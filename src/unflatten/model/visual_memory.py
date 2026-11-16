"""VisualMemory adapter — frozen backbone features into decoder memory.

RFC004 §5 / design `dense-end-to-end-model-v0-design` (doc) §8 M2 item 2. This is
the ONLY place that mentions the backbone's width: `visual_dim` is a
constructor argument taken from the active candidate's config, never a
literal. The backbone is swappable by architectural constraint (CLAUDE.md),
and a hardcoded dimension here would quietly bind the whole model to Qwen.

**The validity mask is not hygiene, it is correctness — and it was measured.**
Pages are not all the same size, so visual token counts differ per sample and
features pad to the batch's longest. `unflatten.model.train` records what that
costs when the padding is left unmasked: `visual_projection` is an
`nn.Linear` *with a bias*, so a zero pad row does not project to zero — it
projects to the learned bias vector and cross-attends as a fully-fledged key.
Measured max |Δ logits| = 0.176 for the same page alone versus co-batched with
a longer one. A model whose output depends on what else happened to be in the
batch is not deterministic, and the failure is silent: nothing errors, the
numbers are merely wrong by an amount nobody looks for.

So `forward` returns the memory AND its key-padding mask together, as one
object. They are not separable: handing a caller the memory without the mask
is handing them the bug. `MemoryBatch` exists to make that impossible to do by
accident.

**Identity at step 0, measured (`scratchpad/audit-train/res96.json`, 96 DPI,
seed 0, 4 real pages, untrained).** The cache holds Qwen's vision tower
output taken AFTER the merger (`candidate_loader.py:301-313`), i.e. already
projected into the LM's own 2 560-wide space — not raw patch features that
still need learning to align. Feeding those features through this module's
own `projection` at its DEFAULT random `nn.Linear` init destroys that
alignment before training ever sees it: native raw features scored
`L_text` 0.88–2.65 nats untrained; the SAME features through a random square
projection scored 3.3–8.8, worse than masking the image out entirely
(2.58–6.89). A random 2560×2560 matrix is not "an untrained adapter that
will learn" here, it is a large, specific corruption applied to features
that were already correct, and one page's gradient at batch size 1 does not
undo it inside the step budgets this project runs. `identity_init` makes the
square case (`visual_dim == d_model`, true for every candidate this repo
currently loads) start as the no-op it should be: `projection.weight = I`,
`bias = 0`, so step 0's memory is bit-identical to the cache's own features,
and training moves the projection away from identity only where the loss
says to. The non-square case has no such no-op to fall back to — a
dimension change is not something an identity matrix can express — so it
keeps `nn.Linear`'s own random init undisturbed; it was never measured to
have this failure mode because RFC004's candidate config is currently
always square, but nothing here promises it stays that way.

Standardisation follows `decoder.StructureDecoder.set_feature_stats`: a
constant channel keeps scale 1 so it becomes all-zero rather than NaN. The
statistics are buffers, so a checkpoint carries the conditioning it was
trained under — a checkpoint that standardises differently than it trained is
a checkpoint that has silently changed model. `normalize=False` is the
default for the same reason as `identity_init`: `LayerNorm` on already
LM-aligned features is not a no-op either — it rescales by ~2.8× and removes
the per-token mean, which is exactly the alignment `identity_init` is trying
to preserve. `normalize=True` remains available (`nn.Identity` otherwise)
for a caller with a measured reason to want it.

**Hashes.** `visual_memory.py` is not in `_SOURCE_BOUND_COMPONENTS` — it was
never lock-chained the way `document_context_encoder.py` / `expert_adapter.py`
/ `cbor.py` are — and `trainable_group_registry` hashes module IDENTITY
(`adapter.projection`), not its numeric init, so neither of those hashes
moves because of this change. `visual_memory_contract` is unaffected because
the contract this module exposes (`MemoryBatch`, `forward`'s signature,
`set_feature_stats`) is unchanged; only what a fresh `nn.Linear` inside it is
initialised to has changed. What DOES change, deliberately: a checkpoint
saved under the old random-init `LayerNorm` carries `visual_memory.norm.*`
parameter keys that a `normalize=False` instance does not have (`nn.Identity`
registers none), so `load_state_dict` (default `strict=True`) refuses it —
correctly, because that checkpoint was trained under a different, and now
known-worse, step-0 pathway.

This module authorizes no gradient update. It declares parameters; it builds
no optimizer and runs no `.backward()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

#: The `trainable_group_registry` group these parameters belong to. The
#: projection is part of the trainable adapter surface, not the frozen base.
VISUAL_MEMORY_GROUP_ID = "adapter.projection"


@dataclass(frozen=True)
class MemoryBatch:
    """Cross-attention memory and the mask that makes it well-defined.

    `key_padding_mask` is True where a position is PADDING, matching
    `nn.TransformerDecoder`'s `memory_key_padding_mask` convention. Getting
    this polarity backwards masks the real tokens and attends to the padding,
    which trains — badly, and without complaint.
    """

    memory: torch.Tensor  # (batch, tokens, d_model)
    key_padding_mask: torch.Tensor  # (batch, tokens), True == padding


class VisualMemory(nn.Module):
    """Standardise, project and normalise frozen visual features."""

    def __init__(self, *, visual_dim: int, d_model: int, normalize: bool = False) -> None:
        super().__init__()
        if visual_dim <= 0:
            raise ValueError(f"visual_dim must be positive, got {visual_dim}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.visual_dim = visual_dim
        self.d_model = d_model
        self.projection = nn.Linear(visual_dim, d_model)
        #: Square case only: the projection starts as the identity, not a
        #: random matrix, because the cached features are already in the
        #: LM's own space (measured, see module docstring). Non-square has
        #: no identity to fall back to, so it keeps `nn.Linear`'s own init.
        self.identity_init = visual_dim == d_model
        if self.identity_init:
            with torch.no_grad():
                self.projection.weight.copy_(torch.eye(d_model))
                self.projection.bias.zero_()
        self.norm = nn.LayerNorm(d_model) if normalize else nn.Identity()
        # Identity until `set_feature_stats`; persistent so a checkpoint
        # carries the conditioning it was trained under.
        self.register_buffer("feature_mean", torch.zeros(visual_dim))
        self.register_buffer("feature_std", torch.ones(visual_dim))

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install the frozen source's per-channel mean and standard deviation.

        Mirrors `decoder.StructureDecoder.set_feature_stats` deliberately: two
        modules that standardise differently would make the same features mean
        different things depending on which one read them.
        """
        for name, value in (("mean", mean), ("std", std)):
            if value.shape != (self.visual_dim,):
                raise ValueError(
                    f"feature {name} has shape {tuple(value.shape)}, expected "
                    f"({self.visual_dim},)"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"feature {name} contains non-finite values")
        if bool((std < 0).any()):
            raise ValueError("feature std contains negative values")
        safe = torch.where(std > 1e-8, std, torch.ones_like(std))
        self.feature_mean.copy_(mean.to(self.feature_mean.dtype))
        self.feature_std.copy_(safe.to(self.feature_std.dtype))

    def forward(
        self,
        visual_features: torch.Tensor,
        *,
        valid_lengths: torch.Tensor | None = None,
    ) -> MemoryBatch:
        """Project `(batch, tokens, visual_dim)` features into decoder memory.

        `valid_lengths` gives the real token count per sample; positions at or
        beyond it are padding. Omitting it declares every position real, which
        is correct only for an unpadded batch — it is not a shortcut for "I
        did not compute the lengths".
        """
        if visual_features.dim() != 3:
            raise ValueError(
                "visual_features must be (batch, tokens, visual_dim), got shape "
                f"{tuple(visual_features.shape)}"
            )
        if visual_features.shape[-1] != self.visual_dim:
            raise ValueError(
                f"visual_features last dim is {visual_features.shape[-1]}, "
                f"expected {self.visual_dim}"
            )
        batch, tokens, _ = visual_features.shape
        standardised = (
            visual_features - self.feature_mean
        ) / self.feature_std
        memory = self.norm(self.projection(standardised))
        if valid_lengths is None:
            mask = torch.zeros(
                (batch, tokens), dtype=torch.bool, device=visual_features.device
            )
        else:
            if valid_lengths.shape != (batch,):
                raise ValueError(
                    f"valid_lengths has shape {tuple(valid_lengths.shape)}, "
                    f"expected ({batch},)"
                )
            if bool((valid_lengths < 0).any()) or bool((valid_lengths > tokens).any()):
                raise ValueError(
                    "valid_lengths must lie in [0, tokens]; got "
                    f"min {int(valid_lengths.min())} max {int(valid_lengths.max())} "
                    f"for tokens={tokens}"
                )
            positions = torch.arange(tokens, device=visual_features.device)
            mask = positions.unsqueeze(0) >= valid_lengths.unsqueeze(1).to(
                positions.device
            )
        return MemoryBatch(memory=memory, key_padding_mask=mask)

"""Shared structural/text/byte decode head over the composite vocabulary --
M2 item 4, `dense-end-to-end-model-v0-design` (doc) §5.2 D2 / §8 M2.

ONE decode stream, not two heads: `composite_vocab.PackedRanges` already
packs `structural | qwen | byte_control | byte_value` into one disjoint
integer id space, and `composite_grammar.CompositeGrammarController` already
gates which sub-range is legal at each step. What was missing (design §8 M2
item 4) is the `nn.Module` that turns a shared decoder hidden state into
logits over that whole packed range, using the SAME embedding/head weights
the grammar and the target materializer already agree on the identity of --
this module does not invent a fourth vocabulary.

Ownership, per `trainable_group_registry.py`'s ``tree_text_decoder`` entry
(three owned tensor identities, all ``s1_allowed_states == {"trainable"}``):

- ``structural_additions`` -- :class:`StructuralAdditions`: input embedding
  and output projection over the 71 structural ids.
- ``text_additions`` -- :class:`ByteFallbackAdditions`: input embedding and
  output projection over the 258 byte-control/byte-value ids.
- ``base_weights`` -- :attr:`CompositeDecodeHead.bridge`: a small trainable
  residual block between the shared decoder hidden state and every one of
  the three sub-heads above.

The Qwen text range is deliberately NOT a fourth trainable identity here.
Design §5.2: "Qwen base embedding/LM-head weights zostávajú frozen... nové
structural/fallback embeddings, output heads a dense adapter sú trainable
groups" -- the pretrained embedding/LM-head weight is reused (design
"prefer the pretrained input embedding and LM head", DEC-007 D4), and that
tensor is already owned by `backbone.language` (frozen). `CompositeDecodeHead`
only takes a *reference* to it (`qwen_embedding_weight`, `qwen_lm_head_weight`)
and must never wrap it in its own `nn.Parameter` -- doing so would silently
duplicate ownership of one tensor identity across two registry groups,
exactly the OPEN-005-016 failure mode CLAUDE.md and the M2 task both warn
about ("a parameter owned twice trains against itself with no error").

Logits are returned as one packed tensor spanning `ranges.total_count`
(design §5.2's disjoint-namespace requirement) so `masked_token_loss`-style
training code can mask the current step's mode-legal sub-range exactly the
way `composite_grammar.AllowedDomain` describes it, without this module
knowing anything about grammar state itself.

This module authorizes no gradient update: it declares `nn.Module`s only.
"""

from __future__ import annotations

import torch
from torch import nn

from unflatten.model.composite_vocab import PackedRanges

#: The one `trainable_group_registry` group id every submodule here belongs
#: to. All three of its owned tensor identities land in this one module.
TREE_TEXT_DECODER_GROUP_ID = "tree_text_decoder"


class StructuralAdditions(nn.Module):
    """Owns the ``structural_additions`` tensor identity: trainable input
    embedding and output projection over the 71 structural local ids."""

    def __init__(self, *, d_model: int, structural_count: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(structural_count, d_model)
        self.output = nn.Linear(d_model, structural_count)

    def embed(self, local_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(local_ids)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(hidden)


class ByteFallbackAdditions(nn.Module):
    """Owns the ``text_additions`` tensor identity: trainable input
    embedding and output projection over `BYTE_BEGIN/BYTE_END` plus the 256
    `BYTE_VALUE` ids (design §5.2's closed byte-fallback frame)."""

    def __init__(self, *, d_model: int, byte_count: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(byte_count, d_model)
        self.output = nn.Linear(d_model, byte_count)

    def embed(self, local_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(local_ids)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(hidden)


class CompositeDecodeHead(nn.Module):
    """`hidden state -> logits over the whole packed composite vocabulary`.

    `ranges` pins the exact packing (`composite_vocab.packed_ranges`);
    `qwen_embedding_weight`/`qwen_lm_head_weight` are frozen references into
    the loaded candidate (`candidate_loader.QwenCandidate`), never copied
    into a trainable parameter here. Both must already have
    `requires_grad=False` -- this module checks that at construction rather
    than silently trusting the caller, because a caller that hands over a
    trainable Qwen weight by mistake would make `backbone.language` train
    through a side door this registry never authorizes.
    """

    def __init__(
        self,
        *,
        d_model: int,
        ranges: PackedRanges,
        qwen_embedding_weight: torch.Tensor,
        qwen_lm_head_weight: torch.Tensor,
        bridge_hidden_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if qwen_embedding_weight.requires_grad or qwen_lm_head_weight.requires_grad:
            raise ValueError(
                "qwen_embedding_weight/qwen_lm_head_weight must be frozen references "
                "(requires_grad=False); CompositeDecodeHead never owns Qwen's tensors, "
                "only reads them"
            )
        if qwen_lm_head_weight.shape[0] != ranges.qwen_count:
            raise ValueError(
                f"qwen_lm_head_weight has {qwen_lm_head_weight.shape[0]} rows, "
                f"but ranges declares qwen_count={ranges.qwen_count}"
            )
        self.d_model = d_model
        self.ranges = ranges
        # Buffers, not Parameters: this is how a frozen reference is held
        # without `nn.Module` treating it as an owned trainable tensor.
        self.register_buffer("qwen_embedding_weight", qwen_embedding_weight, persistent=False)
        self.register_buffer("qwen_lm_head_weight", qwen_lm_head_weight, persistent=False)

        self.structural = StructuralAdditions(
            d_model=d_model, structural_count=ranges.structural_count
        )
        byte_count = ranges.byte_control_count + ranges.byte_value_count
        self.byte_fallback = ByteFallbackAdditions(d_model=d_model, byte_count=byte_count)
        # `base_weights`: the one trainable bridge every sub-head reads from.
        self.bridge = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * bridge_hidden_multiplier),
            nn.GELU(),
            nn.Linear(d_model * bridge_hidden_multiplier, d_model),
        )

    def embed_packed(self, packed_ids: torch.Tensor) -> torch.Tensor:
        """Embed already-packed composite ids for teacher-forced input.

        Routes by range membership: `packed_ids` may mix structural, Qwen
        and byte-fallback ids in one batch (a text span sits between
        structural `TEXT_BEGIN`/`TEXT_END` events in the same sequence).
        """
        r = self.ranges
        if bool((packed_ids < 0).any()) or bool((packed_ids >= r.total_count).any()):
            raise ValueError(
                f"packed_ids out of [0, {r.total_count}) for the configured ranges"
            )
        # dtype follows this module's OWN trainable parameters
        # (`structural.embedding`), not the frozen `qwen_embedding_weight`
        # buffer. DEC-007 D3.1: the bf16 pin covers the frozen base weights
        # and compute, NOT the trainable groups' storage dtype -- an fp32
        # master copy is allowed, and the whole point of keeping one is that
        # bf16's 8 mantissa bits can round a training step back to exactly
        # the starting value. Following the frozen buffer's dtype here would
        # silently narrow every embed_packed() call back to bf16 regardless
        # of what precision the trainable embeddings actually hold -- the
        # Qwen lookup upcasts losslessly into `out` (bf16 -> fp32 loses
        # nothing since bf16 fits inside fp32's mantissa), so upcasting is
        # the safe direction, not the other way around.
        out = torch.zeros(
            *packed_ids.shape,
            self.d_model,
            device=packed_ids.device,
            dtype=self.structural.embedding.weight.dtype,
        )
        structural_mask = packed_ids < r.qwen_start
        qwen_mask = (packed_ids >= r.qwen_start) & (packed_ids < r.byte_control_start)
        byte_mask = packed_ids >= r.byte_control_start

        if bool(structural_mask.any()):
            out[structural_mask] = self.structural.embed(
                packed_ids[structural_mask] - r.structural_start
            )
        if bool(qwen_mask.any()):
            qwen_ids = packed_ids[qwen_mask] - r.qwen_start
            # `Tensor.__setitem__` requires an EXACT dtype match (it raises
            # rather than casting, unlike the elementwise promotion earlier
            # callers of this module may be used to) -- explicit here so a
            # bf16 frozen embedding writing into an fp32 `out` (or any other
            # trainable dtype) does not depend on an implicit rule that does
            # not actually exist for indexed assignment.
            qwen_embedded = nn.functional.embedding(qwen_ids, self.qwen_embedding_weight)
            out[qwen_mask] = qwen_embedded.to(out.dtype)
        if bool(byte_mask.any()):
            out[byte_mask] = self.byte_fallback.embed(
                packed_ids[byte_mask] - r.byte_control_start
            )
        return out

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """`(..., d_model) -> (..., ranges.total_count)` packed composite
        logits, in exactly `structural | qwen | byte_control | byte_value`
        order (matching `composite_vocab.pack`'s layout)."""
        if hidden.shape[-1] != self.d_model:
            raise ValueError(
                f"hidden last dim {hidden.shape[-1]} does not match configured "
                f"d_model {self.d_model}"
            )
        bridged = hidden + self.bridge(hidden)
        structural_logits = self.structural.logits(bridged)
        # SIZE EXCEPTION to the "cast the frozen side to the trainable
        # side" rule (module docstring's dtype-boundary paragraph, and
        # candidate_model.py's): `qwen_lm_head_weight` is vocab-sized.
        # Upcasting it to `bridged`'s dtype on every forward call
        # materialises a full fp32 copy of it -- 1.45 GiB for Qwen3-VL's
        # ~152k-row vocab at d_model -- an OOM found on a real GPU run
        # 2026-09-04, not in any test. DEC-007 D3 also points this
        # direction already: bf16 is the pinned COMPUTE dtype, so the
        # matmul itself belongs in bf16, not fp32. Cast the ACTIVATION down
        # instead and upcast the (small, `(..., qwen_count)`) result --
        # standard mixed precision (bf16 matmul, fp32 master weights,
        # upcast logits before the loss), and it removes the 1.45 GiB copy
        # because only `bridged`, not the frozen weight, needs materialising
        # at a different dtype. This is NOT bit-exact with the fp32-matmul
        # path it replaces -- see `test_composite_head.py`'s characterisation
        # test for the measured logit delta.
        qwen_logits = nn.functional.linear(
            bridged.to(self.qwen_lm_head_weight.dtype), self.qwen_lm_head_weight
        ).to(bridged.dtype)
        byte_logits = self.byte_fallback.logits(bridged)
        return torch.cat([structural_logits, qwen_logits, byte_logits], dim=-1)

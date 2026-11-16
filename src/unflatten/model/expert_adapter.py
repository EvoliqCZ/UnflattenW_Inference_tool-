"""Dense Word Expert Adapter -- RFC004 §22, DEC-005 R4-5 dense V0 baseline.

§22.1 (`docs/rfc/RFC004-ai-model-architecture-v0.1-1.md:1541`) puts the Word
Expert Adapter in the Word Reconstruction Branch, never as a rewrite of the
pretrained backbone, and states its contract precisely: experts transform
*shared hidden representations*; they do not produce separate trees that get
merged by voting later. Dense mode is the mandatory baseline; sparse MoE is a
manifest-selected alternative implementation of the same interface, not
built here (packet R4-5, `docs/plans/dense-v0-ratification-packet.md:204`):

    expert_adapter.mode = dense
    router = not_applicable
    shared_expert = not_applicable
    specialists.* = not_applicable

    "Aktivny je jeden shared residual DenseWordAdapter."

§22.3's reference sparse combination is

    h' = h + alpha_shared * E_shared(h) + sum_i w_i * E_i(h)

Dense V0 has no router and no specialist bank, so that formula collapses to
its only active term, a single shared residual transform:

    h' = h + E_dense(h)

which is exactly `DenseExpertAdapter.forward`. There is no router, no top-k,
no expert bank -- building any of those here would be sparse MoE, which
CLAUDE.md and R4-5 explicitly reserve for a later, separately ratified
RFC005 S6 ablation.

§22.2 lists seven *logical* expert roles (`shared_document`, `structure_flow`,
`text_ocr`, `paragraph_run`, `table_grid`, `section_layout`, `image_resource`)
and states a logical role need not mean exactly one physical FFN -- physical
expert count is a manifest field. Dense V0's manifest pins that field to one:
R4-5 names exactly one active module, so this adapter does not instantiate
seven physical sub-modules keyed by node type. `layers` is a depth knob for
that one physical module (how many residual blocks the shared transform
runs), not a per-node-type expert count; it must not be read as reintroducing
per-role experts through a side door.

No hidden dimension is hardcoded: `d_model` is a required constructor
argument, taken from the caller's active backbone/decoder config, matching
the pattern in `unflatten.model.decoder.StructureDecoder`.

`EXPERT_ADAPTER_DENSE_GROUP_ID` names the one
`trainable_group_registry.trainable_group_registry()` entry
(`expert_adapter.dense`, kind `trainable_primary_s1`) this module's
parameters belong to. That entry owns exactly one tensor identity,
`expert_adapter.dense/base_weights` -- singular, not one identity per
residual block -- so every parameter this module registers (across however
many `layers`) is one undifferentiated owned tensor identity, never split
across several groups.

This module authorizes no gradient update: it declares an `nn.Module` with
its own trainable parameters, but nothing here builds an optimizer, a
training loop, or runs `.backward()`.
"""

from __future__ import annotations

import torch
from torch import nn

#: The one `trainable_group_registry` group ID this module's parameters
#: belong to (packet R4-9, R5-3). Kept here as the single source of truth
#: for the cross-check in `tests/model/test_expert_adapter.py`.
EXPERT_ADAPTER_DENSE_GROUP_ID = "expert_adapter.dense"


class _DenseAdapterBlock(nn.Module):
    """One pre-norm residual FFN block: `h + Down(Act(Up(Norm(h))))`."""

    def __init__(self, d_model: int, hidden_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, d_model * hidden_multiplier)
        self.act = nn.GELU()
        self.down = nn.Linear(d_model * hidden_multiplier, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        delta = self.down(self.act(self.up(self.norm(hidden))))
        return hidden + self.dropout(delta)


class DenseExpertAdapter(nn.Module):
    """§22.1/R4-5 dense Word Expert Adapter: `h' = h + E_dense(h)`.

    Applied directly to the shared decoder hidden state (any leading batch
    and node/token axes; the transform acts on the last, `d_model`, axis
    only), so it never touches tree structure and cannot itself branch
    into per-node-type paths -- that is precisely what dense mode declares
    `not_applicable` per R4-5.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_multiplier: int = 4,
        layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers must be >= 1, got {layers}")
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [_DenseAdapterBlock(d_model, hidden_multiplier, dropout) for _ in range(layers)]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Transform `hidden` (`..., d_model`) into the same shape/dtype."""
        if hidden.shape[-1] != self.d_model:
            raise ValueError(
                f"last dim {hidden.shape[-1]} does not match configured "
                f"d_model {self.d_model}"
            )
        for block in self.layers:
            hidden = block(hidden)
        return hidden

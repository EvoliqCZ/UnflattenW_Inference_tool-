"""Sparse Word Expert Adapter -- RFC004 §22/§23, DEC-047 dormant code.

DEC-047 (`docs/decisions/DEC-047-spiaci-kod-v-predstihu.md`) permits this
module ahead of the S6/G2 ratification CLAUDE.md otherwise requires for any
MoE topology or routing, under five binding conditions this module and its
tests satisfy:

1. **Default off** -- nothing in `candidate_model`, `expert_adapter.py` or any
   construction path imports or instantiates :class:`SparseExpertAdapter`;
   only tests do.
2. **Step 0 == dense** -- :meth:`SparseExpertAdapter.from_dense` zero-inits
   every specialist's output projection, so every specialist transform is
   exactly the zero function at init regardless of routing; combined with
   `alpha_shared == 1.0` and a bit-for-bit copy of the dense adapter's
   weights into the shared expert, `SparseExpertAdapter.forward(h) ==
   DenseExpertAdapter.forward(h)` exactly at construction
   (`tests/model/sparse_expert_adapter/test_adapter.py
   ::test_from_dense_matches_dense_output_exactly`).
3. **No hashed-identity change** -- this package does not import or edit
   `trainable_group_registry.py`, `capability_profile.py`, `loss_unit.py`,
   `training_materializer.py` or any v15b policy file. Its own sparse
   registry lives in :mod:`unflatten.model.sparse_expert_adapter.registry`,
   a new, separately hashed artefact nothing in the production path reads.
4. **Topology is a parameter** -- see `manifest.py`: expert count, `top_k`,
   logical roles, insertion point and eligibility table are all
   :class:`~unflatten.model.sparse_expert_adapter.manifest
   .SparseExpertAdapterManifest` fields, never module constants.
5. **No gradient runs** -- this module declares only `nn.Module` subclasses
   and pure functions; it builds no optimizer and calls no `.backward()`.
   The invariance suite runs on CPU with tiny dims.

§22.3's reference combination for hidden state `h`::

    s_i = router_i(h)                              for eligible specialists
    S   = deterministic_top_k(softmax(s), k, stable_expert_id_tie_break)
    w_i = exp(s_i) / sum_{j in S} exp(s_j)
    h'  = h + alpha_shared * E_shared(h) + sum_{i in S} w_i * E_i(h)

Each `E_*(h)` here is defined as a pure *delta* transform (its residual `h +`
is applied once, at the top level, matching the formula literally) rather
than the `_DenseAdapterBlock` stack's own internal per-block residual --
:class:`_ExpertTransform` computes that delta as `stack(h) - h`. This is what
lets `E_shared` be an exact copy of `DenseExpertAdapter`'s block stack: with
`alpha_shared = 1.0` and every specialist delta forced to zero,
`h' = h + (dense_stack(h) - h) + 0 = dense_stack(h)`, exactly
`DenseExpertAdapter.forward(h)`.

Routing is per-token and batch-independent by construction: the router
linear layer, the eligibility mask, `torch.argsort`-based top-k and the
softmax renormalization are all elementwise along the leading axes and
reduce only over the trailing expert axis, so one token's routing decision
and output never depend on which other tokens share the call
(§22.6 dropless dispatch; no batch-norm, no capacity factor, no cross-token
reduction anywhere in this file). §22.6 also forbids overflow-to-shared and
token drop in the forward path: this implementation computes every
specialist's transform over the *entire* input and only then applies the
per-token selection weight (which is exactly zero for an unselected
specialist), so no token's contribution is ever dropped or capped by a
batch-global capacity factor -- the tradeoff, spelled out because it is not
free, is that this reference implementation is compute-*wasteful* (it runs
every specialist over every token rather than gathering), which is
acceptable for a CPU-tested, not-yet-deployed dormant module and is exactly
what a real dispatch-optimized implementation would later replace without
changing this module's semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from unflatten.model.expert_adapter import DenseExpertAdapter, _DenseAdapterBlock
from unflatten.model.sparse_expert_adapter.manifest import (
    SparseExpertAdapterManifest,
    SparseExpertAdapterManifestError,
)

_NEG_INF = float("-inf")


class MissingExpertWeightError(RuntimeError):
    """A declared specialist/shared/router weight is absent at load time.

    §23: "missing weight pre ktorýkoľvek expert/router deklarovaný aktívnym
    sparse Component Lockom je hard model load error" -- raised by
    :func:`load_sparse_expert_adapter_state_dict` rather than silently
    leaving a sub-module at its randomly initialized weights.
    """


@dataclass(frozen=True)
class RoutingResult:
    """One `route()` call's full decision, kept for telemetry/tests.

    `expert_ids` is the fixed, manifest-ordered tuple of specialist ids that
    `selected`/`weights` columns correspond to (never `shared_document`,
    which is not routed -- §22.3 says it is unconditionally active).
    """

    expert_ids: tuple[str, ...]
    logits: torch.Tensor
    eligibility_mask: torch.Tensor
    selected: torch.Tensor
    weights: torch.Tensor


class _ExpertTransform(nn.Module):
    """A stack of `_DenseAdapterBlock`s, exposed as a pure delta transform.

    `forward(h)` returns `stack(h) - h`, not `stack(h)` -- the outer
    `SparseExpertAdapter.forward` adds the `+ h` term exactly once, per the
    §22.3 reference formula.
    """

    def __init__(self, d_model: int, hidden_multiplier: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_DenseAdapterBlock(d_model, hidden_multiplier, dropout) for _ in range(layers)]
        )

    def forward(self, hidden: torch.Tensor, *, residual_only: bool = True) -> torch.Tensor:
        transformed = hidden
        for block in self.blocks:
            transformed = block(transformed)
        return transformed - hidden if residual_only else transformed

    def zero_output_(self) -> None:
        """Zero every block's output projection so `forward` returns 0 exactly."""
        for block in self.blocks:
            nn.init.zeros_(block.down.weight)
            nn.init.zeros_(block.down.bias)


class _Router(nn.Module):
    """Linear router over `expert_ids`, with a fixed ascending tie-break rank.

    §22.4: input is the hidden state only -- no split/source/task-view id/
    target value is a constructor argument or forward input anywhere in this
    class, so there is nothing forbidden for it to leak.
    """

    def __init__(self, d_model: int, expert_ids: tuple[str, ...]) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, len(expert_ids), bias=False)
        tie_break_rank = [0] * len(expert_ids)
        for rank, index in enumerate(sorted(range(len(expert_ids)), key=lambda i: expert_ids[i])):
            tie_break_rank[index] = rank
        self.register_buffer(
            "tie_break_rank", torch.tensor(tie_break_rank, dtype=torch.long), persistent=False
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


def _deterministic_top_k(
    logits: torch.Tensor,
    eligibility_mask: torch.Tensor,
    tie_break_rank: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """§22.3/§22.4 top-k: deterministic, stable-tied by ascending `expert_id`.

    Returns `(selected, weights)`, both shaped like `logits`. `weights` is
    the softmax renormalized over exactly the selected columns (zero
    elsewhere); a token with zero eligible experts gets an all-zero
    `selected`/`weights` row (the §22.7 "shared-only fallback" case), never
    a NaN.
    """
    masked = logits.masked_fill(~eligibility_mask, _NEG_INF)
    tie_order = torch.argsort(tie_break_rank)
    reordered = masked.index_select(-1, tie_order)
    rank_in_reordered = torch.argsort(reordered, dim=-1, descending=True, stable=True)
    composite_order = tie_order[rank_in_reordered]
    arange = torch.arange(logits.shape[-1], device=logits.device).expand_as(composite_order)
    rank = torch.empty_like(composite_order)
    rank.scatter_(-1, composite_order, arange)

    eligible_count = eligibility_mask.sum(dim=-1, keepdim=True)
    effective_k = torch.clamp(eligible_count, max=top_k)
    selected = (rank < effective_k) & eligibility_mask

    weight_logits = logits.masked_fill(~selected, _NEG_INF)
    weights = torch.softmax(weight_logits, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return selected, weights


def _build_eligibility_mask(
    manifest: SparseExpertAdapterManifest,
    context: str | Sequence[str] | None,
    leading_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    n_experts = len(manifest.specialist_roles)
    if context is None or isinstance(context, str):
        row = manifest.eligibility_mask_for_context(context)
        mask = torch.tensor(row, dtype=torch.bool, device=device)
        return mask.expand(*leading_shape, n_experts)

    contexts = list(context)
    expected = 1
    for dim in leading_shape:
        expected *= dim
    if len(contexts) != expected:
        raise ValueError(
            f"per-token context has {len(contexts)} entries but hidden has "
            f"{expected} leading tokens (shape {tuple(leading_shape)})"
        )
    rows = [manifest.eligibility_mask_for_context(c) for c in contexts]
    return torch.tensor(rows, dtype=torch.bool, device=device).reshape(*leading_shape, n_experts)


class SparseExpertAdapter(nn.Module):
    """§22/§23 sparse Word Expert Adapter: shared expert + top-k specialists.

    `forward(hidden, context=...)` mirrors :class:`DenseExpertAdapter`'s
    single-positional-argument interface (`context` is keyword-only and
    optional, defaulting to "no grammar restriction"), so a caller wired to
    the dense interface can pass a `SparseExpertAdapter` unchanged. Nothing
    in this repository does that substitution -- DEC-047 condition 1.
    """

    def __init__(self, manifest: SparseExpertAdapterManifest) -> None:
        super().__init__()
        self.manifest = manifest
        self.d_model = manifest.d_model
        expert_ids = tuple(role.expert_id for role in manifest.specialist_roles)
        self.expert_ids = expert_ids
        self.router = _Router(manifest.d_model, expert_ids)
        self.shared_expert = _ExpertTransform(
            manifest.d_model, manifest.hidden_multiplier, manifest.layers, manifest.dropout
        )
        self.specialists = nn.ModuleList(
            _ExpertTransform(
                manifest.d_model, manifest.hidden_multiplier, manifest.layers, manifest.dropout
            )
            for _ in manifest.specialist_roles
        )

    def route(
        self, hidden: torch.Tensor, *, context: str | Sequence[str] | None = None
    ) -> RoutingResult:
        """Compute the routing decision for `hidden` without applying experts."""
        if hidden.shape[-1] != self.d_model:
            raise ValueError(
                f"last dim {hidden.shape[-1]} does not match configured d_model {self.d_model}"
            )
        eligibility = _build_eligibility_mask(
            self.manifest, context, hidden.shape[:-1], hidden.device
        )
        logits = self.router(hidden)
        selected, weights = _deterministic_top_k(
            logits, eligibility, self.router.tie_break_rank, self.manifest.top_k
        )
        return RoutingResult(
            expert_ids=self.expert_ids,
            logits=logits,
            eligibility_mask=eligibility,
            selected=selected,
            weights=weights,
        )

    def forward(
        self, hidden: torch.Tensor, *, context: str | Sequence[str] | None = None
    ) -> torch.Tensor:
        """Transform `hidden` (`..., d_model`) into the same shape/dtype."""
        routing = self.route(hidden, context=context)
        # With unit shared weight, preserve the dense stack's exact arithmetic.
        # Subtracting its input and adding it back can change trained bf16 bits.
        if self.manifest.alpha_shared == 1.0:
            out = self.shared_expert(hidden, residual_only=False)
        else:
            out = hidden + self.manifest.alpha_shared * self.shared_expert(hidden)
        for index, specialist in enumerate(self.specialists):
            weight = routing.weights[..., index : index + 1]
            out = out + weight * specialist(hidden)
        return out

    @classmethod
    def from_dense(
        cls, dense: DenseExpertAdapter, manifest: SparseExpertAdapterManifest
    ) -> SparseExpertAdapter:
        """Build a sparse adapter whose step-0 output equals `dense`'s exactly.

        Requires `manifest.alpha_shared == 1.0` and an architecture
        (`d_model`, `layers`, inferred `hidden_multiplier`) matching `dense`
        exactly -- both are needed for the shared expert's copied weights to
        reproduce `dense`'s output bit-for-bit (condition 2, DEC-047).
        """
        if manifest.alpha_shared != 1.0:
            raise SparseExpertAdapterManifestError(
                f"from_dense requires alpha_shared == 1.0 for exact dense-output "
                f"equivalence at step 0, got {manifest.alpha_shared!r}"
            )
        if manifest.d_model != dense.d_model:
            raise SparseExpertAdapterManifestError(
                f"manifest.d_model {manifest.d_model} != dense.d_model {dense.d_model}"
            )
        dense_layers = len(dense.layers)
        if manifest.layers != dense_layers:
            raise SparseExpertAdapterManifestError(
                f"manifest.layers {manifest.layers} != dense adapter's {dense_layers} layers"
            )
        dense_hidden_multiplier = dense.layers[0].up.out_features // dense.d_model
        if manifest.hidden_multiplier != dense_hidden_multiplier:
            raise SparseExpertAdapterManifestError(
                f"manifest.hidden_multiplier {manifest.hidden_multiplier} != dense "
                f"adapter's inferred {dense_hidden_multiplier}"
            )

        adapter = cls(manifest)
        adapter.shared_expert.blocks.load_state_dict(dense.layers.state_dict())
        for specialist in adapter.specialists:
            specialist.zero_output_()
        return adapter

"""Router telemetry and `L_router` -- RFC004 §22.7, RFC005 §27.8.

Standalone functions over a :class:`~unflatten.model.sparse_expert_adapter
.adapter.RoutingResult`, deliberately **not** registered anywhere a
production/training path reads: not in `training_materializer.LOSS_FAMILY_IDS`,
not in `loss_unit.py`, not in the §26.6 allowlist. Registering `l_router`
there would move `loss_unit_schema_hash`, which DEC-047 condition 3 forbids
for dormant code. Nothing calls these functions except tests.

§22.7 asks the runtime to report, without document content: expert selection
counts, load balance, overflow count, shared-only fallback count, routing
entropy histograms, and per-node-type utilization -- all present below.
§20.3/§27.8 name the hard invariants `l_router` itself must not violate:
never reward source/customer routing (this module never sees those fields,
by construction of `RoutingResult`), never suppress the shared expert (the
loss term only touches specialist logits, not `alpha_shared`), never
compensate token drop (dropless dispatch makes drop count structurally
zero), never encode batch-global production capacity semantics (no
capacity factor appears anywhere in this file).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from unflatten.model.sparse_expert_adapter.adapter import RoutingResult

#: §22.6: dropless dispatch structurally drops nothing; kept as an explicit
#: constant so a telemetry consumer never has to guess why it is always 0.
STRUCTURAL_OVERFLOW_COUNT = 0


@dataclass(frozen=True)
class RouterTelemetry:
    """One `route()` call's diagnostic summary -- §22.7 reporting surface."""

    expert_selection_counts: dict[str, int]
    load_balance: dict[str, float]
    overflow_count: int
    shared_only_fallback_count: int
    routing_entropy_mean: float
    per_node_type_utilization: dict[str, dict[str, int]]


def routing_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Per-token Shannon entropy (nats) of the selected-expert weight row.

    Zero-weight columns contribute 0 by convention (`0 * log(0) := 0`), so an
    all-zero row (shared-only fallback) has entropy exactly 0, not NaN.
    """
    safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    terms = torch.where(weights > 0, -weights * safe.log(), torch.zeros_like(weights))
    return terms.sum(dim=-1)


def expert_selection_counts(routing: RoutingResult) -> dict[str, int]:
    """How many tokens selected each specialist at least once (top-k membership)."""
    counts = routing.selected.reshape(-1, len(routing.expert_ids)).sum(dim=0)
    return {expert_id: int(counts[i].item()) for i, expert_id in enumerate(routing.expert_ids)}


def detect_dead_experts(counts: dict[str, int]) -> frozenset[str]:
    """Expert ids selected by zero tokens in `counts` -- §20.3 "dead experts"."""
    return frozenset(expert_id for expert_id, count in counts.items() if count == 0)


def shared_only_fallback_count(routing: RoutingResult) -> int:
    """Tokens with zero eligible/selected specialists (only the shared expert ran)."""
    per_token = routing.selected.reshape(-1, len(routing.expert_ids)).any(dim=-1)
    return int((~per_token).sum().item())


def load_balance(counts: dict[str, int]) -> dict[str, float]:
    """Fraction of total selections each expert received (sums to 1, or all-0 if none)."""
    total = sum(counts.values())
    if total == 0:
        return {expert_id: 0.0 for expert_id in counts}
    return {expert_id: count / total for expert_id, count in counts.items()}


def per_node_type_utilization(
    routing: RoutingResult, node_types: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Selection counts per expert, broken down by a caller-supplied node type per token.

    `node_types` must have one entry per token in `routing.selected`'s
    flattened leading axes -- this is purely a reporting aggregation, it
    never feeds back into routing itself.
    """
    flat = routing.selected.reshape(-1, len(routing.expert_ids))
    if len(node_types) != flat.shape[0]:
        raise ValueError(
            f"node_types has {len(node_types)} entries but routing covers {flat.shape[0]} tokens"
        )
    result: dict[str, dict[str, int]] = {}
    for token_index, node_type in enumerate(node_types):
        bucket = result.setdefault(node_type, {eid: 0 for eid in routing.expert_ids})
        for expert_index, expert_id in enumerate(routing.expert_ids):
            if bool(flat[token_index, expert_index]):
                bucket[expert_id] += 1
    return result


def summarize(
    routing: RoutingResult, *, node_types: Sequence[str] | None = None
) -> RouterTelemetry:
    """Build the full §22.7 telemetry report for one `route()` result."""
    counts = expert_selection_counts(routing)
    return RouterTelemetry(
        expert_selection_counts=counts,
        load_balance=load_balance(counts),
        overflow_count=STRUCTURAL_OVERFLOW_COUNT,
        shared_only_fallback_count=shared_only_fallback_count(routing),
        routing_entropy_mean=float(routing_entropy(routing.weights).mean().item()),
        per_node_type_utilization=(
            per_node_type_utilization(routing, node_types) if node_types is not None else {}
        ),
    )


def stable_across_co_batch(
    routing_alone: RoutingResult, routing_in_batch: RoutingResult, token_index_in_batch: int
) -> bool:
    """True iff a token's routing is identical whether run alone or co-batched.

    `routing_alone` must cover exactly one token (its flattened leading axis
    has length 1); `token_index_in_batch` selects that same token's row out
    of `routing_in_batch`.
    """
    alone_selected = routing_alone.selected.reshape(-1, len(routing_alone.expert_ids))[0]
    alone_weights = routing_alone.weights.reshape(-1, len(routing_alone.expert_ids))[0]
    batch_selected = routing_in_batch.selected.reshape(-1, len(routing_in_batch.expert_ids))[
        token_index_in_batch
    ]
    batch_weights = routing_in_batch.weights.reshape(-1, len(routing_in_batch.expert_ids))[
        token_index_in_batch
    ]
    return bool(torch.equal(alone_selected, batch_selected)) and bool(
        torch.allclose(alone_weights, batch_weights, atol=1e-6)
    )


def l_router(
    routing: RoutingResult,
    *,
    load_balance_coefficient: float = 1e-2,
    z_loss_coefficient: float = 1e-3,
) -> torch.Tensor:
    """RFC005 §27.8 bounded load-balance + router z-loss, training-only.

    Load-balance term follows the Switch-Transformer form
    ``n_experts * sum_e f_e * P_e`` where ``f_e`` is the selected fraction of
    tokens per expert and ``P_e`` is the mean *selected-softmax* probability
    mass per expert (both computed only over routed specialists, never over
    the always-on shared expert, so this term structurally cannot suppress
    it). The z-loss term is ``mean(logsumexp(logits)^2)``, bounding router
    logit magnitude without touching drop compensation or batch-global
    capacity, both of which §27.8 forbids.
    """
    n_experts = len(routing.expert_ids)
    flat_selected = routing.selected.reshape(-1, n_experts).float()
    flat_weights = routing.weights.reshape(-1, n_experts)
    n_tokens = flat_selected.shape[0]
    if n_tokens == 0:
        return torch.zeros((), dtype=routing.logits.dtype)

    f = flat_selected.mean(dim=0)
    p = flat_weights.mean(dim=0)
    load_balance_loss = n_experts * torch.sum(f * p)

    flat_logits = routing.logits.reshape(-1, n_experts)
    z_loss = torch.mean(torch.logsumexp(flat_logits, dim=-1) ** 2)

    return load_balance_coefficient * load_balance_loss + z_loss_coefficient * z_loss

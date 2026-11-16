"""RFC004-ERRATA-005 E5: the `adaptive-document-visual-memory-v0` profile.

DEC-047 (`docs/decisions/DEC-047-spiaci-kod-v-predstihu.md`) authorizes this
as dormant code over a PROPOSED part: default off, no gradient runs, CPU
tests only, no hashed-identity change. Nothing in the model wiring
constructs :class:`AdaptiveVisualMemoryCompressor` yet.

Five risk classes (same values as
:data:`unflatten.model.visual_compression.region_plan.RiskClass`, "protected"
first): a per-token risk assignment head picks one class per visual token,
and tokens merge into `coarse_memory` in contiguous same-class runs, up to
the policy's retention ratio for that class. `protected_detail` always
carries the full, unmerged memory (E5 rule 2: only exact text path, typed
heads, TablePlan, Resource Recovery and the Node Visual Aligner read it) --
so "100% retention of protected regions" holds by construction, not by a
merge decision.

**Step 0 == bypass (E5 rule 1).** ``RiskAssignmentHead`` is a zero-initialized
`Linear`; with an all-zero logit row `torch.argmax` deterministically picks
index 0 ("protected") for every token, so every class gets ratio 1 and no
merge ever happens -- `coarse_memory` is then bit-identical to `memory`,
matching :class:`~unflatten.model.visual_compression.bypass.BypassVisualCompressor`
exactly (see `test_adaptive_v0.py::test_step_zero_matches_bypass_identity`).

Merging only supports a single-page batch (`memory.tokens.shape[0] == 1`),
the same restriction
:func:`unflatten.model.visual_compression.architecture_contract.check_alignment_integrity`
already documents for this interface -- a batch item's risk assignment can
produce a different compressed token count than its neighbour, and nothing
in :class:`~unflatten.model.visual_compression.contracts.VisualMemory` lets
two batch items disagree on the tokens dimension.

**Training-only straight-through relaxation (RFC005-ERRATA-007 E5).** The
group *membership* decision (`torch.argmax`) has no gradient at any step --
that discreteness is not relaxed. What *is* relaxed, and only while
`self.training` is `True`: each token's contribution to its group's merged
mean is scaled by `_straight_through_merge_weight`, a straight-through
estimator whose forward value is always exactly `1.0` (so the merged value
is bit-identical to the plain hard mean at every step, including step 0's
zero-init bypass identity) but whose backward pass routes gradient into the
risk head through the softmax probability of each token's own assigned
class. Eval/inference never builds this weight at all -- it keeps the exact
hard mean this module has always used.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import Tensor, nn

from unflatten.model.hashing import domain_hash
from unflatten.model.visual_compression.contracts import (
    BudgetReport,
    CompressedVisualMemory,
    OriginalTokenAlignment,
    VisualMemory,
    retention_map_to_alignment,
)

#: Same five values as `region_plan.RiskClass`, in the fixed order the
#: zero-init step-0 identity depends on: "protected" MUST be index 0.
RISK_CLASSES: Final[tuple[str, ...]] = ("protected", "high", "medium", "low", "uncertain")

_POLICY_PATH: Final = Path(__file__).parent / "policies" / "adaptive_v0_policy.json"

_POLICY_DOMAIN = "unflatten.visual-compression.adaptive-v0-policy/v1"
_IMPLEMENTATION_DOMAIN = "unflatten.visual-compression.adaptive-v0-implementation/v1"

#: A fixed marker for the implementation identity this module presents to
#: the Component Lock (`check_hashes_present`). It has no versioned fields
#: yet -- bump the payload when the merge algorithm itself changes.
IMPLEMENTATION_HASH: Final = domain_hash(
    _IMPLEMENTATION_DOMAIN, {"module": "adaptive_v0", "algorithm": "contiguous_same_class_merge"}
)


class AdaptiveVisualMemoryPolicyError(ValueError):
    """A policy payload does not match the shape this module reads."""


@dataclass(frozen=True, slots=True)
class AdaptiveVisualMemoryPolicy:
    """§15 PROPOSED starting ratios (E5): "hodnoty sú [PROPOSED] na overenie
    vo VC-1, nie cieľ" -- kept as policy data, never a code literal, exactly
    per DEC-047 condition 4.
    """

    policy_version: str
    retention_ratio: Mapping[str, int]

    def __post_init__(self) -> None:
        if frozenset(self.retention_ratio) != frozenset(RISK_CLASSES):
            raise AdaptiveVisualMemoryPolicyError(
                f"retention_ratio must have exactly the keys {RISK_CLASSES}, "
                f"got {sorted(self.retention_ratio)}"
            )
        for risk_class, ratio in self.retention_ratio.items():
            if ratio < 1:
                raise AdaptiveVisualMemoryPolicyError(
                    f"retention_ratio[{risk_class!r}] must be >= 1, got {ratio}"
                )
        if self.retention_ratio["protected"] != 1:
            raise AdaptiveVisualMemoryPolicyError(
                "retention_ratio['protected'] must be 1 (E5: '100 % retencia "
                "chránených regiónov')"
            )

    def hash(self) -> str:
        return domain_hash(
            _POLICY_DOMAIN,
            {"policy_version": self.policy_version, "retention_ratio": dict(self.retention_ratio)},
        )


def load_default_policy() -> AdaptiveVisualMemoryPolicy:
    """Load the versioned starting-ratio policy from
    `policies/adaptive_v0_policy.json` -- never a code literal (DEC-047
    condition 4)."""
    payload = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    return AdaptiveVisualMemoryPolicy(
        policy_version=payload["policy_version"],
        retention_ratio=dict(payload["retention_ratio"]),
    )


class RiskAssignmentHead(nn.Module):
    """Per-token risk-class logits -- zero-initialized (E5 rule 1 / DEC-047
    condition 2: "krok 0 == dense"). All-zero logits make `argmax` pick
    index 0 ("protected") for every token deterministically.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model!r}")
        self.proj = nn.Linear(d_model, len(RISK_CLASSES))
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()

    def forward(self, tokens: Tensor) -> Tensor:
        """`tokens`: `(batch, n_tokens, d_model)` -> logits `(batch, n_tokens, 5)`."""
        return self.proj(tokens)


class AdaptiveVisualMemoryCompressor(nn.Module):
    """RFC004 §15.2 `VisualCompressor` implementing the E5 profile.

    `policy` defaults to :func:`load_default_policy` -- always overridable
    since the ratios are PROPOSED, never ratified (DEC-047 condition 4).
    """

    def __init__(
        self, d_model: int, *, policy: AdaptiveVisualMemoryPolicy | None = None
    ) -> None:
        super().__init__()
        self.policy = policy if policy is not None else load_default_policy()
        self.risk_head = RiskAssignmentHead(d_model)

    def compress(self, memory: VisualMemory) -> CompressedVisualMemory:
        batch, n_tokens, _ = memory.tokens.shape
        if batch != 1:
            raise ValueError(
                f"AdaptiveVisualMemoryCompressor only supports a single-page "
                f"batch (batch == 1), got batch == {batch}"
            )
        logits = self.risk_head(memory.tokens)  # (1, n_tokens, 5)
        risk_idx = torch.argmax(logits, dim=-1)  # (1, n_tokens)
        group_of_token, group_tokens = _group_contiguous_by_ratio(
            risk_idx[0].tolist(), self.policy.retention_ratio
        )
        retention_map = torch.tensor(
            group_of_token, dtype=torch.long, device=memory.tokens.device
        )
        n_groups = len(group_tokens)
        coarse_tokens = torch.stack(
            [
                _merge_group_tokens(
                    memory.tokens[0, indices, :],
                    logits[0, indices, :],
                    risk_idx[0, indices],
                    training=self.training,
                )
                for indices in group_tokens
            ],
            dim=0,
        ).unsqueeze(0)
        coarse_mask = torch.tensor(
            [
                bool(memory.key_padding_mask[0, indices].all())
                for indices in group_tokens
            ],
            dtype=torch.bool,
            device=memory.tokens.device,
        ).unsqueeze(0)
        coarse_memory = VisualMemory(
            tokens=coarse_tokens, key_padding_mask=coarse_mask, page_ordinals=memory.page_ordinals
        )
        bypassed = int((risk_idx == 0).sum().item())
        budget_report = BudgetReport(
            original_token_count=n_tokens,
            compressed_token_count=n_groups,
            bypassed_region_count=bypassed,
            policy_version=self.policy.policy_version,
        )
        return CompressedVisualMemory(
            coarse_memory=coarse_memory,
            protected_detail=memory,
            risk_map=risk_idx,
            retention_map=retention_map,
            budget_report=budget_report,
        )

    def expand_alignment(self, retention_map: Tensor) -> OriginalTokenAlignment:
        return retention_map_to_alignment(retention_map)

    def soft_compression_ratio(self, memory: VisualMemory) -> Tensor:
        """Differentiable budget signal (VC-1 rehearsal 2026-09-28 finding):
        the EXPECTED compression ratio under the risk head's softmax,
        ``valid_tokens / sum_n sum_c p_c(n) / retention_ratio[c]`` -- a token
        in class c occupies ``1 / ratio_c`` of a compressed slot. Unlike
        ``budget_report.compression_ratio`` (a float of the hard ``argmax``
        grouping) this stays in the autograd graph, so a budget term built
        on it moves a zero-init head that protects every token, where the
        distillation loss is exactly 0 and gives no gradient at all. It
        relaxes only the budget; grouping and the merged values stay hard."""
        valid = ~memory.key_padding_mask
        probs = torch.softmax(self.risk_head(memory.tokens), dim=-1)
        inverse_ratio = torch.tensor(
            [1.0 / self.policy.retention_ratio[c] for c in RISK_CLASSES],
            dtype=probs.dtype,
            device=probs.device,
        )
        expected_slots = (probs * inverse_ratio).sum(dim=-1)[valid].sum()
        return valid.sum().to(probs.dtype) / expected_slots


def _straight_through_merge_weight(logits: Tensor, risk_idx: Tensor) -> Tensor:
    """Straight-through estimator for one group's per-token merge weight.

    `logits`: `(group_size, 5)`, `risk_idx`: `(group_size,)` -- the class
    each token in the group was hard-assigned to (all equal, by
    construction of `_group_contiguous_by_ratio`). The returned tensor's
    forward value is always exactly `1.0` per token: `soft_selected +
    (1 - soft_selected).detach()` evaluates to `1.0` regardless of
    `soft_selected`'s value, so a weighted merge using this weight is
    bit-identical to the unweighted hard mean at every step. Its gradient
    is `d(soft_selected)/d(logits)` only -- the second term is detached --
    which is how gradient reaches `risk_head`'s parameters without touching
    the discrete `argmax` grouping decision itself.
    """
    soft = torch.softmax(logits, dim=-1)  # (group_size, 5)
    soft_selected = soft.gather(-1, risk_idx.unsqueeze(-1)).squeeze(-1)  # (group_size,)
    return soft_selected + (1.0 - soft_selected).detach()


def _merge_group_tokens(
    tokens: Tensor, logits: Tensor, risk_idx: Tensor, *, training: bool
) -> Tensor:
    """Mean-merge one contiguous same-class group of tokens into one coarse
    token. `tokens`/`logits`/`risk_idx` are already sliced to this group.

    Eval/inference (`training=False`) is the exact hard mean this profile
    has always used. Training additionally scales each token by
    `_straight_through_merge_weight` before averaging -- forward-identical
    to the plain mean (the weight is always `1.0`), but it gives the risk
    head's own weights a gradient (RFC005-ERRATA-007 E5: the retention/merge
    path itself has no gradient w.r.t. the discrete risk assignment, so the
    gradient is routed through the merge weights instead, a soft merge in
    training only).
    """
    if not training:
        return tokens.mean(dim=0)
    weight = _straight_through_merge_weight(logits, risk_idx)
    return (tokens * weight.unsqueeze(-1)).sum(dim=0) / tokens.shape[0]


def _group_contiguous_by_ratio(
    risk_idx: list[int], retention_ratio: Mapping[str, int]
) -> tuple[list[int], list[list[int]]]:
    """Walk `risk_idx` in order, starting a new group whenever the class
    changes or the current group has reached that class's ratio.

    Returns `(group_of_token, group_tokens)`: `group_of_token[i]` is the
    compressed group index original token `i` merged into (this is exactly
    `retention_map`'s per-token semantics); `group_tokens[g]` lists the
    original token indices merged into group `g`, in order.
    """
    group_of_token: list[int] = []
    group_tokens: list[list[int]] = []
    current_class: int | None = None
    for token_index, risk_class_index in enumerate(risk_idx):
        ratio = retention_ratio[RISK_CLASSES[risk_class_index]]
        starts_new_group = (
            risk_class_index != current_class or len(group_tokens[-1]) >= ratio
        )
        if starts_new_group:
            group_tokens.append([])
            current_class = risk_class_index
        group_tokens[-1].append(token_index)
        group_of_token.append(len(group_tokens) - 1)
    return group_of_token, group_tokens

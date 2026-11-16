"""ADRF reasoning path composed onto a `SparseExpertAdapter` -- W3-5.

RFC004-ERRATA-005 E1 §22.3 physically combines the sparse branch and the
reasoning path in one formula::

    h' = h + alpha_shared * E_shared(h) + sum_{i in S} w_i * E_i(h) + g(h) * R_spec(h, V)

`SparseExpertAdapter.forward(h)` already computes the first three terms
(``h + alpha_shared * E_shared(h) + sum w_i * E_i(h)``); this module adds the
fourth term, `g(h) * R_spec(h, V)`, as a residual on top of that output,
without touching `SparseExpertAdapter` itself.

`g` and `R_spec` both take the *same* pre-adapter hidden state `h` as their
input, exactly as the formula states -- not the sparse adapter's own output
-- so the gate and the specialization observe the branch's state at the
point uvažovanie attaches, per E1 rule 6 ("vstupy brány a modulu sú
obmedzené rovnako ako vstupy routera").

Normative rules this module is responsible for (RFC004-ERRATA-005 E1):

- **Rule 1 (architectural bypass).** `enabled=False` skips the reasoning
  module call entirely and returns the bare `SparseExpertAdapter` output --
  bit-identical to a pipeline with no reasoning module wired in at all.
- **Rule 4 (init equals bypass).** `reasoning=on` at step 0 equals
  `reasoning=off` bit-for-bit, because every registered specialization ends
  in a `ZeroInitOutputProjection` (E1 rule 4; `HardReasoningGate`'s
  docstring; enforced by the specialization itself, not here).
- **Rule 9 (sparse-only attachment).** Construction refuses a
  `DenseExpertAdapter` with a hard `TypeError` -- reasoning plugs only into
  the sparse branch, never the dense adapter (§61.20).

Nothing in this module is imported by `candidate_model`, `expert_adapter.py`
or any construction path -- like `sparse_expert_adapter.adapter` and
`reasoning.*`, it is dormant per DEC-047 / RFC004-ERRATA-005: default off,
no hashed identity moved, no gradient runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from unflatten.model.expert_adapter import DenseExpertAdapter
from unflatten.model.reasoning.gate import HardReasoningGate
from unflatten.model.reasoning.module import ReasoningTelemetry
from unflatten.model.reasoning.registry import instantiate as instantiate_reasoning
from unflatten.model.sparse_expert_adapter.adapter import SparseExpertAdapter


class DenseAdapterNotSupportedError(TypeError):
    """A `DenseExpertAdapter` was passed where a `SparseExpertAdapter` is required.

    E1 rule 9: "Dense profil uvažovanie nemá." Reasoning is a conditional
    path inside the sparse Word Expert Adapter only.
    """


@dataclass(frozen=True)
class SparseReasoningResult:
    """One `SparseExpertAdapterWithReasoning.forward` call's output.

    `hidden` is `h'` from the E1 §22.3 formula, already including the sparse
    branch's own terms. `telemetry` is always produced -- with `off`, it
    still reports `gate_activated=False` so a caller can tell "reasoning
    wired in but idle" apart from "no reasoning path at all" (E3 §22.7).
    """

    hidden: Tensor
    telemetry: ReasoningTelemetry


class SparseExpertAdapterWithReasoning(nn.Module):
    """Wraps a `SparseExpertAdapter` with an optional ADRF reasoning residual.

    `spec_id` is instantiated eagerly and explicitly through the reasoning
    registry (`reasoning.<family>.<version>`) -- there is no default
    specialization (E1 rule 7), so callers must always name one. Tests use
    only the package's `FixtureReasoningModule` test fixture, never a real
    specialization (none is ratified yet).
    """

    def __init__(
        self,
        sparse_adapter: SparseExpertAdapter,
        *,
        spec_id: str,
        attach_point: str,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(sparse_adapter, DenseExpertAdapter):
            raise DenseAdapterNotSupportedError(
                "reasoning attaches only to a SparseExpertAdapter, never to "
                "DenseExpertAdapter (RFC004-ERRATA-005 E1 rule 9)"
            )
        if not isinstance(sparse_adapter, SparseExpertAdapter):
            raise TypeError(
                f"sparse_adapter must be a SparseExpertAdapter, got {type(sparse_adapter)!r}"
            )
        self.sparse_adapter = sparse_adapter
        self.attach_point = attach_point
        self.enabled = enabled
        self.reasoning_module = instantiate_reasoning(spec_id)
        self.gate = HardReasoningGate(
            d_model=sparse_adapter.d_model, attach_point=attach_point, enabled=enabled
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        visual_state: Tensor,
        attention_mask: Tensor,
        eligibility_mask: Tensor,
        context: Mapping[str, Any] | None = None,
        sparse_context: str | None = None,
    ) -> SparseReasoningResult:
        """`hidden`: `[..., d_model]`, the pre-adapter branch state `h`.

        `sparse_context` is forwarded to `SparseExpertAdapter.route`'s
        grammar context (§22.5); `context` is the reasoning module's own
        context mapping (E1 §19), kept distinct because the two components
        have separate contracts.
        """
        sparse_out = self.sparse_adapter(hidden, context=sparse_context)

        if not self.enabled:
            # Rule 1: the bypass never calls the reasoning module at all, so
            # its output is bit-identical to the bare SparseExpertAdapter.
            telemetry = ReasoningTelemetry(
                gate_activated=False,
                eligibility_masked=bool((eligibility_mask == 0).any()),
                attach_point=self.attach_point,
            )
            return SparseReasoningResult(hidden=sparse_out, telemetry=telemetry)

        gate_decision = self.gate(hidden, eligibility_mask)
        result = self.reasoning_module.forward(
            hidden, visual_state, attention_mask, eligibility_mask, context or {}
        )
        gated_residual = gate_decision.unsqueeze(-1) * result.residual
        out = sparse_out + gated_residual
        return SparseReasoningResult(hidden=out, telemetry=result.telemetry)

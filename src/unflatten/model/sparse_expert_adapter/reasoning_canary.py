"""R-A contract canary -- RFC005-ERRATA-007 E2 R-A (`child_run_kind =
diagnostic_canary`, "ziadne tvrdenie o kvalite").

R-A is the *first* S6R sub-step: before any specialization weight is
trained, this harness proves the wiring itself is sound over a tiny
`SparseExpertAdapterWithReasoning` -- CPU only, no gradient, no quality
claim. It checks exactly the four things R-A names:

1. **Bypass identity** -- `reasoning = off` is bit-identical to the bare
   `SparseExpertAdapter` (RFC004-ERRATA-005 E1 rule 1).
2. **Step 0 == off** -- a freshly instantiated `reasoning = on` module
   matches `reasoning = off` bit-for-bit at construction (E1 rule 4, zero-
   init output projection).
3. **Eligibility respected** -- an all-zero `eligibility_mask` forces the
   gate closed and reproduces the bare sparse-adapter output, mirroring
   check 1 through the real forward path rather than a fresh twin.
4. **Gate telemetry non-degenerate** -- across a few caller-supplied
   synthetic perturbations, the gate's activation is neither always-off,
   always-on, nor dead on any perturbation's slice (`reasoning.telemetry`).
5. **Missing weight is a hard error** -- `reasoning.lock
   .verify_reasoning_component_weights` actually raises when a component
   lock declares an active reasoning component with no matching weight
   (E3, §23).

This module never trains anything and never claims quality; it only
reports whether the wiring contract holds. It refuses outright to run
against anything that is not a `SparseExpertAdapterWithReasoning` --
including a bare `DenseExpertAdapter`, which the constructor it wraps
already refuses to accept (E1 rule 9).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from unflatten.model.reasoning.gate import HardReasoningGate
from unflatten.model.reasoning.lock import (
    MissingReasoningComponentWeightError,
    verify_reasoning_component_weights,
)
from unflatten.model.reasoning.telemetry import ReasoningRecord, ReasoningTelemetryAggregator
from unflatten.model.sparse_expert_adapter.reasoning_integration import (
    SparseExpertAdapterWithReasoning,
)

#: RFC005-ERRATA-007 E2 R-A: "child_run_kind = diagnostic_canary, ziadne
#: tvrdenie o kvalite" -- a fixed label the report always carries.
R_A_CHILD_RUN_KIND = "diagnostic_canary"


@dataclass(frozen=True)
class CanaryCheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RAContractCanaryReport:
    child_run_kind: str
    results: tuple[CanaryCheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def failures(self) -> tuple[CanaryCheckResult, ...]:
        return tuple(result for result in self.results if not result.passed)


@dataclass(frozen=True)
class GatePerturbation:
    """One synthetic input for the gate-telemetry check.

    `eligibility_mask` and `hidden` share the same leading (non-`d_model`)
    shape, exactly like `HardReasoningGate.forward`'s own contract.
    """

    slice_id: str
    hidden: Tensor
    eligibility_mask: Tensor


def check_bypass_bit_identical(
    module: SparseExpertAdapterWithReasoning,
    hidden: Tensor,
    visual_state: Tensor,
    attention_mask: Tensor,
    eligibility_mask: Tensor,
) -> CanaryCheckResult:
    """RFC004-ERRATA-005 E1 rule 1: `reasoning = off` == bare `SparseExpertAdapter`.

    Builds a disabled twin sharing `module.sparse_adapter`'s exact weight
    object (not a copy), so this is a real bit-for-bit comparison against
    the module under test, not against an independently seeded adapter.
    """
    bare_output = module.sparse_adapter(hidden, context=None)
    disabled_twin = SparseExpertAdapterWithReasoning(
        module.sparse_adapter,
        spec_id=module.reasoning_module.spec_id,
        attach_point=module.attach_point,
        enabled=False,
    )
    result = disabled_twin(
        hidden,
        visual_state=visual_state,
        attention_mask=attention_mask,
        eligibility_mask=eligibility_mask,
    )
    ok = torch.equal(result.hidden, bare_output)
    return CanaryCheckResult(
        "bypass_bit_identical",
        ok,
        "reasoning=off must reproduce the bare SparseExpertAdapter output exactly",
    )


def check_step0_on_equals_off(
    module: SparseExpertAdapterWithReasoning,
    hidden: Tensor,
    visual_state: Tensor,
    attention_mask: Tensor,
    eligibility_mask: Tensor,
) -> CanaryCheckResult:
    """RFC004-ERRATA-005 E1 rule 4: a freshly instantiated `reasoning = on`
    twin matches `reasoning = off` bit-for-bit at construction.

    Both twins share `module.sparse_adapter`'s weights; `on`'s reasoning
    module is instantiated fresh here (not reused from `module`), so this
    check exercises the specialization's own zero-init contract rather than
    whatever state `module.reasoning_module` happens to carry.
    """
    spec_id = module.reasoning_module.spec_id
    off = SparseExpertAdapterWithReasoning(
        module.sparse_adapter, spec_id=spec_id, attach_point=module.attach_point, enabled=False
    )
    fresh_on = SparseExpertAdapterWithReasoning(
        module.sparse_adapter, spec_id=spec_id, attach_point=module.attach_point, enabled=True
    )
    kwargs: dict[str, Any] = dict(
        visual_state=visual_state, attention_mask=attention_mask, eligibility_mask=eligibility_mask
    )
    off_result = off(hidden, **kwargs)
    on_result = fresh_on(hidden, **kwargs)
    ok = torch.equal(on_result.hidden, off_result.hidden)
    return CanaryCheckResult(
        "step0_on_equals_off",
        ok,
        "a freshly instantiated reasoning=on module must equal reasoning=off at step 0",
    )


def check_eligibility_respected(
    module: SparseExpertAdapterWithReasoning,
    hidden: Tensor,
    visual_state: Tensor,
    attention_mask: Tensor,
) -> CanaryCheckResult:
    """An all-zero `eligibility_mask` must close the gate on `module` itself
    (not a twin), reproducing the bare sparse-adapter output exactly."""
    eligibility_mask = torch.zeros(hidden.shape[:-1])
    bare_output = module.sparse_adapter(hidden, context=None)
    result = module(
        hidden,
        visual_state=visual_state,
        attention_mask=attention_mask,
        eligibility_mask=eligibility_mask,
    )
    ok = torch.equal(result.hidden, bare_output)
    return CanaryCheckResult(
        "eligibility_respected",
        ok,
        "all-zero eligibility_mask must force the gate closed on every slot",
    )


def check_gate_telemetry_non_degenerate(
    gate: HardReasoningGate, perturbations: Sequence[GatePerturbation]
) -> CanaryCheckResult:
    """Gate activation across `perturbations` must be neither always-off,
    always-on, nor dead on any perturbation's slice (RFC005-ERRATA-007 E2
    R-A "telemetria ... nie je degenerovana", read through
    `ReasoningTelemetryAggregator`)."""
    aggregator = ReasoningTelemetryAggregator()
    gate.eval()
    with torch.no_grad():
        for perturbation in perturbations:
            probs = torch.sigmoid(gate.gate_logit_proj(perturbation.hidden)).squeeze(-1)
            decisions = gate(perturbation.hidden, perturbation.eligibility_mask)
            flat_probs = probs.reshape(-1).tolist()
            flat_decisions = decisions.reshape(-1).tolist()
            flat_eligible = perturbation.eligibility_mask.reshape(-1).tolist()
            for probability, decision, eligible in zip(
                flat_probs, flat_decisions, flat_eligible, strict=True
            ):
                try:
                    aggregator.record(
                        ReasoningRecord(
                            activated=bool(decision == 1),
                            eligible=bool(eligible != 0),
                            gate_probability=float(probability),
                            slice_id=perturbation.slice_id,
                        )
                    )
                except ValueError as exc:
                    # An activation on an ineligible slot is itself a gate
                    # defect (rule 6) -- report it as a failed check rather
                    # than letting it crash the whole canary run.
                    return CanaryCheckResult(
                        "gate_telemetry_non_degenerate",
                        False,
                        f"gate produced an invalid record: {exc}",
                    )
    report = aggregator.report()
    ok = (
        report.eligible_count > 0
        and 0.0 < report.activation_rate_overall < 1.0
        and not report.dead_paths
    )
    detail = (
        f"eligible_count={report.eligible_count} "
        f"activation_rate_overall={report.activation_rate_overall:.4f} "
        f"dead_paths={sorted(report.dead_paths)}"
    )
    return CanaryCheckResult("gate_telemetry_non_degenerate", ok, detail)


def check_missing_weight_hard_error(
    component_lock: Mapping[str, Any], available_weight_hashes: Iterable[str]
) -> CanaryCheckResult:
    """RFC004-ERRATA-005 E3 (§23): a declared active reasoning component
    with no matching weight must be a hard load error, never a silent
    skip. `component_lock` must actually declare such a component for this
    check to be meaningful -- it asserts the *raise*, not just its absence.
    """
    try:
        verify_reasoning_component_weights(component_lock, available_weight_hashes)
    except MissingReasoningComponentWeightError:
        return CanaryCheckResult(
            "missing_weight_hard_error",
            True,
            "a declared active reasoning component with no matching weight raised "
            "MissingReasoningComponentWeightError as required",
        )
    return CanaryCheckResult(
        "missing_weight_hard_error",
        False,
        "expected MissingReasoningComponentWeightError but none was raised -- "
        "component_lock must declare an active reasoning component whose weight "
        "is absent from available_weight_hashes",
    )


def run_r_a_contract_canary(
    module: SparseExpertAdapterWithReasoning,
    *,
    hidden: Tensor,
    visual_state: Tensor,
    attention_mask: Tensor,
    eligibility_mask: Tensor,
    gate_perturbations: Sequence[GatePerturbation],
    component_lock: Mapping[str, Any],
    available_weight_hashes: Iterable[str] = (),
) -> RAContractCanaryReport:
    """Run every R-A check over `module` and return the combined report.

    Refuses outright (`TypeError`) if `module` is not a
    `SparseExpertAdapterWithReasoning` -- in particular a bare
    `DenseExpertAdapter`, which has no reasoning path to canary at all
    (RFC004-ERRATA-005 E1 rule 9).
    """
    if not isinstance(module, SparseExpertAdapterWithReasoning):
        raise TypeError(
            "run_r_a_contract_canary requires a SparseExpertAdapterWithReasoning; "
            f"got {type(module)!r} -- it refuses to run directly against a "
            "DenseExpertAdapter or any other module (RFC004-ERRATA-005 E1 rule 9)"
        )
    results = (
        check_bypass_bit_identical(module, hidden, visual_state, attention_mask, eligibility_mask),
        check_step0_on_equals_off(module, hidden, visual_state, attention_mask, eligibility_mask),
        check_eligibility_respected(module, hidden, visual_state, attention_mask),
        check_gate_telemetry_non_degenerate(module.gate, gate_perturbations),
        check_missing_weight_hard_error(component_lock, available_weight_hashes),
    )
    return RAContractCanaryReport(child_run_kind=R_A_CHILD_RUN_KIND, results=results)

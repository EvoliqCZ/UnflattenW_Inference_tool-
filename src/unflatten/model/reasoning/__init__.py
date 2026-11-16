"""Adaptive Document Reasoning Framework (ADRF) -- RFC004-ERRATA-005 E1, A6.

Wave 1 (RFC004-ERRATA-005 E6, §56 A6 exit: "vypnutá cesta vždy prejde a je
výstupovo totožná s cestou bez modulu") builds the interface, the
specialization registry, the hard gate with its architectural bypass, and
telemetry -- with **no concrete specialization**. Nothing in this package is
wired into `DenseExpertAdapter`, `candidate_model` forward, the
`trainable_group_registry`, or the `property_head_registry`. It is standalone
and unused by any production path: ADRF waits for sparse MoE (RFC005 S6) and
its own G2/G3 selection (RFC005-ERRATA-007).

Submodules:

- `module`: `ReasoningModule` protocol, `ReasoningResult`, `ReasoningTelemetry`
  (E1 §19 contract, ~RFC004-ERRATA-005.md:150-163).
- `registry`: `reasoning.<family>.<version>` specialization registry that
  refuses `continuation.*` (E1 rules 7-8) and instantiates only an explicitly
  selected id -- no default, no auto-discovery.
- `gate`: `HardReasoningGate`, the learned hard gate with a straight-through
  estimator (E1 variant 1), the `reasoning=off` architectural bypass (rule 1),
  and zero-init output projection (rule 4).
- `telemetry`: `ReasoningTelemetryAggregator`, a pure aggregator over
  per-call `ReasoningTelemetry` records (E3, §22.7/§43.1).
- `lock`: the four new Component Lock kinds (`reasoning_module`,
  `reasoning_gate`, `pre_encoder_compressor`, `visual_compressor`) and the
  missing-weight hard load error (E3 §39.2).
"""

from unflatten.model.reasoning.gate import HardReasoningGate
from unflatten.model.reasoning.module import (
    ReasoningModule,
    ReasoningResult,
    ReasoningTelemetry,
)
from unflatten.model.reasoning.registry import (
    ReasoningSpecializationNotRegistered,
    ReasoningSpecializationRejected,
    instantiate,
    register,
)
from unflatten.model.reasoning.telemetry import ReasoningTelemetryAggregator

__all__ = [
    "HardReasoningGate",
    "ReasoningModule",
    "ReasoningResult",
    "ReasoningTelemetry",
    "ReasoningSpecializationNotRegistered",
    "ReasoningSpecializationRejected",
    "ReasoningTelemetryAggregator",
    "instantiate",
    "register",
]

"""RFC005-ERRATA-007 E5 VC-5: collect evidence and call the gate, unchanged.

This driver's only job is composition: it gathers the architecture-contract
report and the per-slice/safety/performance metric maps a real S7 run would
produce, and hands them to
:func:`unflatten.model.visual_compression.vc5_gate_report.build_vc5_gate_report`
verbatim. It makes no gating decision of its own -- duplicating that logic
here would be a second, driftable copy of the one erratum-mandated gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from unflatten.model.visual_compression.architecture_contract import ArchitectureContractReport
from unflatten.model.visual_compression.vc5_gate_report import build_vc5_gate_report


def run_vc5_gate(
    *,
    architecture_contract_report: ArchitectureContractReport,
    bypass_quality_metrics: Mapping[str, float] | None,
    compressed_quality_metrics: Mapping[str, float] | None,
    safety_metrics: Mapping[str, float] | None,
    performance_deltas: Mapping[str, float] | None,
    vc5_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Thin pass-through to :func:`build_vc5_gate_report` -- kept as its own
    entry point so a caller (CLI, notebook, later CI step) has one stable
    place to collect evidence from, without importing the gate module's
    internal condition functions directly."""
    return build_vc5_gate_report(
        architecture_contract_report=architecture_contract_report,
        bypass_quality_metrics=bypass_quality_metrics,
        compressed_quality_metrics=compressed_quality_metrics,
        safety_metrics=safety_metrics,
        performance_deltas=performance_deltas,
        vc5_policy=vc5_policy,
    )

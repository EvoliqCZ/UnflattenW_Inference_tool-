"""VC-5 promotion gate -- RFC005-ERRATA-007 E5.

Four conditions, in the erratum's own order: architecture, quality, safety,
performance. **Architecture dominates**: "Zlyhanie = zamietnutie bez ohladu
na agregatnu metriku" -- a failing `ArchitectureContractReport` (reused
verbatim from `architecture_contract.py`, never recomputed here) forces
`gate_status = "blocked"` regardless of how good the other three conditions
look, and this module reports that fact rather than let a caller average
across conditions.

Every other condition follows the same `insufficient_evidence` discipline as
`s4_stage_gate_report.py` and `g3_reasoning_selection_gate_report.py`: a
threshold still marked `is_placeholder` in `vc5_policy` cannot produce a
`pass`, per the erratum's own "prahy zmrazene pred prvym gradientom".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from unflatten.model.visual_compression.architecture_contract import ArchitectureContractReport

#: RFC005-ERRATA-007 E5, "Kvalita, per-slice non-inferiority voci obtoku".
REQUIRED_QUALITY_SLICES: tuple[str, ...] = (
    "strict_validity",
    "exact_visible_text",
    "small_font",
    "numbers",
    "diacritics",
    "punctuation",
    "table_structure_incl_borderless",
    "paragraph_run_style",
    "image_boundaries",
    "node_visual_aligner_quality",
    "visual_closeness_word",
)

#: DEC-018 accepted ceiling: compression must not push these below the floor
#: already accepted for the primary model, even though it need not improve them.
DEC018_FLOOR_SLICES: tuple[str, ...] = ("spacing", "font_size")

_SAFETY_CEILINGS: tuple[str, ...] = (
    "bypass_rate",
    "protected_region_miss_rate",
    "calibration_error",
    "ood_degradation",
)

VC5_CONDITION_NAMES = ("architecture_contract", "quality_non_inferiority", "safety", "performance")

_BLOCKED_STATUS = "blocked"
_PROMOTED_STATUS = "promoted"


def _condition(name: str, status: str, reason: str) -> dict[str, Any]:
    return {"condition": name, "status": status, "reason": reason}


def _threshold_value(policy: Mapping[str, Any], key: str) -> tuple[Any, str | None]:
    entry = policy.get(key)
    if entry is None:
        return None, f"vc5_policy has no {key!r} threshold"
    if entry.get("is_placeholder", True):
        return None, f"{key!r} threshold is still a placeholder, not frozen before S7"
    return entry.get("value"), None


def condition_architecture_contract(report: ArchitectureContractReport) -> dict[str, Any]:
    """RFC005 §21.2 architecture gate, reused verbatim -- never recomputed."""
    name = "architecture_contract"
    if report.passed:
        return _condition(name, "pass", "architecture contract holds")
    failing = sorted(r.name for r in report.failures())
    return _condition(name, "fail", f"architecture contract failed: {failing}")


def condition_quality_non_inferiority(
    *,
    bypass_metrics: Mapping[str, float] | None,
    compressed_metrics: Mapping[str, float] | None,
    vc5_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Per-slice non-inferiority vs bypass, plus the DEC-018 accepted-ceiling
    floors, which are absolute rather than relative-to-bypass."""
    name = "quality_non_inferiority"
    margin, margin_insufficient = _threshold_value(vc5_policy, "quality_margin")
    floor, floor_insufficient = _threshold_value(vc5_policy, "dec018_floor")
    if margin_insufficient is not None:
        return _condition(name, "insufficient_evidence", margin_insufficient)
    if floor_insufficient is not None:
        return _condition(name, "insufficient_evidence", floor_insufficient)
    if not bypass_metrics or not compressed_metrics:
        return _condition(
            name, "insufficient_evidence", "bypass or compressed quality metrics not measured"
        )
    missing = sorted(
        slice_id
        for slice_id in (*REQUIRED_QUALITY_SLICES, *DEC018_FLOOR_SLICES)
        if slice_id not in bypass_metrics or slice_id not in compressed_metrics
    )
    if missing:
        return _condition(
            name, "insufficient_evidence", f"missing metric(s) for slice(s): {missing}"
        )
    regressed = sorted(
        slice_id
        for slice_id in REQUIRED_QUALITY_SLICES
        if (bypass_metrics[slice_id] - compressed_metrics[slice_id]) > margin
    )
    if regressed:
        return _condition(
            name, "fail", f"slices regressed beyond margin {margin:.4f} vs bypass: {regressed}"
        )
    below_floor = sorted(
        slice_id for slice_id in DEC018_FLOOR_SLICES if compressed_metrics[slice_id] < floor
    )
    if below_floor:
        return _condition(
            name,
            "fail",
            f"compression pushed the DEC-018 accepted ceiling below its floor: {below_floor}",
        )
    return _condition(
        name, "pass", "every required slice is non-inferior and both DEC-018 floors hold"
    )


def condition_safety(
    *, safety_metrics: Mapping[str, float] | None, vc5_policy: Mapping[str, Any]
) -> dict[str, Any]:
    """ "miera obtoku, miera minutia chranenych regionov, kalibracia
    neistoty, degradacia mimo distribucie" -- each a ceiling not to exceed."""
    name = "safety"
    if not safety_metrics:
        return _condition(name, "insufficient_evidence", "no safety metrics supplied")
    missing_metrics = sorted(m for m in _SAFETY_CEILINGS if safety_metrics.get(m) is None)
    if missing_metrics:
        return _condition(
            name, "insufficient_evidence", f"missing safety metric(s): {missing_metrics}"
        )
    exceeded: list[str] = []
    insufficient_thresholds: list[str] = []
    for metric in _SAFETY_CEILINGS:
        ceiling, insufficient = _threshold_value(vc5_policy, f"{metric}_ceiling")
        if insufficient is not None:
            insufficient_thresholds.append(insufficient)
            continue
        if safety_metrics[metric] > ceiling:
            exceeded.append(metric)
    if insufficient_thresholds:
        return _condition(name, "insufficient_evidence", "; ".join(insufficient_thresholds))
    if exceeded:
        return _condition(name, "fail", f"safety ceiling(s) exceeded: {sorted(exceeded)}")
    return _condition(
        name,
        "pass",
        "bypass rate, protected-region miss rate, calibration and OOD all within ceiling",
    )


def condition_performance(
    *, performance_deltas: Mapping[str, float] | None, vc5_policy: Mapping[str, Any]
) -> dict[str, Any]:
    """ "Prinos musi prekrocit vopred deklarovanu materialnu hranicu.
    Kompresia bez merateľneho prinosu sa neinstaluje."""
    name = "performance"
    materiality_line, insufficient = _threshold_value(vc5_policy, "materiality_line")
    if insufficient is not None:
        return _condition(name, "insufficient_evidence", insufficient)
    if not performance_deltas:
        return _condition(name, "insufficient_evidence", "no performance deltas measured")
    best_metric = max(performance_deltas, key=lambda m: performance_deltas[m])
    if performance_deltas[best_metric] < materiality_line:
        return _condition(
            name,
            "fail",
            f"no metric clears the materiality line {materiality_line:.4f} "
            f"(best={best_metric}={performance_deltas[best_metric]:.4f})",
        )
    return _condition(
        name,
        "pass",
        f"{best_metric} improved by {performance_deltas[best_metric]:.4f}, above materiality line",
    )


def build_vc5_gate_report(
    *,
    architecture_contract_report: ArchitectureContractReport,
    bypass_quality_metrics: Mapping[str, float] | None,
    compressed_quality_metrics: Mapping[str, float] | None,
    safety_metrics: Mapping[str, float] | None,
    performance_deltas: Mapping[str, float] | None,
    vc5_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the full VC-5 report. A failing architecture contract makes
    `gate_status = "blocked"` unconditionally, before the other three
    conditions are even allowed to matter (RFC005-ERRATA-007 E5)."""
    architecture = condition_architecture_contract(architecture_contract_report)
    quality = condition_quality_non_inferiority(
        bypass_metrics=bypass_quality_metrics,
        compressed_metrics=compressed_quality_metrics,
        vc5_policy=vc5_policy,
    )
    safety = condition_safety(safety_metrics=safety_metrics, vc5_policy=vc5_policy)
    performance = condition_performance(
        performance_deltas=performance_deltas, vc5_policy=vc5_policy
    )
    conditions = [architecture, quality, safety, performance]

    if architecture["status"] != "pass":
        return {
            "gate": "VC-5",
            "gate_status": _BLOCKED_STATUS,
            "blocked_conditions": ["architecture_contract"],
            "dominated_by_architecture_contract": True,
            "conditions": conditions,
        }
    non_pass = [c["condition"] for c in conditions if c["status"] != "pass"]
    return {
        "gate": "VC-5",
        "gate_status": _PROMOTED_STATUS if not non_pass else _BLOCKED_STATUS,
        "blocked_conditions": non_pass,
        "dominated_by_architecture_contract": False,
        "conditions": conditions,
    }

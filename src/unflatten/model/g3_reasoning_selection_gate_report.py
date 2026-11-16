"""`G3_reasoning_selection` gate report -- RFC005-ERRATA-007 E2.

RFC005 §12.2's "absencia reportu znamena `blocked`, nie implicitny uspech"
applies here exactly as it does to S4's `stage_status`: this module is a
pure *consumer* of arm results (A/B/C and optional D), a
`ReasoningTelemetryReport` and a shortcut-audit result someone else already
produced, and never recomputes a metric itself. Every one of the erratum's
eight `AND`-joined G3 conditions is evaluated independently and reported
`pass` / `fail` / `insufficient_evidence`; a threshold that is still a named
placeholder (`is_placeholder: true` in `g3_policy`) makes its condition
`insufficient_evidence`, never a guessed `pass`.

`gate_status` is `"accepted"` only when every condition is `pass`; any
non-pass condition makes it `"blocked_g2_winner_stands"` -- the erratum's own
words: "Nesplnenie znamena, ze produktom ostava vitaz G2 bez uvazovania."
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

G3_CONDITION_NAMES = (
    "g2_hard_gates_pass_for_c",
    "strict_validity_not_regressing",
    "target_slice_improves",
    "non_target_slices_not_regressing",
    "gate_stable",
    "shortcut_audit_clean",
    "bypass_valid_control",
    "latency_vram_throughput_measured",
)

_BLOCKED_STATUS = "blocked_g2_winner_stands"
_ACCEPTED_STATUS = "accepted"


def _condition(name: str, status: str, reason: str) -> dict[str, Any]:
    return {"condition": name, "status": status, "reason": reason}


def _threshold_value(policy: Mapping[str, Any], key: str) -> tuple[Any, str | None]:
    """Read one named threshold from `policy`. Returns `(value, insufficient_reason)`;
    `insufficient_reason` is `None` iff the threshold is frozen and usable."""
    entry = policy.get(key)
    if entry is None:
        return None, f"g3_policy has no {key!r} threshold"
    if entry.get("is_placeholder", True):
        return None, f"{key!r} threshold is still a placeholder, not frozen before R-B"
    return entry.get("value"), None


def condition_g2_hard_gates_pass_for_c(arm_c_g2_gates: Mapping[str, str] | None) -> dict[str, Any]:
    """RFC005-ERRATA-007 E2 G3: "vsetky hard gates G2 platia aj pre C"."""
    name = "g2_hard_gates_pass_for_c"
    if not arm_c_g2_gates:
        return _condition(name, "insufficient_evidence", "arm C has no recorded G2 gate outcomes")
    failing = sorted(gate for gate, status in arm_c_g2_gates.items() if status != "pass")
    if failing:
        return _condition(name, "fail", f"arm C fails G2 gates: {failing}")
    return _condition(name, "pass", f"arm C passes all {len(arm_c_g2_gates)} G2 hard gates")


def condition_strict_validity_not_regressing(
    *,
    arm_b_strict_validity: float | None,
    arm_c_strict_validity: float | None,
    g3_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """ "strict validita neregreduje nad deklarovany okraj"."""
    name = "strict_validity_not_regressing"
    margin, insufficient = _threshold_value(g3_policy, "strict_validity_margin")
    if insufficient is not None:
        return _condition(name, "insufficient_evidence", insufficient)
    if arm_b_strict_validity is None or arm_c_strict_validity is None:
        return _condition(
            name, "insufficient_evidence", "arm B or arm C strict validity not measured"
        )
    regression = arm_b_strict_validity - arm_c_strict_validity
    if regression > margin:
        return _condition(
            name,
            "fail",
            f"strict validity regressed by {regression:.4f}, exceeding margin {margin:.4f}",
        )
    return _condition(
        name, "pass", f"strict validity regression {regression:.4f} <= margin {margin:.4f}"
    )


def condition_target_slice_improves(
    *,
    arm_b_target_metric: float | None,
    arm_c_target_metric: float | None,
    arm_d_target_metric: float | None,
    g3_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """"cielovy kriticky slice sa zlepsi nad absolutne dno a okraj proti B
    (a proti D, ak bezi)"."""
    name = "target_slice_improves"
    floor, floor_insufficient = _threshold_value(g3_policy, "target_slice_absolute_floor")
    margin, margin_insufficient = _threshold_value(g3_policy, "target_slice_margin")
    if floor_insufficient is not None:
        return _condition(name, "insufficient_evidence", floor_insufficient)
    if margin_insufficient is not None:
        return _condition(name, "insufficient_evidence", margin_insufficient)
    if arm_b_target_metric is None or arm_c_target_metric is None:
        return _condition(
            name, "insufficient_evidence", "arm B or arm C target slice metric not measured"
        )
    if arm_c_target_metric < floor:
        return _condition(
            name,
            "fail",
            f"arm C target slice metric {arm_c_target_metric:.4f} below absolute floor {floor:.4f}",
        )
    if arm_c_target_metric < arm_b_target_metric + margin:
        return _condition(
            name,
            "fail",
            f"arm C target slice metric {arm_c_target_metric:.4f} does not beat arm B "
            f"{arm_b_target_metric:.4f} by margin {margin:.4f}",
        )
    if arm_d_target_metric is not None and arm_c_target_metric < arm_d_target_metric + margin:
        return _condition(
            name,
            "fail",
            f"arm C target slice metric {arm_c_target_metric:.4f} does not beat arm D "
            f"{arm_d_target_metric:.4f} by margin {margin:.4f}",
        )
    return _condition(
        name, "pass", "arm C clears the absolute floor and the margin vs B (and D, if run)"
    )


def condition_non_target_slices_not_regressing(
    *,
    arm_b_non_target: Mapping[str, float] | None,
    arm_c_non_target: Mapping[str, float] | None,
    g3_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """ "necielove kriticke slice neregreduju nad okraj"."""
    name = "non_target_slices_not_regressing"
    margin, insufficient = _threshold_value(g3_policy, "non_target_slice_margin")
    if insufficient is not None:
        return _condition(name, "insufficient_evidence", insufficient)
    if not arm_b_non_target or not arm_c_non_target:
        return _condition(
            name, "insufficient_evidence", "arm B or arm C non-target slice metrics not measured"
        )
    missing = sorted(set(arm_b_non_target) - set(arm_c_non_target))
    if missing:
        return _condition(
            name, "insufficient_evidence", f"arm C has no metric for slices {missing}"
        )
    regressed = sorted(
        slice_id
        for slice_id, b_value in arm_b_non_target.items()
        if (b_value - arm_c_non_target[slice_id]) > margin
    )
    if regressed:
        return _condition(
            name, "fail", f"non-target slices regressed beyond margin {margin:.4f}: {regressed}"
        )
    return _condition(
        name,
        "pass",
        f"all {len(arm_b_non_target)} non-target slices stayed within margin {margin:.4f}",
    )


def condition_gate_stable(
    *,
    dead_paths: frozenset[str] | Sequence[str],
    activation_rate_overall: float | None,
    g3_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """ "brana je stabilna: ziadna mrtva cesta, ziadny kolaps, aktivacia v deklarovanom pasme"."""
    name = "gate_stable"
    if dead_paths:
        return _condition(name, "fail", f"gate has dead path(s): {sorted(dead_paths)}")
    if activation_rate_overall is None:
        return _condition(
            name, "insufficient_evidence", "gate activation_rate_overall not measured"
        )
    if activation_rate_overall in (0.0, 1.0):
        return _condition(
            name, "fail", f"gate collapsed to a constant decision (rate={activation_rate_overall})"
        )
    band, insufficient = _threshold_value(g3_policy, "activation_band")
    if insufficient is not None:
        return _condition(name, "insufficient_evidence", insufficient)
    low, high = band
    if not (low <= activation_rate_overall <= high):
        return _condition(
            name,
            "fail",
            f"activation_rate_overall {activation_rate_overall:.4f} "
            f"outside declared band [{low}, {high}]",
        )
    return _condition(
        name, "pass", f"no dead path, no collapse, activation {activation_rate_overall:.4f} in band"
    )


def condition_shortcut_audit_clean(
    shortcut_audit_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """ "audit skratiek (RFC004 §44.3) je cisty"."""
    name = "shortcut_audit_clean"
    if shortcut_audit_result is None:
        return _condition(name, "insufficient_evidence", "no shortcut-audit result supplied")
    status = shortcut_audit_result.get("status")
    if status == "clean":
        return _condition(name, "pass", "shortcut audit reports status=clean")
    if status is None:
        return _condition(
            name, "insufficient_evidence", "shortcut-audit result has no status field"
        )
    return _condition(name, "fail", f"shortcut audit reports status={status!r}")


def condition_bypass_valid_control(bypass_is_valid_control: bool | None) -> dict[str, Any]:
    """ "obtok ostava platnou kontrolou"."""
    name = "bypass_valid_control"
    if bypass_is_valid_control is None:
        return _condition(name, "insufficient_evidence", "bypass control validity not measured")
    if bypass_is_valid_control:
        return _condition(name, "pass", "bypass remains a valid control")
    return _condition(
        name, "fail", "bypass no longer reproduces a valid control (arm A/B parity broke)"
    )


def condition_latency_vram_throughput_measured(
    performance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """ "rezia latencie, VRAM a priepustnosti je zmerana a zapisana"."""
    name = "latency_vram_throughput_measured"
    required = ("latency_ms", "vram_bytes", "throughput_pages_per_min")
    if performance is None:
        return _condition(name, "insufficient_evidence", "no performance report supplied")
    missing = sorted(field for field in required if performance.get(field) is None)
    if missing:
        return _condition(
            name, "insufficient_evidence", f"performance report missing fields: {missing}"
        )
    return _condition(name, "pass", "latency, VRAM and throughput all recorded")


def build_g3_gate_report(
    *,
    arm_c_g2_gates: Mapping[str, str] | None,
    arm_b_strict_validity: float | None,
    arm_c_strict_validity: float | None,
    arm_b_target_metric: float | None,
    arm_c_target_metric: float | None,
    arm_d_target_metric: float | None,
    arm_b_non_target: Mapping[str, float] | None,
    arm_c_non_target: Mapping[str, float] | None,
    gate_dead_paths: frozenset[str] | Sequence[str],
    gate_activation_rate_overall: float | None,
    shortcut_audit_result: Mapping[str, Any] | None,
    bypass_is_valid_control: bool | None,
    performance: Mapping[str, Any] | None,
    g3_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the full G3 report. `gate_status` is `"accepted"` only when
    every condition is `pass`; otherwise `"blocked_g2_winner_stands"`, with
    the offending conditions named."""
    conditions = [
        condition_g2_hard_gates_pass_for_c(arm_c_g2_gates),
        condition_strict_validity_not_regressing(
            arm_b_strict_validity=arm_b_strict_validity,
            arm_c_strict_validity=arm_c_strict_validity,
            g3_policy=g3_policy,
        ),
        condition_target_slice_improves(
            arm_b_target_metric=arm_b_target_metric,
            arm_c_target_metric=arm_c_target_metric,
            arm_d_target_metric=arm_d_target_metric,
            g3_policy=g3_policy,
        ),
        condition_non_target_slices_not_regressing(
            arm_b_non_target=arm_b_non_target,
            arm_c_non_target=arm_c_non_target,
            g3_policy=g3_policy,
        ),
        condition_gate_stable(
            dead_paths=gate_dead_paths,
            activation_rate_overall=gate_activation_rate_overall,
            g3_policy=g3_policy,
        ),
        condition_shortcut_audit_clean(shortcut_audit_result),
        condition_bypass_valid_control(bypass_is_valid_control),
        condition_latency_vram_throughput_measured(performance),
    ]
    non_pass = [c["condition"] for c in conditions if c["status"] != "pass"]
    gate_status = _ACCEPTED_STATUS if not non_pass else _BLOCKED_STATUS
    return {
        "gate": "G3_reasoning_selection",
        "gate_status": gate_status,
        "blocked_conditions": non_pass,
        "conditions": conditions,
    }

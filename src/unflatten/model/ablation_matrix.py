"""RFC004 §52.1 mandatory ablation switch matrix, §52.2 comparison
invariants -- closed schema, the canonical matrix, and a validator.

RFC004 §52 is explicit that it "neurčuje povinné experimentálne runs ani
ich výsledkové gates" (names no required runs or result gates) -- RFC005
owns which runs happen, dataset mix, metrics, thresholds and statistical
protocol. This module only proves the §52.1 switches are declared and
independently configurable at a fixed output contract, and that a
comparison between two configurations does not silently change one of the
§52.2 invariants or more than one switch at once.

`ABLATION_MATRIX`'s per-switch `status` is a snapshot of *this codebase
today*, not an RFC decision: `not_available` names a switch whose
experimental variant has no implementation yet (e.g. `moe` -- no MoE
router exists outside RFC004 design text; `speculative_decode` -- no
`SpeculativeDrafter` implementation exists yet, per CLAUDE.md "architecture
only needs a pluggable `SpeculativeDrafter` interface"). Declaring these
rows keeps the switch_id enum closed and complete rather than shrinking it
to "whatever happens to be built" (task instruction). `available` rows
cite the modules backing both variants in `label`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.ast.issues import Issue

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_SCHEMA_NAME = "ablation_matrix"

ABLATION_MATRIX_SCHEMA_VERSION = "0.1.0"

#: RFC004 §52.2's seven comparison invariants, exact set and order.
COMPARISON_INVARIANTS: tuple[str, ...] = (
    "dataset_split",
    "target_semantics",
    "head_registry",
    "assembler_policy",
    "renderer_truth_profile",
    "evaluation_cohort",
    "unrelated_training_budget",
)

#: M1-internal codes (§9.3-style); not RFC004 §42.4 registry codes -- §52
#: names no ablation-comparison error code of its own.
ABLATION_INVARIANT_CHANGED = "ABLATION_INVARIANT_CHANGED"
ABLATION_NOT_ISOLATED = "ABLATION_NOT_ISOLATED"
ABLATION_SWITCH_UNAVAILABLE = "ABLATION_SWITCH_UNAVAILABLE"


class AblationMatrixInvalid(ValueError):
    """The ablation matrix fails its closed schema (RFC004 §52.1)."""


def _variant(value_id: str, label: str) -> dict[str, str]:
    return {"value_id": value_id, "label": label}


def _switch(
    switch_id: str,
    primary_question: str,
    control: tuple[str, str],
    experimental: tuple[str, str],
    status: str,
) -> dict[str, Any]:
    return {
        "switch_id": switch_id,
        "primary_question": primary_question,
        "control": _variant(*control),
        "experimental": _variant(*experimental),
        "status": status,
    }


#: The canonical RFC004 §52.1 matrix, current-codebase snapshot (module
#: docstring). Row order matches the RFC table.
ABLATION_MATRIX: dict[str, Any] = {
    "ablation_matrix_schema_version": ABLATION_MATRIX_SCHEMA_VERSION,
    "switches": [
        _switch(
            "backbone",
            "Kvalita vs compute pri rovnakom output contracte",
            ("monkeyocr_bridge", "MonkeyOCR bridge (RFC004 §47.1)"),
            ("dots_or_qwen3_adapter", "dots.mocr / Qwen3-VL adapter (RFC004 §47.2-47.3)"),
            "not_available",
        ),
        _switch(
            "output_representation",
            "Validita, token budget, typed accuracy",
            (
                "typed_structural_text_heads",
                "typed structural/text/heads (typed_property_heads.py)",
            ),
            ("monolithic_json_baseline", "monolithic serialized JSON baseline"),
            "control_only",
        ),
        _switch(
            "moe",
            "Specialization bez router collapse",
            ("compute_matched_dense", "compute-matched dense (expert_adapter.py)"),
            ("shared_sparse_specialists", "shared + sparse specialists (MoE router)"),
            "not_available",
        ),
        _switch(
            "visual_compression",
            "Zachováva fine visual signals?",
            ("bypass", "bypass (visual_memory.py passthrough)"),
            ("compressor_on", "compressor on (RFC004 §15 VisualCompressor)"),
            "not_available",
        ),
        _switch(
            "aligner",
            "Prínos a end-to-end gap",
            ("production_predicted", "production predicted (node_visual_aligner.py)"),
            ("oracle_or_no_aligner", "no aligner / oracle diagnostic"),
            "control_only",
        ),
        _switch(
            "context",
            "Multi-page continuity",
            ("local_only", "local only (no context encoder attached)"),
            ("global_context", "global page/document context (document_context_encoder.py)"),
            "available",
        ),
        _switch(
            "fragmentation",
            "Fidelity a determinizmus",
            ("full_document", "full-document (input_contract.InputMode 'full_document')"),
            ("bounded_windows_assembler", "bounded windows + assembler (Track A)"),
            "available",
        ),
        _switch(
            "typed_values",
            "Numeric validity a calibration",
            ("direct_regression", "direct regression (typed_property_heads.py)"),
            ("bin_residual_codec", "bin + residual/structured codec"),
            "control_only",
        ),
        _switch(
            "table",
            "Grid/span validity",
            ("generic_heads", "generic heads (typed_property_heads.py)"),
            ("table_plan_constraints", "table plan + constraints (table_plan.py, RFC004 §53)"),
            "available",
        ),
        _switch(
            "speculative_decode",
            "Lossless output + latency",
            ("off", "off (no speculative drafter attached)"),
            ("dflash_or_mtp", "DFlash alebo MTP (SpeculativeDrafter, not yet implemented)"),
            "not_available",
        ),
        _switch(
            "resource_recovery",
            "Image fidelity a failure rate",
            ("crop_baseline", "crop baseline (resource_recovery.py)"),
            (
                "mask_rectification_reconstruction",
                "mask/rectification/reconstruction "
                "(recovered-image-materialization.v0, RFC004-ERRATA-004 E1)",
            ),
            "available",
        ),
    ],
    "comparison_invariants": list(COMPARISON_INVARIANTS),
}


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate_ablation_matrix(matrix: Mapping[str, Any]) -> None:
    """Validate `matrix` against `ablation_matrix.schema.json`, plus the
    one closure rule the schema's enum already implies but this makes
    explicit: `switches` must name each of the 11 `switch_id`s exactly
    once (no duplicate, no gap)."""
    errors = list(_validator().iter_errors(matrix))
    if errors:
        raise AblationMatrixInvalid(
            f"schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )
    seen = [row["switch_id"] for row in matrix["switches"]]
    if len(set(seen)) != len(seen):
        raise AblationMatrixInvalid(f"duplicate switch_id(s) in switches: {seen}")


def _switch_by_id(matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["switch_id"]: row for row in matrix["switches"]}


def check_comparison(
    matrix: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    invariants: Sequence[str] = COMPARISON_INVARIANTS,
) -> list[Issue]:
    """§52.2: verify one ablation comparison changes exactly one switch and
    none of `invariants`.

    `baseline`/`candidate` each carry the `invariants` keys plus
    `active_switch_values: {switch_id: value_id}` covering every switch in
    `matrix`. Returns an empty list when the comparison is isolated;
    never raises on a bad comparison (RFC001 §50 style) -- only a
    malformed `matrix` raises `AblationMatrixInvalid`.
    """
    validate_ablation_matrix(matrix)
    issues: list[Issue] = []

    for key in invariants:
        if baseline.get(key) != candidate.get(key):
            issues.append(
                Issue(
                    code=ABLATION_INVARIANT_CHANGED,
                    severity="error",
                    node_id=None,
                    path=f"/{key}",
                    message=(
                        f"comparison invariant {key!r} changed: "
                        f"{baseline.get(key)!r} -> {candidate.get(key)!r} (RFC004 §52.2)."
                    ),
                )
            )

    switches = _switch_by_id(matrix)
    base_values = baseline.get("active_switch_values", {})
    cand_values = candidate.get("active_switch_values", {})
    changed = sorted(
        switch_id
        for switch_id in switches
        if base_values.get(switch_id) != cand_values.get(switch_id)
    )

    if len(changed) == 0:
        issues.append(
            Issue(
                code=ABLATION_NOT_ISOLATED,
                severity="error",
                node_id=None,
                path="/active_switch_values",
                message="no switch changed between baseline and candidate.",
            )
        )
    elif len(changed) > 1:
        issues.append(
            Issue(
                code=ABLATION_NOT_ISOLATED,
                severity="error",
                node_id=None,
                path="/active_switch_values",
                message=f"more than one switch changed: {changed} (RFC004 §52.2).",
            )
        )
    else:
        switch_id = changed[0]
        row = switches[switch_id]
        if row["status"] == "not_available":
            issues.append(
                Issue(
                    code=ABLATION_SWITCH_UNAVAILABLE,
                    severity="error",
                    node_id=None,
                    path=f"/active_switch_values/{switch_id}",
                    message=f"switch {switch_id!r} has no implementation (status=not_available).",
                )
            )

    return issues

"""Entry shapes and the model-side §9.3 axes: the ones whose domain comes
from a code registry rather than from a dataset release.

Split out of `model_target_coverage_report.py` so neither file carries both
the release walk and the registry enumeration. Since S3 (RFC005 §27.5)
`training_materializer.SUPPORTED_LOSS_FAMILIES` is `("L_structure",
"L_text", "L_typed")`, the two typed-property axes are `unmeasured`
rather than a derived zero: the supervision channel is declared, the
counting surface is absent. The rest carry the reason attached to their
state, and five axes have no domain in code to enumerate at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from unflatten.ast.schema_validator import _load_schema
from unflatten.dataset.targets.properties import PROPERTY_FIELDS, pointer_registry
from unflatten.model import training_materializer
from unflatten.model.property_head_registry import (
    EXCLUDED_PROPERTY_POINTERS,
    property_head_registry,
)

#: The loss families whose presence leaves every typed property unsupervised.
#: `L_structure` supervises the token stream and `L_text` the text spans;
#: neither carries a typed property, which is why 441 features on the two
#: property axes can be reported as a derived zero rather than a measured one.
NON_TYPED_LOSS_FAMILIES: frozenset[str] = frozenset({"L_structure", "L_text"})


def typed_property_supervision_gap() -> tuple[str, ...]:
    """Loss families the materializer now supports that this report cannot
    reason about -- read at call time, never frozen at import.

    Empty means the §9.3 premise holds: nothing supervises typed properties,
    so their zero is a derivation. Non-empty means the premise is gone, and
    a zero would then be an unmeasured count wearing a measured one's clothes
    -- which is the exact failure this whole artefact exists to prevent, so
    :func:`typed_property_entry` stops writing counts at all.
    """
    return tuple(
        sorted(set(training_materializer.SUPPORTED_LOSS_FAMILIES) - NON_TYPED_LOSS_FAMILIES)
    )


def typed_property_entry(key: str) -> dict[str, Any]:
    """One typed-property feature: a derived zero, or `unmeasured`.

    The state follows :func:`typed_property_supervision_gap`; it is never a
    literal. The previous version wrote `0/0` with a sentence interpolating
    `SUPPORTED_LOSS_FAMILIES`, so growing that tuple produced 441 zeros
    justified by a sentence that contradicted itself.
    """
    beyond = typed_property_supervision_gap()
    if beyond:
        return unmeasured_entry(
            key,
            "unflatten.model.training_materializer.SUPPORTED_LOSS_FAMILIES now "
            f"includes {list(beyond)} beyond {sorted(NON_TYPED_LOSS_FAMILIES)}: "
            "the L_typed family (RFC005 §27.5) declares a supervision channel "
            "for this feature in which per-row known_mask=false is "
            "no_supervision -- never a numeric zero, an abstain, "
            "not_applicable, a conflict or a negative -- while not_applicable "
            "is derived deterministically from node, profile and registry, and "
            "a target equal to the default value is a supervised prediction, "
            "not a missing one. This report's release walk has no counter for "
            "per-row typed supervision, so its coverage is unmeasured, not "
            "zero; extend the counting surface before this feature may carry "
            "a count",
        )
    return zero_entry(
        key,
        "unflatten.model.training_materializer.SUPPORTED_LOSS_FAMILIES is "
        f"{sorted(NON_TYPED_LOSS_FAMILIES)}: no typed-property loss family is "
        "materialised, so no training target supervises this feature. RFC003 "
        "label presence in the release is not model supervision (RFC005 §9.3 "
        "last sentence)",
        "no_supervision_channel",
    )

def counted_entry(key: str, tally: Mapping[str, tuple[int, int]]) -> dict[str, Any]:
    raw, lineage = tally[key]
    return {
        "feature_key": key,
        "state": "covered",
        "raw_count": raw,
        "lineage_count": lineage,
    }


def zero_entry(key: str, artefact: str, zero_class: str) -> dict[str, Any]:
    """A blocking zero: the missing artefact in prose, its class as data."""
    return {
        "feature_key": key,
        "state": "zero_not_excluded",
        "raw_count": 0,
        "lineage_count": 0,
        "zero_class": zero_class,
        "missing_upstream_artefact": artefact,
    }


def unmeasured_entry(key: str, reason: str) -> dict[str, Any]:
    """A feature nobody counted. Carries no count key, deliberately."""
    return {"feature_key": key, "state": "unmeasured", "unmeasured_reason": reason}


def excluded_entry(key: str, reference: str) -> dict[str, Any]:
    return {
        "feature_key": key,
        "state": "zero_capability_excluded",
        "raw_count": 0,
        "lineage_count": 0,
        "capability_exclusion_ref": reference,
    }


def make_axis(
    axis_id: str,
    clause: str,
    source: str,
    count_unit: str,
    entries: list[dict[str, Any]],
) -> dict:
    """One enumerated axis. `count_unit` is the unit this axis *would* count
    in; it is **downgraded to `not_counted` when no entry carries a count**.

    Derived rather than declared, for the same reason the entry states are:
    with a typed loss family present, every one of the 251 categorical
    entries becomes `unmeasured` and the axis would otherwise still announce
    `typed_property_occurrence` over nothing counted. Declaring a unit for
    counts that do not exist is the axis-level form of a zero that reads as
    measured. `validate_model_target_coverage` enforces the same rule on a
    document this function did not build.
    """
    return {
        "axis_id": axis_id,
        "rfc005_clause": clause,
        "domain_source": source,
        "count_unit": count_unit if any("raw_count" in e for e in entries) else "not_counted",
        "axis_state": "enumerated",
        "missing_upstream_artefact": None,
        "entries": entries,
    }


def undefined_axis(axis_id: str, clause: str, source: str, artefact: str) -> dict:
    return {
        "axis_id": axis_id,
        "rfc005_clause": clause,
        "domain_source": source,
        "count_unit": "not_counted",
        "axis_state": "domain_undefined",
        "missing_upstream_artefact": artefact,
        "entries": [],
    }


def property_pointer_axis() -> dict[str, Any]:
    """Axis 3a: every Property Head Registry pointer."""
    entries = [
        typed_property_entry(f"{node_type}{pointer}")
        for node_type, pointer in sorted(property_head_registry())
    ]
    entries += [
        excluded_entry(f"{node_type}{pointer}", reason)
        for (node_type, pointer), reason in sorted(EXCLUDED_PROPERTY_POINTERS.items())
    ]
    return make_axis(
        "property_head_pointers",
        "kazdy Property Head Registry pointer",
        "unflatten.model.property_head_registry: _HEAD_ASSIGNMENTS and "
        "EXCLUDED_PROPERTY_POINTERS",
        "typed_property_occurrence",
        sorted(entries, key=lambda entry: entry["feature_key"]),
    )


def categorical_value_domain() -> dict[tuple[str, str], tuple[str, ...]]:
    """`(node_type, pointer) -> enum members`, re-derived from the AST schema.

    Deliberately not imported from `diagnostics/typed_property_head/enum_domain.py`,
    which holds the same domain: `tests/model/test_diagnostic_module_not_promoted.py`
    forbids the promotion path from importing `unflatten.diagnostics` at all.
    """
    defs = _load_schema()["$defs"]
    registry = pointer_registry()
    domain: dict[tuple[str, str], tuple[str, ...]] = {}

    def resolve(sub: dict[str, Any]) -> dict[str, Any]:
        while "$ref" in sub:
            sub = defs[sub["$ref"].removeprefix("#/$defs/")]
        return sub

    def walk(sub: dict[str, Any], pointer: str, out: dict[str, set[str]]) -> None:
        sub = resolve(sub)
        if "properties" in sub:
            for name, prop in sub["properties"].items():
                walk(prop, f"{pointer}/{name}", out)
        elif "items" in sub:
            walk(sub["items"], f"{pointer}/*", out)
        elif "oneOf" in sub:
            for branch in sub["oneOf"]:
                if resolve(branch).get("type") != "null":
                    walk(branch, pointer, out)
        elif "enum" in sub:
            out.setdefault(pointer, set()).update(
                str(value) for value in sub["enum"] if value is not None
            )
        elif "const" in sub:
            out.setdefault(pointer, set()).add(str(sub["const"]))

    for node_type, fields in PROPERTY_FIELDS.items():
        found: dict[str, set[str]] = {}
        for field in fields:
            walk(defs[node_type]["properties"][field], f"/{field}", found)
        for pointer, values in found.items():
            if registry[node_type].get(pointer) == "enum":
                domain[(node_type, pointer)] = tuple(sorted(values))
    return domain


def categorical_value_axis() -> dict[str, Any]:
    """Axis 3b: every categorical value of every enum-valued pointer."""
    entries = [
        typed_property_entry(f"{node_type}{pointer}={value}")
        for (node_type, pointer), values in sorted(categorical_value_domain().items())
        for value in values
    ]
    return make_axis(
        "property_categorical_values",
        "kazda categorical value",
        "enum members of every enum-valued pointer in "
        "unflatten.dataset.targets.properties.pointer_registry(), re-derived from "
        "the canonical AST schema",
        "typed_property_occurrence",
        entries,
    )


def undefined_domain_axes() -> list[dict[str, Any]]:
    """The five §9.3 axes whose domain does not exist in code yet."""
    return [
        undefined_axis(
            "numeric_bins",
            "kazdy numeric bin",
            "no module in src/ defines one",
            "no numeric bin policy exists: the materializer emits no numeric "
            "target and no schema or policy artefact defines the bins. "
            "dataset/admission/coverage.py records the same gap as its "
            "_UNMEASURED criterion 5",
        ),
        undefined_axis(
            "repeated_field_variants",
            "repeated-field count/item variant",
            "no module in src/ defines the variants",
            "no count/item variant target exists: five registry pointers are "
            "repeated (section/columns/widths_mpt/*, the three paragraph "
            "layout/tabs/* pointers, table/columns/*/width_mpt), but no head, "
            "count target or stop token distinguishes a count variant from an "
            "item variant",
        ),
        undefined_axis(
            "table_plan_variants",
            "TablePlan variant",
            "no module in src/ defines the variants",
            "TablePlan variants are RFC004 §53.1 TableBoundaryClaim.kind, which "
            "has no code representation: trainable_group_registry owns a "
            "`table_plan/boundary_claims` tensor identity, but no variant enum",
        ),
        undefined_axis(
            "resource_recovery_modes",
            "resource recovery mode",
            "no module in src/ defines the modes",
            "resource recovery modes (RFC004 §28.4/§28.5) have no code "
            "representation; component_lock.schema.json defers the "
            "`resource_recovery/materializer` component kind in its own $comment",
        ),
        undefined_axis(
            "supervised_prediction_status",
            "kazdy supervised prediction status (the status half)",
            "no module in src/ defines the statuses",
            "RFC005 §27.5 supervised prediction status has no enum in code; "
            "prediction_ir.RawDocumentHypothesis.typed_predictions is an untyped "
            "tuple that its own docstring records as always empty until M4",
        ),
    ]



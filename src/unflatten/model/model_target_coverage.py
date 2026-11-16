"""The `ModelTargetCoverageReport` producer (RFC005 §9.3): raw and
independent-lineage counts over the model's own target surface, bound to
the dataset release, the RFC004 Component Lock, the TrainingTargetMaterializer
and the loss policy.

§9.3's last sentence is the reason this artefact is not
`dataset/admission/coverage.py` with more rows: "RFC003 AST coverage sama
osebe nezaruc uje coverage modelovych tokenov/headov." The release coverage
report counts node types and property *presence* in the AST; this one counts
what the model is actually supervised on -- structural symbols, tokenizer
routes, Property Head Registry pointers, categorical values, and the label
authority behind them.

**Most of this report is zero today, and that is its result, not its
failure.** Since S3 (RFC005 §27.5) `training_materializer.SUPPORTED_LOSS_FAMILIES`
is `("L_structure", "L_text", "L_typed")`: the typed-property family is
declared, but the release walk has no counter for it -- per-row
`known_mask=false` is `no_supervision`, never a numeric zero -- so every
property head pointer and every categorical value is `unmeasured`, not
supervised zero times. §9.3 makes a zero consequential ("blokuje product promotion alebo
musi byt capability-excluded upstream"), which only works if a reader can
tell the kinds of zero apart. Hence four states, kept apart by
:func:`_check_entry`:

* **covered** -- a count, necessarily above zero.
* **zero_capability_excluded** -- legitimate, and it must carry the
  reference to what excludes it (a capability profile, or the reason string
  a registry already records).
* **zero_not_excluded** -- blocking, and it must name the **missing upstream
  artefact**, because "0" alone does not say what would have to exist for
  the number to be able to move.
* **unmeasured** -- and this state carries **no count key at all**. A zero
  written for something nobody measured is indistinguishable from a
  measured zero once it is in a JSON file, and the schema's optional count
  fields exist precisely so that absence remains expressible.

The same rule one level up: an axis whose domain does not exist in code is
`axis_state: domain_undefined` with an empty entry list AND a named missing
artefact -- an empty list on an `enumerated` axis would read as "there was
nothing to cover", which is a different (and false) claim.

Ten axis ids for §9.3's eight bullets: the third bullet (registry pointer
*and* categorical value) and the eighth (prediction status *with* label
authority) each name two things whose coverage state differs today, and one
axis record cannot carry two axis states. `rfc005_clause` maps each axis
back to its bullet.

Counts are integers. `lineage_count` counts distinct `lineage_root_id`s, not
RFC003 §42 `independent_lineage_families`: no family field exists on a
document bundle, so a family count cannot be produced and must not be
claimed (the schema's `$comment` on the field says so too).

This module builds, validates and hashes a report; it does not decide
whether the coverage is sufficient. §9.3's promotion consequence is a gate
decision for S8/S9, not a function here. It ships no `nn.Module` and
authorizes no gradient update.

Hashed under `rfc005.model-target-coverage/v1`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.model.hashing import domain_hash

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_SCHEMA_NAME = "model_target_coverage"

#: Design §4.9 registry row `rfc005.model-target-coverage/v1`. Owned here:
#: no other module computes a `model_target_coverage_hash`.
_MODEL_TARGET_COVERAGE_DOMAIN = "rfc005.model-target-coverage/v1"

MODEL_TARGET_COVERAGE_SCHEMA_VERSION = "1.0.0"

#: The ten axis ids, in report order. Closed, and every one must be present
#: exactly once: a §9.3 axis that is simply absent from the document is the
#: silent hole this report exists to prevent.
AXIS_IDS: tuple[str, ...] = (
    "structural_symbols",
    "tokenizer_routes",
    "property_head_pointers",
    "property_categorical_values",
    "numeric_bins",
    "repeated_field_variants",
    "table_plan_variants",
    "resource_recovery_modes",
    "supervised_prediction_status",
    "label_authority",
)

#: Which `zero_class` values each axis may use. A blocking zero on a
#: structural symbol is either something the release lacks or something the
#: target unit cannot emit; a typed-property zero can only be the missing
#: supervision channel; a label authority the release does not contain is
#: only ever absent. An unconstrained required field is a new surface rather
#: than a fix -- it can no longer say *why* a zero is zero and be wrong about
#: *what* it describes. Axes with no possible entry (`domain_undefined`) map
#: to the empty set; widening any row is a deliberate edit.
AXIS_ZERO_CLASSES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "structural_symbols": frozenset(
            {"absent_from_release", "target_unit_incompatible"}
        ),
        "tokenizer_routes": frozenset(
            {"absent_from_release", "target_unit_incompatible"}
        ),
        "property_head_pointers": frozenset({"no_supervision_channel"}),
        "property_categorical_values": frozenset({"no_supervision_channel"}),
        "numeric_bins": frozenset(),
        "repeated_field_variants": frozenset(),
        "table_plan_variants": frozenset(),
        "resource_recovery_modes": frozenset(),
        "supervised_prediction_status": frozenset(),
        "label_authority": frozenset({"absent_from_release"}),
    }
)

#: The unit an axis declares when nothing on it was counted.
NOT_COUNTED = "not_counted"

#: Per state: the reason field it must carry, and whether it carries counts.
_REASON_FIELD: Mapping[str, str] = {
    "zero_capability_excluded": "capability_exclusion_ref",
    "zero_not_excluded": "missing_upstream_artefact",
    "unmeasured": "unmeasured_reason",
}
_ALL_REASON_FIELDS: frozenset[str] = frozenset(_REASON_FIELD.values())
_COUNT_FIELDS: tuple[str, str] = ("raw_count", "lineage_count")


class ModelTargetCoverageInvalid(ValueError):
    """A model target coverage report fails its closed schema, omits or
    repeats a §9.3 axis, or carries an entry whose state, counts and reason
    do not agree.

    M1-internal (design §9.3); not an RFC004 registry code.
    """


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _where(axis_id: str, entry: Mapping[str, Any]) -> str:
    return f"axis {axis_id!r}, entry {entry.get('feature_key')!r}"


def _check_entry(axis_id: str, entry: Mapping[str, Any], scope: Mapping[str, Any]) -> None:
    """The four states, kept apart (module docstring)."""
    state = entry["state"]
    present_counts = [field for field in _COUNT_FIELDS if field in entry]

    if state == "unmeasured":
        if present_counts:
            raise ModelTargetCoverageInvalid(
                f"{_where(axis_id, entry)} is `unmeasured` but carries "
                f"{', '.join(present_counts)} -- an unmeasured feature must carry "
                "no count at all, because a written 0 is indistinguishable from a "
                "measured zero and RFC005 §9.3 attaches a consequence to zeros"
            )
    else:
        missing = [field for field in _COUNT_FIELDS if field not in entry]
        if missing:
            raise ModelTargetCoverageInvalid(
                f"{_where(axis_id, entry)} is {state!r} but omits "
                f"{', '.join(missing)} -- only `unmeasured` may omit counts"
            )
        raw, lineage = entry["raw_count"], entry["lineage_count"]
        if state == "covered" and raw == 0:
            raise ModelTargetCoverageInvalid(
                f"{_where(axis_id, entry)} claims state `covered` with "
                "raw_count 0 -- a zero is one of the three zero states, never "
                "coverage"
            )
        if state == "covered" and lineage == 0:
            raise ModelTargetCoverageInvalid(
                f"{_where(axis_id, entry)} claims state `covered` with "
                f"raw_count {raw} but lineage_count 0 -- §9.3 asks for both "
                "counts because occurrences from few independent sources are not "
                "coverage, and zero sources is the limiting case of that"
            )
        if state != "covered" and (raw, lineage) != (0, 0):
            raise ModelTargetCoverageInvalid(
                f"{_where(axis_id, entry)} claims state {state!r} with "
                f"raw_count {raw} / lineage_count {lineage} -- a zero state must "
                "carry an explicit 0/0"
            )
        if lineage > raw:
            raise ModelTargetCoverageInvalid(
                f"{_where(axis_id, entry)} has lineage_count {lineage} above "
                f"raw_count {raw} -- a lineage root cannot contribute more "
                "independent sources than occurrences"
            )
        if lineage > scope["lineage_roots_counted"]:
            raise ModelTargetCoverageInvalid(
                f"{_where(axis_id, entry)} has lineage_count {lineage} above the "
                f"scope's lineage_roots_counted {scope['lineage_roots_counted']} "
                "-- the count would name lineage roots the report did not read"
            )

    if state == "zero_not_excluded" and "zero_class" not in entry:
        raise ModelTargetCoverageInvalid(
            f"{_where(axis_id, entry)} is `zero_not_excluded` but names no "
            "zero_class -- the prose reason does not survive a gate that reads "
            "states, and 'the release lacks it' and 'the target unit cannot emit "
            "it' are not closable by the same action"
        )
    allowed = AXIS_ZERO_CLASSES[axis_id]
    if state == "zero_not_excluded" and entry.get("zero_class") not in allowed:
        raise ModelTargetCoverageInvalid(
            f"{_where(axis_id, entry)} carries zero_class "
            f"{entry.get('zero_class')!r}, which axis {axis_id!r} cannot produce "
            f"(allowed: {sorted(allowed) or 'none -- this axis has no entries'}) "
            "-- a class that does not belong to the axis misdescribes what would "
            "close the zero"
        )
    if state != "zero_not_excluded" and "zero_class" in entry:
        raise ModelTargetCoverageInvalid(
            f"{_where(axis_id, entry)} is {state!r} but carries a zero_class, "
            "which classifies a blocking zero and belongs to no other state"
        )
    required = _REASON_FIELD.get(state)
    if required is not None and not str(entry.get(required) or "").strip():
        raise ModelTargetCoverageInvalid(
            f"{_where(axis_id, entry)} is {state!r} but names no {required} -- "
            "RFC005 §9.3 lets a zero stand only when what excludes it, or what is "
            "missing upstream, is named, and whitespace names nothing"
        )
    surplus = sorted((_ALL_REASON_FIELDS - {required}) & set(entry))
    if surplus:
        raise ModelTargetCoverageInvalid(
            f"{_where(axis_id, entry)} is {state!r} but carries "
            f"{', '.join(surplus)}, which belongs to a different state -- two "
            "reasons on one entry make it unreadable which one applies"
        )


def _check_axis(axis: Mapping[str, Any], scope: Mapping[str, Any]) -> None:
    axis_id = axis["axis_id"]
    entries = axis["entries"]
    artefact = axis["missing_upstream_artefact"]

    if axis["axis_state"] == "enumerated":
        if not entries:
            raise ModelTargetCoverageInvalid(
                f"axis {axis_id!r} is `enumerated` but lists no entries -- an "
                "empty enumerated axis reads as 'nothing to cover'; an axis whose "
                "domain does not exist is `domain_undefined`"
            )
        if artefact is not None:
            raise ModelTargetCoverageInvalid(
                f"axis {axis_id!r} is `enumerated` but names the missing upstream "
                f"artefact {artefact!r} -- an enumerated domain is not missing; "
                "per-entry `missing_upstream_artefact` is where a zero is explained"
            )
    else:
        if entries:
            raise ModelTargetCoverageInvalid(
                f"axis {axis_id!r} is `domain_undefined` but lists "
                f"{len(entries)} entries -- entries would imply a domain to "
                "enumerate them from"
            )
        if not artefact:
            raise ModelTargetCoverageInvalid(
                f"axis {axis_id!r} is `domain_undefined` but names no missing "
                "upstream artefact -- an undefined domain must say what would "
                "have to exist for it to be defined"
            )

    counted = [entry for entry in entries if "raw_count" in entry]
    if counted and axis["count_unit"] == NOT_COUNTED:
        raise ModelTargetCoverageInvalid(
            f"axis {axis_id!r} declares count_unit {NOT_COUNTED!r} while "
            f"{len(counted)} of its entries carry a count -- the unit is what one "
            "unit of `raw_count` IS, so an axis that counted something must name it"
        )
    if not counted and axis["count_unit"] != NOT_COUNTED:
        raise ModelTargetCoverageInvalid(
            f"axis {axis_id!r} declares count_unit {axis['count_unit']!r} while "
            "none of its entries carries a count -- announcing a unit over "
            "nothing counted is the axis-level form of a zero that reads as "
            f"measured; declare {NOT_COUNTED!r}"
        )

    seen: set[str] = set()
    for entry in entries:
        key = entry["feature_key"]
        if key in seen:
            raise ModelTargetCoverageInvalid(
                f"axis {axis_id!r} lists feature {key!r} twice -- two rows for one "
                "feature make its coverage ambiguous"
            )
        seen.add(key)
        _check_entry(axis_id, entry, scope)


def _check_axes_are_the_closed_set(report: Mapping[str, Any]) -> None:
    seen: list[str] = [axis["axis_id"] for axis in report["axes"]]
    duplicated = sorted({a for a in seen if seen.count(a) > 1})
    if duplicated:
        raise ModelTargetCoverageInvalid(
            f"axes {', '.join(duplicated)} appear more than once -- RFC005 §9.3 "
            "reports each axis once, and two records for one axis can disagree"
        )
    missing = [axis_id for axis_id in AXIS_IDS if axis_id not in seen]
    if missing:
        raise ModelTargetCoverageInvalid(
            f"the report omits axes {', '.join(missing)} -- every §9.3 axis must "
            "be present, including the ones that are entirely zero; an omitted "
            "axis is the unspoken hole this report exists to close"
        )


def _check_scope(report: Mapping[str, Any]) -> None:
    scope = report["scope"]
    if scope["lineage_roots_counted"] > scope["bundles_counted"]:
        raise ModelTargetCoverageInvalid(
            f"scope counts {scope['lineage_roots_counted']} lineage roots over "
            f"{scope['bundles_counted']} bundles -- each bundle has exactly one "
            "lineage root, so there cannot be more roots than bundles"
        )


def validate_model_target_coverage(report: Mapping[str, Any]) -> None:
    """Validate `report` against `model_target_coverage.schema.json`, plus the
    §9.3 rules the schema cannot express (module docstring)."""
    errors = list(_validator().iter_errors(report))
    if errors:
        raise ModelTargetCoverageInvalid(
            f"schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )
    _check_axes_are_the_closed_set(report)
    _check_scope(report)
    for axis in report["axes"]:
        _check_axis(axis, report["scope"])


def build_model_target_coverage(
    *,
    report_version: str,
    stage_id: str,
    bindings: Mapping[str, Any],
    scope: Mapping[str, Any],
    axes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble one coverage report, validated on the way out.

    Every argument is keyword-only and required; there are no defaults and
    no field is filled in for the caller. A missing input is an error --
    never a silently zeroed count, which is the one thing this artefact must
    never produce.
    """
    report = {
        "model_target_coverage_schema_version": MODEL_TARGET_COVERAGE_SCHEMA_VERSION,
        "report_version": report_version,
        "stage_id": stage_id,
        "bindings": dict(bindings),
        "scope": dict(scope),
        "axes": [
            {**dict(axis), "entries": [dict(entry) for entry in axis["entries"]]}
            for axis in axes
        ],
    }
    validate_model_target_coverage(report)
    return report


def load_model_target_coverage(path: Path | str) -> dict[str, Any]:
    """Read a frozen coverage report from disk, validated on the way in."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise ModelTargetCoverageInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ModelTargetCoverageInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_model_target_coverage(payload)
    return payload


def model_target_coverage_hash(report: Mapping[str, Any]) -> str:
    """`model_target_coverage_hash` over the whole validated report."""
    validate_model_target_coverage(report)
    return domain_hash(_MODEL_TARGET_COVERAGE_DOMAIN, dict(report))

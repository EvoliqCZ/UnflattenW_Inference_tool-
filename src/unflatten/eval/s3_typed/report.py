"""T9 -- a teacher-forced typed-property report (design §8, plan row 11).

Pure functions over already-decoded `(predictions, targets)` per
`(head_id, pointer)`: no torch import, no model, no cache reader. `cli.py`
is the only module in this package that runs a forward pass; everything
here is exercised with plain Python fixtures.

Report-only for S3 discovery: no StageGatePolicy is frozen for run 1
(design §8), so nothing here PASSES or FAILS a run -- it only measures.

**Enum decoding source (coordinator note, plan §7).**
`typed_property_pointer_map.enum_domain` orders 52 of 57 enum pointers
differently from the diagnostics module of the same name (the one that
sized the OLDER, pre-promotion heads). Every enum `PointerReport` carries
its `domain` -- the schema-order tuple from the PRODUCTION module, the
same source `s3_typed_loss._encode_target` indexed against and
`candidate_model._build_typed_head_group` sized the head from -- so a
swap to the diagnostics module changes what this report decodes, and the
literal-pinned test catches it.

**Integer range bounds.** `integer_bounds` walks the canonical AST JSON
Schema (`unflatten.ast.schema_validator._load_schema()["$defs"]`) the same
way `unflatten.dataset.targets.properties._walk_schema` derives the value
type per pointer, but reads `minimum`/`maximum` off the leaf schema
instead of a type name. Not every integer pointer has bounds in the
schema (`_walk_schema` merely knows the pointer is `"integer"`); where
neither key is present, `integer_bounds` returns `(None, None)` and no
range check fires for that pointer.

Every `(node_type, pointer)` lookup here normalizes the pointer first
(`normalize_pointer`) -- the same rule T6's census producer and T3's
collator already apply, matching the cache's own indexed-pointer storage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from unflatten.ast.schema_validator import _load_schema
from unflatten.dataset.targets.properties import PROPERTY_FIELDS, normalize_pointer
from unflatten.eval.s2_gate.decode import (
    EvalCacheIdentityRefused,
    refuse_unless_eval_cache_identity_ok,
)
from unflatten.model.hashing import domain_hash
from unflatten.model.typed_property_pointer_map import enum_domain

#: This module is the sole owner of this domain. `report_hash` no longer
#: uses `domain_hash` (see `_payload_hash`) since the report now carries
#: float metrics; `REPORT_DOMAIN` is retired with it.
CACHE_IDENTITY_DOMAIN = "rfc005.s3-typed-eval-cache-identity/v1"

_FAMILIES: frozenset[str] = frozenset({"enum", "boolean", "integer"})


class S3TypedReportRefused(ValueError):
    """A report input is not well-formed: an unknown family, or a cache
    identity mismatch on a field other than `split`. A refusal, never a
    best-effort repair."""


# ---------------------------------------------------------------------------
# Schema-derived integer bounds
# ---------------------------------------------------------------------------


def _resolve_ref(subschema: Any, defs: Mapping[str, Any]) -> Any:
    while isinstance(subschema, dict) and "$ref" in subschema:
        ref = subschema["$ref"]
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"unsupported $ref target: {ref!r}")
        subschema = defs[ref.removeprefix("#/$defs/")]
    return subschema


def _walk_bounds(
    subschema: Any,
    pointer: str,
    out: dict[str, tuple[int | None, int | None]],
    defs: Mapping[str, Any],
) -> None:
    subschema = _resolve_ref(subschema, defs)
    if not isinstance(subschema, dict):
        return
    if isinstance(subschema.get("properties"), dict):
        for name, prop in subschema["properties"].items():
            _walk_bounds(prop, f"{pointer}/{name}", out, defs)
        return
    if "items" in subschema:
        _walk_bounds(subschema["items"], f"{pointer}/*", out, defs)
        return
    if isinstance(subschema.get("oneOf"), list):
        for branch in subschema["oneOf"]:
            resolved = _resolve_ref(branch, defs)
            if isinstance(resolved, dict) and resolved.get("type") == "null":
                continue
            _walk_bounds(branch, pointer, out, defs)
        return
    if "minimum" in subschema or "maximum" in subschema:
        out[normalize_pointer(pointer)] = (subschema.get("minimum"), subschema.get("maximum"))


@lru_cache(maxsize=1)
def _integer_bounds_registry() -> dict[str, dict[str, tuple[int | None, int | None]]]:
    schema = _load_schema()
    defs = schema["$defs"]
    registry: dict[str, dict[str, tuple[int | None, int | None]]] = {}
    for node_type, fields in PROPERTY_FIELDS.items():
        collected: dict[str, tuple[int | None, int | None]] = {}
        node_def = defs[node_type]
        for field_name in fields:
            if field_name not in node_def["properties"]:
                continue
            _walk_bounds(node_def["properties"][field_name], f"/{field_name}", collected, defs)
        registry[node_type] = collected
    return registry


def integer_bounds(node_type: str, pointer: str) -> tuple[int | None, int | None] | None:
    """`(minimum, maximum)` for one integer pointer, read from the
    canonical AST JSON Schema's own `minimum`/`maximum` keywords
    (`src/unflatten/ast/schemas/canonical_ast.schema.json`) -- `None` for
    either side the schema does not bound, and `None` for the whole tuple
    when the pointer carries no bound at all."""
    return _integer_bounds_registry().get(node_type, {}).get(normalize_pointer(pointer))


def _derive_integer_unit(pointer: str) -> str | None:
    """`integer_unit` when the caller passes no explicit `unit`: the
    pointer's final path segment's suffix names its unit -- `_mpt` ->
    `"mpt"` (RFC004 milli-point), `_ppm` -> `"ppm"` (parts-per-million);
    any other suffix (or none) -> `None`, never guessed."""
    segment = pointer.rsplit("/", 1)[-1]
    if segment.endswith("_mpt"):
        return "mpt"
    if segment.endswith("_ppm"):
        return "ppm"
    return None


# ---------------------------------------------------------------------------
# Per-pointer report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PointerReport:
    """One `(head_id, pointer)` group's teacher-forced metrics (design §8).

    `unmeasured=True` -- never a fabricated `0` -- when `supervised_count`
    on this val cohort is 0; every metric field then stays `None`.

    `lineage_count`/`lineage_unmeasured`: when the caller has no
    `--release-dir` (no lineage map at all, `cli._lineage_by_bundle`),
    `lineage_ids` is `None`, `lineage_count` stays `None`, and
    `lineage_unmeasured=True` -- the count is never fabricated as `0`,
    which would read as "measured, zero distinct lineages" rather than
    "not measured"."""

    head_id: str
    node_type: str
    pointer: str
    family: str
    supervised_count: int
    lineage_count: int | None
    lineage_unmeasured: bool
    unmeasured: bool
    enum_accuracy: float | None = None
    enum_domain: tuple[str, ...] | None = None
    boolean_accuracy: float | None = None
    integer_exact_match_rate: float | None = None
    integer_mae: float | None = None
    integer_unit: str | None = None
    out_of_domain: int = 0
    range_violations: int = 0

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "head_id": self.head_id,
            "node_type": self.node_type,
            "pointer": self.pointer,
            "family": self.family,
            "supervised_count": self.supervised_count,
            "lineage_count": self.lineage_count,
            "lineage_unmeasured": self.lineage_unmeasured,
            "unmeasured": self.unmeasured,
            "enum_accuracy": self.enum_accuracy,
            "enum_domain": list(self.enum_domain) if self.enum_domain is not None else None,
            "boolean_accuracy": self.boolean_accuracy,
            "integer_exact_match_rate": self.integer_exact_match_rate,
            "integer_mae": self.integer_mae,
            "integer_unit": self.integer_unit,
            "out_of_domain": self.out_of_domain,
            "range_violations": self.range_violations,
        }
        return payload


def pointer_report(
    *,
    head_id: str,
    node_type: str,
    pointer: str,
    family: str,
    predictions: Sequence[Any],
    targets: Sequence[Any],
    lineage_ids: Sequence[str] | None,
    unit: str | None = None,
) -> PointerReport:
    """One `(head_id, pointer)` group's metrics over already-decoded
    per-example `predictions`/`targets` (index space for enum, `bool` for
    boolean, `int` for integer -- exactly what `typed_property_heads
    .predict_value`/`.forward` argmax produce and `s3_typed_loss
    .collate_typed` encoded the targets as).

    `lineage_ids=None` means the caller has no lineage map at all (no
    `--release-dir`) -- `lineage_count` stays `None` and
    `lineage_unmeasured=True`, never a fabricated `0`.

    `unit`: an explicit override (e.g. a schema-declared unit string). When
    `None` and `family == "integer"`, `integer_unit` is derived from the
    pointer's suffix (`_derive_integer_unit`)."""
    if family not in _FAMILIES:
        raise S3TypedReportRefused(
            f"unknown family {family!r} -- expected one of {sorted(_FAMILIES)}"
        )
    if len(predictions) != len(targets):
        raise S3TypedReportRefused(
            f"{len(predictions)} predictions != {len(targets)} targets for "
            f"{(head_id, pointer)!r}"
        )
    normalized = normalize_pointer(pointer)
    supervised_count = len(targets)
    lineage_unmeasured = lineage_ids is None
    lineage_count = None if lineage_ids is None else len(set(lineage_ids))

    if supervised_count == 0:
        return PointerReport(
            head_id=head_id,
            node_type=node_type,
            pointer=normalized,
            family=family,
            supervised_count=0,
            lineage_count=None if lineage_unmeasured else 0,
            lineage_unmeasured=lineage_unmeasured,
            unmeasured=True,
        )

    if family == "enum":
        domain = enum_domain(node_type, normalized)
        correct = sum(1 for p, t in zip(predictions, targets, strict=True) if p == t)
        return PointerReport(
            head_id=head_id,
            node_type=node_type,
            pointer=normalized,
            family=family,
            supervised_count=supervised_count,
            lineage_count=lineage_count,
            lineage_unmeasured=lineage_unmeasured,
            unmeasured=False,
            enum_accuracy=correct / supervised_count,
            enum_domain=domain,
            out_of_domain=0,
        )

    if family == "boolean":
        correct = sum(
            1 for p, t in zip(predictions, targets, strict=True) if bool(p) == bool(t)
        )
        return PointerReport(
            head_id=head_id,
            node_type=node_type,
            pointer=normalized,
            family=family,
            supervised_count=supervised_count,
            lineage_count=lineage_count,
            lineage_unmeasured=lineage_unmeasured,
            unmeasured=False,
            boolean_accuracy=correct / supervised_count,
            out_of_domain=0,
        )

    # integer
    exact = sum(1 for p, t in zip(predictions, targets, strict=True) if int(p) == int(t))
    abs_errors = sum(
        abs(int(p) - int(t)) for p, t in zip(predictions, targets, strict=True)
    )
    mae = abs_errors / supervised_count
    bounds = integer_bounds(node_type, normalized)
    range_violations = 0
    if bounds is not None:
        lower, upper = bounds
        for p in predictions:
            value = int(p)
            if (lower is not None and value < lower) or (upper is not None and value > upper):
                range_violations += 1
    return PointerReport(
        head_id=head_id,
        node_type=node_type,
        pointer=normalized,
        family=family,
        supervised_count=supervised_count,
        lineage_count=lineage_count,
        lineage_unmeasured=lineage_unmeasured,
        unmeasured=False,
        integer_exact_match_rate=exact / supervised_count,
        integer_mae=mae,
        integer_unit=unit if unit is not None else _derive_integer_unit(normalized),
        out_of_domain=0,
        range_violations=range_violations,
    )


# ---------------------------------------------------------------------------
# Per-head_id macro summary
# ---------------------------------------------------------------------------

_MACRO_METRICS: tuple[str, ...] = (
    "enum_accuracy",
    "boolean_accuracy",
    "integer_exact_match_rate",
    "integer_mae",
)


def head_summary(pointer_reports: Sequence[PointerReport]) -> dict[str, Any]:
    """Macro summary for one `head_id`: one vote per pointer (mirrors
    `s3_typed_loss.s3_total_loss`'s own macro `L_typed` mean), never a
    supervised-count-weighted average that would let one high-volume
    pointer swamp the rest. `unmeasured=True` when every pointer in the
    group had zero val-cohort supervision -- never averaged into `0`."""
    measured = [r for r in pointer_reports if not r.unmeasured]
    lineage_counts = [r.lineage_count for r in pointer_reports if r.lineage_count is not None]
    summary: dict[str, Any] = {
        "pointer_count": len(pointer_reports),
        "measured_pointer_count": len(measured),
        "supervised_count": sum(r.supervised_count for r in pointer_reports),
        "lineage_count": max(lineage_counts) if lineage_counts else None,
        "lineage_unmeasured": len(lineage_counts) == 0,
        "unmeasured": len(measured) == 0,
        "out_of_domain": sum(r.out_of_domain for r in pointer_reports),
        "range_violations": sum(r.range_violations for r in pointer_reports),
    }
    for metric in _MACRO_METRICS:
        values = [getattr(r, metric) for r in measured if getattr(r, metric) is not None]
        summary[metric] = sum(values) / len(values) if values else None
    return summary


# ---------------------------------------------------------------------------
# Cache identity + top-level report
# ---------------------------------------------------------------------------


def refuse_unless_cache_identity_ok(
    *, eval_cache_identity: Mapping[str, Any], checkpoint_cache_identity: Mapping[str, Any]
) -> None:
    """Reuses `eval.s2_gate.decode.refuse_unless_eval_cache_identity_ok`
    verbatim (design §8): `checkpoint_cache_identity` (the `cache_identity`
    a checkpoint's own state recorded at TRAINING time, `split="train"`) is
    fed as `run_manifest_bindings`, and the VAL `eval_cache_identity` this
    CLI runs against is fed as `eval_cache_identity` -- agreement required
    on every field except `split`, which is expected to differ."""
    try:
        refuse_unless_eval_cache_identity_ok(
            eval_cache_identity=eval_cache_identity,
            run_manifest_bindings=checkpoint_cache_identity,
        )
    except EvalCacheIdentityRefused as exc:
        raise S3TypedReportRefused(str(exc)) from exc


# ---------------------------------------------------------------------------
# Both-arms combination -- trained vs. baseline, with a signed delta
# ---------------------------------------------------------------------------

#: Delta sign convention (governance R1: "the untrained-heads baseline arm
#: is the comparison", design §8 D8): **positive always means the trained
#: arm is better.** For `enum_accuracy` / `boolean_accuracy` /
#: `integer_exact_match_rate` -- higher is better -- delta = trained -
#: baseline. For `integer_mae` -- lower is better -- delta = baseline -
#: trained (the sign is flipped so the "positive = better" rule still
#: holds). `unmeasured` in EITHER arm -> the delta is `None`, never a
#: number computed against a fabricated baseline.
_REVERSED_DELTA_METRICS: frozenset[str] = frozenset({"integer_mae"})


def _delta(trained: float | None, baseline: float | None, *, metric: str) -> float | None:
    if trained is None or baseline is None:
        return None
    if metric in _REVERSED_DELTA_METRICS:
        return baseline - trained
    return trained - baseline


def _pointer_delta(trained: PointerReport, baseline: PointerReport) -> dict[str, float | None]:
    if trained.unmeasured or baseline.unmeasured:
        return dict.fromkeys(_MACRO_METRICS)
    return {
        metric: _delta(getattr(trained, metric), getattr(baseline, metric), metric=metric)
        for metric in _MACRO_METRICS
    }


def _head_delta(
    trained_summary: Mapping[str, Any], baseline_summary: Mapping[str, Any]
) -> dict[str, float | None]:
    if trained_summary["unmeasured"] or baseline_summary["unmeasured"]:
        return dict.fromkeys(_MACRO_METRICS)
    return {
        metric: _delta(trained_summary[metric], baseline_summary[metric], metric=metric)
        for metric in _MACRO_METRICS
    }


def _group_by_head(
    pointer_reports: Sequence[PointerReport],
) -> dict[str, list[PointerReport]]:
    by_head: dict[str, list[PointerReport]] = {}
    for r in pointer_reports:
        by_head.setdefault(r.head_id, []).append(r)
    return by_head


def combine_pointer_arms(
    trained_pointer_reports: Sequence[PointerReport],
    baseline_pointer_reports: Sequence[PointerReport],
) -> list[dict[str, Any]]:
    """Pairs both arms' `PointerReport`s by `(head_id, pointer)` -- the two
    arms are evaluated over the SAME dataset and supervision masks
    (`cli.evaluate_arm`), so their key sets must agree; a mismatch is
    refused rather than silently dropped or zero-filled."""
    trained_by_key = {(r.head_id, r.pointer): r for r in trained_pointer_reports}
    baseline_by_key = {(r.head_id, r.pointer): r for r in baseline_pointer_reports}
    if trained_by_key.keys() != baseline_by_key.keys():
        raise S3TypedReportRefused(
            "trained and baseline arms cover different (head_id, pointer) sets: "
            f"{sorted(trained_by_key.keys() ^ baseline_by_key.keys())}"
        )
    combined = []
    for key in sorted(trained_by_key):
        trained = trained_by_key[key]
        baseline = baseline_by_key[key]
        combined.append(
            {
                "head_id": trained.head_id,
                "node_type": trained.node_type,
                "pointer": trained.pointer,
                "family": trained.family,
                "trained": trained.as_dict(),
                "baseline": baseline.as_dict(),
                "delta": _pointer_delta(trained, baseline),
            }
        )
    return combined


def combine_head_summaries(
    trained_pointer_reports: Sequence[PointerReport],
    baseline_pointer_reports: Sequence[PointerReport],
) -> dict[str, dict[str, Any]]:
    """Per-`head_id` macro summary for both arms plus a delta -- same
    pairing/refusal rule as :func:`combine_pointer_arms`."""
    trained_by_head = _group_by_head(trained_pointer_reports)
    baseline_by_head = _group_by_head(baseline_pointer_reports)
    if trained_by_head.keys() != baseline_by_head.keys():
        raise S3TypedReportRefused(
            "trained and baseline arms cover different head_ids: "
            f"{sorted(trained_by_head.keys() ^ baseline_by_head.keys())}"
        )
    heads: dict[str, dict[str, Any]] = {}
    for head_id in sorted(trained_by_head):
        trained_summary = head_summary(trained_by_head[head_id])
        baseline_summary = head_summary(baseline_by_head[head_id])
        heads[head_id] = {
            "trained": trained_summary,
            "baseline": baseline_summary,
            "delta": _head_delta(trained_summary, baseline_summary),
        }
    return heads


def _payload_hash(payload: Mapping[str, Any]) -> str:
    """`report_hash` over the WHOLE payload (task: "put the report hash
    over the whole payload"), including the per-pointer/per-head float
    metrics -- unlike `domain_hash`, which cannot: its CBOR profile
    (`rfc8949-core-det-nfc-int-v0`) has no float in its data model
    (`cbor.py`), and every existing domain-hashed payload in this codebase
    is integer/string/bool-only for that reason. This hash is therefore
    SHA-256 over `json.dumps(payload, sort_keys=True)` instead -- a plain
    content hash, not a cross-language canonical-JSON contract like
    RFC001's `ast_hash`; it exists only so a consumer can tell this
    report's bytes were not silently substituted."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def build_report(
    *,
    trained_pointer_reports: Sequence[PointerReport],
    baseline_pointer_reports: Sequence[PointerReport],
    eval_cache_identity: Mapping[str, Any],
    s3_checkpoint_cache_identity: Mapping[str, Any],
    baseline_checkpoint_cache_identity: Mapping[str, Any],
    s3_checkpoint_id: str,
    baseline_checkpoint_id: str,
) -> dict[str, Any]:
    """The `s3_typed_report.json` payload (design §8): `regime:
    "teacher_forced"`, `free_running: null` (RFC005 R4-8), both arms'
    checkpoint ids, the val cache's identity hash, both arms' metrics per
    `(head_id, pointer)` and per `head_id` plus a signed delta (see
    `_REVERSED_DELTA_METRICS` for the sign convention), and a self-hash.

    Refuses (via :func:`refuse_unless_cache_identity_ok`) unless BOTH
    arms' checkpoints were trained on a cache that agrees with the eval
    cache on every field except `split`; also refuses (via
    :func:`combine_pointer_arms`/:func:`combine_head_summaries`) if the two
    arms do not cover the same `(head_id, pointer)`/`head_id` keys.
    """
    refuse_unless_cache_identity_ok(
        eval_cache_identity=eval_cache_identity,
        checkpoint_cache_identity=s3_checkpoint_cache_identity,
    )
    refuse_unless_cache_identity_ok(
        eval_cache_identity=eval_cache_identity,
        checkpoint_cache_identity=baseline_checkpoint_cache_identity,
    )

    pointers = combine_pointer_arms(trained_pointer_reports, baseline_pointer_reports)
    heads = combine_head_summaries(trained_pointer_reports, baseline_pointer_reports)

    cache_identity_hash = domain_hash(CACHE_IDENTITY_DOMAIN, dict(eval_cache_identity))

    payload: dict[str, Any] = {
        "regime": "teacher_forced",
        "free_running": None,
        "s3_checkpoint_id": s3_checkpoint_id,
        "baseline_checkpoint_id": baseline_checkpoint_id,
        "cache_identity_hash": cache_identity_hash,
        "pointers": pointers,
        "heads": heads,
    }
    payload["report_hash"] = _payload_hash(payload)
    return payload

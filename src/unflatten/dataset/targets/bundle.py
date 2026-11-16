"""Canonical Target Bundle builder and validator (RFC003 §27–§28).

`build_target_bundle` mechanically derives the model-neutral training
target from a canonical AST: structural events (A), text spans (B), typed
property labels (C), known masks + label provenance (D) and the visible
text event stream (E), plus counts and the semantic `ast_hash`.

Identity closure (§27): the bundle carries `document_bundle_id`,
`ast_file_hash`, the `projector` identity and its own
`target_bundle_hash`, so a consumer can bind target and mask authority
to the exact stored AST file rather than to a claim. The first two are
external inputs: when the caller has none (the synthetic factory has no
DocumentBundle), they are explicit `null` — "not bound" — never a
fabricated value. `observable_mask`/`supervision_mask` deliberately do
NOT live here: §27 puts them in the TaskViewProjection, because they
depend on the render, crop and fragment (see
`unflatten.dataset.fragments.envelope`).

RULE (§27): the bundle is never hand-fixed — derived only. To correct a
bundle, fix the AST and re-derive.

Byte determinism: identical canonical AST input yields a byte-identical
bundle under `unflatten.ast.canonical_json.to_canonical_json` (pinned by
a double-run test and a golden sha256 over a corpus fixture).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from unflatten.ast.hashing import ast_hash
from unflatten.ast.issues import Issue
from unflatten.dataset.identity import domain_hash
from unflatten.dataset.targets.events import _TRANSPARENT_CONTAINER_TYPES as FLATTENED_NODE_TYPES
from unflatten.dataset.targets.events import derive_event_streams, reconstruct_visible_events
from unflatten.dataset.targets.properties import (
    emit_property_labels,
    normalize_pointer,
    pointer_registry,
)

TARGET_SCHEMA_VERSION = "0.2.0"

#: RFC003 §27 projector identity. The version tracks the derivation, not
#: the schema; `config_hash` digests the projector's actual configuration
#: (label provenance + the pinned property registry), so two bundles that
#: differ only in how their labels were produced differ in identity.
PROJECTOR_NAME = "canonical-target-projector"
PROJECTOR_VERSION = "0.1.0"

_PROJECTOR_CONFIG_DOMAIN = "rfc003.target-projector-config/v1"
_TARGET_BUNDLE_HASH_DOMAIN = "rfc003.canonical-target-bundle/v1"

#: Fields the §27 `target_bundle_hash` payload covers, in RFC order. The
#: hash field itself, `provenance` and `counts` are excluded (the first
#: would be a cycle, the last two are derived restatements).
_HASHED_FIELDS: tuple[str, ...] = (
    "target_schema_version",
    "document_bundle_id",
    "ast_hash",
    "ast_file_hash",
    "projector",
    "structural",
    "text",
    "typed",
)

_BUNDLE_ID_RE = re.compile(r"^bundle_[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "target_bundle.schema.json"
_TEXT_SPAN_ID_RE = re.compile(r"^txt_(0|[1-9][0-9]*)$")


@lru_cache(maxsize=1)
def _load_schema() -> dict:
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _validator() -> jsonschema.protocols.Validator:
    schema = _load_schema()
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _projector(label_origin: str, verification: str) -> dict[str, Any]:
    return {
        "name": PROJECTOR_NAME,
        "version": PROJECTOR_VERSION,
        "config_hash": domain_hash(
            _PROJECTOR_CONFIG_DOMAIN,
            {
                "label_origin": label_origin,
                "verification": verification,
                "property_registry": pointer_registry(),
            },
        ),
    }


def target_bundle_hash(bundle: dict[str, Any]) -> str:
    """`H("rfc003.canonical-target-bundle/v1", ...)` over the §27 fields.

    Excludes the bundle's own hash field, so embedding the result back
    into the bundle is a fixed point.
    """
    return domain_hash(
        _TARGET_BUNDLE_HASH_DOMAIN, {key: bundle.get(key) for key in _HASHED_FIELDS}
    )


def build_target_bundle(
    canonical_ast: dict[str, Any],
    label_origin: str = "synthetic_generator",
    verification: str = "unverified",
    *,
    document_bundle_id: str | None = None,
    ast_file_hash: str | None = None,
) -> dict[str, Any]:
    """Build the Canonical Target Bundle from a `canonicalize()`d AST.

    `document_bundle_id` and `ast_file_hash` bind the bundle to the
    stored AST artifact it was derived from (§27). Pass them whenever the
    caller has them; `None` records "not bound", which is the truth for a
    freshly generated synthetic document that has no DocumentBundle yet.
    """
    if document_bundle_id is not None and not _BUNDLE_ID_RE.fullmatch(document_bundle_id):
        raise ValueError(
            f"document_bundle_id must be 'bundle_<64 lowercase hex>', "
            f"got {document_bundle_id!r}"
        )
    if ast_file_hash is not None and not _SHA256_RE.fullmatch(ast_file_hash):
        raise ValueError(
            f"ast_file_hash must be 'sha256:<64 lowercase hex>', got {ast_file_hash!r}"
        )
    streams = derive_event_streams(canonical_ast)
    typed = emit_property_labels(
        canonical_ast, label_origin=label_origin, verification=verification
    )
    structural = streams["structural"]
    text = streams["text"]
    bundle = {
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "document_bundle_id": document_bundle_id,
        "ast_hash": ast_hash(canonical_ast),
        "ast_file_hash": ast_file_hash,
        "projector": _projector(label_origin, verification),
        "structural": structural,
        "text": text,
        "typed": typed,
        "provenance": {"label_origin": label_origin, "verification": verification},
        "counts": {
            "nodes": len(structural["node_order"]),
            "structural_events": len(structural["events"]),
            "text_spans": len(text["spans"]),
            "visible_text_events": len(text["visible_text_events"]),
            "typed_fields": sum(len(entry["fields"]) for entry in typed.values()),
        },
    }
    bundle["target_bundle_hash"] = target_bundle_hash(bundle)
    return bundle


def validate_target_bundle(bundle: Any) -> list[Issue]:
    """Validate a target bundle; returns Issues, never raises on invalid input."""
    issues = _schema_issues(bundle)
    if issues:
        return issues
    issues.extend(_structural_issues(bundle))
    issues.extend(_span_issues(bundle))
    issues.extend(_visible_issues(bundle))
    issues.extend(_typed_issues(bundle))
    issues.extend(_count_issues(bundle))
    issues.extend(_identity_issues(bundle))
    return issues


def _identity_issues(bundle: dict[str, Any]) -> list[Issue]:
    """The §27 hash must recompute from the bundle it travels in."""
    recomputed = target_bundle_hash(bundle)
    if bundle["target_bundle_hash"] == recomputed:
        return []
    return [
        Issue(
            code="TARGET_HASH_MISMATCH",
            severity="error",
            node_id=None,
            path="/target_bundle_hash",
            message=f"target_bundle_hash does not recompute from the bundle "
            f"(expected {recomputed}).",
        )
    ]


def _schema_issues(bundle: Any) -> list[Issue]:
    return [
        Issue(
            code="TARGET_SCHEMA_VIOLATION",
            severity="error",
            node_id=None,
            path="/" + "/".join(str(p) for p in error.absolute_path),
            message=error.message,
        )
        for error in sorted(_validator().iter_errors(bundle), key=lambda e: str(e.absolute_path))
    ]


def _balance_issue(path: str, message: str) -> Issue:
    return Issue(
        code="EVENT_BALANCE_VIOLATION",
        severity="error",
        node_id=None,
        path=path,
        message=message,
    )


def _structural_issues(bundle: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    structural = bundle["structural"]
    events = structural["events"]

    for position, event in enumerate(events):
        if event["index"] != position:
            issues.append(
                _balance_issue(
                    f"/structural/events/{position}",
                    f"event index {event['index']} does not match position {position}.",
                )
            )
            return issues

    stack: list[dict[str, Any]] = []
    open_events: list[dict[str, Any]] = []
    replay_parent: list[int] = []
    replay_close: dict[int, int] = {}  # open-event ordinal -> close event index
    ordinal_by_open_index: dict[int, int] = {}
    for event in events:
        op = event["op"]
        if op == "open":
            replay_parent.append(stack[-1]["index"] if stack else -1)
            ordinal_by_open_index[event["index"]] = len(open_events)
            stack.append(event)
            open_events.append(event)
        elif op == "close":
            if not stack:
                issues.append(
                    _balance_issue(
                        f"/structural/events/{event['index']}",
                        "close event without a matching open.",
                    )
                )
                return issues
            top = stack.pop()
            if top["node_id"] != event["node_id"] or top["index"] != event["open_index"]:
                issues.append(
                    _balance_issue(
                        f"/structural/events/{event['index']}",
                        f"close of {event['node_id']!r} (open_index {event['open_index']}) does "
                        f"not match the innermost open {top['node_id']!r} at {top['index']}.",
                    )
                )
                return issues
            replay_close[ordinal_by_open_index[top["index"]]] = event["index"]
        else:  # text
            if not stack or stack[-1]["node_type"] != "run" or (
                stack[-1]["node_id"] != event["node_id"]
            ):
                issues.append(
                    _balance_issue(
                        f"/structural/events/{event['index']}",
                        "text event is not enclosed by the open event of its run node.",
                    )
                )
                return issues
    if stack:
        issues.append(
            _balance_issue(
                "/structural/events",
                f"{len(stack)} open event(s) have no matching close.",
            )
        )
        return issues

    expected_order = [e["node_id"] for e in open_events]
    expected_types = [e["node_type"] for e in open_events]
    expected_close = [replay_close[i] for i in range(len(open_events))]
    if structural["node_order"] != expected_order or structural["node_types"] != expected_types:
        issues.append(
            _balance_issue(
                "/structural/node_order",
                "node_order/node_types do not mirror the open-event sequence.",
            )
        )
    if structural["parent_index"] != replay_parent:
        issues.append(
            _balance_issue(
                "/structural/parent_index",
                "parent_index does not match the parents' open event indices.",
            )
        )
    if structural["close_event_index"] != expected_close:
        issues.append(
            _balance_issue(
                "/structural/close_event_index",
                "close_event_index does not match the matching close events.",
            )
        )
    return issues


def _span_issues(bundle: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    events = bundle["structural"]["events"]
    text_events = [e for e in events if e["op"] == "text"]
    spans = bundle["text"]["spans"]

    for event in text_events:
        span_id = event["text_span_id"]
        if not _TEXT_SPAN_ID_RE.fullmatch(span_id) or int(span_id[4:]) != event["index"]:
            issues.append(
                Issue(
                    code="TEXT_SPAN_ID_INVALID",
                    severity="error",
                    node_id=event["node_id"],
                    path=f"/structural/events/{event['index']}/text_span_id",
                    message=f"text_span_id {span_id!r} must be 'txt_' + the event index "
                    f"({event['index']}) with no leading zeros (RFC003 §27).",
                )
            )
    if [s["text_span_id"] for s in spans] != [e["text_span_id"] for e in text_events]:
        issues.append(
            Issue(
                code="TEXT_SPAN_ID_INVALID",
                severity="error",
                node_id=None,
                path="/text/spans",
                message="spans are not 1:1, in order, with the structural text events.",
            )
        )
        return issues

    spans_by_node: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        spans_by_node.setdefault(span["node_id"], []).append(span)
    for node_id, node_spans in spans_by_node.items():
        ordered = sorted(node_spans, key=lambda s: s["start"])
        cursor = 0
        for span in ordered:
            if span["start"] != cursor or span["end"] - span["start"] != len(span["text"]):
                issues.append(
                    Issue(
                        code="SPAN_PARTITION_VIOLATION",
                        severity="error",
                        node_id=node_id,
                        path="/text/spans",
                        message=f"spans of node {node_id!r} do not partition its run text: "
                        f"span {span['text_span_id']!r} has [{span['start']},{span['end']}) "
                        f"for {len(span['text'])} scalar(s) at cursor {cursor}.",
                    )
                )
                break
            cursor = span["end"]
    return issues


def _visible_issues(bundle: dict[str, Any]) -> list[Issue]:
    """Stream E consistency (D4 should_fix 5): indices contiguous, per-kind
    counts consistent with the structural streams, and every RUN_TEXT
    event attributed to the same node as its text span."""
    issues: list[Issue] = []
    visible = bundle["text"]["visible_text_events"]
    spans = bundle["text"]["spans"]

    for position, event in enumerate(visible):
        if event["index"] != position:
            issues.append(
                Issue(
                    code="VISIBLE_STREAM_VIOLATION",
                    severity="error",
                    node_id=event["node_id"],
                    path=f"/text/visible_text_events/{position}/index",
                    message=f"visible event index {event['index']} does not match "
                    f"position {position}.",
                )
            )
            return issues

    # Order-sensitive check (re-verify should_fix): reconstruct stream E
    # from the structural stream and compare element-wise — a reordered,
    # duplicated or dropped visible event cannot pass.
    expected_visible = reconstruct_visible_events(bundle["structural"])
    if len(visible) != len(expected_visible):
        issues.append(
            Issue(
                code="VISIBLE_STREAM_VIOLATION",
                severity="error",
                node_id=None,
                path="/text/visible_text_events",
                message=f"visible stream has {len(visible)} events, the structural "
                f"stream derives {len(expected_visible)}.",
            )
        )
    else:
        for position, (got, expected) in enumerate(zip(visible, expected_visible)):
            if got != expected:
                issues.append(
                    Issue(
                        code="VISIBLE_STREAM_VIOLATION",
                        severity="error",
                        node_id=got.get("node_id"),
                        path=f"/text/visible_text_events/{position}",
                        message="visible event diverges from the structural-stream "
                        f"derivation (got {got.get('event')!r} for node "
                        f"{got.get('node_id')!r}, expected {expected.get('event')!r} "
                        f"for node {expected.get('node_id')!r}).",
                    )
                )
                break

    spans_by_id = {span["text_span_id"]: span for span in spans}
    for position, event in enumerate(visible):
        if event["event"] != "RUN_TEXT":
            continue
        span = spans_by_id.get(event["text_span_id"])
        if span is None or span["node_id"] != event["node_id"]:
            span_node = span["node_id"] if span is not None else None
            issues.append(
                Issue(
                    code="SPAN_ATTRIBUTION_VIOLATION",
                    severity="error",
                    node_id=event["node_id"],
                    path=f"/text/visible_text_events/{position}",
                    message=f"RUN_TEXT event for node {event['node_id']!r} does not "
                    f"agree with its text span {event['text_span_id']!r} "
                    f"(span node: {span_node!r}).",
                )
            )
    return issues


def _typed_issues(bundle: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    registry = pointer_registry()
    node_types = dict(
        zip(bundle["structural"]["node_order"], bundle["structural"]["node_types"])
    )
    for node_id, entry in bundle["typed"].items():
        node_registry = registry.get(entry["node_type"])
        # DEC-043 §1: a flattened node type (`hyperlink`) never appears in
        # the structural stream by design (`events._TRANSPARENT_CONTAINER_
        # TYPES`) -- `emit_property_labels` still walks the full canonical
        # AST and emits its typed entry (needed for `masks.py`'s own
        # full-AST NO_CHANNEL_POINTERS assessment to have a matching typed
        # label to mask), so its absence from `node_types` is expected,
        # not an inconsistency.
        structurally_present = node_types.get(node_id) == entry["node_type"]
        if node_registry is None or not (
            structurally_present or entry["node_type"] in FLATTENED_NODE_TYPES
        ):
            issues.append(
                Issue(
                    code="UNKNOWN_PROPERTY_POINTER",
                    severity="error",
                    node_id=node_id,
                    path=f"/typed/{node_id}/node_type",
                    message=f"typed entry node_type {entry['node_type']!r} is not a registry "
                    "node type matching the structural stream.",
                )
            )
            continue
        for position, field in enumerate(entry["fields"]):
            pointer = field["node_property_pointer"]
            expected = node_registry.get(normalize_pointer(pointer))
            if expected is None or field["value_type"] != expected:
                issues.append(
                    Issue(
                        code="UNKNOWN_PROPERTY_POINTER",
                        severity="error",
                        node_id=node_id,
                        path=f"/typed/{node_id}/fields/{position}",
                        message=f"pointer {pointer!r} with value_type "
                        f"{field['value_type']!r} is not in the pinned registry for "
                        f"{entry['node_type']!r}.",
                    )
                )
    return issues


def _count_issues(bundle: dict[str, Any]) -> list[Issue]:
    expected = {
        "nodes": len(bundle["structural"]["node_order"]),
        "structural_events": len(bundle["structural"]["events"]),
        "text_spans": len(bundle["text"]["spans"]),
        "visible_text_events": len(bundle["text"]["visible_text_events"]),
        "typed_fields": sum(len(e["fields"]) for e in bundle["typed"].values()),
    }
    return [
        Issue(
            code="COUNT_MISMATCH",
            severity="error",
            node_id=None,
            path=f"/counts/{key}",
            message=f"counts.{key} is {bundle['counts'][key]}, derived value is {value}.",
        )
        for key, value in expected.items()
        if bundle["counts"][key] != value
    ]

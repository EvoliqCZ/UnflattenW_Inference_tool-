"""Minimal ModelIssue Error Registry (RFC004 §42.1/§42.4/§42.7, design §2.7,
T5a of `spec-ast-builder-t5-promotion` P5).

`src/` carries no §42.1 Error Registry today; this module founds it, but
only for the codes the internal `PromotionEnvelope` (`promotion.py`) can
actually emit -- `_build_raw`'s own builder codes (`ast_builder.py`,
`resource_binder.py`) plus the three envelope-level codes design §2.7
names alongside them (`VALIDATOR_CONSISTENCY_MISMATCH`,
`RESULT_ENVELOPE_INVALID`, `RENDERER_GATE_FAILED`). The 23 renderer-native
error codes are deliberately NOT registered one by one (design §2.7,
REJECTED as speculative): a renderer failure carries them in
`details.renderer_codes` under the single `RENDERER_GATE_FAILED` code.

``to_model_issue`` implements plan U11's lookup order for a builder
``UNRESOLVED_REQUIRED_PROPERTY`` issue -- checked BEFORE the general
mapping table below:

1. Is ``(node_type, pointer)`` a whole repeated container
   (`typed_property_pointer_map.repeated_containers()`)? A missing
   container has no typed head at all (no model head predicts a
   *count*) -- interim ``AST_SEMANTIC_INVALID``.
2. Otherwise replace ``pointer``'s numeric segments with ``"*"`` and look
   the exact key up in ``property_head_registry.property_head_registry()``
   (plan U11: NOT ``ast_required.registry_pointer_matches``, which the
   plan calls out as wrong in both directions -- it matches a table entry
   ending in ``/*`` against an UN-normalized query pointer, or vice versa,
   neither of which is the same test as "this exact indexed pointer has a
   head"). A hit is ``TYPED_HEAD_REQUIRED_ABSTAINED``; a miss is interim
   ``AST_SEMANTIC_INVALID`` (covers `EXCLUDED_PROPERTY_POINTERS` and the
   two `_BUILDER_POINTERS`, neither of which ever legitimately reaches
   this function from `_build_raw` itself -- `check_required` never
   reports them -- but a synthetic issue naming one must still resolve
   to "not a head", never a false `TYPED_HEAD_REQUIRED_ABSTAINED`).

The four "interim" builder codes (design §2.7: closure failures with no
RFC004-native code yet -- Q2, proposed errata `AST_NOTE_REF_UNBOUND` /
`AST_STORY_UNRESOLVED` / `AST_REPEATED_INDEX_OUT_OF_RANGE` /
`AST_REQUIRED_PROPERTY_UNRESOLVED`) all borrow `AST_SEMANTIC_INVALID` with
`details.interim = true` and `details.builder_code` naming the real
reason; a `CANONICALIZATION_REFUSED`-sourced `AST_SEMANTIC_INVALID` (an
actual `semantic_validator.py` code, or the internal-anchor/note-ref
renumbering guards) is NOT interim -- it is the correct, final RFC004
code, not a placeholder.

Registry `category`/`stage` for the three envelope-level codes
(`VALIDATOR_CONSISTENCY_MISMATCH`, `RESULT_ENVELOPE_INVALID`,
`RENDERER_GATE_FAILED`) have no RFC004-pinned value; this module's choice
is PROPOSED (design §2.7 founds the registry, does not pin these three):
`VALIDATOR_CONSISTENCY_MISMATCH` sits at `AST_SEMANTIC`/`ast_validate`
(same axis and stage as the S1 re-run that raises it, §34/§2.2 T5b);
`RESULT_ENVELOPE_INVALID` sits at `DETERMINISM`/`package_result` (an
envelope-shape defect discovered only once the whole result is
assembled, not a document-content defect); `RENDERER_GATE_FAILED` sits at
`RENDERER`/`renderer_gate` (RFC002 S1-S11's own name for exactly this
step). Each code gets exactly one registry row regardless of which
builder-issue origin produced it (`AST_SEMANTIC_INVALID`'s `stage` is
therefore one fixed value, `ast_validate`, even though the interim
origin's own failure happens earlier during `_build_raw`/`ast_build` --
the two origins are mutually exclusive within one `BuildResult.issues`
list (`build_strict` gates on `_build_raw`'s issues before ever calling
`canonicalize`), so no single `failure_report` ever needs to disambiguate
them by stage; `details.interim`/`details.builder_code` already do).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jsonschema.validators

from unflatten.model import ast_builder as ab
from unflatten.model import resource_binder as rb
from unflatten.model.property_head_registry import property_head_registry
from unflatten.model.typed_property_pointer_map import repeated_containers

#: RFC004 §42.1's V0 total stage order, explicit zero-based ordinal.
STAGE_ORDER: tuple[str, ...] = (
    "input",
    "preprocess",
    "component_load",
    "backbone",
    "visual_memory",
    "compress",
    "context_encode",
    "decode",
    "align",
    "typed_property",
    "table",
    "resource",
    "fragment",
    "assemble",
    "ast_build",
    "canonicalize",
    "ast_validate",
    "renderer_gate",
    "render",
    "calibrate",
    "accept",
    "package_result",
)
STAGE_ORDINAL: dict[str, int] = {name: index for index, name in enumerate(STAGE_ORDER)}

Severity = Literal["info", "warning", "error", "fatal"]
_SEVERITY_ORDER: dict[str, int] = {"fatal": 0, "error": 1, "warning": 2, "info": 3}

FAILURE_REPORT_SCHEMA_VERSION = "0.1.0"


class ModelIssueRegistryError(ValueError):
    """A code is not in the registry, or `details` fails its schema."""


@dataclass(frozen=True, slots=True)
class ModelIssueRecord:
    code: str
    category: str
    stage: str
    severity: Severity
    recoverability: str
    policy_action: str
    #: Closed JSON Schema (`additionalProperties: false`) `details` must
    #: validate against (§42.1: "details musí celý validovať voči tomuto
    #: per-code schema").
    details_schema: Mapping[str, Any]


def _details_schema(
    *,
    builder_codes: tuple[str, ...] | None = None,
    extra: Mapping[str, Any] | None = None,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    """One shared shape for every code's `details_schema`: `source` (the
    builder's `a<k>` assembled id, plan U13) is always an optional string;
    `builder_code`/`interim`/`inner_code` are added only for codes that
    carry them."""
    properties: dict[str, Any] = {"source": {"type": "string"}}
    if builder_codes is not None:
        properties["builder_code"] = {"type": "string", "enum": list(builder_codes)}
        properties["interim"] = {"type": "boolean"}
        properties["inner_code"] = {"type": "string"}
    if extra is not None:
        properties.update(extra)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": properties,
    }


#: §42.6: `severity = fatal` requires `policy_action = fail_request`.
RECOVERABILITY_VALUES = frozenset(
    {"none", "same_config", "larger_window", "primary_decoder", "human_review"}
)
POLICY_ACTION_VALUES = frozenset(
    {
        "continue",
        "continue_with_warning",
        "retry",
        "return_review_candidate",
        "return_partial",
        "set_acceptance_review_required",
        "reject",
        "fail_request",
    }
)

#: §42.3 closed category enum.
CATEGORY_VALUES = frozenset(
    {
        "INPUT",
        "PREPROCESS",
        "PROFILE",
        "COMPONENT_LOCK",
        "BACKBONE",
        "VISUAL_MEMORY",
        "COMPRESSOR",
        "ALIGNMENT",
        "DECODER",
        "GRAMMAR",
        "TEXT",
        "MOE_ROUTER",
        "TYPED_PROPERTY",
        "TABLE",
        "RESOURCE",
        "FRAGMENT",
        "ASSEMBLY",
        "AST_SCHEMA",
        "AST_SEMANTIC",
        "CAPABILITY",
        "RENDERER",
        "CALIBRATION",
        "ACCEPTANCE",
        "LIMIT",
        "SECURITY",
        "DETERMINISM",
        "RUNTIME",
    }
)


def _rec(
    code: str,
    category: str,
    stage: str,
    severity: Severity,
    recoverability: str,
    policy_action: str,
    *,
    details_schema: Mapping[str, Any] | None = None,
) -> ModelIssueRecord:
    """One §42.4 registry row not sourced from an internal builder issue
    today (design §2.7 only founded the internal-envelope subset; this adds
    the remaining RFC004-native codes as inert rows -- nothing in `src/`
    emits them yet, so their `details_schema` is the shared empty shape
    unless a caller passes one). §42.6's fatal/fail_request pairing and the
    closed severity/category/recoverability/policy_action enums are checked
    by `tests/unit/model/test_model_issue_registry.py`'s registry-closure
    test, not re-checked here at import time (a raised error at import
    would be surprising for a pure data table)."""
    return ModelIssueRecord(
        code=code,
        category=category,
        stage=stage,
        severity=severity,
        recoverability=recoverability,
        policy_action=policy_action,
        details_schema=details_schema if details_schema is not None else _details_schema(),
    )


#: §42.4 codes with no producer in `src/` yet (no TablePlan, speculative
#: decoder, MoE router, etc. exist). Registered so the closed registry and
#: `failure_report`/Component-Lock error-code-registry hash are complete
#: ahead of their producers (RFC004 §39.2 "error-code registry" is a
#: pinned Component Lock artifact independent of which codes a given
#: release path can currently raise).
_UNPRODUCED_RECORDS: tuple[ModelIssueRecord, ...] = (
    _rec(
        "MODEL_INPUT_SCHEMA_INVALID",
        "INPUT",
        "input",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "MODEL_INPUT_MIME_UNSUPPORTED",
        "INPUT",
        "input",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "MODEL_INPUT_PIXELS_INVALID",
        "INPUT",
        "input",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "MODEL_INPUT_FORBIDDEN_CHANNEL",
        "INPUT",
        "input",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "MODEL_INPUT_LIMIT_EXCEEDED",
        "LIMIT",
        "input",
        "error",
        "none",
        "reject",
    ),
    _rec(
        "PREPROCESS_TRANSFORM_INVALID",
        "PREPROCESS",
        "preprocess",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "TOKEN_BUDGET_EXCEEDED",
        "LIMIT",
        "decode",
        "error",
        "larger_window",
        "retry",
    ),
    _rec(
        "MODEL_COMPONENT_LOCK_MISMATCH",
        "COMPONENT_LOCK",
        "component_load",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "MODEL_PROFILE_UNSUPPORTED",
        "PROFILE",
        "component_load",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "BACKBONE_ENCODE_FAILED",
        "BACKBONE",
        "backbone",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "VISUAL_MEMORY_INVALID",
        "VISUAL_MEMORY",
        "visual_memory",
        "error",
        "same_config",
        "retry",
    ),
    _rec(
        "COMPRESSOR_OUTPUT_INVALID",
        "COMPRESSOR",
        "compress",
        "error",
        "same_config",
        "retry",
    ),
    _rec(
        "ALIGNMENT_FAILED",
        "ALIGNMENT",
        "align",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "MOE_ROUTER_INVALID_EXPERT",
        "MOE_ROUTER",
        "decode",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "MOE_ROUTER_CAPACITY_EXCEEDED",
        "MOE_ROUTER",
        "decode",
        "error",
        "larger_window",
        "retry",
    ),
    _rec(
        "DECODER_INVALID_EVENT",
        "DECODER",
        "decode",
        "error",
        "same_config",
        "retry",
    ),
    _rec(
        "DECODER_GRAMMAR_DEAD_END",
        "GRAMMAR",
        "decode",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "DECODER_UNBALANCED_TREE",
        "DECODER",
        "decode",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "DECODER_EVENT_LIMIT",
        "LIMIT",
        "decode",
        "error",
        "larger_window",
        "retry",
    ),
    _rec(
        "DECODER_INCOMPLETE",
        "DECODER",
        "decode",
        "error",
        "human_review",
        "return_partial",
    ),
    _rec(
        "DECODER_NON_TERMINATION",
        "DECODER",
        "decode",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "TEXT_INVALID_UNICODE",
        "TEXT",
        "decode",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "TEXT_TOKENIZER_ROUNDTRIP_FAILED",
        "TEXT",
        "decode",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "TYPED_HEAD_UNKNOWN_POINTER",
        "TYPED_PROPERTY",
        "typed_property",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "TYPED_HEAD_NODE_TYPE_MISMATCH",
        "TYPED_PROPERTY",
        "typed_property",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "TYPED_HEAD_VALUE_INVALID",
        "TYPED_PROPERTY",
        "typed_property",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "TYPED_HEAD_DUPLICATE_PROPERTY",
        "TYPED_PROPERTY",
        "typed_property",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "TABLE_GRID_INVALID",
        "TABLE",
        "table",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "TABLE_SPAN_CONFLICT",
        "TABLE",
        "table",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "FRAGMENT_SEQUENCE_GAP",
        "FRAGMENT",
        "fragment",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "FRAGMENT_SEQUENCE_DUPLICATE",
        "FRAGMENT",
        "fragment",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "FRAGMENT_BOUNDARY_CONFLICT",
        "ASSEMBLY",
        "assemble",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "FRAGMENT_CONTINUATION_AMBIGUOUS",
        "ASSEMBLY",
        "assemble",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "FRAGMENT_OVERLAP_CONFLICT",
        "ASSEMBLY",
        "assemble",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "ASSEMBLER_TEXT_CONFLICT",
        "ASSEMBLY",
        "assemble",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "ASSEMBLER_STRUCTURE_CONFLICT",
        "ASSEMBLY",
        "assemble",
        "error",
        "human_review",
        "return_review_candidate",
    ),
    _rec(
        "ASSEMBLER_NONDETERMINISTIC",
        "DETERMINISM",
        "assemble",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "AST_PROFILE_INVALID",
        "AST_SEMANTIC",
        "ast_validate",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "RENDERER_PREFLIGHT_FAILED",
        "RENDERER",
        "renderer_gate",
        "error",
        "none",
        "reject",
    ),
    _rec(
        "RENDERER_OPEN_PROBE_FAILED",
        "RENDERER",
        "renderer_gate",
        "error",
        "none",
        "reject",
    ),
    _rec(
        "RESULT_ID_MISMATCH",
        "DETERMINISM",
        "package_result",
        "fatal",
        "none",
        "fail_request",
    ),
    _rec(
        "ACCEPTANCE_POLICY_MISMATCH",
        "ACCEPTANCE",
        "accept",
        "fatal",
        "none",
        "fail_request",
    ),
)

#: One record per §42.4 code the internal envelope can emit (design §2.7).
MODEL_ISSUE_REGISTRY: dict[str, ModelIssueRecord] = {
    "TYPED_HEAD_REQUIRED_ABSTAINED": ModelIssueRecord(
        code="TYPED_HEAD_REQUIRED_ABSTAINED",
        category="TYPED_PROPERTY",
        stage="typed_property",
        severity="error",
        recoverability="human_review",
        policy_action="return_review_candidate",
        details_schema=_details_schema(
            builder_codes=(ab.UNRESOLVED_REQUIRED_PROPERTY,),
            required=("builder_code", "interim"),
        ),
    ),
    "AST_SEMANTIC_INVALID": ModelIssueRecord(
        code="AST_SEMANTIC_INVALID",
        category="AST_SEMANTIC",
        stage="ast_validate",
        severity="error",
        recoverability="human_review",
        policy_action="return_review_candidate",
        details_schema=_details_schema(
            builder_codes=(
                ab.UNRESOLVED_REQUIRED_PROPERTY,
                ab.NOTE_REF_UNBOUND,
                ab.STORY_UNRESOLVED,
                ab.REPEATED_INDEX_OUT_OF_RANGE,
                ab.CANONICALIZATION_REFUSED,
            ),
            required=("builder_code", "interim"),
        ),
    ),
    "AST_SCHEMA_INVALID": ModelIssueRecord(
        code="AST_SCHEMA_INVALID",
        category="AST_SCHEMA",
        stage="canonicalize",
        severity="error",
        recoverability="human_review",
        policy_action="return_review_candidate",
        details_schema=_details_schema(
            builder_codes=(ab.CANONICALIZATION_REFUSED,),
            required=("builder_code", "interim", "inner_code"),
        ),
    ),
    "AST_CANONICALIZATION_FAILED": ModelIssueRecord(
        code="AST_CANONICALIZATION_FAILED",
        category="AST_SEMANTIC",
        stage="canonicalize",
        severity="error",
        recoverability="human_review",
        policy_action="return_review_candidate",
        details_schema=_details_schema(
            builder_codes=(ab.CANONICALIZATION_REFUSED,),
            required=("builder_code", "interim", "inner_code"),
        ),
    ),
    "RESOURCE_SLOT_UNRESOLVED": ModelIssueRecord(
        code="RESOURCE_SLOT_UNRESOLVED",
        category="RESOURCE",
        stage="resource",
        severity="error",
        recoverability="human_review",
        policy_action="return_review_candidate",
        details_schema=_details_schema(),
    ),
    "RESOURCE_SLOT_DUPLICATE": ModelIssueRecord(
        code="RESOURCE_SLOT_DUPLICATE",
        category="RESOURCE",
        stage="resource",
        severity="error",
        recoverability="human_review",
        policy_action="return_review_candidate",
        details_schema=_details_schema(),
    ),
    "RESOURCE_BYTES_INVALID": ModelIssueRecord(
        code="RESOURCE_BYTES_INVALID",
        category="RESOURCE",
        stage="resource",
        severity="error",
        recoverability="human_review",
        policy_action="return_review_candidate",
        details_schema=_details_schema(),
    ),
    "RESOURCE_CROP_LOW_CONFIDENCE": ModelIssueRecord(
        code="RESOURCE_CROP_LOW_CONFIDENCE",
        category="RESOURCE",
        stage="resource",
        severity="error",
        recoverability="human_review",
        policy_action="return_review_candidate",
        details_schema=_details_schema(),
    ),
    "RESOURCE_CROP_TOO_SMALL": ModelIssueRecord(
        code="RESOURCE_CROP_TOO_SMALL",
        category="RESOURCE",
        stage="resource",
        severity="error",
        recoverability="human_review",
        policy_action="return_review_candidate",
        details_schema=_details_schema(),
    ),
    "RESOURCE_CROP_BLANK": ModelIssueRecord(
        code="RESOURCE_CROP_BLANK",
        category="RESOURCE",
        stage="resource",
        severity="error",
        recoverability="human_review",
        policy_action="return_review_candidate",
        details_schema=_details_schema(),
    ),
    "ASSEMBLER_PROPERTY_CONFLICT": ModelIssueRecord(
        code="ASSEMBLER_PROPERTY_CONFLICT",
        category="ASSEMBLY",
        stage="assemble",
        severity="error",
        recoverability="human_review",
        policy_action="return_review_candidate",
        details_schema=_details_schema(),
    ),
    "VALIDATOR_CONSISTENCY_MISMATCH": ModelIssueRecord(
        code="VALIDATOR_CONSISTENCY_MISMATCH",
        category="AST_SEMANTIC",
        stage="ast_validate",
        severity="fatal",
        recoverability="none",
        policy_action="fail_request",
        details_schema=_details_schema(
            extra={"inner_codes": {"type": "array", "items": {"type": "string"}}}
        ),
    ),
    "RESULT_ENVELOPE_INVALID": ModelIssueRecord(
        code="RESULT_ENVELOPE_INVALID",
        category="DETERMINISM",
        stage="package_result",
        severity="fatal",
        recoverability="none",
        policy_action="fail_request",
        details_schema=_details_schema(),
    ),
    "RENDERER_GATE_FAILED": ModelIssueRecord(
        code="RENDERER_GATE_FAILED",
        category="RENDERER",
        stage="renderer_gate",
        severity="error",
        recoverability="none",
        policy_action="reject",
        details_schema=_details_schema(
            extra={"renderer_codes": {"type": "array", "items": {"type": "string"}}}
        ),
    ),
    **{record.code: record for record in _UNPRODUCED_RECORDS},
}

#: Builder codes design §2.7 marks interim: closure failures with no
#: RFC004-native code of their own yet (Q2). Always map to
#: `AST_SEMANTIC_INVALID` with `details.interim = true`.
_INTERIM_UNBOUND_CODES = frozenset(
    {ab.NOTE_REF_UNBOUND, ab.STORY_UNRESOLVED, ab.REPEATED_INDEX_OUT_OF_RANGE}
)

#: Builder/binder codes that already ARE their own §42.4 code -- passed
#: through unchanged.
_DIRECT_CODES = frozenset(
    {
        rb.RESOURCE_SLOT_UNRESOLVED,
        rb.RESOURCE_SLOT_DUPLICATE,
        rb.RESOURCE_BYTES_INVALID,
        rb.RESOURCE_CROP_LOW_CONFIDENCE,
        rb.RESOURCE_CROP_TOO_SMALL,
        rb.RESOURCE_CROP_BLANK,
        ab.ASSEMBLER_PROPERTY_CONFLICT,
    }
)

#: `semantic_validator.py`'s own code constants, mirrored here (module
#: docstring point 2 of design §2.7): a `CANONICALIZATION_REFUSED` whose
#: inner code is one of these -- or the internal-anchor/note-ref
#: renumbering guards `canonicalizer.py` raises directly -- is the real,
#: final `AST_SEMANTIC_INVALID`, never interim.
_SEMANTIC_VALIDATOR_CODES = frozenset(
    {
        "BROKEN_INTERNAL_ANCHOR",
        "BROKEN_NOTE_REF",
        "BROKEN_RESOURCE_REF",
        "CELL_LAST_CHILD_NOT_PARAGRAPH",
        "CELL_MISSING_PARAGRAPH",
        "COLUMN_WIDTH_COUNT_MISMATCH",
        "DUPLICATE_NODE_ID",
        "DUPLICATE_RESOURCE_ID",
        "DUPLICATE_STORY_KIND",
        "HEADING_LEVEL_MISSING",
        "INCOMPATIBLE_RESOURCE_REF",
        "INDENT_MUTUAL_EXCLUSION",
        "INVALID_CELL_SPAN",
        "INVALID_FONT_SIZE",
        "INVALID_PARENT_CHILD",
        "MARKER_TEXT_FORBIDDEN",
        "MARKER_TEXT_REQUIRED",
        "NESTED_TABLE_UNSUPPORTED",
        "NESTED_TEXT_FRAME_UNSUPPORTED",
        "NOTE_REF_IN_NOTE",
        "ORPHAN_FOOTNOTE",
        "REFERENCE_CYCLE",
        "RUN_SCRIPT_MUTUAL_EXCLUSION",
        "TABLE_GRID_COLUMN_OUT_OF_RANGE",
        "TABLE_GRID_GAP",
        "TABLE_GRID_OVERLAP",
        "TABLE_ROW_SPAN_OUT_OF_RANGE",
        "TABLE_WIDTH_FIELD_MISMATCH",
        "TEXT_FRAME_REQUIRES_VISUAL_V1_1",
        "V1_FEATURE_IN_V0_PROFILE",
    }
)

_NUMERIC_SEGMENT = re.compile(r"/\d+")


def _normalize_pointer(pointer: str) -> str:
    """Replace every numeric JSON-pointer segment with `*` (plan U11)."""
    return _NUMERIC_SEGMENT.sub("/*", pointer)


def _map(issue: ab.BuilderIssue) -> tuple[str, dict[str, Any]]:
    details: dict[str, Any] = {}
    if issue.source is not None:
        details["source"] = issue.source

    if issue.code == ab.UNRESOLVED_REQUIRED_PROPERTY:
        node_type = issue.node_type or ""
        pointer = issue.pointer or ""
        if (node_type, pointer) in repeated_containers():
            # U11 step 1: a whole missing repeated container -- no head
            # predicts a count -- is interim, checked before the head
            # lookup below.
            details.update(builder_code=issue.code, interim=True)
            return "AST_SEMANTIC_INVALID", details
        key = (node_type, _normalize_pointer(pointer))
        if key in property_head_registry():
            # U11 step 2: exact normalized key in `_HEAD_ASSIGNMENTS`.
            details.update(builder_code=issue.code, interim=False)
            return "TYPED_HEAD_REQUIRED_ABSTAINED", details
        details.update(builder_code=issue.code, interim=True)
        return "AST_SEMANTIC_INVALID", details

    if issue.code in _INTERIM_UNBOUND_CODES:
        details.update(builder_code=issue.code, interim=True)
        return "AST_SEMANTIC_INVALID", details

    if issue.code == ab.CANONICALIZATION_REFUSED:
        inner = (issue.detail or "").split(":", 1)[0]
        details.update(builder_code=issue.code, interim=False, inner_code=inner)
        if inner == "SCHEMA_VALIDATION_ERROR":
            return "AST_SCHEMA_INVALID", details
        if inner in _SEMANTIC_VALIDATOR_CODES:
            return "AST_SEMANTIC_INVALID", details
        # Unreachable in practice (canonicalizer.py never raises any
        # other CANONICALIZATION_REFUSED inner code, design §2.7) --
        # exercised only by a synthetic BuilderIssue in tests.
        return "AST_CANONICALIZATION_FAILED", details

    if issue.code in _DIRECT_CODES:
        return issue.code, details

    # An unrecognized builder code is a registry mismatch (§42.5: "unknown
    # code je registry mismatch").
    return "RESULT_ENVELOPE_INVALID", details


def to_model_issue(issue: ab.BuilderIssue) -> dict[str, Any]:
    """Map one `ast_builder.BuilderIssue` onto RFC004 §42.1's closed
    ModelIssue shape. `node_property_pointer` carries `issue.pointer`;
    every other id field is `None` (plan U13) -- the final canonical
    RFC001 node id (`node_id`) has no meaning for a failed build, and
    `issue.source`/`issue.node_id` travel in `details` instead (plan
    U13: "source (a<k>) sa zapíše do uzavretých details schém")."""
    code, details = _map(issue)
    record = MODEL_ISSUE_REGISTRY[code]
    validate_details(code, details)
    return {
        "code": code,
        "category": record.category,
        "severity": record.severity,
        "stage": record.stage,
        "page_ordinal": None,
        "fragment_sequence_index": None,
        "local_node_id": None,
        "node_id": None,
        "node_property_pointer": issue.pointer,
        "ast_pointer": None,
        "message": issue.detail or issue.code,
        "details": details,
        "recoverability": record.recoverability,
        "policy_action": record.policy_action,
    }


def validate_details(code: str, details: Mapping[str, Any]) -> None:
    """Validate `details` against `code`'s closed `details_schema`.
    Raises `ModelIssueRegistryError` for an unknown code or invalid
    `details` (§42.1: "details musí celý validovať voči tomuto per-code
    schema")."""
    record = MODEL_ISSUE_REGISTRY.get(code)
    if record is None:
        raise ModelIssueRegistryError(f"unknown ModelIssue code {code!r}")
    validator_cls = jsonschema.validators.validator_for(record.details_schema)
    errors = list(validator_cls(record.details_schema).iter_errors(details))
    if errors:
        raise ModelIssueRegistryError(f"{code}: {errors[0].message}")


def primary_code(issues: Sequence[Mapping[str, Any]]) -> str:
    """§42.7: severity order `fatal > error > warning > info`, then
    lowest V0 stage ordinal, then code UTF-8 byte order."""
    if not issues:
        raise ValueError("primary_code requires at least one issue")

    def sort_key(issue: Mapping[str, Any]) -> tuple[int, int, bytes]:
        return (
            _SEVERITY_ORDER[issue["severity"]],
            STAGE_ORDINAL[issue["stage"]],
            issue["code"].encode("utf-8"),
        )

    return min(issues, key=sort_key)["code"]


def failure_report(axis: str, issues: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """§42.7 Failure Report shape. `retry_count`/`attempts` are always
    `0`/`[]`: nothing in this module's callers retries yet."""
    if axis not in ("ast", "renderer"):
        raise ValueError(f"axis must be 'ast' or 'renderer', got {axis!r}")
    if not issues:
        raise ValueError("failure_report requires at least one issue")
    return {
        "failure_report_schema_version": FAILURE_REPORT_SCHEMA_VERSION,
        "axis": axis,
        "terminal_status": "failed",
        "primary_code": primary_code(issues),
        "retry_count": 0,
        "attempts": [],
        "issues": list(issues),
    }

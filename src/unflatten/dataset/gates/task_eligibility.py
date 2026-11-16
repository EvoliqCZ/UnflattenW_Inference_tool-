"""Per-TaskView eligibility (RFC003 §30 task kinds, §33.4 shape).

The document-level gate result answers one question — "is this bundle
usable as a full-AST training sample?" — and RFC003 §33.4 is explicit
that this is the wrong granularity: eligibility belongs to the *task*.

It matters in practice. On the imported real corpus roughly two thirds of
documents are `partial_ready` for exactly one reason: the PDF text layer
does not yield geometry for every text node (`MAPPING_INCOMPLETE`).
Geometry is what grounding needs. It is not what structure prediction,
text prediction or Word-property prediction need — those targets come
from the Canonical AST, which is complete. Judging the whole document by
the grounding requirement discards ~90 % of the available pages, and
discards them non-randomly: long documents fail full coverage far more
often than short ones, so a full-coverage-only training set is
systematically biased toward short, simple documents.

This module is a pure projection of an already-computed `GateResult`
plus the document's node-type histogram. It adds no new evidence — it
reads the evidence the gates already produced and asks, per task, "is
what this task needs actually missing?"

**Evidence, not the absence of a failure.** A task view is eligible only
if the gates that judge the model input, the observation and the target
artifacts actually PASSED. A `not_applicable` gate judged nothing: a
rendered document whose targets were underivable has no target bundle,
no mask projection and no fragment index, and calling it `train_ready`
labels a document that cannot produce a single training item.

**v0 scope.** Task views here are *document level*: the whole page set in,
the whole AST out. Page-fragment task views (RFC003 §26) additionally
need per-node page assignment, which comes from the observation sidecar
and is therefore only as complete as the geometry coverage. That
per-page rule is NOT expressed here as a document-wide geometry
requirement on `structure_v0`/`paragraph_run_v0` — it belongs to the
fragment itself, which records per-slice `page_evidence` and withdraws
typed-property supervision for `unresolved` slices. What this module
guarantees is that the fragment index exists and validated at all.

**v0 conservatism.** `table_v0` and `image_v0` require *document-wide*
geometry completeness, though strictly they need only their own nodes
grounded. Refining that needs per-node coverage, which the observation
gate does not currently report. The effect is to under-count those two
task views, never to over-count them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: RFC003 §30 normative `task_kind` enum for v0.1.
TASK_KINDS: tuple[str, ...] = (
    "layout_v0",
    "paragraph_run_v0",
    "table_v0",
    "image_v0",
    "structure_v0",
    "full_ast_exact_v0",
    "partial_ast_v0",
    "renderer_negative_v0",
)

#: Task views whose targets are geometric: they need every required node
#: mapped to rendered geometry (RFC003 §30.1, §30.3, §30.4, §30.6).
_GEOMETRY_TASKS: frozenset[str] = frozenset(
    {"layout_v0", "table_v0", "image_v0", "full_ast_exact_v0"}
)

#: Task views whose targets come from the Canonical AST alone.
_AST_ONLY_TASKS: frozenset[str] = frozenset({"structure_v0", "paragraph_run_v0"})

#: Gates that must have judged actual evidence before ANY task view is
#: eligible: a hash-verified model input (`integrity`, which also
#: requires a rendered document to carry non-empty page images), the
#: per-node observation the fragments are partitioned by, and the
#: target bundle + mask projection + fragment index (`targets`).
_PREREQUISITE_GATES: tuple[str, ...] = ("integrity", "observation", "targets")

#: Reason recorded when a prerequisite gate judged nothing.
_ABSENT_EVIDENCE: dict[str, str] = {
    "integrity": "INPUT_ABSENT",
    "observation": "OBSERVATION_ABSENT",
    "targets": "TARGETS_ABSENT",
}

#: Node type each task view predicts; absent node type => nothing to learn.
_REQUIRED_NODE: dict[str, tuple[str, str]] = {
    "table_v0": ("table", "NO_TABLE_NODE"),
    "image_v0": ("inline_image", "NO_IMAGE_NODE"),
}


def _view(eligibility: str, reason_codes: list[str]) -> dict[str, Any]:
    return {"eligibility": eligibility, "reason_codes": sorted(set(reason_codes))}


def task_eligibility(
    result: Mapping[str, Any], node_types: Mapping[str, int]
) -> dict[str, dict[str, Any]]:
    """Project one document-level gate result onto the §30 task kinds.

    `result` is a serialized `GateResult`; `node_types` is the document's
    node-type histogram (the `coverage.node_types` block M2c computes).
    """
    reason_codes = list(result.get("reason_codes", []))
    failed = [
        check
        for check in result.get("gate_report", [])
        if check.get("status") == "fail"
    ]

    # A failed hard gate is not task-specific: the bundle's evidence is
    # unusable, so no task view can be built from it.
    if failed:
        return {kind: _view("ineligible", reason_codes) for kind in TASK_KINDS}

    # Every task view's model input is a rendered page image. Without a
    # render there is no input, whatever the labels look like.
    if "RENDER_SKIPPED" in reason_codes:
        return {kind: _view("ineligible", reason_codes) for kind in TASK_KINDS}

    # Positive evidence, not the absence of a failure. `not_applicable`
    # means the gate had nothing to judge — a rendered document whose
    # targets were underivable carries no target bundle, no mask
    # projection and no fragment index, so no task view can be built
    # from it. Treating that as eligible is how a "train_ready" label
    # ends up on a document with no training target at all.
    statuses = {
        check.get("gate_id"): check.get("status")
        for check in result.get("gate_report", [])
    }
    absent = [
        _ABSENT_EVIDENCE[gate]
        for gate in _PREREQUISITE_GATES
        if statuses.get(gate) != "pass"
    ]
    if absent:
        return {kind: _view("ineligible", absent) for kind in TASK_KINDS}

    # A document where NOTHING was observed has no observed placement on
    # any page: every position would be inferred, so every page would be
    # dropped downstream (the feature cache skips a page with no scored
    # position). Claiming train_ready here and dropping the page later is
    # how the corpus loses its hardest documents without a record.
    if any(
        issue.get("code") == "OBSERVATION_NO_PLACEMENT"
        for check in result.get("gate_report", [])
        if check.get("gate_id") == "observation"
        for issue in check.get("issues", [])
    ):
        return {
            kind: _view("ineligible", ["NO_OBSERVED_PLACEMENT"]) for kind in TASK_KINDS
        }

    geometry_complete = "MAPPING_INCOMPLETE" not in reason_codes
    views: dict[str, dict[str, Any]] = {}
    for kind in TASK_KINDS:
        if kind == "partial_ast_v0":
            # §30.6: a TaskView over Candidate Annotation IR, not over a
            # DocumentBundle. This pipeline produces no candidates.
            views[kind] = _view("ineligible", ["NOT_CANDIDATE_IR"])
            continue
        if kind == "renderer_negative_v0":
            views[kind] = _view("ineligible", ["NOT_PRODUCED"])
            continue

        required = _REQUIRED_NODE.get(kind)
        if required is not None and not node_types.get(required[0]):
            views[kind] = _view("ineligible", [required[1]])
            continue

        if kind in _AST_ONLY_TASKS:
            views[kind] = _view("train_ready", [])
        elif kind in _GEOMETRY_TASKS:
            views[kind] = (
                _view("train_ready", [])
                if geometry_complete
                else _view("partial_ready", ["MAPPING_INCOMPLETE"])
            )
        else:  # pragma: no cover - guards the enum against silent drift
            raise ValueError(f"task kind {kind!r} has no eligibility rule")
    return views

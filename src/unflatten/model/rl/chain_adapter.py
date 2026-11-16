"""Rollout trajectory -> reward-engine inputs (S4.5 design D1/D6; PROPOSED, dormant).

``candidate_inputs_from_trajectory`` runs the pinned builder chain
(``eval.builder_chain.run.process_model_page``) over one sampled token stream and
returns the ``CandidateInputs`` the T4 engine scores. Contract:

* **Never raises on a bad stream, never repairs.** ``PageStreamInvalid``,
  ``AssemblyInvariantViolation``, a lifecycle violation, a non-terminated rollout
  (``DecodeMaxStepsExceeded`` / ``DecodeVocabularyExhausted``), a build the strict
  builder refuses, or a candidate whose AST profile is wider than the gold's give
  a FAILED build (``ast is None``) with ``rfc002_render="not_run"``. Any other
  exception is a defect and propagates. Malformed decoded Unicode/byte text is
  likewise a named FAILED candidate, never normalized or repaired.
* **No LibreOffice.** ``rfc002_render`` is the deterministic RFC002 compile
  (``promotion.promote`` with a ``RenderRequest``). It only runs when the caller
  injects ``render_dir``; otherwise it is ``not_run`` (a missing observation).
* **Typed values are read from the built AST**, at the pointers the model's heads
  predicted for that node type, keyed by candidate AST id, then re-keyed onto GOLD
  node ids through ``alignment`` (the decoder's ``local_node_id`` space is not the
  AST's id space: ``n0`` is the first decoded node, ``n1`` the first AST node).
  Typed values and the tree/grid components are scored from the BUILT AST (what
  reaches the Word output); a pointer the model never predicted for a node type
  is not credited, even if a chain policy defaulted it. The text component and
  the text hard gate are the exception: they use the token stream (next item),
  so stream text alone cannot reach the maximum reward -- a built AST that lacks
  a node still loses its tree/structure components.
* **Text** is the candidate's own token stream's text
  (``score.assemble_hypothesis_text``, stories/footnotes/frames included), the
  symmetric counterpart of the gold's ``score.assemble_reference_text`` (RFC003
  text). A failed build has no committed text (``visible_text=None``).
* **Grid** is not an input field: the engine derives candidate cells from the
  built AST and keys both sides by table ordinal and ``(row, col)``; gold without
  cells yields an unobservable (``None``) component, never ``0``.
* ``continuity_micro`` is ``None``: a standalone page has no seam and the
  text-continuity helper needs a frozen policy this module does not own.
* Gold is scorer-side data. It never reaches a model tensor here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from unflatten.assembly.assemble import AssemblyInvariantViolation
from unflatten.assembly.hypothesis import PageStreamInvalid
from unflatten.eval.builder_chain import run
from unflatten.eval.s2_gate import sealed
from unflatten.eval.s2_gate.slices import CONTROL_EVENT_TOKENS
from unflatten.model import composite_vocab as cv
from unflatten.model import events_vocab
from unflatten.model.exact_text import ByteFrameInvalid, TextInvalidUnicode, TextNotNfc
from unflatten.model.node_lifecycle import (
    NodeLifecycleError,
    NodeLifecycleTracker,
    UnsupportedResumeInV0,
)
from unflatten.model.prediction_ir import typed_prediction_projection
from unflatten.model.rl.alignment import (
    align_preorder,
    map_candidate_node_ids,
    node_ids_from_ast,
    rekey_typed,
)
from unflatten.model.rl.policies import CHAIN_FLAGS
from unflatten.model.rl.reward_inputs import (
    CandidateInputs,
    GoldReference,
    gold_from_document_reference,
    preorder_from_ast,
)
from unflatten.model.rl.rollout import STATUS_TERMINATED, RolloutTrajectory

__all__ = [
    "sealed_regions",
    "CONTROL_EVENT_TOKENS",
    "AdapterBuild",
    "ChainAdapterRefused",
    "candidate_inputs_from_trajectory",
    "control_event_counts",
    "gold_node_ids_parallel",
    "gold_reference_from_chain",
]

#: Smallest-first; a candidate may be narrower than the gold, never wider.
_PROFILE_RANK: Final = {"word-core-v0": 0, "word-visual-v1": 1, "word-visual-v1.1": 2}
_TOKEN_NAME_BY_ID: Final = {i: n for n, i in events_vocab.TOKEN_IDS.items()}
_CONTROL_NAME_BY_ID: Final = {i: n for n, i in CONTROL_EVENT_TOKENS.items()}
#: Page outcomes that name a resource/fragment-integrity finding (not an invalid AST).
_INTEGRITY_OUTCOMES: Final = ("AssemblyInvariantViolation",)
#: Named "this page failed" shapes; everything else is a defect and propagates.
_PAGE_FAILURES: Final = (
    PageStreamInvalid,
    AssemblyInvariantViolation,
    NodeLifecycleError,
    UnsupportedResumeInV0,
)
_TEXT_FAILURES: Final = (TextNotNfc, TextInvalidUnicode, ByteFrameInvalid)
#: Pinned table-geometry column source for every RL chain call. Not hashed (DEC-048
#: addendum): it is bound here so the reward does not drift with the chain default.
TABLE_GEOMETRY_COLUMN_SOURCE: Final = "cell_median"
_RENDERED: Final = ("render_ready", "approximation_required")


class ChainAdapterRefused(ValueError):
    """The caller's pairing/gold inputs are inconsistent (a caller defect, not a bad stream)."""


@dataclass(slots=True)  # not frozen: the engine's StrictBuild protocol attributes are writable
class AdapterBuild:
    """The engine's ``StrictBuild`` shape plus why a build failed (``page_outcome``)."""

    status: str
    ast: dict[str, Any] | None
    page_outcome: str


def control_event_counts(
    token_ids: Sequence[int], vocabulary: cv.CompositeVocabulary
) -> dict[str, int]:
    """Occurrences of ``slices.CONTROL_EVENT_TOKENS`` in the accepted stream, by name."""
    counts: Counter[str] = Counter()
    for packed in token_ids:
        ref = vocabulary.unpack(int(packed))
        if isinstance(ref, cv.StructuralRef) and ref.local_id in _CONTROL_NAME_BY_ID:
            counts[_CONTROL_NAME_BY_ID[ref.local_id]] += 1
    return dict(sorted(counts.items()))


def gold_node_ids_parallel(gold_ast: Mapping[str, Any], reference: Any) -> tuple[str, ...]:
    """Gold node ids PARALLEL to ``reference.structure``.

    ``DocumentReference.node_ids`` is an unordered set; ``structure`` is the body
    preorder (stories excluded) restricted to ``node_ids``. The AST preorder of
    ``alignment.node_ids_from_ast`` has the same body order and depths, so the ids
    are that preorder filtered by membership. The result is verified against
    ``reference.structure`` entry by entry (length and ``(depth, type)``).
    """
    members = reference.node_ids
    pairs = [
        (nid, entry)
        for nid, entry in zip(node_ids_from_ast(gold_ast), preorder_from_ast(gold_ast), strict=True)
        if nid in members
    ]
    structure = tuple(tuple(e) for e in reference.structure)
    if tuple(entry for _nid, entry in pairs) != structure:
        raise ChainAdapterRefused(
            f"reference.structure ({len(structure)} entries) does not match the gold AST "
            f"body preorder restricted to reference.node_ids ({len(pairs)} entries): "
            "the gold ids cannot be made parallel to it"
        )
    return tuple(nid for nid, _entry in pairs)


def gold_reference_from_chain(
    reference: Any,
    gold_ast: Mapping[str, Any],
    *,
    document_id: str,
    control_event_counts: Mapping[str, int],
    visible_text: str | None = None,
) -> GoldReference:
    """``GoldReference`` for one document, after checking the id parallelism the
    candidate side relies on (``gold_node_ids_parallel``)."""
    gold_node_ids_parallel(gold_ast, reference)
    return gold_from_document_reference(
        reference,
        document_id=document_id,
        control_event_counts=control_event_counts,
        visible_text=visible_text,
        gold_ast=gold_ast,
    )


# --- chain -------------------------------------------------------------------


def _failed(outcome: str) -> AdapterBuild:
    return AdapterBuild("failed", None, outcome)


def _page_result(
    document_bundle_id: str,
    token_ids: Sequence[int],
    typed_predictions: Sequence[Any],
    vocabulary: cv.CompositeVocabulary,
    region_predictions: Sequence[Any] = (),
) -> sealed.PageDecodeResult:
    """Sealed-shape page over the accepted stream (node states replayed, no anchors).

    ``region_predictions`` are ``sealed.SealedRegionPrediction`` records (what the
    builder chain's resource recovery reads); empty abstains on every image."""
    tracker = NodeLifecycleTracker()
    for index, packed in enumerate(token_ids):
        ref = vocabulary.unpack(int(packed))
        if isinstance(ref, cv.StructuralRef):
            tracker.apply_structural_token(_TOKEN_NAME_BY_ID[ref.local_id], event_index=index)
    states = tuple(
        sealed.NodeStateProjection(
            local_node_id=s.local_node_id,
            slot_kind=s.slot_kind,
            node_type=s.node_type,
            parent_local_node_id=s.parent_local_node_id,
            open_event_index=s.open_event_index,
            leaf_event_index=s.leaf_event_index,
            resume_event_index=s.resume_event_index,
            close_event_index=s.close_event_index,
        )
        for s in tracker.states
    )
    return sealed.PageDecodeResult(
        document_bundle_id=document_bundle_id,
        outcome="terminated",
        accepted_tokens=tuple(int(t) for t in token_ids),
        node_states=states,
        typed_predictions=tuple(typed_prediction_projection(p) for p in typed_predictions),
        region_predictions=tuple(region_predictions),
    )


def sealed_regions(hypotheses: Sequence[Any]) -> tuple[Any, ...]:
    """``region_decode.RegionHypothesis`` tuple -> sealed region predictions."""
    from unflatten.model.prediction_ir import region_prediction_projection

    return sealed._sealed_region_predictions([region_prediction_projection(h) for h in hypotheses])


def _render_state(row: Any, render_dir: Path | None) -> str:
    if render_dir is None:
        return "not_run"
    status = row.envelope.renderer.status
    if status in _RENDERED:
        return "rendered"
    return "failed" if status == "failed" else "not_run"


def _resource_violations(row: Any) -> tuple[str, ...]:
    report = row.envelope.ast.builder_report
    codes = (r.get("code") for r in report.get("unresolved", ()))
    return tuple(sorted(c for c in codes if isinstance(c, str) and c.startswith("RESOURCE_")))


@dataclass(frozen=True, slots=True)
class _Outcome:
    build: AdapterBuild
    render: str = "not_run"
    violations: tuple[str, ...] = ()
    row: Any = None


def _run_chain(
    trajectory: RolloutTrajectory,
    typed_predictions: Sequence[Any],
    *,
    document_bundle_id: str,
    chain_flags: Mapping[str, bool],
    gold_profile: str,
    vocabulary: cv.CompositeVocabulary,
    tokenizer: Any,
    raster_source: Any,
    render_dir: Path | None,
    region_predictions: Sequence[Any] = (),
) -> _Outcome:
    if trajectory.status != STATUS_TERMINATED:
        return _Outcome(_failed(trajectory.status))
    flag_kwargs: dict[str, Any] = dict(chain_flags)
    try:
        page = _page_result(
            document_bundle_id, trajectory.token_ids, typed_predictions, vocabulary,
            region_predictions,
        )
        row = run.process_model_page(
            page,
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            raster_source=raster_source,
            render_dir=render_dir,
            table_geometry_column_source=TABLE_GEOMETRY_COLUMN_SOURCE,
            **flag_kwargs,
        )
    except _PAGE_FAILURES as exc:
        return _Outcome(_failed(type(exc).__name__))
    violations = (row.page_outcome,) if row.page_outcome in _INTEGRITY_OUTCOMES else ()
    if row.envelope is None:
        return _Outcome(_failed(row.page_outcome), violations=violations)
    if row.envelope.ast.status != "strict_valid" or row.envelope.ast.canonical_ast is None:
        return _Outcome(_failed("AstBuildFailed"))
    if _PROFILE_RANK[row.profile or run.MODEL_PROFILE] > _PROFILE_RANK[gold_profile]:
        return _Outcome(_failed("ProfileExceedsGold"))
    build = AdapterBuild("built", row.envelope.ast.canonical_ast, "terminated")
    return _Outcome(build, _render_state(row, render_dir), _resource_violations(row), row)


# --- typed from the built AST -------------------------------------------------

_MISSING: Final = object()


def _pointer_universe(typed_predictions: Sequence[Any]) -> dict[str, frozenset[str]]:
    by_type: dict[str, set[str]] = {}
    for p in typed_predictions:
        by_type.setdefault(p.node_type, set()).add(p.pointer)
    return {t: frozenset(ptrs) for t, ptrs in by_type.items()}


def _resolve(node: Mapping[str, Any], pointer: str) -> Any:
    """Value at ``pointer`` in a canonical AST node; a list is its cardinality
    (the container-count pointers); ``_MISSING`` when absent or an object."""
    value: Any = node
    for part in pointer.strip("/").split("/"):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return _MISSING
    if isinstance(value, list):
        return len(value)
    return _MISSING if isinstance(value, Mapping) else value


def _nodes_in_preorder(ast: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    wanted = set(node_ids_from_ast(ast))

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if value.get("id") in wanted and "type" in value:
                out.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(ast)
    by_id = {n["id"]: n for n in out}
    return [by_id[i] for i in node_ids_from_ast(ast)]


def _typed_by_ast_id(
    ast: Mapping[str, Any], universe: Mapping[str, frozenset[str]]
) -> dict[str, dict[str, Any]]:
    typed: dict[str, dict[str, Any]] = {}
    for node in _nodes_in_preorder(ast):
        props = {}
        for pointer in sorted(universe.get(node["type"], ())):
            value = _resolve(node, pointer)
            if value is not _MISSING:
                props[pointer] = value
        if props:
            typed[node["id"]] = props
    return typed


def _typed_on_gold_ids(
    ast: Mapping[str, Any],
    typed_predictions: Sequence[Any],
    gold: GoldReference,
    gold_node_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    candidate_ids = node_ids_from_ast(ast)
    alignment = align_preorder(gold.structure, preorder_from_ast(ast))
    id_map = map_candidate_node_ids(alignment, candidate_ids, gold_node_ids)
    return rekey_typed(_typed_by_ast_id(ast, _pointer_universe(typed_predictions)), id_map)


def _stream_text(
    token_ids: Sequence[int], vocabulary: cv.CompositeVocabulary, tokenizer: Any
) -> str:
    from unflatten.eval.s2_gate import score

    return score.assemble_hypothesis_text(token_ids, vocabulary=vocabulary, tokenizer=tokenizer)


# --- public ------------------------------------------------------------------


def candidate_inputs_from_trajectory(
    trajectory: RolloutTrajectory,
    typed_predictions: Sequence[Any],
    *,
    candidate_id: str,
    document_bundle_id: str,
    chain_flags: Mapping[str, bool],
    gold: GoldReference,
    gold_node_ids: Sequence[str],
    gold_profile: str,
    vocabulary: cv.CompositeVocabulary,
    tokenizer: Any,
    raster_source: Any = None,
    render_dir: Path | None = None,
    region_predictions: Sequence[Any] = (),
) -> CandidateInputs:
    """Run the chain over ``trajectory`` and return the engine's ``CandidateInputs``.

    ``typed_predictions`` are the ``TypedPrediction`` objects of the A2 walk (the
    ``TypedWalk.predictions``). ``chain_flags`` must be exactly the reward policy's
    pinned set (``score_candidate`` raises ``ChainPolicyMismatch`` otherwise) and
    are passed to the chain verbatim. ``gold_node_ids`` must be parallel to
    ``gold.structure`` (``gold_node_ids_parallel``). ``render_dir`` opts in to the
    deterministic RFC002 compile; ``None`` leaves ``rfc002_render="not_run"``.
    ``region_predictions`` (sealed shape, argmax of the node visual aligner, no gradient)
    let the chain bind image resources exactly as a free-running decode does.
    """
    if set(chain_flags) != set(CHAIN_FLAGS):
        raise ChainAdapterRefused(
            f"chain_flags keys must be exactly {CHAIN_FLAGS}, got {sorted(chain_flags)}"
        )
    if len(gold_node_ids) != len(gold.structure):
        raise ChainAdapterRefused(
            f"gold_node_ids has {len(gold_node_ids)} entries but gold.structure has "
            f"{len(gold.structure)}: they must be parallel"
        )
    if gold_profile not in _PROFILE_RANK:
        raise ChainAdapterRefused(
            f"gold profile {gold_profile!r} is not one of {sorted(_PROFILE_RANK)}"
        )
    try:
        outcome = _run_chain(
            trajectory,
            typed_predictions,
            document_bundle_id=document_bundle_id,
            chain_flags=chain_flags,
            gold_profile=gold_profile,
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            raster_source=raster_source,
            render_dir=render_dir,
            region_predictions=region_predictions,
        )
        # Score the original stream even when a chain policy removed a story.
        # Its text must still satisfy the same strict codec, without repair.
        visible_text = (
            _stream_text(trajectory.token_ids, vocabulary, tokenizer)
            if outcome.build.ast is not None else None
        )
    except _TEXT_FAILURES as exc:
        outcome = _Outcome(_failed(type(exc).__name__))
        visible_text = None
    ast = outcome.build.ast
    typed = (
        _typed_on_gold_ids(ast, typed_predictions, gold, gold_node_ids) if ast is not None else {}
    )
    build = outcome.build
    def build_candidate() -> AdapterBuild:
        return build

    return CandidateInputs(
        candidate_id=candidate_id,
        chain_flags=dict(chain_flags),
        build_candidate=build_candidate,
        control_event_counts=control_event_counts(trajectory.token_ids, vocabulary),
        rfc002_render=outcome.render,
        visible_text=visible_text,
        typed=typed,
        fragment_violations=outcome.violations,
        continuity_micro=None,
    )

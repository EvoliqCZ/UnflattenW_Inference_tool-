"""``RealRunPorts.build_gold``: the scorer-side gold of one page ``(bundle, k)`` (DP5).

Pipeline (the candidate takes the SAME chain, so both sides share an id space and
one projection)::

    cache row tokens + typed targets -> chain (same CHAIN_FLAGS, PageRasterAdapter)
      -> gold AST -> reference_from_ast -> gold_reference_from_chain -> PageGold

Authorities, and where each comes from (never from the chain AST unless stated):

* ``visible_text`` -- ``s2_gate.score.assemble_reference_text`` over the row (RFC003
  text, stories included). The candidate is measured symmetrically from its own token
  stream (``chain_adapter.candidate_inputs_from_trajectory``).
* ``typed`` / ``typed_observable`` -- the row's ``typed_targets`` / ``typed_cardinalities``
  under the RFC003 known / observable / supervision masks. The chain AST is only
  the id space: ``headless_defaults`` would otherwise turn owner defaults into
  "observed gold" (a missing label is never a value).
* structure, node ids, grid -- the chain's gold AST. The chain is fed every KNOWN target,
  so the gold grid uses known cell spans even when unsupervised (DOCX truth); the number
  of such spans is counted per page as ``gold_span_unsupervised``.
* ``source`` / ``cell`` -- the injected maps, never the row.

A page whose gold chain does not end in a strict-valid AST is REFUSED
(:class:`GoldChainFailed`, counted ``gold_chain_failed``), never scored against an
empty gold. Torch-free only at import time: heavy imports are inside functions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from unflatten.model.rl.adapters import gold_typed
from unflatten.model.rl.adapters.gold_errors import (
    GoldChainFailed,
    GoldPageRefused,
    GoldReferenceRefused,
    GoldRefused,
    GoldTypedMappingFailed,
)
from unflatten.model.rl.data import RlPage

if TYPE_CHECKING:
    from unflatten.eval.assembly.reference import DocumentReference
    from unflatten.model.rl.real_run_ports import PageGold

__all__ = [
    "GoldBuilder",
    "GoldChainFailed",
    "GoldPageRefused",
    "GoldRefused",
    "GoldReferenceRefused",
    "GoldScreen",
    "GoldTypedMappingFailed",
    "make_build_gold",
    "reference_from_ast",
    "s4_cells_by_source",
    "screen_pages",
]

_STORY_KINDS: Final = ("header", "footer")  # the story kinds ``reference.stories`` lists


# --- reference_from_ast ----------------------------------------------------------


def _collect_ids(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        if "id" in value and "type" in value:
            out.append(value["id"])
        for child in value.values():
            _collect_ids(child, out)
    elif isinstance(value, list):
        for child in value:
            _collect_ids(child, out)


def _nodes(value: Any, out: dict[str, Mapping[str, Any]]) -> None:
    if isinstance(value, dict):
        if "id" in value and "type" in value:
            out[value["id"]] = value
        for child in value.values():
            _nodes(child, out)
    elif isinstance(value, list):
        for child in value:
            _nodes(child, out)


def _story_triples(nodes: Mapping[str, Mapping[str, Any]]) -> tuple[tuple[int, str, str], ...]:
    """``(page, node id, kind)`` of the header/footer nodes the page carries, as
    ``eval.assembly.reference._stories`` lists them; a standalone page is page 0."""
    out: list[tuple[int, str, str]] = []
    for nid, node in nodes.items():
        if node["type"] in _STORY_KINDS:
            kind = node.get("kind")
            if not isinstance(kind, str):
                raise GoldReferenceRefused(f"{node['type']} {nid!r} has no string 'kind'")
            out.append((0, nid, kind))
    return tuple(out)


def reference_from_ast(
    ast: Mapping[str, Any],
    typed: Mapping[str, Mapping[str, Any]],
    *,
    dropped: frozenset[str] = frozenset(),
) -> DocumentReference:
    """A real ``DocumentReference`` over the gold ``ast`` of ONE standalone page.

    ``node_ids`` keeps present story nodes (as the real reference does) but
    ``structure`` / ``text`` are body-only; ``stories`` lists the header/footer
    nodes of the AST at page 0. A standalone page has no seam, so ``seams`` is
    empty by construction. ``excluded`` / ``attribution`` are document-level
    counters over per-page sidecars this path does not have: they are left EMPTY
    (not derived), never zero-filled. Raises :class:`GoldReferenceRefused` for a
    duplicate node id.
    """
    from collections import Counter as _Counter

    from unflatten.eval.assembly.reference import DocumentReference
    from unflatten.model.rl.alignment import node_ids_from_ast
    from unflatten.model.rl.reward_inputs import preorder_from_ast

    ids: list[str] = []
    _collect_ids(ast, ids)
    if len(set(ids)) != len(ids):
        dup = sorted(i for i, n in _Counter(ids).items() if n > 1)
        raise GoldReferenceRefused(f"gold AST has duplicate node ids {dup[:5]}")
    nodes: dict[str, Mapping[str, Any]] = {}
    _nodes(ast, nodes)
    body = list(zip(node_ids_from_ast(ast), preorder_from_ast(ast), strict=True))
    kept = [(nid, entry) for nid, entry in body if nid not in dropped]
    text = {nid: nodes[nid].get("text") or "" for nid, (_d, t) in kept if t == "run"}
    return DocumentReference(
        node_ids=frozenset(ids) - dropped,
        structure=tuple(entry for _nid, entry in kept),
        text=text,
        typed={n: dict(p) for n, p in typed.items() if n not in dropped},
        stories=_story_triples(nodes),
        seams=(),
        excluded=Counter(),
        attribution=Counter(),
    )


# --- the port --------------------------------------------------------------------


def s4_cells_by_source() -> dict[str, tuple[str, str, str, str]]:
    """``{cohort: (split, pyramid_level, task_kind, cohort)}`` of the S4 matrix cells."""
    from unflatten.model.s4_plan import REAL_CELL, SYNTHETIC_CELL

    keys = ("split", "pyramid_level", "task_kind", "cohort")
    return {c["cohort"]: tuple(c[k] for k in keys) for c in (REAL_CELL, SYNTHETIC_CELL)}  # type: ignore[misc]


@dataclass
class GoldBuilder:
    """``Callable[[RlPage], PageGold]`` plus the counters a manifest builder reads.

    ``counters`` carries ``built``, one entry per refusal kind
    (``gold_chain_failed``, ...) and ``typed_outside_body`` (supervised typed
    targets on story / frame nodes, outside the scored body) and ``gold_span_unsupervised``
    (known, unsupervised cell spans that shaped the gold grid).
    """

    store_dir: Path
    reader: Any
    index_of: Mapping[tuple[str, int], int]
    vocabulary: Any
    tokenizer: Any
    chain_flags: Mapping[str, bool]
    bundle_source: Mapping[str, str]
    cells: Mapping[str, Sequence[str]]
    fetch: Callable[[Path, Mapping[str, Any]], bytes] | None = None
    counters: Counter[str] = field(default_factory=Counter)

    def __call__(self, page: RlPage) -> PageGold:
        try:
            out = self._build(page)
        except GoldRefused as exc:
            self.counters[exc.counter] += 1
            raise
        self.counters["built"] += 1
        return out

    def _row(self, key: tuple[str, int]) -> Any:
        index = self.index_of.get(key)
        if index is None:
            raise GoldPageRefused(f"page {key!r} is not in the pinned expectation target keys")
        row = self.reader[index]
        if (row.document_bundle_id, row.page_index) != key:
            raise GoldPageRefused(
                f"cache index {index} is page {(row.document_bundle_id, row.page_index)!r}, "
                f"the page asked for is {key!r}"
            )
        if row.page_image is None:
            raise GoldPageRefused(f"page {key!r}: a feature row has no page image to rasterise")
        return row

    def _build(self, page: RlPage) -> PageGold:
        from unflatten.eval.s2_gate import score
        from unflatten.model.rl import chain_adapter as ca
        from unflatten.model.rl.adapters.visual import PageRasterAdapter
        from unflatten.model.rl.real_run_ports import PageGold

        key = (page.page_key[0], int(page.page_key[1]))
        row = self._row(key)
        source = self.bundle_source.get(key[0])
        if source is None:
            raise GoldPageRefused(f"bundle {key[0]!r} has no source in the bundle_source map")
        cell = self.cells.get(source)
        if cell is None:
            raise GoldPageRefused(f"source {source!r} has no matrix cell")
        token_ids = [int(t) for t in row.packed_token_ids.tolist()]
        initial = ca._page_result(key[0], token_ids, (), self.vocabulary)
        by_position: dict[int, Any] = {}
        for state in initial.node_states or ():
            at = gold_typed.position(state)
            if at is not None:
                by_position[at] = state
        predictions = gold_typed.predictions(row, by_position)
        spans = gold_typed.unsupervised_spans(row, by_position)
        if spans:  # absent key = none, so existing counter reports are unchanged
            self.counters["gold_span_unsupervised"] += spans
        raster = PageRasterAdapter(
            key[0],
            key[1],
            row.page_image,
            store_dir=self.store_dir,
            expectation_keys=self.index_of,
            fetch=self.fetch,
        )
        regions = gold_typed.region_predictions(row, by_position)
        ast = self._gold_ast(key[0], token_ids, predictions, raster, regions)
        placement = gold_typed.placement(initial, ast)
        typed = gold_typed.typed_authority(row, placement, self.counters)
        reference = reference_from_ast(ast, typed)
        text = score.assemble_reference_text(
            row, vocabulary=self.vocabulary, tokenizer=self.tokenizer
        )
        try:
            gold = ca.gold_reference_from_chain(
                reference,
                ast,
                document_id=f"{key[0]}/pages/{key[1]}",
                control_event_counts=ca.control_event_counts(token_ids, self.vocabulary),
                visible_text=text,
            )
            node_ids = ca.gold_node_ids_parallel(ast, reference)
        except ca.ChainAdapterRefused as exc:
            raise GoldReferenceRefused(str(exc)) from exc
        return PageGold(
            gold=gold,
            gold_node_ids=node_ids,
            gold_profile=ast["profile"],
            document_bundle_id=key[0],
            source=source,
            cell=tuple(cell),  # type: ignore[arg-type]
        )

    def _gold_ast(
        self,
        bundle: str,
        token_ids: list[int],
        predictions: list[Any],
        raster: Any,
        regions: Sequence[Any] = (),
    ) -> dict[str, Any]:
        from unflatten.eval.builder_chain import run
        from unflatten.model.builder_report import BuilderReportInvalid
        from unflatten.model.rl import chain_adapter as ca

        flag_kwargs: dict[str, Any] = dict(self.chain_flags)
        try:
            page = ca._page_result(bundle, token_ids, predictions, self.vocabulary, regions)
            chain_row = run.process_model_page(
                page,
                vocabulary=self.vocabulary,
                tokenizer=self.tokenizer,
                raster_source=raster,
                render_dir=None,
                table_geometry_column_source=ca.TABLE_GEOMETRY_COLUMN_SOURCE,
                **flag_kwargs,
            )
        except (*ca._PAGE_FAILURES, BuilderReportInvalid) as exc:
            raise GoldChainFailed(f"{bundle}: {type(exc).__name__}: {exc}") from exc
        envelope = chain_row.envelope
        if envelope is None:
            raise GoldChainFailed(f"{bundle}: no envelope (page_outcome {chain_row.page_outcome})")
        ast = envelope.ast
        if ast.status != "strict_valid" or ast.canonical_ast is None:
            codes = sorted(
                {
                    str(u.get("code"))
                    for u in (ast.builder_report or {}).get("unresolved", ())
                    if isinstance(u, Mapping)
                }
            )
            raise GoldChainFailed(
                f"{bundle}: gold AST status {ast.status!r}, not strict_valid (unresolved {codes})"
            )
        return ast.canonical_ast


def make_build_gold(
    *,
    store_dir: Path | str,
    reader: Any,
    expectation: Any,
    vocabulary: Any,
    tokenizer: Any,
    chain_flags: Mapping[str, bool],
    bundle_source: Mapping[str, str],
    cells: Mapping[str, Sequence[str]],
    fetch: Callable[[Path, Mapping[str, Any]], bytes] | None = None,
) -> GoldBuilder:
    """``RealRunPorts.build_gold`` over a 7.0.0 page-image cache.

    ``reader[i]`` addresses ``expectation.target_keys()[i]`` (not the physical
    order). ``chain_flags`` must be exactly the reward policy's pinned set (the
    candidate runs the same). ``bundle_source`` is ``{bundle: cohort}`` from the
    store index and ``cells`` ``{cohort: matrix cell}`` (``s4_cells_by_source``);
    neither is ever read from the row. ``fetch`` is the test seam of
    :class:`PageRasterAdapter`.
    """
    from unflatten.model.rl.policies import CHAIN_FLAGS

    if set(chain_flags) != set(CHAIN_FLAGS):
        raise GoldRefused(
            f"chain_flags keys must be exactly {CHAIN_FLAGS}, got {sorted(chain_flags)}"
        )
    keys = tuple(expectation.target_keys())
    return GoldBuilder(
        store_dir=Path(store_dir),
        reader=reader,
        index_of={k: i for i, k in enumerate(keys)},
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        chain_flags=dict(chain_flags),
        bundle_source=bundle_source,
        cells=cells,
        fetch=fetch,
    )


# --- screening (for the manifest builder) ----------------------------------------


@dataclass(frozen=True)
class GoldScreen:
    """Pages that build gold, and the named refusal of each page that does not."""

    passed: tuple[tuple[str, int], ...]
    refused: Mapping[tuple[str, int], tuple[str, str]]  # key -> (counter name, message)

    def summary(self) -> dict[str, Any]:
        by_kind = Counter(kind for kind, _msg in self.refused.values())
        return {"passed": len(self.passed), "refused": len(self.refused), "by_kind": dict(by_kind)}


def screen_pages(build_gold: Callable[[RlPage], PageGold], pages: Iterable[RlPage]) -> GoldScreen:
    """Run ``build_gold`` over ``pages``; a :class:`GoldRefused` page is recorded by
    name (``gold_chain_failed`` ...), any other exception is a defect and propagates."""
    passed: list[tuple[str, int]] = []
    refused: dict[tuple[str, int], tuple[str, str]] = {}
    for page in pages:
        key = (page.page_key[0], int(page.page_key[1]))
        try:
            build_gold(page)
        except GoldRefused as exc:
            refused[key] = (exc.counter, str(exc))
        else:
            passed.append(key)
    return GoldScreen(tuple(passed), refused)

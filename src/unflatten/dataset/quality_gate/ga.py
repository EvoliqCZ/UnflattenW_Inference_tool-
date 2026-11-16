"""G-A of the DEC-057 data-quality gate: detectors over EVERY page of the
train and val caches.

Each detector reads one page's gold token stream (the audit's
`page_metrics`, `token_tree.py`) plus the cache build's own skip rows,
and yields one integer. A page is *flagged* when that value exceeds the
policy's `page_max`; a split *violates* the detector when more flagged
pages remain (after register waivers, `gc.py`) than the policy's
`max_flagged_pages`. There is no warning level: the gate either passes or
exits non-zero with the named pages (DEC-057 point 2).

`manifest_checks` wraps the checks that already existed on the cache
manifest (cohort size, skip-reason shares incl. the token-scan invariant
refusal `skipped_projection_invariant`, the text-coverage floor) under the
same policy file, so one gate result carries all of them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from unflatten.dataset.quality_gate.token_tree import (
    ID,
    QWEN_START,
    Node,
    defect_flags,
    page_metrics,
    parse_tree,
    walk,
)

TEXT_FRAME_OPEN = ID["TEXT_FRAME_OPEN"]
# These are compiler refusals, not detector applicability false positives.
NON_WAIVABLE_DETECTORS = frozenset(
    {"unobservable_images", "unsupported_list_continuations", "unpaired_notes",
     "table_grid_invalid"}
)

#: detector id -> (version, page-metric field). Bump a version whenever a
#: detector's value changes meaning: the version lands in the gate result.
DETECTORS: dict[str, tuple[str, str]] = {
    # audit §3 signatures (`token_tree.defect_flags`)
    "spill": ("1", "spill"),
    "skeleton_rows": ("1", "skeleton_rows"),
    # textless tables and cardinality
    "textless_table_cells": ("1", "textless_table_cells_max"),
    "cells_per_row": ("1", "cells_per_row_max"),
    "rows_per_table": ("1", "rows_per_table_max"),
    "empty_cell_streak": ("1", "empty_cell_streak_max"),
    "empty_row_streak": ("1", "empty_row_streak_max"),
    # withdrawn runs (DEC-055 V1a / V1b)
    "empty_run_chain": ("1", "empty_run_streak_max"),
    "lone_withdrawn_runs": ("1", "empty_runs_sandwiched"),
    # empty text frames (audit I2)
    "empty_text_frames": ("1", "empty_text_frames"),
    # repeated subtrees
    "repeat_textless": ("1", "repeat_textless_max"),
    "repeat_textful": ("1", "repeat_textful_max"),
    # DEC-033 projection invariants (empty paragraphs / TAB runs capped at 3)
    "empty_paragraph_streak": ("1", "empty_par_any_streak_max"),
    "tab_run": ("1", "tab_run_max"),
    # G-C trial classes (docs/runs/target-defects-root-cause-2026-09-30.md)
    "noteref_textless_paragraphs": ("1", "noteref_textless_paragraphs"),
    "footnote_only_page": ("1", "footnote_only_page"),
    "hf_withdrawn_runs": ("1", "hf_withdrawn_runs"),
    # [section] (docs/runs/target-defects-iteration2-2026-09-30.md)
    "bodyless_sections": ("1", "bodyless_sections"),
    # Standalone strict builder requires every note/ref paired on this page.
    "unpaired_notes": ("1", "unpaired_notes"),
    "unobservable_images": ("1", "unobservable_images"),
    "unsupported_list_continuations": ("1", "unsupported_list_continuations"),
    # Standalone strict builder refuses TABLE_GRID_GAP/OVERLAP/OUT_OF_RANGE
    # (GC i6: continuation rows whose leading cells belong to the prior page).
    "table_grid_invalid": ("1", "table_grid_invalid"),
}

#: story roots: a paragraph under one of them is not body text
_STORY_OPEN = frozenset(
    ID[k]
    for k in ("HEADER_OPEN", "FOOTER_OPEN", "FOOTNOTE_OPEN", "ENDNOTE_OPEN", "TEXT_FRAME_OPEN")
)
_HF_OPEN = frozenset((ID["HEADER_OPEN"], ID["FOOTER_OPEN"]))
_NOTE_OPEN = frozenset((ID["FOOTNOTE_OPEN"], ID["ENDNOTE_OPEN"]))
_PO, _RO, _NOTE_REF, _SECTION_OPEN = (
    ID["PARAGRAPH_OPEN"], ID["RUN_OPEN"], ID["NOTE_REF"], ID["SECTION_OPEN"],
)  # fmt: skip
_EMPTY_RUN = (ID["RUN_OPEN"], ID["TEXT_BEGIN"], ID["TEXT_END"], ID["RUN_CLOSE"])


def empty_text_frames(tokens: list[int]) -> int:
    """Textless `TEXT_FRAME` nodes on the page (audit §3 #30, I2): a frame
    whose drawing the import dropped renders nothing, yet the target still
    emits its wrapper."""
    return sum(1 for n in walk(parse_tree(tokens)) if n.kind == TEXT_FRAME_OPEN and n.textless)


def story_defects(tokens: list[int]) -> dict[str, int]:
    """The G-C trial's token-visible classes on one page:

    * `noteref_textless_paragraphs` -- body paragraphs (outside every
      header/footer/note/frame story) that hold a `NOTE_REF` and not one
      text token: the aligner matched none of their words, so every run
      was withdrawn and only the reference survived ([noteref]);
    * `footnote_only_page` -- 1 when the page carries a note story but no
      section: every body paragraph lost its observation, only the notes
      were placed ([noteref], `ORPHAN_FOOTNOTE`);
    * `hf_withdrawn_runs` -- withdrawn run wrappers (`RUN_OPEN TEXT_BEGIN
      TEXT_END RUN_CLOSE`) inside a header or footer ([hftext])."""
    out = {"noteref_textless_paragraphs": 0, "footnote_only_page": 0, "hf_withdrawn_runs": 0}

    def visit(node: Node, in_story: bool, in_hf: bool) -> None:
        in_story = in_story or node.kind in _STORY_OPEN
        in_hf = in_hf or node.kind in _HF_OPEN
        span = tokens[node.start : node.end + 1]
        if node.kind == _PO and not in_story:
            if _NOTE_REF in span and not any(t >= QWEN_START for t in span):
                out["noteref_textless_paragraphs"] += 1
        elif node.kind == _RO and in_hf and tuple(span) == _EMPTY_RUN:
            out["hf_withdrawn_runs"] += 1
        for child in node.children:
            visit(child, in_story, in_hf)

    visit(parse_tree(tokens), False, False)
    present = set(tokens)
    out["footnote_only_page"] = int(bool(present & _NOTE_OPEN) and _SECTION_OPEN not in present)
    return out


def bodyless_sections(tokens: list[int]) -> int:
    """Sections whose children are only headers/footers, on a page with
    more than one section ([section], G-C trial 058): an inherited story
    placed on the heir section's page pulled its own section in with no
    body (DEC-033 §3 wants one standalone section carrying the furniture it
    shows; `v16_target_policy.rehome_inherited_stories`). A page with one
    section that shows only its footer is a closed document, not this."""
    sections = [n for n in walk(parse_tree(tokens)) if n.kind == _SECTION_OPEN]
    if len(sections) < 2:
        return 0
    return sum(1 for s in sections if all(c.kind in _HF_OPEN for c in s.children))


def unobservable_images(tokens: list[int], extra: Mapping[str, Any]) -> int:
    """Image leaves without one visible, supervised region at their actual anchor."""
    images = {ID["INLINE_IMAGE"]: "inline_image", ID["FLOATING_IMAGE"]: "floating_image"}
    by_anchor: dict[int, list[Mapping[str, Any]]] = {}
    for target in extra.get("region_targets") or ():
        by_anchor.setdefault(target["anchor_token_index"], []).append(target)
    invalid = 0
    for anchor, token in enumerate(tokens):
        if token not in images:
            continue
        records = by_anchor.get(anchor, [])
        valid = len(records) == 1
        if valid:
            target = records[0]
            boxes = target.get("regions") or []
            valid = (
                target.get("node_type") == images[token]
                and target.get("page_evidence") == "observed"
                and target.get("observable_mask") is True
                and target.get("supervision_mask") is True
                and len(boxes) == 1
                and _valid_box(boxes[0])
            )
        invalid += int(not valid)
    return invalid


def _valid_box(box: Any) -> bool:
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(type(value) is int and 0 <= value <= 1_000_000 for value in box)
        and box[0] < box[2]
        and box[1] < box[3]
    )


def unsupported_list_continuations(tokens: list[int]) -> int:
    """RFC002 §47: every standalone list item starts with a paragraph."""
    return sum(
        node.kind == ID["LIST_ITEM_OPEN"]
        and (not node.children or node.children[0].kind != ID["PARAGRAPH_OPEN"])
        for node in walk(parse_tree(tokens))
    )


def table_grid_invalid(tokens: list[int], extra: Mapping[str, Any]) -> int:
    """Rows violating the strict grid closure of `semantic_validator.
    _check_table` (gap, overlap or column past the grid), from the typed cell
    targets the gold builder uses. A table without a known `/columns` count is
    not judged; a cell without a known `grid_column` takes the first free one."""
    cell_props: dict[int, dict[str, Any]] = {}
    for target in extra.get("typed_targets") or ():
        if (
            isinstance(target, Mapping)
            and target.get("node_type") == "cell"
            and target.get("known_mask", True)
        ):
            pointer = target.get("node_property_pointer")
            if pointer in ("/grid_column", "/col_span", "/row_span"):
                cell_props.setdefault(target["anchor_token_index"], {})[pointer] = target["value"]
    columns = {
        c["anchor_token_index"]: c["count"]
        for c in extra.get("typed_cardinalities") or ()
        if isinstance(c, Mapping)
        and c.get("container_pointer") == "/columns"
        and c.get("known_mask", True)
    }
    bad = 0
    for table in walk(parse_tree(tokens)):
        if table.kind != ID["TABLE_OPEN"] or table.start not in columns:
            continue
        num_columns = int(columns[table.start])
        active: dict[int, int] = {}
        rows = [r for r in table.children if r.kind == ID["ROW_OPEN"]]
        for ri, row in enumerate(rows):
            occupied = {gc for gc, until in active.items() if until >= ri}
            invalid = False
            for cell in (c for c in row.children if c.kind == ID["CELL_OPEN"]):
                props = cell_props.get(cell.start, {})
                col_span = int(props.get("/col_span", 1))
                row_span = int(props.get("/row_span", 1))
                grid_column = props.get("/grid_column")
                if grid_column is None:
                    grid_column = next(gc for gc in range(num_columns + 1) if gc not in occupied)
                grid_column = int(grid_column)
                if grid_column + col_span > num_columns:
                    invalid = True
                    continue
                for gc in range(grid_column, grid_column + col_span):
                    if gc in occupied:
                        invalid = True
                    occupied.add(gc)
                    active[gc] = max(active.get(gc, -1), ri + row_span - 1)
            if invalid or any(gc not in occupied for gc in range(num_columns)):
                bad += 1
    return bad


def page_values(
    tokens: list[int],
    bundle: str,
    page: int,
    skipped: set[tuple[str, int]],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Every G-A detector's value for one page. `skipped` holds the
    `(bundle, page)` keys the cache build refused (its `page_skip_rows`):
    the `spill` signature reads the neighbours' fate."""
    metrics = dict(page_metrics(tokens))
    metrics["empty_text_frames"] = empty_text_frames(tokens)
    metrics.update(story_defects(tokens))
    metrics["bodyless_sections"] = bodyless_sections(tokens)
    metrics["unobservable_images"] = unobservable_images(tokens, extra or {})
    metrics["unsupported_list_continuations"] = unsupported_list_continuations(tokens)
    metrics["table_grid_invalid"] = table_grid_invalid(tokens, extra or {})
    # This gate admits standalone exact-text cache pages, not multi-page
    # continuations. Match ast_builder.bind_note_refs' equal-count contract;
    # neither missing body references nor orphan note stories may be dropped.
    metrics["unpaired_notes"] = abs(
        tokens.count(_NOTE_REF) - sum(tokens.count(kind) for kind in _NOTE_OPEN)
    )
    metrics["spill"] = int(
        "spill" in defect_flags({"bundle": bundle, "page": page, **metrics}, skipped)
    )
    return {name: int(metrics[field]) for name, (_version, field) in DETECTORS.items()}


def evaluate(
    values: Iterable[tuple[str, Mapping[str, int]]],
    detectors_policy: Mapping[str, Mapping[str, int]],
    waived: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Flag pages per detector and decide each detector for one split.

    `values` yields `(page_id, {detector: value})`; `waived` holds page ids
    the G-C register marks `genuine_keep` (a reviewed, legitimate page).
    A detector missing from the policy is refused -- an unthresholded
    detector would silently never fail."""
    missing = sorted(set(DETECTORS) - set(detectors_policy))
    if missing:
        raise ValueError(f"policy has no G-A threshold for detector(s) {missing}")
    flagged: dict[str, list[tuple[str, int]]] = {name: [] for name in DETECTORS}
    pages = 0
    for page_id, row in values:
        pages += 1
        for name in DETECTORS:
            if row[name] > detectors_policy[name]["page_max"]:
                flagged[name].append((page_id, row[name]))
    out: dict[str, Any] = {"pages": pages, "detectors": {}}
    for name in DETECTORS:
        rule = detectors_policy[name]
        hits = sorted(flagged[name])
        open_hits = [(p, v) for p, v in hits if p not in waived or name in NON_WAIVABLE_DETECTORS]
        out["detectors"][name] = {
            "version": DETECTORS[name][0],
            "page_max": rule["page_max"],
            "max_flagged_pages": rule["max_flagged_pages"],
            "flagged": len(hits),
            "waived": len(hits) - len(open_hits),
            "violation": len(open_hits) > rule["max_flagged_pages"],
            "pages": [{"page": p, "value": v} for p, v in open_hits],
        }
    return out


# -- existing manifest checks --------------------------------------------------


def manifest_checks(manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """The cache manifest's own counters against `policy` (the policy
    file's `manifest` block): cohort floor, per-reason skip-share ceilings
    and the text-coverage floor. Returns `{check: {value, limit, violation}}`;
    a counter the manifest does not carry is a violation, never a pass."""
    skips = Counter(row["reason"] for row in manifest.get("page_skip_rows", ()))
    entries = int(manifest["entries"])
    cohort = entries + sum(skips.values())
    checks: dict[str, Any] = {}

    def put(name: str, value: Any, limit: Any, violation: bool) -> None:
        checks[name] = {"value": value, "limit": limit, "violation": violation}

    put("min_pages", entries, policy["min_pages"], entries < policy["min_pages"])
    for reason, ceiling in sorted(policy["max_skip_share_milli"].items()):
        share = (1000 * skips.get(reason, 0)) // max(cohort, 1)
        put(f"skip_share_milli:{reason}", share, ceiling, share > ceiling)
    # ratios as integer milli (the result is canonical JSON: no floats)
    floor = _milli(manifest.get("min_text_coverage"))
    limit = policy["min_text_coverage_milli"]
    put("min_text_coverage_milli", floor, limit, floor is None or floor < limit)
    median = _milli((manifest.get("text_coverage") or {}).get("median_ratio"))
    limit = policy["min_text_coverage_median_milli"]
    put("text_coverage_median_milli", median, limit, median is None or median < limit)
    return checks


def _milli(ratio: float | None) -> int | None:
    return None if ratio is None else int(1000 * ratio)

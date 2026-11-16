"""Opt-in chain policy `tab-stops-from-regions.v0` (RATIFIED 2026-10-16, DEC-077;
`docs/rfc/RFC004-ERRATA-011.md` E3).

Where gold aligns columns with runs of spaces, a borderless table or tab
stops, the model writes `text TAB..TAB value TAB..TAB value` and predicts
`/layout/tabs = 0` (or one stop) for the paragraph. Without stops the renderer
puts every TAB on Word's default stops, so the values land in different
columns row by row -- the misaligned score table of the S4 v16 cohort page
`2871e204` (123 TABs where gold has none). Measured 2026-10-16 on the 40-page
selection cohort, checkpoint 178 975: 68 TAB paragraphs on 10 pages, 210 TABs
in 79 groups (2.7 TABs per group); 63 paragraphs with predicted stops across
the cohort, 10 of them with more TAB groups than stops.

The model does predict a page region for every run (`SealedRegionPrediction`),
and the left edge of the run that follows a TAB group is exactly where the
tab stop has to be. This policy derives the paragraph's `/layout/tabs` from
those regions BEFORE assembly (after `page-geometry-from-raster.v0`, so a
corrected section geometry is what it reads), edits only `typed_predictions`
and counts every outcome:

- scope: a `paragraph` whose descendants (through `hyperlink` and the like)
  contain at least one `tab` leaf followed by a `run`; a paragraph inside a
  table `cell` is skipped (its stops would be cell-relative), as is one whose
  `/layout/alignment` is not left/justify (tabs behave differently there);
- stop position: `run.bbox_ppm.x0 / 1e6 * page_width_mpt - left_margin_mpt`
  of the paragraph's section, from that section's own typed predictions
  (`/page/width_mpt`, `/margins/left_mpt`) -- the same geometry the renderer
  lays the paragraph out with; a section without them is a counted refusal;
- a group of m TABs gets m stops `pos - (m-1)*STOP_STEP_MPT .. pos`, kept
  strictly increasing, so every TAB of the group advances to the same column
  and the tokens stay untouched; a trailing TAB group (no run after it) is
  ignored and counted;
- any run after a TAB group without a `region` prediction leaves the whole
  paragraph as predicted (counted): half-derived stops would misplace the rest;
- the derived stops REPLACE the model's predicted `/layout/tabs*` (counted
  separately when there were any) -- a predicted single stop under two TAB
  groups is the common failure; stops already equal to the derived ones are
  `paragraph_left_as_predicted` and the page object is returned unchanged.

Offline prototype (same day, 3 cohort pages rendered): the two score columns
of `2871e204` return to the source positions, the form labels of `0f60662d`
too; what remains is the +-0.5-1 cm row-to-row jitter of the run regions
themselves and a run wider than the remaining line wrapping under its stop.
TAB tokens, node types and all other pointers are never changed.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Any

from unflatten.eval.s2_gate import sealed

POLICY_ID = "tab-stops-from-regions.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-011.md"
COUNT_PREFIX = "tab_stops:"

PPM = 1_000_000
TABS_POINTER = "/layout/tabs"
ALIGNMENT_POINTER = "/layout/alignment"
WIDTH_POINTER = "/page/width_mpt"
LEFT_MARGIN_POINTER = "/margins/left_mpt"
#: Paragraph alignments under which a left tab stop means "this column".
ALIGNMENTS = ("left", "justify")
#: 1.5 pt between the substitute stops of one TAB group (prototype 2026-10-16):
#: 2.7 TABs per group on the cohort keep the first substitute within ~4 pt of
#: the target, still right of where the preceding text ends.
STOP_STEP_MPT = 1_500
STOP_ALIGNMENT = "left"
STOP_LEADER = "none"
RULES = (
    "stop_at_following_run_left_edge.v0",
    "section_geometry_from_typed_predictions.v0",
    "one_stop_per_tab_step_before_target.v0",
    "predicted_stops_replaced.v0",
    "cell_and_non_left_paragraphs_skipped.v0",
    "tokens_untouched.v0",
)


def policy_record() -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "alignments": list(ALIGNMENTS),
        "stop_step_mpt": STOP_STEP_MPT,
        "stop_alignment": STOP_ALIGNMENT,
        "stop_leader": STOP_LEADER,
        "scope": {"node_type": "paragraph", "requires": "tab leaf followed by a run with a region"},
        "rules": list(RULES),
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _ancestor(node_id: str, node_type: str, states: dict[str, Any]) -> str | None:
    """The nearest ancestor of `node_type` (the node itself excluded), or None."""
    seen: set[str] = set()
    current = states[node_id].parent_local_node_id
    while current is not None and current not in seen:
        seen.add(current)
        state = states.get(current)
        if state is None:
            return None
        if state.node_type == node_type:
            return current
        current = state.parent_local_node_id
    return None


def _event_index(state: Any) -> int | None:
    return state.open_event_index if state.slot_kind == "opened_node" else state.leaf_event_index


def _paragraph_items(states: dict[str, Any]) -> dict[str, list[tuple[int, str, str]]]:
    """paragraph id -> [(event_index, 'tab' | 'run', node id)] in token order,
    for every tab leaf and run under that paragraph (through hyperlinks)."""
    items: dict[str, list[tuple[int, str, str]]] = {}
    for node_id, state in states.items():
        if state.node_type not in ("tab", "run"):
            continue
        index = _event_index(state)
        paragraph = _ancestor(node_id, "paragraph", states)
        if index is None or paragraph is None:
            continue
        items.setdefault(paragraph, []).append((index, state.node_type, node_id))
    for sequence in items.values():
        sequence.sort()
    return items


def _groups(sequence: list[tuple[int, str, str]]) -> tuple[list[tuple[int, str]], int]:
    """`([(tab_count, following_run_id)], trailing_groups)` for one paragraph."""
    groups: list[tuple[int, str]] = []
    trailing = 0
    j = 0
    while j < len(sequence):
        if sequence[j][1] != "tab":
            j += 1
            continue
        k = j
        while k < len(sequence) and sequence[k][1] == "tab":
            k += 1
        if k < len(sequence):
            groups.append((k - j, sequence[k][2]))
        else:
            trailing += 1
        j = k
    return groups, trailing


def _substitute_stops(stops: list[tuple[int, int]]) -> list[int]:
    """`[(pos, m)] -> m stops per group, step STOP_STEP_MPT, strictly increasing."""
    flat: list[int] = []
    previous = 0
    for pos, m in stops:
        for i in range(m):
            p = max(pos - (m - 1 - i) * STOP_STEP_MPT, previous + STOP_STEP_MPT)
            flat.append(p)
            previous = p
    return flat


def _existing_stops(predictions: dict[str, Any]) -> list[int] | None:
    """The paragraph's predicted stops as positions, or None when they are not
    exactly this policy's shape (count, left alignment, no leader)."""
    count = predictions.get(TABS_POINTER)
    if not _is_int(count):
        return None
    positions: list[int] = []
    for i in range(count):
        pos = predictions.get(f"{TABS_POINTER}/{i}/position_mpt")
        alignment = predictions.get(f"{TABS_POINTER}/{i}/alignment")
        leader = predictions.get(f"{TABS_POINTER}/{i}/leader")
        if not _is_int(pos) or alignment != STOP_ALIGNMENT or leader != STOP_LEADER:
            return None
        positions.append(pos)
    return positions


def apply(page: sealed.PageDecodeResult) -> tuple[sealed.PageDecodeResult, Counter[str]]:
    """`page` with every eligible paragraph's `/layout/tabs*` derived from the
    regions of the runs that follow its TAB groups, and one count per
    paragraph outcome (plus one per ignored trailing TAB group)."""
    counts: Counter[str] = Counter()
    if page.node_states is None:
        return page, counts
    states = {state.local_node_id: state for state in page.node_states}
    items = _paragraph_items(states)
    if not any(kind == "tab" for sequence in items.values() for _, kind, _ in sequence):
        return page, counts
    regions = {p.local_node_id: p for p in page.region_predictions}
    by_node: dict[str, dict[str, Any]] = {}
    for prediction in page.typed_predictions:
        node_props = by_node.setdefault(prediction["local_node_id"], {})
        node_props[prediction["pointer"]] = prediction["value"]

    new_stops: dict[str, list[int]] = {}
    for paragraph, sequence in items.items():
        groups, trailing = _groups(sequence)
        counts[COUNT_PREFIX + "trailing_tab_ignored"] += trailing
        if not groups:
            if trailing:
                counts[COUNT_PREFIX + "paragraph_skipped_no_gap"] += 1
            continue
        props = by_node.get(paragraph, {})
        if _ancestor(paragraph, "cell", states) is not None:
            counts[COUNT_PREFIX + "paragraph_skipped_in_cell"] += 1
            continue
        if props.get(ALIGNMENT_POINTER, "left") not in ALIGNMENTS:
            counts[COUNT_PREFIX + "paragraph_skipped_alignment"] += 1
            continue
        section = _ancestor(paragraph, "section", states)
        geometry = by_node.get(section, {}) if section is not None else {}
        width, margin = geometry.get(WIDTH_POINTER), geometry.get(LEFT_MARGIN_POINTER)
        if not (_is_int(width) and _is_int(margin) and width > 0):
            counts[COUNT_PREFIX + "paragraph_skipped_no_section_geometry"] += 1
            continue
        targets: list[tuple[int, int]] = []
        for m, run_id in groups:
            region = regions.get(run_id)
            if region is None or region.alignment_status != "region" or not region.regions:
                targets = []
                break
            pos = round(region.regions[0].bbox_ppm[0] / PPM * width) - margin
            targets.append((pos, m))
        if not targets:
            counts[COUNT_PREFIX + "paragraph_skipped_run_without_region"] += 1
            continue
        if any(pos <= 0 for pos, _ in targets):
            counts[COUNT_PREFIX + "paragraph_skipped_nonpositive_stop"] += 1
            continue
        flat = _substitute_stops(targets)
        if _existing_stops(props) == flat:
            counts[COUNT_PREFIX + "paragraph_left_as_predicted"] += 1
            continue
        if _is_int(props.get(TABS_POINTER)) and props[TABS_POINTER] > 0:
            counts[COUNT_PREFIX + "predicted_stops_overridden"] += 1
        counts[COUNT_PREFIX + "paragraph_stops_written"] += 1
        new_stops[paragraph] = flat
    if not new_stops:
        return page, counts

    predictions = [
        p for p in page.typed_predictions
        if not (p["local_node_id"] in new_stops and p["pointer"].startswith(TABS_POINTER))
    ]
    for paragraph, flat in new_stops.items():
        def entry(pointer: str, value: Any, paragraph: str = paragraph) -> dict[str, Any]:
            return {"head_id": "policy." + POLICY_ID, "local_node_id": paragraph,
                    "node_type": "paragraph", "pointer": pointer, "value": value}
        predictions.append(entry(TABS_POINTER, len(flat)))
        for i, pos in enumerate(flat):
            predictions.append(entry(f"{TABS_POINTER}/{i}/position_mpt", pos))
            predictions.append(entry(f"{TABS_POINTER}/{i}/alignment", STOP_ALIGNMENT))
            predictions.append(entry(f"{TABS_POINTER}/{i}/leader", STOP_LEADER))
    return dataclasses.replace(page, typed_predictions=tuple(predictions)), counts

"""Opt-in chain policy `run-typography-snap.v0` (RATIFIED 2026-10-16, DEC-077;
`docs/rfc/RFC004-ERRATA-011.md` E2).

The model's run typography heads are regressions and their output carries
noise that gold never has (measured 2026-10-16 on the S4 v16 selection
cohort, 40 pages, 1 255 gold runs, 695 text-matched run pairs):

- gold `/style/font_size_mpt` sits 100 % on the 0.5 pt grid and a paragraph
  carries ONE size in 94-96 % of paragraphs; the model predicted continuous
  values (12 488, 10 697, ...) with a >= 1 pt spread inside 63 % of paragraphs
  -- the "mixed sizes" in the Word output;
- gold `/style/character_spacing_mpt` is 0 on every cohort run (|v| >= 0.25 pt
  on 3.6 % of runs over 150 corpus pages); the model predicted < -0.5 pt on
  30 % of runs and > +0.25 pt on 24 % -- the "compressed text".

`headless-defaults.v0` does not touch these pointers and the builder writes
the predicted values verbatim (`sz = round(mpt / 500)`), so the noise reaches
the renderer unchanged. This policy edits the run predictions before
assembly and counts every outcome:

1. `/style/character_spacing_mpt`: |v| < `CHAR_SPACING_ZERO_BELOW_MPT` becomes
   0 (`char_spacing_zeroed`); a larger value is a deliberate tracking and is
   kept (`char_spacing_kept`);
2. `/style/font_size_mpt`: rounded half-up to the `FONT_SIZE_GRID_MPT` grid
   (`font_size_rounded` when the value moved);
3. per paragraph (the run's nearest `paragraph` ancestor in `node_states`,
   through `hyperlink` and the like): if the runs' rounded sizes differ by at
   most `PARAGRAPH_UNIFY_RANGE_MPT`, every run gets the paragraph's median size
   on the grid (`paragraph_size_unified`); a larger spread is a real contrast
   (drop cap, inline heading) and stays (`paragraph_size_mixed_kept`); runs
   already sharing one size count `paragraph_single_size`.

Bold/italic, alignment, paragraph spacing and indents are NOT touched: their
errors are categorical or measured smaller, and alignment is a model
decision, not noise. The thresholds are read off the cohort statistics above,
not optimised; the page object is returned unchanged when nothing moved.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections import Counter
from typing import Any

from unflatten.eval.s2_gate import sealed

POLICY_ID = "run-typography-snap.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-011.md"
COUNT_PREFIX = "run_typography:"

SIZE_POINTER = "/style/font_size_mpt"
SPACING_POINTER = "/style/character_spacing_mpt"
#: 1 pt. Gold has |spacing| >= 0.25 pt on 3.6 % of runs; the model's noise
#: reaches -0.5 pt on 30 % of runs, so a 0.25 pt threshold would keep most of it.
CHAR_SPACING_ZERO_BELOW_MPT = 1_000
#: 0.5 pt: gold sizes sit on this grid in 100 % of cohort runs.
FONT_SIZE_GRID_MPT = 500
#: 2 pt: gold paragraphs carry one size in 94-96 % of cases; a real in-paragraph
#: contrast (inline heading, drop cap) is larger than this.
PARAGRAPH_UNIFY_RANGE_MPT = 2_000
RULES = (
    "char_spacing_below_threshold_zeroed.v0",
    "font_size_rounded_half_up_to_grid.v0",
    "paragraph_runs_within_range_take_median.v0",
)


def policy_record() -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "char_spacing_zero_below_mpt": CHAR_SPACING_ZERO_BELOW_MPT,
        "font_size_grid_mpt": FONT_SIZE_GRID_MPT,
        "paragraph_unify_range_mpt": PARAGRAPH_UNIFY_RANGE_MPT,
        "scope": {"node_types": ["run"], "pointers": [SIZE_POINTER, SPACING_POINTER]},
        "rules": list(RULES),
    }


def _to_grid(value: int) -> int:
    """Round half-up to the font-size grid (`round` would send 10 250 to 10 000)."""
    return int(math.floor(value / FONT_SIZE_GRID_MPT + 0.5)) * FONT_SIZE_GRID_MPT


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _paragraph_of(node_id: str, parents: dict[str, tuple[str, str | None]]) -> str | None:
    """The nearest `paragraph` ancestor's id, or None."""
    seen: set[str] = set()
    current = parents.get(node_id, (None, None))[1]
    while current is not None and current not in seen:
        seen.add(current)
        node_type, parent = parents.get(current, (None, None))
        if node_type == "paragraph":
            return current
        current = parent
    return None


def apply(page: sealed.PageDecodeResult) -> tuple[sealed.PageDecodeResult, Counter[str]]:
    """`page` with run typography snapped, and one count per outcome."""
    counts: Counter[str] = Counter()
    if page.node_states is None:
        return page, counts
    predictions = list(page.typed_predictions)
    edited = False
    size_index: dict[str, int] = {}
    for i, p in enumerate(predictions):
        if p["node_type"] != "run" or not _is_int(p["value"]):
            continue
        if p["pointer"] == SPACING_POINTER:
            value = p["value"]
            if value == 0:
                continue
            if abs(value) < CHAR_SPACING_ZERO_BELOW_MPT:
                predictions[i] = {**p, "value": 0}
                counts[COUNT_PREFIX + "char_spacing_zeroed"] += 1
                edited = True
            else:
                counts[COUNT_PREFIX + "char_spacing_kept"] += 1
        elif p["pointer"] == SIZE_POINTER:
            snapped = _to_grid(p["value"])
            if snapped != p["value"]:
                predictions[i] = {**p, "value": snapped}
                counts[COUNT_PREFIX + "font_size_rounded"] += 1
                edited = True
            size_index[p["local_node_id"]] = i
    if size_index:
        parents = {s.local_node_id: (s.node_type, s.parent_local_node_id) for s in page.node_states}
        groups: dict[str, list[int]] = {}
        for run_id, i in size_index.items():
            paragraph = _paragraph_of(run_id, parents)
            if paragraph is not None:
                groups.setdefault(paragraph, []).append(i)
        for indices in groups.values():
            sizes = [predictions[i]["value"] for i in indices]
            spread = max(sizes) - min(sizes)
            if spread == 0:
                counts[COUNT_PREFIX + "paragraph_single_size"] += 1
            elif spread <= PARAGRAPH_UNIFY_RANGE_MPT:
                target = _to_grid(int(statistics.median(sizes)))
                for i in indices:
                    if predictions[i]["value"] != target:
                        predictions[i] = {**predictions[i], "value": target}
                        edited = True
                counts[COUNT_PREFIX + "paragraph_size_unified"] += 1
            else:
                counts[COUNT_PREFIX + "paragraph_size_mixed_kept"] += 1
    if not edited:
        return page, counts
    return dataclasses.replace(page, typed_predictions=tuple(predictions)), counts

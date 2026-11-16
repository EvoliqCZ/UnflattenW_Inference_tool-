"""Paint-versus-text bookkeeping for the admission scanner, in device space.

A text run is visible only if no paint can hide it and no paint lies under it:
  * paint whose device bound overlaps a run by at least a quarter of the run's
    glyph size in both directions can cover the glyphs (after) or is an unknown
    background (before) -> ambiguous, whichever came first;
  * thinner overlaps (rules, underlines, borders) are slivers that cannot hide a
    glyph on their own; more than `_MAX_SLIVERS` of them on one run fail closed;
  * paint with an unknown bound fails closed against every run, as before.
Glyph ink never hides glyph ink, so runs are not paint.
"""

from __future__ import annotations

from scanner_geometry import Matrix, Rect
from scanner_state import ContentState, TextRun, Visibility

_SLIVER_FRACTION = 0.25
_MAX_SLIVERS = 8
_ASCENT = 1.0
_DESCENT = 0.35
_EM = 1000.0


def text_run(
    state: ContentState,
    rendering: Matrix,
    origin: tuple[float, float],
    scale: float,
    min_extent: float,
    max_extent: float,
) -> TextRun:
    """Bound one run along its baseline by `[min_extent, max_extent]` (in
    thousandths of an em, relative to the glyph origin) -- the real ink span
    `_shown_extent` computed, using resolved glyph widths where available.

    Only upright, unmirrored text (b == c == 0, a > 0, d > 0) gets a strip; any
    other orientation is bounded by the clip and page, or by nothing.
    """
    graphics = state.graphics
    a, b, c, d, _, _ = rendering
    if b != 0 or c != 0 or a <= 0 or d <= 0:
        rect = graphics.clip if graphics.clip is not None else graphics.page_box
    else:
        x, y = origin
        rect = (
            x + min_extent / _EM * scale,
            y - _DESCENT * scale,
            x + max_extent / _EM * scale,
            y + _ASCENT * scale,
        )
        for bound in (graphics.clip, graphics.page_box):
            if bound is not None:
                rect = (
                    max(rect[0], bound[0]),
                    max(rect[1], bound[1]),
                    min(rect[2], bound[2]),
                    min(rect[3], bound[3]),
                )
    return TextRun(rect=rect, scale=scale)


def background_known(state: ContentState, run: TextRun) -> bool:
    """Whether every earlier paint stays clear of the run (slivers allowed)."""
    visibility = state.visibility
    if visibility.unbounded_paint:
        return False
    slivers = 0
    for region in visibility.paints:
        kind = _overlap(run, region)
        if kind == "cover":
            return False
        slivers += kind == "sliver"
    return slivers <= _MAX_SLIVERS


def record_run(state: ContentState, run: TextRun) -> None:
    state.visibility.runs.append(run)


def record_paint(state: ContentState, region: Rect | None, *, background: bool = False) -> None:
    """Paint laid down now: it may cover earlier runs and shadows later ones."""
    visibility = state.visibility
    if region is None:
        visibility.unbounded_paint = True
        if visibility.runs:
            visibility.ambiguous = True
        return
    for run in visibility.runs:
        if _covers_after(run, region):
            visibility.ambiguous = True
    (visibility.covers if background else visibility.paints).append(region)


def merge_nested(state: ContentState, nested: Visibility) -> None:
    """Fold a Form XObject's runs and paints into the invoking stream's."""
    visibility = state.visibility
    if nested.unbounded_paint and visibility.runs:
        visibility.ambiguous = True
    for region in nested.paints + nested.covers:
        for run in visibility.runs:
            if _covers_after(run, region):
                visibility.ambiguous = True
    for run in nested.runs:
        if not background_known(state, run):
            visibility.ambiguous = True
    visibility.merge(nested)
    visibility.runs.extend(nested.runs)
    visibility.paints.extend(nested.paints)
    visibility.covers.extend(nested.covers)
    visibility.unbounded_paint |= nested.unbounded_paint


def _covers_after(run: TextRun, region: Rect) -> bool:
    kind = _overlap(run, region)
    if kind == "sliver":
        run.slivers += 1
        return run.slivers > _MAX_SLIVERS
    return kind == "cover"


def _overlap(run: TextRun, region: Rect) -> str:
    if run.rect is None:
        return "cover"
    width = min(run.rect[2], region[2]) - max(run.rect[0], region[0])
    height = min(run.rect[3], region[3]) - max(run.rect[1], region[1])
    if width <= 0 or height <= 0:
        return "none"
    threshold = _SLIVER_FRACTION * run.scale
    return "cover" if width >= threshold and height >= threshold else "sliver"

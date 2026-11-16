"""Token-grid labels for the ViT-level probe (discovery, 2026-09-29).

Which Qwen3-VL ViT level carries fine document signals is a per-TOKEN
question: DeepStack injects each level at the visual positions of the
decoder, one row per merged 32 px token. So every label here is assigned
to a merged token of the backbone grid, and every label is either a class
or IGNORE (-1). A missing label is never encoded as a class.

Sources, and why each one:

- **Text-layer words and lines** come from the rendered PDF
  (`pdftotext -bbox-layout`) that the page PNG was rasterised from.
  They are tight glyph boxes, unlike the aligner's node boxes, which are
  unions over whole lines.
- **Word properties** (font size, bold, italic, paragraph spacing) come
  from the AST, which is the target. They are localised by the aligner's
  observation boxes (`bbox_ppm` per node). A run's box is a union over the
  lines it touches, so a token is labelled only when every run box over it
  agrees AND every paragraph over it has all of its glyph runs observed.
  Without that, an unmatched run lying inside a neighbour's union box would
  borrow the neighbour's value.
- **Table lines** come from the render's pixels inside the aligner's
  table box. The aligner's cell boxes are text unions, not cell edges, so
  they cannot tell where a grid line is; the drawn line in the render can.
  The AST border declaration is kept as a per-table cross-check
  (`table_checks`), not as the token label.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from lxml import etree

PPM = 1_000_000
IGNORE = -1

SIGNALS: tuple[str, ...] = (
    "text_blank",
    "font_size",
    "bold",
    "italic",
    "space_before",
    "space_after",
    "line_offset",
    "line_pitch",
    "grid_line",
    "border_weight",
)

#: Upper edges in pt; a value above the last edge is the last class.
#: Word sizes are half-points, so the edges sit between the common sizes.
FONT_SIZE_EDGES_PT: tuple[float, ...] = (8.5, 9.5, 10.5, 11.5, 12.5, 14.5, 18.5)
#: Class 0 is exactly zero spacing; the rest bin (0, 3], (3, 7] (6 pt),
#: (7, 9] (8 pt, Word's default after), (9, 11], (11, 13], > 13.
SPACING_EDGES_PT: tuple[float, ...] = (3.0, 7.0, 9.0, 11.0, 13.0)
LINE_PITCH_EDGES_PT: tuple[float, ...] = (11.0, 13.0, 15.0, 18.0, 24.0)
LINE_OFFSET_BINS = 4

N_CLASSES: dict[str, int] = {
    "text_blank": 2,
    "font_size": len(FONT_SIZE_EDGES_PT) + 1,
    "bold": 2,
    "italic": 2,
    "space_before": len(SPACING_EDGES_PT) + 2,
    "space_after": len(SPACING_EDGES_PT) + 2,
    "line_offset": LINE_OFFSET_BINS,
    "line_pitch": len(LINE_PITCH_EDGES_PT) + 1,
    "grid_line": 2,
    "border_weight": 3,
}

#: A token counts as text when word boxes cover at least this fraction of
#: it, and as blank only when they cover none of it; in between it is
#: IGNORE rather than a guess.
TEXT_FRACTION = 0.15
#: Aligner matches below this confidence (ppm) are not used for labels.
MIN_CONFIDENCE = 900_000
#: Table-line detector (pixels at 300 DPI): dark threshold, minimum run
#: length (two tokens, longer than any body-text glyph stroke), maximum
#: thickness (thicker is a filled area, not a line), and the line-pixel
#: count that makes a token "on a line".
DARK = 160
MIN_LINE_PX = 64
MAX_LINE_THICK_PX = 12
LINE_TOKEN_PX = 16
#: Thin/thick split: 2 px at 300 DPI is 0.48 pt, so "thin" is up to a
#: half-point border and "thick" is 0.75 pt and up.
THIN_MAX_PX = 2


def font_size_class(size_pt: float) -> int:
    return bisect.bisect_left(FONT_SIZE_EDGES_PT, size_pt)


def spacing_class(spacing_pt: float) -> int:
    if spacing_pt <= 0:
        return 0
    return 1 + bisect.bisect_left(SPACING_EDGES_PT, spacing_pt)


def line_pitch_class(pitch_pt: float) -> int:
    return bisect.bisect_left(LINE_PITCH_EDGES_PT, pitch_pt)


# ------------------------------------------------------------------ grid --


@dataclass(frozen=True)
class TokenGrid:
    """The merged-token grid laid over a page raster.

    The Qwen3-VL processor resizes the whole page to a multiple of 32 px,
    and neither crops nor pads it, so token (r, c) covers the same FRACTION
    of the page at any size. Tokens are row-major, like the pooler rows.
    """

    rows: int
    cols: int
    width_px: int
    height_px: int

    @property
    def count(self) -> int:
        return self.rows * self.cols

    @property
    def token_h(self) -> float:
        return self.height_px / self.rows

    @property
    def token_w(self) -> float:
        return self.width_px / self.cols

    def boxes(self) -> np.ndarray:
        """(count, 4) float pixel boxes x0, y0, x1, y1, row-major."""
        r = np.repeat(np.arange(self.rows), self.cols).astype(np.float64)
        c = np.tile(np.arange(self.cols), self.rows).astype(np.float64)
        return np.stack(
            [c * self.token_w, r * self.token_h, (c + 1) * self.token_w, (r + 1) * self.token_h],
            axis=1,
        )


def grid_from_thw(grid_thw: Sequence[int], merge: int, width_px: int, height_px: int) -> TokenGrid:
    _, h, w = (int(v) for v in grid_thw)
    if h % merge or w % merge:
        raise ValueError(f"patch grid {h}x{w} is not divisible by merge {merge}")
    return TokenGrid(rows=h // merge, cols=w // merge, width_px=width_px, height_px=height_px)


def ppm_to_px(box: Sequence[float], width_px: int, height_px: int) -> tuple[float, ...]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return (x0 * width_px / PPM, y0 * height_px / PPM, x1 * width_px / PPM, y1 * height_px / PPM)


def _overlap(boxes: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """Tokens whose box intersects `box` with positive area."""
    x0, y0, x1, y1 = box
    return (boxes[:, 0] < x1) & (boxes[:, 2] > x0) & (boxes[:, 1] < y1) & (boxes[:, 3] > y0)


class _Agree:
    """Per-token 'every claim gives the same value' accumulator."""

    def __init__(self, count: int) -> None:
        self.lo = np.full(count, np.iinfo(np.int32).max, dtype=np.int64)
        self.hi = np.full(count, -1, dtype=np.int64)

    def claim(self, where: np.ndarray, value: int) -> None:
        self.lo[where] = np.minimum(self.lo[where], value)
        self.hi[where] = np.maximum(self.hi[where], value)

    def result(self) -> np.ndarray:
        out = np.full(self.lo.shape, IGNORE, dtype=np.int64)
        ok = (self.hi >= 0) & (self.lo == self.hi)
        out[ok] = self.hi[ok]
        return out


# ------------------------------------------------------------ text layer --


@dataclass(frozen=True)
class TextLine:
    """One pdftotext line: its box and its word boxes (ppm)."""

    box: tuple[int, int, int, int]
    words: tuple[tuple[int, int, int, int], ...]


def _local(element: etree._Element) -> str:
    tag = element.tag
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def parse_text_lines(xhtml: bytes) -> tuple[list[list[TextLine]], list[float]]:
    """Per page: the text lines, plus the page height in pt.

    Coordinates become ppm of the page, the same frame as the aligner's
    `bbox_ppm`. Degenerate word boxes are dropped (they point at no pixels).
    """
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    root = etree.fromstring(xhtml, parser=parser)
    pages: list[list[TextLine]] = []
    heights: list[float] = []
    for page in (e for e in root.iter() if _local(e) == "page"):
        width, height = float(page.get("width")), float(page.get("height"))
        heights.append(height)

        def ppm(el: etree._Element) -> tuple[int, int, int, int]:
            return (
                round(float(el.get("xMin")) / width * PPM),
                round(float(el.get("yMin")) / height * PPM),
                round(float(el.get("xMax")) / width * PPM),
                round(float(el.get("yMax")) / height * PPM),
            )

        lines: list[TextLine] = []
        for block in (e for e in page.iter() if _local(e) == "block"):
            for line in (e for e in block if _local(e) == "line"):
                boxes = [ppm(w) for w in line if _local(w) == "word" and (w.text or "").strip()]
                words = tuple(b for b in boxes if b[0] < b[2] and b[1] < b[3])
                if words:
                    lines.append(TextLine(box=ppm(line), words=words))
        pages.append(lines)
    return pages, heights


def _integral(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    out[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
    return out


def _box_sums(integral: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Sum of the mask inside every (rounded) token box."""
    h, w = integral.shape[0] - 1, integral.shape[1] - 1
    x0 = np.clip(np.round(boxes[:, 0]).astype(int), 0, w)
    y0 = np.clip(np.round(boxes[:, 1]).astype(int), 0, h)
    x1 = np.clip(np.round(boxes[:, 2]).astype(int), 0, w)
    y1 = np.clip(np.round(boxes[:, 3]).astype(int), 0, h)
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def _box_areas(boxes: np.ndarray) -> np.ndarray:
    return (np.round(boxes[:, 2]) - np.round(boxes[:, 0])) * (
        np.round(boxes[:, 3]) - np.round(boxes[:, 1])
    )


def text_fraction(grid: TokenGrid, lines: Iterable[TextLine]) -> np.ndarray:
    """Fraction of each token covered by a word box."""
    mask = np.zeros((grid.height_px, grid.width_px), dtype=bool)
    for line in lines:
        for word in line.words:
            x0, y0, x1, y1 = ppm_to_px(word, grid.width_px, grid.height_px)
            mask[int(round(y0)) : int(round(y1)), int(round(x0)) : int(round(x1))] = True
    boxes = grid.boxes()
    return _box_sums(_integral(mask), boxes) / np.maximum(_box_areas(boxes), 1)


# ----------------------------------------------------------- table lines --


def run_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    """For every True pixel, the length of its run of Trues along `axis`."""
    m = np.moveaxis(mask, axis, -1)
    rows, n = m.shape
    padded = np.zeros((rows, n + 2), dtype=np.int8)
    padded[:, 1:-1] = m
    flat = padded.ravel()
    edges = np.diff(flat)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    lengths = ends - starts
    # each run [start+1, end] in the padded flat index gets its length
    delta = np.zeros(flat.shape[0] + 1, dtype=np.int64)
    np.add.at(delta, starts + 1, lengths)
    np.add.at(delta, ends + 1, -lengths)
    out = np.cumsum(delta)[:-1].astype(np.int32)
    out = out.reshape(rows, n + 2)[:, 1:-1]
    return np.moveaxis(out, -1, axis)


def line_pixels(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(line mask, thickness map) of long, thin, straight dark strokes.

    A horizontal line pixel sits in a horizontal dark run of at least
    MIN_LINE_PX, and its thickness is the vertical extent of such pixels;
    vertical lines symmetrically. Anything thicker than MAX_LINE_THICK_PX is
    a filled area and is dropped.
    """
    dark = gray < DARK
    h_long = run_lengths(dark, axis=1) >= MIN_LINE_PX
    h_thick = run_lengths(h_long, axis=0)
    h_line = h_long & (h_thick <= MAX_LINE_THICK_PX)
    v_long = run_lengths(dark, axis=0) >= MIN_LINE_PX
    v_thick = run_lengths(v_long, axis=1)
    v_line = v_long & (v_thick <= MAX_LINE_THICK_PX)
    thickness = np.where(h_line, h_thick, 0)
    thickness = np.where(v_line & ((thickness == 0) | (v_thick < thickness)), v_thick, thickness)
    return h_line | v_line, thickness


# ------------------------------------------------------------------- AST --


_GLYPHLESS = set(" \t\n\r ​")


@dataclass
class AstIndex:
    nodes: dict[str, dict[str, Any]]
    #: paragraph id -> its runs with glyph text (nearest-paragraph ownership)
    paragraph_runs: dict[str, list[str]]


def index_ast(ast: Mapping[str, Any]) -> AstIndex:
    nodes: dict[str, dict[str, Any]] = {}
    paragraph_runs: dict[str, list[str]] = {}

    def walk(value: Any, paragraph: str | None) -> None:
        if isinstance(value, dict):
            node_id, kind = value.get("id"), value.get("type")
            if isinstance(node_id, str) and isinstance(kind, str):
                nodes[node_id] = value
                if kind == "paragraph":
                    paragraph = node_id
                    paragraph_runs.setdefault(node_id, [])
                elif kind in ("table", "text_frame", "cell"):
                    paragraph = None
                elif kind == "run" and paragraph is not None:
                    text = value.get("text") or ""
                    if any(ch not in _GLYPHLESS for ch in text):
                        paragraph_runs[paragraph].append(node_id)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child, paragraph)
        elif isinstance(value, list):
            for child in value:
                walk(child, paragraph)

    walk(ast, None)
    return AstIndex(nodes=nodes, paragraph_runs=paragraph_runs)


def _visible_border(border: Any) -> bool:
    return (
        isinstance(border, dict)
        and border.get("style") not in (None, "none", "nil")
        and (border.get("width_mpt") or 0) > 0
    )


def table_declares_borders(table: Mapping[str, Any]) -> bool:
    for row in table.get("rows") or []:
        for cell in row.get("cells") or []:
            borders = ((cell.get("style") or {}).get("borders")) or {}
            if any(_visible_border(b) for b in borders.values()):
                return True
    borders = ((table.get("style") or {}).get("borders")) or {}
    return any(_visible_border(b) for b in borders.values())


# --------------------------------------------------------------- labels --


@dataclass
class PageLabels:
    labels: dict[str, np.ndarray]
    table_checks: list[dict[str, Any]]


def page_labels(
    grid: TokenGrid,
    observations: Sequence[Mapping[str, Any]],
    ast: AstIndex,
    lines: Sequence[TextLine],
    page_height_pt: float,
    gray: np.ndarray,
) -> PageLabels:
    """Every signal for every token of one page; IGNORE where unknown."""
    if gray.shape != (grid.height_px, grid.width_px):
        raise ValueError(f"raster {gray.shape} does not match grid {grid}")
    boxes = grid.boxes()
    n = grid.count
    fraction = text_fraction(grid, lines)
    is_text = fraction >= TEXT_FRACTION

    labels: dict[str, np.ndarray] = {}
    tb = np.full(n, IGNORE, dtype=np.int64)
    tb[is_text] = 1
    tb[fraction == 0] = 0
    labels["text_blank"] = tb

    observed: dict[str, tuple[float, ...]] = {}
    for obs in observations:
        box = obs.get("bbox_ppm")
        node_id = obs.get("node_id")
        if not box or len(box) != 4 or node_id not in ast.nodes:
            continue
        if (obs.get("mapping_confidence") or 0) < MIN_CONFIDENCE:
            continue
        px = ppm_to_px(box, grid.width_px, grid.height_px)
        if px[0] >= px[2] or px[1] >= px[3]:
            continue
        if node_id in observed:  # continuation fragments: union
            old = observed[node_id]
            px = (min(old[0], px[0]), min(old[1], px[1]), max(old[2], px[2]), max(old[3], px[3]))
        observed[node_id] = px

    def kind(node_id: str) -> str:
        return ast.nodes[node_id].get("type", "")

    # --- run properties: every run box agrees, every paragraph complete
    incomplete = np.zeros(n, dtype=bool)
    paragraph_cover = np.zeros(n, dtype=np.int64)
    for node_id, box in observed.items():
        if kind(node_id) != "paragraph":
            continue
        over = _overlap(boxes, box)
        paragraph_cover += over
        if any(run not in observed for run in ast.paragraph_runs.get(node_id, [])):
            incomplete |= over
    size, bold, italic = _Agree(n), _Agree(n), _Agree(n)
    any_run = np.zeros(n, dtype=bool)
    for node_id, box in observed.items():
        if kind(node_id) != "run":
            continue
        style = ast.nodes[node_id].get("style") or {}
        over = _overlap(boxes, box)
        any_run |= over
        if style.get("font_size_mpt") is not None:
            size.claim(over, font_size_class(style["font_size_mpt"] / 1000.0))
        if style.get("bold") is not None:
            bold.claim(over, int(bool(style["bold"])))
        if style.get("italic") is not None:
            italic.claim(over, int(bool(style["italic"])))
    run_ok = is_text & any_run & ~incomplete
    for name, acc in (("font_size", size), ("bold", bold), ("italic", italic)):
        value = acc.result()
        value[~run_ok] = IGNORE
        labels[name] = value

    # --- paragraph spacing on the first / last token row of the paragraph
    before, after = _Agree(n), _Agree(n)
    rows = np.repeat(np.arange(grid.rows), grid.cols)
    for node_id, box in observed.items():
        if kind(node_id) != "paragraph":
            continue
        spacing = (ast.nodes[node_id].get("layout") or {}).get("spacing") or {}
        x_over = (boxes[:, 0] < box[2]) & (boxes[:, 2] > box[0])
        top_row = int(box[1] // grid.token_h)
        bottom_row = int(max(box[1], box[3] - 1e-6) // grid.token_h)
        if spacing.get("before_mpt") is not None:
            before.claim(x_over & (rows == top_row), spacing_class(spacing["before_mpt"] / 1000.0))
        if spacing.get("after_mpt") is not None:
            after.claim(x_over & (rows == bottom_row), spacing_class(spacing["after_mpt"] / 1000.0))
    alone = is_text & (paragraph_cover == 1)
    for name, acc in (("space_before", before), ("space_after", after)):
        value = acc.result()
        value[~alone] = IGNORE
        labels[name] = value

    # --- text lines: bottom-edge offset inside the token, and line pitch.
    # pdftotext splits widely spaced lines into separate blocks, so lines
    # are grouped by the aligner's paragraph box that holds their centre.
    offset_claims = np.zeros(n, dtype=np.int64)
    offset = np.full(n, IGNORE, dtype=np.int64)
    pitch = _Agree(n)
    pt_per_px = page_height_pt / grid.height_px
    paragraphs = [(i, b) for i, b in observed.items() if kind(i) == "paragraph"]
    by_paragraph: dict[str, list[tuple[float, ...]]] = {}
    for line in lines:
        box = ppm_to_px(line.box, grid.width_px, grid.height_px)
        bottom = box[3]
        row = int(min(bottom, grid.height_px - 1e-6) // grid.token_h)
        hit = (rows == row) & (boxes[:, 0] < box[2]) & (boxes[:, 2] > box[0])
        offset_claims += hit
        offset[hit] = min(
            LINE_OFFSET_BINS - 1,
            int((bottom - row * grid.token_h) / grid.token_h * LINE_OFFSET_BINS),
        )
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        owners = [i for i, b in paragraphs if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
        if len(owners) == 1:
            by_paragraph.setdefault(owners[0], []).append(box)
    for members in by_paragraph.values():
        members.sort(key=lambda b: b[3])
        for box, nxt in zip(members, members[1:]):
            same_column = nxt[0] < box[2] and nxt[2] > box[0]
            step_pt = (nxt[3] - box[3]) * pt_per_px
            if same_column and 0 < step_pt < 60:
                pitch.claim(_overlap(boxes, box) & is_text, line_pitch_class(step_pt))
    offset[offset_claims != 1] = IGNORE
    labels["line_offset"] = offset
    value = pitch.result()
    value[~is_text] = IGNORE
    labels["line_pitch"] = value

    # --- table lines from the render, inside the (widened) table box
    grid_line = np.full(n, IGNORE, dtype=np.int64)
    weight = np.full(n, IGNORE, dtype=np.int64)
    checks: list[dict[str, Any]] = []
    tables = [(i, b) for i, b in observed.items() if kind(i) == "table"]
    if tables:
        line_mask, thickness = line_pixels(gray)
        counts = _box_sums(_integral(line_mask), boxes)
        centre_x = (boxes[:, 0] + boxes[:, 2]) / 2
        centre_y = (boxes[:, 1] + boxes[:, 3]) / 2
        region_all = np.zeros(n, dtype=bool)
        for node_id, box in tables:
            region = (
                (centre_x >= box[0] - grid.token_w)
                & (centre_x <= box[2] + grid.token_w)
                & (centre_y >= box[1] - grid.token_h)
                & (centre_y <= box[3] + grid.token_h)
            )
            region_all |= region
            checks.append(
                {
                    "table": node_id,
                    "declares_borders": table_declares_borders(ast.nodes[node_id]),
                    "line_tokens": int((region & (counts >= LINE_TOKEN_PX)).sum()),
                    "region_tokens": int(region.sum()),
                }
            )
        grid_line[region_all & (counts == 0)] = 0
        weight[region_all & (counts == 0)] = 0
        on_line = region_all & (counts >= LINE_TOKEN_PX)
        grid_line[on_line] = 1
        for t in np.flatnonzero(on_line):
            x0, y0, x1, y1 = (int(round(v)) for v in boxes[t])
            patch = thickness[y0:y1, x0:x1]
            median = float(np.median(patch[patch > 0]))
            weight[t] = 1 if median <= THIN_MAX_PX else 2
    labels["grid_line"] = grid_line
    labels["border_weight"] = weight
    return PageLabels(labels={k: labels[k] for k in SIGNALS}, table_checks=checks)

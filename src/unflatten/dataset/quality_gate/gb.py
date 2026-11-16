"""G-B of the DEC-057 data-quality gate: what the target claims against
what the page shows, automatically, on CPU.

Three measurements per page, all from artefacts the store already holds
(page PNG, document PDF) and the page's own target tokens:

* **trailing table rows vs the page** (`grid_rows`) -- the defect it exists
  for is `spill` (audit §3): textless rows the envelope borrowed from a
  skipped page, so the target ends a table with rows the image does not
  show. For every table that ends in textless rows, the target's last row
  WITH text and the first text after the table are located in the PDF
  text layer; between them the page shows either horizontal rules (one
  per visible row border, `rule_positions`) or, for a borderless table,
  empty space that holds at most `gap / min_row_height` rows. Rows the
  target carries beyond that are `grid_excess_rows`.
* **target text vs visible text** -- the target's letters/digits against
  those of the page's PDF text layer (`pdftotext -bbox`, poppler, already
  used by M2b). The PNG is rasterised from that PDF, so its text layer IS
  the visible text, exact where OCR would guess; no OCR engine is needed
  (tesseract exists locally and on prod but only with eng/slk data, and
  the corpus is multilingual). Characters rather than words, because the
  layer splits letter-spaced and vertical text into one-letter words.
  `visible_missing_*` = visible text the target lacks (withdrawn visible
  text, audit §3 #7/#13/#18, V1c); `target_unseen_*` = target text the
  page does not show.
* **target text without ink** -- every text-layer word the target also
  carries, and the PNG inside its box: a claimed word whose box holds no
  mark at all (white or hidden text) is not visible supervision.
* **page edges** (the G-C trial's [pagenum] and [boundary] classes,
  `docs/runs/target-defects-root-cause-2026-09-30.md`) -- the text layer
  grouped into lines (`layer_lines`): a running page number on the page's
  first/last line the target carries neither as a header/footer `FIELD`
  nor as story text (`page_number_missing`), and the word-pair coverage
  of the first and last body line (`edge_line_coverage`).

`measure` returns integers only; `flags` applies the policy file.
"""

from __future__ import annotations

import html
import re
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unflatten.dataset.quality_gate.token_tree import ID, QWEN_START, parse_tree, walk

GB_VERSION = "3"  # 2: page_number + edge_lines (G-C trial classes, 2026-09-30)
# 3: edge_lines covers a line the target spells letter for letter (small caps);
#    underscores separate words (fill-in leaders)

#: Qwen3-VL native vocabulary (composite_vocab.packed_ranges(151936)):
#: after it come the two byte-frame controls, then 256 byte values.
QWEN_VOCAB = 151936
BYTE_BEGIN = QWEN_START + QWEN_VOCAB
BYTE_END = BYTE_BEGIN + 1
BYTE_VALUE_START = BYTE_BEGIN + 2

TB, TE = ID["TEXT_BEGIN"], ID["TEXT_END"]
RO, RC = ID["RUN_OPEN"], ID["RUN_CLOSE"]
TO, ROW_O = ID["TABLE_OPEN"], ID["ROW_OPEN"]
FRAME_O, FIELD_T = ID["TEXT_FRAME_OPEN"], ID["FIELD"]
_STORY_OPEN = frozenset((ID["HEADER_OPEN"], ID["FOOTER_OPEN"], FRAME_O))

#: a table-rule pixel is darker than this (0..255 grey)
RULE_LEVEL = 192
#: a word box has ink when any pixel is darker than this (light grey text
#: counts; anti-aliasing leaves no glyph box pure white)
INK_LEVEL = 232
#: a horizontal rule is a dark run at least this share of the page width
RULE_MIN_WIDTH_MILLI = 100
#: ... and at most this thick (share of page height; 3 milli = 10 px at 300 DPI)
RULE_MAX_THICKNESS_MILLI = 3
#: a word must have at least this many letters/digits to locate a row
MIN_WORD_CHARS = 3
#: words used to locate a row / the text after a table
LOCATE_WORDS = 6
#: an edge line whose letters/digits (at least this many) occur contiguously
#: in the target's letters/digits is covered, whatever its word split
EDGE_SPELLED_MIN_CHARS = 12


# -- the target ------------------------------------------------------------------


@dataclass(frozen=True)
class TableTail:
    trailing_textless_rows: int
    last_row_words: tuple[str, ...]  # of the table's last row with text
    next_words: tuple[str, ...]  # first words after the table


@dataclass(frozen=True)
class TargetFacts:
    chars: Counter[str]
    words: frozenset[str]
    tails: tuple[TableTail, ...]  # tables that end in >= 1 textless row
    #: text of the page's header/footer/text-frame stories
    story_text: str = ""
    #: a `FIELD` inside a header or footer (a running page number)
    hf_field: bool = False
    #: adjacent pairs of `words_of(all target text)`
    word_pairs: frozenset[tuple[str, str]] = frozenset()
    #: every letter/digit of the target text, in order, casefolded
    alnum: str = ""


def decode_text_tokens(tokens: Sequence[int], tokenizer: Any) -> str:
    """One `TEXT_BEGIN..TEXT_END` body back to text: Qwen runs through the
    pinned tokenizer, byte frames as UTF-8 (lenient -- this is a
    comparison, not the exact-text validator)."""
    out: list[str] = []
    qwen: list[int] = []
    frame: bytearray | None = None
    for t in tokens:
        if QWEN_START <= t < BYTE_BEGIN:
            qwen.append(t - QWEN_START)
            continue
        if qwen:
            out.append(tokenizer.decode(qwen))
            qwen = []
        if t == BYTE_BEGIN:
            frame = bytearray()
        elif t == BYTE_END and frame is not None:
            out.append(bytes(frame).decode("utf-8", errors="replace"))
            frame = None
        elif t >= BYTE_VALUE_START and frame is not None:
            frame.append(t - BYTE_VALUE_START)
    if qwen:
        out.append(tokenizer.decode(qwen))
    return "".join(out)


def stream_text(tokens: Sequence[int], start: int, end: int, tokenizer: Any) -> str:
    """Text of `tokens[start:end]`. Runs of one paragraph join without a
    separator (a word may be split across runs); every other structural
    token (paragraph/cell bounds, TAB, LINE_BREAK, ...) separates words."""
    out: list[str] = []
    i = start
    while i < end:
        t = tokens[i]
        if t == TB:
            j = i + 1
            while j < end and tokens[j] != TE:
                j += 1
            out.append(decode_text_tokens(tokens[i + 1 : j], tokenizer))
            i = j
        elif t < QWEN_START and t not in (RO, RC):
            out.append(" ")
        i += 1
    return "".join(out)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def chars_of(text: str) -> Counter[str]:
    return Counter(ch for ch in _norm(text) if ch.isalnum())


def words_of(text: str) -> list[str]:
    """Letter/digit words of at least `MIN_WORD_CHARS`; an underscore is a
    separator (a fill-in leader renders with its own length, GB 3)."""
    return [
        w
        for w in re.findall(r"[^\W_]+", _norm(text))
        if sum(c.isalnum() for c in w) >= MIN_WORD_CHARS
    ]


def target_facts(tokens: Sequence[int], tokenizer: Any) -> TargetFacts:
    text = stream_text(tokens, 0, len(tokens), tokenizer)
    tails = []
    stories = [n for n in walk(parse_tree(tokens)) if n.kind in _STORY_OPEN]
    story_text = " ".join(stream_text(tokens, n.start, n.end + 1, tokenizer) for n in stories)
    hf_field = any(n.kind != FRAME_O and FIELD_T in tokens[n.start : n.end + 1] for n in stories)
    all_words = words_of(text)
    for node in walk(parse_tree(tokens)):
        if node.kind != TO:
            continue
        rows = [c for c in node.children if c.kind == ROW_O]
        trailing = 0
        for r in reversed(rows):
            if not r.textless:
                break
            trailing += 1
        if trailing == 0:
            continue
        # the last row with text; for an all-textless table, the text
        # right before the table
        textful = [r for r in rows if not r.textless]
        if textful:
            last = words_of(stream_text(tokens, textful[-1].start, textful[-1].end, tokenizer))
        else:
            last = words_of(stream_text(tokens, 0, node.start, tokenizer))
        after = words_of(stream_text(tokens, node.end + 1, len(tokens), tokenizer))
        tails.append(TableTail(trailing, tuple(last[-LOCATE_WORDS:]), tuple(after[:LOCATE_WORDS])))
    return TargetFacts(
        chars_of(text),
        frozenset(all_words),
        tuple(tails),
        story_text=story_text,
        hf_field=hf_field,
        word_pairs=frozenset(zip(all_words, all_words[1:], strict=False)),
        alnum=_alnum(text),
    )


# -- the page image ------------------------------------------------------------


def longest_runs(mask: np.ndarray) -> np.ndarray:
    """Per row, the longest run of True pixels (vectorised)."""
    ones = mask.astype(np.int32)
    csum = np.cumsum(ones, axis=1)
    reset = np.where(mask, 0, csum)
    np.maximum.accumulate(reset, axis=1, out=reset)
    return (csum - reset).max(axis=1)


def rule_positions(gray: np.ndarray, *, min_width_milli: int = RULE_MIN_WIDTH_MILLI) -> list[int]:
    """Horizontal rules on the page, as page-height milli of each rule's
    first pixel row: maximal blocks of pixel rows whose longest dark run
    spans at least `min_width_milli`/1000 of the width, and that are thin
    (<= RULE_MAX_THICKNESS_MILLI of the page height -- a table border, not
    a logo, a filled cell or a picture)."""
    h, w = gray.shape
    hits = longest_runs(gray < RULE_LEVEL) >= (w * min_width_milli) // 1000
    edges = np.diff(np.concatenate(([0], hits.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    thick = (h * RULE_MAX_THICKNESS_MILLI) // 1000
    return [int(1000 * s) // h for s, e in zip(starts, ends, strict=True) if e - s <= thick]


def load_gray(png: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(png) as im:
        return np.asarray(im.convert("L"))


# -- the PDF text layer --------------------------------------------------------

_PAGE = re.compile(r'<page width="([\d.]+)" height="([\d.]+)">')
_WORD = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>'
)


@dataclass(frozen=True)
class LayerWord:
    text: str
    box: tuple[int, int, int, int]  # page milli (x0, y0, x1, y1)


def parse_bbox_html(doc: str) -> list[LayerWord]:
    page = _PAGE.search(doc)
    if page is None:
        raise ValueError("pdftotext -bbox output has no <page> element")
    w, h = float(page.group(1)), float(page.group(2))
    out = []
    for m in _WORD.finditer(doc):
        x0, y0, x1, y1 = (float(m.group(k)) for k in range(1, 5))
        box = (int(1000 * x0 / w), int(1000 * y0 / h), int(1000 * x1 / w), int(1000 * y1 / h))
        out.append(LayerWord(html.unescape(m.group(5)), box))
    return out


def text_layer(pdf: Path, page_index: int) -> list[LayerWord]:
    """Words of one PDF page (`pdftotext -bbox`, 1-based page numbers)."""
    n = str(page_index + 1)
    result = subprocess.run(
        ["pdftotext", "-q", "-f", n, "-l", n, "-bbox", str(pdf), "-"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return parse_bbox_html(result.stdout)


# -- page edges: running page number, first/last body line ---------------------

#: "N of M" separators and "Page"-words of the corpus languages (the G-C
#: trial saw: 34/34, - N -, Seite 2 von 2, Pag. 1 di 2, 1 | Strona, Page 5 of 25)
_SEP = r"(?:/|of|von|z|di|de|from|od|sur|van|av|из)"
_PAGE_WORD = (
    r"(?:page|pages|seite|strona|pag\.?|pág\.?|página|pagina|p\.|str\.?|strana|stránka"
    r"|side|sida|blad|lap|sivu|oldal|стр\.?)"
)
#: a whole line that is nothing but a page number
PAGE_NUMBER_LINE = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\d{1,4}",
        rf"\d{{1,4}}\s*{_SEP}\s*\d{{1,4}}",
        r"[-–—]\s*\d{1,4}\s*[-–—]",
        rf"{_PAGE_WORD}\s*\d{{1,4}}(?:\s*{_SEP}\s*\d{{1,4}})?",
        r"\d{1,4}\s*\|\s*\w+",
        r"\w+\s*\|\s*\d{1,4}",
        r"[ivxlc]{1,7}",
    )
)
#: an explicit "Page N of M" anywhere inside a line
PAGE_NUMBER_IN_LINE = re.compile(
    rf"\b{_PAGE_WORD}\s*\d{{1,4}}\s*{_SEP}\s*\d{{1,4}}\b", re.IGNORECASE
)


@dataclass(frozen=True)
class LayerLine:
    text: str
    top: int  # page milli
    bottom: int


def layer_lines(layer: Sequence[LayerWord]) -> list[LayerLine]:
    """Text-layer words grouped into visual lines: a word joins the first
    line (top to bottom) whose vertical extent overlaps its own -- so a
    raised footnote mark stays on its line -- and a line reads left to
    right."""
    groups: list[list[Any]] = []  # [top, bottom, words]
    for word in sorted(layer, key=lambda lw: (lw.box[1], lw.box[0])):
        for group in groups:
            if word.box[1] < group[1] and word.box[3] > group[0]:
                group[0], group[1] = min(group[0], word.box[1]), max(group[1], word.box[3])
                group[2].append(word)
                break
        else:
            groups.append([word.box[1], word.box[3], [word]])
    lines = [
        LayerLine(" ".join(lw.text for lw in sorted(g[2], key=lambda lw: lw.box[0])), g[0], g[1])
        for g in groups
    ]
    return sorted(lines, key=lambda ln: (ln.top, ln.bottom))


def _is_page_number_line(text: str) -> bool:
    return any(p.fullmatch(text.strip()) for p in PAGE_NUMBER_LINE)


def page_number_lines(lines: Sequence[LayerLine]) -> list[str]:
    """Lines that show a running page number: the page's first or last
    line when it is nothing but a page number, or one of the first/last
    two lines when it contains an explicit "Page N of M"."""
    if not lines:
        return []
    edge = [lines[0], lines[-1]]
    out: list[str] = []
    for line in (*edge, *lines[:2], *lines[-2:]):
        text = line.text.strip()
        hit = (line in edge and _is_page_number_line(text)) or PAGE_NUMBER_IN_LINE.search(text)
        if hit and text not in out:
            out.append(text)
    return out


def _alnum(text: str) -> str:
    return "".join(ch for ch in _norm(text) if ch.isalnum())


def page_number_missing(lines: Sequence[LayerLine], facts: TargetFacts) -> int:
    """1 when the page shows a running page number (`page_number_lines`)
    while the target has no `FIELD` in a header/footer and no story text
    spelling that number -- the [pagenum] class (DEC-033 §3: visible page
    furniture belongs to the page). Body text does not count: a stray "7"
    in a paragraph is not the running footer."""
    if facts.hf_field:
        return 0
    story = _alnum(facts.story_text)
    return int(any(_alnum(c) and _alnum(c) not in story for c in page_number_lines(lines)))


def edge_line_coverage(lines: Sequence[LayerLine], facts: TargetFacts) -> int:
    """Page milli of the worse-covered of the page's first and last BODY
    line: the share of the line's adjacent word pairs (`words_of`) the
    target also carries adjacently; a one-word line counts the word
    itself, and a line of >= 3 words that all occur in the target counts as
    covered (one visual line across table cells or text columns). A body
    line skips page numbers, lines without a counted word,
    and lines whose every word is in the target's story text (furniture
    the target has); leading pure-number words (list labels, which the
    target carries as numbering, not text) are dropped. 1000 = no body
    line or both fully covered. The [boundary] class: the top or bottom
    piece of a paragraph split by the page break is missing -- a loss a
    whole-page ratio (`visible_missing_milli`) dilutes below any usable
    threshold."""
    return min((milli for _line, milli in edge_lines(lines, facts)), default=1000)


def edge_lines(lines: Sequence[LayerLine], facts: TargetFacts) -> list[tuple[LayerLine, int]]:
    """The page's first and last body line (`edge_line_coverage`), each
    with its coverage in milli -- the lines a G-C reviewer checks against
    the page image when `edge_lines` fires. A line of at least
    `EDGE_SPELLED_MIN_CHARS` letters/digits that the target spells
    contiguously is covered however the text layer split it into words:
    small caps paint a capital and a smaller rest ("U NIVERSITÀ"), an
    ordinal suffix is raised ("26 th") -- the 1 false alarm of the 40-hit
    G-C sample, 7 of 344 hits on the 200-document sample (2026-09-30)."""
    story_words = set(words_of(facts.story_text))
    body = []
    for line in lines:
        words = words_of(line.text)
        if _is_page_number_line(line.text) or not words or all(w in story_words for w in words):
            continue
        while words and words[0].isdigit():
            words = words[1:]
        if words:
            body.append((line, words))
    out = []
    for line, words in body[:1] + body[-1:]:
        spelled = _alnum(line.text)
        if len(spelled) >= EDGE_SPELLED_MIN_CHARS and spelled in facts.alnum:
            out.append((line, 1000))
        elif len(words) >= 2:
            pairs = list(zip(words, words[1:], strict=False))
            paired = 1000 * sum(p in facts.word_pairs for p in pairs) // len(pairs)
            # a visual line that joins table cells or text columns: its words
            # are all in the target, just not adjacent (v16 corpus check)
            every_word = len(words) >= 3 and all(w in facts.words for w in words)
            out.append((line, 1000 if every_word else paired))
        else:
            out.append((line, 1000 * int(words[0] in facts.words)))
    return out


# -- the three measurements ------------------------------------------------------


def _located(words: Iterable[str], layer: Sequence[LayerWord]) -> list[LayerWord]:
    wanted = set(words)
    return [lw for lw in layer if wanted.intersection(words_of(lw.text))]


def _distinctive(words: Sequence[str], layer: Sequence[LayerWord]) -> list[LayerWord]:
    """Layer boxes of the words that occur exactly once on the page (a
    common word like "the" occurs everywhere and locates nothing)."""
    counts = Counter(w for lw in layer for w in words_of(lw.text))
    return _located([w for w in words if counts[w] == 1], layer)


def row_bottom(words: Sequence[str], layer: Sequence[LayerWord]) -> int:
    """Page milli of the bottom of a target row located in the text layer:
    the lowest distinctive box; without one, the median bottom of all
    matches; 0 when nothing matches."""
    unique = _distinctive(words, layer)
    if unique:
        return max(lw.box[3] for lw in unique)
    bottoms = sorted(lw.box[3] for lw in _located(words, layer))
    return bottoms[len(bottoms) // 2] if bottoms else 0


def text_top_below(words: Sequence[str], layer: Sequence[LayerWord], top: int) -> int:
    """Page milli of the top of the first target text after a table that
    lies below `top`: distinctive boxes first, all matches as a fallback,
    the page bottom (1000) when nothing matches below."""
    for boxes in (_distinctive(words, layer), _located(words, layer)):
        below = [lw.box[1] for lw in boxes if lw.box[1] > top]
        if below:
            return min(below)
    return 1000


def tail_excess(
    tail: TableTail, layer: Sequence[LayerWord], rules: Sequence[int], min_row_milli: int
) -> tuple[int, int, int]:
    """(rows the target carries beyond what the page shows, rules between,
    gap milli) for one table tail. The window runs from the bottom of the
    last row with text (`row_bottom`) to the first located text after the
    table below it (the page bottom when there is none)."""
    top = row_bottom(tail.last_row_words, layer)
    bottom = text_top_below(tail.next_words, layer, top)
    between = sum(1 for y in rules if top < y < bottom)
    visible = between - 1 if between >= 2 else (bottom - top) // max(min_row_milli, 1)
    return tail.trailing_textless_rows - visible, between, bottom - top


def measure(
    facts: TargetFacts,
    gray: np.ndarray,
    layer: Sequence[LayerWord],
    *,
    min_row_milli: int,
) -> dict[str, int]:
    """Every G-B value for one page (see module docstring)."""
    rules = rule_positions(gray)
    worst = (0, 0, 0, 0)  # excess, trailing, rules between, gap
    for tail in facts.tails:
        excess, between, gap = tail_excess(tail, layer, rules, min_row_milli)
        if excess > worst[0]:
            worst = (excess, tail.trailing_textless_rows, between, gap)
    visible = Counter()
    for lw in layer:
        visible.update(chars_of(lw.text))
    n_visible, n_target = sum(visible.values()), sum(facts.chars.values())
    missing = sum((visible - facts.chars).values())
    unseen = sum((facts.chars - visible).values())
    ink = gray < INK_LEVEL
    h, w = ink.shape
    claimed = no_ink = 0
    for lw in layer:
        if not facts.words.intersection(words_of(lw.text)):
            continue
        x0, y0, x1, y1 = lw.box
        ys, ye = y0 * h // 1000, max(y1 * h // 1000, y0 * h // 1000 + 1)
        xs, xe = x0 * w // 1000, max(x1 * w // 1000, x0 * w // 1000 + 1)
        claimed += 1
        no_ink += int(not ink[ys:ye, xs:xe].any())
    lines = layer_lines(layer)
    return {
        "grid_excess_rows": worst[0],
        "grid_trailing_textless_rows": worst[1],
        "grid_rules_between": worst[2],
        "grid_gap_milli": worst[3],
        "image_rules": len(rules),
        "visible_chars": n_visible,
        "target_chars": n_target,
        "visible_missing_chars": missing,
        "visible_missing_milli": (1000 * missing) // max(n_visible, 1),
        "target_unseen_chars": unseen,
        "target_unseen_milli": (1000 * unseen) // max(n_target, 1),
        "claimed_words": claimed,
        "claimed_words_without_ink": no_ink,
        "page_number_lines": len(page_number_lines(lines)),
        "page_number_missing": page_number_missing(lines, facts),
        "edge_line_coverage_milli": edge_line_coverage(lines, facts),
    }


GB_CHECKS: tuple[str, ...] = (
    "grid_rows", "visible_text", "target_text", "text_ink", "page_number", "edge_lines",
)  # fmt: skip


def flags(values: Mapping[str, int], policy: Mapping[str, Any]) -> list[str]:
    """Which G-B checks this page fails under `policy` (the policy file's
    `gb` block):

    * `grid_rows`: `grid_excess_rows >= min_excess_rows`;
    * `visible_text`: >= `min_chars` visible characters missing from the
      target AND more than `max_milli` of the visible ones;
    * `target_text`: the same for target characters the page lacks;
    * `text_ink`: more than `max_words_without_ink` claimed words without
      ink;
    * `page_number`: the page shows a running page number the target has
      neither as a header/footer `FIELD` nor as story text;
    * `edge_lines`: the page's first or last body line is covered below
      `min_coverage_milli`."""
    out = []
    if values["grid_excess_rows"] >= policy["grid_rows"]["min_excess_rows"]:
        out.append("grid_rows")
    rule = policy["visible_text"]
    if (
        values["visible_missing_chars"] >= rule["min_chars"]
        and values["visible_missing_milli"] > rule["max_milli"]
    ):
        out.append("visible_text")
    rule = policy["target_text"]
    if (
        values["target_unseen_chars"] >= rule["min_chars"]
        and values["target_unseen_milli"] > rule["max_milli"]
    ):
        out.append("target_text")
    if values["claimed_words_without_ink"] > policy["text_ink"]["max_words_without_ink"]:
        out.append("text_ink")
    if values["page_number_missing"] > 0:
        out.append("page_number")
    if values["edge_line_coverage_milli"] < policy["edge_lines"]["min_coverage_milli"]:
        out.append("edge_lines")
    return out

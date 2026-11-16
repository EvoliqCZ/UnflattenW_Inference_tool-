"""RFC003 §20 deterministic Canonical AST generator (word-core-v0, D2).

All randomness flows through `random.Random(identity.operation_seed(...))`
seeded per §20.3: `synthetic_root_key` (namespace, generator version,
recipe id+hash, root instance index) → one seed per named generative
operation. No global RNG, no wall clock, no uuid — same inputs are
byte-identical outputs, forever.

Construct-correct-by-design: the generator only ever assembles nodes that
the RFC001 schema + semantic validators accept. A final validation pass
runs as an *internal bug detector* — on any blocking issue it raises
`GeneratorBugError` (a generate-then-resample loop is forbidden by §20).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from unflatten.ast.schema_validator import validate_schema
from unflatten.ast.semantic_validator import validate_semantics
from unflatten.dataset import identity
from unflatten.dataset.synthetic.recipes import (
    DATASET_NAMESPACE,
    GENERATOR_VERSION,
    IMAGE_ROLE,
    RECIPE_COHORT_ID,
    Recipe,
)

#: semantic path (RFC001 Canonical Resource Root) of the one D2 image.
IMAGE_AST_PATH = "images/tiny_2x2.png"

#: RFC003 §20.3 named generative operations. Each is seeded independently
#: from the synthetic root key, so appending a name leaves the existing
#: streams byte-identical; renaming or reordering reseeds the corpus.
_OPERATION_IDS = ("structure", "content", "style", "table_plan", "anchor")

_WORDS = (
    "alpha", "bravo", "canonical", "delta", "document", "echo", "foxtrot",
    "grid", "hotel", "india", "juliet", "kilo", "layout", "margin", "node",
    "oscar", "paragraph", "quebec", "render", "section", "table", "uniform",
    "vector", "whiskey", "xray", "yankee", "zulu",
)

_FONT_SIZES_MPT = (9000, 10000, 11000, 12000, 14000)
_COLORS = ("#000000", "#1a1a1a", "#003366", "#660000", "#004400")
_ALIGNMENTS = ("left", "left", "justify", "center")

_DEFAULT_RUN_STYLE: dict[str, Any] = {
    "font_family": "Arial",
    "font_size_mpt": 11000,
    "bold": False,
    "italic": False,
    "underline": "none",
    "strike": False,
    "color": "#000000",
    "highlight": None,
    "superscript": False,
    "subscript": False,
    "character_spacing_mpt": 0,
    "small_caps": False,
    "all_caps": False,
}


class GeneratorBugError(RuntimeError):
    """The generator produced an invalid AST — an internal bug, never resampled."""


class _Ids:
    """Deterministic sequential node-id allocator (canonicalizer renumbers)."""

    def __init__(self) -> None:
        self._next = 0

    def new(self) -> str:
        self._next += 1
        return f"n{self._next}"


def generate_document(
    recipe: Recipe,
    root_instance_index: int,
    resource_bytes: Mapping[str, bytes] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Generate one Canonical AST + its resource files for `recipe`.

    Returns `(ast, resources)` where `resources` maps RFC001 resource
    paths to raw bytes (empty for resource-free recipes). `resource_bytes`
    maps recipe resource-input *roles* to bytes; each entry is verified
    against the recipe's pinned sha256/size (mismatch is a `ValueError` —
    the recipe identity would no longer describe the output).
    """
    resource_bytes = dict(resource_bytes or {})
    _check_resource_bytes(recipe, resource_bytes)

    root_key = identity.synthetic_root_key(
        dataset_namespace=DATASET_NAMESPACE,
        generator_version=GENERATOR_VERSION,
        recipe_id=recipe.recipe_id,
        recipe_hash=recipe.hash(),
        root_instance_index=root_instance_index,
    )
    rng = {
        op: random.Random(
            identity.operation_seed(
                synthetic_root_key=root_key,
                recipe_cohort_id=RECIPE_COHORT_ID,
                operation_id=op,
                variant_index=0,
            )
        )
        for op in _OPERATION_IDS
    }

    ids = _Ids()
    layout_kind = recipe.body["layout_kind"]
    builder = _BUILDERS.get(layout_kind)
    if builder is None:
        raise ValueError(f"unknown layout_kind {layout_kind!r} in recipe {recipe.recipe_id!r}")
    built = builder(recipe, ids, rng, resource_bytes)

    ast: dict[str, Any] = {
        "ast_version": "0.1.0",
        "profile": recipe.body.get("profile", "word-core-v0"),
        "document_id": f"synthetic:{recipe.recipe_id}:{root_instance_index}",
        "metadata": {"language": "en-US", "title": None, "subject": None},
        "resources": built.resources_decl,
        "sections": built.sections,
        "extensions": {},
    }
    # Root note stories are added ONLY when non-empty: a present-but-empty
    # key is a different canonical document and would move every ast_hash.
    if built.footnotes:
        ast["footnotes"] = list(built.footnotes)
    if built.endnotes:
        ast["endnotes"] = list(built.endnotes)

    blocking = [
        issue
        for issue in (validate_schema(ast) + validate_semantics(ast))
        if issue.severity in ("error", "fatal")
    ]
    if blocking:
        raise GeneratorBugError(
            f"recipe {recipe.recipe_id!r} index {root_instance_index} produced an "
            f"invalid AST (internal bug, not resampled): "
            + "; ".join(f"{i.code}: {i.message}" for i in blocking)
        )
    return ast, built.resources_out


def _check_resource_bytes(recipe: Recipe, resource_bytes: dict[str, bytes]) -> None:
    for record in recipe.resource_inputs:
        role = record["role"]
        data = resource_bytes.get(role)
        if data is None:
            raise ValueError(f"recipe {recipe.recipe_id!r} requires resource bytes for {role!r}")
        actual = f"sha256:{hashlib.sha256(data).hexdigest()}"
        if actual != record["sha256"] or len(data) != record["size_bytes"]:
            raise ValueError(
                f"resource bytes for role {role!r} do not match the recipe's pinned "
                f"identity ({record['sha256']}/{record['size_bytes']}B)"
            )


# ---------------------------------------------------------------- content --


def _sentence(content: random.Random) -> str:
    words = [content.choice(_WORDS) for _ in range(content.randint(4, 9))]
    return words[0].capitalize() + " " + " ".join(words[1:]) + "."


def _text(content: random.Random, sentences: int) -> str:
    return " ".join(_sentence(content) for _ in range(sentences))


def _span(recipe: Recipe, structure: random.Random, key: str) -> int:
    return structure.randint(recipe.body[f"{key}_min"], recipe.body[f"{key}_max"])


# ------------------------------------------------------------ node helpers --


def _run(ids: _Ids, text: str, **style_overrides: Any) -> dict[str, Any]:
    return {
        "id": ids.new(),
        "type": "run",
        "text": text,
        "language": "en-US",
        "style": {**_DEFAULT_RUN_STYLE, **style_overrides},
    }


def _paragraph(
    ids: _Ids, children: list[dict[str, Any]], *, alignment: str = "left"
) -> dict[str, Any]:
    return {
        "id": ids.new(),
        "type": "paragraph",
        "semantic": {"role": "body", "heading_level": None, "outline_level": None},
        "layout": {
            "alignment": alignment,
            "direction": "ltr",
            "indent": {"left_mpt": 0, "right_mpt": 0, "first_line_mpt": 0, "hanging_mpt": 0},
            "spacing": {"before_mpt": 0, "after_mpt": 0, "line": {"mode": "single"}},
            "pagination": {
                "keep_with_next": False,
                "keep_lines": False,
                "page_break_before": False,
                "widow_control": True,
            },
            "tabs": [],
        },
        "style": {"background": None, "borders": None},
        "children": children,
    }


#: A4 portrait, the only page size the word-core-v0 recipes use.
_PAGE_WIDTH_MPT = 595276
_PAGE_HEIGHT_MPT = 841890


def _section(
    ids: _Ids,
    children: list[dict[str, Any]],
    *,
    column_count: int = 1,
    equal_width: bool = True,
    widths_mpt: list[int] | None = None,
    gap_mpt: int = 18000,
    break_type: str = "next_page",
    orientation: str = "portrait",
    headers: list[dict[str, Any]] | None = None,
    footers: list[dict[str, Any]] | None = None,
    margins_extra: dict[str, int] | None = None,
    vertical_alignment: str | None = None,
    page_numbering: dict[str, Any] | None = None,
    line_numbering: dict[str, Any] | None = None,
    page_borders: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One section. Every keyword default is the word-core-v0 constant this
    helper used before it was widened, so existing callers are byte-identical;
    the optional stories/decor keys are absent unless asked for."""
    landscape = orientation == "landscape"
    section: dict[str, Any] = {
        "id": ids.new(),
        "type": "section",
        "break_type": break_type,
        "page": {
            "width_mpt": _PAGE_HEIGHT_MPT if landscape else _PAGE_WIDTH_MPT,
            "height_mpt": _PAGE_WIDTH_MPT if landscape else _PAGE_HEIGHT_MPT,
            "orientation": orientation,
        },
        "margins": {
            "top_mpt": 72000,
            "right_mpt": 72000,
            "bottom_mpt": 72000,
            "left_mpt": 72000,
            "gutter_mpt": 0,
            **(margins_extra or {}),
        },
        "columns": {
            "count": column_count,
            "equal_width": equal_width,
            "gap_mpt": gap_mpt,
            "widths_mpt": list(widths_mpt or []),
        },
        "children": children,
    }
    optional = {
        "headers": headers,
        "footers": footers,
        "vertical_alignment": vertical_alignment,
        "page_numbering": page_numbering,
        "line_numbering": line_numbering,
        "page_borders": page_borders,
    }
    section.update({k: v for k, v in optional.items() if v is not None})
    return section


def _text_paragraph(
    ids: _Ids, content: random.Random, sentences: int, *, alignment: str = "left"
) -> dict[str, Any]:
    return _paragraph(ids, [_run(ids, _text(content, sentences))], alignment=alignment)


def _cell(
    ids: _Ids,
    content: random.Random,
    grid_column: int,
    *,
    col_span: int = 1,
    row_span: int = 1,
    width_mpt: int = 100000,
    padding_mpt: int = 1000,
    vertical_alignment: str = "top",
    background: str | None = None,
    borders: dict[str, Any] | None = None,
    children: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One cell. `text_direction` is deliberately PINNED to 'horizontal': the
    renderer never emits w:textDirection, so varying it would create a label
    with no pixel evidence behind it."""
    return {
        "id": ids.new(),
        "type": "cell",
        "grid_column": grid_column,
        "col_span": col_span,
        "row_span": row_span,
        "layout": {
            "width_mpt": width_mpt,
            "padding": {
                "top_mpt": padding_mpt,
                "right_mpt": padding_mpt,
                "bottom_mpt": padding_mpt,
                "left_mpt": padding_mpt,
            },
            "vertical_alignment": vertical_alignment,
            "text_direction": "horizontal",
        },
        "style": {"background": background, "borders": borders},
        "children": (
            children if children is not None else [_text_paragraph(ids, content, 1)]
        ),
    }


def _row(
    ids: _Ids,
    cells: list[dict[str, Any]],
    *,
    height_mode: str = "auto",
    height_mpt: int = 0,
    repeat_header: bool = False,
    cant_split: bool = False,
) -> dict[str, Any]:
    return {
        "id": ids.new(),
        "type": "row",
        "layout": {
            "height": {"mode": height_mode, "value_mpt": height_mpt},
            "repeat_header": repeat_header,
            "cant_split": cant_split,
        },
        "cells": cells,
    }


def _table(
    ids: _Ids,
    columns: int,
    rows: list[dict[str, Any]],
    *,
    column_widths: list[int] | None = None,
    width: dict[str, Any] | None = None,
    alignment: str = "left",
    indent_mpt: int = 0,
    autofit: bool = True,
    background: str | None = None,
    borders: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": ids.new(),
        "type": "table",
        "layout": {
            "alignment": alignment,
            "indent_mpt": indent_mpt,
            "width": dict(width) if width is not None else {"mode": "auto", "value_mpt": 0},
            "autofit": autofit,
        },
        "columns": (
            [{"width_mpt": w} for w in column_widths]
            if column_widths is not None
            else [{"width_mpt": 100000} for _ in range(columns)]
        ),
        "style": {"background": background, "borders": borders},
        "rows": rows,
    }


def _list(
    ids: _Ids,
    kind: str,
    items: list[dict[str, Any]],
    *,
    indent_mpt: int = 36000,
) -> dict[str, Any]:
    marker = (
        {"format": "decimal", "start": 1, "suffix": "."}
        if kind == "ordered"
        else {"format": "bullet", "start": None, "suffix": None}
    )
    return {
        "id": ids.new(),
        "type": "list",
        "list_kind": kind,
        "marker": marker,
        "layout": {"left_indent_mpt": indent_mpt, "hanging_mpt": 18000},
        "items": items,
    }


def _list_item(ids: _Ids, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": ids.new(), "type": "list_item", "children": children}


# --------------------------------------------------------- tier-D helpers --

#: A4 portrait minus the 72000 mpt margins every section uses — the width a
#: full-bleed table or body paragraph has to fill.
_USABLE_WIDTH_MPT = _PAGE_WIDTH_MPT - 2 * 72000

#: Distinct integer weights, so a multi-column split never yields two equal
#: column widths (a constant is coverage without variation).
_COLUMN_WEIGHTS = (3, 4, 5, 6, 7, 8)

#: Mean advance of one character plus its trailing space at the 11000 mpt
#: default run size. It sizes filler text only — nothing derived from it
#: ever enters the AST as a length.
_FILL_MPT_PER_CHAR = 5200

_BORDER_STYLES = ("single", "double", "dotted", "dashed")
_BORDER_WIDTHS_MPT = (4000, 6000, 8000, 12000)
_SHADE_COLORS = ("#f2f2f2", "#e6eef7", "#fdf1e0", "#eaf3ea")
_VERTICAL_ALIGNMENTS = ("top", "center", "bottom")
_BORDER_SIDES = ("top", "right", "bottom", "left")


def _split_widths(total_mpt: int, count: int, table_rng: random.Random) -> list[int]:
    """Split `total_mpt` into `count` distinct positive integer widths.

    Integer arithmetic only: one `/` here puts a float in the tree, which the
    schema validator rejects inside `generate_document` (and which
    `canonicalizer._assert_no_floats` would otherwise raise on). The division
    remainder goes to the FIRST column by a fixed rule, so the widths sum to
    `total_mpt` exactly.
    """
    if not 1 <= count <= len(_COLUMN_WEIGHTS):
        raise ValueError(f"column count {count} outside 1..{len(_COLUMN_WEIGHTS)}")
    weights = table_rng.sample(_COLUMN_WEIGHTS, count)
    total_weight = sum(weights)
    widths = [total_mpt * weight // total_weight for weight in weights]
    widths[0] += total_mpt - sum(widths)
    return widths


def _plan_grid(
    table_rng: random.Random, rows: int, columns: int, *, span_plan: str
) -> list[list[tuple[int, int, int]]]:
    """Plan a `rows` x `columns` occupancy CONSTRUCTIVELY — never
    sample-and-retry (the module contract at the top of this file).

    Returns one `(grid_column, col_span, row_span)` list per row. Each row is
    filled left to right across the columns not already covered by an earlier
    row's row_span, so every row band tiles the grid exactly and no
    continuation cell is ever authored for a row_span origin.
    """
    allow_col_span = span_plan in ("colspan", "colspan_rowspan")
    allow_row_span = span_plan == "colspan_rowspan"
    carried: dict[int, int] = {}
    plan: list[list[tuple[int, int, int]]] = []
    for row_index in range(rows):
        cells: list[tuple[int, int, int]] = []
        column = 0
        while column < columns:
            if carried.get(column, 0) > 0:
                column += 1
                continue
            free = column
            while free < columns and carried.get(free, 0) == 0:
                free += 1
            col_span = table_rng.randint(1, free - column) if allow_col_span else 1
            rows_left = rows - row_index
            row_span = 1
            if (
                allow_row_span
                and col_span == 1
                and rows_left > 1
                and table_rng.randint(0, 2) == 0
            ):
                row_span = table_rng.randint(2, rows_left)
            cells.append((column, col_span, row_span))
            if row_span > 1:
                for covered in range(column, column + col_span):
                    carried[covered] = row_span
            column += col_span
        plan.append(cells)
        carried = {c: left - 1 for c, left in carried.items() if left - 1 > 0}
    return plan


def _text_of_width(content: random.Random, target_chars: int) -> str:
    """Words up to `target_chars` including the trailing '.', NEVER past it.

    MEASURED, and the reason this stops short instead of overshooting: the
    old loop appended one more word AFTER the budget was spent, so a cell
    whose target was 12 characters routinely rendered ~18 and wrapped onto a
    second line. A cell two lines high next to one-line neighbours makes
    pdftotext read that row band column-major instead of row-major, and the
    table's own text then fails the strict synthetic observation gate —
    `d-table-spans-styled` scored 6/22 that way. Never exceeding the budget
    is what keeps a cell one line high.
    """
    words = [content.choice(_WORDS)]
    length = len(words[0]) + 1
    while True:
        word = content.choice(_WORDS)
        if length + len(word) + 1 > target_chars:
            break
        words.append(word)
        length += len(word) + 1
    return words[0].capitalize() + "".join(" " + w for w in words[1:]) + "."


def _fill_paragraph(
    ids: _Ids,
    content: random.Random,
    width_mpt: int,
    fill_ppm: int,
    *,
    alignment: str = "left",
) -> dict[str, Any]:
    """A paragraph whose text is sized to `fill_ppm` of `width_mpt`.

    Measured: pdftotext's reading order is an emergent property of the whole
    page, so a table surrounded by FULL-WIDTH body paragraphs stays row-major
    while the same table between short ones does not.
    """
    target_chars = max(4, (width_mpt * fill_ppm) // (1_000_000 * _FILL_MPT_PER_CHAR))
    return _paragraph(
        ids, [_run(ids, _text_of_width(content, target_chars))], alignment=alignment
    )


def _borders(style: random.Random) -> dict[str, Any]:
    return {
        side: {
            "style": style.choice(_BORDER_STYLES),
            "width_mpt": style.choice(_BORDER_WIDTHS_MPT),
            "color": style.choice(_COLORS),
        }
        for side in _BORDER_SIDES
    }


# --------------------------------------------------------- tier-F helpers --

#: Anchor vocabulary. Every one of these is a TUPLE and is only ever indexed
#: or passed to `choice`: iterating a set here would make the generated bytes
#: depend on PYTHONHASHSEED (guarded by the cross-process determinism test).
_ANCHOR_H_RELATIVE = ("margin", "column", "page")
_ANCHOR_H_ALIGN = ("left", "right", "center")
_ANCHOR_V_RELATIVE = ("margin", "paragraph", "page")
_ANCHOR_V_ALIGN = ("top", "bottom", "center")
_WRAP_SIDES = ("both_sides", "left", "right", "largest")
#: wrapTight / wrapThrough are deliberately absent: both require a
#: wp:wrapPolygon, i.e. pixel geometry, which RFC001 §0.1 keeps out of the AST.
_WRAP_MODES = ("square", "none", "top_and_bottom")

#: `(horizontal mode, vertical mode)`. The first two entries are
#: COMPLEMENTARY: a document that emits both of them covers each `oneOf`
#: branch on each axis BY CONSTRUCTION, not by a lucky draw over indices.
_ANCHOR_MODE_PAIRS = (
    ("align", "offset"),
    ("offset", "align"),
    ("align", "align"),
    ("offset", "offset"),
)

_SECTION_BREAK_TYPES = ("next_page", "continuous", "even_page", "odd_page", "next_column")
_SECTION_VERTICAL_ALIGNMENTS = ("top", "center", "both", "bottom")
_PAGE_NUMBER_FORMATS = ("decimal", "lower_roman", "upper_roman")
_PAGE_BORDER_DISPLAYS = ("all_pages", "first_page", "not_first_page")
_LINE_NUMBER_RESTARTS = ("continuous", "new_page", "new_section")

#: MEASURED, and the reason this is a constant instead of a range: with
#: `count_by` 5 the printed line number lands INSIDE a wrapped paragraph's
#: word stream and the observation aligner reports
#: OBSERVATION_ALIGNMENT_MISMATCH. Pinned to 1, with one-line paragraphs.
_LINE_NUMBERING_COUNT_BY = 1

#: MEASURED, and the reason the frame recipe does not draw its wrap mode: see
#: `_build_text_frames`. `none` and `square` both leave the frame sitting in
#: the body's own space.
_FRAME_WRAP_MODE = "top_and_bottom"


def _anchor(
    anchor_rng: random.Random,
    *,
    h_mode: str,
    v_mode: str,
    wrap_mode: str,
    behind_text: bool,
    allow_overlap: bool,
    z_order: int,
) -> dict[str, Any]:
    """One `anchor`. `mode` and its dependent key are emitted TOGETHER — the
    schema's two `oneOf` branches forbid `align` and `offset_mpt` coexisting,
    and a leaf-only edit of `mode` is a defect this repo has already seen."""
    horizontal: dict[str, Any] = {
        "relative_from": anchor_rng.choice(_ANCHOR_H_RELATIVE),
        "mode": h_mode,
    }
    if h_mode == "align":
        horizontal["align"] = anchor_rng.choice(_ANCHOR_H_ALIGN)
    else:
        horizontal["offset_mpt"] = anchor_rng.randrange(9000, 90001, 1000)
    vertical: dict[str, Any] = {
        "relative_from": anchor_rng.choice(_ANCHOR_V_RELATIVE),
        "mode": v_mode,
    }
    if v_mode == "align":
        vertical["align"] = anchor_rng.choice(_ANCHOR_V_ALIGN)
    else:
        vertical["offset_mpt"] = anchor_rng.randrange(9000, 120001, 1000)
    wrap: dict[str, Any] = {"mode": wrap_mode}
    if wrap_mode == "square":
        wrap["side"] = anchor_rng.choice(_WRAP_SIDES)
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "wrap": wrap,
        "behind_text": behind_text,
        "allow_overlap": allow_overlap,
        "z_order": z_order,
    }


def _floating_image(
    ids: _Ids,
    resource_ref: str,
    anchor: dict[str, Any],
    *,
    width_mpt: int,
    height_mpt: int,
    crop: dict[str, int],
    rotation_mdeg: int,
) -> dict[str, Any]:
    return {
        "id": ids.new(),
        "type": "floating_image",
        "resource_ref": resource_ref,
        "size": {"width_mpt": width_mpt, "height_mpt": height_mpt},
        "crop": crop,
        "rotation_mdeg": rotation_mdeg,
        "alt_text": "Synthetic floating figure",
        "anchor": anchor,
    }


def _text_frame(
    ids: _Ids, anchor: dict[str, Any], children: list[dict[str, Any]]
) -> dict[str, Any]:
    """An anchored block container. Callers must pass PARAGRAPHS only — see
    `_build_text_frames` for the measured reason a list may never go in."""
    return {"id": ids.new(), "type": "text_frame", "anchor": anchor, "children": children}


def _story(
    ids: _Ids, story_type: str, kind: str, children: list[dict[str, Any]]
) -> dict[str, Any]:
    return {"id": ids.new(), "type": story_type, "kind": kind, "children": children}


def _note(ids: _Ids, note_type: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": ids.new(), "type": note_type, "children": children}


def _note_ref(ids: _Ids, note_id: str, marker_text: str) -> dict[str, Any]:
    """`marker_text` is the visible mark AS RENDERED (schema $comment), so it
    carries no padding: a trailing space would write a pdftotext artefact into
    a supervised label."""
    return {
        "id": ids.new(),
        "type": "note_ref",
        "note_id": note_id,
        "marker_text": marker_text,
    }


def _field(ids: _Ids, field_type: str) -> dict[str, Any]:
    return {"id": ids.new(), "type": "field", "field_type": field_type}


def _hyperlink(
    ids: _Ids, kind: str, uri: str, children: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "id": ids.new(),
        "type": "hyperlink",
        "target": {"kind": kind, "uri": uri},
        "children": children,
    }


def _image_resource(recipe: Recipe) -> tuple[str, list[dict[str, Any]]]:
    """`(resource_id, resources_decl)` for the recipe's one PNG input."""
    record = next(r for r in recipe.resource_inputs if r["role"] == IMAGE_ROLE)
    raw_hex = record["sha256"].removeprefix("sha256:")
    resource_id = f"res_{raw_hex[:12]}"
    return resource_id, [
        {
            "id": resource_id,
            "kind": "image",
            "mime_type": record["mime_type"],
            "sha256": raw_hex,
            "path": IMAGE_AST_PATH,
            "pixel_width": recipe.body["image_pixel_width"],
            "pixel_height": recipe.body["image_pixel_height"],
        }
    ]


# ---------------------------------------------------------- recipe layouts --

@dataclass(frozen=True)
class _Built:
    """What one layout builder produces.

    `footnotes`/`endnotes` are the RFC001 root note stories; they stay empty
    for every word-core-v0 layout and reach the AST only when non-empty.
    """

    sections: list[dict[str, Any]]
    resources_decl: list[dict[str, Any]]
    resources_out: dict[str, bytes]
    footnotes: tuple[dict[str, Any], ...] = ()
    endnotes: tuple[dict[str, Any], ...] = ()


def _build_plain(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    paragraphs = [
        _text_paragraph(
            ids,
            rng["content"],
            _span(recipe, rng["structure"], "sentences"),
            alignment=rng["style"].choice(_ALIGNMENTS),
        )
        for _ in range(_span(recipe, rng["structure"], "paragraph_count"))
    ]
    return _Built([_section(ids, paragraphs)], [], {})


def _styled_runs(
    ids: _Ids, content: random.Random, style: random.Random, count: int
) -> list[dict[str, Any]]:
    """`count` runs where adjacent styles always differ (construct-by-design:
    the canonicalizer would merge equal-styled neighbours, changing shape)."""
    runs: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for _ in range(count):
        overrides: dict[str, Any] = {
            "bold": style.random() < 0.4,
            "italic": style.random() < 0.3,
            "underline": style.choice(("none", "none", "single")),
            "color": style.choice(_COLORS),
            "font_size_mpt": style.choice(_FONT_SIZES_MPT),
        }
        if overrides == previous:
            overrides["bold"] = not overrides["bold"]
        previous = overrides
        runs.append(_run(ids, _sentence(content) + " ", **overrides))
    return runs


def _build_mixed_runs(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    paragraphs = [
        _paragraph(
            ids,
            _styled_runs(
                ids, rng["content"], rng["style"], _span(recipe, rng["structure"], "runs")
            ),
        )
        for _ in range(_span(recipe, rng["structure"], "paragraph_count"))
    ]
    return _Built([_section(ids, paragraphs)], [], {})


def _build_lists(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    structure, content = rng["structure"], rng["content"]
    ordered_items = [
        _list_item(ids, [_text_paragraph(ids, content, 1)])
        for _ in range(_span(recipe, structure, "ordered_items"))
    ]
    nested = _list(
        ids,
        "unordered",
        [
            _list_item(ids, [_text_paragraph(ids, content, 1)])
            for _ in range(_span(recipe, structure, "nested_items"))
        ],
        indent_mpt=72000,
    )
    ordered_items[-1]["children"].append(nested)
    unordered = _list(
        ids,
        "unordered",
        [
            _list_item(ids, [_text_paragraph(ids, content, 1)])
            for _ in range(_span(recipe, structure, "unordered_items"))
        ],
    )
    children = [
        _text_paragraph(ids, content, 1),
        _list(ids, "ordered", ordered_items),
        unordered,
    ]
    return _Built([_section(ids, children)], [], {})


def _build_tables(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    structure, content = rng["structure"], rng["content"]
    cols = _span(recipe, structure, "cols")
    rows = [
        _row(ids, [_cell(ids, content, gc) for gc in range(cols)])
        for _ in range(_span(recipe, structure, "rows"))
    ]
    children: list[dict[str, Any]] = [
        _text_paragraph(ids, content, 1),
        _table(ids, cols, rows),
    ]
    if recipe.body["include_span_table"]:
        # Fixed valid span layout over a 3-column grid (mirrors the RFC001
        # golden corpus case): row1 = [colspan-2, rowspan-2], row2 covers
        # the remaining two grid columns.
        span_rows = [
            _row(
                ids,
                [
                    _cell(ids, content, 0, col_span=2),
                    _cell(ids, content, 2, row_span=2),
                ],
            ),
            _row(ids, [_cell(ids, content, 0), _cell(ids, content, 1)]),
        ]
        children.append(_text_paragraph(ids, content, 1))
        children.append(_table(ids, 3, span_rows))
    return _Built([_section(ids, children)], [], {})


def _build_sections_columns(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    structure, content = rng["structure"], rng["content"]
    two_col_index = recipe.body["two_column_section_index"]
    sections = []
    for section_index in range(2):
        paragraphs = [
            _text_paragraph(ids, content, 1)
            for _ in range(_span(recipe, structure, "paragraphs_per_section"))
        ]
        columns = 2 if section_index == two_col_index else 1
        sections.append(_section(ids, paragraphs, column_count=columns))
    return _Built(sections, [], {})


def _build_column_breaks(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    """One multi-column section whose columns are separated by explicit
    `column_break` nodes (RFC001 §42.1 ColumnBreak) — never leading,
    trailing or adjacent, so every break separates real content."""
    structure, content = rng["structure"], rng["content"]
    column_count = recipe.body["column_count"]
    children: list[dict[str, Any]] = []
    for column_index in range(column_count):
        if column_index:
            children.append({"id": ids.new(), "type": "column_break"})
        children.extend(
            _text_paragraph(ids, content, 1)
            for _ in range(_span(recipe, structure, "paragraphs_per_column"))
        )
    return _Built([_section(ids, children, column_count=column_count)], [], {})


def _build_inline_image(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], resource_bytes: dict[str, bytes]
) -> _Built:
    structure, content = rng["structure"], rng["content"]
    record = next(r for r in recipe.resource_inputs if r["role"] == IMAGE_ROLE)
    raw_hex = record["sha256"].removeprefix("sha256:")
    resource_id = f"res_{raw_hex[:12]}"
    resources_decl = [
        {
            "id": resource_id,
            "kind": "image",
            "mime_type": record["mime_type"],
            "sha256": raw_hex,
            "path": IMAGE_AST_PATH,
            "pixel_width": recipe.body["image_pixel_width"],
            "pixel_height": recipe.body["image_pixel_height"],
        }
    ]
    image_node = {
        "id": ids.new(),
        "type": "inline_image",
        "resource_ref": resource_id,
        "size": {
            "width_mpt": recipe.body["image_width_mpt"],
            "height_mpt": recipe.body["image_height_mpt"],
        },
        "crop": {"left_ppm": 0, "top_ppm": 0, "right_ppm": 0, "bottom_ppm": 0},
        "rotation_mdeg": 0,
        "alt_text": "Synthetic figure",
    }
    children = [
        _text_paragraph(ids, content, 1)
        for _ in range(_span(recipe, structure, "paragraph_count"))
    ]
    children.append(_paragraph(ids, [_run(ids, "Figure: "), image_node]))
    return _Built(
        [_section(ids, children)],
        resources_decl,
        {IMAGE_AST_PATH: resource_bytes[IMAGE_ROLE]},
    )


def _table_width(body: Mapping[str, Any], total_mpt: int) -> dict[str, Any]:
    mode = body["width_mode"]
    if mode == "absolute":
        return {"mode": "absolute", "value_mpt": total_mpt}
    if mode == "percent":
        return {"mode": "percent", "value_ppm": body["width_ppm"]}
    return {"mode": "auto", "value_mpt": 0}


def _build_table_grid(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    """One flat table between two full-width body paragraphs.

    Hazards this builder refuses to author, each measured against the strict
    synthetic observation gate: `row.layout.height.mode == 'exact'` clips text
    out of the PDF text layer at every value tested, and a non-`top`
    `cell.layout.vertical_alignment` on a row-SPANNING cell reorders the
    rendered reading order — so varied vertical alignment is only offered to
    span-free recipes.

    Spans themselves are still PLANNED here and still guarded at the AST
    level, but no ROSTERED recipe authors one: a spanning cell makes
    poppler read the row band column-major and the forward-only aligner
    loses the rest of the table. `recipes.held_out_recipes` carries the span
    recipe and the numbers that refused it.
    """
    body = recipe.body
    structure, content, style = rng["structure"], rng["content"], rng["style"]
    table_rng = rng["table_plan"]

    rows = _span(recipe, structure, "rows")
    columns = _span(recipe, structure, "cols")
    styled, shaded = body["styled"], body["shaded"]
    fill_ppm = body["fill_ppm"]
    body_fill_ppm = body["body_paragraph_fill_ppm"]
    indent_mpt = body["indent_mpt"]
    height_mode = body.get("row_height_mode", "auto")
    if height_mode == "exact":
        raise ValueError(
            "row height mode 'exact' clips cell text out of the rendered PDF "
            "text layer; use 'at_least'"
        )
    vary_valign = body.get("vary_vertical_alignment", False)
    if vary_valign and body["span_plan"] != "none":
        raise ValueError(
            "a non-top vertical_alignment on a row-spanning cell reorders the "
            "rendered reading order; vary it only on span-free tables"
        )

    total_width_mpt = _USABLE_WIDTH_MPT - indent_mpt
    column_widths = _split_widths(total_width_mpt, columns, table_rng)
    plan = _plan_grid(table_rng, rows, columns, span_plan=body["span_plan"])

    leading = _fill_paragraph(ids, content, _USABLE_WIDTH_MPT, body_fill_ppm)

    table_rows: list[dict[str, Any]] = []
    for row_index, row_plan in enumerate(plan):
        cells = []
        for grid_column, col_span, row_span in row_plan:
            width_mpt = sum(column_widths[grid_column : grid_column + col_span])
            cells.append(
                _cell(
                    ids,
                    content,
                    grid_column,
                    col_span=col_span,
                    row_span=row_span,
                    width_mpt=width_mpt,
                    padding_mpt=style.randint(
                        body.get("padding_mpt_min", 1000),
                        body.get("padding_mpt_max", 1000),
                    ),
                    vertical_alignment=(
                        style.choice(_VERTICAL_ALIGNMENTS) if vary_valign else "top"
                    ),
                    background=(
                        style.choice(_SHADE_COLORS)
                        if shaded and style.randrange(1_000_000) < body["shaded_cells_ppm"]
                        else None
                    ),
                    borders=_borders(style) if styled else None,
                    children=[_fill_paragraph(ids, content, width_mpt, fill_ppm)],
                )
            )
        table_rows.append(
            _row(
                ids,
                cells,
                height_mode=height_mode,
                height_mpt=(
                    0
                    if height_mode == "auto"
                    else table_rng.randint(
                        body["row_height_mpt_min"], body["row_height_mpt_max"]
                    )
                ),
                repeat_header=body.get("repeat_header", False) and row_index == 0,
                cant_split=body.get("cant_split", False),
            )
        )

    table = _table(
        ids,
        columns,
        table_rows,
        column_widths=column_widths,
        width=_table_width(body, total_width_mpt),
        alignment=body["alignment"],
        indent_mpt=indent_mpt,
        autofit=body["autofit"],
        background=style.choice(_SHADE_COLORS) if shaded else None,
        borders=_borders(style) if styled else None,
    )
    children = [
        leading,
        table,
        _fill_paragraph(ids, content, _USABLE_WIDTH_MPT, body_fill_ppm),
    ]
    return _Built([_section(ids, children)], [], {})


def _build_table_nested(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    """One nested table, depth 1, inside a SINGLE-COLUMN outer table.

    Measured: a sibling cell beside the nested table scrambles pdftotext's
    reading order, and a cell may never end on a table
    (CELL_LAST_CHILD_NOT_PARAGRAPH), so the nested table is always followed by
    a trailing paragraph in its host cell.
    """
    body = recipe.body
    content, style, table_rng = rng["content"], rng["style"], rng["table_plan"]
    outer_columns = body["outer_cols"]
    if outer_columns != 1:
        raise ValueError("the outer table of a nested-table recipe must be single-column")
    padding_mpt = body["padding_mpt"]
    fill_ppm = body["fill_ppm"]
    body_fill_ppm = body["body_paragraph_fill_ppm"]
    inner_columns = body["inner_cols"]

    outer_widths = _split_widths(_USABLE_WIDTH_MPT, outer_columns, table_rng)
    inner_total_mpt = _USABLE_WIDTH_MPT - 2 * padding_mpt
    inner_widths = _split_widths(inner_total_mpt, inner_columns, table_rng)

    leading = _fill_paragraph(ids, content, _USABLE_WIDTH_MPT, body_fill_ppm)
    inner_rows = [
        _row(
            ids,
            [
                _cell(
                    ids,
                    content,
                    column,
                    width_mpt=inner_widths[column],
                    padding_mpt=padding_mpt,
                    borders=_borders(style),
                    children=[
                        _fill_paragraph(ids, content, inner_widths[column], fill_ppm)
                    ],
                )
                for column in range(inner_columns)
            ],
        )
        for _ in range(body["inner_rows"])
    ]
    inner_table = _table(
        ids,
        inner_columns,
        inner_rows,
        column_widths=inner_widths,
        width={"mode": "absolute", "value_mpt": inner_total_mpt},
        autofit=False,
        borders=_borders(style),
    )

    outer_rows: list[dict[str, Any]] = []
    for row_index in range(body["outer_rows"]):
        cells = []
        for column in range(outer_columns):
            width_mpt = outer_widths[column]
            cell_children = [_fill_paragraph(ids, content, width_mpt, fill_ppm)]
            if row_index == 0 and column == 0:
                cell_children.append(inner_table)
                cell_children.append(
                    _fill_paragraph(ids, content, width_mpt, fill_ppm)
                )
            cells.append(
                _cell(
                    ids,
                    content,
                    column,
                    width_mpt=width_mpt,
                    padding_mpt=padding_mpt,
                    borders=_borders(style),
                    children=cell_children,
                )
            )
        outer_rows.append(_row(ids, cells))

    outer_table = _table(
        ids,
        outer_columns,
        outer_rows,
        column_widths=outer_widths,
        width={"mode": "absolute", "value_mpt": _USABLE_WIDTH_MPT},
        autofit=False,
        borders=_borders(style),
    )
    children = [
        leading,
        outer_table,
        _fill_paragraph(ids, content, _USABLE_WIDTH_MPT, body_fill_ppm),
    ]
    return _Built([_section(ids, children)], [], {})


# ---------------------------------------------------------- tier-F layouts --


def _kinds(body: Mapping[str, Any], key: str) -> tuple[str, ...]:
    """Story kinds from a comma-separated body knob (bodies are int/bool/str
    only — a list would still hash, but the JCS layer keeps bodies flat)."""
    kinds = tuple(k.strip() for k in body[key].split(",") if k.strip())
    if len(set(kinds)) != len(kinds):
        raise ValueError(f"{key} repeats a kind ({body[key]!r}) — DUPLICATE_STORY_KIND")
    return kinds


def _build_stories_multipage(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    """One section carrying default/first/even headers AND footers over a
    body long enough to render THREE pages.

    MEASURED, and the reason `paragraph_count` starts at 130: `titlePg` plus
    `evenAndOddHeaders` means the `default` header and footer are rendered
    only on an ODD page after the first, i.e. only from page 3 on. Below
    three pages the aligner has a story with no glyphs to place and the
    strict synthetic observation gate fails with
    OBSERVATION_STORY_UNPLACED.

    An earlier version of this docstring recorded "80 paragraphs -> 3 pages
    -> 172/172". That does not reproduce here: 70-90 paragraphs render TWO
    pages and fail every index of `range(16)` (175/179 at index 0). 130-150
    render three pages and are 16/16. The number is re-measured, not
    remembered — a docstring that reports a measurement nobody can reproduce
    makes the next reader faster at the wrong reasoning.
    """
    body = recipe.body
    structure, content = rng["structure"], rng["content"]
    header_kinds = _kinds(body, "header_kinds")
    footer_kinds = _kinds(body, "footer_kinds")

    headers = [
        _story(
            ids,
            "header",
            kind,
            [_paragraph(ids, [_run(ids, f"{kind.capitalize()} header {_sentence(content)}")])],
        )
        for kind in header_kinds
    ]
    footers = []
    for kind in footer_kinds:
        inline: list[dict[str, Any]] = [
            _run(ids, f"{kind.capitalize()} footer {_sentence(content)} page ")
        ]
        # The PAGE field lives in exactly one footer: its rendered result is a
        # different string on every page, so one is coverage and three is noise.
        if kind == body["field_footer_kind"]:
            inline.append(_field(ids, "page"))
        footers.append(_story(ids, "footer", kind, [_paragraph(ids, inline)]))

    paragraph_count = _span(recipe, structure, "paragraph_count")
    link_at = body["hyperlink_paragraph_index"]
    if not 0 <= link_at < paragraph_count:
        raise ValueError(
            f"hyperlink_paragraph_index {link_at} outside 0..{paragraph_count - 1}"
        )
    children: list[dict[str, Any]] = []
    for index in range(paragraph_count):
        if index == link_at:
            children.append(
                _paragraph(
                    ids,
                    [
                        _run(ids, _sentence(content) + " "),
                        _hyperlink(
                            ids,
                            body["hyperlink_kind"],
                            body["hyperlink_uri"],
                            [_run(ids, _sentence(content))],
                        ),
                    ],
                )
            )
        else:
            children.append(_text_paragraph(ids, content, 1))

    section = _section(
        ids,
        children,
        headers=headers,
        footers=footers,
        margins_extra={
            "header_mpt": body["margins_header_mpt"],
            "footer_mpt": body["margins_footer_mpt"],
        },
        page_numbering={
            "format": body["page_numbering_format"],
            "start": body["page_numbering_start"],
        },
    )
    return _Built([section], [], {})


def _build_notes(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    """Body paragraphs carrying `note_ref`s into root footnote/endnote
    stories. Every ref resolves and every note is referenced.

    HELD OUT OF EVERY ROSTER (owner decision OPEN-1), and the root cause is
    NOT in this package: `observations/stories.py::_match_across_gaps` refuses
    a body word that straddles a gap, and LibreOffice's PDF text layer glues
    the custom note mark to the body's first word ("1Footnote ..."), so the
    note story is UNPLACED and the strict synthetic observation gate fails
    (measured 10/12). The one-character workaround — a LEADING SPACE in the
    note body — scores 12/12 and is REJECTED: it writes a rendering artefact
    into a supervised label. This builder therefore never emits one, and
    `stories.py` is not edited here.
    """
    body = recipe.body
    structure, content = rng["structure"], rng["content"]
    children: list[dict[str, Any]] = [
        _text_paragraph(ids, content, 1)
        for _ in range(_span(recipe, structure, "lead_paragraphs"))
    ]
    footnotes: list[dict[str, Any]] = []
    endnotes: list[dict[str, Any]] = []

    def _referenced(note_type: str, marker: str) -> None:
        # The note body is built FIRST so the ref can carry its real id, and
        # its first run starts with a word, never with padding.
        note = _note(ids, note_type, [_text_paragraph(ids, content, 1)])
        (footnotes if note_type == "footnote" else endnotes).append(note)
        children.append(
            _paragraph(
                ids,
                [
                    _run(ids, _sentence(content) + " "),
                    _note_ref(ids, note["id"], marker),
                ],
            )
        )

    for index in range(_span(recipe, structure, "notes")):
        _referenced("footnote", str(index + 1))
    if body["include_endnote"]:
        _referenced("endnote", "i")

    children.append(_text_paragraph(ids, content, 1))
    return _Built(
        [_section(ids, children)], [], {}, tuple(footnotes), tuple(endnotes)
    )


def _build_floating_objects(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], resource_bytes: dict[str, bytes]
) -> _Built:
    """Anchored images in a SINGLE-column body.

    MEASURED: a floating object in a two-column section broke body alignment
    (probe F1); at column_count 1 the same document is 4/4. The first two
    images always take the COMPLEMENTARY anchor mode pair, so both schema
    `oneOf` branches appear on both axes in every generated document rather
    than across a lucky spread of indices.
    """
    body = recipe.body
    structure, content, anchor_rng = rng["structure"], rng["content"], rng["anchor"]
    resource_id, resources_decl = _image_resource(recipe)

    count = _span(recipe, structure, "images_per_doc")
    if count < 2:
        raise ValueError(
            "a floating-object document needs >= 2 images: the complementary "
            "anchor mode pair is what makes both oneOf branches reachable"
        )
    pairs = list(_ANCHOR_MODE_PAIRS[:2])
    anchor_rng.shuffle(pairs)
    pairs += [anchor_rng.choice(_ANCHOR_MODE_PAIRS) for _ in range(count - len(pairs))]

    children: list[dict[str, Any]] = [
        _text_paragraph(ids, content, 2)
        for _ in range(_span(recipe, structure, "lead_paragraphs"))
    ]
    for h_mode, v_mode in pairs:
        anchor = _anchor(
            anchor_rng,
            h_mode=h_mode,
            v_mode=v_mode,
            wrap_mode=anchor_rng.choice(_WRAP_MODES),
            behind_text=anchor_rng.choice((True, False)),
            # `overlap_text` is on so the four overlap-conditional pointers
            # are observable rather than merely present.
            allow_overlap=anchor_rng.choice((True, False)),
            z_order=anchor_rng.randrange(0, 4),
        )
        # Non-zero on exactly two sides, and non-zero rotation: a crop or a
        # rotation of 0 is a pointer with no pixel evidence behind it.
        crop = {
            "left_ppm": body["crop_ppm"],
            "top_ppm": 0,
            "right_ppm": 0,
            "bottom_ppm": body["crop_ppm"],
        }
        children.append(
            _paragraph(
                ids,
                [
                    _run(ids, _sentence(content) + " "),
                    _floating_image(
                        ids,
                        resource_id,
                        anchor,
                        width_mpt=body["image_width_mpt"],
                        height_mpt=body["image_height_mpt"],
                        crop=crop,
                        rotation_mdeg=body["rotation_mdeg"],
                    ),
                ],
            )
        )
        children.append(_text_paragraph(ids, content, 2))

    section = _section(ids, children, column_count=1)
    return _Built(
        [section], resources_decl, {IMAGE_AST_PATH: resource_bytes[IMAGE_ROLE]}
    )


def _build_text_frames(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    """Anchored block containers in a SINGLE-column body.

    A `list` is NEVER placed inside a frame, and that is a live renderer
    defect this generator routes around rather than papers over:
    `renderer/ooxml/numbering.py::collect_list_nodes` walks list items and
    table cells but has NO text_frame branch, so the frame's list is never
    registered, `document_builder.py:344` looks its id up with `.get()` and
    writes `<w:numId w:val="None"/>`, no numbering.xml is emitted, the compile
    still returns success_with_warnings, and LibreOffice drops every marker.
    Both existing guards are structurally blind to it (XSD validation strips
    the whole wps subtree; text conservation ignores frame content on both
    sides), so `test_no_list_is_ever_generated_inside_a_text_frame` is the
    only thing standing between a future edit and silent label corruption.
    """
    body = recipe.body
    structure, content, anchor_rng = rng["structure"], rng["content"], rng["anchor"]
    children: list[dict[str, Any]] = [
        _text_paragraph(ids, content, 2)
        for _ in range(_span(recipe, structure, "lead_paragraphs"))
    ]
    frames: list[dict[str, Any]] = []
    for _frame_index in range(body["frames_per_doc"]):
        h_mode, v_mode = anchor_rng.choice(_ANCHOR_MODE_PAIRS)
        anchor = _anchor(
            anchor_rng,
            h_mode=h_mode,
            v_mode=v_mode,
            wrap_mode=_FRAME_WRAP_MODE,
            behind_text=False,
            allow_overlap=False,
            z_order=anchor_rng.randrange(0, 4),
        )
        frames.append(
            _text_frame(
                ids,
                anchor,
                [
                    _text_paragraph(ids, content, 1)
                    for _ in range(_span(recipe, structure, "frame_paragraphs"))
                ],
            )
        )
    # The frames go on a PAGE OF THEIR OWN, after an explicit page break.
    # MEASURED, and none of the gentler arrangements work: LibreOffice does
    # NOT reflow body text around a `wps` text box the way it does around a
    # picture, so whatever the wrap mode says, a frame sharing a page with
    # body text is drawn on top of it, poppler reads the two overlapping
    # word streams as columns, and the forward-only aligner loses the body
    # paragraphs (interleaved 4/8 indices admitted; wrap pinned to
    # `top_and_bottom` 14/16; anchor frame of reference pinned as well
    # 11/16; frames merely last in document order 12/16). On their own page
    # the frames still overlap — but only each other, and their glyphs are
    # orphan to the aligner in every arrangement anyway.
    children.append({"id": ids.new(), "type": "page_break"})
    children.extend(frames)
    return _Built([_section(ids, children, column_count=1)], [], {})


def _build_columns_breaks_rich(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    """An unequal-width multi-column section with explicit breaks.

    `equal_width` false makes `widths_mpt` load-bearing: the semantic
    validator raises COLUMN_WIDTH_COUNT_MISMATCH unless it holds exactly
    `count` entries. Widths come from `_split_widths`, so they are integers
    that sum to the usable width minus the gaps — no float ever reaches the
    tree. Breaks separate real content: never leading, never trailing, never
    adjacent, and never inside a story.
    """
    body = recipe.body
    structure, content, table_rng = rng["structure"], rng["content"], rng["table_plan"]
    count = _span(recipe, structure, "column_count")
    gap_mpt = structure.randrange(body["gap_mpt_min"], body["gap_mpt_max"] + 1, 1000)
    widths = _split_widths(_USABLE_WIDTH_MPT - gap_mpt * (count - 1), count, table_rng)

    def _group() -> list[dict[str, Any]]:
        return [
            _text_paragraph(ids, content, 1)
            for _ in range(_span(recipe, structure, "paragraphs_per_group"))
        ]

    children: list[dict[str, Any]] = []
    for column_index in range(count):
        if column_index:
            children.append({"id": ids.new(), "type": "column_break"})
        children.extend(_group())
    children.append({"id": ids.new(), "type": "page_break"})
    children.extend(_group())

    section = _section(
        ids,
        children,
        column_count=count,
        equal_width=False,
        widths_mpt=widths,
        gap_mpt=gap_mpt,
    )
    return _Built([section], [], {})


def _build_section_decor(
    recipe: Recipe, ids: _Ids, rng: dict[str, random.Random], _: dict[str, bytes]
) -> _Built:
    """Sections carrying the four section-level decor properties.

    Every one of them (page_borders, line_numbering, page_numbering,
    vertical_alignment, orientation, break_type) is a SECTION property, not a
    node type, so this stays on `word-core-v0`: the semantic validator's
    profile gate is on node types only, and there is no reason to drag this
    coverage into the unratified visual-profile question.

    `line_numbering.count_by` is pinned to 1 with one-line paragraphs — see
    `_LINE_NUMBERING_COUNT_BY`.
    """
    body = recipe.body
    structure, content, style = rng["structure"], rng["content"], rng["style"]
    sections: list[dict[str, Any]] = []
    for section_index in range(_span(recipe, structure, "sections")):
        paragraphs = [
            _text_paragraph(ids, content, 1)
            for _ in range(_span(recipe, structure, "paragraphs_per_section"))
        ]
        sections.append(
            _section(
                ids,
                paragraphs,
                break_type=(
                    "next_page"
                    if section_index == 0
                    else structure.choice(_SECTION_BREAK_TYPES)
                ),
                orientation=structure.choice(("portrait", "landscape")),
                vertical_alignment=style.choice(_SECTION_VERTICAL_ALIGNMENTS),
                page_numbering={
                    "format": style.choice(_PAGE_NUMBER_FORMATS),
                    # `null` means "continue the previous section", so it is
                    # only meaningful after the first one.
                    "start": body["page_numbering_start"] if section_index == 0 else None,
                },
                line_numbering={
                    "count_by": _LINE_NUMBERING_COUNT_BY,
                    "start": style.randint(0, 1),
                    "restart": style.choice(_LINE_NUMBER_RESTARTS),
                    "distance_mpt": style.randrange(9000, 27001, 1000),
                },
                page_borders={
                    "sides": _borders(style),
                    "display": style.choice(_PAGE_BORDER_DISPLAYS),
                    "offset_from": style.choice(("page", "text")),
                },
            )
        )
    return _Built(sections, [], {})


_BUILDERS = {
    "plain": _build_plain,
    "mixed_runs": _build_mixed_runs,
    "lists": _build_lists,
    "tables": _build_tables,
    "sections_columns": _build_sections_columns,
    "column_breaks": _build_column_breaks,
    "inline_image": _build_inline_image,
    "table_grid": _build_table_grid,
    "table_nested": _build_table_nested,
    "stories_multipage": _build_stories_multipage,
    "notes": _build_notes,
    "floating_objects": _build_floating_objects,
    "text_frames": _build_text_frames,
    "columns_breaks_rich": _build_columns_breaks_rich,
    "section_decor": _build_section_decor,
}

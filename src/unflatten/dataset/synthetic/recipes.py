"""RFC003 §20 synthetic recipes for the `word-core-v0` profile (D2 smoke).

A `Recipe` is the *only* semantic input to the generator: its identity is
`identity.recipe_hash` over `rfc003.synthetic-recipe/v1`, so any change to
the body knobs, resource inputs or dependencies yields a new recipe hash and
therefore new lineage roots (§12.2). Bodies are integer/bool/string only —
the JCS serializer underneath the identity layer rejects floats by design.

`resource_inputs` records carry exactly the §12.2 identity fields
`{role, sha256, size_bytes, mime_type}`; transport paths never enter
identity. The one D2 resource is a caller-supplied PNG (the test asset
`tests/data/assets/tiny_2x2.png` in this repo), passed in as bytes so this
package never hardcodes a test-tree path.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from unflatten.dataset import identity

DATASET_NAMESPACE = "unflatten.synthetic.smoke"
GENERATOR_VERSION = "0.1.0"
RECIPE_SCHEMA_VERSION = "0.1.0"
RECIPE_COHORT_ID = "word-core-v0-smoke"

#: role key for the single D2 image resource input.
IMAGE_ROLE = "inline_image_png"


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze dicts to MappingProxyType and lists to tuples."""
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    return value


@dataclass(frozen=True, eq=False)
class Recipe:
    """One immutable synthetic recipe (RFC003 §20.2).

    `body` and `resource_inputs` are deep-copied and deep-frozen at
    construction (nested MappingProxyType / tuples), and the recipe hash
    is computed eagerly from that private copy — mutating the recipe
    after construction raises, and mutating the caller's source dicts is
    harmless: the cached hash always describes the stored body.
    """

    recipe_id: str
    body: dict[str, Any]
    recipe_schema_version: str = RECIPE_SCHEMA_VERSION
    resource_inputs: tuple[dict[str, Any], ...] = ()
    dependency_recipe_hashes: tuple[str, ...] = ()
    _recipe_hash: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.recipe_id, str) or not self.recipe_id:
            raise ValueError(f"recipe_id must be a non-empty string, got {self.recipe_id!r}")
        if not isinstance(self.body, dict) or "layout_kind" not in self.body:
            raise ValueError("recipe body must be a dict carrying 'layout_kind'")
        body = copy.deepcopy(dict(self.body))
        resource_inputs = tuple(copy.deepcopy(dict(r)) for r in self.resource_inputs)
        recipe_hash = identity.recipe_hash(
            recipe_schema_version=self.recipe_schema_version,
            recipe_id=self.recipe_id,
            recipe_body=body,
            resource_inputs=[dict(r) for r in resource_inputs],
            dependency_recipe_hashes=list(self.dependency_recipe_hashes),
        )
        object.__setattr__(self, "body", _deep_freeze(body))
        object.__setattr__(
            self, "resource_inputs", tuple(_deep_freeze(r) for r in resource_inputs)
        )
        object.__setattr__(self, "_recipe_hash", recipe_hash)

    def hash(self) -> str:
        """`sha256:<64hex>` recipe hash (§12.2), computed once at init."""
        return self._recipe_hash


def image_resource_input(png_bytes: bytes) -> dict[str, Any]:
    """§12.2 identity record for the caller-supplied PNG resource."""
    return {
        "role": IMAGE_ROLE,
        "sha256": f"sha256:{hashlib.sha256(png_bytes).hexdigest()}",
        "size_bytes": len(png_bytes),
        "mime_type": "image/png",
    }


def word_core_v0_recipes(image_png: bytes) -> tuple[Recipe, ...]:
    """The eight D2 smoke recipes covering the word-core-v0 node families
    (six single-page families, one multi-page overflow variant and one
    column-break variant).

    `image_png` is the raw bytes of the PNG used by the inline-image
    recipe (its sha256/size bind into that recipe's identity).
    """
    return (
        Recipe(
            recipe_id="wcv0-plain-paragraphs",
            body={
                "layout_kind": "plain",
                "paragraph_count_min": 2,
                "paragraph_count_max": 5,
                "sentences_min": 1,
                "sentences_max": 3,
            },
        ),
        Recipe(
            recipe_id="wcv0-mixed-run-styles",
            body={
                "layout_kind": "mixed_runs",
                "paragraph_count_min": 2,
                "paragraph_count_max": 4,
                "runs_min": 2,
                "runs_max": 5,
            },
        ),
        Recipe(
            recipe_id="wcv0-lists",
            body={
                "layout_kind": "lists",
                "ordered_items_min": 2,
                "ordered_items_max": 4,
                "unordered_items_min": 2,
                "unordered_items_max": 3,
                "nested_items_min": 1,
                "nested_items_max": 2,
            },
        ),
        Recipe(
            recipe_id="wcv0-tables",
            body={
                "layout_kind": "tables",
                "rows_min": 2,
                "rows_max": 4,
                "cols_min": 2,
                "cols_max": 3,
                "include_span_table": True,
            },
        ),
        Recipe(
            recipe_id="wcv0-sections-columns",
            body={
                "layout_kind": "sections_columns",
                "paragraphs_per_section_min": 1,
                "paragraphs_per_section_max": 3,
                "two_column_section_index": 1,
            },
        ),
        Recipe(
            # multi-page by natural overflow (25+ paragraphs of 3-5 sentences
            # cannot fit one A4 page) so fragment ownership across a page
            # boundary is exercised end-to-end; no page_break node needed.
            recipe_id="wcv0-multipage-plain",
            body={
                "layout_kind": "plain",
                "paragraph_count_min": 25,
                "paragraph_count_max": 30,
                "sentences_min": 3,
                "sentences_max": 5,
            },
        ),
        Recipe(
            # RFC001 §42.1 makes ColumnBreak a mandatory V0 node, but no
            # other recipe emits one (sections_columns builds multi-column
            # sections and lets text flow), so the corpus was short of the
            # §43 minimum. Three columns => two breaks per document.
            recipe_id="wcv0-column-breaks",
            body={
                "layout_kind": "column_breaks",
                "column_count": 3,
                "paragraphs_per_column_min": 1,
                "paragraphs_per_column_max": 3,
            },
        ),
        Recipe(
            recipe_id="wcv0-inline-image",
            body={
                "layout_kind": "inline_image",
                "paragraph_count_min": 1,
                "paragraph_count_max": 2,
                "image_pixel_width": 2,
                "image_pixel_height": 2,
                "image_width_mpt": 50000,
                "image_height_mpt": 50000,
            },
            resource_inputs=(image_resource_input(image_png),),
        ),
    )


def pyramid_d_recipes(image_png: bytes) -> tuple[Recipe, ...]:
    """Pyramid tier-D (table) recipes: the three table width modes, row/cell
    layout decor, and one depth-1 nested table.

    Four authoring facts here are measured against the strict synthetic
    observation gate, not reasoned: surrounding body paragraphs are
    FULL-WIDTH (measured on the SPAN-BEARING body, where short ones dropped
    coverage to 4/12 — re-measured after the roster went span-free,
    `body_paragraph_fill_ppm` 100000 still observed every text node at
    indices 0..7, so full width is no longer load-bearing for THESE bodies
    and is kept only because the held-out span recipe shares the knob);
    the decor recipe is authored SINGLE-COLUMN, which costs no coverage
    because every border/shading/padding/vertical_alignment pointer is
    per-cell or per-table and never per-column (22/22 vs 10/22 for the
    three-column form); `row.layout.height.mode == 'exact'` is never
    emitted, because it clips cell text out of the rendered PDF text layer
    at every value tested; and NO ROSTERED RECIPE AUTHORS A SPAN — see
    `held_out_recipes`, which carries the span recipe and the measurement
    that refused it.

    `image_png` is unused — tier D declares no resources — and is accepted
    so every roster function has the one registry signature.
    """
    return (
        Recipe(
            # The styled/shaded absolute-width table. `span_plan` is `none`
            # by MEASUREMENT, not by preference: see `held_out_recipes`.
            recipe_id="d-table-styled",
            body={
                "layout_kind": "table_grid",
                "rows_min": 3,
                "rows_max": 5,
                "cols_min": 3,
                "cols_max": 4,
                "span_plan": "none",
                "width_mode": "absolute",
                "alignment": "left",
                "indent_mpt": 0,
                "autofit": False,
                "styled": True,
                "shaded": True,
                "shaded_cells_ppm": 400000,
                "fill_ppm": 700000,
                "body_paragraph_fill_ppm": 1000000,
            },
        ),
        Recipe(
            recipe_id="d-table-width-percent",
            body={
                "layout_kind": "table_grid",
                "rows_min": 3,
                "rows_max": 4,
                "cols_min": 3,
                "cols_max": 3,
                "span_plan": "none",
                "width_mode": "percent",
                "width_ppm": 1000000,
                "alignment": "center",
                "indent_mpt": 0,
                "autofit": False,
                "styled": True,
                "shaded": False,
                "shaded_cells_ppm": 0,
                "fill_ppm": 700000,
                "body_paragraph_fill_ppm": 1000000,
            },
        ),
        Recipe(
            # The deliberate NULL-state recipe: `styled`/`shaded` false keeps
            # table and cell `style.borders` at null, which is the ONLY way
            # the object-typed container pointers are emitted at all — a dict
            # value recurses to its leaves and never emits the container.
            recipe_id="d-table-width-auto-plain",
            body={
                "layout_kind": "table_grid",
                "rows_min": 2,
                "rows_max": 3,
                "cols_min": 2,
                "cols_max": 3,
                "span_plan": "none",
                "width_mode": "auto",
                "alignment": "left",
                "indent_mpt": 0,
                "autofit": True,
                "styled": False,
                "shaded": False,
                "shaded_cells_ppm": 0,
                "fill_ppm": 600000,
                "body_paragraph_fill_ppm": 1000000,
            },
        ),
        Recipe(
            # The decor recipe, authored SINGLE-COLUMN on purpose (see the
            # docstring above). `span_plan` none is what makes a varied
            # vertical_alignment safe: it is only the row-SPANNING case that
            # reorders the rendered stream.
            recipe_id="d-table-row-cell-layout",
            body={
                "layout_kind": "table_grid",
                "rows_min": 4,
                "rows_max": 6,
                "cols_min": 1,
                "cols_max": 1,
                "span_plan": "none",
                "width_mode": "absolute",
                "alignment": "left",
                "indent_mpt": 18000,
                "autofit": False,
                "styled": True,
                "shaded": True,
                "shaded_cells_ppm": 400000,
                "fill_ppm": 900000,
                "body_paragraph_fill_ppm": 1000000,
                "row_height_mode": "at_least",
                "row_height_mpt_min": 14000,
                "row_height_mpt_max": 26000,
                "repeat_header": True,
                "cant_split": True,
                "vary_vertical_alignment": True,
                "padding_mpt_min": 500,
                "padding_mpt_max": 3000,
            },
        ),
        Recipe(
            # Nested tables are a word-visual-v1 node arrangement: under
            # word-core-v0 the semantic validator raises
            # NESTED_TABLE_UNSUPPORTED. Structural/grammar coverage only —
            # a nested table exercises no registry pointer the flat tables
            # above do not already reach.
            recipe_id="d-table-nested",
            body={
                "layout_kind": "table_nested",
                "profile": "word-visual-v1",
                "outer_cols": 1,
                "outer_rows": 2,
                "inner_cols": 2,
                "inner_rows": 2,
                "nesting_depth": 1,
                "padding_mpt": 1000,
                "fill_ppm": 700000,
                "body_paragraph_fill_ppm": 1000000,
            },
        ),
    )


def pyramid_f_recipes(image_png: bytes) -> tuple[Recipe, ...]:
    """Pyramid tier-F (full-document) recipes: header/footer stories over a
    multi-page body, anchored images, text frames, unequal multi-column
    sections with explicit breaks, and section-level decor.

    Four authoring facts here are MEASURED against the strict synthetic
    observation gate, not reasoned, and each is the reason a knob is a
    constant rather than a range:

    * `f-headers-footers-multipage` needs 130-150 paragraphs, because THREE
      rendered pages are what places all three header/footer kinds. On a
      one- or two-page document `titlePg` + `evenAndOddHeaders` means the
      `default` header and footer are never rendered at all, the aligner has
      a story with no glyphs to place, and the gate turns
      OBSERVATION_STORY_UNPLACED into a FAIL. Measured over `range(16)`:
      70-90 paragraphs renders TWO pages and fails every index (175/179 at
      index 0); 130-150 renders three and is 16/16.
    * `f-floating-image` is SINGLE-column. A floating object combined with a
      two-column section broke body alignment; at column_count 1 it is 16/16.
    * `f-text-frame` declares `word-visual-v1.1` (the only profile that
      admits a `text_frame`), never puts a list inside the frame — see
      `generator._build_text_frames` for the renderer defect that would
      silently drop every marker — and puts its frames on a page of their
      own, which is the only arrangement measured at 16/16.
    * `f-multicolumn-breaks` needs a gutter of 36000-54000 mpt. At the Word
      minimum of 9000 mpt poppler does not see two columns at all and reads
      the page line by line ACROSS them, which loses the body (16/20 at
      index 1). Measured over `range(16)`: 9000-27000 fails, 36000-54000 is
      16/16.
    * `f-section-decor` pins `line_numbering.count_by` to 1 with one-line
      paragraphs: `count_by` 5 injects the printed line number INSIDE a
      wrapped paragraph's word stream (OBSERVATION_ALIGNMENT_MISMATCH).

    `f-multicolumn-breaks` and `f-section-decor` stay on `word-core-v0`: they
    add section PROPERTIES, not `word-visual-v1` node types, and there is no
    reason to drag them into the unratified visual-profile question.

    `f-footnotes` is deliberately NOT here — see `held_out_recipes`.
    """
    return (
        Recipe(
            recipe_id="f-headers-footers-multipage",
            body={
                "layout_kind": "stories_multipage",
                "profile": "word-visual-v1",
                "paragraph_count_min": 130,
                "paragraph_count_max": 150,
                "header_kinds": "default,first,even",
                "footer_kinds": "default,first,even",
                "field_footer_kind": "default",
                "margins_header_mpt": 36000,
                "margins_footer_mpt": 36000,
                "page_numbering_format": "decimal",
                "page_numbering_start": 1,
                "hyperlink_paragraph_index": 3,
                "hyperlink_kind": "external",
                "hyperlink_uri": "https://example.org/unflatten",
            },
        ),
        Recipe(
            recipe_id="f-floating-image",
            body={
                "layout_kind": "floating_objects",
                "profile": "word-visual-v1",
                "lead_paragraphs_min": 2,
                "lead_paragraphs_max": 3,
                # >= 2 so the COMPLEMENTARY anchor mode pair fits in every
                # document: both oneOf branches on both axes by construction.
                "images_per_doc_min": 2,
                "images_per_doc_max": 3,
                "image_pixel_width": 2,
                "image_pixel_height": 2,
                "image_width_mpt": 90000,
                "image_height_mpt": 60000,
                "crop_ppm": 50000,
                "rotation_mdeg": 15000,
            },
            resource_inputs=(image_resource_input(image_png),),
        ),
        Recipe(
            recipe_id="f-text-frame",
            body={
                "layout_kind": "text_frames",
                # A text_frame is admitted by exactly one profile
                # (TEXT_FRAME_REQUIRES_VISUAL_V1_1).
                "profile": "word-visual-v1.1",
                "lead_paragraphs_min": 2,
                "lead_paragraphs_max": 3,
                "frames_per_doc": 2,
                "frame_paragraphs_min": 1,
                "frame_paragraphs_max": 2,
            },
        ),
        Recipe(
            recipe_id="f-multicolumn-breaks",
            body={
                "layout_kind": "columns_breaks_rich",
                "column_count_min": 2,
                "column_count_max": 3,
                "gap_mpt_min": 36000,
                "gap_mpt_max": 54000,
                "paragraphs_per_group_min": 2,
                "paragraphs_per_group_max": 4,
            },
        ),
        Recipe(
            recipe_id="f-section-decor",
            body={
                "layout_kind": "section_decor",
                "sections_min": 2,
                "sections_max": 3,
                "paragraphs_per_section_min": 3,
                "paragraphs_per_section_max": 5,
                "page_numbering_start": 1,
            },
        ),
    )


def held_out_recipes(image_png: bytes) -> tuple[Recipe, ...]:
    """Recipes whose GENERATOR is shipped and tested but which no roster —
    including `all` — may produce.

    `d-table-spans` is held out BY MEASUREMENT. A table cell that spans
    columns or rows makes poppler's layout analysis read the row band
    column-major instead of row-major, the forward-only aligner then loses
    every paragraph after the first out-of-order cell, and the strict
    synthetic observation gate FAILS. It is not a tuning problem: every
    variant tried failed some index of `range(16)` —
    `colspan_rowspan` at 3-4 columns 10/16, at 2-3 columns 14/16, at exactly
    2 columns 14/16, `colspan` only at 3-4 columns 10/16, at exactly 3
    columns 14/16 — while the SAME bodies with `span_plan` `none` are 16/16.
    Spans are the variable; nothing else in the body changed. The generator
    keeps `_plan_grid`'s span planning and its AST-level guards; only the
    rostered recipes are span-free, and the consequence is stated rather than
    hidden: `cell/col_span` and `cell/row_span` are still emitted (value 1)
    but the roster exercises no value other than 1.

    `f-footnotes` is held out by owner decision (OPEN-1). The root cause is
    outside this package and is NOT worked around here:
    `src/unflatten/dataset/observations/stories.py::_match_across_gaps`
    refuses a body word that straddles a gap ("A word that straddles a gap
    has no clean segmentation and is refused rather than approximated"), and
    LibreOffice's PDF text layer glues the custom note mark to the body's
    first word ("1Footnote ..."). The note story is therefore UNPLACED and
    the strict synthetic observation gate fails, measured 10/12. Inserting a
    LEADING SPACE into the note body scores 12/12 and is REJECTED: it writes
    a rendering artefact into a supervised label. `stories.py` is not edited
    by this work.

    `image_png` is unused (this roster declares no resources) and is accepted
    so every roster function has the one registry signature.
    """
    return (
        Recipe(
            recipe_id="d-table-spans",
            body={
                "layout_kind": "table_grid",
                "rows_min": 3,
                "rows_max": 5,
                "cols_min": 3,
                "cols_max": 4,
                "span_plan": "colspan_rowspan",
                "width_mode": "absolute",
                "alignment": "left",
                "indent_mpt": 0,
                "autofit": False,
                "styled": True,
                "shaded": True,
                "shaded_cells_ppm": 400000,
                "fill_ppm": 700000,
                "body_paragraph_fill_ppm": 1000000,
            },
        ),
        Recipe(
            recipe_id="f-footnotes",
            body={
                "layout_kind": "notes",
                "profile": "word-visual-v1",
                "lead_paragraphs_min": 1,
                "lead_paragraphs_max": 2,
                "notes_min": 1,
                "notes_max": 3,
                "include_endnote": True,
            },
        ),
    )


class UnknownRecipeError(LookupError):
    """Raised for a recipe_id with no DEC-013 pyramid level."""


#: RFC003 §31 pyramid level per recipe, transcribed from
#: `docs/decisions/DEC-013-pyramid-level-assignment.md` (the owner's ratified
#: assignment: P1 is one property family, P2 is a composed page, and the two
#: deliberately multi-page recipes are P3).
#:
#: This is a module-level mapping keyed by `recipe_id`, NOT a `Recipe` field
#: and NOT a body knob, precisely so it cannot enter `identity.recipe_hash`
#: (computed in `Recipe.__post_init__` over the body/resources/dependencies)
#: and therefore cannot perturb `label_revision_hash` or
#: `document_bundle_id`.
PYRAMID_LEVELS: Mapping[str, str] = MappingProxyType(
    {
        "wcv0-plain-paragraphs": "P1",
        "wcv0-mixed-run-styles": "P1",
        "wcv0-lists": "P1",
        "wcv0-tables": "P1",
        "wcv0-column-breaks": "P1",
        "wcv0-inline-image": "P1",
        "d-table-styled": "P1",
        "d-table-width-percent": "P1",
        "d-table-width-auto-plain": "P1",
        "d-table-row-cell-layout": "P1",
        "d-table-spans": "P1",
        "f-multicolumn-breaks": "P1",
        "wcv0-sections-columns": "P2",
        "d-table-nested": "P2",
        "f-floating-image": "P2",
        "f-text-frame": "P2",
        "f-section-decor": "P2",
        "f-footnotes": "P2",
        "wcv0-multipage-plain": "P3",
        "f-headers-footers-multipage": "P3",
    }
)


def pyramid_level(recipe_id: str) -> str:
    """DEC-013 pyramid level (`P1`/`P2`/`P3`) for `recipe_id`.

    There is deliberately NO default: an unmapped recipe silently landing in
    some level would put a cohort in the wrong `StageDataMatrix` cell with no
    symptom, so it fails loudly instead.
    """
    try:
        return PYRAMID_LEVELS[recipe_id]
    except KeyError:
        raise UnknownRecipeError(
            f"recipe_id {recipe_id!r} has no pyramid level; add it to "
            f"PYRAMID_LEVELS (DEC-013)"
        ) from None


def all_recipes(image_png: bytes) -> tuple[Recipe, ...]:
    """Every roster, in registry order."""
    return (
        word_core_v0_recipes(image_png)
        + pyramid_d_recipes(image_png)
        + pyramid_f_recipes(image_png)
    )


#: Named rosters. The factory CLI defaults to `core`, so the tier-D/F
#: documents are opt-in and cannot enter the gates/split fixtures by accident.
ROSTERS = {
    "core": word_core_v0_recipes,
    "d": pyramid_d_recipes,
    "f": pyramid_f_recipes,
    "all": all_recipes,
}

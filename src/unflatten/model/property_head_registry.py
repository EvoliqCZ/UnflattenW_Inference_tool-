"""RFC005 §9.1 item 8 / RFC004 §24.2 Property Head Registry — S0 model_admission
artefact, closed by `fix-import-fidelity-defects`.

RFC004 §24.2 defines a Property Head Registry record as a versioned mapping
from a node-relative RFC001 pointer to the typed head that predicts it,
grouped by "Skupina" (§26.1-§26.4) or, for Table/Row/Cell/InlineImage
(§26.5-§26.6), by node type directly. §26 is explicit that "RFC004 nesmie
vymyslieť chýbajúci key" (RFC004 must not invent a missing key) — so every
pointer this module does not map is *excluded*, with a citable reason, never
silently defaulted to a head (mirrors the load-bearing rule already applied
by `trainable_group_registry`).

Grammar (`grammar.py`) and token (`events_vocab.py`) registries already exist
for the other two RFC005 §9.1 item 8 sub-registries; this is the third,
Property Head, half.

Ownership agreement with `trainable_group_registry`
-----------------------------------------------------
`trainable_group_registry` already closed, at tensor-identity granularity,
that `table/columns/*/width_mpt` and the three cell span fields
(`/grid_column`, `/col_span`, `/row_span`) are owned by `table_plan`, and
explicitly *not* by `typed_heads.table` / `typed_heads.row_cell` (packet R4-9).
Every mapped pointer below carries a `trainable_group_id` that must name an
existing `trainable_group_registry` group, and the table/cell carve-out is
reproduced pointer-for-pointer rather than re-derived — see
`tests/model/test_property_head_registry.py::TestCrossCheckAgainstTrainableGroupRegistry`,
which is the proof this deliverable is built around: two registries that
silently drift apart are how a property gets predicted twice or never.

Two RFC004/RFC001 discrepancies resolved here rather than guessed silently
(see the module docstring cross-reference in the delivery report):

1. §26.6's prose lists `/size/*`, `/crop/*`, `/rotation_mdeg` as "Typed heads
   pokrývajú" for `inline_image` but discusses `/alt_text` separately ("must
   remain explicitly null" for image-only V0). `trainable_group_registry`
   already resolved this the other way — `typed_heads.inline_image` owns an
   `inline_image/alt_text` tensor identity — so this registry follows the
   more specific, already-ratified evidence and maps `/alt_text` to
   `typed_heads.inline_image` too (same resolution principle
   `trainable_group_registry` documents for its own two ambiguities).
2. `pointer_registry()` is mechanically derived from the *current* schema,
   which has grown fields RFC004 §26 (written for RFC001 v0.1 / word-core-v0)
   never enumerates: word-visual-v1 additions on `section`
   (`/vertical_alignment`, `/page_numbering/*`, `/line_numbering/*`,
   `/page_borders/*`, `/margins/header_mpt`, `/margins/footer_mpt`), every
   `header`/`footer`/`note_ref`/`field`/`hyperlink`/`floating_image` pointer,
   and every `text_frame` pointer (word-visual-v1.1, RFC001-ERRATA-004,
   postdates RFC004 v0.1 entirely). §26's own instruction is "RFC004 nesmie
   vymyslieť chýbajúci key" — so these are explicit, reasoned exclusions in
   `EXCLUDED_PROPERTY_POINTERS`, not invented heads. `list/marker/text` is a
   narrower case of the same rule: §26.4's prose explicitly excludes the
   `custom_text` literal marker from default strict model capability because
   RFC001 v0.1 gives it no canonical field, and `hyperlink//target/uri` is
   excluded for the separate, already-decided reason that DEC-003 §1.3
   withdraws its supervision (known value, no model channel).
"""

from __future__ import annotations

from typing import Any

from unflatten.model.hashing import domain_hash
from unflatten.model.trainable_group_registry import is_known_group_id

_REGISTRY_DOMAIN = "rfc004.property-head-registry.v0"

_REASON_WORD_VISUAL_V1_NO_RFC004_HEAD = (
    "word-visual-v1(.1) property; RFC004 §26 ('Typed Head groups pre "
    "word-core-v0') maps only fields explicitly named or shown in RFC001 "
    "v0.1 and does not yet cover this node type/pointer — RFC004 §26 forbids "
    "inventing a missing key, so this pointer stays unmapped rather than "
    "guessing a head"
)
_REASON_LIST_MARKER_TEXT_CAPABILITY_EXCLUDED = (
    "RFC004 §26.4: the custom_text literal marker has no separate canonical "
    "RFC001 v0.1 field, so default strict model capability explicitly "
    "excludes it until RFC001 changes"
)
_REASON_HYPERLINK_TARGET_URI_NO_MODEL_CHANNEL = (
    "DEC-003 §1.3: /target/uri is a known value with no model channel; its "
    "supervision is withdrawn by masks.build_channel_mask, so no head "
    "predicts it (contrast /target/kind, which is supervisable structure "
    "but is still word-visual-v1 and has no RFC004 §26 head either)"
)


class PropertyHeadRegistryError(KeyError):
    """A pointer is neither in the mapped registry nor the exclusion table."""


#: (node_type, node-relative pointer) -> head_id. Exact enumeration, no
#: wildcards — the ``*`` segments below are literal, inherited from
#: `pointer_registry()`'s own array-index normalization, not a pattern this
#: registry introduces.
_HEAD_ASSIGNMENTS: dict[tuple[str, str], str] = {
    # --- section (§26.1); "expert_adapter.dense" is the single dense V0
    # tensor identity node-type experts collapse into (CLAUDE.md: "V1
    # routing is deterministic by node type ... MoE lives only in our
    # document-reconstruction branch"; no separate frozen group is named
    # for section/paragraph/run/list in trainable_group_registry, unlike
    # the packet-carved table/row/cell/inline_image heads). ------------
    ("section", "/break_type"): "expert_adapter.dense",
    ("section", "/page/width_mpt"): "expert_adapter.dense",
    ("section", "/page/height_mpt"): "expert_adapter.dense",
    ("section", "/page/orientation"): "expert_adapter.dense",
    ("section", "/margins/top_mpt"): "expert_adapter.dense",
    ("section", "/margins/right_mpt"): "expert_adapter.dense",
    ("section", "/margins/bottom_mpt"): "expert_adapter.dense",
    ("section", "/margins/left_mpt"): "expert_adapter.dense",
    ("section", "/margins/gutter_mpt"): "expert_adapter.dense",
    ("section", "/columns/count"): "expert_adapter.dense",
    ("section", "/columns/equal_width"): "expert_adapter.dense",
    ("section", "/columns/gap_mpt"): "expert_adapter.dense",
    ("section", "/columns/widths_mpt/*"): "expert_adapter.dense",
    # --- paragraph (§26.2) ---------------------------------------------
    ("paragraph", "/semantic/role"): "expert_adapter.dense",
    ("paragraph", "/semantic/heading_level"): "expert_adapter.dense",
    ("paragraph", "/semantic/outline_level"): "expert_adapter.dense",
    ("paragraph", "/layout/alignment"): "expert_adapter.dense",
    ("paragraph", "/layout/direction"): "expert_adapter.dense",
    ("paragraph", "/layout/indent/left_mpt"): "expert_adapter.dense",
    ("paragraph", "/layout/indent/right_mpt"): "expert_adapter.dense",
    ("paragraph", "/layout/indent/first_line_mpt"): "expert_adapter.dense",
    ("paragraph", "/layout/indent/hanging_mpt"): "expert_adapter.dense",
    ("paragraph", "/layout/spacing/before_mpt"): "expert_adapter.dense",
    ("paragraph", "/layout/spacing/after_mpt"): "expert_adapter.dense",
    ("paragraph", "/layout/spacing/line/mode"): "expert_adapter.dense",
    ("paragraph", "/layout/spacing/line/multiplier_ppm"): "expert_adapter.dense",
    ("paragraph", "/layout/spacing/line/value_mpt"): "expert_adapter.dense",
    ("paragraph", "/layout/pagination/keep_with_next"): "expert_adapter.dense",
    ("paragraph", "/layout/pagination/keep_lines"): "expert_adapter.dense",
    ("paragraph", "/layout/pagination/page_break_before"): "expert_adapter.dense",
    ("paragraph", "/layout/pagination/widow_control"): "expert_adapter.dense",
    ("paragraph", "/layout/tabs/*/alignment"): "expert_adapter.dense",
    ("paragraph", "/layout/tabs/*/leader"): "expert_adapter.dense",
    ("paragraph", "/layout/tabs/*/position_mpt"): "expert_adapter.dense",
    ("paragraph", "/style/background"): "expert_adapter.dense",
    ("paragraph", "/style/borders"): "expert_adapter.dense",
    ("paragraph", "/style/borders/bottom/color"): "expert_adapter.dense",
    ("paragraph", "/style/borders/bottom/style"): "expert_adapter.dense",
    ("paragraph", "/style/borders/bottom/width_mpt"): "expert_adapter.dense",
    ("paragraph", "/style/borders/left/color"): "expert_adapter.dense",
    ("paragraph", "/style/borders/left/style"): "expert_adapter.dense",
    ("paragraph", "/style/borders/left/width_mpt"): "expert_adapter.dense",
    ("paragraph", "/style/borders/right/color"): "expert_adapter.dense",
    ("paragraph", "/style/borders/right/style"): "expert_adapter.dense",
    ("paragraph", "/style/borders/right/width_mpt"): "expert_adapter.dense",
    ("paragraph", "/style/borders/top/color"): "expert_adapter.dense",
    ("paragraph", "/style/borders/top/style"): "expert_adapter.dense",
    ("paragraph", "/style/borders/top/width_mpt"): "expert_adapter.dense",
    # --- run (§26.3) -----------------------------------------------------
    ("run", "/language"): "expert_adapter.dense",
    ("run", "/style/font_family"): "expert_adapter.dense",
    ("run", "/style/font_size_mpt"): "expert_adapter.dense",
    ("run", "/style/bold"): "expert_adapter.dense",
    ("run", "/style/italic"): "expert_adapter.dense",
    ("run", "/style/underline"): "expert_adapter.dense",
    ("run", "/style/strike"): "expert_adapter.dense",
    ("run", "/style/color"): "expert_adapter.dense",
    ("run", "/style/highlight"): "expert_adapter.dense",
    ("run", "/style/superscript"): "expert_adapter.dense",
    ("run", "/style/subscript"): "expert_adapter.dense",
    ("run", "/style/character_spacing_mpt"): "expert_adapter.dense",
    ("run", "/style/small_caps"): "expert_adapter.dense",
    ("run", "/style/all_caps"): "expert_adapter.dense",
    # --- list (§26.4); /marker/text is capability-excluded, see module
    # docstring point 2 and EXCLUDED_PROPERTY_POINTERS. ------------------
    ("list", "/list_kind"): "expert_adapter.dense",
    ("list", "/marker/format"): "expert_adapter.dense",
    ("list", "/marker/start"): "expert_adapter.dense",
    ("list", "/marker/suffix"): "expert_adapter.dense",
    ("list", "/layout/left_indent_mpt"): "expert_adapter.dense",
    ("list", "/layout/hanging_mpt"): "expert_adapter.dense",
    # --- table (§26.5). /columns/*/width_mpt is the R4-9 table_plan
    # carve-out; every other table pointer is typed_heads.table. ---------
    ("table", "/columns/*/width_mpt"): "table_plan",
    ("table", "/layout/alignment"): "typed_heads.table",
    ("table", "/layout/indent_mpt"): "typed_heads.table",
    ("table", "/layout/autofit"): "typed_heads.table",
    ("table", "/layout/width/mode"): "typed_heads.table",
    ("table", "/layout/width/value_mpt"): "typed_heads.table",
    ("table", "/layout/width/value_ppm"): "typed_heads.table",
    ("table", "/style/background"): "typed_heads.table",
    ("table", "/style/borders"): "typed_heads.table",
    ("table", "/style/borders/bottom/color"): "typed_heads.table",
    ("table", "/style/borders/bottom/style"): "typed_heads.table",
    ("table", "/style/borders/bottom/width_mpt"): "typed_heads.table",
    ("table", "/style/borders/left/color"): "typed_heads.table",
    ("table", "/style/borders/left/style"): "typed_heads.table",
    ("table", "/style/borders/left/width_mpt"): "typed_heads.table",
    ("table", "/style/borders/right/color"): "typed_heads.table",
    ("table", "/style/borders/right/style"): "typed_heads.table",
    ("table", "/style/borders/right/width_mpt"): "typed_heads.table",
    ("table", "/style/borders/top/color"): "typed_heads.table",
    ("table", "/style/borders/top/style"): "typed_heads.table",
    ("table", "/style/borders/top/width_mpt"): "typed_heads.table",
    # --- row (§26.5), all typed_heads.row_cell --------------------------
    ("row", "/layout/height/mode"): "typed_heads.row_cell",
    ("row", "/layout/height/value_mpt"): "typed_heads.row_cell",
    ("row", "/layout/repeat_header"): "typed_heads.row_cell",
    ("row", "/layout/cant_split"): "typed_heads.row_cell",
    # --- cell (§26.5). grid_column/col_span/row_span are the R4-9
    # table_plan carve-out; every other cell pointer is
    # typed_heads.row_cell. ----------------------------------------------
    ("cell", "/grid_column"): "table_plan",
    ("cell", "/col_span"): "table_plan",
    ("cell", "/row_span"): "table_plan",
    ("cell", "/layout/width_mpt"): "typed_heads.row_cell",
    ("cell", "/layout/padding/top_mpt"): "typed_heads.row_cell",
    ("cell", "/layout/padding/right_mpt"): "typed_heads.row_cell",
    ("cell", "/layout/padding/bottom_mpt"): "typed_heads.row_cell",
    ("cell", "/layout/padding/left_mpt"): "typed_heads.row_cell",
    ("cell", "/layout/vertical_alignment"): "typed_heads.row_cell",
    ("cell", "/layout/text_direction"): "typed_heads.row_cell",
    ("cell", "/style/background"): "typed_heads.row_cell",
    ("cell", "/style/borders"): "typed_heads.row_cell",
    ("cell", "/style/borders/bottom/color"): "typed_heads.row_cell",
    ("cell", "/style/borders/bottom/style"): "typed_heads.row_cell",
    ("cell", "/style/borders/bottom/width_mpt"): "typed_heads.row_cell",
    ("cell", "/style/borders/left/color"): "typed_heads.row_cell",
    ("cell", "/style/borders/left/style"): "typed_heads.row_cell",
    ("cell", "/style/borders/left/width_mpt"): "typed_heads.row_cell",
    ("cell", "/style/borders/right/color"): "typed_heads.row_cell",
    ("cell", "/style/borders/right/style"): "typed_heads.row_cell",
    ("cell", "/style/borders/right/width_mpt"): "typed_heads.row_cell",
    ("cell", "/style/borders/top/color"): "typed_heads.row_cell",
    ("cell", "/style/borders/top/style"): "typed_heads.row_cell",
    ("cell", "/style/borders/top/width_mpt"): "typed_heads.row_cell",
    # --- inline_image (§26.6); /alt_text included per docstring point 1. -
    ("inline_image", "/size/width_mpt"): "typed_heads.inline_image",
    ("inline_image", "/size/height_mpt"): "typed_heads.inline_image",
    ("inline_image", "/crop/left_ppm"): "typed_heads.inline_image",
    ("inline_image", "/crop/top_ppm"): "typed_heads.inline_image",
    ("inline_image", "/crop/right_ppm"): "typed_heads.inline_image",
    ("inline_image", "/crop/bottom_ppm"): "typed_heads.inline_image",
    ("inline_image", "/rotation_mdeg"): "typed_heads.inline_image",
    # ERRATA-001 E7.3: ('inline_image', '/alt_text') tu UZ NIE JE — vlastnik
    # ho 2026-09-04 vylucil pre OBA typy obrazkov (alt text sa nikdy
    # nedostane na vyrenderovanu stranu, takze je z pixelov nenaucitelny).
    # Je to ODOBRANIE uz nasadeneho mapovania, nie len odmietnutie pridat.
    # --- section (§26.1 rozsirena, RFC004-ERRATA-001 E2): stranove
    # vlastnosti toho isteho uzla; nova trainable group by nekupila nic. -
    ("section", "/vertical_alignment"): "expert_adapter.dense",
    ("section", "/margins/header_mpt"): "expert_adapter.dense",
    ("section", "/margins/footer_mpt"): "expert_adapter.dense",
    ("section", "/page_numbering/format"): "expert_adapter.dense",
    ("section", "/page_numbering/start"): "expert_adapter.dense",
    ("section", "/line_numbering/count_by"): "expert_adapter.dense",
    ("section", "/line_numbering/distance_mpt"): "expert_adapter.dense",
    ("section", "/line_numbering/restart"): "expert_adapter.dense",
    ("section", "/line_numbering/start"): "expert_adapter.dense",
    ("section", "/page_borders/display"): "expert_adapter.dense",
    ("section", "/page_borders/offset_from"): "expert_adapter.dense",
    ("section", "/page_borders/sides"): "expert_adapter.dense",
    ("section", "/page_borders/sides/bottom/color"): "expert_adapter.dense",
    ("section", "/page_borders/sides/bottom/style"): "expert_adapter.dense",
    ("section", "/page_borders/sides/bottom/width_mpt"): "expert_adapter.dense",
    ("section", "/page_borders/sides/left/color"): "expert_adapter.dense",
    ("section", "/page_borders/sides/left/style"): "expert_adapter.dense",
    ("section", "/page_borders/sides/left/width_mpt"): "expert_adapter.dense",
    ("section", "/page_borders/sides/right/color"): "expert_adapter.dense",
    ("section", "/page_borders/sides/right/style"): "expert_adapter.dense",
    ("section", "/page_borders/sides/right/width_mpt"): "expert_adapter.dense",
    ("section", "/page_borders/sides/top/color"): "expert_adapter.dense",
    ("section", "/page_borders/sides/top/style"): "expert_adapter.dense",
    ("section", "/page_borders/sides/top/width_mpt"): "expert_adapter.dense",
    # --- floating_image geometria (§26.6 rozsirena, ERRATA-001 E4):
    # zdiela tenzorovu identitu s inline_image — ta ista rodina
    # vlastnosti nesmie byt trenovatelna/zmrazena podla toho, ci obrazok
    # plava. /alt_text a /anchor/z_order su VYLUCENE (E7.3, E7.1). ------
    ("floating_image", "/size/width_mpt"): "typed_heads.inline_image",
    ("floating_image", "/size/height_mpt"): "typed_heads.inline_image",
    ("floating_image", "/crop/left_ppm"): "typed_heads.inline_image",
    ("floating_image", "/crop/top_ppm"): "typed_heads.inline_image",
    ("floating_image", "/crop/right_ppm"): "typed_heads.inline_image",
    ("floating_image", "/crop/bottom_ppm"): "typed_heads.inline_image",
    ("floating_image", "/rotation_mdeg"): "typed_heads.inline_image",
    # --- kotva (§26.9 nova, ERRATA-001 E3): JEDNA spolocna hlava pre
    # floating_image aj text_frame, lebo $defs/anchor je jeden
    # ratifikovany kontrakt a dve hlavy by ho zdvojili na urovni
    # parametrov (RFC001-ERRATA-004 E1 to uz raz odmietla). ------------
    ("floating_image", "/anchor/allow_overlap"): "typed_heads.anchor",
    ("floating_image", "/anchor/behind_text"): "typed_heads.anchor",
    ("floating_image", "/anchor/horizontal/align"): "typed_heads.anchor",
    ("floating_image", "/anchor/horizontal/mode"): "typed_heads.anchor",
    ("floating_image", "/anchor/horizontal/offset_mpt"): "typed_heads.anchor",
    ("floating_image", "/anchor/horizontal/relative_from"): "typed_heads.anchor",
    ("floating_image", "/anchor/vertical/align"): "typed_heads.anchor",
    ("floating_image", "/anchor/vertical/mode"): "typed_heads.anchor",
    ("floating_image", "/anchor/vertical/offset_mpt"): "typed_heads.anchor",
    ("floating_image", "/anchor/vertical/relative_from"): "typed_heads.anchor",
    ("floating_image", "/anchor/wrap/mode"): "typed_heads.anchor",
    ("floating_image", "/anchor/wrap/side"): "typed_heads.anchor",
    ("text_frame", "/anchor/allow_overlap"): "typed_heads.anchor",
    ("text_frame", "/anchor/behind_text"): "typed_heads.anchor",
    ("text_frame", "/anchor/horizontal/align"): "typed_heads.anchor",
    ("text_frame", "/anchor/horizontal/mode"): "typed_heads.anchor",
    ("text_frame", "/anchor/horizontal/offset_mpt"): "typed_heads.anchor",
    ("text_frame", "/anchor/horizontal/relative_from"): "typed_heads.anchor",
    ("text_frame", "/anchor/vertical/align"): "typed_heads.anchor",
    ("text_frame", "/anchor/vertical/mode"): "typed_heads.anchor",
    ("text_frame", "/anchor/vertical/offset_mpt"): "typed_heads.anchor",
    ("text_frame", "/anchor/vertical/relative_from"): "typed_heads.anchor",
    ("text_frame", "/anchor/wrap/mode"): "typed_heads.anchor",
    ("text_frame", "/anchor/wrap/side"): "typed_heads.anchor",
    # --- male uzavrete enumy (§26.10 nova, ERRATA-001 E5) --------------
    ("header", "/kind"): "expert_adapter.dense",
    ("footer", "/kind"): "expert_adapter.dense",
    ("field", "/field_type"): "expert_adapter.dense",
    ("hyperlink", "/target/kind"): "expert_adapter.dense",
}

#: RFC004-ERRATA-001 E7.3, ratifikovane vlastnikom 2026-09-04. Alt text sa
#: NIKDY nedostane na vyrenderovanu stranu — nie je v pixeloch, takze je
#: z obrazu nenaucitelny. Hlava by model ucila sebavedome si vymyslat text,
#: ktory nema odkial vediet. Ta ista trieda ako nameraný strop font packu.
_REASON_ALT_TEXT_NEVER_RENDERED = (
    "alt text never reaches the rendered page, so it is unlearnable from "
    "pixels; a head would teach confident invention (RFC004-ERRATA-001 E7.3)"
)

#: RFC004-ERRATA-001 E7.1, ratifikovane vlastnikom 2026-09-04. Ked sa
#: plavajuce objekty neprekryvaju — drviva vacsina — strana nenesie ziadny
#: dokaz o tom, ktory je vpredu. Ordinalne poradie patri Tree Decoderu.
_REASON_Z_ORDER_UNOBSERVABLE = (
    "z_order is unobservable on a page where the floating objects do not "
    "overlap, and ordinal stacking belongs to the Tree Decoder, not a "
    "property head (RFC004-ERRATA-001 E7.1)"
)

#: RFC004-ERRATA-001 E6: nie je to text runu ani enum, a supervizia mu je
#: uz odobrata cez masks.build_channel_mask (DEC-003 §1.3).
_REASON_NOTE_REF_MARKER_TEXT_NO_CHANNEL = (
    "note_ref marker text is neither run text nor an enum; supervision is "
    "already withheld by masks.build_channel_mask (RFC004-ERRATA-001 E6)"
)

#: (node_type, node-relative pointer) -> reason. Exact enumeration, no
#: wildcards. Every pointer `pointer_registry()` can ever produce is either
#: here or in `_HEAD_ASSIGNMENTS`, never both, never neither (proved by
#: `tests/model/test_property_head_registry.py::TestCompleteness`).
EXCLUDED_PROPERTY_POINTERS: dict[tuple[str, str], str] = {
    # Sedem vylucenych po ratifikacii RFC004-ERRATA-001 (2026-09-04).
    # Predtym ich bolo 65; 59 dostalo hlavu, jedno (inline_image /alt_text)
    # sa naopak z mapped ODOBRALO. Sucet ostava 183 + 7 = 190.
    ("list", "/marker/text"): _REASON_LIST_MARKER_TEXT_CAPABILITY_EXCLUDED,
    ("hyperlink", "/target/uri"): _REASON_HYPERLINK_TARGET_URI_NO_MODEL_CHANNEL,
    ("note_ref", "/marker_text"): _REASON_NOTE_REF_MARKER_TEXT_NO_CHANNEL,
    ("inline_image", "/alt_text"): _REASON_ALT_TEXT_NEVER_RENDERED,
    ("floating_image", "/alt_text"): _REASON_ALT_TEXT_NEVER_RENDERED,
    ("floating_image", "/anchor/z_order"): _REASON_Z_ORDER_UNOBSERVABLE,
    ("text_frame", "/anchor/z_order"): _REASON_Z_ORDER_UNOBSERVABLE,
}


def property_head_registry() -> dict[tuple[str, str], str]:
    """Return the closed pointer -> head_id mapping.

    A copy is returned so callers cannot mutate the module-level singleton
    (mirrors `trainable_group_registry.trainable_group_registry`).
    """
    return dict(_HEAD_ASSIGNMENTS)


def head_for_pointer(node_type: str, pointer: str) -> str:
    """The head_id predicting `pointer` on `node_type`.

    Raises `PropertyHeadRegistryError` (a `KeyError`) if the pair is neither
    mapped nor explicitly excluded — an unmapped pointer must fail loudly,
    never default to a head.
    """
    key = (node_type, pointer)
    if key in _HEAD_ASSIGNMENTS:
        return _HEAD_ASSIGNMENTS[key]
    if key in EXCLUDED_PROPERTY_POINTERS:
        raise PropertyHeadRegistryError(
            f"{key!r} is explicitly excluded, not mapped to a head: "
            f"{EXCLUDED_PROPERTY_POINTERS[key]}"
        )
    raise PropertyHeadRegistryError(
        f"{key!r} is neither in the property head registry nor in "
        "EXCLUDED_PROPERTY_POINTERS — this is a registry-completeness bug, "
        "not a missing head"
    )


def excluded_pointer_reason(node_type: str, pointer: str) -> str:
    """The documented reason `(node_type, pointer)` carries no head.

    Raises `KeyError` if the pair is not in `EXCLUDED_PROPERTY_POINTERS`.
    """
    return EXCLUDED_PROPERTY_POINTERS[(node_type, pointer)]


def is_known_head_id(head_id: str) -> bool:
    """True iff `head_id` is used by at least one mapped pointer.

    Every head_id used here also names a `trainable_group_registry` group
    (see `registry_projection`'s hash-bound cross-check), so this is
    equivalent to "known and reachable", not just "spelled correctly".
    """
    return head_id in set(_HEAD_ASSIGNMENTS.values())


def registry_projection() -> dict[str, Any]:
    """Deterministic hash projection (RFC004 §3.1 shape)."""
    return {
        "property_head_registry_schema_version": "0.1.0",
        "mapped": [
            {"node_type": nt, "node_property_pointer": p, "head_id": head}
            for (nt, p), head in sorted(_HEAD_ASSIGNMENTS.items())
        ],
        "excluded": [
            {"node_type": nt, "node_property_pointer": p, "reason": reason}
            for (nt, p), reason in sorted(EXCLUDED_PROPERTY_POINTERS.items())
        ],
    }


PROPERTY_HEAD_REGISTRY_HASH: str = domain_hash(_REGISTRY_DOMAIN, registry_projection())

# Every head_id referenced above must be a real trainable_group_registry
# group id — checked at import time so a typo or drift fails immediately,
# not only when the cross-check test happens to run. Guarded explicitly
# for an empty `_HEAD_ASSIGNMENTS`: without the guard, a loop that never
# runs would fall through to `del _head_id` and raise a confusing
# `NameError` instead of naming what is actually wrong.
_head_ids_to_check = set(_HEAD_ASSIGNMENTS.values())
if not _head_ids_to_check:
    raise PropertyHeadRegistryError(
        "property_head_registry._HEAD_ASSIGNMENTS is empty; there is "
        "nothing to cross-check against trainable_group_registry"
    )
for _head_id in _head_ids_to_check:
    if not is_known_group_id(_head_id):
        raise PropertyHeadRegistryError(
            f"property_head_registry references head_id {_head_id!r}, which "
            "is not a known trainable_group_registry group id"
        )
del _head_id, _head_ids_to_check


def head_registry_hash() -> str:
    """The `head_registry_hash` an `s1_run_manifest` / `LossUnit` would
    bind, over :func:`registry_projection` under this module's own domain.

    **Using this value is a milestone decision with a full cache rebuild
    attached, not a configuration change.** As of spec-s1-run-manifest
    (2026-09-06), this function has exactly one caller, and it lives
    outside this repository: the owner's bindings script
    (`~/unflatten-data/build_s1_bindings.py`) is the sole producer. Its
    output enters the exact-text cache identity as the seventh
    `--bindings` key, `head_registry_hash` (cache version 2.0.0), and
    from there into every `LossUnit`. Inside this repository,
    `run_binding.build_s1_run_manifest` and
    `unflatten.model.freeze_s1_run_manifest.freeze` only *read* the value off
    that already-bound identity -- neither one calls this function, and
    `tests/model/test_policy_producers.py:537-572`
    guards that separation. The mechanical availability of a hash is still
    not authorization to bind one; that decision is made once, outside the
    repo, by the owner.

    Two facts make that concrete:

    * `RunManifestBindingVerifier` refuses `gradient_bearing=True` while
      `head_registry_hash` is null (`run_binding.py:192-197`), and the M1
      exact-text design (§8 rule 3, §16 criterion 14) is explicit about
      why -- "M1 ships no head module (§12); the field exists so the
      omission is visible and lock-checkable, not so it can be
      forgotten". Supplying
      a hash here removes the *symptom* of the missing head module, not
      the missing head module.
    * Every M1 `LossUnit` carries `head_registry_hash = null` **inside its
      identity** (M1 exact-text design §3, §18.1 item 6), so a non-null
      value changes every
      `loss_unit_id`, therefore `loss_unit_index`, therefore every shard
      row and the cache identity that binds them. There is no valid
      in-place edit: the entire exact-text cache must be re-materialized
      from the immutable RFC003 authority with the hash bound from the
      start.

    So this function exists to make the value *producible when the owner
    decides to produce it*, and for no other purpose. It computes; it does
    not authorize.
    """
    return PROPERTY_HEAD_REGISTRY_HASH

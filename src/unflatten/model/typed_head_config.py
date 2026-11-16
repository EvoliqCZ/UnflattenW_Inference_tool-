"""RFC004 typed head config literals -- design `spec-typed-heads-missing-
families` §10.1 T1, §7, §9 (R12/R13).

Three closed, versioned pieces of configuration the new typed heads need
that the exact-text cache identity does not carry (design §7's
`typed_head_config_hash` gap, T8): the per-repeated-container cardinality
cap (`K_TABS`/`K_TABLE_COLUMNS`/`K_SECTION_WIDTHS`), the shared colour
palette (`COLOR_PALETTE_V0`), and the `list:/marker/suffix` vocabulary
(`MARKER_SUFFIX_VOCABULARY_V0`). `FONT_VOCABULARY_V0` is prepared but
unused until Q1/erratum RFC003 §29 (design §5.2).

**Measured, not guessed** (`docs/runs/typed-heads-train-census.md`, T1):
this module's pure functions (`cardinality_caps_from_census`,
`color_palette_from_census`, `marker_suffix_vocabulary_from_census`) derive
each literal from a `s4_typed_census.build_census` payload's `counts`; the
literals below are exactly what those functions returned on
`cache-exact-text-v14-r4/train` (cardinality caps: `train` UNION `val_iid`,
because a cap a validation page exceeds would silently misclassify a
legitimate count as `cardinality_over_cap` -- §4.2's "never truncate"
sits on caps that must cover both cohorts, not train alone). Pinned as
literals, not recomputed at import time, so a cache rebuild cannot
silently re-choose them (design §9 R12/R13): `tests/model/
test_typed_head_config.py` reproduces every one of them from the checked-
in census fixtures and fails if the literal and the recomputation diverge.

Not consulted by any T1 code path -- `typed_head_config_hash()` exists so
a LATER task (T8) can bind it into the run manifest, checkpoint and census
payload; nothing here authorises a gradient update.
"""

from __future__ import annotations

import re as _re
from collections.abc import Mapping
from typing import Any

from unflatten.model import node_visual_aligner
from unflatten.model.hashing import domain_hash
from unflatten.model.property_head_registry import head_for_pointer
from unflatten.model.typed_property_pointer_map import DEFERRED, _defs, _resolve
from unflatten.renderer.fonts import DEFAULT_FALLBACK_MAP, PILOT_FONT_PACK

#: Design §4.9 registry row `rfc004.typed-head-config/v0`. Owned here: no
#: other module computes a `typed_head_config_hash`.
_TYPED_HEAD_CONFIG_DOMAIN = "rfc004.typed-head-config/v0"

#: `s4_typed_census.value_hist`'s key for `list:/marker/suffix`, excluded
#: from `color_palette_from_census` on purpose: it shares the raw
#: `value_hist` shape with the 21 colour pointers, but its values are
#: never colours.
_SUFFIX_HIST_KEY = "list|/marker/suffix"

#: Values every colour/suffix histogram already carries as an explicit
#: non-colour state -- excluded from the palette body, never from the
#: value totals (design §5.1: `null`/`auto` are their own discrete states,
#: not the least-common hex tail this threshold is pruning).
_NON_PALETTE_LABELS = frozenset({"null", "auto"})

#: The three repeated containers `s4_typed_census.cardinality_hist`/
#: `cardinality_max` key by container pointer (`typed_property_pointer_map.
#: repeated_containers()`), each named by the config attribute its cap
#: belongs to.
_CARDINALITY_CAP_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("/layout/tabs", "K_TABS"),
    ("/columns", "K_TABLE_COLUMNS"),
    ("/columns/widths_mpt", "K_SECTION_WIDTHS"),
)


# ---------------------------------------------------------------------------
# T4 (design §5.1/§10.1): colour and border-object pointers, derived from
# `DEFERRED` -- never hand-listed, so a registry change moves these sets
# instead of silently drifting from them.
# ---------------------------------------------------------------------------

#: The 3 DEFERRED `string` pointers that get their OWN family, not the
#: shared colour palette (design §5.1's rozpis: 21 colour + these 3 = 24
#: DEFERRED strings).
_LANGUAGE_POINTER: tuple[str, str] = ("run", "/language")
_FONT_FAMILY_POINTER: tuple[str, str] = ("run", "/style/font_family")
_MARKER_SUFFIX_POINTER: tuple[str, str] = ("list", "/marker/suffix")
_NON_COLOR_STRING_POINTERS = frozenset(
    {_LANGUAGE_POINTER, _FONT_FAMILY_POINTER, _MARKER_SUFFIX_POINTER}
)

#: The 21 colour pointers (design §5.1 rozpis): every DEFERRED `string`
#: pointer except the 3 above. Sorted so iteration order is deterministic
#: for anything that builds a hash or a report from it.
COLOR_POINTERS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        key for key, value_type in DEFERRED.items()
        if value_type == "string" and key not in _NON_COLOR_STRING_POINTERS
    )
)

#: The 4 DEFERRED `object` pointers -- every nullable `borders`/`sides`
#: container (design §5.1: `{paragraph,table,cell}:/style/borders`,
#: `section:/page_borders/sides`).
BORDER_OBJECT_POINTERS: tuple[tuple[str, str], ...] = tuple(
    sorted(key for key, value_type in DEFERRED.items() if value_type == "object")
)


def _is_nullable_color(node_type: str, pointer: str) -> bool:
    """R9 (design §9/§10.1 T4): `True` iff the pointer's RAW (unresolved)
    schema is exactly `{"$ref": "#/$defs/nullable_color"}` -- nullability
    comes from the schema, never a hand-maintained list. Intermediate
    segments ARE resolved (`_resolve`, following `$ref`s such as `style` ->
    `paragraph_style`); only the FINAL leaf's raw subschema is compared
    unresolved, because resolving it would erase the very distinction
    (`color` vs `nullable_color`) this function exists to read.
    """
    segments = pointer.strip("/").split("/")
    subschema: Any = _defs()[node_type]
    for index, segment in enumerate(segments):
        subschema = _resolve(subschema)
        if not isinstance(subschema, dict) or not isinstance(subschema.get("properties"), dict):
            return False
        raw = subschema["properties"].get(segment)
        if raw is None:
            return False
        if index == len(segments) - 1:
            return isinstance(raw, dict) and raw.get("$ref") == "#/$defs/nullable_color"
        subschema = raw
    return False


#: Exactly 4 (design §9 R9, rev 3 §5.1): `run:/style/highlight` and the 3
#: `{paragraph,table,cell}:/style/background` pointers -- `border_side.
#: color` is `$defs/color` (never nullable), so none of the 12 border-side
#: colour pointers or the 4 `section:/page_borders/sides/*/color` pointers
#: are in this set. The remaining 17 `COLOR_POINTERS` get a `PaletteHead`
#: with NO `null` class at all (`palette_size + 1` outputs, not `+ 2`).
NULLABLE_COLOR_POINTERS: frozenset[tuple[str, str]] = frozenset(
    key for key in COLOR_POINTERS if _is_nullable_color(*key)
)

_COLOR_POINTER_SET = frozenset(COLOR_POINTERS)
_BORDER_OBJECT_POINTER_SET = frozenset(BORDER_OBJECT_POINTERS)


def family_override(node_type: str, pointer: str) -> str | None:
    """`"palette"`, `"borders_null"`, or `None` -- the ONE place `collate_
    typed`, `candidate_model._build_typed_head_group` and `typed_decode.
    decode_typed` consult BEFORE `FAMILY`/`DEFERRED` (design §10.1 T4), so
    a colour/borders-object pointer is recognised identically at all three
    call sites instead of each re-deriving membership by hand."""
    key = (node_type, pointer)
    if key in _COLOR_POINTER_SET:
        return "palette"
    if key in _BORDER_OBJECT_POINTER_SET:
        return "borders_null"
    return None


def palette_domain(*, nullable: bool) -> tuple[str, ...]:
    """The shared palette's class order: `("auto", *COLOR_PALETTE_V0)`,
    plus a trailing `"null"` class ONLY when `nullable` -- a non-nullable
    pointer's head has one FEWER class, not a masked-out one (R9)."""
    domain = ("auto", *COLOR_PALETTE_V0)
    if nullable:
        domain = (*domain, "null")
    return domain


#: The 4 border sides and 3 leaf fields every `borders` object carries
#: (schema `$defs/border_side`) -- R11 (design §9): decode must emit all
#: 12, in this fixed (side, field) order, or a missing one is a schema
#: refusal (`SCHEMA_VALIDATION_ERROR`), never a silent partial object.
BORDER_SIDES: tuple[str, ...] = ("top", "right", "bottom", "left")
BORDER_LEAF_FIELDS: tuple[str, ...] = ("style", "width_mpt", "color")


def border_leaf_pointers(container_pointer: str) -> tuple[str, ...]:
    """The 12 leaf pointers of one `borders`-typed container, e.g.
    `/style/borders/top/style`, `/style/borders/top/width_mpt`, ...,
    `/style/borders/left/color` -- side-major, field-minor, matching
    `BORDER_SIDES`/`BORDER_LEAF_FIELDS` order exactly."""
    return tuple(
        f"{container_pointer}/{side}/{field}"
        for side in BORDER_SIDES
        for field in BORDER_LEAF_FIELDS
    )


def cardinality_caps_from_census(*censuses: Mapping[str, Any]) -> dict[str, int]:
    """`{config attribute: cap}` = the MAXIMUM `cardinality_max[container]`
    across every presented census's `counts` (or bare `counts` payload).

    Takes one or more census payloads (or their already-unwrapped `counts`
    dict) so a caller can combine `train` and `val_iid`: a cap a validation
    page exceeds would misclassify a real count as `cardinality_over_cap`
    (design §4.2), so the cap must cover every cohort a run will read, not
    only the one it happened to be measured on first.
    """
    merged_max: dict[str, int] = {}
    for census in censuses:
        counts = census["counts"] if "counts" in census else census
        for container_pointer, cap in counts["cardinality_max"].items():
            merged_max[container_pointer] = max(merged_max.get(container_pointer, 0), cap)
    return {
        attribute: merged_max[container_pointer]
        for container_pointer, attribute in _CARDINALITY_CAP_ATTRIBUTES
    }


def _color_totals(value_hist: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for pointer_key, histogram in value_hist.items():
        if pointer_key == _SUFFIX_HIST_KEY:
            continue
        for label, n in histogram.items():
            if label in _NON_PALETTE_LABELS:
                continue
            totals[label] = totals.get(label, 0) + n
    return totals


def min_count_for_coverage(
    totals: Mapping[str, int], *, coverage_micro: int = 995_000
) -> tuple[int, int]:
    """`(min_count, achieved_coverage_micro)`: the LARGEST occurrence-count
    threshold whose inclusion still covers at least `coverage_micro` / 1e6
    of `sum(totals.values())` -- the tightest (smallest) palette that does
    not drop below the coverage floor (design §5.1/Q6). Integer micro-units
    throughout, matching the project's other rate literals (no float in a
    pinned config).
    """
    total_all = sum(totals.values())
    if total_all == 0:
        raise ValueError("min_count_for_coverage: totals is empty")
    best_threshold = 1
    best_coverage_micro = 1_000_000
    for threshold in sorted(set(totals.values())):
        covered = sum(n for n in totals.values() if n >= threshold)
        achieved_micro = (covered * 1_000_000) // total_all
        if achieved_micro >= coverage_micro:
            best_threshold = threshold
            best_coverage_micro = achieved_micro
    return best_threshold, best_coverage_micro


def color_palette_from_census(
    value_hist: Mapping[str, Mapping[str, int]], *, min_count: int
) -> tuple[str, ...]:
    """Sorted tuple of hex colours (excludes `null`/`auto`) whose total
    occurrence across every colour pointer's histogram is `>= min_count`
    (design §5.1: one shared palette, not one per pointer)."""
    totals = _color_totals(value_hist)
    return tuple(sorted(label for label, n in totals.items() if n >= min_count))


def marker_suffix_vocabulary_from_census(
    value_hist: Mapping[str, Mapping[str, int]], *, min_count: int
) -> tuple[str, ...]:
    """Sorted tuple of `list:/marker/suffix` values (including the literal
    string `"null"` for the observed-null state) whose occurrence is
    `>= min_count`.

    The measured train-wide domain is far larger than the 24-fixture/512-
    page val sample the design's §1 row 4 / §5.4 cited (**143** distinct
    values, not 6) -- almost all of it numbered-list `lvlText` templates in
    a long tail of languages (`"Article %1"`, `"Rozdział %1"`, `"K čl.
    %1:"`, ...). The same coverage-threshold reasoning as the colour
    palette (`min_count_for_coverage`) is applied here for the same
    reason: a closed vocabulary head cannot emit a value it was never
    given a class for, so the tail is EXCLUDED and counted, never folded
    into the nearest kept value.
    """
    histogram = value_hist[_SUFFIX_HIST_KEY]
    return tuple(sorted(label for label, n in histogram.items() if n >= min_count))


# ---------------------------------------------------------------------------
# Pinned literals -- measured 2026-09-23 on `cache-exact-text-v14-r4`
# (`docs/runs/typed-heads-train-census.md`). Reproduced from the checked-in
# fixtures by `tests/model/test_typed_head_config.py`.
# ---------------------------------------------------------------------------

#: `train` UNION `val_iid` cardinality maxima. Matches the pre-existing
#: `TypedRepeatedCardinality` docstring's cited full-release maxima
#: (8 / 64 / 118) exactly once the two cohorts are combined -- `train`
#: alone under-measures `K_SECTION_WIDTHS` (max 6; `val_iid` reaches 8).
K_TABS: int = 64
K_TABLE_COLUMNS: int = 118
K_SECTION_WIDTHS: int = 8

#: The colour-occurrence threshold that keeps train coverage `>= 99.5%`
#: while excluding the rarest tail (measured coverage: 99.5009 %, i.e.
#: `995_009` micro, integer-floor).
MIN_COUNT: int = 35
MIN_COUNT_ACHIEVED_COVERAGE_MICRO: int = 995_009

#: 733 hex values (measured; excludes the separately-modelled `null`/
#: `auto` states).
COLOR_PALETTE_V0: tuple[str, ...] = (
    '#000000', '#000001', '#000005', '#000006', '#000009', '#00000A',
    '#000066', '#00007F', '#000080', '#000090', '#000096', '#000099',
    '#0000CC', '#0000DC', '#0000FF', '#000101', '#001F5F', '#002060',
    '#002147', '#002664', '#002858', '#002A5C', '#002F63', '#002F8E',
    '#003087', '#003366', '#00338D', '#0033CC', '#00346D', '#003865',
    '#003E7E', '#004379', '#004400', '#004682', '#004976', '#004C97',
    '#004EA8', '#004F86', '#005288', '#005941', '#005A77', '#005A84',
    '#005B67', '#005C40', '#005DAA', '#006000', '#0060A5', '#0060A8',
    '#006272', '#0063AA', '#006600', '#0066CC', '#0066FF', '#006AB3',
    '#006F90', '#006FC0', '#0070C0', '#0072CC', '#007670', '#0076BD',
    '#008000', '#008080', '#0085AC', '#0085CA', '#0092DD', '#009391',
    '#0093A1', '#009DDC', '#009EE0', '#00AEEF', '#00AF50', '#00B050',
    '#00B0F0', '#00B9F2', '#00FF00', '#00FFFF', '#010000', '#010101',
    '#013366', '#013C59', '#0200C9', '#02180D', '#03202F', '#032093',
    '#0432FF', '#0462C1', '#050505', '#051532', '#05497E', '#0562C1',
    '#0563C1', '#06080F', '#062172', '#070707', '#080808', '#083E53',
    '#08C2DD', '#0983D0', '#0A0A0A', '#0B0C0C', '#0C1975', '#0C327B',
    '#0C4A9F', '#0D0D0D', '#0E0E0E', '#0E101A', '#0E4194', '#0F0F00',
    '#0F0F0C', '#0F0F0F', '#0F243E', '#0F4761', '#101000', '#101010',
    '#10137D', '#104F75', '#111100', '#111111', '#1155CC', '#121212',
    '#12263F', '#122BBA', '#131313', '#13171A', '#133C66', '#141414',
    '#146CFD', '#149613', '#15191E', '#156082', '#158045', '#1593CB',
    '#159A90', '#161616', '#169FDA', '#16AD85', '#171717', '#17365D',
    '#17375D', '#17375F', '#1790D9', '#181717', '#181818', '#18181A',
    '#191919', '#191970', '#193C67', '#1A1617', '#1A181D', '#1A1A1A',
    '#1B1B1B', '#1B1C1D', '#1C1C1C', '#1C1E21', '#1C1F23', '#1C2024',
    '#1C4586', '#1CCB1C', '#1D1B11', '#1D1D1B', '#1D2126', '#1D2256',
    '#1D2A57', '#1D70B8', '#1D71B8', '#1E1E1E', '#1E2228', '#1E4778',
    '#1F1F1F', '#1F1F20', '#1F3763', '#1F3864', '#1F467A', '#1F487C',
    '#1F497D', '#1F4D78', '#1F4E79', '#1F62AA', '#20007E', '#201547',
    '#201F1E', '#201F1F', '#202020', '#202122', '#202124', '#202428',
    '#202429', '#203764', '#205E9E', '#211D1E', '#211E1E', '#211F1F',
    '#212121', '#212529', '#212B32', '#212B55', '#21347B', '#215868',
    '#221E1F', '#221F1F', '#222222', '#22272B', '#22413A', '#22A469',
    '#231F1F', '#231F20', '#232323', '#232C67', '#2419F5', '#242424',
    '#243F60', '#244061', '#244062', '#252525', '#25313B', '#254061',
    '#25478B', '#262626', '#263238', '#26352A', '#263673', '#266F8B',
    '#2683C6', '#272727', '#272F38', '#282828', '#283C46', '#291F51',
    '#292929', '#292C2E', '#2980B9', '#2A2A24', '#2A2A2A', '#2A2D35',
    '#2B2B2B', '#2B2C38', '#2B3D4F', '#2B6B83', '#2B92BE', '#2C2927',
    '#2C2A2A', '#2C2C2C', '#2D2D2D', '#2D3336', '#2D3338', '#2D6D6E',
    '#2E6D30', '#2E74B5', '#2E759E', '#2F0E72', '#2F2F2F', '#2F3338',
    '#2F343A', '#2F5496', '#303030', '#303152', '#3051A5', '#3078BA',
    '#312E89', '#313131', '#313335', '#313341', '#313638', '#313D4C',
    '#316F72', '#31849B', '#3190C9', '#323130', '#323232', '#32323E',
    '#32325D', '#323E4F', '#32566E', '#333300', '#333333', '#333399',
    '#3333FF', '#333E48', '#336699', '#3366CC', '#3366FF', '#337AB7',
    '#339933', '#339966', '#343433', '#343641', '#347186', '#3494BA',
    '#363435', '#363636', '#363838', '#363839', '#36384A', '#365F91',
    '#366091', '#373737', '#3844AC', '#385623', '#385C6B', '#392C69',
    '#3A1D54', '#3A3939', '#3A3A3A', '#3A4972', '#3B3838', '#3B3C42',
    '#3B5463', '#3B6E8F', '#3C3C3B', '#3C3C3C', '#3C960E', '#3D4249',
    '#3E3E3E', '#3F3A38', '#3F3F3F', '#400080', '#40382D', '#403F41',
    '#404040', '#404246', '#414042', '#414751', '#424242', '#425563',
    '#426092', '#42ADAD', '#434343', '#43D596', '#443C2F', '#444444',
    '#44546A', '#4471C4', '#4472C4', '#464646', '#467886', '#4681B3',
    '#484848', '#494949', '#495F6C', '#4A4A4A', '#4A6822', '#4AA55B',
    '#4AACC5', '#4B4949', '#4B4B4B', '#4BACC6', '#4C4C4C', '#4D4D4C',
    '#4D4D4D', '#4D4D4F', '#4D5356', '#4DAED0', '#4E5065', '#4F4F4E',
    '#4F5CD6', '#4F6228', '#4F81BC', '#4F81BD', '#505050', '#505150',
    '#5074B9', '#522E91', '#523888', '#523D91', '#53575A', '#538135',
    '#545860', '#545D8A', '#548DD4', '#555554', '#555555', '#565656',
    '#571925', '#585858', '#58595B', '#595958', '#595959', '#596CDD',
    '#5A5A5A', '#5B5B5B', '#5B6770', '#5B9BD4', '#5B9BD5', '#5D8DB6',
    '#5DBA47', '#5F497A', '#5F5F5F', '#5F6062', '#622423', '#632423',
    '#63696C', '#646363', '#652F76', '#660000', '#660033', '#666666',
    '#690039', '#696969', '#6A0E75', '#6A2875', '#6B2876', '#6B2976',
    '#6BE7E7', '#6C6C6C', '#6CB33F', '#6CBE7F', '#6D5560', '#6D6E71',
    '#6E6E6E', '#6F2F9F', '#6F6D5C', '#7030A0', '#707070', '#707173',
    '#70AD47', '#70AFD9', '#737373', '#74737B', '#757171', '#75B6C0',
    '#767171', '#767878', '#76923C', '#774B30', '#776E64', '#777777',
    '#78004F', '#7800C8', '#78C0D4', '#797979', '#7A003C', '#7BA0CD',
    '#7C7C7C', '#7CADD9', '#7D7D7D', '#7F7F7F', '#7FA1B6', '#7FC0DB',
    '#800000', '#800080', '#804E88', '#806000', '#8064A2', '#808000',
    '#808080', '#82EAB1', '#833C0B', '#83F880', '#8496B0', '#84B3DF',
    '#85C440', '#87189D', '#87888A', '#87DCFF', '#8B1A4A', '#8B5FBF',
    '#8C8C96', '#8DB3E2', '#8DB4E2', '#8DBC40', '#8EA9E9', '#8EAADB',
    '#909090', '#92CDDC', '#92D050', '#939598', '#93D07C', '#943634',
    '#948A54', '#949699', '#94C600', '#95B3D7', '#95D0E3', '#96004B',
    '#969696', '#98361E', '#984806', '#989C9E', '#98BE3B', '#9900FF',
    '#993300', '#993366', '#999999', '#99CCFF', '#99FF99', '#9BBB59',
    '#9C9C9C', '#9CC2E5', '#9EC5D6', '#9F9F9F', '#9FC5E8', '#A00045',
    '#A078AA', '#A2AD00', '#A3A3A3', '#A558A0', '#A5A5A5', '#A5C9EB',
    '#A6A6A6', '#A7CCEE', '#A80000', '#A8D08D', '#A98899', '#AAAAAA',
    '#AAD3E4', '#ABABAB', '#ADDDFF', '#AEAAAA', '#B00A32', '#B0B7BB',
    '#B1B3B6', '#B2A1C7', '#B3B3B3', '#B4C6E7', '#B52D79', '#B6985E',
    '#B6DDE8', '#B7ECFF', '#B8CCE3', '#B8CCE4', '#B962C8', '#B99ED8',
    '#B9B9B9', '#B9BAB9', '#BA0C2F', '#BA94FF', '#BAE2DF', '#BC1A25',
    '#BCCBDC', '#BCD5ED', '#BDD6EE', '#BDD7EE', '#BE8F00', '#BEBEBE',
    '#BF8F00', '#BFBFBF', '#C00000', '#C0504D', '#C0C0C0', '#C1C1C1',
    '#C1C7D0', '#C26161', '#C2D69B', '#C45911', '#C4BC96', '#C5CFD9',
    '#C5E0B3', '#C6BCDB', '#C6D9F1', '#C7DAF1', '#C8C8C8', '#C8CACB',
    '#C9211E', '#C9C9C9', '#C9DAF8', '#C9DFFF', '#CAC8D9', '#CACAD9',
    '#CBCBCB', '#CC00CC', '#CC99FF', '#CCC0D9', '#CCCCCC', '#CCCCFF',
    '#CCD8E6', '#CCECFF', '#CCFFCC', '#CCFFFF', '#CDCDCD', '#CDD5D9',
    '#CDD5FE', '#CE202F', '#CECECE', '#CED6E7', '#CED7E7', '#CEEAC7',
    '#CFD7E7', '#D0CECE', '#D0D8E8', '#D22D0F', '#D2AACF', '#D2EAF1',
    '#D31932', '#D3D3D3', '#D3D9D6', '#D3DFEE', '#D3E9ED', '#D3F2E9',
    '#D49988', '#D5D5D5', '#D5DCE4', '#D5F2FE', '#D6E3BC', '#D7D7D7',
    '#D8D1C8', '#D8D8D8', '#D8DFDE', '#D9D9D9', '#D9E1F2', '#D9E2F3',
    '#D9EAD3', '#D9F0FB', '#D9F2D0', '#DADADA', '#DAE2F3', '#DAE9F7',
    '#DAEEF3', '#DBDBDB', '#DBE4F0', '#DBE5F1', '#DCDCDC', '#DCE6F1',
    '#DDD9C3', '#DDDDDD', '#DDE2F0', '#DDEBF7', '#DDFFFF', '#DE0000',
    '#DEEAF6', '#DEEBF6', '#E0E0E0', '#E1C39A', '#E1F6FF', '#E2E2E2',
    '#E2EFD9', '#E2EFDA', '#E3147E', '#E36C09', '#E36C0A', '#E3E2E2',
    '#E3E3E3', '#E3F1F3', '#E4E5E6', '#E5B8B7', '#E5DFEC', '#E5E5E5',
    '#E61E0F', '#E6331B', '#E6E6E6', '#E7E6E6', '#E8E8E8', '#E8EBF3',
    '#E8ECF4', '#E8F8F8', '#E94E1B', '#E9EDF4', '#EAEAEA', '#EAECEE',
    '#EAF1DD', '#EBF1DE', '#EBF9DF', '#EC6607', '#ECECEC', '#ED7D31',
    '#EDEDED', '#EDF0F7', '#EEECE1', '#EEEEEE', '#EEF7F4', '#EFEFEC',
    '#EFEFEF', '#EFF6FF', '#F0A22E', '#F0F0F0', '#F0F0F4', '#F1F1F1',
    '#F2DBDB', '#F2DCDB', '#F2F2F2', '#F2F4FB', '#F30000', '#F39200',
    '#F3CC5F', '#F3F3F3', '#F3FBFB', '#F49515', '#F4B083', '#F4F4F4',
    '#F5844C', '#F59AC3', '#F5F5F5', '#F5F7FD', '#F5FFF3', '#F79646',
    '#F7CAAC', '#F7F7F7', '#F7F8FF', '#F8F8F8', '#F8F9FA', '#F9B074',
    '#F9F900', '#F9F9F9', '#FABF8F', '#FAFAFA', '#FAFAFD', '#FAFCE5',
    '#FBD4B4', '#FBE4D5', '#FBFBFB', '#FCD5B4', '#FCE4D6', '#FCFCFC',
    '#FCFDFD', '#FD8209', '#FDE9D9', '#FDF6E7', '#FDFCFA', '#FDFDFD',
    '#FEE99C', '#FEFEFE', '#FF0000', '#FF00FF', '#FF1F64', '#FF5050',
    '#FF6600', '#FF8040', '#FF9300', '#FF9900', '#FF9999', '#FFC000',
    '#FFCC00', '#FFCC99', '#FFCCCC', '#FFCCFF', '#FFD966', '#FFDF94',
    '#FFE599', '#FFF2CC', '#FFFAEB', '#FFFF00', '#FFFF99', '#FFFFCC',
    '#FFFFFF'
)

#: Marker-suffix threshold (§1 row 4, module docstring above): the
#: coverage-threshold reasoning is the same as the colour palette, applied
#: separately because the suffix domain is a distinct, unrelated string
#: family. 53 values at coverage 99.5242 % (`995_242` micro).
MIN_COUNT_MARKER_SUFFIX: int = 12
MIN_COUNT_MARKER_SUFFIX_ACHIEVED_COVERAGE_MICRO: int = 995_242

MARKER_SUFFIX_VOCABULARY_V0: tuple[str, ...] = (
    "", " -", "(%1)", ")", ") ", ").", "*", "-", ".", ". ", ".)", ".0",
    "1.%1", "1.0%1", "2.%1", "3.%1", "4.%1", "4.%1.", "5.%1", "5.%1.",
    "7.%1", "8.%1", "8.%1.", "9-%1.", "9.%1", "9.%1.", "AGENDA ITEM %1",
    "Annex %1", "Art. %1.", "Article %1", "Article %1.", "Artículo %1.",
    "Artículo %1.-", "B.%1.", "C.%1.", "D%1.", "D.", "Item %1", "Item %1.",
    "K čl. %1:", "PART %1 - ", "R%1.", "Requirement %1", "Rozdział %1",
    "Schedule %1", "Section %1.", "[%1]", "null", "§ %1.", "°", "°)",
    "Článek %1.", "ČÁST\xa0%1",
)

#: Prepared for T6 (design §5.2, BLOCKED on Q1 + RFC003 §29 erratum), not
#: used by any T1 code path. Derived from the pinned font policy, not from
#: observed data: `PILOT_FONT_PACK ∪ keys(DEFAULT_FALLBACK_MAP)`. Measured
#: coverage under this policy (`docs/runs/typed-heads-train-census.md`):
#: `in_pack` 3 659 + `fallback_map` 2 328 594 of 2 513 118 font_family
#: targets (92.68 %), `fallback_default` (DejaVu Sans, excluded from this
#: vocabulary) 180 865 (7.20 %), `unresolved` 0.
#:
#: **Deliberately OUTSIDE `typed_head_config_projection()`** (design §7 rev
#: 3, fix loop 1 should_fix 3): no typed head reads this vocabulary until
#: T6 lands, so binding it into the hash today would refuse every stamped
#: checkpoint/census over an edit to `DEFAULT_FALLBACK_MAP`/
#: `PILOT_FONT_PACK` that no head cares about yet.
FONT_VOCABULARY_V0: tuple[str, ...] = tuple(sorted(PILOT_FONT_PACK | set(DEFAULT_FALLBACK_MAP)))

#: Load-time derivation policy versions (design §7 rev 3): a SEMANTIC
#: change to either policy -- not just a literal it reads -- must move
#: `typed_head_config_hash` even though neither `zero_count.v0`
#: (`s4_typed_loss._zero_count_records`) nor `borders_null.v0`
#: (`s4_typed_loss._borders_null_records`) has a literal configuration
#: table of its own to bind.
ZERO_COUNT_VERSION = "zero_count.v0"
BORDERS_NULL_VERSION = "borders_null.v0"


# ---------------------------------------------------------------------------
# T8 (design §7, §9 R14a-d, §10.1 T8) -- identity for the `*`-pointer heads
# T3 replaces with a `SlotHead`, and the warm-start extension point T2/T4
# grow into. Neither T3's `SlotHead` nor any T2/T4 head module exists yet;
# everything below is a NAMING CONVENTION `s4_train.replaced_keys` /
# `s4_train.declared_new_keys` read off a real model's `state_dict()`, so
# T3 only has to build a module that MATCHES it -- never a class this
# module imports.
# ---------------------------------------------------------------------------

#: The child attribute a `SlotHead`'s `nn.ModuleList` of per-slot leaf
#: heads MUST be named -- `T3` must build `SlotHead` with an attribute
#: named exactly `slots` (an `nn.ModuleList`), so its state_dict keys are
#: `"{child_attribute}.{pointer_attribute}.slots.{i}.<leaf-suffix>"`; the
#: `<leaf-suffix>` (e.g. `linear.weight`, `regression_head.weight`) is
#: whatever the SAME family's pre-T3 single-instance head already uses
#: under `"{child_attribute}.{pointer_attribute}.<leaf-suffix>"` -- warm
#: start matches slot 0..K-1 to the single parent tensor by that shared
#: suffix, purely from `model.state_dict()` key strings.
SLOT_HEAD_LIST_ATTRIBUTE = "slots"

#: The 5 `*`-pointer star groups T3 replaces (design §7 rev 3, MEASURED on
#: `runs/s4-discovery-v14-r4/checkpoint.pt`, 299 keys): each entry names
#: the `candidate_model.typed_head_children()` child attribute the head
#: lives under, the `candidate_model._pointer_attribute()`-derived module
#: name under that child, and the `typed_head_config` cap attribute above
#: that bounds its slot count. Part of `typed_head_config_projection()`
#: (design §7: "the list of `replaced_keys`") -- adding, removing or
#: repointing a star group changes `typed_head_config_hash`.
STAR_POINTERS: tuple[dict[str, str], ...] = (
    {
        "child_attribute": "expert_adapter_dense_heads",
        "pointer_attribute": "layout_tabs_alignment",
        "cap_attribute": "K_TABS",
    },
    {
        "child_attribute": "expert_adapter_dense_heads",
        "pointer_attribute": "layout_tabs_leader",
        "cap_attribute": "K_TABS",
    },
    {
        "child_attribute": "expert_adapter_dense_heads",
        "pointer_attribute": "layout_tabs_position_mpt",
        "cap_attribute": "K_TABS",
    },
    {
        "child_attribute": "expert_adapter_dense_heads",
        "pointer_attribute": "columns_widths_mpt",
        "cap_attribute": "K_SECTION_WIDTHS",
    },
    {
        "child_attribute": "table_plan",
        "pointer_attribute": "columns_width_mpt",
        "cap_attribute": "K_TABLE_COLUMNS",
    },
)

#: The repeated containers that get a COUNT head (design §4.1, §10.1 T2) --
#: `(node_type, container_pointer, cap_attribute)`. Deliberately NOT
#: `repeated_containers()` (`typed_property_pointer_map.py`): of the three
#: pinned containers, only `paragraph:/layout/tabs` has no other source of
#: its element count. `table:/columns` (`table.columns.from_grid.v0`) and
#: `section:/columns/widths_mpt` (`section.columns.widths_mpt.length.v0`)
#: are derived deterministically from what the model already emits (§4.1's
#: REJECTED row: a second head would risk `TYPED_HEAD_DUPLICATE_PROPERTY`
#: and a count that disagrees with the model's own column grid). This list
#: is what `candidate_model._build_typed_head_group` and
#: `s4_typed_loss.collate_typed`/`typed_decode.decode_typed` consult
#: instead of the property head registry -- a count is a tensor identity,
#: not an RFC001 pointer (`head_registry_hash` precedent:
#: `table_plan/grid_column_count`).
CARDINALITY_HEADS: tuple[tuple[str, str, str], ...] = (
    ("paragraph", "/layout/tabs", "K_TABS"),
)

def pointer_supervision_key(head_id: str, node_type: str, pointer: str) -> str:
    """The stable string identity `s4_run_manifest`'s per-pointer
    supervision floor (design §9 R8) keys its disclosed-pointer entries
    by. Built from the SAME `(head_id, node_type, pointer)` triple
    `new_supervised_pointers` returns -- never re-derived by a caller."""
    return f"{head_id}|{node_type}:{pointer}"


def new_supervised_pointers() -> tuple[tuple[str, str, str], ...]:
    """`(head_id, node_type, pointer)` for every NEW pointer a T2/T4 head
    was built for that `s4_typed_census` can ACTUALLY measure a real
    `supervised` count for (design §9 R8, §10.1 T8 point 4; fix loop 2,
    replacing fix loop 1's wrong membership) -- the 21 colour pointers
    ONLY.

    Two categories are deliberately excluded, disclosed here rather than
    silently measured as a fabricated `0` (fix loop 1's actual defect,
    caught in review):

    - The `/layout/tabs` cardinality COUNT (`CARDINALITY_HEADS`) is not a
      `TypedPropertyTarget` at all -- it lives in `typed_cardinalities`, a
      record type `s4_typed_census._partial_census` never folds into
      `counts.per_pointer` (that dict is built solely from
      `row["typed_targets"]`). Its `zero_count.v0` synthetic supervision
      exists only at S4 training load time (`s4_typed_loss.
      _zero_count_records`), not in the census. A real prod census
      therefore reports this key as ABSENT (read as `0` by
      `new_pointer_supervised_counts`), while the true count is in the
      hundreds of thousands -- a fabricated, not measured, zero.
    - The 4 `BORDER_OBJECT_POINTERS` (`borders_null.v0`) are classified
      `deferred` by `s4_typed_census._classify` for BOTH the `True` and
      `False` states unconditionally (that module's own docstring: the
      `False` ("has border leaves") branch needs cross-record synthesis
      the per-record classifier does not attempt, out of scope there by
      design). Their `supervised` bucket is 0 by construction on every
      census, never a fact about how much real supervision exists.

    Should either become census-measurable (T1's `_partial_census`
    extended with a `typed_cardinalities`-derived supervised tally, or a
    `borders_null` cross-record pass), this function is where they would
    be added back -- WITH a test proving they are (design's own guard-
    must-be-provably-failable rule), not restored on the strength of this
    docstring alone.

    Star-pointer SLOTS (design §4.3) are excluded for a different, older
    reason: each slot's per-index census entry is a FRACTION of its
    leaf's total supervision by construction (`K_TABS = 64` slots share
    one paragraph's worth of tabs), so a per-slot floor would refuse
    almost every real run.
    """
    return tuple(
        sorted(
            (head_for_pointer(node_type, pointer), node_type, pointer)
            for node_type, pointer in COLOR_POINTERS
        )
    )


def star_pointer_cap_attributes() -> dict[str, str]:
    """`{pointer_attribute: cap_attribute}` for every `STAR_POINTERS` entry
    -- what T3's `SlotHead` wiring (`candidate_model._build_typed_head_
    group`) and decode (`typed_decode.decode_typed`) key off of, keyed by
    the SAME `_pointer_attribute` string `s4_typed_loss.collate_typed`
    groups typed targets under, so all three sites recognise a star
    pointer identically without re-deriving `STAR_POINTERS` membership by
    hand. Callers needing the full entry (child attribute, parent-key
    derivation for warm start) still read `STAR_POINTERS` directly --
    this is a narrower, single-purpose view of the same 5 rows."""
    return {entry["pointer_attribute"]: entry["cap_attribute"] for entry in STAR_POINTERS}


#: State-dict key prefixes of head modules a task (T2's `CardinalityHead`,
#: T4's `PaletteHead`/`borders_is_null`) adds that the S3 parent checkpoint
#: never carried AT ALL -- `s4_train.declared_new_keys` reads this list so
#: it never has to be restated in `s4_train.py`, which stays byte-frozen
#: while a run is live (design §5 [governance R1]).
#:
#: T2 (§10.1): `expert_adapter_dense_heads.layout_tabs.` -- the real
#: `runs/s4-discovery-v14-r4/checkpoint.pt` parent (MEASURED, 299 keys)
#: never carried this head at all, so its 2 keys (`linear.weight`,
#: `linear.bias`) must warm-start from THIS module's own initialization
#: (`fresh_source = "absent_in_parent"`), never `S4TrainingRefused` for a
#: real, expected `missing` key.
def _new_head_pointer_attribute(pointer: str) -> str:
    """Duplicated from `candidate_model._pointer_attribute` on purpose --
    importing `candidate_model` here would be circular (it imports THIS
    module). Both apply the same regex to the same pointer strings, so a
    test pins them equal rather than trusting the duplication silently."""
    return _re.sub(r"[^0-9a-zA-Z]+", "_", pointer).strip("_")


def _new_head_child_attribute(head_id: str) -> str:
    """Duplicated from `candidate_model.typed_head_children`'s explicit
    `expert_adapter.dense -> expert_adapter_dense_heads` mapping, for the
    same circular-import reason as `_new_head_pointer_attribute`."""
    if head_id == "expert_adapter.dense":
        return "expert_adapter_dense_heads"
    return head_id.replace(".", "_")


def _new_palette_and_borders_key_prefixes() -> tuple[str, ...]:
    """T4 (§10.1): one `"{child}.{pointer_attr}."` prefix per `COLOR_
    POINTERS`/`BORDER_OBJECT_POINTERS` entry -- every one of these 25
    pointers gets a head the S4 parent checkpoint never carried at all,
    exactly like T2's `layout_tabs`. `BORDER_OBJECT_POINTERS` gets the
    `_is_null` suffix `candidate_model._build_typed_head_group` builds its
    module name under."""
    prefixes = []
    for node_type, pointer in COLOR_POINTERS:
        head_id = head_for_pointer(node_type, pointer)
        child = _new_head_child_attribute(head_id)
        attr = _new_head_pointer_attribute(pointer)
        prefixes.append(f"{child}.{attr}.")
    for node_type, pointer in BORDER_OBJECT_POINTERS:
        head_id = head_for_pointer(node_type, pointer)
        child = _new_head_child_attribute(head_id)
        attr = _new_head_pointer_attribute(pointer) + "_is_null"
        prefixes.append(f"{child}.{attr}.")
    return tuple(sorted(set(prefixes)))


NEW_HEAD_KEY_PREFIXES: tuple[str, ...] = (
    "expert_adapter_dense_heads.layout_tabs.",
    *_new_palette_and_borders_key_prefixes(),
    "node_visual_aligner.",
)


def typed_head_config_projection() -> dict[str, Any]:
    """The hashable projection: every literal above except `MIN_COUNT_*_
    ACHIEVED_COVERAGE_MICRO` (measurement metadata, not configuration --
    changing how well a fixed palette happens to cover a re-measured
    corpus must not move the hash a loaded checkpoint is compared against)
    and `FONT_VOCABULARY_V0` (design §7 rev 3 should_fix 3: prepared for
    T6, consulted by no head today, so it stays out of the hash until T6
    lands).

    `star_pointers` binds the warm-start "list of `replaced_keys`" design
    §7 names: which `*`-pointer groups T3's `SlotHead` replaces, and under
    what cap. `NEW_HEAD_KEY_PREFIXES` is NOT included -- it names key
    PREFIXES, not a closed vocabulary/cap a loaded checkpoint's tensors
    must agree with, so it carries no identity a hash needs to bind.
    `zero_count_v0`/`borders_null_v0` bind the load-time derivation
    policies' own version identifiers (design §7 rev 3).
    """
    return {
        "k_tabs": K_TABS,
        "k_table_columns": K_TABLE_COLUMNS,
        "k_section_widths": K_SECTION_WIDTHS,
        "min_count": MIN_COUNT,
        "color_palette_v0": list(COLOR_PALETTE_V0),
        "min_count_marker_suffix": MIN_COUNT_MARKER_SUFFIX,
        "marker_suffix_vocabulary_v0": list(MARKER_SUFFIX_VOCABULARY_V0),
        "zero_count_v0": ZERO_COUNT_VERSION,
        "borders_null_v0": BORDERS_NULL_VERSION,
        "star_pointers": [dict(entry) for entry in STAR_POINTERS],
        # The aligner is not a property head, but its tensor architecture and
        # load-time status derivation must be checkpoint/census-compatible.
        "node_visual_aligner": {
            "alignment_status_v0": "alignment_status.v0",
            "status_vocabulary": list(node_visual_aligner.STATUS_VOCABULARY),
            "d_align": node_visual_aligner.DEFAULT_D_ALIGN,
            "num_heads": node_visual_aligner.DEFAULT_NUM_HEADS,
            "bbox_ppm_v0": node_visual_aligner.BBOX_PPM_VERSION,
        },
    }


def typed_head_config_hash() -> str:
    """`domain_hash(_TYPED_HEAD_CONFIG_DOMAIN, typed_head_config_projection())`."""
    return domain_hash(_TYPED_HEAD_CONFIG_DOMAIN, typed_head_config_projection())

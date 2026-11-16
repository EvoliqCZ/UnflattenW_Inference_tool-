"""DOCX -> Canonical AST importer (RFC003 §17.2 stages I0-I10, D1 scope).

Scope: `word-core-v0` documents — primarily packages produced by our own
renderer (minimal `styles.xml`, fully explicit effective properties),
with tolerance for simple external DOCX (docDefaults + full basedOn
named-style chains, cycle-safe, depth-capped). The inverse value maps are imported from
`unflatten.renderer.ooxml.*` so the importer can never silently drift
from the renderer's forward mapping.

No silent loss (RFC003 §45): every construct the importer cannot map is
reported as a typed Issue; nothing is dropped without a report entry.
Renderer-technical XML (vMerge continuation cells, synthesized section
boundary paragraphs) is *detected* — since import has no build context
the boundary-paragraph detection is an assumption, always surfaced as
`IMPORT_BOUNDARY_PARAGRAPH_ASSUMED` and recorded in the source map,
never applied silently.

Complex-field nesting rule (RFC001 §30): everything inside a field's
instruction region is field code — including the RESULT runs of an
inner field that properly terminates there. When the inner field ends,
those runs are dropped from the AST like any other instruction-region
content; the drop is recorded in the source map (import_status
"dropped" on each run, plus the field's own `field:` entry) — a
recorded drop, never a silent one. Only if the OUTER field never
terminates is its own pending content re-emitted as visible result
content on drain.
"""

from __future__ import annotations

import copy
import hashlib
import posixpath
import re
import struct
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from io import BytesIO
from typing import Any, Literal

from lxml import etree

from unflatten.ast.canonicalizer import CanonicalizationError, canonicalize
from unflatten.ast.issues import Issue
from unflatten.ast.xml_text import find_illegal_xml_char
from unflatten.dataset.import_ import visual_v1
from unflatten.dataset.import_.rasterize import (
    TARGET_DPI,
    RasterizationError,
    RasterizerUnavailableError,
    rasterize_graphic,
)
from unflatten.dataset.import_.secure_scan import ScanLimits, scan_docx
from unflatten.renderer.ooxml.document_builder import (
    _ALIGNMENT_MAP,
    _BORDER_STYLE_MAP,
    _BREAK_TYPE_MAP,
    _TAB_LEADER_MAP,
    _UNDERLINE_MAP,
)
from unflatten.renderer.ooxml.numbering import _ORDERED_FORMAT_MAP, _UNORDERED_GLYPH
from unflatten.renderer.units import round_half_away_from_zero
from unflatten.renderer.xml_writer import NSMAP

_W = NSMAP["w"]
_WP_NS = NSMAP["wp"]
_A_NS = NSMAP["a"]
_R_NS = NSMAP["r"]
_VML_NS = "urn:schemas-microsoft-com:vml"
_VML_OFFICE_NS = "urn:schemas-microsoft-com:office:office"
_VML_WORD_NS = "urn:schemas-microsoft-com:office:word"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_DC_NS = "http://purl.org/dc/elements/1.1/"

ImportStatus = Literal["success", "success_with_warnings", "failed"]

# Exact inverses of the renderer's forward maps.
_INV_ALIGNMENT = {v: k for k, v in _ALIGNMENT_MAP.items()}
_INV_TAB_LEADER = {v: k for k, v in _TAB_LEADER_MAP.items()}
_INV_UNDERLINE = {v: k for k, v in _UNDERLINE_MAP.items()}
_INV_BORDER_STYLE = {v: k for k, v in _BORDER_STYLE_MAP.items()}
# Additive input-only spellings `_INV_BORDER_STYLE` cannot carry: ECMA-376
# `ST_Border` has both `none` and `nil` (visually identical), but
# `_BORDER_STYLE_MAP` must stay injective (`_append_border_side`,
# `table_builder.py:176`, indexes it directly by AST style value) so `nil`
# cannot be added there without breaking the forward direction. Kept
# disjoint from `_INV_BORDER_STYLE` by construction (D-1 §1.3, E-1.7) —
# `nil` is not a `_BORDER_STYLE_MAP` value.
_EXTRA_BORDER_STYLE_IN = {"nil": "none"}
_INV_BREAK_TYPE = {v: k for k, v in _BREAK_TYPE_MAP.items()}
_INV_ORDERED_FORMAT = {v: k for k, v in _ORDERED_FORMAT_MAP.items()}
_INV_BULLET_GLYPH = {glyph: fmt for fmt, (glyph, _font) in _UNORDERED_GLYPH.items()}

# RFC002 §12.1 (RATIFIED 2026-09-02) names the V0 MIME whitelist as
# png, jpeg, gif, bmp and tiff; `renderer/resources.py:MIME_EXTENSION`
# already carries all five. The importer knowing only two was drift from
# a ratified RFC, not a profile line — RFC001 §42.1 says nothing about
# image formats at all (D3).
_EXT_MIME = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "gif": "image/gif", "bmp": "image/bmp", "tif": "image/tiff",
    "tiff": "image/tiff",
}
_MIME_CANONICAL_EXT = {
    "image/png": "png", "image/jpeg": "jpg", "image/gif": "gif",
    "image/bmp": "bmp", "image/tiff": "tiff",
}
#: Magic-byte signatures, longest first. The MIME type is decided by
#: CONTENT and only then by extension: packages that store an image as
#: `image1.bin` or `.tmp` were refused on the strength of a filename
#: while their first bytes said `\x89PNG` (D4).
_MIME_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"BM", "image/bmp"),
)


def _sniff_mime(data: bytes) -> str | None:
    """The image's MIME type from its magic bytes, or None."""
    for signature, mime in _MIME_SIGNATURES:
        if data.startswith(signature):
            return mime
    return None

# ST_HighlightColor (wml.xsd) is a CLOSED 17-value enumeration, so this
# table is total, not a best effort. The values are Word's highlight
# palette. There is no renderer forward map to invert here — the
# renderer emits `run_style.highlight` as `w:shd` fill, never as a named
# `w:highlight` — so this is the one import map that is not the inverse
# of a renderer map, and it is exact rather than approximate (D5).
_HIGHLIGHT_HEX = {
    "black": "#000000", "blue": "#0000FF", "cyan": "#00FFFF",
    "green": "#00FF00", "magenta": "#FF00FF", "red": "#FF0000",
    "yellow": "#FFFF00", "white": "#FFFFFF", "darkBlue": "#000080",
    "darkCyan": "#008080", "darkGreen": "#008000", "darkMagenta": "#800080",
    "darkRed": "#800000", "darkYellow": "#808000", "darkGray": "#808080",
    "lightGray": "#C0C0C0",
}

_TAB_ALIGNMENTS = frozenset({"left", "center", "right", "decimal", "bar"})
# `hyperlink_child` in the schema: the inline set minus hyperlink,
# floating_image and note_ref.
_HYPERLINK_CHILD_TYPES = frozenset({"run", "tab", "line_break", "inline_image"})
# footnotePr/endnotePr are read (numFmt/numStart/numRestart drive the
# imported note marker text) — handled, not dropped.
_KNOWN_SECTPR_CHILDREN = frozenset({
    "type", "pgSz", "pgMar", "cols", "footnotePr", "endnotePr",
})
# Section features outside word-core-v0 that do not change body-content
# meaning: dropped with a typed issue, import continues (approximate).
# Header/footer stories are V1 — their references are counted and
# recorded in the source map, never silently lost.
_IGNORABLE_SECTPR_CHILDREN = frozenset({
    "headerReference", "footerReference", "lnNumType", "pgNumType", "docGrid",
    "formProt", "vAlign", "titlePg", "textDirection", "bidi", "rtlGutter",
})
# sectPr children word-visual-v1 imports instead of dropping (DEC-003:
# headers/footers and advanced section settings). `titlePg` and
# `pgBorders` are not dropped either — the first selects the `first`
# story, the second becomes `section.page_borders`.
_V1_SECTPR_CHILDREN = frozenset({
    "headerReference", "footerReference", "titlePg", "vAlign", "lnNumType",
    "pgBorders",
})
# Non-content range markers and annotations: no visual effect, so they
# are ignored with a typed info instead of rejecting the document.
# permStart/permEnd are editing-permission range markers — the same
# ECMA-376 EG_RangeMarkupElements class as bookmarkStart/End, and
# RFC001 §42.1 does not exclude them.
_IGNORED_PARAGRAPH_CHILDREN = frozenset({
    "bookmarkStart", "bookmarkEnd", "proofErr", "permStart", "permEnd",
})
_STYLE_CHAIN_DEPTH_CAP = 10

# Per-document rasterization budget (charts + EMF/WMF media). Measured
# against the 965 `m2a-v4` documents carrying IMPORT_GRAPHIC_RASTERIZED /
# IMPORT_GRAPHIC_RASTERIZATION_FAILED: a cap of 5 dropped 757 graphics
# across 47 tail documents (up to 164 graphics in one file); 50 drops
# only 208 and costs 549 extra conversions across the whole corpus
# (under a minute of wall clock over 22 workers). 50 is still a bound
# against a pathological document, chosen from the measured distribution.
_RASTERIZATION_BUDGET = 50

# Baseline effective values used only where neither direct formatting,
# a named style, nor docDefaults observes a property (external DOCX
# tolerance; our own renderer output always writes these explicitly).
_BASELINE_FONT = "Calibri"
_BASELINE_FONT_SIZE_MPT = 11000


def _w(tag: str) -> str:
    return f"{{{_W}}}{tag}"


def _twip_to_mpt(twip: int) -> int:
    return twip * 50


def _emu_to_mpt(emu: int) -> int:
    return round_half_away_from_zero(emu * 10, 127)


def _mpt_to_emu(mpt: int) -> int:
    """Inverse of `_emu_to_mpt`, needed because a VML shape's size lives
    in `v:shape/@style` as a physical length (millipoints once parsed)
    while `_rasterize_or_drop` — shared with the DrawingML path — wants
    EMU."""
    return round_half_away_from_zero(mpt * 127, 10)


@dataclass(frozen=True)
class ImportOptions:
    """Import policy knobs (RFC003 §17.2 — approximate import is
    legitimate, but lossy drops are opt-in).

    `allow_floating_drop`: when True, anchored (floating) drawings and
    VML `w:pict` shapes — unrepresentable in word-core-v0 — are dropped
    with a typed `IMPORT_FLOATING_OBJECT_DROPPED` issue and a source-map
    record instead of failing the import. Default False preserves the
    strict behavior. Under `word-visual-v1` an anchored `w:drawing` is
    imported as a `floating_image`, so the policy then governs only the
    VML `w:pict` shapes that stay outside the profile.

    `profile`: the target AST profile (DEC-003). `word-core-v0` is the
    default and reproduces the pre-DEC-003 import byte for byte —
    headers/footers dropped, hyperlinks flattened, nested tables
    rejected — so every pinned golden `ast_hash` is unmoved until the
    caller opts in. `word-visual-v1` imports the constructs RFC001
    §42.2 assigns to that profile.

    `drop_empty_text_frames` (DEC-055 I2): a `text_frame` whose content has
    no ink -- only paragraphs holding whitespace runs, tabs or line breaks
    -- is not emitted, with a typed `IMPORT_EMPTY_TEXT_FRAME_DROPPED` issue
    and a source-map record. The node carries no border or fill, so it
    renders nothing. Default False keeps every existing import byte for
    byte; `migration.m2a` turns it on only with a `MIGRATION_VERSION` bump.
    """

    allow_floating_drop: bool = False
    profile: Literal[
        "word-core-v0", "word-visual-v1", "word-visual-v1.1"
    ] = "word-core-v0"
    drop_empty_text_frames: bool = False

    @property
    def visual_v1(self) -> bool:
        return self.profile in ("word-visual-v1", "word-visual-v1.1")

    @property
    def text_frames(self) -> bool:
        """Whether the target profile admits RFC001-ERRATA-004 text frames."""
        return self.profile == "word-visual-v1.1"


@dataclass
class ImportResult:
    """Outcome of an `import_docx` call (RFC003 §17.2).

    `ast` is the IMPORTER-FORM tree: its node ids (`i*`) are the keys of
    `source_map` and the `node_id` of every `report` issue, so the three
    fields describe one tree. `canonicalize()` renumbers ids to preorder
    `n*` and therefore breaks that correspondence — a caller that stores
    or hashes an AST must canonicalize it itself and store exactly what
    it hashed (see `migration.m2a`), never treat this tree as canonical.
    """

    status: ImportStatus
    ast: dict[str, Any] | None
    source_map: dict[str, dict[str, Any]]
    report: list[Issue]
    resources: dict[str, bytes] = field(default_factory=dict)
    resource_provenance: dict[str, str] = field(default_factory=dict)

    def get_resources(self) -> dict[str, bytes]:
        """Extracted media bytes keyed by the AST resource `path`."""
        return dict(self.resources)

    def get_resource_provenance(self) -> dict[str, str]:
        """Derived resource path -> source package part (out-of-band)."""
        return dict(self.resource_provenance)


@dataclass
class _NumDef:
    list_kind: str
    marker_format: str
    suffix: str | None
    start: int | None
    left_indent_mpt: int
    hanging_mpt: int
    approximate: bool = False
    text: str | None = None  # ERRATA-002 E6: literal marker text for custom_text
    #: The level renders no marker at all (empty `w:lvlText`), so it
    #: cannot become a `list` — the AST has no "no marker" format.
    no_marker: bool = False


@dataclass
class _Ctx:
    tree: etree._ElementTree
    options: ImportOptions = field(default_factory=ImportOptions)
    issues: list[Issue] = field(default_factory=list)
    source_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    dd_ppr: etree._Element | None = None
    dd_rpr: etree._Element | None = None
    styles: dict[str, etree._Element] = field(default_factory=dict)
    # `w:style[@w:type="table"][@w:default="1"]` — the style Word applies
    # to any table lacking `w:tblStyle` (border resolution rank 3b, D-1
    # §1.2). `_load_styles` indexes by `styleId` only, so this needs its
    # own slot.
    default_table_style_id: str | None = None
    hopped_styles: set[str] = field(default_factory=set)
    numdefs: dict[int, _NumDef] = field(default_factory=dict)
    # V0's reported higher-level approximation uses abstract level zero.
    # An instance override for level zero must not rewrite other levels.
    numbering_abstract_defs: dict[int, _NumDef] = field(default_factory=dict)
    # OOXML counters survive paragraph/table/section interruptions. Count
    # source paragraphs while parsing, not while assembling AST lists: table
    # cells are assembled before preceding body paragraphs. A reused header
    # part is parsed again in each section but must not consume new ordinals.
    numbering_next: dict[tuple[str, str, int], int] = field(default_factory=dict)
    numbering_ordinals: dict[tuple[str, str], int] = field(default_factory=dict)
    # Relationships are PER PART: `rId4` in word/header1.xml.rels and in
    # word/document.xml.rels are unrelated ids pointing at different
    # targets, so one flat map would silently resolve a header image to
    # the body's target. 25.5 % of a 1500-document sample ship a header
    # or footer part with its own .rels, which makes this the common
    # case, not an edge case. Read it through `ctx.rels`, which always
    # answers for the part currently being parsed.
    rels_by_part: dict[str, dict[str, str]] = field(default_factory=dict)
    # The part currently being parsed; `tree` is that part's tree, so
    # `prov()` yields real XPaths inside headers and note bodies too.
    part: str = "word/document.xml"
    parts: dict[str, bytes] = field(default_factory=dict)
    resources_by_sha: dict[str, dict[str, Any]] = field(default_factory=dict)
    resource_bytes: dict[str, bytes] = field(default_factory=dict)
    resource_provenance: dict[str, str] = field(default_factory=dict)
    rasterization_count: int = 0
    # Open complex-field frames (w:fldChar begin..end), document-wide —
    # fields legally span runs and paragraphs (e.g. TOC).
    field_stack: list[dict[str, Any]] = field(default_factory=list)
    footnote_count: int = 0
    endnote_count: int = 0
    # Per-section footnote/endnote numbering settings (sectPr-level
    # w:footnotePr / w:endnotePr, prescanned in document order) and the
    # index of the section currently being parsed.
    note_conf: list[dict[str, dict[str, Any]]] = field(default_factory=list)
    section_index: int = 0
    # word-visual-v1 note stories: imported bodies keyed by ("footnote"
    # | "endnote", source w:id), and the ids of the notes some note_ref
    # actually reached. An unreferenced body is content Word never
    # renders and `_check_note_refs` rejects, so it is never emitted.
    note_bodies: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    referenced_notes: set[tuple[str, str]] = field(default_factory=set)
    # True while parsing a footnote/endnote body: a note_ref there is
    # forbidden by the semantic validator, so nested references degrade
    # to their marker text instead of producing an invalid AST.
    in_note_body: bool = False
    # A note_ref that declared w:customMarkFollows and is still waiting
    # for the run text carrying its visible mark.
    pending_custom_mark: dict[str, Any] | None = None
    # Text frames may contain ordinary story blocks, including drawings,
    # but RFC001-ERRATA-004 rejects a text_frame inside another text_frame.
    # The depth makes that residual drop path explicit while reusing the
    # ordinary paragraph/table parsers for the supported content.
    text_frame_depth: int = 0
    # The note body currently being parsed, and the body-mark runs whose
    # text is filled in when the referencing note_ref resolves the mark.
    note_mark_key: tuple[str, str] | None = None
    note_mark_runs: dict[tuple[str, str], list[dict[str, Any]]] = field(
        default_factory=dict
    )
    # Cached `w:evenAndOddHeaders` (settings.xml); None until first read.
    even_and_odd_headers: bool | None = None
    # Word bookmark name -> the AST node id the bookmark sits on, and the
    # internal hyperlinks waiting on one. RFC001-ERRATA-001 E5 makes an
    # `internal_anchor` uri `#<node_id>`, and the canonicalizer REJECTS a
    # dangling anchor — but a w:hyperlink names a BOOKMARK, and a
    # forward reference (every TOC entry) cannot be resolved while the
    # body is parsed, so resolution is a fix-up pass.
    bookmarks: dict[str, str] = field(default_factory=dict)
    _counter: int = 0

    @property
    def rels(self) -> dict[str, str]:
        """Relationships of the part currently being parsed."""
        return self.rels_by_part.get(self.part, {})

    @contextmanager
    def in_part(self, part: str, tree: etree._ElementTree) -> Iterator[None]:
        """Parse `part` — swapping the relationship scope and the tree
        `prov()` resolves XPaths against — then restore the caller's."""
        saved = (self.part, self.tree)
        self.part, self.tree = part, tree
        try:
            yield
        finally:
            self.part, self.tree = saved

    def in_field_code(self) -> bool:
        """True while inside any field's instruction region (begin
        without separate yet): visible content there is field code, not
        a stable result — dropped, recorded on the field frame."""
        return any(not frame["separated"] for frame in self.field_stack)

    def new_id(self) -> str:
        self._counter += 1
        return f"i{self._counter}"

    def issue(self, code: str, severity: str, message: str, el: etree._Element | None = None,
              node_id: str | None = None, **details: Any) -> None:
        self.issues.append(
            Issue(
                code=code,
                severity=severity,  # type: ignore[arg-type]
                node_id=node_id,
                path=self.prov(el) if el is not None else f"/{self.part}",
                message=message,
                details=dict(details),
            )
        )

    def prov(self, el: etree._Element) -> str:
        try:
            return self.tree.getpath(el)
        except ValueError:  # element from another part (styles/numbering)
            return "."

    def map_node(self, node_id: str, el: etree._Element, status: str = "exact",
                 part: str | None = None) -> None:
        self.source_map[node_id] = {
            "part": part or self.part,
            "provenance": self.prov(el),
            "import_status": status,
        }

    def degrade(self, node_id: str, status: str) -> None:
        entry = self.source_map.get(node_id)
        if entry is not None and entry["import_status"] == "exact":
            entry["import_status"] = status


def _int_attr(ctx: _Ctx, el: etree._Element, attr: str, *, default: int | None = None,
              node_id: str | None = None, required: bool = False) -> int | None:
    """Parse an integer XML attribute defensively (external DOCX tolerance).

    A missing attribute returns `default` (with an `IMPORT_MALFORMED_VALUE`
    issue when `required`); a non-integer value returns `default` with an
    `IMPORT_MALFORMED_VALUE` warning and degrades `node_id` to approximate
    — a malformed value must never crash the import (RFC003 §45).
    """
    local = etree.QName(attr).localname if attr.startswith("{") else attr
    raw = el.get(attr)
    if raw is None:
        if required:
            ctx.issue(
                "IMPORT_MALFORMED_VALUE", "warning",
                f"required attribute '{local}' is missing on 'w:"
                f"{etree.QName(el).localname}'; element imported approximately.",
                el=el, node_id=node_id, attribute=local,
            )
            if node_id is not None:
                ctx.degrade(node_id, "approximate")
        return default
    try:
        return int(raw)
    except ValueError:
        ctx.issue(
            "IMPORT_MALFORMED_VALUE", "warning",
            f"attribute '{local}' value {raw!r} is not a valid integer; "
            "imported approximately.",
            el=el, node_id=node_id, attribute=local, value=raw,
        )
        if node_id is not None:
            ctx.degrade(node_id, "approximate")
        return default


def _nonneg(ctx: _Ctx, value: int, what: str, node_id: str | None = None,
            el: etree._Element | None = None) -> int:
    """Clamp a length the schema requires non-negative (D1).

    Word accepts a negative `w:tcMar`, `w:spacing` or `w:firstLine`; the
    AST's `length_mpt_nonneg` does not, and the value used to reach
    `canonicalize()` unchecked and fail the WHOLE document there. Clamping
    with a typed issue follows the existing `IMPORT_NEGATIVE_LIST_INDENT`
    and `IMPORT_NEGATIVE_TABLE_INDENT` precedent; widening the schema
    would amend a ratified RFC001 for the same visible result
    (team-lead decision §8.2).
    """
    if value >= 0:
        return value
    ctx.issue(
        "IMPORT_NEGATIVE_LENGTH_CLAMPED", "warning",
        f"{what} is {value} mpt, which the AST cannot represent; clamped to 0 "
        "(approximate import).",
        el=el, node_id=node_id, property=what, computed_mpt=value,
    )
    if node_id is not None:
        ctx.degrade(node_id, "approximate")
    return 0


def _parse_bool(el: etree._Element | None, default: bool) -> bool:
    if el is None:
        return default
    val = el.get(_w("val"))
    if val is None:
        return True
    return val not in ("0", "false", "off")


def _chain_layer_with(
    chain: list[etree._Element | None], tag: str
) -> etree._Element | None:
    """Leaf-first: the first chain layer that CONTAINS `tag` (the layer,
    not the tag element) — for helpers that parse a whole layer."""
    for layer in chain:
        if layer is not None and layer.find(_w(tag)) is not None:
            return layer
    return None


def _chain_find(chain: list[etree._Element | None], tag: str) -> etree._Element | None:
    for layer in chain:
        if layer is None:
            continue
        child = layer.find(_w(tag))
        if child is not None:
            return child
    return None


_HEX6 = re.compile(r"^[0-9A-Fa-f]{6}$")
_HEX8 = re.compile(r"^[0-9A-Fa-f]{8}$")


def _color_from_val(ctx: _Ctx, val: str | None, default: str,
                    node_id: str | None = None,
                    el: etree._Element | None = None) -> str:
    """An OOXML colour value -> an AST colour, validated HERE.

    `"#" + val.upper()` used to be unconditional, so a source writing
    `BLUE`, `RGB(0, 0, 0)` or an empty value produced `'#BLUE'` — which
    only failed at the very end, in `canonicalize()`, taking the WHOLE
    document with it (D1). The importer already owns the mechanism for
    this (`IMPORT_MALFORMED_VALUE`); it just was not used on this path.

    An 8-digit ARGB value keeps its RGB half: that is what Word
    displays, and keeping the visible property beats defaulting it.
    """
    if val is None:
        return default
    if val == "auto":
        return "auto"
    if _HEX6.match(val):
        return "#" + val.upper()
    if _HEX8.match(val):
        rgb = "#" + val[2:].upper()
        ctx.issue(
            "IMPORT_COLOR_ALPHA_DROPPED", "warning",
            f"colour value {val!r} carries an alpha channel the AST does not model; "
            f"imported as {rgb} (the colour Word displays; approximate import).",
            el=el, node_id=node_id, value=val,
        )
        if node_id is not None:
            ctx.degrade(node_id, "approximate")
        return rgb
    ctx.issue(
        "IMPORT_MALFORMED_VALUE", "warning",
        f"colour value {val!r} is not an OOXML hex colour; the default {default!r} "
        "was used (approximate import).",
        el=el, node_id=node_id, value=val, attribute="color",
    )
    if node_id is not None:
        ctx.degrade(node_id, "approximate")
    return default


def _shd_background(ctx: _Ctx, layer: etree._Element | None,
                    node_id: str | None = None) -> str | None:
    if layer is None:
        return None
    shd = layer.find(_w("shd"))
    if shd is None:
        return None
    fill = shd.get(_w("fill"))
    if fill is None:
        return None
    if fill == "auto":
        return "auto"
    colour = _color_from_val(ctx, fill, "auto", node_id=node_id, el=shd)
    # An unparseable fill falls back to the default, which for a shading
    # fill means "no background" rather than the literal `auto` colour.
    return None if colour == "auto" else colour


def _shd_background_chained(
    ctx: _Ctx, direct: etree._Element | None,
    style_chain: list[etree._Element], style_pr_tag: str,
    node_id: str | None = None,
) -> str | None:
    """Table/cell background resolved through the style chain (D-1
    §1.8), the identical precedence rule as `_table_border_context`:
    direct `w:tblPr`/`w:tcPr` `w:shd` > the table's own `w:tblStyle`
    chain's matching `style_pr_tag` sub-element `w:shd` > the default
    table style (already folded into `style_chain` by
    `_resolve_table_style`) > nothing. `w:fill="auto"` is not ink, but
    it is still an explicit layer — it wins at whatever rank it is
    found, exactly like a real colour."""
    if direct is not None and direct.find(_w("shd")) is not None:
        return _shd_background(ctx, direct, node_id)
    for style in style_chain:
        style_pr = style.find(_w(style_pr_tag))
        if style_pr is not None and style_pr.find(_w("shd")) is not None:
            return _shd_background(ctx, style_pr, node_id)
    return None


# --- styles.xml ---------------------------------------------------------


def _load_styles(ctx: _Ctx, data: bytes | None) -> None:
    if data is None:
        return
    root = etree.fromstring(data)
    dd = root.find(_w("docDefaults"))
    if dd is not None:
        rpr_default = dd.find(_w("rPrDefault"))
        if rpr_default is not None:
            ctx.dd_rpr = rpr_default.find(_w("rPr"))
        ppr_default = dd.find(_w("pPrDefault"))
        if ppr_default is not None:
            ctx.dd_ppr = ppr_default.find(_w("pPr"))
    for style in root.findall(_w("style")):
        style_id = style.get(_w("styleId"))
        if style_id:
            ctx.styles[style_id] = style
            if (
                style.get(_w("type")) == "table"
                and style.get(_w("default")) not in (None, "0", "false")
            ):
                ctx.default_table_style_id = style_id


def _resolve_style_chain(ctx: _Ctx, style_id: str | None,
                         node_id: str | None) -> tuple[list[etree._Element], str | None]:
    """Resolve a named style's full `basedOn` chain, leaf -> root.

    The effective properties are the chain walked root -> leaf with
    overrides — realized by the leaf-first lookup order of
    `_chain_find`. Resolution is deterministic and exact; a chain deeper
    than one hop is reported once as `IMPORT_STYLE_CHAIN_RESOLVED`
    (info). A cycle (`IMPORT_STYLE_CHAIN_CYCLE`) or a chain deeper than
    `_STYLE_CHAIN_DEPTH_CAP` (`IMPORT_UNSUPPORTED_STYLE_CHAIN`) is
    truncated with a typed warning and an approximate import — never a
    crash or an unbounded walk.
    """
    if style_id is None:
        return [], None
    chain: list[etree._Element] = []
    seen: set[str] = set()
    truncation: tuple[str, str] | None = None  # (code, message)
    current: str | None = style_id
    while current is not None:
        style = ctx.styles.get(current)
        if style is None:
            ctx.issue(
                "IMPORT_STYLE_UNRESOLVED", "warning",
                f"style '{current}' is referenced but not defined in styles.xml.",
                node_id=node_id, style_id=current,
            )
            break
        if current in seen:
            truncation = (
                "IMPORT_STYLE_CHAIN_CYCLE",
                f"style '{style_id}' has a basedOn cycle (at '{current}'); the chain "
                "was truncated at the repeated style (approximate import).",
            )
            break
        seen.add(current)
        chain.append(style)
        based_on = style.find(_w("basedOn"))
        current = based_on.get(_w("val")) if based_on is not None else None
        if current is not None and len(chain) >= _STYLE_CHAIN_DEPTH_CAP:
            truncation = (
                "IMPORT_UNSUPPORTED_STYLE_CHAIN",
                f"style '{style_id}' has a basedOn chain deeper than the supported "
                f"cap of {_STYLE_CHAIN_DEPTH_CAP}; deeper chain properties are not "
                "applied (approximate import).",
            )
            break
    if truncation is not None:
        if node_id is not None:
            ctx.degrade(node_id, "approximate")
        if style_id not in ctx.hopped_styles:
            ctx.hopped_styles.add(style_id)
            ctx.issue(truncation[0], "warning", truncation[1],
                      node_id=node_id, style_id=style_id)
        return chain, "approximate"

    if len(chain) > 1:
        # Element-level shadowing (leaf element wholly replaces the
        # parent's), NOT OOXML attribute-level merge / toggle semantics
        # — a deterministic approximation, so degrade to normalized.
        if node_id is not None:
            ctx.degrade(node_id, "normalized")
        if style_id not in ctx.hopped_styles:
            ctx.hopped_styles.add(style_id)
            ctx.issue(
                "IMPORT_STYLE_CHAIN_RESOLVED", "info",
                f"style '{style_id}' resolved through a basedOn chain of depth "
                f"{len(chain)} (element-level shadowing approximation).",
                node_id=node_id, style_id=style_id, depth=len(chain),
            )
        return chain, "normalized"
    return chain, None


# --- numbering.xml ------------------------------------------------------


def _load_numbering(ctx: _Ctx, data: bytes | None) -> None:
    if data is None:
        return
    root = etree.fromstring(data)
    abstract: dict[int, _NumDef] = {}
    style_links: dict[int, str] = {}
    for abstract_num in root.findall(_w("abstractNum")):
        abstract_id = _int_attr(ctx, abstract_num, _w("abstractNumId"), required=True)
        if abstract_id is None:
            continue
        lvl0 = None
        for lvl in abstract_num.findall(_w("lvl")):
            if lvl.get(_w("ilvl")) == "0":
                lvl0 = lvl
                break
        if lvl0 is None:
            # An abstractNum with no levels of its own may delegate to a
            # numbering style (`w:numStyleLink`), which is where its real
            # definition lives. Resolved below, once the numId ->
            # abstractNumId map exists.
            link = abstract_num.find(_w("numStyleLink"))
            if link is not None and link.get(_w("val")):
                style_links[abstract_id] = link.get(_w("val"))
            continue
        abstract[abstract_id] = _numdef_from_lvl(ctx, lvl0)

    num_to_abstract: dict[int, int] = {}
    overrides: dict[int, etree._Element] = {}
    for num in root.findall(_w("num")):
        num_id = _int_attr(ctx, num, _w("numId"), required=True)
        ref = num.find(_w("abstractNumId"))
        if num_id is None or ref is None:
            continue
        ref_val = _int_attr(ctx, ref, _w("val"), required=True)
        if ref_val is not None:
            num_to_abstract[num_id] = ref_val
            for override in num.findall(_w("lvlOverride")):
                if override.get(_w("ilvl")) == "0":
                    overrides[num_id] = override

    for abstract_id, style_id in style_links.items():
        definition = _numdef_via_style_link(ctx, style_id, num_to_abstract, abstract)
        if definition is not None:
            abstract[abstract_id] = definition

    for num_id, ref_val in num_to_abstract.items():
        definition = abstract.get(ref_val)
        if definition is not None:
            ctx.numbering_abstract_defs[num_id] = definition
            override = overrides.get(num_id)
            if override is not None:
                level = override.find(_w("lvl"))
                if level is not None:
                    definition = _numdef_from_lvl(ctx, level)
                start = override.find(_w("startOverride"))
                if start is not None and definition.list_kind == "ordered":
                    definition = replace(
                        definition, start=_int_attr(ctx, start, _w("val"), default=1),
                    )
            ctx.numdefs[num_id] = definition


def _numdef_via_style_link(ctx: _Ctx, style_id: str, num_to_abstract: dict[int, int],
                           abstract: dict[int, _NumDef]) -> _NumDef | None:
    """Follow `w:numStyleLink` -> numbering style -> `w:numId` -> the
    abstract definition that actually carries the levels.

    One hop only: a numbering style that delegates again is a chain the
    corpus does not contain, and following it blindly risks a cycle.
    """
    style = ctx.styles.get(style_id)
    if style is None:
        return None
    num_id_el = style.find(f"{_w('pPr')}/{_w('numPr')}/{_w('numId')}")
    if num_id_el is None:
        return None
    num_id = _int_attr(ctx, num_id_el, _w("val"))
    if num_id is None:
        return None
    definition = abstract.get(num_to_abstract.get(num_id, -1))
    if definition is None:
        ctx.issue(
            "IMPORT_NUMBERING_STYLE_LINK_UNRESOLVED", "warning",
            f"w:numStyleLink points at numbering style '{style_id}', whose numbering "
            "definition could not be resolved; paragraphs using it import as plain "
            "paragraphs (approximate import).",
            style_id=style_id,
        )
    return definition


def _numdef_from_lvl(ctx: _Ctx, lvl: etree._Element) -> _NumDef:
    num_fmt_el = lvl.find(_w("numFmt"))
    num_fmt = num_fmt_el.get(_w("val")) if num_fmt_el is not None else "decimal"
    lvl_text_el = lvl.find(_w("lvlText"))
    lvl_text = lvl_text_el.get(_w("val")) if lvl_text_el is not None else ""
    start_el = lvl.find(_w("start"))
    start = _int_attr(ctx, start_el, _w("val"), default=1) if start_el is not None else 1
    left_mpt = hanging_mpt = 0
    p_pr = lvl.find(_w("pPr"))
    if p_pr is not None:
        ind = p_pr.find(_w("ind"))
        if ind is not None:
            left_mpt = _twip_to_mpt(_int_attr(ctx, ind, _w("left"), default=0))
            hanging_mpt = _twip_to_mpt(_int_attr(ctx, ind, _w("hanging"), default=0))
            if left_mpt < 0 or hanging_mpt < 0:
                ctx.issue(
                    "IMPORT_NEGATIVE_LIST_INDENT", "warning",
                    "numbering level defines a negative marker indent, which "
                    "word-core-v0 cannot represent; clamped to 0 (approximate).",
                    el=ind, left_mpt=left_mpt, hanging_mpt=hanging_mpt,
                )
                left_mpt = max(left_mpt, 0)
                hanging_mpt = max(hanging_mpt, 0)

    if not lvl_text:
        # `w:lvlText` empty (usually with `w:numFmt w:val="none"`): the
        # level renders NO marker. `custom_text` requires a non-empty
        # `text`, and inventing a bullet glyph would put ink on the page
        # that the source does not have — so the paragraphs import as
        # plain paragraphs instead (handled in `_assemble_blocks`).
        # 103 documents used to be rejected whole for this (D1).
        return _NumDef("unordered", "custom_text", None, None, left_mpt, hanging_mpt,
                       approximate=True, text=None, no_marker=True)
    if num_fmt in _INV_ORDERED_FORMAT:
        suffix = lvl_text[2:] if lvl_text.startswith("%1") else lvl_text
        return _NumDef("ordered", _INV_ORDERED_FORMAT[num_fmt], suffix, start,
                       left_mpt, hanging_mpt)
    # bullet family: the AST marker suffix is not observable for bullet
    # glyphs (the renderer's lvlText carries only the glyph); the
    # canonical generator default "." is assumed (normalized import).
    fmt = _INV_BULLET_GLYPH.get(lvl_text)
    if fmt is not None:
        return _NumDef("unordered", fmt, ".", None, left_mpt, hanging_mpt, approximate=True)
    return _NumDef("unordered", "custom_text", None, None, left_mpt, hanging_mpt,
                   approximate=True, text=lvl_text)


# --- run / paragraph properties ----------------------------------------


def _run_style(ctx: _Ctx, chain: list[etree._Element | None], node_id: str) -> dict[str, Any]:
    rfonts = _chain_find(chain, "rFonts")
    font = _BASELINE_FONT
    if rfonts is not None:
        font = rfonts.get(_w("ascii")) or rfonts.get(_w("hAnsi")) or _BASELINE_FONT
    sz = _chain_find(chain, "sz")
    sz_val = _int_attr(ctx, sz, _w("val"), node_id=node_id) if sz is not None else None
    size_mpt = sz_val * 500 if sz_val is not None else _BASELINE_FONT_SIZE_MPT
    if size_mpt <= 0:
        # `w:sz` of 0 (or negative) is not a renderable size and the
        # schema forbids it (`exclusiveMinimum: 0`). Falling back to the
        # baseline keeps the run's text; failing the whole document over
        # it does not (D1).
        ctx.issue(
            "IMPORT_MALFORMED_VALUE", "warning",
            f"font size {size_mpt} mpt is not a renderable size; the baseline "
            f"{_BASELINE_FONT_SIZE_MPT} mpt was used (approximate import).",
            el=sz, node_id=node_id, attribute="sz", value=sz_val,
        )
        ctx.degrade(node_id, "approximate")
        size_mpt = _BASELINE_FONT_SIZE_MPT
    color_el = _chain_find(chain, "color")
    color = _color_from_val(
        ctx, color_el.get(_w("val")) if color_el is not None else None, "#000000",
        node_id=node_id, el=color_el,
    )
    spacing_el = _chain_find(chain, "spacing")
    char_spacing = _twip_to_mpt(
        _int_attr(ctx, spacing_el, _w("val"), default=0, node_id=node_id)
        if spacing_el is not None
        else 0
    )

    underline = "none"
    u = _chain_find(chain, "u")
    if u is not None:
        u_val = u.get(_w("val")) or "single"
        underline = _INV_UNDERLINE.get(u_val)
        if underline is None:
            underline = "single"
            ctx.issue(
                "IMPORT_UNDERLINE_APPROXIMATED", "warning",
                f"underline value '{u_val}' has no word-core-v0 equivalent; imported as "
                "'single' (approximate).",
                node_id=node_id, value=u_val,
            )
            ctx.degrade(node_id, "approximate")

    # w:shd may come from the direct rPr OR any style in the chain —
    # read leaf-first like every other chain property (review should_fix:
    # chain-level shading was silently ignored while w:highlight scanned
    # the whole chain).
    highlight = _shd_background(ctx, _chain_layer_with(chain, "shd"), node_id)
    highlight_el = _chain_find(chain, "highlight")
    if highlight is None and highlight_el is not None:
        name = highlight_el.get(_w("val")) or "none"
        if name != "none":
            highlight = _HIGHLIGHT_HEX.get(name)
            if highlight is None:
                ctx.issue(
                    "IMPORT_RUN_HIGHLIGHT_UNSUPPORTED", "warning",
                    f"w:highlight value '{name}' is not in ST_HighlightColor; the "
                    "highlight was dropped (approximate import).",
                    node_id=node_id, value=name,
                )
                ctx.degrade(node_id, "approximate")

    vert = _chain_find(chain, "vertAlign")
    vert_val = vert.get(_w("val")) if vert is not None else "baseline"
    return {
        "font_family": font,
        "font_size_mpt": size_mpt,
        "bold": _parse_bool(_chain_find(chain, "b"), False),
        "italic": _parse_bool(_chain_find(chain, "i"), False),
        "underline": underline,
        "strike": _parse_bool(_chain_find(chain, "strike"), False),
        "color": color,
        "highlight": highlight,
        "superscript": vert_val == "superscript",
        "subscript": vert_val == "subscript",
        "character_spacing_mpt": char_spacing,
        "small_caps": _parse_bool(_chain_find(chain, "smallCaps"), False),
        "all_caps": _parse_bool(_chain_find(chain, "caps"), False),
    }


def _indent_from_chain(ctx: _Ctx, chain: list[etree._Element | None],
                       node_id: str) -> dict[str, int]:
    ind = _chain_find(chain, "ind")
    if ind is None:
        return {"left_mpt": 0, "right_mpt": 0, "first_line_mpt": 0, "hanging_mpt": 0}
    hanging = _nonneg(
        ctx,
        _twip_to_mpt(_int_attr(ctx, ind, _w("hanging"), default=0, node_id=node_id)),
        "paragraph hanging indent", node_id, ind,
    )
    first_line = _nonneg(
        ctx,
        _twip_to_mpt(_int_attr(ctx, ind, _w("firstLine"), default=0, node_id=node_id)),
        "paragraph first-line indent", node_id, ind,
    )
    if hanging:
        first_line = 0  # w:hanging shadows w:firstLine (Word semantics)
    return {
        "left_mpt": _twip_to_mpt(_int_attr(ctx, ind, _w("left"), default=0, node_id=node_id)),
        "right_mpt": _twip_to_mpt(_int_attr(ctx, ind, _w("right"), default=0, node_id=node_id)),
        "first_line_mpt": first_line,
        "hanging_mpt": hanging,
    }


def _spacing_from_chain(ctx: _Ctx, chain: list[etree._Element | None],
                        node_id: str) -> dict[str, Any]:
    spacing = _chain_find(chain, "spacing")
    if spacing is None:
        return {"before_mpt": 0, "after_mpt": 0, "line": {"mode": "single"}}
    line_attr = spacing.get(_w("line"))
    rule = spacing.get(_w("lineRule"), "auto")
    line: dict[str, Any] = {"mode": "single"}
    value = (
        _int_attr(ctx, spacing, _w("line"), node_id=node_id)
        if line_attr is not None
        else None
    )
    if value is not None:
        if rule == "auto":
            if value != 240:
                line = {
                    "mode": "multiple",
                    "multiplier_ppm": round_half_away_from_zero(value * 1_000_000, 240),
                }
        elif rule == "exact":
            line = {"mode": "exact", "value_mpt": _twip_to_mpt(value)}
        else:  # atLeast
            line = {"mode": "at_least", "value_mpt": _twip_to_mpt(value)}
    return {
        "before_mpt": _nonneg(
            ctx,
            _twip_to_mpt(
                _int_attr(ctx, spacing, _w("before"), default=0, node_id=node_id)
            ),
            "paragraph spacing before", node_id, spacing,
        ),
        "after_mpt": _nonneg(
            ctx,
            _twip_to_mpt(
                _int_attr(ctx, spacing, _w("after"), default=0, node_id=node_id)
            ),
            "paragraph spacing after", node_id, spacing,
        ),
        "line": line,
    }


def _tabs_from_chain(ctx: _Ctx, chain: list[etree._Element | None],
                     node_id: str) -> list[dict[str, Any]]:
    tabs_el = _chain_find(chain, "tabs")
    if tabs_el is None:
        return []
    tabs: list[dict[str, Any]] = []
    for tab in tabs_el.findall(_w("tab")):
        alignment = tab.get(_w("val"), "left")
        if alignment == "clear":
            continue
        if alignment not in _TAB_ALIGNMENTS:
            ctx.issue(
                "IMPORT_TAB_STOP_APPROXIMATED", "warning",
                f"tab stop alignment '{alignment}' has no word-core-v0 equivalent; imported "
                "as 'left' (approximate).",
                node_id=node_id, value=alignment,
            )
            ctx.degrade(node_id, "approximate")
            alignment = "left"
        leader = tab.get(_w("leader"))
        tabs.append({
            "position_mpt": _twip_to_mpt(
                _int_attr(ctx, tab, _w("pos"), default=0, node_id=node_id)
            ),
            "alignment": alignment,
            "leader": _INV_TAB_LEADER.get(leader, "none") if leader else "none",
        })
    return tabs


def _border_side(ctx: _Ctx, side_el: etree._Element, node_id: str) -> dict[str, Any]:
    val = side_el.get(_w("val"), "single")
    style = _INV_BORDER_STYLE.get(val, _EXTRA_BORDER_STYLE_IN.get(val))
    if style is None:
        style = "single"
        ctx.issue(
            "IMPORT_BORDER_STYLE_APPROXIMATED", "warning",
            f"border style '{val}' has no word-core-v0 equivalent; imported as 'single' "
            "(approximate).",
            node_id=node_id, value=val,
        )
        ctx.degrade(node_id, "approximate")
    return {
        "style": style,
        "width_mpt": _int_attr(ctx, side_el, _w("sz"), default=0, node_id=node_id) * 125,
        "color": _color_from_val(
            ctx, side_el.get(_w("color")), "auto", node_id=node_id, el=side_el,
        ),
    }


def _borders_from(ctx: _Ctx, container: etree._Element | None,
                  node_id: str) -> dict[str, Any] | None:
    if container is None:
        return None
    borders: dict[str, Any] = {}
    for side in ("top", "left", "bottom", "right"):
        side_el = container.find(_w(side))
        if side_el is not None:
            borders[side] = _border_side(ctx, side_el, node_id)
    return borders or None


_BORDER_SIDES = ("top", "left", "bottom", "right", "insideH", "insideV")


def _resolve_table_style(
    ctx: _Ctx, tbl_pr: etree._Element | None, node_id: str
) -> tuple[bool, list[etree._Element]]:
    """Resolve the table's effective named-style chain ONCE: the
    table's own `w:tblStyle` if present, else the default table style
    (D-1 §1.2 rank 3b) if there is one. Returns `(is_direct, chain)` so
    a caller that must NOT fall back to the default (the `w:tblStylePr`
    non-goal check, §2) can tell the two apart without a second
    `_resolve_style_chain` call — which would re-emit
    `IMPORT_STYLE_UNRESOLVED`/chain-truncation warnings a second time
    for the same unresolved or over-deep style id.
    """
    tbl_style_el = tbl_pr.find(_w("tblStyle")) if tbl_pr is not None else None
    style_id = tbl_style_el.get(_w("val")) if tbl_style_el is not None else None
    is_direct = style_id is not None
    if style_id is None:
        style_id = ctx.default_table_style_id
    style_chain, _status = _resolve_style_chain(ctx, style_id, node_id)
    return is_direct, style_chain


def _table_border_context(
    ctx: _Ctx, tbl_pr: etree._Element | None, node_id: str,
    style_chain: list[etree._Element],
) -> dict[str, etree._Element | None]:
    """Resolve a table's own 6-slot border set (D-1 §1.2 ranks 2/3/3b),
    leaf-first: table direct `w:tblPr/w:tblBorders`, else the given
    style chain (table's own `w:tblStyle`, or the default table style
    when the table has none — `_resolve_table_style`), else nothing.
    Returns raw `<w:top>`/`<w:insideH>`/... elements (or `None` per
    side), not yet parsed into AST border dicts, so callers keep the
    option of falling through to a lower rank.
    """
    chain: list[etree._Element | None] = []
    direct = tbl_pr.find(_w("tblBorders")) if tbl_pr is not None else None
    if direct is not None:
        chain.append(direct)
    for style in style_chain:
        style_tbl_pr = style.find(_w("tblPr"))
        style_borders = style_tbl_pr.find(_w("tblBorders")) if style_tbl_pr is not None else None
        if style_borders is not None:
            chain.append(style_borders)
    return {side: _chain_find(chain, side) for side in _BORDER_SIDES}


def _tbl_look_enables_bands(tbl_pr: etree._Element | None) -> bool:
    """Mirrors `scripts/census/import_fidelity_census.py`
    `tbl_look_enables_bands` (:169-186) exactly — that predicate is what
    the 27/1500 (1.8 %) `IMPORT_TABLE_STYLE_CONDITIONAL_APPROXIMATED`
    measurement was produced with (D-1 §2). Keep them in lockstep: the
    "0 of 27 import clean" conclusion is conditioned on the predicate
    staying this narrow.
    """
    look = tbl_pr.find(_w("tblLook")) if tbl_pr is not None else None
    if look is None:
        return True
    on = any(
        look.get(_w(k)) in ("1", "true", "on")
        for k in ("firstRow", "lastRow", "firstColumn", "lastColumn")
    )
    mask = look.get(_w("val"))
    if mask is not None:
        try:
            on = on or int(mask, 16) != 0
        except ValueError:
            pass
    no_h = look.get(_w("noHBand")) in ("1", "true", "on")
    no_v = look.get(_w("noVBand")) in ("1", "true", "on")
    return on or not (no_h and no_v)


def _check_table_style_conditional_bands(
    ctx: _Ctx, tbl_pr: etree._Element | None, node_id: str,
    is_direct: bool, style_chain: list[etree._Element],
) -> None:
    """§2 non-goal (D-1): `w:tblStylePr` conditional band formatting
    (firstRow, band1Horz, ...) is NOT resolved — Word's full conditional
    cascade plus band arithmetic is a multiple of D-1 for a banding
    effect the base grid does not depend on. Converts the silence into a
    KNOWN gap: warn when the table's OWN `w:tblStyle` (never the default
    table style — matches the census predicate, `is_direct`) resolves
    through a style whose chain carries ANY `w:tblStylePr` AND
    `w:tblLook` enables that band. Takes the ALREADY-resolved chain
    (`_resolve_table_style`) rather than resolving again, so an
    unresolved or over-deep style id is never reported twice.
    """
    if tbl_pr is None or not is_direct or not _tbl_look_enables_bands(tbl_pr):
        return
    if any(style.find(_w("tblStylePr")) is not None for style in style_chain):
        tbl_style_el = tbl_pr.find(_w("tblStyle"))
        style_id = tbl_style_el.get(_w("val")) if tbl_style_el is not None else None
        ctx.issue(
            "IMPORT_TABLE_STYLE_CONDITIONAL_APPROXIMATED", "warning",
            f"table style '{style_id}' carries w:tblStylePr conditional band "
            "formatting (firstRow/band1Horz/...), which word-core-v0 does not "
            "resolve; only the table's base border/shading frame is imported "
            "(approximate import).",
            el=tbl_pr, node_id=node_id, style_id=style_id,
        )
        ctx.degrade(node_id, "approximate")


def _table_frame_borders(
    ctx: _Ctx, border_ctx: dict[str, etree._Element | None], node_id: str
) -> dict[str, Any] | None:
    """`table.style.borders` (D-1 §1.5 Option C): the resolved perimeter
    frame only — cell direct formatting never reaches the table node."""
    frame = {
        side: _border_side(ctx, border_ctx[side], node_id)
        for side in ("top", "left", "bottom", "right")
        if border_ctx[side] is not None
    }
    return frame or None


def _resolve_cell_borders(
    ctx: _Ctx, tc_borders: etree._Element | None,
    border_ctx: dict[str, etree._Element | None], node_id: str, *,
    top_perimeter: bool, bottom_perimeter: bool,
    left_perimeter: bool, right_perimeter: bool,
) -> dict[str, Any] | None:
    """Effective per-edge border for one cell (D-1 §1.2 rank 1 first,
    else the table-level context for that edge's perimeter/interior
    slot). `cell.style.borders` records this cell's CONTRIBUTION to
    each of its four edges, not the arbitrated ink Word finally paints
    on a shared edge — D-1 §1.4, an explicit non-goal."""
    is_perimeter = {
        "top": top_perimeter, "bottom": bottom_perimeter,
        "left": left_perimeter, "right": right_perimeter,
    }
    context_slot = {"top": "insideH", "bottom": "insideH", "left": "insideV", "right": "insideV"}
    borders: dict[str, Any] = {}
    for side in ("top", "left", "bottom", "right"):
        direct_el = tc_borders.find(_w(side)) if tc_borders is not None else None
        if direct_el is not None:
            borders[side] = _border_side(ctx, direct_el, node_id)
            continue
        context_el = border_ctx[side if is_perimeter[side] else context_slot[side]]
        if context_el is not None:
            borders[side] = _border_side(ctx, context_el, node_id)
    return borders or None


def _resolve_table_borders(
    ctx: _Ctx, table_node: dict[str, Any], border_ctx: dict[str, etree._Element | None]
) -> None:
    """Per-cell pass over the FINAL grid (D-1 §1.2): must run after
    `_resolve_row_spans` (row_span not settled before) and after
    `_pad_grid_gaps` — synthesized pad cells are real cells in the
    rendered grid and get the same per-edge resolution, not a hole."""
    num_rows = len(table_node["rows"])
    num_columns = len(table_node["columns"])
    for row_index, row in enumerate(table_node["rows"]):
        for cell in row["cells"]:
            tc_borders_el = cell.pop("_tc_borders_el", None)
            grid_column = cell["grid_column"]
            col_span = cell["col_span"]
            row_span = cell["row_span"]
            cell["style"]["borders"] = _resolve_cell_borders(
                ctx, tc_borders_el, border_ctx, cell["id"],
                top_perimeter=row_index == 0,
                bottom_perimeter=row_index + row_span == num_rows,
                left_perimeter=grid_column == 0,
                right_perimeter=grid_column + col_span == num_columns,
            )


def _paragraph_node(ctx: _Ctx, p: etree._Element, p_pr: etree._Element | None) -> dict[str, Any]:
    node_id = ctx.new_id()
    ctx.map_node(node_id, p)
    style_chain: list[etree._Element] = []
    if p_pr is not None:
        p_style = p_pr.find(_w("pStyle"))
        if p_style is not None:
            style_chain, _p_chain_status = _resolve_style_chain(
                ctx, p_style.get(_w("val")), node_id
            )
    chain: list[etree._Element | None] = [
        p_pr, *(style.find(_w("pPr")) for style in style_chain), ctx.dd_ppr,
    ]

    outline = _chain_find(chain, "outlineLvl")
    semantic = {"role": "body", "heading_level": None, "outline_level": None}
    if outline is not None:
        outline_val = _int_attr(
            ctx, outline, _w("val"), default=0, node_id=node_id, required=True
        )
        if 0 <= outline_val <= 8:
            semantic = {
                "role": "heading",
                "heading_level": outline_val + 1,
                "outline_level": None,
            }
        elif outline_val != 9:
            # OOXML: outlineLvl 9 means body text (no heading role) —
            # exact semantics, no issue. Anything else is malformed.
            ctx.issue(
                "IMPORT_MALFORMED_VALUE", "warning",
                f"w:outlineLvl value {outline_val} is outside the valid range 0-9; "
                "paragraph imported as body text (approximate).",
                el=outline, node_id=node_id, attribute="val", value=outline_val,
            )
            ctx.degrade(node_id, "approximate")

    jc = _chain_find(chain, "jc")
    alignment = "left"
    if jc is not None:
        jc_val = jc.get(_w("val"), "left")
        alignment = _INV_ALIGNMENT.get(jc_val)
        if alignment is None:
            alignment = "left"
            ctx.issue(
                "IMPORT_ALIGNMENT_APPROXIMATED", "warning",
                f"paragraph alignment '{jc_val}' has no word-core-v0 equivalent; imported "
                "as 'left' (approximate).",
                el=p, node_id=node_id, value=jc_val,
            )
            ctx.degrade(node_id, "approximate")

    node = {
        "id": node_id,
        "type": "paragraph",
        "semantic": semantic,
        "layout": {
            "alignment": alignment,
            "direction": "ltr",
            "indent": _indent_from_chain(ctx, chain, node_id),
            "spacing": _spacing_from_chain(ctx, chain, node_id),
            "pagination": {
                "keep_with_next": _parse_bool(_chain_find(chain, "keepNext"), False),
                "keep_lines": _parse_bool(_chain_find(chain, "keepLines"), False),
                "page_break_before": _parse_bool(_chain_find(chain, "pageBreakBefore"), False),
                "widow_control": _parse_bool(_chain_find(chain, "widowControl"), True),
            },
            "tabs": _tabs_from_chain(ctx, chain, node_id),
        },
        "style": {
            "background": (
                _shd_background(ctx, p_pr, node_id) if p_pr is not None else None
            ),
            "borders": _borders_from(
                ctx, p_pr.find(_w("pBdr")) if p_pr is not None else None, node_id
            ),
        },
        "children": [],
    }
    node["_style_rpr"] = [style.find(_w("rPr")) for style in style_chain]
    # The resolved pPr chain, so `numPr` can be read leaf-first like
    # every other paragraph property (D6). It used to be read from the
    # direct `w:pPr` alone, which lost numbering inherited through
    # `w:pStyle` -> styles.xml -> `w:numPr` with NO issue, NO degradation
    # and NO source-map entry — the only silent loss of visible content
    # in the corpus, and a violation of RFC003 §18 item 14.
    node["_ppr_chain"] = chain
    return node


# --- inline content -----------------------------------------------------


def _parse_run(ctx: _Ctx, r: etree._Element, style_rprs: list[etree._Element | None],
               out: list[dict[str, Any]]) -> None:
    r_pr = r.find(_w("rPr"))
    rstyle_chain: list[etree._Element] = []
    rstyle_status: str | None = None
    if r_pr is not None:
        r_style = r_pr.find(_w("rStyle"))
        if r_style is not None:
            rstyle_chain, rstyle_status = _resolve_style_chain(
                ctx, r_style.get(_w("val")), None
            )
    chain: list[etree._Element | None] = [
        r_pr, *(style.find(_w("rPr")) for style in rstyle_chain),
        *style_rprs, ctx.dd_rpr,
    ]
    lang_el = _chain_find(chain, "lang")
    language = lang_el.get(_w("val")) if lang_el is not None else None

    def text_run(el: etree._Element, text: str,
                 style_override: dict[str, Any] | None = None,
                 status: str | None = rstyle_status) -> dict[str, Any]:
        node_id = ctx.new_id()
        ctx.map_node(node_id, el)
        if status is not None:
            ctx.degrade(node_id, status)
        style = _run_style(ctx, chain, node_id)
        if style_override:
            style.update(style_override)
        return {
            "id": node_id,
            "type": "run",
            "text": text,
            "language": language,
            "style": style,
        }

    for child in r:
        tag = etree.QName(child).localname
        if tag == "rPr":
            continue
        if tag in ("fldChar", "instrText", "delInstrText"):
            _parse_field_control(ctx, child, tag)
            continue
        if ctx.in_field_code():
            # Inside a field's instruction region: any visible child is
            # field code, never a stable result (RFC001 §30) — recorded
            # on the open field frame, reported when the field closes.
            # EVERY inline item the importer normally handles (text,
            # tabs, breaks, hyphens, symbols, note refs, images) is
            # still parsed and emitted, held pending on the frame: a
            # field that terminates drops it from the AST (recorded
            # dropped in the source map); a field that never terminates
            # keeps it visible (its swallowed content is treated as
            # visible result content, never silently lost).
            if tag == "t" and child.text:
                ctx.field_stack[-1]["instr"].append(child.text)
            held: list[dict[str, Any]] = []
            _parse_run_child(ctx, child, tag, text_run, held)
            frame = ctx.field_stack[-1]
            for item in held:
                item["_field_pending"] = True
                frame["pending"].append(item)
                out.append(item)
            continue
        before = len(out)
        _parse_run_child(ctx, child, tag, text_run, out)
        if ctx.field_stack:
            frame = ctx.field_stack[-1]
            frame["result_out"] = out
            frame["result"].extend(out[before:])


def _parse_run_child(ctx: _Ctx, child: etree._Element, tag: str, text_run: Any,
                     out: list[dict[str, Any]]) -> None:
    """Dispatch one non-field run child into `out` — shared by the
    normal path and the field instruction-region pending capture."""
    if tag == "t":
        text = child.text or ""
        pending = ctx.pending_custom_mark
        if pending is not None:
            ctx.pending_custom_mark = None
            if text:
                # This IS the custom mark the preceding reference
                # announced: it belongs to the note_ref, not beside it.
                pending["marker_text"] = text
                ctx.issue(
                    "IMPORT_NOTE_CUSTOM_MARK_CONSUMED", "info",
                    f"custom note mark {text!r} imported as the preceding note_ref's "
                    "marker_text instead of a separate run.",
                    el=child, node_id=pending["id"], marker=text,
                )
                return
        out.append(text_run(child, text))
    elif tag == "sym":
        _parse_sym(ctx, child, text_run, out)
    elif tag == "softHyphen":
        # U+00AD IS the soft hyphen — an exact import.
        out.append(text_run(child, "\u00ad"))
    elif tag == "noBreakHyphen":
        out.append(text_run(child, "\u2011"))
    elif tag in ("footnoteReference", "endnoteReference"):
        _parse_note_reference(ctx, child, tag, text_run, out)
    elif tag == "tab":
        node_id = ctx.new_id()
        ctx.map_node(node_id, child)
        out.append({"id": node_id, "type": "tab"})
    elif tag == "br":
        br_type = child.get(_w("type"))
        if br_type in ("page", "column"):
            # Block-level break in the middle of a paragraph: the
            # paragraph is split around a break node by _parse_p.
            out.append({"__break__": br_type, "el": child})
        else:
            node_id = ctx.new_id()
            ctx.map_node(node_id, child)
            out.append({"id": node_id, "type": "line_break"})
    elif tag == "cr":
        # w:cr is a carriage return — same visible effect as a
        # textWrapping w:br.
        node_id = ctx.new_id()
        ctx.map_node(node_id, child)
        out.append({"id": node_id, "type": "line_break"})
    elif tag == "drawing":
        node = _parse_drawing(ctx, child)
        if node is not None:
            if node.get("type") == "text_frame":
                out.append({"__block__": node, "el": child})
            else:
                out.append(node)
    elif tag == "pict":
        node = _parse_vml_pict(ctx, child)
        if node is not None:
            if node.get("type") == "text_frame":
                out.append({"__block__": node, "el": child})
            else:
                out.append(node)
    elif tag == "AlternateContent":
        _parse_alternate_content(ctx, child, out)
    elif tag == "ptab":
        # An absolute-position tab (CT_PTab): it advances to a position
        # on the line, which the AST's `tab` expresses only as "advance
        # to the next stop". Imported as a tab with the difference
        # declared (M3) — it is a gap in both profiles, not a v1
        # feature, so it is not gated on one.
        node_id = ctx.new_id()
        ctx.map_node(node_id, child, status="approximate")
        out.append({"id": node_id, "type": "tab"})
        ctx.issue(
            "IMPORT_POSITIONAL_TAB_APPROXIMATED", "warning",
            "w:ptab is an absolute-position tab; imported as a plain tab, which "
            "advances to the next tab stop instead (approximate import).",
            el=child, node_id=node_id,
            alignment=child.get(_w("alignment")),
            relative_to=child.get(_w("relativeTo")),
        )
    elif tag in ("footnoteRef", "endnoteRef") and ctx.note_mark_key is not None:
        # Word repeats the note's mark at the START OF THE BODY, and
        # that mark is ordinary visible ink at the foot of the page —
        # distinct from `note_ref.marker_text`, which is the
        # REFERENCE-side mark up in the text. The AST has no mark node,
        # so if this were dropped the body mark would be gone for good
        # and no round-trip could catch it. It becomes a real run whose
        # text is filled in once the referencing note_ref resolves the
        # number (`_emit_note_ref`).
        run = text_run(child, "")
        ctx.note_mark_runs.setdefault(ctx.note_mark_key, []).append(run)
        out.append(run)
        ctx.issue(
            "IMPORT_NOTE_BODY_MARK_RESOLVED", "info",
            f"note-body mark 'w:{tag}' imported as a run carrying the resolved "
            "mark: it is visible ink, and nothing is synthesized at render time.",
            el=child, node_id=run["id"], element=tag,
        )
    elif tag == "lastRenderedPageBreak":
        return
    else:
        ctx.issue(
            "IMPORT_UNSUPPORTED_ELEMENT", "error",
            f"run child element 'w:{tag}' is not supported by the word-core-v0 importer.",
            el=child, element=tag,
        )


def _parse_field_control(ctx: _Ctx, child: etree._Element, tag: str) -> None:
    """Complex-field control elements (RFC001 §30): the visible result
    runs (between separate and end) import as plain runs; the
    instruction text is dropped into the source map only, never the AST.
    A field with no separate showed no stable result — dropped whole.
    """
    if tag in ("instrText", "delInstrText"):
        if ctx.field_stack:
            if child.text:
                ctx.field_stack[-1]["instr"].append(child.text)
        else:
            ctx.issue(
                "IMPORT_FIELD_FLATTENED", "warning",
                f"orphan w:{tag} outside any open field; instruction text dropped "
                "(recorded in the source map).",
                el=child,
            )
            prov = ctx.prov(child)
            ctx.source_map[f"field:{prov}"] = {
                "part": ctx.part,
                "provenance": prov,
                "import_status": "dropped",
                "instruction": child.text or None,
            }
        return
    fld_type = child.get(_w("fldCharType"))
    if fld_type == "begin":
        ctx.field_stack.append(
            {
                "separated": False, "instr": [], "el": child, "pending": [],
                # Items emitted after w:fldChar separate — the field's
                # CACHED result. Kept so a whitelisted field can replace
                # them with a live `field` node at their exact position.
                "result": [], "result_out": None,
            }
        )
    elif fld_type == "separate":
        if ctx.field_stack:
            ctx.field_stack[-1]["separated"] = True
        else:
            ctx.issue(
                "IMPORT_FIELD_FLATTENED", "warning",
                "orphan w:fldChar separate outside any open field; ignored.",
                el=child,
            )
    elif fld_type == "end":
        if not ctx.field_stack:
            ctx.issue(
                "IMPORT_FIELD_FLATTENED", "warning",
                "unmatched w:fldChar end ignored.", el=child,
            )
            return
        frame = ctx.field_stack.pop()
        # The field terminated: its instruction-region items were field
        # code, never a stable result — dropped from the AST. Items
        # without an id (inline page/column break sentinels) are dropped
        # by _split_paragraph_on_breaks instead.
        for item in frame["pending"]:
            item["_field_drop"] = True
            entry = ctx.source_map.get(item.get("id", ""))
            if entry is not None:
                entry["import_status"] = "dropped"
        instruction = "".join(frame["instr"]).strip() or None
        prov = ctx.prov(frame["el"])
        if ctx.options.visual_v1 and frame["separated"]:
            field_type = visual_v1.parse_field_instruction(ctx, instruction, child)
            if field_type is not None:
                # The cached result runs are the field code's OUTPUT, not
                # content: a stored "1" would be wrong on every page but
                # the first. They are dropped for the live field node.
                _replace_field_result(ctx, frame, child, field_type, instruction)
                return
        if frame["separated"]:
            ctx.source_map[f"field:{prov}"] = {
                "part": ctx.part,
                "provenance": prov,
                "import_status": "approximate",
                "instruction": instruction,
            }
            ctx.issue(
                "IMPORT_FIELD_FLATTENED", "warning",
                "field flattened to its visible result runs; the instruction text is "
                "recorded in the source map only, never the AST (approximate import).",
                el=child,
            )
        else:
            ctx.source_map[f"field:{prov}"] = {
                "part": ctx.part,
                "provenance": prov,
                "import_status": "dropped",
                "instruction": instruction,
            }
            ctx.issue(
                "IMPORT_FIELD_NO_RESULT", "warning",
                "field has no separate/result region and showed nothing stable; the "
                "whole field was dropped (instruction recorded in the source map).",
                el=child,
            )


def _field_node(ctx: _Ctx, el: etree._Element, field_type: str,
                instruction: str | None) -> dict[str, Any]:
    node_id = ctx.new_id()
    ctx.map_node(node_id, el)
    prov = ctx.prov(el)
    ctx.source_map[f"field:{prov}"] = {
        "part": ctx.part,
        "provenance": prov,
        "import_status": "exact",
        "instruction": instruction,
    }
    return {"id": node_id, "type": "field", "field_type": field_type}


def _replace_field_result(ctx: _Ctx, frame: dict[str, Any], child: etree._Element,
                          field_type: str, instruction: str | None) -> None:
    """Swap a terminated complex field's cached result runs for a live
    `field` node, in the list those runs were emitted into.

    The result runs were appended to the paragraph as they were parsed,
    so the node takes the position of the FIRST of them and the rest are
    marked for the sweep — the field lands exactly where it rendered.
    """
    node = _field_node(ctx, child, field_type, instruction)
    siblings = frame.get("result_out")
    result = frame.get("result", [])
    index = None
    if siblings is not None:
        index = next(
            (i for i, item in enumerate(siblings)
             if any(item is r for r in result)),
            None,
        )
    for item in result:
        item["_field_drop"] = True
        entry = ctx.source_map.get(item.get("id", ""))
        if entry is not None:
            entry["import_status"] = "dropped"
    if index is None:
        # No result runs to position against (an empty result region):
        # the field belongs where the field ended.
        siblings = frame.get("result_out")
        if siblings is None:
            ctx.issue(
                "IMPORT_FIELD_FLATTENED", "warning",
                f"{field_type} field could not be positioned in its paragraph; "
                "flattened instead (approximate import).",
                el=child,
            )
            return
        siblings.append(node)
        return
    siblings.insert(index, node)


def _drain_field_stack(ctx: _Ctx) -> None:
    """Close every complex-field frame still open at end of body.

    An unterminated field (fldChar begin with no end) has swallowed all
    subsequent visible text as instruction; Word's rendering of such a
    field is ambiguous, so the safest faithful rule — documented here —
    is: an unterminated field's swallowed content is treated as visible
    result text and re-emitted as plain runs (the pending runs simply
    stay in the AST). The captured instruction goes to the source map
    and a typed `IMPORT_FIELD_UNTERMINATED` warning is reported — total
    silent text loss is never acceptable (RFC003 §45).
    """
    for frame in ctx.field_stack:
        instruction = "".join(frame["instr"]).strip() or None
        prov = ctx.prov(frame["el"])
        ctx.source_map[f"field:{prov}"] = {
            "part": ctx.part,
            "provenance": prov,
            "import_status": "approximate",
            "instruction": instruction,
        }
        ctx.issue(
            "IMPORT_FIELD_UNTERMINATED", "warning",
            "w:fldChar begin has no matching end; the field's swallowed content "
            "was re-emitted as visible result text and the instruction recorded "
            "in the source map (approximate import).",
            el=frame["el"],
        )
        # pending runs stay visible; the sweep strips their markers.
    ctx.field_stack.clear()


def _sweep_field_markers(node: Any) -> None:
    """Strip `_field_pending` markers and remove runs marked
    `_field_drop` (instruction-region runs of fields that terminated)
    from the finished AST tree.

    `headers`/`footers` are on the key list because a section's stories
    hang off their own keys, not off `children` — and a page-number
    field, the single most common field in the corpus, lives almost only
    there (38.3 % of documents, against 0.3 % in the body). Missing them
    left dropped result runs in the tree, which the schema rejects.
    """
    if not isinstance(node, dict):
        return
    node.pop("_field_pending", None)
    for key in ("children", "items", "cells", "rows", "headers", "footers"):
        value = node.get(key)
        if isinstance(value, list):
            value[:] = [
                child for child in value
                if not (isinstance(child, dict) and child.pop("_field_drop", False))
            ]
            for child in value:
                _sweep_field_markers(child)


def _flatten_fld_simple(ctx: _Ctx, fld: etree._Element,
                        style_rprs: list[etree._Element | None],
                        out: list[dict[str, Any]]) -> None:
    """`w:fldSimple`: unwrap the child runs as the visible field result;
    the `w:instr` attribute goes to the source map only (RFC001 §30).
    """
    instruction = fld.get(_w("instr"))
    if ctx.options.visual_v1:
        field_type = visual_v1.parse_field_instruction(ctx, instruction, fld)
        if field_type is not None:
            out.append(_field_node(ctx, fld, field_type, instruction))
            return
    prov = ctx.prov(fld)
    ctx.source_map[f"field:{prov}"] = {
        "part": ctx.part,
        "provenance": prov,
        "import_status": "approximate",
        "instruction": instruction,
    }
    ctx.issue(
        "IMPORT_FIELD_FLATTENED", "warning",
        "w:fldSimple flattened to its visible result runs; the instruction is "
        "recorded in the source map only, never the AST (approximate import).",
        el=fld,
    )
    for child in _iter_content(ctx, fld):
        tag = etree.QName(child).localname
        if tag == "r":
            _parse_run(ctx, child, style_rprs, out)
        elif tag == "fldSimple":
            _flatten_fld_simple(ctx, child, style_rprs, out)
        elif tag in _IGNORED_PARAGRAPH_CHILDREN or tag == "fldData":
            continue
        else:
            ctx.issue(
                "IMPORT_UNSUPPORTED_ELEMENT", "error",
                f"fldSimple child element 'w:{tag}' is not supported by the "
                "word-core-v0 importer.",
                el=child, element=tag,
            )


def _parse_sym(ctx: _Ctx, child: etree._Element, text_run: Any,
               out: list[dict[str, Any]]) -> None:
    """`w:sym`: a single glyph addressed by font + hex char code.
    Imported as run text — the F0xx private-use high byte is stripped to
    the base codepoint and the symbol font becomes the run's font, so
    the glyph identity (font + codepoint) survives (approximate import).
    """
    raw_char = child.get(_w("char")) or ""
    font = child.get(_w("font"))
    try:
        codepoint = int(raw_char, 16)
    except ValueError:
        ctx.issue(
            "IMPORT_MALFORMED_VALUE", "warning",
            f"w:sym char value {raw_char!r} is not valid hex; symbol dropped.",
            el=child, attribute="char", value=raw_char,
        )
        return
    if not (0 <= codepoint <= 0x10FFFF) or 0xD800 <= codepoint <= 0xDFFF:
        ctx.issue(
            "IMPORT_MALFORMED_VALUE", "warning",
            f"w:sym char value {raw_char!r} is outside the valid Unicode "
            "codepoint range (or a surrogate); symbol dropped.",
            el=child, attribute="char", value=raw_char,
        )
        return
    if 0xF000 <= codepoint <= 0xF0FF:
        codepoint -= 0xF000
    if find_illegal_xml_char(chr(codepoint)) is not None:
        # The F0xx unmapping can land on the C0 control range (e.g.
        # w:char="F001" -> \x01): a private-use symbol-font glyph with
        # no XML-legal textual representation. word-core-v0/word-visual
        # AST text must stay renderable (RFC001) — the renderer cannot
        # emit a literal control character into `w:t` — so the glyph is
        # dropped rather than smuggling a non-renderable byte into the
        # AST for the renderer to choke on three stages later.
        ctx.issue(
            "IMPORT_SYMBOL_CONTROL_CHAR_DROPPED", "warning",
            f"w:sym char value {raw_char!r} maps to XML-illegal control "
            f"codepoint U+{codepoint:04X}; symbol dropped (no renderable "
            "glyph text exists for it).",
            el=child, attribute="char", value=raw_char,
        )
        return
    override = {"font_family": font} if font else None
    run = text_run(child, chr(codepoint), override, "approximate")
    out.append(run)
    ctx.issue(
        "IMPORT_SYMBOL_APPROXIMATED", "warning",
        "w:sym imported as run text with the symbol font as the run font "
        "(approximate import).",
        el=child, node_id=run["id"], font=font, char=raw_char,
    )


_NOTE_MARKER_FORMATS = frozenset({
    "decimal", "lowerRoman", "upperRoman", "lowerLetter", "upperLetter",
})
_ROMAN_NUMERALS = (
    (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
    (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
)
_DEFAULT_NOTE_SETTINGS: dict[str, Any] = {
    "fmt": "decimal", "start": 1, "restart": "continuous", "fallback_from": None,
}


def _note_settings(ctx: _Ctx, sect_pr: etree._Element) -> dict[str, dict[str, Any]]:
    """Read one sectPr's `w:footnotePr`/`w:endnotePr` numbering settings
    (numFmt, numStart, numRestart). Unknown numFmt values (chicago,
    symbol lists, …) fall back to decimal; the original value is kept so
    each affected reference reports it in its typed issue details.
    """
    conf: dict[str, dict[str, Any]] = {}
    for tag, kind in (("footnotePr", "footnote"), ("endnotePr", "endnote")):
        entry = dict(_DEFAULT_NOTE_SETTINGS)
        pr = sect_pr.find(_w(tag))
        if pr is not None:
            fmt_el = pr.find(_w("numFmt"))
            if fmt_el is not None:
                fmt = fmt_el.get(_w("val")) or "decimal"
                if fmt in _NOTE_MARKER_FORMATS:
                    entry["fmt"] = fmt
                else:
                    entry["fallback_from"] = fmt
            start_el = pr.find(_w("numStart"))
            if start_el is not None:
                entry["start"] = _int_attr(ctx, start_el, _w("val"), default=1) or 1
            restart_el = pr.find(_w("numRestart"))
            if restart_el is not None:
                entry["restart"] = restart_el.get(_w("val")) or "continuous"
        conf[kind] = entry
    return conf


def _current_note_settings(ctx: _Ctx, kind: str) -> dict[str, Any]:
    if ctx.section_index < len(ctx.note_conf):
        return ctx.note_conf[ctx.section_index][kind]
    return _DEFAULT_NOTE_SETTINGS


def _advance_note_section(ctx: _Ctx) -> None:
    """Move the note-numbering machinery to the next section: restart
    the counters when the entered section's numRestart is eachSect.
    """
    ctx.section_index += 1
    if ctx.section_index < len(ctx.note_conf):
        conf = ctx.note_conf[ctx.section_index]
        if conf["footnote"]["restart"] == "eachSect":
            ctx.footnote_count = 0
        if conf["endnote"]["restart"] == "eachSect":
            ctx.endnote_count = 0


def _format_note_marker(number: int, fmt: str) -> str:
    if number < 1 or fmt == "decimal":
        return str(number)
    if fmt in ("lowerRoman", "upperRoman"):
        parts: list[str] = []
        n = number
        for value, symbol in _ROMAN_NUMERALS:
            while n >= value:
                parts.append(symbol)
                n -= value
        marker = "".join(parts)
        return marker.upper() if fmt == "upperRoman" else marker
    # lowerLetter / upperLetter: a..z, aa, bb, ... (Word semantics)
    letter = chr(ord("a") + (number - 1) % 26) * ((number - 1) // 26 + 1)
    return letter.upper() if fmt == "upperLetter" else letter


def _parse_note_reference(ctx: _Ctx, child: etree._Element, tag: str, text_run: Any,
                          out: list[dict[str, Any]]) -> None:
    """Footnote/endnote reference: the visible marker Word renders is
    the note number — derived by counting references in document order
    (footnotes and endnotes separately), formatted per the governing
    section's `w:footnotePr`/`w:endnotePr` numFmt and offset by
    numStart. The note body part stays unimported (word-core-v0
    excludes notes) and is recorded in the source map. A custom-marked
    reference emits no number: its visible marker arrives as the
    following run text.
    """
    kind = "footnote" if tag == "footnoteReference" else "endnote"
    if ctx.options.visual_v1 and not ctx.in_note_body:
        if _emit_note_ref(ctx, child, kind, out):
            return
    prov = ctx.prov(child)
    ctx.source_map[f"{kind}:{prov}"] = {
        "part": ctx.part,
        "provenance": prov,
        "import_status": "dropped",
        "note_id": child.get(_w("id")),
    }
    custom_mark = child.get(_w("customMarkFollows")) in ("1", "true")
    number: int | None = None
    details: dict[str, Any] = {}
    if not custom_mark:
        settings = _current_note_settings(ctx, kind)
        if kind == "footnote":
            ctx.footnote_count += 1
            count = ctx.footnote_count
        else:
            ctx.endnote_count += 1
            count = ctx.endnote_count
        number = settings["start"] + count - 1
        marker = _format_note_marker(number, settings["fmt"])
        details["marker"] = marker
        if settings["fallback_from"] is not None:
            details["format_fallback"] = settings["fallback_from"]
            details["note"] = (
                f"numFmt '{settings['fallback_from']}' has no word-core-v0 "
                "marker mapping; decimal used."
            )
        run = text_run(
            child, marker,
            {"superscript": True, "subscript": False}, "approximate",
        )
        out.append(run)
    ctx.issue(
        "IMPORT_FOOTNOTE_REF_AS_TEXT", "warning",
        f"{kind} reference imported as its visible marker text"
        f"{' (custom mark follows in run text)' if custom_mark else ''}; the note "
        "body is outside word-core-v0 and stays unimported (recorded in the source "
        "map; approximate import).",
        el=child, kind=kind, number=number, **details,
    )


def _emit_note_ref(ctx: _Ctx, child: etree._Element, kind: str,
                   out: list[dict[str, Any]]) -> bool:
    """`w:footnoteReference`/`w:endnoteReference` -> a `note_ref` node.

    `marker_text` is the mark Word renders — the number derived from the
    governing section's numFmt/numStart, or the custom mark. RFC001 has
    no section-level note-numbering property, so the resolved literal
    mark is where that setting is preserved (a `note_ref` inside a note
    body is forbidden by the semantic validator, and never reaches here).

    Returns False when the referenced body is absent from the package,
    so the caller falls back to importing the visible marker as text
    rather than emitting a dangling `note_id`.
    """
    source_id = child.get(_w("id"))
    body = visual_v1.note_body_for(ctx, kind, source_id)
    if body is None:
        ctx.issue(
            "IMPORT_NOTE_BODY_MISSING", "warning",
            f"{kind} reference w:id={source_id!r} resolves to no body in the "
            f"{kind}s part; imported as its visible marker text (approximate import).",
            el=child, kind=kind, note_id=source_id,
        )
        return False
    settings = _current_note_settings(ctx, kind)
    custom_mark = child.get(_w("customMarkFollows")) in ("1", "true")
    if custom_mark:
        # The visible mark is the text that FOLLOWS the reference (Word's
        # w:customMarkFollows). It is the note_ref's marker_text, so it
        # is consumed there rather than also emitted as a run — which
        # would print the mark twice. A provisional number stands in
        # until that text arrives, because `marker_text` may not be empty.
        marker = _format_note_marker(settings["start"], settings["fmt"])
    else:
        if kind == "footnote":
            ctx.footnote_count += 1
            count = ctx.footnote_count
        else:
            ctx.endnote_count += 1
            count = ctx.endnote_count
        marker = _format_note_marker(
            settings["start"] + count - 1, settings["fmt"]
        )
        if settings["fallback_from"] is not None:
            ctx.issue(
                "IMPORT_NOTE_FORMAT_FALLBACK", "warning",
                f"{kind} numFmt '{settings['fallback_from']}' has no marker mapping; "
                f"decimal marker {marker!r} used (approximate import).",
                el=child, kind=kind, format_fallback=settings["fallback_from"],
            )
    for run in ctx.note_mark_runs.get((kind, source_id or ""), ()):
        run["text"] = marker
    node_id = ctx.new_id()
    ctx.map_node(node_id, child)
    node = {
        "id": node_id, "type": "note_ref",
        "note_id": body["id"], "marker_text": marker,
    }
    out.append(node)
    if custom_mark:
        ctx.pending_custom_mark = node
    return True


def _drop_floating_or_report(ctx: _Ctx, el: etree._Element, what: str) -> None:
    """A floating object (anchored drawing / VML shape): dropped with a
    typed issue and a source-map record under the `allow_floating_drop`
    policy, otherwise a typed error — never silent (RFC003 §45).

    A floating object that carries `w:txbxContent` (legacy VML
    `v:textbox`; DrawingML `wps:txbx` reaches this function only via the
    anchor-not-allowed path in `_parse_drawing`) is a text box, and its
    loss is reported by `_report_text_box_content_dropped` instead — a
    generic "floating object dropped" would misname the same D-2a
    problem `IMPORT_IMAGE_RESOURCE_MISSING` used to have for the
    no-blip case.
    """
    if ctx.options.allow_floating_drop:
        txbx_contents = list(el.iter(f"{{{_W}}}txbxContent"))
        if txbx_contents:
            for txbx_content in txbx_contents:
                _report_text_box_content_dropped(ctx, txbx_content)
            return
        prov = ctx.prov(el)
        ctx.source_map[f"dropped:{prov}"] = {
            "part": ctx.part,
            "provenance": prov,
            "import_status": "dropped",
        }
        ctx.issue(
            "IMPORT_FLOATING_OBJECT_DROPPED", "warning",
            f"{what}; dropped under the allow_floating_drop policy and recorded in the "
            "source map (approximate import).",
            el=el,
        )
    else:
        ctx.issue(
            "IMPORT_UNSUPPORTED_FLOATING_OBJECT", "error",
            f"{what} (set allow_floating_drop to import the document without it).",
            el=el,
        )


def _report_text_box_content_dropped(
    ctx: _Ctx, txbx_content: etree._Element, *, reason: str | None = None,
) -> None:
    """`w:txbxContent` (a text box body — DrawingML `wps:txbx` or legacy
    VML `v:textbox`) has no word-core-v0 representation (D-2b's
    `text_frame` node is a later, separate deliverable — §5A). Its
    paragraphs are dropped; reported once per text box with the dropped
    paragraph count, instead of the misattributed
    `IMPORT_IMAGE_RESOURCE_MISSING` a no-blip drawing used to produce
    (D-2a, RFC003 §45: no silent loss).
    """
    prov = ctx.prov(txbx_content)
    paragraph_count = len(txbx_content.findall(_w("p")))
    ctx.source_map[f"dropped:{prov}"] = {
        "part": ctx.part,
        "provenance": prov,
        "import_status": "dropped",
    }
    explanation = reason or "has no representation in the selected AST profile"
    ctx.issue(
        "IMPORT_TEXT_BOX_CONTENT_DROPPED", "warning",
        f"text box content ({paragraph_count} paragraph(s)) {explanation} and was "
        "dropped; recorded in the source map "
        "(approximate import).",
        el=txbx_content, paragraph_count=paragraph_count, reason=reason,
    )


def _top_level_txbx_contents(host: etree._Element) -> list[etree._Element]:
    """Text-box bodies owned by ``host``, excluding nested drawings/shapes."""
    contents: list[etree._Element] = []
    for content in host.iter(_w("txbxContent")):
        ancestor = content.getparent()
        nested = False
        while ancestor is not None and ancestor is not host:
            if ancestor.tag == _w("txbxContent"):
                nested = True
                break
            ancestor = ancestor.getparent()
        if not nested:
            contents.append(content)
    return contents


def _parse_text_frame(
    ctx: _Ctx,
    host: etree._Element,
    txbx_contents: list[etree._Element],
    *,
    anchor_el: etree._Element | None = None,
    vml_shape: etree._Element | None = None,
) -> dict[str, Any] | None:
    """Normalize one supported DrawingML/VML text box to ``text_frame``."""
    if not ctx.options.text_frames:
        for content in txbx_contents:
            _report_text_box_content_dropped(ctx, content)
        return None
    if ctx.text_frame_depth:
        for content in txbx_contents:
            _report_text_box_content_dropped(
                ctx, content, reason="is nested inside another text frame",
            )
        return None
    if anchor_el is None and vml_shape is None:
        for content in txbx_contents:
            _report_text_box_content_dropped(
                ctx, content, reason="has no supported anchored host",
            )
        return None
    if len(txbx_contents) != 1:
        target = txbx_contents or [host]
        for content in target:
            _report_text_box_content_dropped(
                ctx, content,
                reason=f"has a malformed host with {len(txbx_contents)} text bodies",
            )
        return None
    if _text_frame_is_rotated(anchor_el, vml_shape):
        _report_text_box_content_dropped(
            ctx, txbx_contents[0],
            reason="is rotated, and the ratified text_frame anchor (RFC001-ERRATA-004 E1) "
            "carries no rotation; rendering it unrotated would misplace it worse than "
            "dropping it (RFC001 §0.1: no pixel/angle geometry enters the AST)",
        )
        return None

    issues_at_frame = len(ctx.issues)
    map_keys_at_frame = set(ctx.source_map)
    node_id = ctx.new_id()
    ctx.map_node(node_id, host)
    anchor = (
        visual_v1.parse_anchor(ctx, anchor_el, node_id)
        if anchor_el is not None
        else visual_v1.parse_vml_anchor(ctx, vml_shape, node_id)
    )
    ctx.text_frame_depth += 1
    issues_before = len(ctx.issues)
    map_keys_before = set(ctx.source_map)
    try:
        children = visual_v1.parse_story_blocks(ctx, txbx_contents[0], "text frame")
    finally:
        ctx.text_frame_depth -= 1
    if any(issue.severity in ("error", "fatal") for issue in ctx.issues[issues_before:]):
        # An element inside the frame's own content (an OMML equation, a
        # tracked-change w:del, w:pgNum, ...) has no importer support.
        # Scoped to THIS frame: the whole walk that produced the error(s)
        # is confined to `txbx_contents[0]`, so undoing its issues and
        # source-map entries and falling back to the pre-existing
        # `IMPORT_TEXT_BOX_CONTENT_DROPPED` path costs only the frame,
        # never the document. `_parse_body` and `parse_story_blocks` are
        # untouched, so the SAME unsupported element in the document body
        # (or directly in a header/footer story) still raises its own
        # "error" issue there and fails the document exactly as before.
        del ctx.issues[issues_before:]
        for key in set(ctx.source_map) - map_keys_before:
            del ctx.source_map[key]
        del ctx.source_map[node_id]
        _report_text_box_content_dropped(
            ctx, txbx_contents[0],
            reason="contains an element unsupported by the word-visual-v1 importer",
        )
        return None
    if ctx.options.drop_empty_text_frames and not _story_blocks_have_ink(children):
        # DEC-055 I2: undo everything this frame recorded (anchor and
        # content issues, source-map entries) and leave one typed drop.
        del ctx.issues[issues_at_frame:]
        for key in set(ctx.source_map) - map_keys_at_frame:
            del ctx.source_map[key]
        prov = ctx.prov(host)
        ctx.source_map[f"dropped:{prov}"] = {
            "part": ctx.part,
            "provenance": prov,
            "import_status": "dropped",
        }
        ctx.issue(
            "IMPORT_EMPTY_TEXT_FRAME_DROPPED", "warning",
            "text frame has no visible content (only empty or whitespace paragraphs) "
            "and renders nothing; not emitted, recorded in the source map "
            "(approximate import).",
            el=host,
        )
        return None
    return {
        "id": node_id,
        "type": "text_frame",
        "anchor": anchor,
        "children": children,
    }


#: Inline node types that put no ink on the page by themselves (DEC-055 I2).
_INKLESS_INLINE_TYPES = frozenset({"run", "tab", "line_break"})


def _story_blocks_have_ink(blocks: list[dict[str, Any]]) -> bool:
    """Whether story content renders anything: any non-paragraph block
    (table, list, ...), any inline node other than a run/tab/line break
    (image, field, note reference, hyperlink, ...), or any run text with a
    non-whitespace character."""
    for block in blocks:
        if block.get("type") != "paragraph":
            return True
        for inline in block.get("children", ()):
            if inline.get("type") not in _INKLESS_INLINE_TYPES:
                return True
            if inline.get("type") == "run" and inline.get("text", "").strip():
                return True
    return False


def _parse_vml_pict(ctx: _Ctx, pict: etree._Element) -> dict[str, Any] | None:
    """Import the supported VML spellings: ``v:shape/v:textbox`` (text
    frame) and, when there is no textbox, ``v:shape/v:imagedata`` (the
    legacy raster picture every pre-2007-style ``w:pict`` uses).

    A ``w:pict`` that carries both a textbox and a separate imagedata
    shape still resolves through the textbox branch only, unchanged
    from before this function knew about imagedata — a genuinely mixed
    host is rare enough that reshaping the single-node return here
    isn't worth the risk to the textbox path this function already
    gets right.
    """
    textboxes = list(pict.iter(f"{{{_VML_NS}}}textbox"))
    if not textboxes:
        return _parse_vml_image(ctx, pict)
    if not ctx.options.text_frames:
        _drop_floating_or_report(
            ctx, pict,
            "a VML text box cannot be represented in the selected AST profile",
        )
        return None
    if len(textboxes) != 1:
        for textbox in textboxes:
            contents = _top_level_txbx_contents(textbox)
            for content in contents or [textbox]:
                _report_text_box_content_dropped(
                    ctx, content,
                    reason=f"has a malformed VML host with {len(textboxes)} text boxes",
                )
        return None
    textbox = textboxes[0]
    shape = textbox.getparent()
    if shape is None or shape.tag != f"{{{_VML_NS}}}shape":
        _report_text_box_content_dropped(
            ctx, textbox, reason="has a VML textbox outside v:shape",
        )
        return None
    return _parse_text_frame(
        ctx, pict, _top_level_txbx_contents(textbox), vml_shape=shape,
    )


_VML_LENGTH_RE = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(pt|in|cm|mm|px)?")
_VML_LENGTH_TO_PT = {
    "pt": 1.0, "in": 72.0, "cm": 72.0 / 2.54, "mm": 72.0 / 25.4, "px": 0.75,
}


def _vml_shape_style(shape: etree._Element) -> dict[str, str]:
    """`v:shape/@style`'s CSS-like declarations, lowercased by key (the
    same dialect `visual_v1.parse_vml_anchor` reads for positioning)."""
    declarations: dict[str, str] = {}
    for declaration in shape.get("style", "").split(";"):
        key, separator, value = declaration.partition(":")
        if separator and key.strip():
            declarations[key.strip().lower()] = value.strip()
    return declarations


def _vml_dimension_mpt(ctx: _Ctx, shape: etree._Element, raw: str | None,
                       name: str) -> int:
    """One `width`/`height` declaration -> millipoints, mirroring the
    DrawingML side's `_int_attr(..., default=0)` permissiveness for a
    missing or malformed value (D1: a malformed source degrades the
    import, it never fails it)."""
    if raw is None:
        return 0
    match = _VML_LENGTH_RE.fullmatch(raw.strip())
    if match is None:
        ctx.issue(
            "IMPORT_MALFORMED_VALUE", "warning",
            f"VML shape {name} value {raw!r} is not a supported physical length; "
            "imported approximately.",
            el=shape, attribute=name, value=raw,
        )
        return 0
    value = float(match.group(1))
    unit = match.group(2) or "pt"
    return round(value * _VML_LENGTH_TO_PT[unit] * 1000)


def _parse_vml_image(ctx: _Ctx, pict: etree._Element) -> dict[str, Any] | None:
    """Import the legacy VML raster picture spelling: `v:shape/v:imagedata`
    with an `r:id` (also spelled `o:relid`) into the owning part's media
    relationship — the same construct a DrawingML `a:blip` reaches,
    mirroring `_parse_drawing`'s relationship resolution, EMF/WMF
    rasterization and resource registration rather than a second image
    pipeline.
    """
    imagedatas = list(pict.iter(f"{{{_VML_NS}}}imagedata"))
    if not imagedatas:
        _drop_floating_or_report(
            ctx, pict, "a VML w:pict shape cannot be represented in the selected profile",
        )
        return None
    if len(imagedatas) != 1:
        _drop_floating_or_report(
            ctx, pict,
            f"a VML w:pict carries {len(imagedatas)} v:imagedata pictures, which is not "
            "representable",
        )
        return None
    imagedata = imagedatas[0]
    shape = imagedata.getparent()
    if shape is None or shape.tag != f"{{{_VML_NS}}}shape":
        _drop_floating_or_report(
            ctx, pict, "a VML v:imagedata outside v:shape cannot be represented",
        )
        return None
    if _text_frame_is_rotated(None, shape):
        # Consistent with the text_frame decision (RFC001-ERRATA-004
        # E1): the ratified anchor carries no VML rotation dialect, and
        # placing the picture unrotated would misplace it worse than
        # dropping it.
        _drop_floating_or_report(
            ctx, pict, "a rotated VML picture cannot be represented in the selected profile",
        )
        return None
    style = _vml_shape_style(shape)
    is_floating = style.get("position") == "absolute"
    if is_floating and not ctx.options.visual_v1:
        _drop_floating_or_report(
            ctx, pict,
            "an anchored (floating) VML picture cannot be represented in word-core-v0 "
            "(only an inline VML picture)",
        )
        return None
    relation_id = (
        imagedata.get(f"{{{_R_NS}}}id") or imagedata.get(f"{{{_VML_OFFICE_NS}}}relid")
    )
    if relation_id is None:
        _drop_missing_image(ctx, pict, "VML v:imagedata has no resolvable relationship id")
        return None
    target = ctx.rels.get(relation_id)
    if target is None:
        _drop_missing_image(ctx, pict, "VML picture has no resolvable graphic relationship")
        return None
    part_name = _resolve_target(ctx, target)
    data = ctx.parts.get(part_name)
    if data is None:
        _drop_missing_image(
            ctx, pict, f"graphic part '{part_name}' is missing from the package"
        )
        return None

    width_mpt = _vml_dimension_mpt(ctx, shape, style.get("width"), "width")
    height_mpt = _vml_dimension_mpt(ctx, shape, style.get("height"), "height")

    extension = part_name.rsplit(".", 1)[-1].lower()
    mime = _sniff_mime(data)
    raster = None
    if extension in {"emf", "wmf"}:
        raster = _rasterize_or_drop(
            ctx, pict, data, part_name, extension, "media",
            _mpt_to_emu(width_mpt), _mpt_to_emu(height_mpt),
        )
        if raster is None:
            return None
        data, mime = raster.png, "image/png"
    if mime is None:
        mime = _EXT_MIME.get(extension)
    elif _EXT_MIME.get(extension) != mime:
        ctx.issue(
            "IMPORT_RESOURCE_TYPE_FROM_CONTENT", "info",
            f"image part '{part_name}' is {mime} by content; its extension "
            f"'{extension}' says otherwise and the content was trusted.",
            el=pict, extension=extension, mime_type=mime,
        )
    if mime is None:
        if not ctx.options.visual_v1:
            ctx.issue(
                "IMPORT_UNSUPPORTED_RESOURCE_TYPE", "error",
                f"image extension '{extension}' is outside the word-core-v0 whitelist "
                "(png/jpeg).",
                el=pict, extension=extension,
            )
            return None
        prov = ctx.prov(pict)
        ctx.source_map[f"dropped:{prov}"] = {
            "part": ctx.part,
            "provenance": prov,
            "import_status": "dropped",
        }
        ctx.issue(
            "IMPORT_UNSUPPORTED_RESOURCE_TYPE", "warning",
            f"image extension '{extension}' is outside the resource whitelist "
            "(png/jpeg); the image was dropped and recorded in the source map, and "
            "the rest of the document imported (approximate import).",
            el=pict, extension=extension,
        )
        return None

    node_id = ctx.new_id()
    ctx.map_node(node_id, pict)
    resource_id = _register_resource(ctx, data, mime, node_id)
    if raster is not None:
        resource = ctx.resources_by_sha[hashlib.sha256(data).hexdigest()]
        ctx.resource_provenance[resource["path"]] = part_name

    node: dict[str, Any] = {
        "id": node_id,
        "type": "floating_image" if is_floating else "inline_image",
        "resource_ref": resource_id,
        "size": {"width_mpt": width_mpt, "height_mpt": height_mpt},
        "crop": {"left_ppm": 0, "top_ppm": 0, "right_ppm": 0, "bottom_ppm": 0},
        "rotation_mdeg": 0,
        "alt_text": shape.get("alt"),
    }
    if is_floating:
        node["anchor"] = visual_v1.parse_vml_anchor(ctx, shape, node_id)
    return node


def _report_drawing_no_image(ctx: _Ctx, drawing: etree._Element) -> None:
    """A `w:drawing` carrying no `a:blip` at all: a DrawingML shape with
    no image content and not a text box (that case is reported by
    `_report_text_box_content_dropped`). Distinct from
    `IMPORT_IMAGE_RESOURCE_MISSING`, which means image bytes were
    expected (a relationship or media part existed) but absent — the
    two populations must stay disjoint (D-2a §4.3, E-3.4).
    """
    prov = ctx.prov(drawing)
    ctx.source_map[f"dropped:{prov}"] = {
        "part": ctx.part,
        "provenance": prov,
        "import_status": "dropped",
    }
    ctx.issue(
        "IMPORT_DRAWING_NO_IMAGE", "warning",
        "w:drawing carries no image (no a:blip); the shape was dropped and "
        "recorded in the source map (approximate import).",
        el=drawing,
    )


def _parse_alternate_content(ctx: _Ctx, ac: etree._Element,
                             out: list[dict[str, Any]]) -> None:
    """mc:AlternateContent: import the mc:Choice branch's drawings (the
    richest representation), falling back to the mc:Fallback VML shape.
    Exactly one branch is imported — mc semantics pick one alternative.
    """
    choice = next(
        (c for c in ac if etree.QName(c).localname == "Choice"), None,
    )
    drawings = (
        [d for d in choice.iter() if etree.QName(d).localname == "drawing"]
        if choice is not None
        else []
    )
    if drawings:
        ctx.issue(
            "IMPORT_ALTERNATE_CONTENT_RESOLVED", "info",
            "mc:AlternateContent resolved to its mc:Choice drawing content; the "
            "mc:Fallback alternative was not imported.",
            el=ac,
        )
        for drawing in drawings:
            node = _parse_drawing(ctx, drawing)
            if node is not None:
                if node.get("type") == "text_frame":
                    out.append({"__block__": node, "el": drawing})
                else:
                    out.append(node)
        return
    fallback = next(
        (c for c in ac if etree.QName(c).localname == "Fallback"), None,
    )
    pict = (
        next((p for p in fallback.iter() if etree.QName(p).localname == "pict"), None)
        if fallback is not None
        else None
    )
    if pict is not None:
        if ctx.options.text_frames:
            node = _parse_vml_pict(ctx, pict)
            if node is not None:
                what = "text frame" if node.get("type") == "text_frame" else "picture"
                if node.get("type") == "text_frame":
                    out.append({"__block__": node, "el": pict})
                else:
                    out.append(node)
                ctx.issue(
                    "IMPORT_ALTERNATE_CONTENT_RESOLVED", "info",
                    "mc:AlternateContent had no usable Choice drawing and resolved to "
                    f"its VML fallback {what}.",
                    el=ac,
                )
            return
        _drop_floating_or_report(
            ctx, ac,
            "an mc:AlternateContent VML fallback shape cannot be represented in "
            "word-core-v0",
        )
        return
    ctx.issue(
        "IMPORT_UNSUPPORTED_ELEMENT", "error",
        "run child element 'mc:AlternateContent' carries no importable drawing or "
        "VML alternative.",
        el=ac, element="AlternateContent",
    )


_MAIL_SCHEME = "mailto:"


def _hyperlink_target(ctx: _Ctx, hyperlink: etree._Element) -> dict[str, Any] | None:
    """Classify a `w:hyperlink` destination into the AST target object.

    An internal jump is `w:anchor` (a bookmark name, rendered as
    `#<name>` per RFC001-ERRATA-001 E5); an external one resolves
    through the CURRENT PART's relationships, since `rId4` in
    header1.xml.rels and document.xml.rels are different targets.
    """
    rel_id = hyperlink.get(f"{{{_R_NS}}}id")
    anchor = hyperlink.get(_w("anchor"))
    if rel_id:
        uri = ctx.rels.get(rel_id)
        if uri:
            lowered = uri.lower()
            if lowered.startswith(_MAIL_SCHEME):
                kind = "email"
            elif "://" in uri:
                kind = "external"
            else:
                # A relationship target with no scheme is a package- or
                # filesystem-relative document reference.
                kind = "file"
            return {"kind": kind, "uri": uri}
    if anchor:
        # Resolved later against `ctx.bookmarks`: the uri must be a NODE
        # id, and the bookmark may be declared further down the document.
        return {"kind": "internal_anchor", "uri": f"#{anchor}", "_bookmark": anchor}
    return None


def _record_bookmark(ctx: _Ctx, bookmark: etree._Element, node_id: str) -> None:
    """Bind a `w:bookmarkStart` name to the node it marks.

    The first binding wins: a duplicated name is an authoring defect and
    Word jumps to the first occurrence.
    """
    name = bookmark.get(_w("name"))
    if name and name not in ctx.bookmarks:
        ctx.bookmarks[name] = node_id


def _resolve_internal_anchors(ctx: _Ctx, tree: Any) -> None:
    """Rewrite every `internal_anchor` uri to `#<node_id>`.

    Walking the finished tree rather than tracking the parent list at
    parse time is deliberate: a paragraph that splits on an inline break
    REBINDS its children list, so a reference captured earlier can point
    at a list the AST no longer holds.

    A link to a bookmark the document never declares cannot be a
    hyperlink node — the canonicalizer refuses a dangling anchor — so it
    is unwrapped into its children, keeping the visible text.
    """
    if isinstance(tree, dict):
        for key, value in tree.items():
            if isinstance(value, list):
                tree[key] = _resolve_in_list(ctx, value)
            else:
                _resolve_internal_anchors(ctx, value)
    elif isinstance(tree, list):  # pragma: no cover - lists arrive via the dict branch
        for item in tree:
            _resolve_internal_anchors(ctx, item)


def _resolve_in_list(ctx: _Ctx, items: list[Any]) -> list[Any]:
    resolved: list[Any] = []
    for item in items:
        _resolve_internal_anchors(ctx, item)
        bookmark = (
            item.get("target", {}).pop("_bookmark", None)
            if isinstance(item, dict) and item.get("type") == "hyperlink"
            else None
        )
        if bookmark is None:
            resolved.append(item)
            continue
        target_id = ctx.bookmarks.get(bookmark)
        if target_id is not None:
            item["target"]["uri"] = f"#{target_id}"
            resolved.append(item)
            continue
        ctx.issue(
            "IMPORT_INTERNAL_ANCHOR_UNRESOLVED", "warning",
            f"w:hyperlink jumps to bookmark {bookmark!r}, which the document never "
            "declares; the link was unwrapped into plain content so its text "
            "survives (approximate import).",
            node_id=item["id"], bookmark=bookmark,
        )
        resolved.extend(item["children"])
    return resolved


def _parse_hyperlink(ctx: _Ctx, hyperlink: etree._Element,
                     style_rprs: list[etree._Element | None],
                     para_node: dict[str, Any]) -> None:
    """`w:hyperlink` -> a `hyperlink` node holding its inline children.

    A hyperlink whose destination cannot be resolved, or that has no
    inline children (the schema requires `minItems: 1`), cannot be a
    hyperlink node — it falls back to the word-core-v0 unwrap, so the
    visible text is never lost.
    """
    target = _hyperlink_target(ctx, hyperlink)
    if target is None:
        ctx.issue(
            "IMPORT_HYPERLINK_UNRESOLVED", "warning",
            "w:hyperlink resolves to neither a relationship target nor a w:anchor; "
            "its runs were imported as plain runs (approximate import).",
            el=hyperlink, node_id=para_node["id"],
        )
        _flatten_hyperlink(ctx, hyperlink, style_rprs, para_node)
        return

    node_id = ctx.new_id()
    ctx.map_node(node_id, hyperlink)
    children: list[dict[str, Any]] = []
    for child in _iter_content(ctx, hyperlink):
        tag = etree.QName(child).localname
        if tag == "r":
            _parse_run(ctx, child, style_rprs, children)
        elif tag == "hyperlink":
            # A nested w:hyperlink cannot nest in the AST (a hyperlink
            # is not a hyperlink_child): its runs join the outer link.
            ctx.issue(
                "IMPORT_NESTED_HYPERLINK_FLATTENED", "warning",
                "a w:hyperlink nested inside another is imported as part of the outer "
                "link; the inner target is dropped (approximate import).",
                el=child, node_id=node_id,
            )
            ctx.degrade(node_id, "approximate")
            _collect_hyperlink_runs(ctx, child, style_rprs, children, node_id)
        elif tag in _IGNORED_PARAGRAPH_CHILDREN:
            ctx.issue(
                "IMPORT_IGNORED_ELEMENT", "info",
                f"non-content element 'w:{tag}' ignored.", el=child, element=tag,
            )
        else:
            ctx.issue(
                "IMPORT_UNSUPPORTED_ELEMENT", "error",
                f"hyperlink child element 'w:{tag}' is not supported by the "
                "word-visual-v1 importer.",
                el=child, element=tag,
            )
    kept = [c for c in children if c.get("type") in _HYPERLINK_CHILD_TYPES]
    kept_ids = {id(c) for c in kept}
    lifted = [c for c in children if id(c) not in kept_ids]
    if lifted:
        # floating_image / note_ref / an inline break sentinel are not
        # `hyperlink_child`s; they are lifted out beside the link so no
        # visible content is lost.
        ctx.issue(
            "IMPORT_HYPERLINK_CHILD_LIFTED", "warning",
            f"{len(lifted)} child(ren) of the hyperlink are outside the hyperlink "
            "content set and were imported as siblings of the link (approximate "
            "import).",
            el=hyperlink, node_id=node_id, count=len(lifted),
        )
        ctx.degrade(node_id, "approximate")
    if not kept:
        ctx.issue(
            "IMPORT_HYPERLINK_EMPTY", "warning",
            "w:hyperlink has no importable inline content; the link was not imported "
            "as a hyperlink node (approximate import).",
            el=hyperlink, node_id=para_node["id"],
        )
        ctx.degrade(para_node["id"], "approximate")
        para_node["children"].extend(lifted)
        return
    para_node["children"].append({
        "id": node_id, "type": "hyperlink", "target": target, "children": kept,
    })
    para_node["children"].extend(lifted)


def _collect_hyperlink_runs(ctx: _Ctx, hyperlink: etree._Element,
                            style_rprs: list[etree._Element | None],
                            out: list[dict[str, Any]], node_id: str) -> None:
    for child in _iter_content(ctx, hyperlink):
        tag = etree.QName(child).localname
        if tag == "r":
            _parse_run(ctx, child, style_rprs, out)
        elif tag == "hyperlink":
            _collect_hyperlink_runs(ctx, child, style_rprs, out, node_id)


def _flatten_hyperlink(ctx: _Ctx, hyperlink: etree._Element,
                       style_rprs: list[etree._Element | None],
                       para_node: dict[str, Any]) -> None:
    """word-core-v0 has no hyperlink node, but the link text is visible
    content: import the `w:hyperlink` children as plain runs (unwrap),
    with a typed `IMPORT_HYPERLINK_FLATTENED` issue. The target URI is
    recorded in the source map only — never in the AST.
    """
    prov = ctx.prov(hyperlink)
    rel_id = hyperlink.get(f"{{{_R_NS}}}id")
    anchor = hyperlink.get(_w("anchor"))
    target = ctx.rels.get(rel_id) if rel_id else (f"#{anchor}" if anchor else None)
    ctx.source_map[f"hyperlink:{prov}"] = {
        "part": ctx.part,
        "provenance": prov,
        "import_status": "approximate",
        "target": target,
    }
    ctx.issue(
        "IMPORT_HYPERLINK_FLATTENED", "warning",
        "w:hyperlink is outside word-core-v0; its runs were imported as plain runs "
        "and the target URI recorded in the source map only (approximate import).",
        el=hyperlink, node_id=para_node["id"],
    )
    ctx.degrade(para_node["id"], "approximate")
    for child in _iter_content(ctx, hyperlink):
        tag = etree.QName(child).localname
        if tag == "r":
            _parse_run(ctx, child, style_rprs, para_node["children"])
        elif tag == "hyperlink":
            _flatten_hyperlink(ctx, child, style_rprs, para_node)
        elif tag in _IGNORED_PARAGRAPH_CHILDREN:
            ctx.issue(
                "IMPORT_IGNORED_ELEMENT", "info",
                f"non-content element 'w:{tag}' ignored.", el=child, element=tag,
            )
        else:
            ctx.issue(
                "IMPORT_UNSUPPORTED_ELEMENT", "error",
                f"hyperlink child element 'w:{tag}' is not supported by the "
                "word-core-v0 importer.",
                el=child, element=tag,
            )


def _parse_drawing(ctx: _Ctx, drawing: etree._Element) -> dict[str, Any] | None:
    """A `w:drawing` -> `inline_image`, or -> `floating_image` when the
    drawing is anchored and the target profile is `word-visual-v1`
    (18.3 % of documents carry an anchored drawing in the body)."""
    inline = drawing.find(f"{{{_WP_NS}}}inline")
    anchor_el: etree._Element | None = None
    if inline is None:
        anchor_el = drawing.find(f"{{{_WP_NS}}}anchor")
        if anchor_el is None or not ctx.options.visual_v1:
            _drop_floating_or_report(
                ctx, drawing,
                "an anchored (floating) drawing cannot be represented in word-core-v0 "
                "(only wp:inline)",
            )
            return None
        inline = anchor_el
    txbx_contents = _top_level_txbx_contents(inline)
    if txbx_contents:
        frame = _parse_text_frame(
            ctx, drawing, txbx_contents,
            anchor_el=anchor_el,
        )
        if frame is not None:
            return frame
    a_ext = inline.find(f".//{{{_A_NS}}}xfrm/{{{_A_NS}}}ext")
    extent = a_ext if a_ext is not None else inline.find(f"{{{_WP_NS}}}extent")
    cx = _int_attr(ctx, extent, "cx", default=0) if extent is not None else 0
    cy = _int_attr(ctx, extent, "cy", default=0) if extent is not None else 0
    chart = inline.find(
        ".//{http://schemas.openxmlformats.org/drawingml/2006/chart}chart"
    )
    blip = inline.find(f".//{{{_A_NS}}}blip")
    embed = blip.get(f"{{{_R_NS}}}embed") if blip is not None else None
    chart_rel = chart.get(f"{{{_R_NS}}}id") if chart is not None else None
    relation_id = chart_rel or embed
    if relation_id is None:
        # No blip at all: either a text box (already reported above) or
        # a plain DrawingML shape with no image (D-2a §4.3 split).
        if not txbx_contents:
            _report_drawing_no_image(ctx, drawing)
        return None
    target = ctx.rels.get(relation_id)
    if target is None:
        _drop_missing_image(
            ctx, drawing, "drawing has no resolvable graphic relationship"
        )
        return None
    part_name = _resolve_target(ctx, target)
    data = ctx.parts.get(part_name)
    if data is None:
        _drop_missing_image(
            ctx, drawing, f"graphic part '{part_name}' is missing from the package"
        )
        return None
    extension = part_name.rsplit(".", 1)[-1].lower()
    mime = None if chart is not None else _sniff_mime(data)
    raster_kind = "chart" if chart is not None else (
        "media" if extension in {"emf", "wmf"} else None
    )
    raster = None
    if raster_kind is not None:
        raster = _rasterize_or_drop(
            ctx, drawing, data, part_name, extension, raster_kind, cx or 0, cy or 0
        )
        if raster is None:
            return None
        data, mime = raster.png, "image/png"
    if mime is None:
        mime = _EXT_MIME.get(extension)
    elif _EXT_MIME.get(extension) != mime:
        ctx.issue(
            "IMPORT_RESOURCE_TYPE_FROM_CONTENT", "info",
            f"image part '{part_name}' is {mime} by content; its extension "
            f"'{extension}' says otherwise and the content was trusted.",
            el=drawing, extension=extension, mime_type=mime,
        )
    if mime is None:
        if not ctx.options.visual_v1:
            ctx.issue(
                "IMPORT_UNSUPPORTED_RESOURCE_TYPE", "error",
                f"image extension '{extension}' is outside the word-core-v0 whitelist "
                "(png/jpeg).",
                el=drawing, extension=extension,
            )
            return None
        # Under word-visual-v1 the same finding drops the IMAGE, not the
        # DOCUMENT. Header logos in emf/wmf/tiff/gif are the common case
        # (80 occurrences across a 1500-document sample, essentially all
        # of them in a story), and rejecting the whole document over one
        # unconvertible logo throws away all of its text. The loss is
        # typed and source-mapped, which is what RFC003 §45 requires;
        # it does not require it to be fatal.
        prov = ctx.prov(drawing)
        ctx.source_map[f"dropped:{prov}"] = {
            "part": ctx.part,
            "provenance": prov,
            "import_status": "dropped",
        }
        ctx.issue(
            "IMPORT_UNSUPPORTED_RESOURCE_TYPE", "warning",
            f"image extension '{extension}' is outside the resource whitelist "
            "(png/jpeg); the image was dropped and recorded in the source map, and "
            "the rest of the document imported (approximate import).",
            el=drawing, extension=extension,
        )
        return None

    node_id = ctx.new_id()
    ctx.map_node(node_id, drawing)
    resource_id = _register_resource(ctx, data, mime, node_id)
    if raster is not None:
        resource = ctx.resources_by_sha[hashlib.sha256(data).hexdigest()]
        ctx.resource_provenance[resource["path"]] = part_name

    # Size comes from `a:ext` (the picture's own, UNROTATED extent) and
    # only falls back to `wp:extent`. Word writes `wp:extent` as the
    # ROTATED bounding box, so reading it into `size` would grow a
    # rotated image a little on every round-trip — silently, and
    # cumulatively. `a:ext` is the measurement the AST's single `size`
    # actually means.
    rotation_mdeg = _drawing_rotation(ctx, inline, node_id)
    doc_pr = inline.find(f"{{{_WP_NS}}}docPr")
    alt_text = doc_pr.get("descr") if doc_pr is not None else None
    src_rect = inline.find(f".//{{{_A_NS}}}srcRect")
    crop = {"left_ppm": 0, "top_ppm": 0, "right_ppm": 0, "bottom_ppm": 0}
    if src_rect is not None:
        for attr, key in (("l", "left_ppm"), ("t", "top_ppm"),
                          ("r", "right_ppm"), ("b", "bottom_ppm")):
            crop[key] = _int_attr(ctx, src_rect, attr, default=0, node_id=node_id) * 10
    node: dict[str, Any] = {
        "id": node_id,
        "type": "floating_image" if anchor_el is not None else "inline_image",
        "resource_ref": resource_id,
        "size": {"width_mpt": _emu_to_mpt(cx), "height_mpt": _emu_to_mpt(cy)},
        "crop": crop,
        "rotation_mdeg": rotation_mdeg,
        "alt_text": alt_text,
    }
    if anchor_el is not None:
        node["anchor"] = visual_v1.parse_anchor(ctx, anchor_el, node_id)
    return node


_THEME_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme"


def _rel_target_by_type(ctx: _Ctx, owner_part: str, rel_type: str) -> str | None:
    """The Target of `owner_part`'s relationship of type `rel_type`, or
    None. `ctx.rels_by_part` drops the Type attribute (it is keyed by
    Id only, since ids are what document content actually references),
    so this re-reads the raw .rels bytes directly for the one caller
    that needs Type: theme resolution.
    """
    rels_part = _rels_part_for(owner_part)
    data = ctx.parts.get(rels_part)
    if data is None:
        return None
    try:
        root = etree.fromstring(data)
    except etree.XMLSyntaxError:
        return None
    for rel in root.findall(f"{{{_REL_NS}}}Relationship"):
        if rel.get("Type") == rel_type:
            target = rel.get("Target")
            if target:
                return posixpath.normpath(
                    posixpath.join(posixpath.dirname(owner_part), target.lstrip("/"))
                )
    return None


def _document_theme(ctx: _Ctx) -> bytes | None:
    """The document theme part, needed so a chart's `<a:schemeClr>`
    resolves to real colours instead of rendering black. Looked up from
    `word/document.xml.rels` (RFC003's minimal package only ever hosts
    the graphic inside `word/document.xml`, never a header/footer/note
    story, so that is always the right owner part to ask).

    A chart can also carry its own `themeOverride` relationship, which
    is deliberately NOT resolved here — verified against a real corpus
    document that the pinned LibreOffice (26.2.5.2) ignores it and
    always falls back to this document theme, so wiring it would be
    untested dead code.
    """
    theme_part = _rel_target_by_type(ctx, "word/document.xml", _THEME_REL_TYPE)
    return ctx.parts.get(theme_part) if theme_part else None


def _rasterize_or_drop(
    ctx: _Ctx, drawing: etree._Element, source: bytes, source_part: str,
    extension: str, kind: str, width_emu: int, height_emu: int,
) -> Any:
    """Run one bounded conversion, or record the common loud drop path."""
    if ctx.rasterization_count >= _RASTERIZATION_BUDGET:
        _drop_rasterization_failure(ctx, drawing, source_part, "BUDGET_EXCEEDED", "")
        return None
    ctx.rasterization_count += 1
    theme = _document_theme(ctx) if kind == "chart" else None
    try:
        result = rasterize_graphic(
            source, source_extension=extension, kind=kind,
            width_emu=width_emu, height_emu=height_emu, theme=theme,
        )
    except (RasterizationError, RasterizerUnavailableError) as exc:
        reason = exc.reason if isinstance(exc, RasterizationError) else "UNAVAILABLE"
        _drop_rasterization_failure(ctx, drawing, source_part, reason, str(exc))
        return None
    source_mime = (
        "application/vnd.openxmlformats-officedocument.drawingml.chart+xml"
        if kind == "chart" else f"image/x-{extension}"
    )
    ctx.issue(
        "IMPORT_GRAPHIC_RASTERIZED", "info",
        f"graphic part '{source_part}' was rasterized deterministically to PNG.",
        el=drawing, source_part=source_part, source_mime=source_mime,
        target_dpi=TARGET_DPI, pixel_width=result.pixel_width,
        pixel_height=result.pixel_height, raster_pin_hash=result.raster_pin_hash,
    )
    return result


def _drop_rasterization_failure(
    ctx: _Ctx, drawing: etree._Element, source_part: str, reason: str, detail: str,
) -> None:
    prov = ctx.prov(drawing)
    ctx.source_map[f"dropped:{prov}"] = {
        "part": ctx.part, "provenance": prov, "import_status": "dropped",
    }
    ctx.issue(
        "IMPORT_GRAPHIC_RASTERIZATION_FAILED", "warning",
        f"graphic part '{source_part}' could not be rasterized ({reason}); it was "
        "dropped and recorded in the source map, and the rest of the document imported.",
        el=drawing, source_part=source_part, reason=reason, detail=detail,
    )


_VML_ROTATION_RE = re.compile(r"rotation\s*:\s*(-?[\d.]+)")


def _text_frame_is_rotated(
    anchor_el: etree._Element | None, vml_shape: etree._Element | None
) -> bool:
    """A rotated text box has no home in the ratified `text_frame` anchor
    (RFC001-ERRATA-004 E1 reuses `floating_image`'s anchor verbatim, and
    that contract carries no angle). Importing it unrotated would put the
    right words in the wrong shape on the page — worse than a typed drop,
    per the owner's ruling that missing ink beats wrong ink.

    DrawingML rotation lives in `wps:spPr/a:xfrm/@rot` (1/60000 degree);
    VML rotation is the `rotation:<deg>` declaration in `v:shape/@style`
    (plain degrees, the same dialect `visual_v1._vml_style` parses).
    """
    if anchor_el is not None:
        xfrm = anchor_el.find(f".//{{{_A_NS}}}xfrm")
        if xfrm is not None:
            rot = xfrm.get("rot")
            if rot is not None:
                try:
                    return int(rot) != 0
                except ValueError:
                    return False
        return False
    if vml_shape is not None:
        match = _VML_ROTATION_RE.search(vml_shape.get("style", ""))
        if match is not None:
            try:
                return float(match.group(1)) != 0
            except ValueError:
                return False
    return False


def _drawing_rotation(ctx: _Ctx, inline: etree._Element, node_id: str) -> int:
    """`a:xfrm/@rot` -> `rotation_mdeg`.

    DrawingML measures rotation in 1/60000 of a degree (CT_Transform2D,
    the exact inverse of the renderer's `mdeg_to_ooxml_angle`). The
    attribute was previously a hardcoded 0, which lost every rotation
    with no issue, no degradation and no source-map entry — the silent
    kind of loss RFC003 §18 item 14 forbids.
    """
    xfrm = inline.find(f".//{{{_A_NS}}}xfrm")
    if xfrm is None or xfrm.get("rot") is None:
        return 0
    rot = _int_attr(ctx, xfrm, "rot", default=0, node_id=node_id) or 0
    return round_half_away_from_zero(rot, 60)


def _drop_missing_image(ctx: _Ctx, drawing: etree._Element, what: str) -> None:
    """A missing relationship/media target is a source defect: the only
    faithful option is to drop the image with a typed warning and a
    source-map record (RFC003 §45) — there are no bytes to import.
    """
    prov = ctx.prov(drawing)
    ctx.source_map[f"dropped:{prov}"] = {
        "part": ctx.part,
        "provenance": prov,
        "import_status": "dropped",
    }
    ctx.issue(
        "IMPORT_IMAGE_RESOURCE_MISSING", "warning",
        f"{what}; the image was dropped and recorded in the source map "
        "(approximate import).",
        el=drawing,
    )


def _resolve_target(ctx: _Ctx, target: str) -> str:
    """Resolve a relationship target against the directory of the part
    that declared it — `media/x.png` in word/_rels/header1.xml.rels and
    in word/_rels/document.xml.rels both mean word/media/x.png, but the
    base must come from the part, never from a hardcoded "word".
    """
    return posixpath.normpath(
        posixpath.join(posixpath.dirname(ctx.part), target.lstrip("/"))
    )


def _register_resource(ctx: _Ctx, data: bytes, mime: str, node_id: str) -> str:
    sha = hashlib.sha256(data).hexdigest()
    existing = ctx.resources_by_sha.get(sha)
    if existing is not None:
        return existing["id"]
    dims = _image_pixel_size(data, mime)
    if dims is None:
        dims = (1, 1)
        ctx.issue(
            "IMPORT_RESOURCE_PIXEL_DIMS_UNKNOWN", "warning",
            "image pixel dimensions could not be parsed from the media bytes; placeholder "
            "1x1 recorded (normalized import).",
            node_id=node_id,
        )
        ctx.degrade(node_id, "normalized")
    extension = _MIME_CANONICAL_EXT[mime]
    path = f"media/{sha[:16]}.{extension}"
    resource = {
        "id": f"res_{sha[:16]}",
        "kind": "image",
        "mime_type": mime,
        "sha256": sha,
        "path": path,
        "pixel_width": dims[0],
        "pixel_height": dims[1],
    }
    ctx.resources_by_sha[sha] = resource
    ctx.resource_bytes[path] = data
    return resource["id"]


def _image_pixel_size(data: bytes, mime: str) -> tuple[int, int] | None:
    if mime == "image/gif":
        # Logical screen descriptor: width, height as little-endian
        # uint16 immediately after the 6-byte signature.
        if len(data) >= 10:
            width, height = struct.unpack("<HH", data[6:10])
            if width and height:
                return width, height
        return None
    if mime == "image/bmp":
        # BITMAPINFOHEADER at offset 14; height is signed and negative
        # for a top-down bitmap, which is a storage order, not a size.
        if len(data) >= 26:
            width, height = struct.unpack("<ii", data[18:26])
            if width and height:
                return abs(width), abs(height)
        return None
    if mime == "image/tiff":
        return _tiff_pixel_size(data)
    if mime == "image/png":
        if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", data[16:24])
            if width and height:
                return width, height
        return None
    # image/jpeg: scan for a start-of-frame marker.
    offset = 2
    while offset + 9 < len(data):
        if data[offset] != 0xFF:
            return None
        marker = data[offset + 1]
        length = struct.unpack(">H", data[offset + 2:offset + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            height, width = struct.unpack(">HH", data[offset + 5:offset + 9])
            if width and height:
                return width, height
            return None
        offset += 2 + length
    return None


def _tiff_pixel_size(data: bytes) -> tuple[int, int] | None:
    """ImageWidth (tag 256) and ImageLength (tag 257) of the first IFD.

    TIFF carries its dimensions in a tag directory rather than a fixed
    header, and the byte order is declared by the file itself, so this
    walks the directory instead of unpacking an offset. Only SHORT and
    LONG tag types occur for these two tags.
    """
    if len(data) < 8:
        return None
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        return None
    try:
        offset = struct.unpack(f"{endian}I", data[4:8])[0]
        if offset + 2 > len(data):
            return None
        count = struct.unpack(f"{endian}H", data[offset:offset + 2])[0]
        dimensions: dict[int, int] = {}
        for index in range(count):
            entry = offset + 2 + index * 12
            if entry + 12 > len(data):
                break
            tag, kind = struct.unpack(f"{endian}HH", data[entry:entry + 4])
            if tag not in (256, 257):
                continue
            if kind == 3:  # SHORT
                dimensions[tag] = struct.unpack(
                    f"{endian}H", data[entry + 8:entry + 10])[0]
            elif kind == 4:  # LONG
                dimensions[tag] = struct.unpack(
                    f"{endian}I", data[entry + 8:entry + 12])[0]
        width, height = dimensions.get(256), dimensions.get(257)
        if width and height:
            return width, height
    except struct.error:
        return None
    return None


# --- paragraph / block parsing -----------------------------------------


def _iter_content(ctx: _Ctx, parent: etree._Element):
    """Iterate `parent`'s children with `w:sdt` content controls and
    `w:smartTag` wrappers unwrapped in place: the wrapper chrome
    (sdtPr/sdtEndPr, smartTagPr) is dropped and the wrapped children are
    yielded where the wrapper stood — at block, row, and inline level
    alike (typed info, RFC003 §45). Neither wrapper has a visual effect
    and RFC001 §42.1 excludes neither.
    """
    for child in parent:
        tag = etree.QName(child).localname
        if tag == "sdt":
            ctx.issue(
                "IMPORT_SDT_UNWRAPPED", "info",
                "w:sdt content control unwrapped; its w:sdtContent children were "
                "imported in place.",
                el=child,
            )
            content = child.find(_w("sdtContent"))
            if content is not None:
                yield from _iter_content(ctx, content)
        elif tag == "smartTag":
            ctx.issue(
                "IMPORT_SMART_TAG_UNWRAPPED", "info",
                "w:smartTag unwrapped; its children were imported in place "
                "(a smart tag carries classification metadata only — no "
                "formatting and no visual effect).",
                el=child, element=tag,
            )
            for inner in _iter_content(ctx, child):
                if etree.QName(inner).localname != "smartTagPr":
                    yield inner
        else:
            yield child


def _is_technical_break(p: etree._Element) -> str | None:
    """The renderer's technical break paragraph: no pPr, one run whose
    only child is `w:br` with type page/column.
    """
    children = list(p)
    if len(children) != 1 or etree.QName(children[0]).localname != "r":
        return None
    if p.find(_w("pPr")) is not None:
        return None
    run_children = list(children[0])
    if len(run_children) != 1 or etree.QName(run_children[0]).localname != "br":
        return None
    br_type = run_children[0].get(_w("type"))
    return br_type if br_type in ("page", "column") else None


def _has_content(p: etree._Element) -> bool:
    return any(etree.QName(child).localname != "pPr" for child in p)


def _parse_p(ctx: _Ctx, p: etree._Element, *, in_cell: bool = False) -> list[dict[str, Any]]:
    """Parse one `w:p` into raw block items for the assembler.

    Usually one item; an inline page/column break in a mixed-content
    run splits the paragraph into paragraph / break / paragraph items
    (`IMPORT_INLINE_BREAK_SPLIT`, approximate).
    """
    p_pr = p.find(_w("pPr"))
    sect_pr = p_pr.find(_w("sectPr")) if p_pr is not None else None
    if sect_pr is not None and in_cell:
        # A sectPr on a paragraph inside a table cell is invalid
        # OOXML-in-practice (sections cannot end inside a cell): the
        # paragraph is imported as a normal paragraph and the sectPr
        # ignored — recorded in the source map, typed, never a crash.
        prov = ctx.prov(sect_pr)
        ctx.source_map[f"sectpr:{prov}"] = {
            "part": ctx.part,
            "provenance": prov,
            "import_status": "dropped",
        }
        ctx.issue(
            "IMPORT_SECTPR_IN_CELL_IGNORED", "warning",
            "w:sectPr on a paragraph inside a table cell cannot end a section "
            "there; the paragraph was imported as a normal paragraph and the "
            "sectPr ignored (recorded in the source map; approximate import).",
            el=sect_pr,
        )
        sect_pr = None

    br_type = _is_technical_break(p)
    if br_type is not None:
        node_id = ctx.new_id()
        ctx.map_node(node_id, p)
        kind = "page_break" if br_type == "page" else "column_break"
        return [{"kind": "block", "node": {"id": node_id, "type": kind}}]

    # Renderer-synthesized section boundary paragraph: carries sectPr,
    # no content, and none of the pPr the renderer always writes for a
    # real paragraph (no w:jc). Import has no build context, so this is
    # an assumption — reported, never silent (RFC002/RFC003 lesson).
    if sect_pr is not None and not _has_content(p) and (
        p_pr is None or p_pr.find(_w("jc")) is None
    ):
        ctx.issue(
            "IMPORT_BOUNDARY_PARAGRAPH_ASSUMED", "warning",
            "paragraph carrying w:sectPr with no content and no explicit paragraph "
            "properties is assumed to be a renderer-synthesized section boundary and was "
            "not imported as an AST paragraph (approximate import).",
            el=p,
        )
        ctx.source_map[f"technical:{ctx.prov(p)}"] = {
            "part": ctx.part,
            "provenance": ctx.prov(p),
            "import_status": "approximate",
        }
        return [{"kind": "section_end", "sect_pr": sect_pr, "el": p}]

    node = _paragraph_node(ctx, p, p_pr)
    style_rpr = node.pop("_style_rpr")
    ppr_chain = node.pop("_ppr_chain")
    num_id, ilvl = _numbering_of(ctx, node, p_pr, ppr_chain)
    number_start = _source_number_start(ctx, p, num_id, ilvl)
    for child in _iter_content(ctx, p):
        tag = etree.QName(child).localname
        if tag == "pPr":
            continue
        if tag == "r":
            _parse_run(ctx, child, style_rpr, node["children"])
        elif tag == "hyperlink":
            if ctx.options.visual_v1:
                _parse_hyperlink(ctx, child, style_rpr, node)
            else:
                _flatten_hyperlink(ctx, child, style_rpr, node)
        elif tag == "fldSimple":
            _flatten_fld_simple(ctx, child, style_rpr, node["children"])
        elif tag == "br" and child.get(_w("type")) is None:
            # `w:br` belongs to EG_RunInnerContent (wml.xsd:1743-1745),
            # i.e. under `w:r` — so the PARSER is not wrong here, the
            # INPUT is invalid. Word tolerates it, PHPWord produced it in
            # 60 of 60 sampled cases, and an attribute-less `w:br` is
            # unambiguously a line break. Adopted with a typed warning
            # rather than rejecting 86 documents over a producer bug (M1).
            break_id = ctx.new_id()
            ctx.map_node(break_id, child, status="approximate")
            node["children"].append({"id": break_id, "type": "line_break"})
            ctx.issue(
                "IMPORT_STRAY_BREAK_ADOPTED", "warning",
                "w:br appears directly under w:p, where OOXML does not allow it "
                "(EG_RunInnerContent requires w:r); adopted as a line break, which "
                "is how Word renders it (approximate import).",
                el=child, node_id=break_id,
            )
        elif tag in _IGNORED_PARAGRAPH_CHILDREN:
            if tag == "bookmarkStart":
                _record_bookmark(ctx, child, node["id"])
            ctx.issue(
                "IMPORT_IGNORED_ELEMENT", "info",
                f"non-content element 'w:{tag}' ignored.", el=child, element=tag,
            )
        else:
            ctx.issue(
                "IMPORT_UNSUPPORTED_ELEMENT", "error",
                f"paragraph child element 'w:{tag}' is not supported by the word-core-v0 "
                "importer.",
                el=child, element=tag,
            )

    items = _split_paragraph_on_breaks(ctx, p, node, num_id, ilvl)
    for item in items:
        if item.get("num_id") is not None and number_start is not None:
            item["number_start"] = number_start
    if sect_pr is not None:
        items[-1]["sect_pr"] = sect_pr
        items[-1]["el"] = p
    return items


def _source_number_start(ctx: _Ctx, p: etree._Element, num_id: int | None,
                         ilvl: int) -> int | None:
    """Effective ordinal for a supported level-zero source marker.

    Nested levels keep the existing explicit approximation; they cannot
    advance the level-zero counter. Separate stories do not consume body
    ordinals, including text frames recursively parsed from a body paragraph.
    """
    definition = ctx.numdefs.get(num_id) if num_id is not None else None
    if ilvl != 0 or definition is None or definition.list_kind != "ordered":
        return None
    paragraph_key = (ctx.part, ctx.prov(p))
    if paragraph_key in ctx.numbering_ordinals:
        return ctx.numbering_ordinals[paragraph_key]
    story = next((ancestor for ancestor in p.iterancestors()
                  if etree.QName(ancestor).localname in
                  {"body", "hdr", "ftr", "footnote", "endnote", "txbxContent"}), None)
    if story is None:
        return None
    key = (ctx.part, ctx.prov(story), num_id)
    ordinal = ctx.numbering_next.get(key, definition.start)
    if ordinal is None:
        return None
    ctx.numbering_ordinals[paragraph_key] = ordinal
    ctx.numbering_next[key] = ordinal + 1
    return ordinal


def _cancels_numbering(num_pr: etree._Element) -> bool:
    num_id_el = num_pr.find(_w("numId"))
    return num_id_el is not None and num_id_el.get(_w("val")) == "0"


def _numbering_of(ctx: _Ctx, node: dict[str, Any], p_pr: etree._Element | None,
                  chain: list[etree._Element | None]) -> tuple[int | None, int]:
    """The paragraph's effective `w:numPr`, resolved leaf-first.

    Direct formatting wins over the style chain, which is what makes
    `w:numId w:val="0"` able to cancel numbering a style imposes — the
    two halves of the same OOXML mechanism, and neither was read before
    (D6 and D2).
    """
    num_pr = _chain_find(chain, "numPr")
    if num_pr is None:
        return None, 0
    if _cancels_numbering(num_pr):
        # OOXML: `w:numId w:val="0"` is not a reference at all — it
        # CANCELS numbering the paragraph would otherwise inherit from
        # its style, so the paragraph is simply un-numbered. Reading it
        # as a dangling reference produced a false warning, a false
        # label-grade degradation, and an unconditional list close that
        # split whatever list the paragraph sat next to. Measured over
        # 300 sampled documents: 276 are this and NONE is a genuinely
        # dangling reference (D2).
        return None, 0
    if p_pr is None or p_pr.find(_w("numPr")) is not num_pr:
        # Numbering the paragraph inherits from its style. Typed so the
        # recovery is measurable, at `info` because the import is exact.
        ctx.issue(
            "IMPORT_NUMBERING_FROM_STYLE", "info",
            "paragraph numbering resolved through its w:pStyle chain rather than "
            "direct formatting.",
            el=num_pr, node_id=node["id"],
        )
    num_id_el = num_pr.find(_w("numId"))
    ilvl_el = num_pr.find(_w("ilvl"))
    num_id = (
        _int_attr(ctx, num_id_el, _w("val"), node_id=node["id"], required=True)
        if num_id_el is not None
        else None
    )
    ilvl = (
        _int_attr(ctx, ilvl_el, _w("val"), default=0, node_id=node["id"])
        if ilvl_el is not None
        else 0
    )
    return num_id, ilvl


def _split_paragraph_on_breaks(ctx: _Ctx, p: etree._Element, node: dict[str, Any],
                               num_id: int | None, ilvl: int) -> list[dict[str, Any]]:
    """Lift block text frames out of their OOXML host paragraph, then split breaks."""
    if not any("__block__" in child for child in node["children"]):
        return _split_paragraph_breaks_only(ctx, p, node, num_id, ilvl)

    original_children = node["children"]
    segments: list[list[dict[str, Any]]] = [[]]
    frames: list[dict[str, Any]] = []
    for child in original_children:
        if "__block__" in child:
            frames.append(child["__block__"])
            segments.append([])
        else:
            segments[-1].append(child)

    has_flow_content = any(segments)
    if has_flow_content:
        ctx.issue(
            "IMPORT_TEXT_FRAME_PARAGRAPH_SPLIT", "warning",
            "a text frame shares its OOXML host paragraph with flow content; the flow "
            "content was split around the anchored block so text and placement survive "
            "(approximate import).",
            el=p, node_id=node["id"], frame_count=len(frames),
        )
        ctx.degrade(node["id"], "approximate")

    items: list[dict[str, Any]] = []
    template = {key: value for key, value in node.items() if key not in ("children", "id")}
    for index, segment in enumerate(segments):
        # A numbered empty first segment is retained: it opens the list item
        # that owns a following text frame. Renderer-generated anchor-only
        # host paragraphs are otherwise technical and have no AST paragraph.
        if segment or (index == 0 and num_id is not None):
            paragraph = node if index == 0 else copy.deepcopy(template)
            if index:
                paragraph["id"] = ctx.new_id()
                ctx.map_node(paragraph["id"], p, status="approximate")
            paragraph["children"] = segment
            items.extend(_split_paragraph_breaks_only(
                ctx, p, paragraph, num_id if index == 0 else None, ilvl if index == 0 else 0,
            ))
        if index < len(frames):
            items.append({"kind": "block", "node": frames[index]})

    if not has_flow_content and num_id is None:
        source = ctx.source_map.pop(node["id"], None)
        if source is not None:
            ctx.source_map[f"technical:{source['provenance']}"] = {
                **source, "import_status": "technical",
            }
    return items


def _split_paragraph_breaks_only(ctx: _Ctx, p: etree._Element, node: dict[str, Any],
                                 num_id: int | None, ilvl: int) -> list[dict[str, Any]]:
    """Resolve inline page/column break sentinels left by `_parse_run`.

    The renderer's convention for a block break is its own technical
    paragraph, so a mid-paragraph `w:br w:type=page/column` is imported
    by splitting the paragraph around a block break node, cloning the
    paragraph properties onto the continuation (position preserved;
    approximate — spacing before/after is duplicated across the split).
    """
    # An inline break inside a terminated field's instruction region is
    # field code (marked _field_drop by _parse_field_control) — dropped
    # here, before splitting; the drop was recorded on the field frame.
    node["children"] = [
        c for c in node["children"]
        if not ("__break__" in c and c.get("_field_drop"))
    ]
    if not any("__break__" in c for c in node["children"]):
        return [{"kind": "paragraph", "node": node, "num_id": num_id, "ilvl": ilvl}]

    segments: list[list[dict[str, Any]]] = [[]]
    break_els: list[tuple[str, etree._Element]] = []
    for child in node["children"]:
        if "__break__" in child:
            break_els.append((child["__break__"], child["el"]))
            segments.append([])
        else:
            segments[-1].append(child)
    node["children"] = segments[0]
    ctx.issue(
        "IMPORT_INLINE_BREAK_SPLIT", "warning",
        f"{len(break_els)} inline page/column break(s) inside the paragraph; the "
        "paragraph was split around block break node(s), cloning its properties "
        "onto the continuation(s) (approximate import).",
        el=p, node_id=node["id"], break_count=len(break_els),
    )
    ctx.degrade(node["id"], "approximate")
    items: list[dict[str, Any]] = [
        {"kind": "paragraph", "node": node, "num_id": num_id, "ilvl": ilvl},
    ]
    template = {k: v for k, v in node.items() if k not in ("children", "id")}
    for (br_type, br_el), segment in zip(break_els, segments[1:], strict=True):
        break_id = ctx.new_id()
        ctx.map_node(break_id, br_el)
        items.append({
            "kind": "block",
            "node": {
                "id": break_id,
                "type": "page_break" if br_type == "page" else "column_break",
            },
        })
        continuation = copy.deepcopy(template)
        continuation["id"] = ctx.new_id()
        continuation["children"] = segment
        ctx.map_node(continuation["id"], p, status="approximate")
        items.append({
            "kind": "paragraph", "node": continuation, "num_id": None, "ilvl": 0,
        })
    return items


# --- tables -------------------------------------------------------------


def _parse_table(ctx: _Ctx, tbl: etree._Element) -> dict[str, Any] | None:
    node_id = ctx.new_id()
    ctx.map_node(node_id, tbl)
    tbl_pr = tbl.find(_w("tblPr"))

    alignment = "left"
    jc = tbl_pr.find(_w("jc")) if tbl_pr is not None else None
    if jc is not None and jc.get(_w("val")) in ("left", "center", "right"):
        alignment = jc.get(_w("val"))
    indent_mpt = 0
    tbl_ind = tbl_pr.find(_w("tblInd")) if tbl_pr is not None else None
    if tbl_ind is not None:
        indent_mpt = _twip_to_mpt(
            _int_attr(ctx, tbl_ind, _w("w"), default=0, node_id=node_id)
        )
        if indent_mpt < 0:
            ctx.issue(
                "IMPORT_NEGATIVE_TABLE_INDENT", "warning",
                "table indent is negative, which word-core-v0 cannot represent; "
                "clamped to 0 (approximate import).",
                el=tbl_ind, node_id=node_id, computed_mpt=indent_mpt,
            )
            ctx.degrade(node_id, "approximate")
            indent_mpt = 0
    width: dict[str, Any] = {"mode": "auto", "value_mpt": 0}
    tbl_w = tbl_pr.find(_w("tblW")) if tbl_pr is not None else None
    if tbl_w is not None:
        w_type = tbl_w.get(_w("type"), "auto")
        w_val = _int_attr(ctx, tbl_w, _w("w"), default=0, node_id=node_id)
        if w_type == "dxa":
            width = {
                "mode": "absolute",
                "value_mpt": _nonneg(
                    ctx, _twip_to_mpt(w_val), "table width", node_id, tbl_w,
                ),
            }
        elif w_type == "pct":
            value_ppm = w_val * 200
            if value_ppm > 1_000_000:
                # Word allows percent widths > 100%; the schema caps
                # value_ppm at 1_000_000 — clamped, typed, never silent.
                ctx.issue(
                    "IMPORT_TABLE_WIDTH_CLAMPED", "warning",
                    f"table percent width {value_ppm} ppm exceeds 100%; clamped to "
                    "1000000 ppm (approximate import).",
                    el=tbl_w, node_id=node_id, value_ppm=value_ppm,
                )
                ctx.degrade(node_id, "approximate")
                value_ppm = 1_000_000
            width = {"mode": "percent", "value_ppm": value_ppm}
    autofit = True
    tbl_layout = tbl_pr.find(_w("tblLayout")) if tbl_pr is not None else None
    if tbl_layout is not None:
        autofit = tbl_layout.get(_w("type")) != "fixed"

    columns = [
        {"width_mpt": _nonneg(
            ctx,
            _twip_to_mpt(_int_attr(ctx, col, _w("w"), default=0, node_id=node_id)),
            "table column width", node_id, col,
        )}
        for grid in tbl.findall(_w("tblGrid"))
        for col in grid.findall(_w("gridCol"))
    ]
    if not columns:
        ctx.issue(
            "IMPORT_TABLE_MISSING_GRID", "error",
            "table has no w:tblGrid; a word-core-v0 table requires explicit columns.",
            el=tbl, node_id=node_id,
        )
        return None

    is_direct_style, style_chain = _resolve_table_style(ctx, tbl_pr, node_id)
    border_ctx = _table_border_context(ctx, tbl_pr, node_id, style_chain)
    _check_table_style_conditional_bands(
        ctx, tbl_pr, node_id, is_direct_style, style_chain
    )
    node = {
        "id": node_id,
        "type": "table",
        "layout": {
            "alignment": alignment,
            "indent_mpt": indent_mpt,
            "width": width,
            "autofit": autofit,
        },
        "columns": columns,
        "style": {
            "background": _shd_background_chained(
                ctx, tbl_pr, style_chain, "tblPr", node_id
            ),
            "borders": _table_frame_borders(ctx, border_ctx, node_id),
        },
        "rows": [],
    }

    parsed_rows: list[list[dict[str, Any]]] = []
    for tr in (
        c for c in _iter_content(ctx, tbl) if etree.QName(c).localname == "tr"
    ):
        parsed_rows.append(_parse_row(ctx, tr, node, style_chain))
    _resolve_row_spans(ctx, parsed_rows)
    for row_node, row_cells in zip(node["rows"], parsed_rows, strict=True):
        row_node["cells"] = [c["node"] for c in row_cells if not c["continuation"]]
    _pad_grid_gaps(ctx, node)
    _resolve_table_borders(ctx, node, border_ctx)
    return node


def _pad_grid_gaps(ctx: _Ctx, table_node: dict[str, Any]) -> None:
    """Pad rows whose cells leave uncovered grid positions with
    synthesized empty cells (`IMPORT_TABLE_GRID_PADDED`, approximate) —
    real DOCX tables may end a row short of the grid, and Word renders
    the uncovered area as empty space; `validate_semantics` rejects it
    as `TABLE_GRID_GAP`. Every pad is reported, never silent (RFC003
    §45). Rows with malformed grid geometry are left untouched — the
    final canonicalize guard in `import_docx` fails those imports with a
    typed issue instead.
    """
    num_columns = len(table_node["columns"])
    active_row_spans: dict[int, int] = {}
    rows = table_node["rows"]
    for ri, row in enumerate(rows):
        covered = {gc for gc, until in active_row_spans.items() if until >= ri}
        for cell in row["cells"]:
            grid_column = cell.get("grid_column")
            col_span = cell.get("col_span")
            row_span = cell.get("row_span")
            if not (
                isinstance(grid_column, int)
                and isinstance(col_span, int)
                and col_span >= 1
                and isinstance(row_span, int)
                and row_span >= 1
            ):
                return
            for gc in range(grid_column, grid_column + col_span):
                covered.add(gc)
                active_row_spans[gc] = max(active_row_spans.get(gc, -1), ri + row_span - 1)
        missing = [gc for gc in range(num_columns) if gc not in covered]
        for start, span in _contiguous_runs(missing):
            row["cells"].append(_synthesized_cell(ctx, row, start, span))
        if missing:
            row["cells"].sort(key=lambda c: c["grid_column"])


def _contiguous_runs(columns: list[int]) -> list[tuple[int, int]]:
    """Group sorted column indices into (start, span) runs."""
    runs: list[tuple[int, int]] = []
    for gc in columns:
        if runs and runs[-1][0] + runs[-1][1] == gc:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((gc, 1))
    return runs


def _synthesized_cell(ctx: _Ctx, row_node: dict[str, Any], grid_column: int,
                      col_span: int) -> dict[str, Any]:
    """An empty pad cell mirroring what Word renders for an uncovered
    grid position. Recorded in the source map as `technical` — it has no
    source content of its own.
    """
    node_id = ctx.new_id()
    paragraph = _synthesized_paragraph(ctx)
    row_entry = ctx.source_map.get(row_node["id"], {})
    for synthesized_id in (node_id, paragraph["id"]):
        ctx.source_map[synthesized_id] = {
            "part": row_entry.get("part", "word/document.xml"),
            "provenance": row_entry.get("provenance", "."),
            "import_status": "technical",
        }
    ctx.issue(
        "IMPORT_TABLE_GRID_PADDED", "warning",
        f"table row leaves grid columns {grid_column}..{grid_column + col_span - 1} "
        "uncovered; an empty cell was synthesized to complete the grid, matching Word's "
        "rendering of the uncovered area (approximate import).",
        node_id=node_id, grid_column=grid_column, col_span=col_span, row_id=row_node["id"],
    )
    return {
        "id": node_id,
        "type": "cell",
        "grid_column": grid_column,
        "col_span": col_span,
        "row_span": 1,
        "layout": {
            "width_mpt": 0,
            "padding": {"top_mpt": 0, "right_mpt": 0, "bottom_mpt": 0, "left_mpt": 0},
            "vertical_alignment": "top",
            "text_direction": "horizontal",
        },
        "style": {"background": None, "borders": None},
        "children": [paragraph],
    }


def _parse_row(ctx: _Ctx, tr: etree._Element, table_node: dict[str, Any],
               style_chain: list[etree._Element]) -> list[dict[str, Any]]:
    node_id = ctx.new_id()
    ctx.map_node(node_id, tr)
    tr_pr = tr.find(_w("trPr"))
    height: dict[str, Any] = {"mode": "auto", "value_mpt": 0}
    repeat_header = cant_split = False
    if tr_pr is not None:
        cant_split = tr_pr.find(_w("cantSplit")) is not None
        repeat_header = tr_pr.find(_w("tblHeader")) is not None
        tr_height = tr_pr.find(_w("trHeight"))
        if tr_height is not None:
            mode = "exact" if tr_height.get(_w("hRule")) == "exact" else "at_least"
            height = {
                "mode": mode,
                "value_mpt": _twip_to_mpt(
                    _int_attr(ctx, tr_height, _w("val"), default=0, node_id=node_id)
                ),
            }
    row_node = {
        "id": node_id,
        "type": "row",
        "layout": {"height": height, "repeat_header": repeat_header, "cant_split": cant_split},
        "cells": [],
    }
    table_node["rows"].append(row_node)

    cells: list[dict[str, Any]] = []
    cursor = 0
    # Iterate w:tr children through _iter_content so a w:sdt wrapping a
    # w:tc is unwrapped instead of silently dropping the cell; every
    # non-tc child is either known-inert or reported (RFC003 §45).
    for tc in _iter_content(ctx, tr):
        tag = etree.QName(tc).localname
        if tag != "tc":
            if tag == "trPr":
                continue  # parsed above
            if tag == "tblPrEx" or tag in _IGNORED_PARAGRAPH_CHILDREN:
                ctx.issue(
                    "IMPORT_IGNORED_ELEMENT", "info",
                    f"non-content table row element 'w:{tag}' ignored.",
                    el=tc, element=tag,
                )
            else:
                ctx.issue(
                    "IMPORT_UNSUPPORTED_ELEMENT", "warning",
                    f"table row child element 'w:{tag}' is not supported by the "
                    "word-core-v0 importer (approximate import).",
                    el=tc, node_id=node_id, element=tag,
                )
                ctx.degrade(node_id, "approximate")
            continue
        tc_pr = tc.find(_w("tcPr"))
        grid_span_el = tc_pr.find(_w("gridSpan")) if tc_pr is not None else None
        col_span = (
            _int_attr(ctx, grid_span_el, _w("val"), default=1, node_id=node_id, required=True)
            if grid_span_el is not None
            else 1
        )
        v_merge = tc_pr.find(_w("vMerge")) if tc_pr is not None else None
        is_continuation = v_merge is not None and v_merge.get(_w("val"), "continue") == "continue"
        if is_continuation:
            if _continuation_cell_has_content(tc):
                ctx.issue(
                    "IMPORT_CONTENT_IN_CONTINUATION_CELL", "error",
                    "a vMerge continuation cell carries visible content; word-core-v0 "
                    "cannot represent it.",
                    el=tc,
                )
            cells.append({
                "continuation": True, "grid_column": cursor, "col_span": col_span, "el": tc,
            })
        else:
            cells.append({
                "continuation": False,
                "grid_column": cursor,
                "col_span": col_span,
                "restart": v_merge is not None,
                "node": _parse_cell(
                    ctx, tc, tc_pr, cursor, col_span, style_chain, table_node
                ),
            })
        cursor += col_span
    return cells


def _continuation_cell_has_content(tc: etree._Element) -> bool:
    """Visible content anywhere under a continuation cell — descendant
    `w:t` with non-empty text (also inside `w:hyperlink`), a nested
    `w:tbl`, a `w:drawing`, legacy VML (`w:pict`/`w:object`), or an
    explicit `w:tab`/`w:br` control — must never be silently dropped.
    """
    for el in tc.iter():
        local = etree.QName(el).localname
        if local == "t" and el.text:
            return True
        if local in ("tbl", "drawing", "pict", "object") and el is not tc:
            return True
        if local in ("tab", "br"):
            parent = el.getparent()
            if parent is not None and etree.QName(parent).localname == "r":
                # A visible tab/break run child — NOT a w:pPr/w:tabs/w:tab
                # tab-stop definition, which is invisible layout metadata.
                return True
    return False


def _synthesized_paragraph(ctx: _Ctx) -> dict[str, Any]:
    """A minimal empty paragraph matching the renderer's minimal-cell
    convention (RFC001 word-core-v0: every cell holds >= 1 paragraph and
    ends with one). The caller records the source-map entry.
    """
    return {
        "id": ctx.new_id(),
        "type": "paragraph",
        "semantic": {"role": "body", "heading_level": None, "outline_level": None},
        "layout": {
            "alignment": "left",
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
        "children": [],
    }


def _repair_cell_children(ctx: _Ctx, tc: etree._Element, node_id: str,
                          children: list[dict[str, Any]]) -> None:
    """Make a cell's children satisfy `validate_semantics`' cell rules
    (>= 1 paragraph; last child a paragraph) by appending a synthesized
    empty paragraph — typed, never silent (RFC003 §45).
    """
    if not children:
        code = "IMPORT_EMPTY_CELL_PARAGRAPH"
        message = (
            "cell has no importable paragraph; an empty paragraph was synthesized to "
            "match the renderer's minimal-cell convention (approximate import)."
        )
    elif children[-1].get("type") != "paragraph":
        code = "IMPORT_CELL_TRAILING_PARAGRAPH_ADDED"
        message = (
            f"cell's last importable child is a {children[-1].get('type')!r}; a trailing "
            "empty paragraph was synthesized to satisfy the word-core-v0 cell rule "
            "(approximate import)."
        )
    else:
        return
    paragraph = _synthesized_paragraph(ctx)
    ctx.map_node(paragraph["id"], tc, status="approximate")
    children.append(paragraph)
    ctx.issue(code, "warning", message, el=tc, node_id=node_id)
    ctx.degrade(node_id, "approximate")


def _cell_width_reference(table_node: dict[str, Any]) -> tuple[int, str] | None:
    """The absolute length a percent `w:tcW` is resolved against: the
    table's own absolute width when it has one, else the grid sum (the
    layout Word lays a table out on); `None` when the document offers
    neither. A percent TABLE width is not a length and is never used."""
    width = table_node["layout"]["width"]
    if width["mode"] == "absolute" and width["value_mpt"] > 0:
        return width["value_mpt"], "table_width"
    grid_sum = sum(column["width_mpt"] for column in table_node["columns"])
    if grid_sum > 0:
        return grid_sum, "grid_sum"
    return None


def _resolve_pct_cell_width(ctx: _Ctx, tc_w: etree._Element, node_id: str,
                            table_node: dict[str, Any]) -> int:
    """RFC005-ERRATA-008 §E2 (measured 2026-09-26): a `w:tcW w:type='pct'`
    used to fall through to width 0 silently, so the label disagreed with
    the document. The cell AST has only `/layout/width_mpt`; the percent
    (fiftieths of a percent, as `w:tblW` reads it) is resolved against
    `_cell_width_reference` deterministically and reported as approximate,
    or -- without a reference or with a malformed value -- stays 0 and is
    counted, never silent (RFC003 §45)."""
    fiftieths = _int_attr(ctx, tc_w, _w("w"), default=None, node_id=node_id)
    reference = _cell_width_reference(table_node)
    value_ppm = None if fiftieths is None else fiftieths * 200
    if fiftieths is None or reference is None:
        ctx.issue(
            "IMPORT_CELL_WIDTH_UNRESOLVED", "warning",
            "cell percent width cannot be resolved to a length (no absolute table "
            "width or grid sum, or a malformed value); kept 0 (approximate import).",
            el=tc_w, node_id=node_id, value_ppm=value_ppm,
        )
        ctx.degrade(node_id, "approximate")
        return 0
    reference_mpt, reference_kind = reference
    width_mpt = _nonneg(
        ctx,
        round_half_away_from_zero(reference_mpt * fiftieths, 5000),
        "cell width", node_id, tc_w,
    )
    ctx.issue(
        "IMPORT_CELL_WIDTH_PCT_RESOLVED", "warning",
        f"cell percent width {value_ppm} ppm resolved against the {reference_kind} "
        f"({reference_mpt} mpt) to {width_mpt} mpt (approximate import).",
        el=tc_w, node_id=node_id, value_ppm=value_ppm,
        reference=reference_kind, reference_mpt=reference_mpt,
    )
    ctx.degrade(node_id, "approximate")
    return width_mpt


def _parse_cell(ctx: _Ctx, tc: etree._Element, tc_pr: etree._Element | None,
                grid_column: int, col_span: int,
                style_chain: list[etree._Element],
                table_node: dict[str, Any]) -> dict[str, Any]:
    node_id = ctx.new_id()
    ctx.map_node(node_id, tc)
    width_mpt = 0
    tc_w = tc_pr.find(_w("tcW")) if tc_pr is not None else None
    tc_w_type = tc_w.get(_w("type"), "dxa") if tc_w is not None else None
    if tc_w_type == "dxa":
        width_mpt = _nonneg(
            ctx,
            _twip_to_mpt(_int_attr(ctx, tc_w, _w("w"), default=0, node_id=node_id)),
            "cell width", node_id, tc_w,
        )
    elif tc_w_type == "pct":
        width_mpt = _resolve_pct_cell_width(ctx, tc_w, node_id, table_node)
    padding = {"top_mpt": 0, "right_mpt": 0, "bottom_mpt": 0, "left_mpt": 0}
    tc_mar = tc_pr.find(_w("tcMar")) if tc_pr is not None else None
    if tc_mar is not None:
        for side in ("top", "right", "bottom", "left"):
            side_el = tc_mar.find(_w(side))
            if side_el is not None:
                padding[f"{side}_mpt"] = _nonneg(
                    ctx,
                    _twip_to_mpt(
                        _int_attr(ctx, side_el, _w("w"), default=0, node_id=node_id)
                    ),
                    f"cell {side} padding", node_id, side_el,
                )
    v_align_el = tc_pr.find(_w("vAlign")) if tc_pr is not None else None
    vertical_alignment = "top"
    if v_align_el is not None and v_align_el.get(_w("val")) in ("top", "center", "bottom"):
        vertical_alignment = v_align_el.get(_w("val"))

    raw_blocks: list[dict[str, Any]] = []
    for child in _iter_content(ctx, tc):
        tag = etree.QName(child).localname
        if tag == "tcPr":
            continue
        if tag == "p":
            raw_blocks.extend(_parse_p(ctx, child, in_cell=True))
        elif tag == "tbl":
            if not ctx.options.visual_v1:
                ctx.issue(
                    "IMPORT_UNSUPPORTED_NESTED_TABLE", "error",
                    "nested tables are not supported in word-core-v0.",
                    el=child, node_id=node_id,
                )
            else:
                nested = _parse_table(ctx, child)
                if nested is not None:
                    raw_blocks.append({"kind": "block", "node": nested})
        elif tag in _IGNORED_PARAGRAPH_CHILDREN:
            ctx.issue(
                "IMPORT_IGNORED_ELEMENT", "info",
                f"non-content element 'w:{tag}' ignored.", el=child, element=tag,
            )
        else:
            ctx.issue(
                "IMPORT_UNSUPPORTED_ELEMENT", "error",
                f"cell child element 'w:{tag}' is not supported by the word-core-v0 importer.",
                el=child, element=tag,
            )
    children = _assemble_blocks(ctx, raw_blocks)
    _repair_cell_children(ctx, tc, node_id, children)
    return {
        "id": node_id,
        "type": "cell",
        "grid_column": grid_column,
        "col_span": col_span,
        "row_span": 1,
        "layout": {
            "width_mpt": width_mpt,
            "padding": padding,
            "vertical_alignment": vertical_alignment,
            "text_direction": "horizontal",
        },
        "style": {
            "background": _shd_background_chained(
                ctx, tc_pr, style_chain, "tcPr", node_id
            ),
            # Resolved in `_resolve_table_borders` once final grid
            # geometry (row_span, grid padding) is known — D-1 §1.2.
            "borders": None,
        },
        "_tc_borders_el": tc_pr.find(_w("tcBorders")) if tc_pr is not None else None,
        "children": children,
    }


def _resolve_row_spans(ctx: _Ctx, parsed_rows: list[list[dict[str, Any]]]) -> None:
    """Fold vMerge continuation placeholders back into `row_span` on the
    origin cells (the inverse of the renderer's `_expand_grid`). A
    continuation no restart claims is an orphan — a grid hole the AST
    cannot represent, reported as `IMPORT_ORPHAN_VMERGE`, never dropped
    silently.
    """
    for row_index, row in enumerate(parsed_rows):
        for cell in row:
            if cell["continuation"] or not cell.get("restart"):
                continue
            span = 1
            for later in parsed_rows[row_index + 1:]:
                match = next(
                    (
                        c for c in later
                        if c["continuation"] and c["grid_column"] == cell["grid_column"]
                    ),
                    None,
                )
                if match is None:
                    break
                match["claimed"] = True
                span += 1
            cell["node"]["row_span"] = span
    for row in parsed_rows:
        for cell in row:
            if cell["continuation"] and not cell.get("claimed"):
                ctx.issue(
                    "IMPORT_ORPHAN_VMERGE", "error",
                    "a vMerge continuation cell has no restart cell above it in the same "
                    "grid column; the merged grid cannot be reconstructed.",
                    el=cell["el"], grid_column=cell["grid_column"],
                )


# --- list assembly ------------------------------------------------------


def _assemble_blocks(ctx: _Ctx, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild `list` AST nodes from flat numPr paragraphs.

    OOXML does not record list nesting; a marker paragraph whose numId
    is not on the open-list stack while another list is open is assumed
    to start a *nested* list inside the current item (reported, since a
    sibling list is equally valid OOXML).
    """
    blocks: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []  # {"list": node, "item": node, "num_id": int, "def": _NumDef}

    def close_all() -> None:
        stack.clear()

    for item in raw:
        if item["kind"] == "block":
            if stack and item["node"].get("type") == "text_frame":
                # A text frame hosted in a numbered paragraph belongs to
                # that open list item. OOXML flattens list ownership, just
                # as it does for continuation paragraphs; retaining the
                # currently open item is the only placement-preserving
                # reading and matches the renderer's emission order.
                stack[-1]["item"]["children"].append(item["node"])
                continue
            close_all()
            blocks.append(item["node"])
            continue
        node = item["node"]
        num_id = item.get("num_id")
        if num_id is not None:
            definitions = ctx.numdefs if item.get("ilvl", 0) == 0 else ctx.numbering_abstract_defs
            definition = definitions.get(num_id)
            if definition is None:
                ctx.issue(
                    "IMPORT_NUMBERING_UNRESOLVED", "warning",
                    f"paragraph references numId {num_id} with no numbering definition; "
                    "imported as a plain paragraph (list identity lost, text preserved; "
                    "approximate import).",
                    node_id=node["id"], num_id=num_id,
                )
                ctx.degrade(node["id"], "approximate")
                close_all()
                blocks.append(node)
                continue
            if definition.no_marker:
                ctx.issue(
                    "IMPORT_NUMBERING_NO_MARKER", "warning",
                    f"numId {num_id} defines a level with no marker text, which the "
                    "AST has no list format for; imported as a plain paragraph "
                    "rather than inventing a marker glyph (approximate import).",
                    node_id=node["id"], num_id=num_id,
                )
                ctx.degrade(node["id"], "approximate")
                close_all()
                blocks.append(node)
                continue
            if item.get("ilvl", 0) != 0:
                ctx.issue(
                    "IMPORT_LIST_LEVEL_APPROXIMATED", "warning",
                    f"list level {item['ilvl']} imported as its own single-level list "
                    "(approximate).",
                    node_id=node["id"], ilvl=item["ilvl"],
                )
                ctx.degrade(node["id"], "approximate")
            _adjust_marker_indent(ctx, node, definition)
            entry = next(
                (e for e in stack if e["num_id"] == num_id and e["def"] == definition),
                None,
            )
            if entry is not None:
                while stack[-1] is not entry:
                    stack.pop()
            else:
                list_node = _new_list_node(ctx, definition, node)
                if "number_start" in item:
                    list_node["marker"]["start"] = item["number_start"]
                    if item["number_start"] != definition.start:
                        ctx.issue(
                            "IMPORT_LIST_COUNTER_CONTINUED", "info",
                            "a new AST list segment continues its source numbering instance.",
                            node_id=list_node["id"], num_id=num_id,
                            start=item["number_start"],
                        )
                if stack:
                    stack[-1]["item"]["children"].append(list_node)
                    ctx.issue(
                        "IMPORT_LIST_NESTING_ASSUMED", "warning",
                        "a list starting while another list is open is imported as nested "
                        "inside the current list item (OOXML does not record nesting; "
                        "approximate).",
                        node_id=list_node["id"],
                    )
                    ctx.degrade(list_node["id"], "approximate")
                else:
                    blocks.append(list_node)
                stack.append({"list": list_node, "item": None, "num_id": num_id,
                              "def": definition})
            list_item = {"id": ctx.new_id(), "type": "list_item", "children": [node]}
            ctx.source_map[list_item["id"]] = dict(ctx.source_map[node["id"]])
            stack[-1]["list"]["items"].append(list_item)
            stack[-1]["item"] = list_item
        else:
            if stack and _is_continuation(node, stack[-1]["def"]):
                node["layout"]["indent"]["left_mpt"] = 0
                ctx.issue(
                    "IMPORT_LIST_CONTINUATION_ASSUMED", "warning",
                    "an unnumbered paragraph at the open list's indent is imported as a "
                    "continuation paragraph of the current list item (approximate).",
                    node_id=node["id"],
                )
                ctx.degrade(node["id"], "approximate")
                stack[-1]["item"]["children"].append(node)
            else:
                close_all()
                blocks.append(node)
    return blocks


def _new_list_node(ctx: _Ctx, definition: _NumDef, marker_node: dict[str, Any]) -> dict[str, Any]:
    list_id = ctx.new_id()
    ctx.source_map[list_id] = dict(ctx.source_map[marker_node["id"]])
    if definition.approximate:
        ctx.source_map[list_id]["import_status"] = "normalized"
    return {
        "id": list_id,
        "type": "list",
        "list_kind": definition.list_kind,
        "marker": {
            "format": definition.marker_format,
            "start": definition.start,
            "suffix": definition.suffix,
            **({"text": definition.text} if definition.marker_format == "custom_text" else {}),
        },
        "layout": {
            "left_indent_mpt": definition.left_indent_mpt,
            "hanging_mpt": definition.hanging_mpt,
        },
        "items": [],
    }


def _adjust_marker_indent(ctx: _Ctx, node: dict[str, Any], definition: _NumDef) -> None:
    """Un-fold the renderer's marker-indent folding: the emitted `w:ind`
    carries paragraph.left + list.left and paragraph.hanging or
    list.hanging (RFC002 "Lists"). A marker paragraph indented less than
    the list definition would yield a negative paragraph indent — clamped
    to 0 with a typed issue (approximate import).
    """
    indent = node["layout"]["indent"]
    indent["left_mpt"] = indent["left_mpt"] - definition.left_indent_mpt
    if indent["left_mpt"] < 0:
        ctx.issue(
            "IMPORT_NEGATIVE_LIST_INDENT", "warning",
            "un-folding the list marker indent yields a negative paragraph left indent; "
            "clamped to 0 (approximate).",
            node_id=node["id"], computed_mpt=indent["left_mpt"],
        )
        ctx.degrade(node["id"], "approximate")
        indent["left_mpt"] = 0
    if indent["hanging_mpt"] == definition.hanging_mpt:
        indent["hanging_mpt"] = 0


def _is_continuation(node: dict[str, Any], definition: _NumDef) -> bool:
    indent = node["layout"]["indent"]
    return (
        indent["left_mpt"] == definition.left_indent_mpt
        and indent["hanging_mpt"] == 0
        and indent["first_line_mpt"] == 0
    )


# --- sections -----------------------------------------------------------


def _section_node(ctx: _Ctx, sect_pr: etree._Element,
                  blocks: list[dict[str, Any]]) -> dict[str, Any]:
    node_id = ctx.new_id()
    ctx.map_node(node_id, sect_pr)
    for child in sect_pr:
        tag = etree.QName(child).localname
        if tag in _KNOWN_SECTPR_CHILDREN:
            continue
        if ctx.options.visual_v1 and tag in _V1_SECTPR_CHILDREN:
            # Imported below by `visual_v1`, not dropped.
            continue
        if tag in _IGNORABLE_SECTPR_CHILDREN:
            details: dict[str, Any] = {"element": tag}
            if tag in ("headerReference", "footerReference"):
                rel_id = child.get(f"{{{_R_NS}}}id")
                details["reference_type"] = child.get(_w("type"))
                prov = ctx.prov(child)
                ctx.source_map[f"dropped:{prov}"] = {
                    "part": ctx.part,
                    "provenance": prov,
                    "import_status": "dropped",
                    "target": ctx.rels.get(rel_id) if rel_id else None,
                }
            ctx.issue(
                "IMPORT_SECTION_FEATURE_APPROXIMATED", "warning",
                f"section property 'w:{tag}' is outside word-core-v0 and does not "
                "change body-content meaning; dropped with this typed report "
                "(approximate import).",
                el=child, node_id=node_id, **details,
            )
            ctx.degrade(node_id, "approximate")
        else:
            ctx.issue(
                "IMPORT_UNSUPPORTED_SECTION_FEATURE", "warning",
                f"section property 'w:{tag}' is not supported in word-core-v0 and was not "
                "imported (approximate).",
                el=child, node_id=node_id, element=tag,
            )
            ctx.degrade(node_id, "approximate")

    type_el = sect_pr.find(_w("type"))
    break_type = "next_page"
    if type_el is not None:
        break_type = _INV_BREAK_TYPE.get(type_el.get(_w("val")), "next_page")

    pg_sz = sect_pr.find(_w("pgSz"))
    page = {"width_mpt": 595300, "height_mpt": 841900, "orientation": "portrait"}
    if pg_sz is not None:
        page = {
            "width_mpt": _twip_to_mpt(
                _int_attr(ctx, pg_sz, _w("w"), default=11906, node_id=node_id)
            ),
            "height_mpt": _twip_to_mpt(
                _int_attr(ctx, pg_sz, _w("h"), default=16838, node_id=node_id)
            ),
            "orientation": "landscape" if pg_sz.get(_w("orient")) == "landscape" else "portrait",
        }

    pg_mar = sect_pr.find(_w("pgMar"))
    margins = {"top_mpt": 72000, "right_mpt": 72000, "bottom_mpt": 72000, "left_mpt": 72000,
               "gutter_mpt": 0}
    if pg_mar is not None:
        margins = {
            "top_mpt": _twip_to_mpt(
                _int_attr(ctx, pg_mar, _w("top"), default=1440, node_id=node_id)
            ),
            "right_mpt": _twip_to_mpt(
                _int_attr(ctx, pg_mar, _w("right"), default=1440, node_id=node_id)
            ),
            "bottom_mpt": _twip_to_mpt(
                _int_attr(ctx, pg_mar, _w("bottom"), default=1440, node_id=node_id)
            ),
            "left_mpt": _twip_to_mpt(
                _int_attr(ctx, pg_mar, _w("left"), default=1440, node_id=node_id)
            ),
            "gutter_mpt": _twip_to_mpt(
                _int_attr(ctx, pg_mar, _w("gutter"), default=0, node_id=node_id)
            ),
        }
        # `w:top` and `w:bottom` are ST_SignedTwipsMeasure: Word allows a
        # negative page margin (a header that prints into the margin),
        # which the AST's `length_mpt_nonneg` cannot hold. Clamped with a
        # typed issue, following IMPORT_NEGATIVE_LIST_INDENT — the
        # alternative, widening the schema, amends a ratified RFC001 for
        # the same visible result (D1, team-lead decision §8.2).
        for key, value in list(margins.items()):
            if value < 0:
                ctx.issue(
                    "IMPORT_NEGATIVE_PAGE_MARGIN", "warning",
                    f"page margin '{key}' is {value} mpt, which word-core-v0 cannot "
                    "represent; clamped to 0 (approximate import).",
                    el=pg_mar, node_id=node_id, margin=key, computed_mpt=value,
                )
                ctx.degrade(node_id, "approximate")
                margins[key] = 0

    columns: dict[str, Any] = {"count": 1, "equal_width": True, "gap_mpt": 18000,
                               "widths_mpt": []}
    cols = sect_pr.find(_w("cols"))
    if cols is not None:
        equal_width = cols.get(_w("equalWidth")) != "0"
        columns = {
            "count": _int_attr(ctx, cols, _w("num"), default=1, node_id=node_id),
            "equal_width": equal_width,
            "gap_mpt": _twip_to_mpt(
                _int_attr(ctx, cols, _w("space"), default=360, node_id=node_id)
            ),
            "widths_mpt": [],
        }
        col_children = cols.findall(_w("col"))
        if not equal_width and col_children:
            columns["widths_mpt"] = [
                _twip_to_mpt(_int_attr(ctx, c, _w("w"), default=0, node_id=node_id))
                for c in col_children
            ]
            columns["gap_mpt"] = _twip_to_mpt(
                _int_attr(ctx, col_children[0], _w("space"), default=0, node_id=node_id)
            )

    node: dict[str, Any] = {
        "id": node_id,
        "type": "section",
        "break_type": break_type,
        "page": page,
        "margins": margins,
        "columns": columns,
        "children": blocks,
    }
    if ctx.options.visual_v1:
        margins.update(visual_v1.header_footer_margins(ctx, pg_mar, node_id))
        # Optional keys sit between `columns` and `children` in the
        # canonical key order the schema documents; canonicalize()
        # orders them, so insertion order here is presentational only.
        node.update(visual_v1.section_settings(ctx, sect_pr, node_id))
        node.update(visual_v1.import_stories(ctx, sect_pr, node_id))
        node["children"] = blocks
    return node


def _parse_body(ctx: _Ctx, body: etree._Element) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    final_sect_pr: etree._Element | None = None

    for child in _iter_content(ctx, body):
        tag = etree.QName(child).localname
        if tag == "p":
            for item in _parse_p(ctx, child):
                if item["kind"] == "section_end":
                    sections.append(
                        _section_node(ctx, item["sect_pr"], _assemble_blocks(ctx, pending))
                    )
                    pending = []
                    _advance_note_section(ctx)
                    continue
                pending.append(item)
                if "sect_pr" in item:
                    sections.append(
                        _section_node(ctx, item["sect_pr"], _assemble_blocks(ctx, pending))
                    )
                    pending = []
                    _advance_note_section(ctx)
        elif tag == "tbl":
            table = _parse_table(ctx, child)
            if table is not None:
                pending.append({"kind": "block", "node": table})
        elif tag == "sectPr":
            final_sect_pr = child
        elif tag in _IGNORED_PARAGRAPH_CHILDREN:
            ctx.issue(
                "IMPORT_IGNORED_ELEMENT", "info",
                f"non-content element 'w:{tag}' ignored.", el=child, element=tag,
            )
        else:
            ctx.issue(
                "IMPORT_UNSUPPORTED_ELEMENT", "error",
                f"body child element 'w:{tag}' is not supported by the word-core-v0 importer.",
                el=child, element=tag,
            )

    if final_sect_pr is None:
        ctx.issue(
            "IMPORT_MISSING_SECTPR", "warning",
            "document body has no final w:sectPr; a default A4 section was "
            "synthesized (approximate import).",
        )
        sections.append({
            "id": ctx.new_id(),
            "type": "section",
            "break_type": "next_page",
            "page": {"width_mpt": 595300, "height_mpt": 841900, "orientation": "portrait"},
            "margins": {"top_mpt": 72000, "right_mpt": 72000, "bottom_mpt": 72000,
                        "left_mpt": 72000, "gutter_mpt": 0},
            "columns": {"count": 1, "equal_width": True, "gap_mpt": 18000, "widths_mpt": []},
            "children": _assemble_blocks(ctx, pending),
        })
    else:
        sections.append(_section_node(ctx, final_sect_pr, _assemble_blocks(ctx, pending)))
    _drain_field_stack(ctx)
    for section in sections:
        _sweep_field_markers(section)
    return sections


def _sweep_notes(notes: dict[str, list[dict[str, Any]]]) -> None:
    """The same sweep for the root-level note stories."""
    for bodies in notes.values():
        for body in bodies:
            _sweep_field_markers(body)


# --- metadata / rels ----------------------------------------------------


def _rels_part_for(part: str) -> str:
    """The .rels part describing `part` (word/header1.xml ->
    word/_rels/header1.xml.rels)."""
    directory, _, name = part.rpartition("/")
    return f"{directory}/_rels/{name}.rels" if directory else f"_rels/{name}.rels"


def _load_rels(ctx: _Ctx) -> None:
    """Load every relationship part into its OWNING part's scope.

    Relationship ids are part-local: merging them into one map would
    resolve a header's `rId4` to the body's target. A malformed .rels
    part yields an empty scope with a typed warning — an unreadable
    relationship file must not fail an otherwise importable document
    (RFC003 §45).
    """
    for rels_part, data in sorted(ctx.parts.items()):
        if not rels_part.endswith(".rels"):
            continue
        directory, _, name = rels_part.rpartition("/")
        if not directory.endswith("_rels"):
            continue
        owner = f"{directory[: -len('_rels')]}{name[: -len('.rels')]}"
        try:
            root = etree.fromstring(data)
        except etree.XMLSyntaxError as exc:
            ctx.issues.append(
                Issue(
                    code="IMPORT_RELS_MALFORMED", severity="warning", node_id=None,
                    path=f"/{rels_part}",
                    message=f"relationship part '{rels_part}' is not well-formed XML; "
                            f"its targets cannot be resolved: {exc}",
                    details={"part": rels_part},
                )
            )
            continue
        scope = ctx.rels_by_part.setdefault(owner, {})
        for rel in root.findall(f"{{{_REL_NS}}}Relationship"):
            rel_id, target = rel.get("Id"), rel.get("Target")
            if rel_id and target:
                scope[rel_id] = target


def _metadata(ctx: _Ctx, core_xml: bytes | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"language": None, "title": None, "subject": None}
    if core_xml is None:
        return metadata
    try:
        root = etree.fromstring(core_xml)
    except etree.XMLSyntaxError as exc:
        ctx.issues.append(
            Issue(
                code="IMPORT_METADATA_MALFORMED", severity="warning", node_id=None,
                path="/docProps/core.xml",
                message=f"docProps/core.xml is not well-formed XML; metadata defaults "
                        f"used: {exc}",
                details={},
            )
        )
        return metadata
    for key in ("language", "title", "subject"):
        el = root.find(f"{{{_DC_NS}}}{key}")
        if el is not None and el.text:
            metadata[key] = el.text
    return metadata


# --- entry point --------------------------------------------------------


def import_docx(
    docx_bytes: bytes,
    *,
    scan_limits: ScanLimits | None = None,
    document_id: str = "doc",
    options: ImportOptions | None = None,
) -> ImportResult:
    """Import a `word-core-v0` DOCX package into a Canonical AST.

    Runs the security scan first (any error-level `IMPORT_SECURITY_*`
    finding fails the import with `ast=None`; warning-level scan
    findings carry into the report), then parses `[Content_Types].xml`,
    `document.xml`, `styles.xml` and `numbering.xml`, resolving
    effective properties for the word-core-v0 subset. Returns the AST,
    the ImportSourceMap (AST node id -> part/provenance/import_status),
    the issue report, and the extracted media bytes.
    """
    security = scan_docx(docx_bytes, scan_limits)
    if any(i.severity in ("error", "fatal") for i in security):
        return ImportResult(status="failed", ast=None, source_map={}, report=security)

    with zipfile.ZipFile(BytesIO(docx_bytes)) as zf:
        parts = {info.filename: zf.read(info.filename) for info in zf.infolist()}

    document_xml = parts.get("word/document.xml")
    if document_xml is None:
        return ImportResult(
            status="failed", ast=None, source_map={},
            report=[Issue(
                code="IMPORT_MISSING_DOCUMENT_PART", severity="error", node_id=None,
                path="/word/document.xml",
                message="package has no word/document.xml part.", details={},
            )],
        )
    try:
        root = etree.fromstring(document_xml)
    except etree.XMLSyntaxError as exc:
        return ImportResult(
            status="failed", ast=None, source_map={},
            report=[Issue(
                code="IMPORT_MALFORMED_XML", severity="error", node_id=None,
                path="/word/document.xml",
                message=f"word/document.xml is not well-formed XML: {exc}", details={},
            )],
        )

    ctx = _Ctx(tree=etree.ElementTree(root), parts=parts,
               options=options or ImportOptions())
    # Warning-level scan findings (inert external-template refs) carry
    # into the import report without blocking the import.
    ctx.issues.extend(security)
    try:
        _load_styles(ctx, parts.get("word/styles.xml"))
        _load_numbering(ctx, parts.get("word/numbering.xml"))
    except etree.XMLSyntaxError as exc:
        return ImportResult(
            status="failed", ast=None, source_map={},
            report=[Issue(
                code="IMPORT_MALFORMED_XML", severity="error", node_id=None,
                path="/word/styles.xml",
                message=f"a styles/numbering part is not well-formed XML: {exc}", details={},
            )],
        )
    _load_rels(ctx)

    body = root.find(_w("body"))
    if body is None:
        return ImportResult(
            status="failed", ast=None, source_map={},
            report=[Issue(
                code="IMPORT_MALFORMED_XML", severity="error", node_id=None,
                path="/word/document.xml",
                message="word/document.xml has no w:body element.", details={},
            )],
        )

    # Prescan sectPr elements (document order — each sectPr terminates
    # its section) for footnote/endnote numbering settings, so note
    # markers can be formatted while the body is parsed linearly.
    ctx.note_conf = [_note_settings(ctx, sp) for sp in body.iter(_w("sectPr"))]

    # Note bodies are parsed BEFORE the body text so a w:footnoteReference
    # can resolve to a real node id as the body is parsed linearly.
    if ctx.options.visual_v1:
        visual_v1.load_note_bodies(ctx)

    sections = _parse_body(ctx, body)
    notes = visual_v1.emitted_notes(ctx) if ctx.options.visual_v1 else {}
    _sweep_notes(notes)
    for story in (sections, *notes.values()):
        _resolve_internal_anchors(ctx, {"root": story})

    # Prune resources orphaned by the field sweep (a w:drawing dropped
    # inside a terminated field region registered its media before the
    # node was recorded-dropped): only resources referenced by a
    # surviving inline_image ship in the AST and in get_resources().
    referenced: set[str] = set()

    def _collect_refs(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("resource_ref")
            if isinstance(ref, str):
                referenced.add(ref)
            for value in node.values():
                _collect_refs(value)
        elif isinstance(node, list):
            for item in node:
                _collect_refs(item)

    # Every story is walked, not just the body: a header's logo is
    # referenced from inside `section.headers`, and a note body's image
    # from inside `footnotes` — pruning on the body alone would delete
    # exactly the header images this profile exists to keep.
    _collect_refs(sections)
    _collect_refs(notes)
    kept_resources = []
    for resource in sorted(ctx.resources_by_sha.values(), key=lambda r: r["id"]):
        if resource["id"] in referenced:
            kept_resources.append(resource)
        else:
            ctx.resource_bytes.pop(resource["path"], None)
            ctx.issue(
                "IMPORT_ORPHAN_RESOURCE_PRUNED", "info",
                f"resource '{resource['id']}' has no referencing image node "
                "(its image node was recorded-dropped); pruned from the AST "
                "and the extracted resources.",
            )

    ast = {
        "ast_version": "0.1.0",
        "profile": ctx.options.profile,
        "document_id": document_id,
        "metadata": _metadata(ctx, parts.get("docProps/core.xml")),
        "resources": kept_resources,
        "sections": sections,
        **notes,
        "extensions": {},
    }

    # Contract guard (RFC003 §17.2/§45): the importer must never emit an
    # AST its own downstream rejects. canonicalize() runs schema +
    # semantic validation; a rejection downgrades the import to `failed`
    # with a typed issue carrying the underlying codes — the crash never
    # escapes to callers, and the document takes the quarantine path.
    # The canonical tree is deliberately DISCARDED: `ast`, `source_map`
    # and `report` stay keyed on importer ids (see `ImportResult`).
    try:
        canonicalize(ast)
    except CanonicalizationError as exc:
        codes = sorted({i.code for i in exc.issues})
        ctx.issues.append(
            Issue(
                code="IMPORT_OUTPUT_NOT_CANONICALIZABLE", severity="error", node_id=None,
                path="/",
                message="imported AST was rejected by canonicalize/validate_semantics "
                        f"(codes: {', '.join(codes)}); import failed rather than emitting "
                        "a non-canonicalizable AST.",
                details={"codes": codes},
            )
        )

    has_error = any(i.severity in ("error", "fatal") for i in ctx.issues)
    has_warning = any(i.severity == "warning" for i in ctx.issues)
    if has_error:
        status: ImportStatus = "failed"
    elif has_warning:
        status = "success_with_warnings"
    else:
        status = "success"
    return ImportResult(
        status=status,
        ast=ast,
        source_map=ctx.source_map,
        report=ctx.issues,
        resources=ctx.resource_bytes,
        resource_provenance=ctx.resource_provenance,
    )

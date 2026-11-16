"""Pilot font pack registry (RFC002 §"Fonts").

V0 font pack: Liberation family (metric-compatible with the classic MS
core fonts) plus DejaVu as a secondary fallback family. No embedding —
`fontTable.xml` only lists family names; determinism across machines
comes from pinning renderer output to this fixed pack, never to
whatever fonts happen to be installed locally.
"""

from __future__ import annotations

from unflatten.ast.issues import Issue

# Fonts available in the pinned pilot pack (canonical family names).
PILOT_FONT_PACK: frozenset[str] = frozenset(
    {
        "Liberation Serif",
        "Liberation Sans",
        "Liberation Mono",
        "DejaVu Serif",
        "DejaVu Sans",
        "DejaVu Mono",
        # Carlito (crosextra) is metric-compatible with Calibri; it is in
        # the pack as the fallback target for Calibri requests.
        "Carlito",
        # Caladea (crosextra) is metric-compatible with Cambria.
        "Caladea",
        # Arabic-script body face: the imported corpus is largely Persian
        # and Arabic, whose documents request faces (IranNastaliq,
        # B Nazanin, Traditional Arabic, ...) that no Latin pack font
        # covers. Without a pack member carrying the script, LibreOffice
        # would glyph-fallback to an unpinned system font.
        "Noto Naskh Arabic",
    }
)

# Default fallback map for common proprietary font requests -> pilot pack.
DEFAULT_FALLBACK_MAP: dict[str, str] = {
    "Times New Roman": "Liberation Serif",
    "Arial": "Liberation Sans",
    "Courier New": "Liberation Mono",
    # Carlito is metric-compatible with Calibri (no metric match for
    # Calibri Light in the pack; Carlito is the closest substitute).
    "Calibri": "Carlito",
    "Calibri Light": "Carlito",
    # PostScript names seen in imported real DOCX font tables.
    "TimesNewRomanPSMT": "Liberation Serif",
    "TimesNewRomanPS-BoldMT": "Liberation Serif",
    "TimesNewRomanPS-ItalicMT": "Liberation Serif",
    "ArialMT": "Liberation Sans",
    "Arial-BoldMT": "Liberation Sans",
    # --- observed across the imported real corpus (full-store font scan)
    # Latin faces with a close pack equivalent.
    "Cambria": "Caladea",
    "Cambria Math": "Caladea",
    "Segoe UI": "Liberation Sans",
    "Segoe UI Symbol": "DejaVu Sans",
    "Segoe UI Emoji": "DejaVu Sans",
    "Tahoma": "DejaVu Sans",
    "Verdana": "DejaVu Sans",
    "Trebuchet MS": "Liberation Sans",
    "Comic Sans MS": "DejaVu Sans",
    "Arial Narrow": "Liberation Sans",
    "Arial Black": "Liberation Sans",
    "Arial Unicode MS": "DejaVu Sans",
    "Garamond": "Liberation Serif",
    "Book Antiqua": "Liberation Serif",
    "Palatino Linotype": "Liberation Serif",
    "Georgia": "Liberation Serif",
    "Times New Roman Bold": "Liberation Serif",
    "Times-Roman": "Liberation Serif",
    "TimesNewRoman": "Liberation Serif",
    "TimesNewRomanPS-BoldItalicMT": "Liberation Serif",
    "Consolas": "Liberation Mono",
    "Lucida Console": "Liberation Mono",
    "Literata": "Liberation Serif",
    # East-Asian family names that real Word files attach to *symbol*
    # runs (☏, ✆) rather than to CJK text — verified across the imported
    # corpus: 0 of the affected documents contain CJK characters. DejaVu
    # Sans carries those symbol codepoints; genuine CJK content would be
    # glyph-fallback territory and should add a CJK pack member instead.
    "MS Gothic": "DejaVu Sans",
    "MS PGothic": "DejaVu Sans",
    "MS Mincho": "DejaVu Serif",
    "MS PMincho": "DejaVu Serif",
    "Yu Gothic": "DejaVu Sans",
    "Yu Gothic UI": "DejaVu Sans",
    # Arabic/Persian faces -> the pack's Arabic-script member.
    "Traditional Arabic": "Noto Naskh Arabic",
    "Simplified Arabic": "Noto Naskh Arabic",
    "Sakkal Majalla": "Noto Naskh Arabic",
    "IranNastaliq": "Noto Naskh Arabic",
    "B Nazanin": "Noto Naskh Arabic",
    "IRNazanin": "Noto Naskh Arabic",
    "IRMitra": "Noto Naskh Arabic",
    "IRYakout": "Noto Naskh Arabic",
    "IRTitr": "Noto Naskh Arabic",
    "DecoType Naskh": "Noto Naskh Arabic",
    "Lotus Linotype": "Noto Naskh Arabic",
    "KFGQPC Uthmanic Script HAFS": "Noto Naskh Arabic",
    "KFGQPC Uthman Taha Naskh": "Noto Naskh Arabic",
}


def resolve_font(
    requested: str,
    node_id: str | None,
    path: str,
    mode: str,
    fallback_map: dict[str, str],
    fallback_default: str | None = None,
) -> tuple[str, Issue | None]:
    """Resolve `requested` font family against the pilot pack.

    `require_exact`: unknown font -> `(requested, FONT_NOT_IN_PACK error)`.
    `allow_fallback`: unknown font mapped via `fallback_map` -> the pack
    font plus a `FONT_FALLBACK` warning.

    An unmapped unknown font falls back to `fallback_default` when one is
    configured (`via: "default"` in the issue details), else it errors.
    The default exists because real corpora request a long tail of faces
    no map can enumerate — a full scan of the imported corpus found 544
    distinct unknown family names. The default is part of the pinned font
    policy (and therefore of `font_pack_hash`), and every substitution is
    still reported per run, so determinism and no-silent-loss hold.

    NOTE (labelling): a substituted run renders in the pack font while
    the AST keeps the *requested* family. Training on `/style/font_family`
    for such runs would teach a value the pixels do not show; those
    properties belong under a supervision mask (RFC003 §29). The emitted
    `FONT_FALLBACK` issues carry `node_id`, which is what a mask builder
    needs.
    """
    if requested in PILOT_FONT_PACK:
        return requested, None
    if mode == "allow_fallback" and requested in fallback_map:
        used = fallback_map[requested]
        return used, Issue(
            code="FONT_FALLBACK",
            severity="warning",
            node_id=node_id,
            path=path,
            message=f"font '{requested}' is not in the pilot pack; substituted '{used}'.",
            details={"requested": requested, "used": used, "via": "map"},
        )
    if mode == "allow_fallback" and fallback_default is not None:
        return fallback_default, Issue(
            code="FONT_FALLBACK",
            severity="warning",
            node_id=node_id,
            path=path,
            message=(
                f"font '{requested}' is not in the pilot pack and has no explicit "
                f"mapping; substituted the pinned default '{fallback_default}'."
            ),
            details={"requested": requested, "used": fallback_default, "via": "default"},
        )
    return requested, Issue(
        code="FONT_NOT_IN_PACK",
        severity="error",
        node_id=node_id,
        path=path,
        message=f"font '{requested}' is not in the pilot font pack and has no fallback mapping.",
        details={"requested": requested},
    )

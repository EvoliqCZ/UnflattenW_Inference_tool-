from unflatten.renderer.fonts import DEFAULT_FALLBACK_MAP, PILOT_FONT_PACK, resolve_font


def test_resolve_font_in_pack_no_issue():
    used, issue = resolve_font("Liberation Sans", "n1", "/path", "require_exact", {})
    assert used == "Liberation Sans"
    assert issue is None


def test_resolve_font_require_exact_missing_is_error():
    used, issue = resolve_font("Arial", "n1", "/path", "require_exact", DEFAULT_FALLBACK_MAP)
    assert used == "Arial"
    assert issue is not None
    assert issue.code == "FONT_NOT_IN_PACK"
    assert issue.severity == "error"
    assert issue.details == {"requested": "Arial"}


def test_resolve_font_allow_fallback_maps_and_warns():
    used, issue = resolve_font("Arial", "n1", "/path", "allow_fallback", DEFAULT_FALLBACK_MAP)
    assert used == "Liberation Sans"
    assert issue is not None
    assert issue.code == "FONT_FALLBACK"
    assert issue.severity == "warning"
    assert issue.details == {"requested": "Arial", "used": "Liberation Sans", "via": "map"}
    assert used in PILOT_FONT_PACK


def test_resolve_font_allow_fallback_unmapped_still_errors():
    # No pinned default configured => an unmapped face is still an error.
    # (The family must be absent from DEFAULT_FALLBACK_MAP, which now
    # covers the whole observed corpus tail.)
    used, issue = resolve_font(
        "No Such Family 1234", "n1", "/path", "allow_fallback", DEFAULT_FALLBACK_MAP
    )
    assert used == "No Such Family 1234"
    assert issue is not None
    assert issue.code == "FONT_NOT_IN_PACK"
    assert issue.severity == "error"


def test_default_fallback_map_targets_are_in_pilot_pack():
    for target in DEFAULT_FALLBACK_MAP.values():
        assert target in PILOT_FONT_PACK


def test_calibri_maps_to_carlito():
    # Carlito is metric-compatible with Calibri and part of the pack
    assert "Carlito" in PILOT_FONT_PACK
    for requested in ("Calibri", "Calibri Light"):
        used, issue = resolve_font(requested, "n1", "/path", "allow_fallback", DEFAULT_FALLBACK_MAP)
        assert used == "Carlito"
        assert issue is not None and issue.code == "FONT_FALLBACK"


def test_postscript_name_variants_map_to_liberation():
    expected = {
        "TimesNewRomanPSMT": "Liberation Serif",
        "TimesNewRomanPS-BoldMT": "Liberation Serif",
        "TimesNewRomanPS-ItalicMT": "Liberation Serif",
        "ArialMT": "Liberation Sans",
        "Arial-BoldMT": "Liberation Sans",
    }
    for requested, target in expected.items():
        used, issue = resolve_font(requested, "n1", "/path", "allow_fallback", DEFAULT_FALLBACK_MAP)
        assert used == target
        assert issue is not None and issue.code == "FONT_FALLBACK"


def test_calibri_document_compiles_with_font_fallback_warning(tmp_path):
    import copy
    import json
    from pathlib import Path

    from unflatten.renderer.api import RendererOptions
    from unflatten.renderer.compiler import compile_docx

    corpus = Path(__file__).parents[1] / "data" / "corpus" / "valid" / "minimal_paragraph.json"
    ast = copy.deepcopy(json.loads(corpus.read_text()))
    ast["sections"][0]["children"][0]["children"][0]["style"]["font_family"] = "Calibri"
    result = compile_docx(ast, tmp_path / "calibri.docx", RendererOptions(strict=True))
    assert result.status in ("success", "success_with_warnings")
    fallbacks = [i for i in result.issues if i.code == "FONT_FALLBACK"]
    assert fallbacks and fallbacks[0].details == {
        "requested": "Calibri", "used": "Carlito", "via": "map",
    }


class TestPinnedDefaultFallback:
    """A pinned default substitute for the long tail of real-corpus fonts
    (544 distinct unknown family names in the imported corpus)."""

    def test_unmapped_font_uses_pinned_default_with_typed_warning(self):
        from unflatten.renderer.fonts import resolve_font

        used, issue = resolve_font(
            "IranNastaliq-Unmapped", "n1", "/p", "allow_fallback", {},
            fallback_default="DejaVu Sans",
        )
        assert used == "DejaVu Sans"
        assert issue is not None
        assert issue.code == "FONT_FALLBACK" and issue.severity == "warning"
        assert issue.details["via"] == "default"
        assert issue.details["requested"] == "IranNastaliq-Unmapped"
        assert issue.node_id == "n1"  # a mask builder needs this

    def test_without_default_unmapped_font_still_errors(self):
        from unflatten.renderer.fonts import resolve_font

        used, issue = resolve_font(
            "Whatever", "n1", "/p", "allow_fallback", {}, fallback_default=None
        )
        assert issue is not None and issue.code == "FONT_NOT_IN_PACK"
        assert issue.severity == "error"

    def test_explicit_map_wins_over_default(self):
        from unflatten.renderer.fonts import resolve_font

        used, issue = resolve_font(
            "Cambria", "n1", "/p", "allow_fallback",
            {"Cambria": "Caladea"}, fallback_default="DejaVu Sans",
        )
        assert used == "Caladea"
        assert issue.details["via"] == "map"

    def test_require_exact_ignores_the_default(self):
        from unflatten.renderer.fonts import resolve_font

        used, issue = resolve_font(
            "Cambria", "n1", "/p", "require_exact", {}, fallback_default="DejaVu Sans"
        )
        assert used == "Cambria"
        assert issue.code == "FONT_NOT_IN_PACK"

    def test_font_pack_hash_tracks_the_default(self):
        from unflatten.renderer.api import FontPolicy
        from unflatten.renderer.manifest import compute_font_pack_hash

        a = compute_font_pack_hash(FontPolicy(fallback_default="DejaVu Sans"))
        b = compute_font_pack_hash(FontPolicy(fallback_default="Liberation Sans"))
        c = compute_font_pack_hash(FontPolicy(fallback_default=None))
        assert len({a, b, c}) == 3

    def test_corpus_font_families_all_resolve(self):
        """The families the full-corpus scan found must not error."""
        from unflatten.renderer.api import FontPolicy
        from unflatten.renderer.fonts import resolve_font

        policy = FontPolicy()
        observed = [
            "MS Mincho", "Tahoma", "IranNastaliq", "Traditional Arabic",
            "Sakkal Majalla", "Cambria", "Simplified Arabic", "MS Gothic",
            "Wingdings", "Segoe UI Symbol", "Symbol", "B Nazanin",
            "Arial Narrow", "Comic Sans MS", "Book Antiqua", "Verdana",
            "totally-unknown-face-xyz",
        ]
        for family in observed:
            _used, issue = resolve_font(
                family, "n1", "/p", policy.mode, policy.fallback_map,
                policy.fallback_default,
            )
            assert issue is None or issue.severity == "warning", family

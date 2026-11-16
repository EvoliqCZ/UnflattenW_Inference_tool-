"""R7 — round-trip validation and TRAIN_READY gate (RFC002 "Round-trip
gate is mandatory for TRAIN_READY" / validation layers V5-V7).

For every `tests/data/corpus/valid` fixture: canonicalize -> compile ->
assert success, 100% text conservation, XSD validity, and determinism
(two independent compiles are byte-identical). The LibreOffice open
probe (V5) additionally opens every compiled fixture when the probe is
available in this environment; it is skipped (never faked) otherwise.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from unflatten.ast.canonicalizer import canonicalize
from unflatten.renderer.api import RendererOptions
from unflatten.renderer.compiler import compile_docx
from unflatten.renderer.probe import libreoffice_open_probe, probe_available
from unflatten.renderer.resources import DirResourceProvider
from unflatten.renderer.text_extract import verify_text_conservation
from unflatten.renderer.xsd_validation import validate_part_against_xsd

VALID_DIR = Path(__file__).parent.parent / "data" / "corpus" / "valid"
ASSETS_DIR = Path(__file__).parent.parent / "data" / "assets"

_FIXTURE_NAMES = sorted(p.stem for p in VALID_DIR.glob("*.json"))


def _load(name: str) -> dict:
    with open(VALID_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _prepare(name: str) -> tuple[dict, DirResourceProvider | None]:
    ast = _load(name)
    resources = None
    if name == "inline_image_with_resource":
        data = (ASSETS_DIR / "tiny_2x2.png").read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        resource = ast["resources"][0]
        resource["sha256"] = sha256
        resource["path"] = "tiny_2x2.png"
        resource["pixel_width"] = 2
        resource["pixel_height"] = 2
        resources = DirResourceProvider(ASSETS_DIR)
    return ast, resources


@pytest.mark.parametrize("name", _FIXTURE_NAMES)
def test_roundtrip_gate_for_every_valid_corpus_fixture(tmp_path, name):
    ast, resources = _prepare(name)

    # canonicalize (part of compile_docx, but exercised standalone too
    # so a canonicalization failure is attributable, not hidden inside
    # a generic compile failure).
    canonical = canonicalize(ast)
    assert canonical["profile"] == "word-core-v0"

    out1 = tmp_path / f"{name}_1.docx"
    out2 = tmp_path / f"{name}_2.docx"
    result1 = compile_docx(ast, out1, RendererOptions(), resources=resources)
    result2 = compile_docx(ast, out2, RendererOptions(), resources=resources)

    assert result1.status in ("success", "success_with_warnings"), result1.issues
    assert not any(i.severity in ("error", "fatal") for i in result1.issues)

    # determinism: two independent compiles, byte-identical package.
    assert out1.read_bytes() == out2.read_bytes()
    assert result1.package_hash == result2.package_hash

    # text conservation: 100% of visible text preserved. Pass the
    # renderer's explicit technical-boundary-paragraph positions so a
    # genuinely empty AST paragraph ending a section is never confused
    # with a synthesized one.
    positions = result1.metrics["technical_boundary_paragraph_positions"]
    assert verify_text_conservation(ast, out1.read_bytes(), positions) == []

    # XSD validity (V4) of every wml.xsd-rooted part.
    with zipfile.ZipFile(out1) as zf:
        for part_name in ("word/document.xml", "word/numbering.xml", "word/styles.xml",
                            "word/settings.xml", "word/fontTable.xml"):
            if part_name not in zf.namelist():
                continue
            issues = validate_part_against_xsd(f"/{part_name}", zf.read(part_name))
            assert issues == [], [i.message for i in issues]


# --- LibreOffice open probe (V5) ------------------------------------------------


@pytest.mark.skipif(not probe_available(), reason="LibreOffice headless probe unavailable here")
@pytest.mark.parametrize("name", _FIXTURE_NAMES)
def test_libreoffice_opens_every_compiled_corpus_fixture(tmp_path, name):
    ast, resources = _prepare(name)
    out = tmp_path / f"{name}.docx"
    result = compile_docx(ast, out, RendererOptions(), resources=resources)
    assert result.status in ("success", "success_with_warnings")

    issue = libreoffice_open_probe(out)
    assert issue is None, issue.message if issue else None

"""R6 — determinism hardening: limits, package manifest, ZIP hygiene,
no-silent-loss node accounting, docProps/core.xml metadata, and
subprocess/hash-seed-randomized determinism (RFC002 "Determinism
guarantees" / "Performance" / "Sidecar artifacts").
"""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from unflatten.renderer.api import RendererOptions
from unflatten.renderer.compiler import compile_docx
from unflatten.renderer.limits import check_limits
from unflatten.renderer.manifest import build_package_manifest, write_package_manifest
from unflatten.renderer.ooxml.package_parts import (
    build_core_properties,
    has_mappable_core_properties,
)
from unflatten.renderer.opc import PackagePart, build_package
from unflatten.renderer.version import RENDERER_VERSION

VALID_DIR = Path(__file__).parent.parent / "data" / "corpus" / "valid"


def _load(name: str) -> dict:
    with open(VALID_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


# --- limits ------------------------------------------------------------------


def test_check_limits_passes_for_small_ast():
    ast = _load("minimal_paragraph")
    issues = check_limits(
        ast,
        max_nodes=100_000,
        max_tree_depth=64,
        max_text_chars=10_000_000,
        max_table_cells=100_000,
    )
    assert issues == []


def test_check_limits_reports_node_count_exceeded():
    ast = _load("minimal_paragraph")
    issues = check_limits(
        ast, max_nodes=1, max_tree_depth=64, max_text_chars=10_000_000, max_table_cells=100_000
    )
    assert any(i.code == "LIMIT_EXCEEDED_NODES" for i in issues)


def test_check_limits_reports_tree_depth_exceeded():
    ast = _load("ordered_unordered_nested_list")
    issues = check_limits(
        ast, max_nodes=100_000, max_tree_depth=1, max_text_chars=10_000_000, max_table_cells=100_000
    )
    assert any(i.code == "LIMIT_EXCEEDED_TREE_DEPTH" for i in issues)


def test_check_limits_reports_text_chars_exceeded():
    ast = _load("minimal_paragraph")
    issues = check_limits(
        ast, max_nodes=100_000, max_tree_depth=64, max_text_chars=1, max_table_cells=100_000
    )
    assert any(i.code == "LIMIT_EXCEEDED_TEXT_CHARS" for i in issues)


def test_check_limits_reports_table_cells_exceeded():
    ast = _load("simple_table_2x2")
    issues = check_limits(
        ast, max_nodes=100_000, max_tree_depth=64, max_text_chars=10_000_000, max_table_cells=1
    )
    assert any(i.code == "LIMIT_EXCEEDED_TABLE_CELLS" for i in issues)


def test_compile_docx_fails_when_limit_exceeded(tmp_path):
    ast = _load("minimal_paragraph")
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions(max_nodes=1))
    assert result.status == "failed"
    assert not (tmp_path / "out.docx").exists()
    assert any(i.code == "LIMIT_EXCEEDED_NODES" for i in result.issues)


def _deep_nested_table_ast(depth: int) -> dict:
    """A table nested `depth` levels deep via `cell.children == [table]`
    at every level — few total nodes (~2 per level), but tree depth
    `depth`. Deliberately not a schema-valid, renderable AST: only
    exercises `check_limits`'s generic tree walk (type/children/rows/
    cells), which doesn't validate the rest of the node shape.
    """
    node: dict = {"id": "leaf", "type": "paragraph", "children": []}
    for i in range(depth):
        cell = {"id": f"c{i}", "children": [node]}
        row = {"id": f"row{i}", "cells": [cell]}
        node = {"id": f"t{i}", "type": "table", "rows": [row]}
    return {"sections": [{"children": [node]}]}


def test_check_limits_is_iterative_and_never_raises_recursion_error_for_deep_small_ast():
    import sys

    depth = sys.getrecursionlimit() + 500
    ast = _deep_nested_table_ast(depth)
    issues = check_limits(
        ast,
        max_nodes=100_000,
        max_tree_depth=64,
        max_text_chars=10_000_000,
        max_table_cells=100_000,
    )
    assert any(i.code == "LIMIT_EXCEEDED_TREE_DEPTH" for i in issues)


def test_compile_docx_rejects_deep_nested_ast_before_canonicalize_without_crashing(tmp_path):
    import sys

    depth = sys.getrecursionlimit() + 500
    ast = _deep_nested_table_ast(depth)
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    assert result.status == "failed"
    assert any(i.code == "LIMIT_EXCEEDED_TREE_DEPTH" for i in result.issues)
    assert not (tmp_path / "out.docx").exists()


# --- no-silent-loss node accounting -------------------------------------------


def _all_ast_node_ids(ast: dict) -> set[str]:
    ids: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("id"), str):
                ids.add(node["id"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(ast.get("sections", []))
    return ids


@pytest.mark.parametrize(
    "name",
    [
        "minimal_paragraph",
        "mixed_run_styles",
        "paragraph_with_tabs_and_line_break",
        "multi_paragraph_with_page_break",
        "ordered_unordered_nested_list",
        "simple_table_2x2",
        "table_with_colspan_rowspan",
        "two_section_two_column",
    ],
)
def test_no_silent_loss_every_node_id_is_accounted_for(tmp_path, name):
    from unflatten.ast.canonicalizer import canonicalize

    ast = _load(name)
    result = compile_docx(ast, tmp_path / f"{name}.docx", RendererOptions())
    assert result.status in ("success", "success_with_warnings")

    # `compile_docx` renders the *canonicalized* AST (RFC001 stable
    # preorder id renumbering, §35 step 5), so the accounting must be
    # checked against those renumbered ids, not the authored fixture's.
    canonical = canonicalize(ast)
    rendered = set(result.metrics["rendered_node_ids"])
    unsupported_ids = {i.node_id for i in result.unsupported if i.node_id is not None}
    accounted = rendered | unsupported_ids
    missing = _all_ast_node_ids(canonical) - accounted
    assert missing == set()


# --- ZIP hygiene ---------------------------------------------------------------


def test_zip_entry_order_matches_parts_order_and_fixed_timestamp(tmp_path):
    parts = [
        PackagePart("[Content_Types].xml", b"<a/>"),
        PackagePart("_rels/.rels", b"<b/>"),
        PackagePart("word/document.xml", b"<c/>"),
    ]
    data = build_package(parts)
    out = tmp_path / "p.docx"
    out.write_bytes(data)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert names == [p.name for p in parts]
        for info in zf.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)


def test_zip_builder_rejects_duplicate_part_names():
    parts = [
        PackagePart("word/document.xml", b"<a/>"),
        PackagePart("word/document.xml", b"<b/>"),
    ]
    with pytest.raises(ValueError, match="duplicate part name"):
        build_package(parts)


def test_zip_has_no_duplicate_entries_for_a_real_compile(tmp_path):
    ast = _load("simple_table_2x2")
    out = tmp_path / "out.docx"
    compile_docx(ast, out, RendererOptions())
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert len(names) == len(set(names))


# --- docProps/core.xml metadata -------------------------------------------------


def test_has_mappable_core_properties_false_when_all_absent():
    assert has_mappable_core_properties({"language": None, "title": None, "subject": None}) is False


def test_has_mappable_core_properties_true_when_any_present():
    metadata = {"language": "en-US", "title": None, "subject": None}
    assert has_mappable_core_properties(metadata) is True


def test_build_core_properties_uses_fixed_creator_and_omits_absent_fields():
    xml = build_core_properties({"language": None, "title": "T", "subject": None})
    text = xml.decode()
    assert "<dc:title>T</dc:title>" in text
    assert "<dc:creator>unflatten</dc:creator>" in text
    assert "<dc:subject>" not in text
    assert "<dc:language>" not in text
    assert "created" not in text.lower()
    assert "modified" not in text.lower()


def test_build_core_properties_no_fields_still_has_creator():
    xml = build_core_properties({"language": None, "title": None, "subject": None})
    assert b"<dc:creator>unflatten</dc:creator>" in xml


def test_build_core_properties_element_order_matches_opc_schema():
    xml = build_core_properties({"language": "en-US", "title": "T", "subject": "S"}).decode()
    creator_pos = xml.index("<dc:creator>")
    language_pos = xml.index("<dc:language>")
    subject_pos = xml.index("<dc:subject>")
    title_pos = xml.index("<dc:title>")
    assert creator_pos < language_pos < subject_pos < title_pos


# --- Package Manifest -----------------------------------------------------------


_REQUIRED_MANIFEST_FIELDS = {
    "renderer_version",
    "ast_version",
    "ast_hash",
    "package_hash",
    "profile",
    "ooxml_conformance",
    "compatibility_mode",
    "font_policy",
    "font_pack_hash",
    "status",
    "parts",
    "issue_counts",
    "issues",
}


def test_build_package_manifest_has_all_required_fields(tmp_path):
    ast = _load("minimal_paragraph")
    options = RendererOptions()
    result = compile_docx(ast, tmp_path / "out.docx", options)
    manifest = build_package_manifest(result, options)
    assert _REQUIRED_MANIFEST_FIELDS <= set(manifest)
    assert manifest["renderer_version"] == RENDERER_VERSION
    assert manifest["ast_version"] == result.ast_version
    assert manifest["ast_version"] is not None
    assert manifest["ast_hash"] == result.ast_hash
    assert manifest["package_hash"] == result.package_hash
    assert manifest["profile"] == result.profile
    assert manifest["status"] == result.status
    assert {p["name"] for p in manifest["parts"]} == set(result.parts)
    assert sum(manifest["issue_counts"].values()) == len(result.issues)
    assert len(manifest["issues"]) == len(result.issues)
    assert manifest["issues"] == [
        {"code": i.code, "severity": i.severity, "node_id": i.node_id} for i in result.issues
    ]


def test_build_package_manifest_font_policy_reflects_options(tmp_path):
    ast = _load("minimal_paragraph")
    from unflatten.renderer.api import FontPolicy

    options = RendererOptions(font_policy=FontPolicy(mode="require_exact", fallback_map={}))
    ast_with_pack_font = json.loads(json.dumps(ast))
    ast_with_pack_font["sections"][0]["children"][0]["children"][0]["style"]["font_family"] = (
        "Liberation Sans"
    )
    result = compile_docx(ast_with_pack_font, tmp_path / "out.docx", options)
    manifest = build_package_manifest(result, options)
    assert manifest["font_policy"] == {
        "mode": "require_exact",
        "fallback_map": {},
        "fallback_default": "DejaVu Sans",  # recorded even when unused
    }


def test_build_package_manifest_ooxml_conformance_and_compat_mode_are_options_sensitive(
    tmp_path,
):
    ast = _load("minimal_paragraph")
    default_options = RendererOptions()
    result = compile_docx(ast, tmp_path / "out.docx", default_options)
    manifest = build_package_manifest(result, default_options)
    assert manifest["ooxml_conformance"] == "transitional"
    assert manifest["compatibility_mode"] == "office2013_plus"

    custom_options = RendererOptions(ooxml_conformance="strict-preview")
    manifest2 = build_package_manifest(result, custom_options)
    assert manifest2["ooxml_conformance"] == "strict-preview"


def test_build_package_manifest_font_pack_hash_changes_with_fallback_map(tmp_path):
    from unflatten.renderer.api import FontPolicy
    from unflatten.renderer.manifest import compute_font_pack_hash

    ast = _load("minimal_paragraph")
    result = compile_docx(ast, tmp_path / "out.docx", RendererOptions())
    options_a = RendererOptions(font_policy=FontPolicy(fallback_map={"Arial": "Liberation Sans"}))
    options_b = RendererOptions(
        font_policy=FontPolicy(fallback_map={"Arial": "Liberation Sans", "Georgia": "DejaVu Serif"})
    )
    manifest_a = build_package_manifest(result, options_a)
    manifest_b = build_package_manifest(result, options_b)
    assert manifest_a["font_pack_hash"] != manifest_b["font_pack_hash"]
    assert manifest_a["font_pack_hash"] == compute_font_pack_hash(options_a.font_policy)


def test_write_package_manifest_writes_sidecar_json(tmp_path):
    ast = _load("minimal_paragraph")
    output_path = tmp_path / "out.docx"
    options = RendererOptions()
    result = compile_docx(ast, output_path, options)
    manifest_path = write_package_manifest(result, output_path, options)
    assert manifest_path == output_path.with_suffix(".docx.manifest.json")
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["ast_hash"] == result.ast_hash
    assert _REQUIRED_MANIFEST_FIELDS <= set(data)


def test_compile_docx_emits_manifest_when_option_set(tmp_path):
    ast = _load("minimal_paragraph")
    output_path = tmp_path / "out.docx"
    compile_docx(ast, output_path, RendererOptions(emit_manifest=True))
    assert output_path.with_suffix(".docx.manifest.json").exists()


def test_compile_docx_does_not_emit_manifest_by_default(tmp_path):
    ast = _load("minimal_paragraph")
    output_path = tmp_path / "out.docx"
    compile_docx(ast, output_path, RendererOptions())
    assert not output_path.with_suffix(".docx.manifest.json").exists()


# --- determinism: subprocess + randomized hash seed -----------------------------


_SUBPROCESS_COMPILE_SCRIPT = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from unflatten.renderer.compiler import compile_docx
from unflatten.renderer.api import RendererOptions
import json

ast = json.load(open(sys.argv[2]))
out = Path(sys.argv[3])
compile_docx(ast, out, RendererOptions())
print(out.read_bytes().hex())
"""


def _compile_in_subprocess(ast_path: Path, out_path: Path, hash_seed: str) -> bytes:
    src_root = Path(__file__).parent.parent.parent / "src"
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    args = [
        sys.executable,
        "-c",
        _SUBPROCESS_COMPILE_SCRIPT,
        str(src_root),
        str(ast_path),
        str(out_path),
    ]
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=60,
    )
    return bytes.fromhex(result.stdout.strip())


def test_compile_is_byte_deterministic_across_subprocesses_and_hash_seeds(tmp_path):
    ast_path = VALID_DIR / "mixed_run_styles.json"
    bytes_seed0 = _compile_in_subprocess(ast_path, tmp_path / "a.docx", "0")
    bytes_seed42 = _compile_in_subprocess(ast_path, tmp_path / "b.docx", "42")
    assert bytes_seed0 == bytes_seed42


def test_compile_is_byte_deterministic_over_ten_in_process_runs(tmp_path):
    ast = _load("table_with_colspan_rowspan")
    hashes = set()
    for i in range(10):
        out = tmp_path / f"run_{i}.docx"
        compile_docx(ast, out, RendererOptions())
        hashes.add(out.read_bytes())
    assert len(hashes) == 1

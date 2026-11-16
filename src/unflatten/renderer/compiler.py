"""Compile pipeline: Canonical AST -> DOCX (RFC002 stages S0-S10, R0/R1 scope).

`compile_docx` is the sole public entry point. It never guesses or
repairs an invalid AST (S0/S1 reuse `unflatten.ast.canonicalizer`
verbatim); every node that R0/R1 cannot render yet is reported as an
`unsupported_with_error` Issue, never silently dropped.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from lxml import etree

from unflatten.ast.canonicalizer import CanonicalizationError, canonicalize
from unflatten.ast.hashing import ast_hash as compute_ast_hash
from unflatten.ast.issues import Issue
from unflatten.renderer.api import RendererOptions, RendererResult
from unflatten.renderer.limits import check_limits
from unflatten.renderer.manifest import write_package_manifest
from unflatten.renderer.ooxml.context import DOCUMENT_PART
from unflatten.renderer.ooxml.document_builder import build_document
from unflatten.renderer.ooxml.numbering import build_numbering_xml
from unflatten.renderer.ooxml.package_parts import (
    CONTENT_TYPE_CORE_PROPERTIES,
    CONTENT_TYPE_DOCUMENT,
    CONTENT_TYPE_FONT_TABLE,
    CONTENT_TYPE_NUMBERING,
    CONTENT_TYPE_SETTINGS,
    CONTENT_TYPE_STYLES,
    build_content_types,
    build_core_properties,
    build_font_table,
    build_root_rels,
    build_settings,
    build_styles,
    has_mappable_core_properties,
)
from unflatten.renderer.ooxml.story_builder import build_notes_root, build_story_root
from unflatten.renderer.opc import PackagePart, build_package
from unflatten.renderer.package_plan import PackagePlan, plan_package
from unflatten.renderer.resources import ResourceProvider
from unflatten.renderer.version import RENDERER_VERSION
from unflatten.renderer.xml_writer import serialize
from unflatten.renderer.xsd_validation import validate_all_parts

_UNSUPPORTED_PREFIX = "UNSUPPORTED"


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def compile_docx(
    ast: dict[str, Any],
    output_path: Path,
    options: RendererOptions,
    resources: ResourceProvider | None = None,
) -> RendererResult:
    """Compile a strict Canonical AST document into a deterministic `.docx`.

    Rejects an invalid AST before any package part is built (S1); a
    strict-mode error anywhere in S2-S9 aborts the compile without
    writing `output_path`. Determinism: same `ast` + `options` +
    `resources` bytes yields byte-identical output — no wall-clock
    time, no random IDs. `resources` is only required when the AST
    contains `inline_image` nodes (RFC002 stage S3).
    """
    # Checked on the *raw* input, before `canonicalize` — the
    # canonicalizer itself recurses over the tree (id renumbering, run
    # merging, ...), so a pathologically deep AST must be rejected here
    # first, or canonicalize() would hit Python's recursion limit
    # before this guard ever ran (RFC002 "Performance").
    limit_issues = check_limits(
        ast if isinstance(ast, dict) else {},
        max_nodes=options.max_nodes,
        max_tree_depth=options.max_tree_depth,
        max_text_chars=options.max_text_chars,
        max_table_cells=options.max_table_cells,
    )
    if limit_issues:
        return _failed(limit_issues, ast_hash=None)

    try:
        canonical = canonicalize(ast)
    except CanonicalizationError as exc:
        return _failed(exc.issues, ast_hash=None)

    doc_ast_hash = compute_ast_hash(canonical)

    plan = plan_package(canonical, options, resources)
    ctx = plan.ctx
    resource_blocking = [i for i in plan.issues if i.severity in ("error", "fatal")]
    if resource_blocking and options.strict:
        return _failed(plan.issues, ast_hash=doc_ast_hash)

    build_result = build_document(canonical, options.font_policy, ctx)
    all_issues = [*plan.issues, *build_result.issues]
    if build_result.fatal or build_result.root is None:
        return _failed(all_issues, ast_hash=doc_ast_hash)

    story_roots, story_issues = _build_story_roots(plan, options, build_result.used_fonts)
    all_issues.extend(story_issues)

    blocking = [i for i in all_issues if i.severity in ("error", "fatal")]
    if blocking and options.strict:
        return _failed(all_issues, ast_hash=doc_ast_hash)

    parts = _build_parts(
        build_result.root,
        story_roots,
        build_result.used_fonts,
        options,
        plan,
        canonical.get("metadata", {}),
    )

    if options.validate_package:
        validation_issue = _validate_package(parts)
        if validation_issue is not None:
            return _failed([*all_issues, validation_issue], ast_hash=doc_ast_hash)

    if options.validate_xsd:
        xsd_issues = validate_all_parts({part.name: part.data for part in parts})
        if xsd_issues and options.strict:
            return _failed([*all_issues, *xsd_issues], ast_hash=doc_ast_hash)
        all_issues.extend(xsd_issues)

    package_bytes = build_package(parts)
    output_path.write_bytes(package_bytes)

    status = "success_with_warnings" if all_issues else "success"
    approximations = [i for i in all_issues if i.code == "FONT_FALLBACK"]
    unsupported = [i for i in all_issues if i.code.startswith(_UNSUPPORTED_PREFIX)]
    part_hashes = {part.name: _sha256(part.data) for part in parts}

    result = RendererResult(
        status=status,
        output_path=output_path,
        ast_hash=doc_ast_hash,
        package_hash=_sha256(package_bytes),
        renderer_version=RENDERER_VERSION,
        profile=options.profile,
        ast_version=canonical.get("ast_version"),
        issues=all_issues,
        approximations=approximations,
        unsupported=unsupported,
        parts=part_hashes,
        metrics={
            "used_fonts": sorted(build_result.used_fonts),
            "technical_cells": ctx.technical_cells,
            "technical_paragraphs": ctx.technical_paragraphs,
            "rendered_node_ids": sorted(ctx.rendered_node_ids),
            "technical_boundary_paragraph_positions": sorted(
                ctx.technical_boundary_paragraph_positions
            ),
            "technical_paragraph_paths": {
                part: sorted(paths) for part, paths in sorted(ctx.technical_paragraph_paths.items())
            },
        },
    )
    if options.emit_manifest:
        write_package_manifest(result, output_path, options)
    return result


def _build_story_roots(
    plan: PackagePlan, options: RendererOptions, used_fonts: set[str]
) -> tuple[dict[str, bytes], list[Issue]]:
    """Serialize every header/footer and notes part, in plan order.

    Each part is built with `ctx.current_part` set to its own name, so
    any synthesized paragraph is recorded against the right part —
    technical-paragraph positions are per part, exactly as relationship
    ids are.
    """
    ctx = plan.ctx
    roots: dict[str, bytes] = {}
    issues: list[Issue] = []
    for story in plan.story_parts:
        ctx.current_part = story.part_name
        root, story_issues = build_story_root(story, options.font_policy, used_fonts, ctx)
        roots[story.part_name] = serialize(root)
        issues.extend(story_issues)
    for notes in plan.notes_parts:
        ctx.current_part = notes.part_name
        root, notes_issues = build_notes_root(notes, options.font_policy, used_fonts, ctx)
        roots[notes.part_name] = serialize(root)
        issues.extend(notes_issues)
    ctx.current_part = DOCUMENT_PART
    return roots, issues


def _failed(issues: list[Issue], *, ast_hash: str | None) -> RendererResult:
    return RendererResult(
        status="failed",
        output_path=None,
        ast_hash=ast_hash,
        package_hash=None,
        renderer_version=RENDERER_VERSION,
        profile="word-core-v0",
        issues=issues,
        unsupported=[i for i in issues if i.code.startswith(_UNSUPPORTED_PREFIX)],
    )


def _build_parts(
    document_root: etree._Element,
    story_roots: dict[str, bytes],
    used_fonts: set[str],
    options: RendererOptions,
    plan: PackagePlan,
    metadata: dict,
) -> list[PackagePart]:
    ctx = plan.ctx
    overrides = {
        "/word/document.xml": CONTENT_TYPE_DOCUMENT,
        "/word/styles.xml": CONTENT_TYPE_STYLES,
        "/word/settings.xml": CONTENT_TYPE_SETTINGS,
        "/word/fontTable.xml": CONTENT_TYPE_FONT_TABLE,
    }
    emit_core_properties = has_mappable_core_properties(metadata)
    parts = [
        PackagePart(
            "_rels/.rels", build_root_rels(include_core_properties=emit_core_properties)
        ),
        PackagePart("word/document.xml", serialize(document_root)),
        PackagePart(
            "word/_rels/document.xml.rels", plan.relationships[DOCUMENT_PART].to_bytes()
        ),
        PackagePart("word/styles.xml", build_styles()),
        PackagePart(
            "word/settings.xml",
            build_settings(
                options.compatibility_mode,
                even_and_odd_headers=plan.even_and_odd_headers,
                update_fields=plan.has_fields,
            ),
        ),
        PackagePart("word/fontTable.xml", build_font_table(sorted(used_fonts))),
    ]
    if not ctx.numbering.is_empty:
        overrides["/word/numbering.xml"] = CONTENT_TYPE_NUMBERING
        parts.append(PackagePart("word/numbering.xml", build_numbering_xml(ctx.numbering)))
    if emit_core_properties:
        overrides["/docProps/core.xml"] = CONTENT_TYPE_CORE_PROPERTIES
        parts.append(PackagePart("docProps/core.xml", build_core_properties(metadata)))

    # Header/footer/notes parts, each with its own `_rels` — omitted
    # when that part references nothing, matching how Word writes them.
    for story in [*plan.story_parts, *plan.notes_parts]:
        overrides[f"/{story.part_name}"] = story.content_type
        parts.append(PackagePart(story.part_name, story_roots[story.part_name]))
        rels = plan.relationships[story.part_name]
        if len(rels):
            parts.append(PackagePart(rels.rels_part_name, rels.to_bytes()))

    parts.extend(plan.media_parts)
    content_types = build_content_types(overrides, plan.extra_defaults)
    parts.insert(0, PackagePart("[Content_Types].xml", content_types))
    return parts


def _validate_package(parts: list[PackagePart]) -> Issue | None:
    """S10: every XML part must be well-formed. A parse failure here is a
    renderer bug, not an AST problem, but is still surfaced as an Issue
    rather than raising, so callers get a structured result.
    """
    for part in parts:
        if not part.name.endswith(".xml"):
            continue
        try:
            etree.fromstring(part.data)
        except etree.XMLSyntaxError as exc:
            return Issue(
                code="PACKAGE_INVALID",
                severity="fatal",
                node_id=None,
                path=f"/{part.name}",
                message=f"generated part is not well-formed XML: {exc}",
                details={"part": part.name},
            )
    return None

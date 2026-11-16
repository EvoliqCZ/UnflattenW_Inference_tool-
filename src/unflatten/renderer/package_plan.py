"""Whole-package pre-scan (RFC002 stages S3 "resource resolution" and
S6 "build registries"), extended for `word-visual-v1` (DEC-003).

Everything that cannot be decided while walking one part is decided
here: which package parts exist, which relationship id each reference
gets *in the part that declares it*, which `wp:docPr` id each image
gets across the whole package, which numeric id each footnote/endnote
gets, and which node ids need a bookmark.

The relationship trap this module exists to avoid: `rel_id_by_sha256`
used to be one document-wide sha256 -> rel-id map, which is only
correct while every image lives in `document.xml`. 24.7 % of the corpus
has a header part whose own rels reference an image, so ids are
allocated per `PartRelationships` here and an `ImageRef` is resolved
against the table of the part its node lives in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from unflatten.ast.issues import Issue
from unflatten.renderer.api import RendererOptions
from unflatten.renderer.ooxml.context import DOCUMENT_PART, BuildContext
from unflatten.renderer.ooxml.drawing_builder import build_image_refs, collect_inline_image_nodes
from unflatten.renderer.ooxml.numbering import build_numbering_registry
from unflatten.renderer.ooxml.package_parts import new_document_relationships
from unflatten.renderer.ooxml.relationships import (
    RELTYPE_HYPERLINK,
    RELTYPE_IMAGE,
    RELTYPE_NUMBERING,
    PartRelationships,
)
from unflatten.renderer.ooxml.story_builder import (
    FIRST_NOTE_ID,
    NotesPart,
    StoryPart,
    plan_notes_parts,
    plan_story_parts,
)
from unflatten.renderer.opc import PackagePart
from unflatten.renderer.resources import MIME_EXTENSION, ResourceProvider, resolve_resource
from unflatten.renderer.units import mpt_to_emu


@dataclass
class PackagePlan:
    ctx: BuildContext
    issues: list[Issue] = field(default_factory=list)
    media_parts: list[PackagePart] = field(default_factory=list)
    extra_defaults: dict[str, str] = field(default_factory=dict)
    story_parts: list[StoryPart] = field(default_factory=list)
    notes_parts: list[NotesPart] = field(default_factory=list)
    relationships: dict[str, PartRelationships] = field(default_factory=dict)
    even_and_odd_headers: bool = False
    has_fields: bool = False


def _story_blocks(story: StoryPart | NotesPart) -> list[dict[str, Any]]:
    if isinstance(story, StoryPart):
        return list(story.node.get("children", []))
    return [child for note in story.notes for child in note.get("children", [])]


def plan_package(
    canonical: dict[str, Any], options: RendererOptions, resources: ResourceProvider | None
) -> PackagePlan:
    """Build the `BuildContext` and every per-part relationship table."""
    sections = canonical.get("sections", [])
    body_blocks = [child for section in sections for child in section.get("children", [])]
    story_parts = plan_story_parts(sections)
    notes_parts = plan_notes_parts(canonical)

    # part name -> the blocks rendered into it, in package part order.
    blocks_by_part: dict[str, list[dict[str, Any]]] = {DOCUMENT_PART: body_blocks}
    for part in [*story_parts, *notes_parts]:
        blocks_by_part[part.part_name] = _story_blocks(part)

    all_blocks = [block for blocks in blocks_by_part.values() for block in blocks]
    numbering = build_numbering_registry(all_blocks)

    image_nodes_by_part = {
        part_name: collect_inline_image_nodes(blocks)
        for part_name, blocks in blocks_by_part.items()
    }
    all_image_nodes = [node for nodes in image_nodes_by_part.values() for node in nodes]

    issues, sha256_by_resource_id, media_by_sha256 = _resolve_resources(
        canonical, all_image_nodes, resources
    )

    relationships: dict[str, PartRelationships] = {DOCUMENT_PART: new_document_relationships()}
    for part in [*story_parts, *notes_parts]:
        relationships[part.part_name] = PartRelationships(part.part_name)

    document_rels = relationships[DOCUMENT_PART]
    if not numbering.is_empty:
        document_rels.add(RELTYPE_NUMBERING, "numbering.xml")

    media_targets_by_sha256 = {
        sha256: f"media/{sha256[:16]}.{extension}"
        for sha256, (extension, _data) in media_by_sha256.items()
    }
    ctx = BuildContext(font_policy=options.font_policy, numbering=numbering, images={})

    for part_name, image_nodes in image_nodes_by_part.items():
        rel_id_by_sha256 = _register_media(
            relationships[part_name],
            image_nodes,
            sha256_by_resource_id,
            media_targets_by_sha256,
            part_name,
        )
        refs, image_issues = build_image_refs(
            image_nodes,
            rel_id_by_sha256,
            sha256_by_resource_id,
            docpr_start=len(ctx.images) + 1,
        )
        ctx.images.update(refs)
        issues.extend(image_issues)

    text_frames = [node for node in _preorder(all_blocks) if node.get("type") == "text_frame"]
    ctx.text_frame_docpr_ids = {
        node["id"]: index
        for index, node in enumerate(text_frames, start=len(ctx.images) + 1)
    }
    ctx.text_frame_width_emu = _text_frame_widths(sections, story_parts, notes_parts)

    # Header/footer relationships and their sectPr references: declared
    # by document.xml, so they are added to *its* table only.
    for story in story_parts:
        rel_id = document_rels.add(story.rel_type, story.target)
        ctx.story_refs_by_section.setdefault(story.section_index, []).append(
            (story.reference_local, story.kind, rel_id)
        )
    for notes in notes_parts:
        document_rels.add(notes.rel_type, notes.target)

    for part_name, blocks in blocks_by_part.items():
        _register_hyperlinks(relationships[part_name], blocks, ctx)
    issues.extend(_register_bookmarks(blocks_by_part, ctx))
    _register_notes(notes_parts, ctx)

    extension_to_mime = {ext: mime for mime, ext in MIME_EXTENSION.items()}
    return PackagePlan(
        ctx=ctx,
        issues=issues,
        media_parts=[
            PackagePart(f"word/{target}", media_by_sha256[sha256][1])
            for sha256, target in media_targets_by_sha256.items()
        ],
        extra_defaults={
            extension: extension_to_mime[extension]
            for extension, _data in media_by_sha256.values()
        },
        story_parts=story_parts,
        notes_parts=notes_parts,
        relationships=relationships,
        even_and_odd_headers=any(story.kind == "even" for story in story_parts),
        has_fields=any(node.get("type") == "field" for node in _preorder(all_blocks)),
    )


def _resolve_resources(
    canonical: dict[str, Any],
    image_nodes: list[dict[str, Any]],
    resources: ResourceProvider | None,
) -> tuple[list[Issue], dict[str, str], dict[str, tuple[str, bytes]]]:
    """Resolve every referenced resource once for the whole package.

    `resources` is document-root and deduped by sha256 already, so a
    logo shared by a header and the body is stored as one media part
    no matter how many parts reference it.
    """
    issues: list[Issue] = []
    sha256_by_resource_id: dict[str, str] = {}
    media_by_sha256: dict[str, tuple[str, bytes]] = {}
    resources_by_id = {r["id"]: r for r in canonical.get("resources", [])}

    if image_nodes and resources is None:
        issues.append(
            Issue(
                code="RESOURCE_PROVIDER_MISSING",
                severity="error",
                node_id=None,
                path="/resources",
                message="AST references inline_image nodes but no ResourceProvider was given.",
                details={"image_count": len(image_nodes)},
            )
        )
        return issues, sha256_by_resource_id, media_by_sha256
    if resources is None:
        return issues, sha256_by_resource_id, media_by_sha256

    for index, resource_id in enumerate(sorted({n["resource_ref"] for n in image_nodes})):
        entry = resources_by_id.get(resource_id)
        if entry is None:
            issues.append(
                Issue(
                    code="RESOURCE_REF_UNRESOLVED",
                    severity="error",
                    node_id=None,
                    path=f"/resources/{index}",
                    message=f"resource '{resource_id}' referenced by an inline_image is not "
                    "declared in the AST resources array.",
                    details={"resource_ref": resource_id},
                )
            )
            continue
        resolved, issue = resolve_resource(entry, resources, f"/resources/{index}")
        if issue is not None:
            issues.append(issue)
            continue
        sha256_by_resource_id[resource_id] = resolved.sha256
        media_by_sha256[resolved.sha256] = (resolved.extension, resolved.data)
    return issues, sha256_by_resource_id, media_by_sha256


def _register_media(
    rels: PartRelationships,
    image_nodes: list[dict[str, Any]],
    sha256_by_resource_id: dict[str, str],
    media_targets_by_sha256: dict[str, str],
    part_name: str,
) -> dict[str, str]:
    """Add an image relationship to `rels` for every distinct sha256 the
    part actually uses, sorted by target so the ids never depend on
    traversal order.
    """
    used = {
        sha256_by_resource_id[node["resource_ref"]]
        for node in image_nodes
        if node["resource_ref"] in sha256_by_resource_id
    }
    targets = sorted((media_targets_by_sha256[sha256], sha256) for sha256 in used)
    # A media target is relative to the declaring part's own folder;
    # every part this renderer emits lives directly in `word/`.
    prefix = "" if part_name.startswith("word/") else "word/"
    return {
        sha256: rels.add(RELTYPE_IMAGE, f"{prefix}{target}") for target, sha256 in targets
    }


def _register_hyperlinks(
    rels: PartRelationships, blocks: list[dict[str, Any]], ctx: BuildContext
) -> None:
    """External hyperlink targets become `TargetMode="External"`
    relationships of the part that contains the link. An
    `internal_anchor` gets none: it renders as `w:anchor`.
    """
    for node in _hyperlinks(blocks):
        target = node["target"]
        if target["kind"] == "internal_anchor":
            continue
        ctx.hyperlink_rel_ids[node["id"]] = rels.add(
            RELTYPE_HYPERLINK, target["uri"], external=True
        )


def _register_bookmarks(
    blocks_by_part: dict[str, list[dict[str, Any]]], ctx: BuildContext
) -> list[Issue]:
    """Give every `internal_anchor` target a `w:bookmarkStart`/`End`.

    RFC001 ERRATA-001 E5 fixes the anchor convention as `#<node_id>`
    and the semantic validator already guarantees the target resolves,
    so the bookmark name is the node id verbatim. OOXML anchors a
    bookmark to a paragraph; a target that is not a paragraph gets no
    bookmark, and the resulting dangling `w:anchor` is reported as an
    approximation rather than emitted silently.
    """
    issues: list[Issue] = []
    paragraph_ids = {
        node["id"]
        for blocks in blocks_by_part.values()
        for node in _preorder(blocks)
        if node.get("type") == "paragraph" and node.get("id")
    }
    for blocks in blocks_by_part.values():
        for node in _hyperlinks(blocks):
            target = node["target"]
            if target["kind"] != "internal_anchor":
                continue
            anchor = target["uri"].removeprefix("#")
            if anchor not in paragraph_ids:
                issues.append(
                    Issue(
                        code="APPROXIMATED_ANCHOR_TARGET_NOT_BOOKMARKABLE",
                        severity="warning",
                        node_id=node.get("id"),
                        path="/target/uri",
                        message=f"internal_anchor '#{anchor}' resolves to a node that is not a "
                        "paragraph; the link is emitted without a bookmark at its target.",
                        details={"anchor": anchor},
                    )
                )
                continue
            if anchor not in ctx.bookmark_targets:
                ctx.bookmark_targets[anchor] = len(ctx.bookmark_targets) + 1
    return issues


def _register_notes(notes_parts: list[NotesPart], ctx: BuildContext) -> None:
    """Assign the `w:id` of each footnote/endnote.

    Ids restart at 1 in each part because `footnotes.xml` and
    `endnotes.xml` are separate id spaces; -1 and 0 are reserved for
    the separator notes.
    """
    for part in notes_parts:
        for offset, note in enumerate(part.notes):
            ctx.note_numeric_ids[note["id"]] = FIRST_NOTE_ID + offset
            ctx.note_kinds[note["id"]] = part.item_local


def _hyperlinks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [node for node in _preorder(blocks) if node.get("type") == "hyperlink"]


def _section_content_width_emu(section: dict[str, Any] | None) -> int | None:
    """Usable width (page minus left/right margin, EMU) for one section,
    or `None` when the section's own geometry cannot answer that (never
    reconstructed from anything outside the AST's own `page`/`margins`).
    """
    if section is None:
        return None
    page = section.get("page") or {}
    margins = section.get("margins") or {}
    width_mpt = page.get("width_mpt")
    left_mpt = margins.get("left_mpt")
    right_mpt = margins.get("right_mpt")
    if width_mpt is None or left_mpt is None or right_mpt is None:
        return None
    content_mpt = width_mpt - left_mpt - right_mpt
    if content_mpt <= 0:
        return None
    return mpt_to_emu(content_mpt)


def _text_frame_widths(
    sections: list[dict[str, Any]],
    story_parts: list[StoryPart],
    notes_parts: list[NotesPart],
) -> dict[str, int]:
    """Section-owned usable width (EMU), keyed by `text_frame` node id.

    `drawing_builder.append_text_frame_drawing` only consults this for
    alignment-positioned anchors, where a wrong width visibly misplaces
    the frame (see its docstring). Footnotes/endnotes are not owned by
    one section; they fall back to the first section's width, the same
    single-section assumption the renderer already makes for page
    geometry outside `w:sectPr`.
    """
    widths: dict[str, int] = {}
    for section in sections:
        width = _section_content_width_emu(section)
        if width is None:
            continue
        for node in _preorder(section.get("children", [])):
            if node.get("type") == "text_frame":
                widths[node["id"]] = width
    for story in story_parts:
        section = (
            sections[story.section_index] if story.section_index < len(sections) else None
        )
        width = _section_content_width_emu(section)
        if width is None:
            continue
        for node in _preorder(story.node.get("children", [])):
            if node.get("type") == "text_frame":
                widths[node["id"]] = width
    fallback_width = _section_content_width_emu(sections[0]) if sections else None
    if fallback_width is not None:
        for notes in notes_parts:
            for note in notes.notes:
                for node in _preorder(note.get("children", [])):
                    if node.get("type") == "text_frame":
                        widths[node["id"]] = fallback_width
    return widths


def _preorder(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every node under `nodes`, in preorder, through every container
    kind a paragraph can hide behind.
    """
    found: list[dict[str, Any]] = []
    for node in nodes:
        found.append(node)
        for key in ("children", "items", "rows", "cells"):
            children = node.get(key)
            if isinstance(children, list):
                found.extend(_preorder(children))
    return found

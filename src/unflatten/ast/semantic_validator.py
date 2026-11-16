"""Semantic AST validation pass (RFC001 §15, §49.2).

Covers rules that JSON Schema cannot express: parent-child taxonomy,
unique IDs, cross-reference integrity, table grid consistency, mutually
exclusive properties and numeric sanity checks. Never raises on an
invalid document — every rule appends an Issue instead.
"""

from __future__ import annotations

from typing import Any

from unflatten.ast.issues import Issue

_BLOCK_TYPES = {"paragraph", "list", "table", "text_frame", "page_break", "column_break"}
_INLINE_TYPES = {"run", "hyperlink", "tab", "line_break", "inline_image",
                 "floating_image", "note_ref", "field"}
_HYPERLINK_CHILD_TYPES = {"run", "tab", "line_break", "inline_image"}
# Story content (header/footer/footnote): the block set minus the two
# breaks, which are admissible syntax and nonsense inside a story.
_STORY_BLOCK_TYPES = _BLOCK_TYPES - {"page_break", "column_break"}

# Parent-child matrix (RFC001 §15): node type -> allowed child node type set,
# or the sentinel "block" / "story_block" / "inline" / "hyperlink" for the
# taxonomies above. Stories are NOT in `_BLOCK_TYPES`, so "a header inside a
# header" is unreachable by the same mechanism that stops a cell containing
# a section — no extra rule.
_PARENT_CHILD: dict[str, set[str] | str] = {
    "document": {"section"},
    "section": "block",
    "paragraph": "inline",
    "list_item": "block",
    "cell": "block",
    "hyperlink": "hyperlink",
    "list": {"list_item"},
    "table": {"row"},
    "row": {"cell"},
    "header": "story_block",
    "footer": "story_block",
    "footnote": "story_block",
    "endnote": "story_block",
    "text_frame": "story_block",
}

# The `word-visual-v1` node types (RFC001 §42.2, DEC-003 §1). Under
# `word-core-v0` their presence is an error, not a silent downgrade.
# `hyperlink` is on this list even though it has been in the schema since
# v0.1: §42.2 assigns it to this profile, and DEC-003 §1 (2026-09-03)
# makes it a first-class member rather than the anomaly it was.
_V1_NODE_TYPES = {
    "header", "footer", "footnote", "endnote", "floating_image", "note_ref",
    "hyperlink", "field",
}
_VISUAL_PROFILES = {"word-visual-v1", "word-visual-v1.1"}
_TEXT_FRAME_PROFILE = "word-visual-v1.1"

# Root keys holding note stories, in canonical order.
_NOTE_KEYS: tuple[tuple[str, str], ...] = (("footnotes", "footnote"), ("endnotes", "endnote"))


def validate_semantics(doc: Any) -> list[Issue]:
    """Validate `doc` against RFC001 semantic rules.

    Assumes `doc` already passed schema validation shape-wise; degrades
    gracefully (skips checks it cannot perform) on malformed input rather
    than raising.
    """
    issues: list[Issue] = []
    if not isinstance(doc, dict):
        return issues

    node_ids_seen: dict[str, str] = {}  # id -> path of first occurrence
    nodes_by_id: dict[str, dict] = {}
    resource_ids_seen: dict[str, str] = {}
    resource_kinds: dict[str, str | None] = {}
    ref_edges: list[tuple[str, str, str]] = []  # (from_id, to_id, path)
    note_refs: list[tuple[str, str]] = []  # (note_id, path of the note_ref)
    profile = doc.get("profile")
    v1_allowed = profile in _VISUAL_PROFILES
    text_frame_allowed = profile == _TEXT_FRAME_PROFILE

    resources = doc.get("resources", [])
    if isinstance(resources, list):
        for i, res in enumerate(resources):
            if not isinstance(res, dict):
                continue
            path = f"/resources/{i}"
            rid = res.get("id")
            if isinstance(rid, str):
                if rid in resource_ids_seen:
                    issues.append(
                        Issue(
                            code="DUPLICATE_RESOURCE_ID",
                            severity="error",
                            node_id=rid,
                            path=path,
                            message=f"Resource id '{rid}' is used more than once.",
                        )
                    )
                else:
                    resource_ids_seen[rid] = path
                    resource_kinds[rid] = res.get("kind")

    walk_kwargs = {
        "issues": issues,
        "node_ids_seen": node_ids_seen,
        "nodes_by_id": nodes_by_id,
        "resource_ids_seen": resource_ids_seen,
        "resource_kinds": resource_kinds,
        "ref_edges": ref_edges,
        "note_refs": note_refs,
        "v1_allowed": v1_allowed,
        "text_frame_allowed": text_frame_allowed,
    }

    sections = doc.get("sections", [])
    if isinstance(sections, list):
        for i, section in enumerate(sections):
            _walk(section, f"/sections/{i}", parent_type="document", **walk_kwargs)

    for key, _node_type in _NOTE_KEYS:
        notes = doc.get(key, [])
        if isinstance(notes, list):
            for i, note in enumerate(notes):
                _walk(
                    note,
                    f"/{key}/{i}",
                    parent_type=None,
                    inside_footnote=True,
                    **walk_kwargs,
                )

    _check_reference_cycles(ref_edges, issues)
    _check_broken_anchors(ref_edges, nodes_by_id, issues)
    _check_note_refs(note_refs, doc, nodes_by_id, issues)

    return issues


def _walk(
    node: Any,
    path: str,
    parent_type: str | None,
    issues: list[Issue],
    node_ids_seen: dict[str, str],
    nodes_by_id: dict[str, dict],
    resource_ids_seen: dict[str, str],
    resource_kinds: dict[str, str | None],
    ref_edges: list[tuple[str, str, str]],
    note_refs: list[tuple[str, str]],
    v1_allowed: bool,
    text_frame_allowed: bool,
    inside_table: bool = False,
    inside_footnote: bool = False,
    inside_text_frame: bool = False,
) -> None:
    if not isinstance(node, dict):
        return

    node_type = node.get("type")
    if node_type in _V1_NODE_TYPES and not v1_allowed:
        issues.append(
            Issue(
                code="V1_FEATURE_IN_V0_PROFILE",
                severity="error",
                node_id=node.get("id"),
                path=path,
                message=f"Node of type '{node_type}' requires profile 'word-visual-v1'.",
            )
        )
    if node_type == "text_frame" and not text_frame_allowed:
        issues.append(
            Issue(
                code="TEXT_FRAME_REQUIRES_VISUAL_V1_1",
                severity="error",
                node_id=node.get("id"),
                path=path,
                message="Node of type 'text_frame' requires profile 'word-visual-v1.1'.",
            )
        )
    if node_type == "text_frame" and inside_text_frame:
        issues.append(
            Issue(
                code="NESTED_TEXT_FRAME_UNSUPPORTED",
                severity="error",
                node_id=node.get("id"),
                path=path,
                message="A text_frame cannot be nested inside another text_frame.",
            )
        )
    if node_type == "table" and inside_table and not v1_allowed:
        # word-core-v0 forbids nested tables (RFC001 §42.1: "bez nested
        # tables v training profile, hoci schema ich môže povoliť
        # neskôr"); the decode grammar and the renderer both enforce
        # this — the semantic validator must agree so a nested-table AST
        # can never be TRAIN_READY yet grammar-unreachable.
        # word-visual-v1 IS that "neskôr" (§42.2, DEC-003 §1): 495
        # documents are rejected outright today for this alone.
        issues.append(
            Issue(
                code="NESTED_TABLE_UNSUPPORTED",
                severity="error",
                node_id=node.get("id"),
                path=path,
                message="Nested tables are not supported in profile word-core-v0.",
            )
        )
    node_id = node.get("id")

    if isinstance(node_id, str):
        if node_id in node_ids_seen:
            issues.append(
                Issue(
                    code="DUPLICATE_NODE_ID",
                    severity="error",
                    node_id=node_id,
                    path=path,
                    message=f"Node id '{node_id}' is used more than once.",
                )
            )
        else:
            node_ids_seen[node_id] = path
            nodes_by_id[node_id] = node

    _check_parent_child(node_type, parent_type, node, path, issues)

    if node_type == "paragraph":
        _check_paragraph(node, path, issues)
    elif node_type == "run":
        _check_run(node, path, issues)
    elif node_type == "section":
        _check_section(node, path, issues)
    elif node_type == "list":
        _check_list(node, path, issues)
    elif node_type == "table":
        _check_table(node, path, issues)
    elif node_type == "cell":
        _check_cell(node, path, issues)
    elif node_type in ("inline_image", "floating_image"):
        _check_resource_ref(node, path, issues, resource_ids_seen, resource_kinds)
    elif node_type == "hyperlink":
        _check_hyperlink_ref(node, path, ref_edges)
    elif node_type == "note_ref":
        _check_note_ref(node, path, issues, note_refs, inside_footnote)

    for child_path, child, child_parent_type in _children_with_paths(node, path):
        _walk(
            child,
            child_path,
            parent_type=child_parent_type,
            issues=issues,
            node_ids_seen=node_ids_seen,
            nodes_by_id=nodes_by_id,
            resource_ids_seen=resource_ids_seen,
            resource_kinds=resource_kinds,
            ref_edges=ref_edges,
            note_refs=note_refs,
            v1_allowed=v1_allowed,
            text_frame_allowed=text_frame_allowed,
            inside_table=inside_table or node_type == "table",
            inside_footnote=inside_footnote,
            inside_text_frame=inside_text_frame or node_type == "text_frame",
        )


def _children_with_paths(node: dict, path: str) -> list[tuple[str, Any, str | None]]:
    """Children as `(path, child, parent_type_for_the_matrix)` triples.

    A section's header/footer stories hang off their own keys, not off
    `children`, so they are walked with parent type `None`: the §15
    matrix says a section's *children* are block nodes, and a story is
    not one.
    """
    result: list[tuple[str, Any, str | None]] = []
    node_type = node.get("type")
    if node_type == "list":
        for i, item in enumerate(node.get("items", []) or []):
            result.append((f"{path}/items/{i}", item, node_type))
    elif node_type == "table":
        for i, row in enumerate(node.get("rows", []) or []):
            result.append((f"{path}/rows/{i}", row, node_type))
    elif node_type == "row":
        for i, cell in enumerate(node.get("cells", []) or []):
            result.append((f"{path}/cells/{i}", cell, node_type))
    else:
        if node_type == "section":
            for key in ("headers", "footers"):
                for i, story in enumerate(node.get(key) or []):
                    result.append((f"{path}/{key}/{i}", story, None))
        children = node.get("children")
        if isinstance(children, list):
            for i, child in enumerate(children):
                result.append((f"{path}/children/{i}", child, node_type))
    return result


def _check_parent_child(
    node_type: str | None, parent_type: str | None, node: dict, path: str, issues: list[Issue]
) -> None:
    if parent_type is None:
        return
    rule = _PARENT_CHILD.get(parent_type)
    if rule is None:
        return
    if rule == "block":
        allowed = node_type in _BLOCK_TYPES
    elif rule == "story_block":
        allowed = node_type in _STORY_BLOCK_TYPES
    elif rule == "inline":
        allowed = node_type in _INLINE_TYPES
    elif rule == "hyperlink":
        allowed = node_type in _HYPERLINK_CHILD_TYPES
    elif isinstance(rule, set):
        allowed = node_type in rule
    else:
        allowed = True
    if not allowed:
        issues.append(
            Issue(
                code="INVALID_PARENT_CHILD",
                severity="error",
                node_id=node.get("id"),
                path=path,
                message=f"Node of type '{node_type}' is not a valid child of '{parent_type}'.",
            )
        )


def _check_paragraph(node: dict, path: str, issues: list[Issue]) -> None:
    semantic = node.get("semantic") or {}
    if semantic.get("role") == "heading":
        level = semantic.get("heading_level")
        if not isinstance(level, int) or not (1 <= level <= 9):
            issues.append(
                Issue(
                    code="HEADING_LEVEL_MISSING",
                    severity="error",
                    node_id=node.get("id"),
                    path=f"{path}/semantic/heading_level",
                    message="Paragraph role 'heading' requires heading_level 1-9.",
                )
            )

    indent = (node.get("layout") or {}).get("indent") or {}
    first_line = indent.get("first_line_mpt", 0)
    hanging = indent.get("hanging_mpt", 0)
    if _is_nonzero(first_line) and _is_nonzero(hanging):
        issues.append(
            Issue(
                code="INDENT_MUTUAL_EXCLUSION",
                severity="error",
                node_id=node.get("id"),
                path=f"{path}/layout/indent",
                message="first_line_mpt and hanging_mpt must not both be non-zero.",
            )
        )


def _check_run(node: dict, path: str, issues: list[Issue]) -> None:
    style = node.get("style") or {}
    if style.get("superscript") is True and style.get("subscript") is True:
        issues.append(
            Issue(
                code="RUN_SCRIPT_MUTUAL_EXCLUSION",
                severity="error",
                node_id=node.get("id"),
                path=f"{path}/style",
                message="superscript and subscript must not both be true.",
            )
        )
    font_size = style.get("font_size_mpt")
    if isinstance(font_size, (int, float)) and font_size <= 0:
        issues.append(
            Issue(
                code="INVALID_FONT_SIZE",
                severity="error",
                node_id=node.get("id"),
                path=f"{path}/style/font_size_mpt",
                message="font_size_mpt must be > 0.",
            )
        )


def _check_section(node: dict, path: str, issues: list[Issue]) -> None:
    _check_story_kinds(node, path, issues)
    columns = node.get("columns") or {}
    if columns.get("equal_width") is False:
        count = columns.get("count")
        widths = columns.get("widths_mpt")
        if isinstance(count, int) and isinstance(widths, list) and len(widths) != count:
            issues.append(
                Issue(
                    code="COLUMN_WIDTH_COUNT_MISMATCH",
                    severity="error",
                    node_id=node.get("id"),
                    path=f"{path}/columns/widths_mpt",
                    message=f"widths_mpt must contain exactly {count} entries when "
                    "equal_width is false.",
                )
            )


def _check_story_kinds(node: dict, path: str, issues: list[Issue]) -> None:
    """DEC-003 §2.1: a section carries at most one story per Word variant.

    Two `default` headers would make "which header does this page
    render" unanswerable, which is exactly the question header
    supervision is keyed on.
    """
    for key in ("headers", "footers"):
        seen: set[str] = set()
        for i, story in enumerate(node.get(key) or []):
            if not isinstance(story, dict):
                continue
            kind = story.get("kind")
            if not isinstance(kind, str):
                continue
            if kind in seen:
                issues.append(
                    Issue(
                        code="DUPLICATE_STORY_KIND",
                        severity="error",
                        node_id=story.get("id"),
                        path=f"{path}/{key}/{i}",
                        message=f"Section carries more than one '{kind}' entry in {key}.",
                    )
                )
            seen.add(kind)


def _check_note_ref(
    node: dict,
    path: str,
    issues: list[Issue],
    note_refs: list[tuple[str, str]],
    inside_footnote: bool,
) -> None:
    note_id = node.get("note_id")
    if isinstance(note_id, str):
        note_refs.append((note_id, path))
    if inside_footnote:
        issues.append(
            Issue(
                code="NOTE_REF_IN_NOTE",
                severity="error",
                node_id=node.get("id"),
                path=path,
                message="A footnote or endnote body must not contain a note_ref.",
            )
        )


def _check_note_refs(
    note_refs: list[tuple[str, str]],
    doc: dict,
    nodes_by_id: dict[str, dict],
    issues: list[Issue],
) -> None:
    """Every `note_ref` resolves to a note, and every note is referenced.

    An unreferenced footnote body is content Word never renders, so it
    could carry no page evidence — the dataset layer would have to
    impute it.
    """
    referenced: set[str] = set()
    for note_id, path in note_refs:
        referenced.add(note_id)
        target = nodes_by_id.get(note_id)
        if target is None or target.get("type") not in ("footnote", "endnote"):
            issues.append(
                Issue(
                    code="BROKEN_NOTE_REF",
                    severity="error",
                    node_id=None,
                    path=f"{path}/note_id",
                    message=f"note_id '{note_id}' does not resolve to a footnote or endnote.",
                )
            )

    for key, node_type in _NOTE_KEYS:
        notes = doc.get(key)
        if not isinstance(notes, list):
            continue
        for i, note in enumerate(notes):
            if not isinstance(note, dict):
                continue
            note_id = note.get("id")
            if isinstance(note_id, str) and note_id not in referenced:
                issues.append(
                    Issue(
                        code="ORPHAN_FOOTNOTE",
                        severity="error",
                        node_id=note_id,
                        path=f"/{key}/{i}",
                        message=f"{node_type.capitalize()} '{note_id}' is never referenced "
                        "by a note_ref.",
                    )
                )


def _check_list(node: dict, path: str, issues: list[Issue]) -> None:
    """RFC001 ERRATA-002 E6: `custom_text` markers require a non-empty
    literal `text`; every other format forbids `text`.
    """
    marker = node.get("marker")
    if not isinstance(marker, dict):
        return
    marker_format = marker.get("format")
    text = marker.get("text")
    if marker_format == "custom_text":
        if not (isinstance(text, str) and text != ""):
            issues.append(
                Issue(
                    code="MARKER_TEXT_REQUIRED",
                    severity="error",
                    node_id=node.get("id"),
                    path=f"{path}/marker/text",
                    message="marker.text must be a non-empty string when format is "
                    "'custom_text'.",
                )
            )
    elif "text" in marker:
        issues.append(
            Issue(
                code="MARKER_TEXT_FORBIDDEN",
                severity="error",
                node_id=node.get("id"),
                path=f"{path}/marker/text",
                message=f"marker.text is only allowed when format is 'custom_text' "
                f"(got format '{marker_format}').",
            )
        )


def _check_table_width(node: dict, path: str, issues: list[Issue]) -> None:
    """RFC001 ERRATA-002 E7: exactly the field matching `mode` may be set —
    absolute carries `value_mpt`, percent carries `value_ppm`, auto carries
    no value (a legacy `value_mpt: 0` is tolerated on auto).
    """
    width = (node.get("layout") or {}).get("width")
    if not isinstance(width, dict):
        return
    mode = width.get("mode")
    has_mpt = "value_mpt" in width
    has_ppm = "value_ppm" in width
    if mode == "absolute":
        mismatch = not has_mpt or has_ppm
    elif mode == "percent":
        mismatch = not has_ppm or has_mpt
    elif mode == "auto":
        mismatch = has_ppm or (has_mpt and width.get("value_mpt") != 0)
    else:
        return
    if mismatch:
        issues.append(
            Issue(
                code="TABLE_WIDTH_FIELD_MISMATCH",
                severity="error",
                node_id=node.get("id"),
                path=f"{path}/layout/width",
                message=f"table width mode '{mode}' carries the wrong value field: "
                "absolute requires value_mpt, percent requires value_ppm, auto "
                "carries no value.",
            )
        )


def _check_table(node: dict, path: str, issues: list[Issue]) -> None:
    _check_table_width(node, path, issues)
    columns = node.get("columns") or []
    num_columns = len(columns) if isinstance(columns, list) else 0
    # grid_column -> last row index still covered by an earlier cell's
    # row_span. Occupancy from row_span=1 cells must not carry over to
    # the next row band, or every ordinary multi-row table (no spans)
    # would falsely collide on itself row after row.
    active_row_spans: dict[int, int] = {}

    rows = node.get("rows") or []
    for ri, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        row_path = f"{path}/rows/{ri}"
        occupancy: dict[int, str] = {
            gc: "<row_span from earlier row>"
            for gc, until_ri in active_row_spans.items()
            if until_ri >= ri
        }
        for ci, cell in enumerate(row.get("cells") or []):
            if not isinstance(cell, dict):
                continue
            cell_path = f"{row_path}/cells/{ci}"
            grid_column = cell.get("grid_column")
            col_span = cell.get("col_span", 1)
            row_span = cell.get("row_span", 1)

            for span_name, span_value in (("col_span", col_span), ("row_span", row_span)):
                if not isinstance(span_value, int) or span_value < 1:
                    issues.append(
                        Issue(
                            code="INVALID_CELL_SPAN",
                            severity="error",
                            node_id=cell.get("id"),
                            path=f"{cell_path}/{span_name}",
                            message=f"{span_name} must be >= 1.",
                        )
                    )

            if isinstance(row_span, int) and row_span >= 1 and ri + row_span > len(rows):
                issues.append(
                    Issue(
                        code="TABLE_ROW_SPAN_OUT_OF_RANGE",
                        severity="error",
                        node_id=cell.get("id"),
                        path=f"{cell_path}/row_span",
                        message="Cell row_span extends past the table's final row.",
                    )
                )

            if not isinstance(grid_column, int) or not isinstance(col_span, int):
                continue

            if grid_column + col_span > num_columns:
                issues.append(
                    Issue(
                        code="TABLE_GRID_COLUMN_OUT_OF_RANGE",
                        severity="error",
                        node_id=cell.get("id"),
                        path=f"{cell_path}/grid_column",
                        message="Cell grid_column + col_span exceeds the declared column count.",
                    )
                )
                continue

            for gc in range(grid_column, grid_column + col_span):
                if gc in occupancy:
                    issues.append(
                        Issue(
                            code="TABLE_GRID_OVERLAP",
                            severity="error",
                            node_id=cell.get("id"),
                            path=cell_path,
                            message=f"Cell span overlaps an existing cell at grid_column {gc}.",
                        )
                    )
                else:
                    occupancy[gc] = cell_path
                if isinstance(row_span, int) and row_span >= 1:
                    until_ri = ri + row_span - 1
                    active_row_spans[gc] = max(active_row_spans.get(gc, -1), until_ri)

        # Gap check is per logical row band: every grid_column must be
        # covered by exactly one cell, counting columns still covered by
        # an earlier row's row_span (tracked via active_row_spans). This
        # runs for every row, not just the last.
        missing = [gc for gc in range(num_columns) if gc not in occupancy]
        if missing:
            issues.append(
                Issue(
                    code="TABLE_GRID_GAP",
                    severity="error",
                    node_id=node.get("id"),
                    path=row_path,
                    message=f"Table grid has uncovered columns: {missing}.",
                )
            )


def _check_cell(node: dict, path: str, issues: list[Issue]) -> None:
    children = node.get("children") or []
    has_paragraph = any(
        isinstance(c, dict) and c.get("type") == "paragraph" for c in children
    )
    if not has_paragraph:
        issues.append(
            Issue(
                code="CELL_MISSING_PARAGRAPH",
                severity="error",
                node_id=node.get("id"),
                path=path,
                message="Cell must contain at least one Paragraph in profile word-core-v0.",
            )
        )
    # RFC004 §18.5 cell rule, second half: the LAST child must be a
    # paragraph too. The decode grammar (unflatten.model.grammar) enforces
    # this on the model side; enforcing it here keeps dataset validity and
    # grammar legality in sync.
    last = children[-1] if children else None
    if (
        has_paragraph
        and isinstance(last, dict)
        and last.get("type") != "paragraph"
    ):
        issues.append(
            Issue(
                code="CELL_LAST_CHILD_NOT_PARAGRAPH",
                severity="error",
                node_id=node.get("id"),
                path=path,
                message=(
                    "Cell's last child must be a Paragraph in profile word-core-v0 "
                    f"(got {last.get('type')!r})."
                ),
            )
        )


def _check_resource_ref(
    node: dict,
    path: str,
    issues: list[Issue],
    resource_ids_seen: dict[str, str],
    resource_kinds: dict[str, str | None],
) -> None:
    ref = node.get("resource_ref")
    if not isinstance(ref, str):
        return
    if ref not in resource_ids_seen:
        issues.append(
            Issue(
                code="BROKEN_RESOURCE_REF",
                severity="error",
                node_id=node.get("id"),
                path=f"{path}/resource_ref",
                message=f"resource_ref '{ref}' does not resolve to a declared resource.",
            )
        )
    elif resource_kinds.get(ref) != "image":
        issues.append(
            Issue(
                code="INCOMPATIBLE_RESOURCE_REF",
                severity="error",
                node_id=node.get("id"),
                path=f"{path}/resource_ref",
                message=f"resource_ref '{ref}' does not resolve to a compatible resource kind.",
            )
        )


def _check_hyperlink_ref(node: dict, path: str, ref_edges: list[tuple[str, str, str]]) -> None:
    target = node.get("target") or {}
    if target.get("kind") != "internal_anchor":
        return
    uri = target.get("uri")
    node_id = node.get("id")
    if isinstance(uri, str) and uri.startswith("#") and isinstance(node_id, str):
        ref_edges.append((node_id, uri[1:], path))


def _check_reference_cycles(
    ref_edges: list[tuple[str, str, str]], issues: list[Issue]
) -> None:
    graph: dict[str, list[str]] = {}
    for src, dst, _path in ref_edges:
        graph.setdefault(src, []).append(dst)

    visiting: set[str] = set()
    visited: set[str] = set()
    reported: set[str] = set()

    def dfs(n: str) -> bool:
        if n in visiting:
            return True
        if n in visited:
            return False
        visiting.add(n)
        for neighbor in graph.get(n, []):
            if dfs(neighbor):
                return True
        visiting.discard(n)
        visited.add(n)
        return False

    for src, _dst, path in ref_edges:
        if src in reported:
            continue
        if dfs(src):
            issues.append(
                Issue(
                    code="REFERENCE_CYCLE",
                    severity="error",
                    node_id=src,
                    path=path,
                    message="Cross-reference graph contains a cycle.",
                )
            )
            reported.add(src)


def _check_broken_anchors(
    ref_edges: list[tuple[str, str, str]], nodes_by_id: dict[str, dict], issues: list[Issue]
) -> None:
    """RFC001 §11.4, ERRATA-001 E5: `#<node_id>` internal anchors MUST
    resolve to an existing node id in the same document.
    """
    for src, dst, path in ref_edges:
        if dst not in nodes_by_id:
            issues.append(
                Issue(
                    code="BROKEN_INTERNAL_ANCHOR",
                    severity="error",
                    node_id=src,
                    path=path,
                    message=f"internal_anchor target '#{dst}' does not resolve to an existing "
                    "node id.",
                )
            )


def _is_nonzero(value: Any) -> bool:
    return isinstance(value, (int, float)) and value != 0

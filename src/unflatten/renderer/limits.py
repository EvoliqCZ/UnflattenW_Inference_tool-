"""Compile-time resource limits (RFC002 "Performance" — `max_nodes`,
`max_tree_depth`, `max_text_chars`, `max_table_cells`).

Checked before any package part is built (between S1 strict validation
and S5 DOCX IR construction) so a pathological AST fails fast with a
structured `Issue` instead of exhausting memory/CPU partway through a
build. Every counter is a single linear walk over the already-validated
canonical AST.
"""

from __future__ import annotations

from typing import Any

from unflatten.ast.issues import Issue


def check_limits(
    ast: dict[str, Any],
    *,
    max_nodes: int,
    max_tree_depth: int,
    max_text_chars: int,
    max_table_cells: int,
) -> list[Issue]:
    """Return one `LIMIT_EXCEEDED_*` `Issue` per violated limit, `[]` if none.

    Iterative (explicit stack), never Python-recursive: a pathologically
    deep-but-narrow AST (depth > Python's default recursion limit, but
    few total nodes) must always come back as a normal
    `LIMIT_EXCEEDED_TREE_DEPTH` `Issue`, never crash the process with
    `RecursionError` before that limit is even evaluated. Intended to
    run on the *raw*, not-yet-canonicalized input dict, since
    `unflatten.ast.canonicalizer.canonicalize` itself recurses and needs
    this same depth guard to run before it, not after.
    """
    node_count = 0
    text_chars = 0
    table_cells = 0
    max_depth_seen = 0

    # Stack entries are (node, depth); depth 1 = a section's direct child.
    # `word-visual-v1` moves content into header/footer stories and into
    # root-level footnotes/endnotes, so those roots are seeded too: a
    # 10 MB header must hit the same limit a 10 MB body does.
    stack: list[tuple[dict[str, Any], int]] = []
    for section in ast.get("sections", []) or []:
        if not isinstance(section, dict):
            continue
        for key in ("children", "headers", "footers"):
            for child in section.get(key, []) or []:
                stack.append((child, 1))
    for key in ("footnotes", "endnotes"):
        for note in ast.get(key, []) or []:
            stack.append((note, 1))

    while stack:
        node, depth = stack.pop()
        if not isinstance(node, dict):
            continue
        node_count += 1
        if depth > max_depth_seen:
            max_depth_seen = depth
        node_type = node.get("type")
        if node_type == "run":
            text_chars += len(node.get("text", ""))
        if node_type == "table":
            for row in node.get("rows", []) or []:
                if not isinstance(row, dict):
                    continue
                for cell in row.get("cells", []) or []:
                    if not isinstance(cell, dict):
                        continue
                    table_cells += 1
                    for child in cell.get("children", []) or []:
                        stack.append((child, depth + 1))
            continue
        for key in ("children", "items"):
            for child in node.get(key, []) or []:
                stack.append((child, depth + 1))

    issues: list[Issue] = []
    if node_count > max_nodes:
        issues.append(
            _limit_issue("LIMIT_EXCEEDED_NODES", "/", node_count, max_nodes, "AST node count")
        )
    if max_depth_seen > max_tree_depth:
        issues.append(
            _limit_issue(
                "LIMIT_EXCEEDED_TREE_DEPTH", "/", max_depth_seen, max_tree_depth, "AST tree depth"
            )
        )
    if text_chars > max_text_chars:
        issues.append(
            _limit_issue(
                "LIMIT_EXCEEDED_TEXT_CHARS", "/", text_chars, max_text_chars, "total run text chars"
            )
        )
    if table_cells > max_table_cells:
        issues.append(
            _limit_issue(
                "LIMIT_EXCEEDED_TABLE_CELLS",
                "/",
                table_cells,
                max_table_cells,
                "total authored table cells",
            )
        )
    return issues


def _limit_issue(code: str, path: str, actual: int, limit: int, description: str) -> Issue:
    return Issue(
        code=code,
        severity="error",
        node_id=None,
        path=path,
        message=f"{description} is {actual}, exceeding the configured limit of {limit}.",
        details={"actual": actual, "limit": limit},
    )

"""RFC004 §53.1-53.2 -- closed `TablePlan` contract and its deterministic
constraint verifier.

This module is standalone: it validates a `TablePlan` document (§53.1's
`table_local_node_id` / `row_local_node_ids` / `columns` / `cells` /
`boundary_claims` / `repeated_header_observations`) against the closed
schema, then checks the §53.2 grid/span/order invariants that the schema
alone cannot express. It never touches the AST, the tree decoder, the
existing `semantic_validator.py` grid checks, or the RATIFIED
`table-geometry.v0` chain policy (`eval/builder_chain/table_geometry.py`)
-- those own their own identities and stay untouched.

The verifier only REFUSES: it returns a list of `Issue`s (RFC001 §50
style, never raises on a semantically invalid plan) and never repairs a
plan itself. Per §53.2's own text, every semantic mismatch is one of
exactly two named RFC004 §42.4 codes: `TABLE_GRID_INVALID` (grid shape --
column count, width-array count, occupancy, span bounds, overlap, gap) or
`TABLE_SPAN_CONFLICT` (TablePlan-vs-tree/typed pointer inconsistency --
row/cell identity, order, uniqueness). A malformed plan that fails the
closed JSON Schema itself raises `TablePlanInvalid` before any semantic
check runs, mirroring `runtime_limits.py`'s schema-first validation.

Cross-fragment merge (§53.3) and repeated-header disambiguation (§53.4)
are out of scope here -- this module only proves ONE plan is internally
consistent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.ast.issues import Issue

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_SCHEMA_NAME = "table_plan"

TABLE_PLAN_SCHEMA_VERSION = "0.1.0"

#: The only two §53.2 semantic mismatch codes (RFC004 §42.4).
TABLE_GRID_INVALID = "TABLE_GRID_INVALID"
TABLE_SPAN_CONFLICT = "TABLE_SPAN_CONFLICT"


class TablePlanInvalid(ValueError):
    """A TablePlan fails its closed schema (RFC004 §53.1)."""


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate_table_plan_schema(plan: Mapping[str, Any]) -> None:
    """Validate `plan` against `table_plan.schema.json`. Raises
    `TablePlanInvalid` on the first schema error."""
    errors = list(_validator().iter_errors(plan))
    if errors:
        raise TablePlanInvalid(
            f"schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )


def verify_table_plan(plan: Mapping[str, Any]) -> list[Issue]:
    """§53.2 deterministic constraint verifier.

    `plan` must already be schema-valid (`validate_table_plan_schema` is
    called first and raises `TablePlanInvalid` on a malformed plan --
    schema shape is a precondition, not a §53.2 finding). Returns a list
    of `Issue`s; an empty list means the plan is consistent. Never
    mutates or repairs `plan`.
    """
    validate_table_plan_schema(plan)
    issues: list[Issue] = []

    row_ids: list[str] = list(plan["row_local_node_ids"])
    row_index_of: dict[str, int] = {}
    for index, row_id in enumerate(row_ids):
        if row_id in row_index_of:
            issues.append(
                Issue(
                    code=TABLE_SPAN_CONFLICT,
                    severity="error",
                    node_id=plan.get("table_local_node_id"),
                    path="/row_local_node_ids",
                    message=f"row_local_node_ids has a duplicate entry: {row_id!r}.",
                )
            )
        else:
            row_index_of[row_id] = index

    grid_column_count = plan["grid_column_count"]
    columns = plan["columns"]
    if len(columns) != grid_column_count:
        issues.append(
            Issue(
                code=TABLE_GRID_INVALID,
                severity="error",
                node_id=plan.get("table_local_node_id"),
                path="/columns",
                message=(
                    f"columns has {len(columns)} entries, "
                    f"grid_column_count declares {grid_column_count}."
                ),
            )
        )
    seen_column_indices: set[int] = set()
    for ci, column in enumerate(columns):
        idx = column["column_index"]
        if idx in seen_column_indices or idx >= grid_column_count:
            issues.append(
                Issue(
                    code=TABLE_GRID_INVALID,
                    severity="error",
                    node_id=plan.get("table_local_node_id"),
                    path=f"/columns/{ci}/column_index",
                    message=f"column_index {idx} is duplicate or out of bounds.",
                )
            )
        seen_column_indices.add(idx)

    cells = plan["cells"]
    seen_cell_ids: set[str] = set()
    # grid_column -> last row index still covered by an earlier cell's
    # row_span, mirroring semantic_validator._check_table's occupancy model.
    active_row_spans: dict[int, int] = {}
    cells_by_row_index: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        resolved_ri = row_index_of.get(cell["row_local_node_id"])
        if resolved_ri is None:
            issues.append(
                Issue(
                    code=TABLE_SPAN_CONFLICT,
                    severity="error",
                    node_id=cell["cell_local_node_id"],
                    path=f"/cells/{cells.index(cell)}/row_local_node_id",
                    message=(
                        f"cell references row_local_node_id "
                        f"{cell['row_local_node_id']!r}, absent from row_local_node_ids."
                    ),
                )
            )
            continue
        if cell["row_index"] != resolved_ri:
            issues.append(
                Issue(
                    code=TABLE_SPAN_CONFLICT,
                    severity="error",
                    node_id=cell["cell_local_node_id"],
                    path=f"/cells/{cells.index(cell)}/row_index",
                    message=(
                        f"cell row_index {cell['row_index']} does not match "
                        f"row_local_node_id {cell['row_local_node_id']!r}'s position "
                        f"{resolved_ri} in row_local_node_ids."
                    ),
                )
            )
            continue
        cells_by_row_index.setdefault(resolved_ri, []).append(cell)

    for ri, row_id in enumerate(row_ids):
        occupancy: dict[int, str] = {
            gc: "<row_span from earlier row>"
            for gc, until_ri in active_row_spans.items()
            if until_ri >= ri
        }
        for cell in cells_by_row_index.get(ri, []):
            cell_id = cell["cell_local_node_id"]
            cell_path = f"/cells/{cells.index(cell)}"
            if cell_id in seen_cell_ids:
                issues.append(
                    Issue(
                        code=TABLE_SPAN_CONFLICT,
                        severity="error",
                        node_id=cell_id,
                        path=cell_path,
                        message=f"cell_local_node_id {cell_id!r} appears more than once.",
                    )
                )
            seen_cell_ids.add(cell_id)

            grid_column = cell["grid_column"]
            col_span = cell["col_span"]
            row_span = cell["row_span"]

            if grid_column + col_span > grid_column_count:
                issues.append(
                    Issue(
                        code=TABLE_GRID_INVALID,
                        severity="error",
                        node_id=cell_id,
                        path=f"{cell_path}/grid_column",
                        message="Cell grid_column + col_span exceeds grid_column_count.",
                    )
                )
                continue

            if ri + row_span > len(row_ids):
                issues.append(
                    Issue(
                        code=TABLE_GRID_INVALID,
                        severity="error",
                        node_id=cell_id,
                        path=f"{cell_path}/row_span",
                        message="Cell row_span extends past the plan's final row.",
                    )
                )

            for gc in range(grid_column, grid_column + col_span):
                if gc in occupancy:
                    issues.append(
                        Issue(
                            code=TABLE_GRID_INVALID,
                            severity="error",
                            node_id=cell_id,
                            path=cell_path,
                            message=f"Cell span overlaps an existing cell at grid_column {gc}.",
                        )
                    )
                else:
                    occupancy[gc] = cell_path
                active_row_spans[gc] = max(active_row_spans.get(gc, -1), ri + row_span - 1)

        missing = [gc for gc in range(grid_column_count) if gc not in occupancy]
        if missing:
            issues.append(
                Issue(
                    code=TABLE_GRID_INVALID,
                    severity="error",
                    node_id=plan.get("table_local_node_id"),
                    path=f"/row_local_node_ids/{ri}",
                    message=f"Row {row_id!r} has uncovered grid columns: {missing}.",
                )
            )

    return issues

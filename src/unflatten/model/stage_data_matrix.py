"""The `StageDataMatrix` producer (RFC005 §30.2, §30.3): the closed joint
`(split, pyramid_level, task_kind, cohort) -> probability mass` a
promotion-class run must freeze before its first gradient update.

The pyramid and TaskView percentages in RFC005 §§15--18 are *human-readable
marginals*. This artefact is the normative object the sampler draws from,
and mass applies **before** fragment, view and augmentation expansion, so
a document with many pages or augmentations cannot pick up multiple weight.

Four §30.2/§30.3 rules are enforced here as code, because each of them
fails silently otherwise:

* **A missing cell is an error, never an implicit zero.** "Každá infeasible
  kombinácia musí byť explicitne `0`; missing cell je schema error." The
  document declares its axis domains and `cells` must be exactly their
  Cartesian product -- a dropped zero-mass cell still reconciles by sum,
  which is precisely why absence cannot be allowed to mean anything.
* **Declared marginals reconcile exactly** ("musia presne reconcile").
  Mass is carried as `mass_micro`, integer millionths summing to exactly
  :data:`TOTAL_MASS_MICRO`, following the `loss_weights_milli` precedent in
  `stage_policy.py`: the deterministic CBOR profile these identities hash
  under (`rfc8949-core-det-nfc-int-v0`) has no float in its data model, and
  an exact integer is the only way "presne" can be checked without a
  tolerance argument that would then need choosing.
* **Reads cannot fall back to zero.** :func:`cell_mass` raises on any
  coordinate outside the declared domains rather than returning `0`, which
  is the same rule as the first one, applied at read time.
* **Conditional renormalization against the loader pool is forbidden.**
  :func:`resolve_available_mass` is that rule as a function rather than a
  comment: given what the loader can actually serve, it returns the frozen
  masses unchanged and refuses if any non-zero cell is unavailable. It
  never redistributes.

§30.3's `w_lineage = bounded(w_feature_coverage × w_hardness)` is a
*within-cell* weighting and is not implemented here; the pyramid/task/domain
weights of §§15--18 are matrix-compilation inputs, and this module holds the
compiled result, not the compiler.

**This module builds and hashes a matrix; it does not choose one.** The
masses are an owner content decision and no matrix document is shipped.

Hashed under `rfc005.stage-data-matrix/v1`, the same `domain_hash`
convention as every other identity in this package.

Ships no `nn.Module`, no sampler, no loader, and authorizes no gradient
update.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.dataset.gates.task_eligibility import TASK_KINDS
from unflatten.model.hashing import domain_hash

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_SCHEMA_NAME = "stage_data_matrix"

#: Design §4.9 registry row `rfc005.stage-data-matrix/v1`. Owned here: no
#: other module computes a `stage_data_matrix_hash`.
_STAGE_DATA_MATRIX_DOMAIN = "rfc005.stage-data-matrix/v1"

STAGE_DATA_MATRIX_SCHEMA_VERSION = "1.0.0"

#: Probability mass is stored in millionths and the cells sum to exactly
#: this (module docstring).
TOTAL_MASS_MICRO = 1_000_000

#: The four axes of the §30.2 joint, in the order :func:`cell_mass` takes
#: them and in the order a coordinate tuple carries them.
AXES: tuple[str, ...] = ("split", "pyramid_level", "task_kind", "cohort")

#: RFC003 splits. `train`/`val`/`test` is also what the schema enumerates.
SPLITS: tuple[str, ...] = ("train", "val", "test")

#: The RFC003 dataset pyramid levels (P0 structure .. P5 full document).
PYRAMID_LEVELS: tuple[str, ...] = ("P0", "P1", "P2", "P3", "P4", "P5")

Coordinate = tuple[str, str, str, str]


class StageDataMatrixInvalid(ValueError):
    """A stage data matrix is not the exact product of its declared axes,
    fails to reconcile its declared marginals, is read or resolved outside
    those axes, or fails its closed schema.

    M1-internal (design §9.3); not an RFC004 registry code.
    """


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _domains(matrix: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        "split": tuple(matrix["splits"]),
        "pyramid_level": tuple(matrix["pyramid_levels"]),
        "task_kind": tuple(matrix["task_kinds"]),
        "cohort": tuple(matrix["cohorts"]),
    }


def _coordinate(cell: Mapping[str, Any]) -> Coordinate:
    return (cell["split"], cell["pyramid_level"], cell["task_kind"], cell["cohort"])


def _format(coord: Coordinate) -> str:
    return "(" + ", ".join(f"{axis}={value}" for axis, value in zip(AXES, coord)) + ")"


def _check_axis_domains(matrix: Mapping[str, Any]) -> None:
    """The declared task kinds must be exact RFC003 kinds.

    The other three axes are closed by the schema (`splits`,
    `pyramid_levels`) or free-form by design (`cohorts`, §30.2 names no
    cohort enum).
    """
    unknown = [kind for kind in matrix["task_kinds"] if kind not in TASK_KINDS]
    if unknown:
        raise StageDataMatrixInvalid(
            f"task_kinds names {', '.join(unknown)}, which is not an exact RFC003 "
            "task_kind -- a §30.2 cell axis must be drawn from the closed RFC003 "
            f"enum, one of {', '.join(TASK_KINDS)}"
        )


def _check_cells_are_the_full_product(matrix: Mapping[str, Any]) -> None:
    """`cells` must be exactly the Cartesian product of the declared axes.

    Three separate failures, each with its own message: a duplicated cell,
    a cell outside the declared domains, and a missing cell. RFC005 §30.2
    makes the last one a schema error rather than an implicit zero.
    """
    domains = _domains(matrix)
    seen: set[Coordinate] = set()
    for cell in matrix["cells"]:
        coord = _coordinate(cell)
        if coord in seen:
            raise StageDataMatrixInvalid(
                f"duplicate cell {_format(coord)} -- a §30.2 joint carries each "
                "combination exactly once, and two rows for one cell make its "
                "mass ambiguous"
            )
        seen.add(coord)
        outside = [
            f"{axis}={value}"
            for axis, value in zip(AXES, coord)
            if value not in domains[axis]
        ]
        if outside:
            raise StageDataMatrixInvalid(
                f"cell {_format(coord)} lies outside the declared axis domains "
                f"({', '.join(outside)}) -- RFC005 §30.2 draws only from the "
                "declared joint, so an undeclared cell could never be sampled"
            )

    missing = sorted(set(product(*(domains[axis] for axis in AXES))) - seen)
    if missing:
        shown = ", ".join(_format(coord) for coord in missing[:4])
        raise StageDataMatrixInvalid(
            f"{len(missing)} cell(s) missing from the declared product: {shown}"
            + (", ..." if len(missing) > 4 else "")
            + " -- RFC005 §30.2: every infeasible combination must be an explicit "
            "`0` and a missing cell is a schema error, because an absent cell and "
            "a zero cell are indistinguishable by sum"
        )


def _check_total_mass(matrix: Mapping[str, Any]) -> None:
    total = sum(cell["mass_micro"] for cell in matrix["cells"])
    if total != TOTAL_MASS_MICRO:
        raise StageDataMatrixInvalid(
            f"cell mass sums to {total} micro-units, not {TOTAL_MASS_MICRO} -- "
            "RFC005 §30.2 mass is a probability over lineage-root draws and the "
            "integer unit exists so the total is exact, not approximate"
        )


def _check_declared_marginals(matrix: Mapping[str, Any]) -> None:
    """Each declared axis marginal must be complete and exact (§30.2)."""
    domains = _domains(matrix)
    for axis, declared in matrix["declared_marginals"].items():
        actual: dict[str, int] = {value: 0 for value in domains[axis]}
        for cell in matrix["cells"]:
            actual[cell[axis]] += cell["mass_micro"]
        if set(declared) != set(actual):
            raise StageDataMatrixInvalid(
                f"declared_marginals[{axis!r}] is incomplete or names undeclared "
                f"values: declared {sorted(declared)}, axis domain "
                f"{sorted(actual)} -- a partial marginal reconciles nothing, so "
                "RFC005 §30.2 takes a declared axis to be declared in full"
            )
        mismatched = [
            f"{value}: declared {declared[value]}, cells sum to {actual[value]}"
            for value in sorted(actual)
            if declared[value] != actual[value]
        ]
        if mismatched:
            raise StageDataMatrixInvalid(
                f"declared_marginals[{axis!r}] does not reconcile with the cells "
                f"({'; '.join(mismatched)}) -- RFC005 §30.2 requires row/column "
                "marginals to reconcile exactly with the declared stage profile"
            )


def validate_stage_data_matrix(matrix: Mapping[str, Any]) -> None:
    """Validate `matrix` against `stage_data_matrix.schema.json`, plus the
    §30.2 rules the schema cannot express (module docstring)."""
    errors = list(_validator().iter_errors(matrix))
    if errors:
        raise StageDataMatrixInvalid(
            f"schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )
    _check_axis_domains(matrix)
    _check_cells_are_the_full_product(matrix)
    _check_total_mass(matrix)
    _check_declared_marginals(matrix)


def build_stage_data_matrix(
    *,
    matrix_version: str,
    stage_id: str,
    splits: Sequence[str],
    pyramid_levels: Sequence[str],
    task_kinds: Sequence[str],
    cohorts: Sequence[str],
    cells: Sequence[Mapping[str, Any]],
    declared_marginals: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Assemble one stage data matrix, validated on the way out.

    Every argument is keyword-only and required; there are no defaults and
    no cell is generated for the caller. `declared_marginals={}` is the
    declaration that no marginal was pinned -- it must be passed, not
    omitted.
    """
    matrix = {
        "stage_data_matrix_schema_version": STAGE_DATA_MATRIX_SCHEMA_VERSION,
        "matrix_version": matrix_version,
        "stage_id": stage_id,
        "splits": list(splits),
        "pyramid_levels": list(pyramid_levels),
        "task_kinds": list(task_kinds),
        "cohorts": list(cohorts),
        "cells": [dict(cell) for cell in cells],
        "declared_marginals": {
            axis: dict(values) for axis, values in declared_marginals.items()
        },
    }
    validate_stage_data_matrix(matrix)
    return matrix


def load_stage_data_matrix(path: Path | str) -> dict[str, Any]:
    """Read a frozen stage data matrix from disk, validated on the way in."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise StageDataMatrixInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise StageDataMatrixInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_stage_data_matrix(payload)
    return payload


def stage_data_matrix_hash(matrix: Mapping[str, Any]) -> str:
    """`stage_data_matrix_hash` over the whole validated matrix."""
    validate_stage_data_matrix(matrix)
    return domain_hash(_STAGE_DATA_MATRIX_DOMAIN, dict(matrix))


def cell_mass(
    matrix: Mapping[str, Any],
    split: str,
    pyramid_level: str,
    task_kind: str,
    cohort: str,
) -> int:
    """The frozen `mass_micro` of one cell.

    Raises rather than returning `0` for a coordinate outside the declared
    domains: a silent zero would turn "this combination was never declared"
    into "this combination is infeasible", which is exactly the collapse
    RFC005 §30.2 forbids at write time.
    """
    coord = (split, pyramid_level, task_kind, cohort)
    domains = _domains(matrix)
    outside = [
        f"{axis}={value}" for axis, value in zip(AXES, coord) if value not in domains[axis]
    ]
    if outside:
        raise StageDataMatrixInvalid(
            f"cell {_format(coord)} lies outside the declared axis domains "
            f"({', '.join(outside)}) -- RFC005 §30.2: an undeclared coordinate is "
            "an error, not zero mass"
        )
    for cell in matrix["cells"]:
        if _coordinate(cell) == coord:
            return cell["mass_micro"]
    raise StageDataMatrixInvalid(  # pragma: no cover - unreachable after validation
        f"cell {_format(coord)} is within the declared axis domains but absent "
        "from `cells`; the matrix was not validated"
    )


def resolve_available_mass(
    matrix: Mapping[str, Any], available_cells: Iterable[Coordinate]
) -> dict[Coordinate, int]:
    """Return the frozen masses for a loader pool, or refuse.

    `available_cells` is the set of coordinates the loader can actually
    serve. The frozen masses come back **unchanged**; if any cell with
    non-zero declared mass is unavailable, this raises. RFC005 §30.2:
    "Conditional renormalization podľa momentálne dostupného loader poolu je
    zakázaná" -- redistributing that mass over what remains would silently
    change the stage's data profile while every hash still matched.

    An unavailable *zero*-mass cell is not an error: there is nothing to
    draw from it, so its absence cannot bias the distribution.
    """
    available = set(available_cells)
    domains = _domains(matrix)
    undeclared = sorted(
        coord
        for coord in available
        if any(value not in domains[axis] for axis, value in zip(AXES, coord))
    )
    if undeclared:
        raise StageDataMatrixInvalid(
            "the loader pool offers cell(s) outside the declared axis domains: "
            f"{', '.join(_format(coord) for coord in undeclared)} -- RFC005 §30.2 "
            "draws only from the frozen joint, so an undeclared cell has no mass "
            "and must not be served"
        )

    frozen = {_coordinate(cell): cell["mass_micro"] for cell in matrix["cells"]}
    unavailable = sorted(
        coord for coord, mass in frozen.items() if mass > 0 and coord not in available
    )
    if unavailable:
        raise StageDataMatrixInvalid(
            "the loader pool cannot serve cell(s) carrying non-zero frozen mass: "
            + ", ".join(
                f"{_format(coord)} mass_micro={frozen[coord]}" for coord in unavailable
            )
            + " -- RFC005 §30.2 forbids conditional renormalization against the "
            "currently available loader pool, so this mass may not be "
            "redistributed over the remaining cells; fix the pool or freeze a "
            "different matrix"
        )
    return frozen

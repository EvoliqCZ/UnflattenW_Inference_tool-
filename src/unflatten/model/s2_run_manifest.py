"""S2 run manifest (RFC005 §44.1, addendum A construction).

The run manifest is the identity an S2 run executes under. Every field is
a derivation from a frozen artefact, never a free choice:

- `run_id` is derived from all the other fields (the addendum A
  construction, the same one S1's `s1run_` ids use): a run that changes
  any bound — cohort, matrix, policies, plan, seed — is a DIFFERENT run,
  and its `run_id` says so without any bookkeeping;
- every hash pins the exact artefact the run may draw from, and the
  builder re-derives each hash from the presented document rather than
  trusting a value a caller might have typed in;
- the ordered exposure plan is cross-checked against the other inputs:
  its own self-hash, its matrix/policy hashes and its `run_seed` must all
  agree with the documents the manifest pins, so a plan built for another
  matrix, another policy or another seed can never be smuggled into a
  run's identity.

`gradient_bearing` is DECLARED, never inferred (DEC-009: an S2 run starts
as `discovery`; whether it bears gradient is an owner decision frozen
here). The CLI (`main`) freezes the manifest to `s2_run_manifest.json`
and refuses to overwrite an existing one, exactly like S1's freeze CLI.

**1.1.0 (DEC-027 D1/D2, T4b).** S2 has no run-manifest shape of its own
for cache identity, allowlist, ancestry or the run's optimizer budget --
those bindings already exist, verified, on the S1 side. Rather than
duplicate them, this manifest names a COMPANION S1-format run manifest
(frozen with the existing `freeze_s1_run_manifest`, whose
`run_config.sampler.order` is `s2_ordered_exposure_plan`, T4a) by hash:
`binding_s1_run_manifest_hash` is re-derived by the builder from the
presented companion document
(`run_binding.compute_s1_run_manifest_hash`), never a caller-typed value,
and the companion is refused unless its own `run_config` agrees with the
S2 inputs it is paired with -- its sampler order names the S2 plan, its
`steps x batch_size` equals the pinned sampler policy's
`virtual_epoch_size`, and its sampler seed equals `run_seed`. `cohort_hash`
is nullable: DEC-027 D2 keeps `S2Cohort` for the `synthetic_p1p2` cells
only (today's matrix pins them at zero mass), so a cohort is required iff
some matrix cell of that cohort carries mass, and refused as an unused pin
otherwise.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema

from unflatten.model import run_binding as rb
from unflatten.model.hashing import domain_hash, make_id
from unflatten.model.s2_cohort import S2Cohort, load_s2_cohort
from unflatten.model.s2_sampler import (
    ORDERED_EXPOSURE_PLAN_DOMAIN,
    load_s2_exposure_policy,
    load_s2_sampler_policy,
    s2_exposure_policy_hash,
    s2_sampler_policy_hash,
    validate_s2_exposure_policy,
    validate_s2_sampler_policy,
)
from unflatten.model.stage_data_matrix import (
    load_stage_data_matrix,
    stage_data_matrix_hash,
    validate_stage_data_matrix,
)

_SCHEMAS_DIR = Path(__file__).parent / "schemas"

S2_RUN_MANIFEST_SCHEMA_VERSION = "1.1.0"

#: DEC-027 D2/D3: the `S2Cohort` this module loads pins ONLY the
#: synthetic P1/P2 development cohort; `docxcorp` cells are pinned through
#: the companion S1 manifest instead. A cohort is required iff some matrix
#: cell of exactly this name carries mass.
_SYNTHETIC_COHORT_NAME = "synthetic_p1p2"

#: The one sampler order a companion manifest may declare (T4a widened the
#: S1 schema to admit it additively).
_COMPANION_SAMPLER_ORDER = "s2_ordered_exposure_plan"

#: The shared filename contract (the S1 analogue owns
#: `s1_run_manifest.json`; the two prefixes cannot collide).
S2_RUN_MANIFEST_FILENAME = "s2_run_manifest.json"

RUN_ID_PREFIX = "s2run"

#: This module is the sole owner of this domain; the S1 run id lives under
#: `rfc005.s1-run-id/v1` with a deliberately different prefix/domain so
#: the two can never be confused.
RUN_ID_DOMAIN = "rfc005.s2-run-id/v1"
S2_RUN_MANIFEST_DOMAIN = "rfc005.s2-run-manifest/v1"

_RUN_CLASSES = ("discovery", "promotion")


class S2RunManifestInvalid(ValueError):
    """The manifest (or one of the documents it binds) is not well-formed:
    a schema violation, a self-hash that does not verify, or a cross
    check that fails (a plan built for a different matrix/policy/seed than
    the manifest pins). A refusal, not a repair."""


@cache
def _manifest_validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMAS_DIR / "s2_run_manifest.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


@cache
def _plan_validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMAS_DIR / "ordered_exposure_plan.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _refuse(message: str) -> S2RunManifestInvalid:
    return S2RunManifestInvalid(f"s2 run manifest refused: {message}")


def verify_ordered_exposure_plan_against(
    *,
    plan: Mapping[str, Any],
    run_seed: int,
    matrix: Mapping[str, Any],
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
) -> str:
    """Verify the plan's own integrity and that it was built from EXACTLY
    the documents the manifest pins: schema, self-hash, matrix/policy
    hashes and `run_seed`. Returns the plan's hash."""
    if not isinstance(plan, Mapping):
        raise _refuse(f"ordered exposure plan must be an object, got {type(plan).__name__}")
    errors = list(_plan_validator().iter_errors(plan))
    if errors:
        raise _refuse(
            f"plan schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )
    payload = {k: v for k, v in plan.items() if k != "ordered_exposure_plan_hash"}
    if domain_hash(ORDERED_EXPOSURE_PLAN_DOMAIN, payload) != plan["ordered_exposure_plan_hash"]:
        raise _refuse("ordered_exposure_plan_hash does not verify against the plan payload")

    cross_checks = {
        "matrix_hash": stage_data_matrix_hash(matrix),
        "sampler_policy_hash": s2_sampler_policy_hash(sampler_policy),
        "exposure_policy_hash": s2_exposure_policy_hash(exposure_policy),
    }
    for field, expected in cross_checks.items():
        if plan[field] != expected:
            raise _refuse(
                f"plan {field} ({plan[field]}) does not match the pinned "
                f"{expected} — the plan was built from different inputs"
            )
    if plan["run_seed"] != run_seed:
        raise _refuse(
            f"plan run_seed ({plan['run_seed']}) does not match the manifest "
            f"run_seed ({run_seed})"
        )
    if plan["virtual_epoch_size"] != sampler_policy["virtual_epoch_size"]:
        raise _refuse(
            "plan virtual_epoch_size does not match the pinned sampler policy"
        )
    return plan["ordered_exposure_plan_hash"]


def _verify_companion(companion: Mapping[str, Any]) -> str:
    """Verify the presented companion S1-format run manifest is well-formed
    (schema + self-hash) and return its own `s1_run_manifest_hash` --
    RE-DERIVED from the presented document, never read as a caller-typed
    value (1.1.0, DEC-027 D2)."""
    if not isinstance(companion, Mapping):
        raise _refuse(
            f"companion s1_run_manifest must be an object, got {type(companion).__name__}"
        )
    errors = list(rb._validator().iter_errors(companion))
    if errors:
        raise _refuse(
            f"companion s1_run_manifest schema validation failed: {errors[0].message} "
            f"at {list(errors[0].absolute_path)}"
        )
    payload = {k: v for k, v in companion.items() if k != "s1_run_manifest_hash"}
    derived = rb.compute_s1_run_manifest_hash(payload)
    if derived != companion["s1_run_manifest_hash"]:
        raise _refuse(
            f"companion s1_run_manifest_hash does not verify: stored "
            f"{companion['s1_run_manifest_hash']!r}, recomputed {derived!r} -- "
            "the companion document was mutated after freezing, or its hash "
            "was never computed correctly"
        )
    return derived


def _verify_companion_consistency(
    companion: Mapping[str, Any],
    *,
    run_seed: int,
    sampler_policy: Mapping[str, Any],
) -> None:
    """DEC-027 D1/D2/T4b: the companion's own frozen `run_config` must
    agree with the S2 inputs it is paired with -- its sampler order must
    name the S2 plan (never claim a random permutation it does not use),
    its budget must spend exactly one virtual epoch, and its seed must be
    the manifest's `run_seed`."""
    run_config = companion.get("run_config")
    if run_config is None:
        raise _refuse(
            "companion s1_run_manifest has a null run_config -- an S2 "
            "companion must freeze the budget it spends (DEC-027 D1/T4b)"
        )
    order = run_config["sampler"]["order"]
    if order != _COMPANION_SAMPLER_ORDER:
        raise _refuse(
            f"companion run_config.sampler.order is {order!r}, expected "
            f"{_COMPANION_SAMPLER_ORDER!r} -- a companion manifest must not "
            "claim a random permutation it does not use (DEC-027 D2)"
        )
    exposure = run_config["steps"] * run_config["sampler"]["batch_size"]
    if exposure != sampler_policy["virtual_epoch_size"]:
        raise _refuse(
            f"companion run_config steps x batch_size ({exposure}) does not "
            f"equal the pinned sampler policy virtual_epoch_size "
            f"({sampler_policy['virtual_epoch_size']})"
        )
    seed = run_config["sampler"]["seed"]
    if seed != run_seed:
        raise _refuse(
            f"companion run_config.sampler.seed ({seed}) does not match "
            f"run_seed ({run_seed})"
        )


def _resolve_cohort_hash(
    cohort: S2Cohort | None, matrix: Mapping[str, Any]
) -> str | None:
    """DEC-027 D2: a `synthetic_p1p2` cohort is required iff a matrix cell
    of that cohort carries mass -- refused as an unused pin when one is
    presented but none does, refused as unpinned when one does and none is
    presented."""
    if cohort is not None and not isinstance(cohort, S2Cohort):
        raise _refuse(
            "cohort must be a verified S2Cohort (load_s2_cohort) or None, "
            f"got {type(cohort).__name__}"
        )
    synthetic_has_mass = any(
        cell["cohort"] == _SYNTHETIC_COHORT_NAME and cell["mass_micro"] > 0
        for cell in matrix["cells"]
    )
    if synthetic_has_mass and cohort is None:
        raise _refuse(
            f"the matrix carries mass on a {_SYNTHETIC_COHORT_NAME!r} cell but "
            "no cohort was presented -- an unpinned cohort with used mass is "
            "the exact gap DEC-MP-4 closes (DEC-027 D2)"
        )
    if not synthetic_has_mass and cohort is not None:
        raise _refuse(
            f"a cohort was presented but no matrix cell of "
            f"{_SYNTHETIC_COHORT_NAME!r} carries mass -- an unused pin is a "
            "lie too (DEC-027 D2)"
        )
    return cohort.cohort_hash if cohort is not None else None


def _validated_base(
    *,
    run_class: str,
    gradient_bearing: bool,
    run_seed: int,
    companion: Mapping[str, Any],
    cohort: S2Cohort | None,
    matrix: Mapping[str, Any],
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate every input and re-derive every hash from the presented
    documents. Returns the manifest payload with `run_id` and the
    self-hash OMITTED — the exact projection the addendum A `run_id` is
    derived from, so the id cannot drift from the contents it names.
    """
    if run_class not in _RUN_CLASSES:
        raise _refuse(f"run_class must be one of {_RUN_CLASSES}, got {run_class!r}")
    if type(gradient_bearing) is not bool:
        raise _refuse(
            f"gradient_bearing must be a declared boolean, got {gradient_bearing!r}"
        )
    # `type is int` on purpose: `bool` is an `int` subclass and a `true`
    # seed would otherwise pass an `isinstance` check.
    if type(run_seed) is not int or run_seed < 0:
        raise _refuse(f"run_seed must be an integer >= 0, got {run_seed!r}")

    validate_stage_data_matrix(matrix)
    validate_s2_sampler_policy(sampler_policy)
    validate_s2_exposure_policy(exposure_policy)
    plan_hash = verify_ordered_exposure_plan_against(
        plan=plan,
        run_seed=run_seed,
        matrix=matrix,
        sampler_policy=sampler_policy,
        exposure_policy=exposure_policy,
    )
    binding_hash = _verify_companion(companion)
    _verify_companion_consistency(
        companion, run_seed=run_seed, sampler_policy=sampler_policy
    )
    cohort_hash = _resolve_cohort_hash(cohort, matrix)
    return {
        "s2_run_manifest_schema_version": S2_RUN_MANIFEST_SCHEMA_VERSION,
        "run_class": run_class,
        "gradient_bearing": gradient_bearing,
        "stage_id": "S2",
        "run_seed": run_seed,
        "binding_s1_run_manifest_hash": binding_hash,
        "cohort_hash": cohort_hash,
        "stage_data_matrix_hash": stage_data_matrix_hash(matrix),
        "s2_sampler_policy_hash": s2_sampler_policy_hash(sampler_policy),
        "s2_exposure_policy_hash": s2_exposure_policy_hash(exposure_policy),
        "ordered_exposure_plan_hash": plan_hash,
    }


def build_s2_run_manifest(
    *,
    run_id: str,
    run_class: str,
    gradient_bearing: bool,
    run_seed: int,
    companion: Mapping[str, Any],
    matrix: Mapping[str, Any],
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    cohort: S2Cohort | None = None,
) -> dict[str, Any]:
    """Build the hash-frozen S2 run manifest for the pinned inputs.

    Every hash is re-derived from the presented document (never copied
    from a caller-supplied value) and the plan is cross-checked against
    the other pinned inputs. The caller supplies `run_id`; :func:`freeze`
    is the only place a `run_id` is derived (addendum A). `companion` is
    the S1-format run manifest this S2 run is bound through (1.1.0,
    DEC-027 D1); `cohort` is optional and required only when the matrix
    pins mass on a `synthetic_p1p2` cell (DEC-027 D2).
    """
    if not isinstance(run_id, str) or not run_id:
        raise _refuse(f"run_id must be a non-empty string, got {run_id!r}")
    base = _validated_base(
        run_class=run_class,
        gradient_bearing=gradient_bearing,
        run_seed=run_seed,
        companion=companion,
        cohort=cohort,
        matrix=matrix,
        sampler_policy=sampler_policy,
        exposure_policy=exposure_policy,
        plan=plan,
    )
    payload: dict[str, Any] = {**base, "run_id": run_id}
    payload["s2_run_manifest_hash"] = domain_hash(
        S2_RUN_MANIFEST_DOMAIN,
        {k: v for k, v in payload.items() if k != "s2_run_manifest_hash"},
    )
    errors = list(_manifest_validator().iter_errors(payload))
    if errors:
        raise _refuse(
            f"manifest schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )
    return payload


def verify_s2_run_manifest(manifest: Mapping[str, Any]) -> None:
    """Schema validation PLUS the self-hash check: a manifest whose
    `s2_run_manifest_hash` does not match its payload was tampered with
    (or never produced by this module) and is refused."""
    errors = list(_manifest_validator().iter_errors(manifest))
    if errors:
        raise _refuse(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )
    payload = {k: v for k, v in manifest.items() if k != "s2_run_manifest_hash"}
    if domain_hash(S2_RUN_MANIFEST_DOMAIN, payload) != manifest["s2_run_manifest_hash"]:
        raise _refuse("s2_run_manifest_hash does not verify against the payload")


def load_s2_run_manifest(path: Path | str) -> dict[str, Any]:
    """Read a frozen S2 run manifest from disk, verified on the way in
    (schema + self-hash)."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise S2RunManifestInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise S2RunManifestInvalid(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise S2RunManifestInvalid(f"{path} must hold a JSON object")
    verify_s2_run_manifest(payload)
    return payload


def freeze(
    *,
    companion: Mapping[str, Any],
    matrix: Mapping[str, Any],
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    run_class: str,
    gradient_bearing: bool,
    run_seed: int,
    out_dir: Path,
    cohort: S2Cohort | None = None,
) -> dict[str, Any]:
    """Freeze the S2 run manifest for the pinned inputs, writing
    `s2_run_manifest.json` under `out_dir` (refusing to overwrite).

    The `run_id` is derived the addendum A way: `make_id` over every
    manifest field except `run_id` and the self-hash, so the id is a pure
    function of the pinned contents — a run that changes any bound is a
    different run, and its id says so without bookkeeping.
    """
    base = _validated_base(
        run_class=run_class,
        gradient_bearing=gradient_bearing,
        run_seed=run_seed,
        companion=companion,
        cohort=cohort,
        matrix=matrix,
        sampler_policy=sampler_policy,
        exposure_policy=exposure_policy,
        plan=plan,
    )
    run_id = make_id(RUN_ID_PREFIX, RUN_ID_DOMAIN, dict(base))
    manifest = build_s2_run_manifest(
        run_id=run_id,
        run_class=run_class,
        gradient_bearing=gradient_bearing,
        run_seed=run_seed,
        companion=companion,
        cohort=cohort,
        matrix=matrix,
        sampler_policy=sampler_policy,
        exposure_policy=exposure_policy,
        plan=plan,
    )

    out_dir = Path(out_dir)
    manifest_path = out_dir / S2_RUN_MANIFEST_FILENAME
    if manifest_path.exists():
        raise S2RunManifestInvalid(
            f"{manifest_path} already exists -- refusing to overwrite a "
            "frozen run manifest"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    return {
        "run_id": manifest["run_id"],
        "s2_run_manifest_hash": manifest["s2_run_manifest_hash"],
        "out_dir": str(out_dir),
        "cohort_hash": manifest["cohort_hash"],
        "ordered_exposure_plan_hash": manifest["ordered_exposure_plan_hash"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze an s2_run_manifest for the pinned S2 inputs."
    )
    parser.add_argument(
        "--binding-run-manifest",
        type=Path,
        required=True,
        help="the companion S1-format run manifest this S2 run is bound "
        "through (1.1.0, DEC-027 D1)",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=None,
        help="the synthetic_p1p2 S2Cohort document; required iff the matrix "
        "pins mass on a synthetic_p1p2 cell (DEC-027 D2)",
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--sampler-policy", type=Path, required=True)
    parser.add_argument("--exposure-policy", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--run-class", choices=_RUN_CLASSES, required=True
    )
    parser.add_argument("--gradient-bearing", action="store_true")
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = freeze(
        companion=json.loads(args.binding_run_manifest.read_text()),
        cohort=load_s2_cohort(args.cohort) if args.cohort else None,
        matrix=load_stage_data_matrix(args.matrix),
        sampler_policy=load_s2_sampler_policy(args.sampler_policy),
        exposure_policy=load_s2_exposure_policy(args.exposure_policy),
        plan=json.loads(args.plan.read_text()),
        run_class=args.run_class,
        gradient_bearing=args.gradient_bearing,
        run_seed=args.run_seed,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

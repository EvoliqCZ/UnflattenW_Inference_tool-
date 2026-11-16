"""S3 run manifest (RFC005 §12/§26, design §7.1, addendum A construction).

The manifest is the identity one S3 discovery run executes under, built the
same way `s2_run_manifest` builds S2's: every field is a derivation from a
frozen artefact, `run_id` is a pure function of the other fields, and every
hash is re-derived from the presented document rather than trusted as a
caller-typed value.

S3 has no run-manifest shape of its own for cache identity, allowlist or
ancestry either -- exactly like S2, it names a companion S1-format run
manifest by hash (`binding_s1_run_manifest_hash`, re-derived via
`s2_run_manifest._verify_companion`, reused verbatim) and refuses one whose
own `run_config` disagrees with the S3 inputs it is paired with (sampler
order, budget, seed -- the same three checks `s2_run_manifest` applies,
reused via `s2_run_manifest._verify_companion_consistency`).

Three things are new relative to S2:

- **`s3_stage_policy_hash`** is bound to the committed `s3_stage_policy.json`
  (via `stage_policy.load_stage_policy` / `stage_policy_hash`), and the
  `L_typed` head ids the census/floor logic below reasons about come from
  that policy's `task_view_allowlist`, never restated here.
- **The typed supervision census.** `typed_supervision_census_hash` pins a
  frozen `s3_typed_census` payload; its own `census_hash` must verify, and
  its `cache_identity_hash` / `expectation_hash` must equal the companion's
  (`exact_text_cache_identity_hash` / `exact_text_cache_expectation_hash`)
  -- a census computed over a different cache than the run's own companion
  binds is refused, not silently trusted. Every `heads_frozen_for_no_supervision`
  entry whose census family-bearing `supervised` count is at or above
  `typed_supervision_floor.min_supervised_per_head` is an unused freeze (a
  lie) and is refused by name; freezing every `L_typed` head is refused too
  (a typed run that trains no typed head).
- **`init_from`** (design §6), an object or null. `parent_run_class` is
  never caller-supplied: it is re-derived here from the presented parent
  companion manifest (another S1-format run manifest, this run's parent's
  own companion), whose hash must equal the pinned
  `parent_s1_run_manifest_hash`; a disagreeing pin is refused, and so is a
  `discovery` parent warm-started into a `promotion` run (RFC005 §12.1).

`grad_clip_micro` deliberately does not appear in this manifest -- it lives
only in `run_config.json` ([governance R1] amendment to design §7.1).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema

from unflatten.model import checkpoint_ancestry as ca
from unflatten.model import s2_run_manifest as s2m
from unflatten.model import stage_policy as sp
from unflatten.model.hashing import domain_hash, make_id
from unflatten.model.s2_sampler import (
    load_s2_exposure_policy,
    load_s2_sampler_policy,
    s2_exposure_policy_hash,
    s2_sampler_policy_hash,
    validate_s2_exposure_policy,
    validate_s2_sampler_policy,
)
from unflatten.model.s3_typed_census import _CENSUS_DOMAIN
from unflatten.model.stage_data_matrix import (
    load_stage_data_matrix,
    stage_data_matrix_hash,
    validate_stage_data_matrix,
)

_SCHEMAS_DIR = Path(__file__).parent / "schemas"

S3_RUN_MANIFEST_SCHEMA_VERSION = "1.0.0"

#: The shared filename contract (S1 owns `s1_run_manifest.json`, S2 owns
#: `s2_run_manifest.json`; the three prefixes cannot collide).
S3_RUN_MANIFEST_FILENAME = "s3_run_manifest.json"

RUN_ID_PREFIX = "s3run"

#: This module is the sole owner of these two domains.
RUN_ID_DOMAIN = "rfc005.s3-run-id/v1"
S3_RUN_MANIFEST_DOMAIN = "rfc005.s3-run-manifest/v1"

_RUN_CLASSES = ("discovery", "promotion")

#: D5: the model's only supported integer codec today (candidate_model.py);
#: pinned, never a caller choice (design §11 D5).
TYPED_INTEGER_CODEC = "direct_regression"

#: The one loss family whose task_view_allowlist rows name the typed heads
#: this manifest's census/floor logic reasons about.
_L_TYPED_FAMILY = "L_typed"


class S3RunManifestInvalid(ValueError):
    """The manifest (or one of the documents it binds) is not well-formed:
    a schema violation, a self-hash that does not verify, or a cross check
    that fails. A refusal, not a repair."""


@cache
def _manifest_validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMAS_DIR / "s3_run_manifest.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _refuse(message: str) -> S3RunManifestInvalid:
    return S3RunManifestInvalid(f"s3 run manifest refused: {message}")


def _l_typed_head_ids(stage_policy: Mapping[str, Any]) -> set[str]:
    """The `L_typed` head ids the committed stage policy declares, read
    from its `task_view_allowlist` rows (design §11: "the L_typed head_ids
    come from the committed policies/s3_stage_policy.json") -- never
    restated as a literal here."""
    heads: set[str] = set()
    for row in stage_policy["task_view_allowlist"]:
        if row["loss_family"] == _L_TYPED_FAMILY:
            heads.update(row["head_ids"])
    return heads


def _verify_census(census: Mapping[str, Any]) -> tuple[str, str, str]:
    """Verify the presented `s3_typed_census` payload's own self-hash and
    return `(census_hash, cache_identity_hash, expectation_hash)`."""
    if not isinstance(census, Mapping):
        raise _refuse(f"census must be an object, got {type(census).__name__}")
    required = {"cache_identity_hash", "expectation_hash", "counts", "census_hash"}
    missing = sorted(required - set(census))
    if missing:
        raise _refuse(f"census is missing field(s): {', '.join(missing)}")
    payload = {k: v for k, v in census.items() if k != "census_hash"}
    derived = domain_hash(_CENSUS_DOMAIN, payload)
    if derived != census["census_hash"]:
        raise _refuse(
            f"census_hash does not verify: stored {census['census_hash']!r}, "
            f"recomputed {derived!r} -- the census document was mutated after "
            "freezing, or its hash was never computed correctly"
        )
    return census["census_hash"], census["cache_identity_hash"], census["expectation_hash"]


def _verify_census_against_companion(
    census: Mapping[str, Any], companion: Mapping[str, Any]
) -> str:
    """Cross-check: the census's own `cache_identity_hash` /
    `expectation_hash` must equal the companion's -- a census counted over a
    different cache than the run's own companion binds is refused, not
    silently trusted."""
    census_hash, cache_identity_hash, expectation_hash = _verify_census(census)
    if cache_identity_hash != companion["exact_text_cache_identity_hash"]:
        raise _refuse(
            f"census cache_identity_hash ({cache_identity_hash}) does not match "
            f"the companion's ({companion['exact_text_cache_identity_hash']})"
        )
    if expectation_hash != companion["exact_text_cache_expectation_hash"]:
        raise _refuse(
            f"census expectation_hash ({expectation_hash}) does not match the "
            f"companion's ({companion['exact_text_cache_expectation_hash']})"
        )
    return census_hash


def _validate_typed_supervision_floor(floor: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(floor, Mapping):
        raise _refuse(
            f"typed_supervision_floor must be an object, got {type(floor).__name__}"
        )
    required = ("min_supervised_per_head", "min_lineages_per_head")
    missing = [field for field in required if field not in floor]
    if missing:
        raise _refuse(f"typed_supervision_floor is missing field(s): {', '.join(missing)}")
    result: dict[str, int] = {}
    for field in required:
        value = floor[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _refuse(
                f"typed_supervision_floor.{field} must be an integer >= 0, got {value!r}"
            )
        result[field] = value
    return result


def head_meets_floor(
    per_head_census_entry: Mapping[str, Any] | None, floor: Mapping[str, int]
) -> bool:
    """True iff a head's census entry meets BOTH `typed_supervision_floor`
    dimensions (design §7.3 step 5): `supervised >= min_supervised_per_head`
    AND `lineage_count >= min_lineages_per_head`. A head with plenty of
    supervised targets but too few distinct lineages does not meet the
    floor -- and is therefore correctly freezable, not an unused freeze.

    Public and named for reuse: `s3_train`'s admission gate (design §7.3
    step 5) applies the identical two-dimension rule to decide whether an
    UNFROZEN head may train, and must never restate it.
    """
    entry = per_head_census_entry or {}
    supervised = entry.get("supervised", 0)
    lineage_count = entry.get("lineage_count", 0)
    return (
        supervised >= floor["min_supervised_per_head"]
        and lineage_count >= floor["min_lineages_per_head"]
    )


def _validate_heads_frozen_for_no_supervision(
    heads_frozen_for_no_supervision: Sequence[str],
    *,
    l_typed_head_ids: set[str],
    census: Mapping[str, Any],
    floor: Mapping[str, int],
) -> list[str]:
    if not isinstance(heads_frozen_for_no_supervision, Sequence) or isinstance(
        heads_frozen_for_no_supervision, (str, bytes)
    ):
        raise _refuse(
            "heads_frozen_for_no_supervision must be a sequence of head ids, got "
            f"{type(heads_frozen_for_no_supervision).__name__}"
        )
    frozen = list(heads_frozen_for_no_supervision)
    per_head = census["counts"]["per_head"]
    for head_id in frozen:
        entry = per_head.get(head_id, {})
        if head_meets_floor(entry, floor):
            raise _refuse(
                f"{head_id!r} is listed in heads_frozen_for_no_supervision but its "
                f"census entry meets both floor dimensions (supervised="
                f"{entry.get('supervised', 0)} >= {floor['min_supervised_per_head']}, "
                f"lineage_count={entry.get('lineage_count', 0)} >= "
                f"{floor['min_lineages_per_head']}) -- an unused freeze is a lie too "
                "(DEC-027 D2 rule)"
            )
    if set(frozen) >= l_typed_head_ids and l_typed_head_ids:
        raise _refuse(
            "every L_typed head is listed in heads_frozen_for_no_supervision -- a "
            "typed run that trains no typed head"
        )
    return frozen


def init_from_block_from_checkpoint(
    checkpoint_path: Path | str, parent_companion: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive the S3 manifest's `init_from` block (design §6) from the real
    parent CHECKPOINT artefacts -- never accepted as a caller-typed JSON
    document. `s3_train.load_init_weights`'s `pinned` argument reads
    exactly the subset of these seven keys it needs
    (`checkpoint_sha256`, `checkpoint_id`, `parent_s1_run_manifest_hash`,
    `parent_base_ancestry_hash`, `step`); this function computes all seven,
    including the two (`parent_run_id`, `parent_run_class`) the manifest
    schema also requires.

    Verification order: the checkpoint file's own bytes hash
    (`checkpoint_ancestry.file_sha256`); the sidecar at
    `checkpoint_ancestry.sidecar_path`, verified in full against ITSELF
    (its own claimed `base_ancestry_hash` / `s1_run_manifest_hash`, and the
    `ancestry_hash` embedded in the checkpoint file) via
    `checkpoint_ancestry.verify_sidecar`; `checkpoint_id` recomputed via
    `checkpoint_ancestry.compute_checkpoint_id` and checked against the
    sidecar's own; the presented `parent_companion`'s own hash
    (`s2_run_manifest._verify_companion`) checked against the sidecar's
    `s1_run_manifest_hash`.
    """
    import torch  # local: the only function in this module that needs it

    checkpoint_path = Path(checkpoint_path)
    checkpoint_sha256 = ca.file_sha256(checkpoint_path)

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise _refuse(
            f"init_from checkpoint {checkpoint_path} could not be loaded: {exc}"
        ) from exc

    sidecar_path = ca.sidecar_path(checkpoint_path)
    try:
        raw_sidecar = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise _refuse(f"init_from sidecar {sidecar_path} could not be read: {exc}") from exc
    raw_checkpoint = raw_sidecar.get("checkpoint") or {}

    try:
        doc = ca.verify_sidecar(
            checkpoint_path,
            expect_base_ancestry_hash=raw_checkpoint.get("base_ancestry_hash"),
            expect_s1_run_manifest_hash=raw_checkpoint.get("s1_run_manifest_hash"),
            embedded_ancestry_hash=state.get("ancestry_hash"),
        )
    except ca.CheckpointAncestryInvalid as exc:
        raise _refuse(f"init_from checkpoint sidecar refused: {exc}") from exc

    checkpoint_id = ca.compute_checkpoint_id(doc["ancestry_hash"], checkpoint_sha256)
    if checkpoint_id != doc["checkpoint"]["checkpoint_id"]:
        raise _refuse(
            f"init_from checkpoint_id recomputed as {checkpoint_id!r} does not match "
            f"the sidecar's own {doc['checkpoint']['checkpoint_id']!r}"
        )

    try:
        parent_hash = s2m._verify_companion(parent_companion)
    except s2m.S2RunManifestInvalid as exc:
        raise _refuse(str(exc)) from exc
    if parent_hash != doc["checkpoint"]["s1_run_manifest_hash"]:
        raise _refuse(
            f"presented parent companion's own s1_run_manifest_hash ({parent_hash}) "
            "does not match the checkpoint sidecar's checkpoint.s1_run_manifest_hash "
            f"({doc['checkpoint']['s1_run_manifest_hash']})"
        )

    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": checkpoint_id,
        "parent_run_id": doc["checkpoint"]["run_id"],
        "parent_s1_run_manifest_hash": parent_hash,
        "parent_base_ancestry_hash": doc["checkpoint"]["base_ancestry_hash"],
        "step": doc["checkpoint"]["step"],
        "parent_run_class": parent_companion["run_class"],
    }


def _resolve_init_from(
    init_from: Mapping[str, Any] | None,
    parent_companion: Mapping[str, Any] | None,
    *,
    run_class: str,
) -> dict[str, Any] | None:
    """Resolve and cross-check `init_from` (design §6). `parent_run_class`
    is re-derived from the presented `parent_companion` (another
    S1-format run manifest -- the parent run's own companion), never read
    as a caller-typed pin."""
    if init_from is None:
        return None
    if not isinstance(init_from, Mapping):
        raise _refuse(f"init_from must be an object or null, got {type(init_from).__name__}")
    if parent_companion is None:
        raise _refuse(
            "init_from is present but no parent companion manifest was presented "
            "to derive parent_run_class from"
        )
    try:
        parent_hash = s2m._verify_companion(parent_companion)
    except s2m.S2RunManifestInvalid as exc:
        raise _refuse(str(exc)) from exc
    if parent_hash != init_from["parent_s1_run_manifest_hash"]:
        raise _refuse(
            f"init_from.parent_s1_run_manifest_hash ({init_from['parent_s1_run_manifest_hash']}) "
            f"does not match the presented parent companion manifest's own hash ({parent_hash})"
        )
    derived_parent_run_class = parent_companion["run_class"]
    if init_from["parent_run_class"] != derived_parent_run_class:
        raise _refuse(
            f"init_from.parent_run_class ({init_from['parent_run_class']!r}) disagrees "
            "with the one derived from the presented parent companion manifest "
            f"({derived_parent_run_class!r}) -- parent_run_class is never caller-supplied"
        )
    if derived_parent_run_class == "discovery" and run_class == "promotion":
        raise _refuse(
            "init_from.parent_run_class is 'discovery' but run_class is 'promotion' "
            "-- a discovery parent may not be warm-started into a promotion run "
            "(RFC005 §12.1)"
        )
    return dict(init_from)


def _validated_base(
    *,
    run_class: str,
    gradient_bearing: bool,
    run_seed: int,
    companion: Mapping[str, Any],
    stage_policy: Mapping[str, Any],
    matrix: Mapping[str, Any],
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    census: Mapping[str, Any],
    typed_supervision_floor: Mapping[str, Any],
    heads_frozen_for_no_supervision: Sequence[str],
    lr_reduced_micro: int,
    init_from: Mapping[str, Any] | None,
    parent_companion: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate every input, re-derive every hash from the presented
    documents, and return the manifest payload with `run_id` and the
    self-hash OMITTED (the addendum A projection)."""
    if run_class not in _RUN_CLASSES:
        raise _refuse(f"run_class must be one of {_RUN_CLASSES}, got {run_class!r}")
    if type(gradient_bearing) is not bool:
        raise _refuse(
            f"gradient_bearing must be a declared boolean, got {gradient_bearing!r}"
        )
    # `type is int` on purpose: `bool` is an `int` subclass.
    if type(run_seed) is not int or run_seed < 0:
        raise _refuse(f"run_seed must be an integer >= 0, got {run_seed!r}")
    if type(lr_reduced_micro) is not int or lr_reduced_micro < 0:
        raise _refuse(
            f"lr_reduced_micro must be an integer >= 0, got {lr_reduced_micro!r}"
        )

    validate_stage_data_matrix(matrix)
    validate_s2_sampler_policy(sampler_policy)
    validate_s2_exposure_policy(exposure_policy)
    # `s2_run_manifest`'s helpers raise its own `S2RunManifestInvalid`; this
    # module re-raises everything it reuses as `S3RunManifestInvalid` so a
    # caller catching this module's refusal type sees every refusal,
    # including the three checks reused verbatim from the S2 builder.
    try:
        plan_hash = s2m.verify_ordered_exposure_plan_against(
            plan=plan,
            run_seed=run_seed,
            matrix=matrix,
            sampler_policy=sampler_policy,
            exposure_policy=exposure_policy,
        )
        binding_hash = s2m._verify_companion(companion)
        s2m._verify_companion_consistency(
            companion, run_seed=run_seed, sampler_policy=sampler_policy
        )
    except s2m.S2RunManifestInvalid as exc:
        raise _refuse(str(exc)) from exc

    sp.validate_stage_policy(stage_policy)
    l_typed_head_ids = _l_typed_head_ids(stage_policy)

    floor = _validate_typed_supervision_floor(typed_supervision_floor)
    census_hash = _verify_census_against_companion(census, companion)
    frozen_heads = _validate_heads_frozen_for_no_supervision(
        heads_frozen_for_no_supervision,
        l_typed_head_ids=l_typed_head_ids,
        census=census,
        floor=floor,
    )

    resolved_init_from = _resolve_init_from(
        init_from, parent_companion, run_class=run_class
    )

    return {
        "s3_run_manifest_schema_version": S3_RUN_MANIFEST_SCHEMA_VERSION,
        "run_class": run_class,
        "gradient_bearing": gradient_bearing,
        "stage_id": "S3",
        "run_seed": run_seed,
        "binding_s1_run_manifest_hash": binding_hash,
        "s3_stage_policy_hash": sp.stage_policy_hash(stage_policy),
        "stage_data_matrix_hash": stage_data_matrix_hash(matrix),
        "s2_sampler_policy_hash": s2_sampler_policy_hash(sampler_policy),
        "s2_exposure_policy_hash": s2_exposure_policy_hash(exposure_policy),
        "ordered_exposure_plan_hash": plan_hash,
        "typed_supervision_census_hash": census_hash,
        "typed_supervision_floor": floor,
        "heads_frozen_for_no_supervision": sorted(frozen_heads),
        "typed_integer_codec": TYPED_INTEGER_CODEC,
        "lr_reduced_micro": lr_reduced_micro,
        "init_from": resolved_init_from,
    }


def build_s3_run_manifest(
    *,
    run_id: str,
    run_class: str,
    gradient_bearing: bool,
    run_seed: int,
    companion: Mapping[str, Any],
    stage_policy: Mapping[str, Any],
    matrix: Mapping[str, Any],
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    census: Mapping[str, Any],
    typed_supervision_floor: Mapping[str, Any],
    heads_frozen_for_no_supervision: Sequence[str],
    lr_reduced_micro: int,
    init_from: Mapping[str, Any] | None = None,
    parent_companion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the hash-frozen S3 run manifest for the pinned inputs.

    Every hash is re-derived from the presented document (never copied from
    a caller-supplied value). The caller supplies `run_id`; :func:`freeze`
    is the only place a `run_id` is derived (addendum A).
    """
    if not isinstance(run_id, str) or not run_id:
        raise _refuse(f"run_id must be a non-empty string, got {run_id!r}")
    base = _validated_base(
        run_class=run_class,
        gradient_bearing=gradient_bearing,
        run_seed=run_seed,
        companion=companion,
        stage_policy=stage_policy,
        matrix=matrix,
        sampler_policy=sampler_policy,
        exposure_policy=exposure_policy,
        plan=plan,
        census=census,
        typed_supervision_floor=typed_supervision_floor,
        heads_frozen_for_no_supervision=heads_frozen_for_no_supervision,
        lr_reduced_micro=lr_reduced_micro,
        init_from=init_from,
        parent_companion=parent_companion,
    )
    payload: dict[str, Any] = {**base, "run_id": run_id}
    payload["s3_run_manifest_hash"] = domain_hash(
        S3_RUN_MANIFEST_DOMAIN,
        {k: v for k, v in payload.items() if k != "s3_run_manifest_hash"},
    )
    errors = list(_manifest_validator().iter_errors(payload))
    if errors:
        raise _refuse(
            f"manifest schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )
    return payload


def verify_s3_run_manifest(manifest: Mapping[str, Any]) -> None:
    """Schema validation PLUS the self-hash check: a manifest whose
    `s3_run_manifest_hash` does not match its payload was tampered with (or
    never produced by this module) and is refused."""
    errors = list(_manifest_validator().iter_errors(manifest))
    if errors:
        raise _refuse(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )
    payload = {k: v for k, v in manifest.items() if k != "s3_run_manifest_hash"}
    if domain_hash(S3_RUN_MANIFEST_DOMAIN, payload) != manifest["s3_run_manifest_hash"]:
        raise _refuse("s3_run_manifest_hash does not verify against the payload")


def load_s3_run_manifest(path: Path | str) -> dict[str, Any]:
    """Read a frozen S3 run manifest from disk, verified on the way in
    (schema + self-hash)."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise S3RunManifestInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise S3RunManifestInvalid(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise S3RunManifestInvalid(f"{path} must hold a JSON object")
    verify_s3_run_manifest(payload)
    return payload


def freeze(
    *,
    companion: Mapping[str, Any],
    stage_policy: Mapping[str, Any],
    matrix: Mapping[str, Any],
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
    plan: Mapping[str, Any],
    census: Mapping[str, Any],
    typed_supervision_floor: Mapping[str, Any],
    heads_frozen_for_no_supervision: Sequence[str],
    run_class: str,
    gradient_bearing: bool,
    run_seed: int,
    lr_reduced_micro: int,
    out_dir: Path,
    init_from: Mapping[str, Any] | None = None,
    parent_companion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the S3 run manifest for the pinned inputs, writing
    `s3_run_manifest.json` under `out_dir` (refusing to overwrite).

    The `run_id` is derived the addendum A way: `make_id` over every
    manifest field except `run_id` and the self-hash.
    """
    base = _validated_base(
        run_class=run_class,
        gradient_bearing=gradient_bearing,
        run_seed=run_seed,
        companion=companion,
        stage_policy=stage_policy,
        matrix=matrix,
        sampler_policy=sampler_policy,
        exposure_policy=exposure_policy,
        plan=plan,
        census=census,
        typed_supervision_floor=typed_supervision_floor,
        heads_frozen_for_no_supervision=heads_frozen_for_no_supervision,
        lr_reduced_micro=lr_reduced_micro,
        init_from=init_from,
        parent_companion=parent_companion,
    )
    run_id = make_id(RUN_ID_PREFIX, RUN_ID_DOMAIN, dict(base))
    manifest = build_s3_run_manifest(
        run_id=run_id,
        run_class=run_class,
        gradient_bearing=gradient_bearing,
        run_seed=run_seed,
        companion=companion,
        stage_policy=stage_policy,
        matrix=matrix,
        sampler_policy=sampler_policy,
        exposure_policy=exposure_policy,
        plan=plan,
        census=census,
        typed_supervision_floor=typed_supervision_floor,
        heads_frozen_for_no_supervision=heads_frozen_for_no_supervision,
        lr_reduced_micro=lr_reduced_micro,
        init_from=init_from,
        parent_companion=parent_companion,
    )

    out_dir = Path(out_dir)
    manifest_path = out_dir / S3_RUN_MANIFEST_FILENAME
    if manifest_path.exists():
        raise S3RunManifestInvalid(
            f"{manifest_path} already exists -- refusing to overwrite a frozen "
            "run manifest"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    return {
        "run_id": manifest["run_id"],
        "s3_run_manifest_hash": manifest["s3_run_manifest_hash"],
        "out_dir": str(out_dir),
        "typed_supervision_census_hash": manifest["typed_supervision_census_hash"],
        "ordered_exposure_plan_hash": manifest["ordered_exposure_plan_hash"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze an s3_run_manifest for the pinned S3 inputs."
    )
    parser.add_argument(
        "--binding-run-manifest",
        type=Path,
        required=True,
        help="the companion S1-format run manifest this S3 run is bound through",
    )
    parser.add_argument("--stage-policy", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--sampler-policy", type=Path, required=True)
    parser.add_argument("--exposure-policy", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--min-supervised-per-head", type=int, default=1000)
    parser.add_argument("--min-lineages-per-head", type=int, default=100)
    parser.add_argument(
        "--head-frozen-for-no-supervision",
        dest="heads_frozen_for_no_supervision",
        action="append",
        default=[],
    )
    parser.add_argument("--run-class", choices=_RUN_CLASSES, required=True)
    parser.add_argument("--gradient-bearing", action="store_true")
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--lr-reduced-micro", type=int, required=True)
    parser.add_argument("--parent-run-manifest", type=Path, default=None)
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="the parent S2 checkpoint (.pt) to warm-start from; requires "
        "--parent-run-manifest",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if bool(args.init_from) != bool(args.parent_run_manifest):
        raise _refuse(
            "--init-from and --parent-run-manifest must be supplied together or "
            "not at all"
        )
    parent_companion = (
        json.loads(args.parent_run_manifest.read_text())
        if args.parent_run_manifest
        else None
    )
    init_from = (
        init_from_block_from_checkpoint(args.init_from, parent_companion)
        if args.init_from
        else None
    )

    summary = freeze(
        companion=json.loads(args.binding_run_manifest.read_text()),
        stage_policy=sp.load_stage_policy(args.stage_policy),
        matrix=load_stage_data_matrix(args.matrix),
        sampler_policy=load_s2_sampler_policy(args.sampler_policy),
        exposure_policy=load_s2_exposure_policy(args.exposure_policy),
        plan=json.loads(args.plan.read_text()),
        census=json.loads(args.census.read_text()),
        typed_supervision_floor={
            "min_supervised_per_head": args.min_supervised_per_head,
            "min_lineages_per_head": args.min_lineages_per_head,
        },
        heads_frozen_for_no_supervision=args.heads_frozen_for_no_supervision,
        run_class=args.run_class,
        gradient_bearing=args.gradient_bearing,
        run_seed=args.run_seed,
        lr_reduced_micro=args.lr_reduced_micro,
        init_from=init_from,
        parent_companion=parent_companion,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

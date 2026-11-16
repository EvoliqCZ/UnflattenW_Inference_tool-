"""S4 run manifest (RFC005 §12/§26, design §7.1, addendum A construction).

The manifest is the identity one S4 discovery run executes under, built the
same way `s2_run_manifest` builds S2's: every field is a derivation from a
frozen artefact, `run_id` is a pure function of the other fields, and every
hash is re-derived from the presented document rather than trusted as a
caller-typed value.

S4 has no run-manifest shape of its own for cache identity, allowlist or
ancestry either -- exactly like S2, it names a companion S1-format run
manifest by hash (`binding_s1_run_manifest_hash`, re-derived via
`s2_run_manifest._verify_companion`, reused verbatim) and refuses one whose
own `run_config` disagrees with the S4 inputs it is paired with (sampler
order, budget, seed -- the same three checks `s2_run_manifest` applies,
reused via `s2_run_manifest._verify_companion_consistency`).

Three things are new relative to S2:

- **`s4_stage_policy_hash`** is bound to the committed `s4_stage_policy.json`
  (via `stage_policy.load_stage_policy` / `stage_policy_hash`), and the
  `L_typed` head ids the census/floor logic below reasons about come from
  that policy's `task_view_allowlist`, never restated here.
- **The typed supervision census.** `typed_supervision_census_hash` pins a
  frozen `s4_typed_census` payload; its own `census_hash` must verify, and
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

**`selection_policy_hash`** (1.1.0, RFC005-ERRATA-005 E3): for
`run_class="promotion"` the caller's `selection_policy` (RFC005 §40.1
lexicographic gates) is REQUIRED, validated
(`s4_selection_policy.validate_s4_selection_policy`), bound by its own
`s4_selection_policy_hash`, and freezing is REFUSED while any gate still
carries a placeholder floor (`has_placeholder_floor`) -- a promotion run may
not freeze before its selection thresholds are ratified. For
`run_class="discovery"` the policy is optional: bound if presented, never
required, and never floor-checked. The field is deliberately absent from
the schema's unconditional `required` list (only a `run_class="promotion"`
manifest requires it, via the schema's own `if`/`then`), so a discovery
manifest frozen under the prior 1.0.0 schema -- which never wrote this key
at all -- still loads.
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
from unflatten.model import events_vocab
from unflatten.model import lr_schedule as lrs
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
from unflatten.model.s4_selection_policy import (
    S4SelectionPolicyInvalid,
    has_placeholder_floor,
    load_s4_selection_policy,
    s4_selection_policy_hash,
)
from unflatten.model.s4_typed_census import _CENSUS_DOMAIN
from unflatten.model.stage_data_matrix import (
    load_stage_data_matrix,
    stage_data_matrix_hash,
    validate_stage_data_matrix,
)
from unflatten.model.typed_head_config import new_supervised_pointers, pointer_supervision_key
from unflatten.model.typed_head_config import typed_head_config_hash as _current_thc_hash

_SCHEMAS_DIR = Path(__file__).parent / "schemas"

#: 1.3.0: `consistency_lambda_milli` (RFC004-ERRATA-004 item R3, draft
#: RFC004-ERRATA-006, v16 only) -- optional, never required, absent from
#: the unconditional `required` list on purpose so a manifest frozen under
#: the prior 1.2.0 schema (no such key at all) still loads;
#: `s4_train.main()` treats an absent key as `0` (off). 1.2.0:
#: `mask_inactive_union_arms` (RFC005-ERRATA-008 E1, PROPOSED, v16
#: only) -- optional, never required, absent from the unconditional
#: `required` list on purpose so a manifest frozen under the prior 1.1.0
#: schema (no such key at all) still loads; `s4_train.main()` treats an
#: absent key as `False`. 1.1.0: `selection_policy_hash`
#: (RFC005-ERRATA-005 E3) -- required for `run_class="promotion"` (via the
#: schema's own `if`/`then`), optional and never required for `discovery`.
#: 1.0.0: the initial S4 manifest shape.
S4_RUN_MANIFEST_SCHEMA_VERSION = "1.3.0"

#: 1.4.0 (v16): optional keys -- `level_router_policy_hash` (RFC007 variant B,
#: PROPOSED), `structural_table_region` (DEC-055 V6b), `visual_input`
#: (DEC-056 addendum A: the on-the-fly ViT, its policy hash, DeepStack tap
#: and augmentation seed), `data_quality_gate_hash` (DEC-057 point 6: the
#: passing gate result binding the v16 caches; `quality_gate.gate.
#: require_data_quality_gate` returns `None` for pre-v16 caches, so the key
#: is then absent) and `lr_schedule` (the dormant learning-rate tail of
#: `model.lr_schedule`; absent == the constant LR every frozen run used).
#: Each is WRITTEN ONLY WHEN SELECTED, and the
#: version string is 1.4.0 only when at least one is written: a manifest
#: that selects none of them is byte-identical to a 1.3.0 one (same keys,
#: same hash, same `run_id`), so nothing a v15b-shaped freeze produces
#: moves. `s4_train.main()` reads an absent key as the v15b behaviour and
#: cross-checks each against its CLI.
#:
#: R1 (RFC004-ERRATA-006 E1, `explicit_zero_bin_pointers`) is REFUSED here,
#: not written: the explicit zero bin exists only on a `bins_plus_residual`
#: integer head (`IntegerHeadConfig` raises under `direct_regression`), and
#: an S4 manifest pins `typed_integer_codec = direct_regression` (D5). R1 in
#: the dense fallback therefore needs an integer-codec decision first -- and
#: a clean start, since it widens two heads' state_dict (E1).
S4_RUN_MANIFEST_SCHEMA_VERSION_V16 = "1.4.0"

#: 1.5.0 (v17, RFC004-ERRATA-012 E4): optional keys `collapsed_region_rule`
#: (written only for "exclude"), `story_decision_weight_milli` (int >= 1,
#: only when != 1000), `alignment_type_weights_milli` ({events_vocab node
#: type: int >= 1}, only when non-empty) and `page_uniform_boost` (the
#: `boost` block of the page-uniform plan sidecar, only when it has one).
#: The version is 1.5.0 iff at least one is written, else 1.4.0 iff a v16
#: key is, else 1.3.0 -- a manifest selecting none of them is byte-identical
#: to the earlier one. Zero weights are refused (a present label would turn
#: into "no label"; an all-zero batch would divide 0/0).
S4_RUN_MANIFEST_SCHEMA_VERSION_V17 = "1.5.0"

_COLLAPSED_REGION_RULE_DEFAULT = "abstain_container"
_COLLAPSED_REGION_RULE_VALUES = ("abstain_container", "exclude")
_STORY_DECISION_WEIGHT_MILLI_DEFAULT = 1000
_PAGE_UNIFORM_BOOST_KEYS = ("sets_sha256", "factor", "draws")

#: DEC-055 V6b values (`s4_typed_loss.STRUCTURAL_TABLE_REGION_POLICIES`,
#: repeated here as data so this module does not import the loss module).
_STRUCTURAL_TABLE_REGION_DEFAULT = "abstain_not_visible"
_STRUCTURAL_TABLE_REGION_VALUES = ("abstain_not_visible", "exclude")
_VISUAL_INPUT_KEYS = frozenset(
    {"mode", "visual_input_policy_hash", "visual_feature_tap", "augmentation"}
)
#: Fresh-S4 speed knobs that change the model's inputs, present only when on
#: (`onfly_visual.visual_input_manifest_block`); the schema pins their values.
_VISUAL_INPUT_OPTIONAL_KEYS = frozenset({"feature_cache", "batch_pages", "external_views"})
_VISUAL_INPUT_AUGMENTATION_KEYS = frozenset(
    {"seed", "policy_id", "policy_hash", "draw_domain", "selection"}
)

#: The shared filename contract (S1 owns `s1_run_manifest.json`, S2 owns
#: `s2_run_manifest.json`; the three prefixes cannot collide).
S4_RUN_MANIFEST_FILENAME = "s4_run_manifest.json"

RUN_ID_PREFIX = "s4run"

#: This module is the sole owner of these two domains.
RUN_ID_DOMAIN = "rfc005.s4-run-id/v1"
S4_RUN_MANIFEST_DOMAIN = "rfc005.s4-run-manifest/v1"

_RUN_CLASSES = ("discovery", "promotion")

#: D5: the model's only supported integer codec today (candidate_model.py);
#: pinned, never a caller choice (design §11 D5).
TYPED_INTEGER_CODEC = "direct_regression"

#: Why R1 cannot be frozen into an S4 manifest (module constant
#: `S4_RUN_MANIFEST_SCHEMA_VERSION_V16`'s note); `s4_train` repeats it.
R1_REFUSAL = (
    "R1 (explicit_zero_bin_pointers, RFC004-ERRATA-006 E1) applies only to "
    "bins_plus_residual integer heads, and an S4 run pins typed_integer_codec "
    f"{TYPED_INTEGER_CODEC!r} (D5) -- enabling R1 in the dense fallback needs an "
    "integer-codec decision first"
)

#: The one loss family whose task_view_allowlist rows name the typed heads
#: this manifest's census/floor logic reasons about.
_L_TYPED_FAMILY = "L_typed"


class S4RunManifestInvalid(ValueError):
    """The manifest (or one of the documents it binds) is not well-formed:
    a schema violation, a self-hash that does not verify, or a cross check
    that fails. A refusal, not a repair."""


@cache
def _manifest_validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMAS_DIR / "s4_run_manifest.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _refuse(message: str) -> S4RunManifestInvalid:
    return S4RunManifestInvalid(f"s4 run manifest refused: {message}")


def _l_typed_head_ids(stage_policy: Mapping[str, Any]) -> set[str]:
    """The `L_typed` head ids the committed stage policy declares, read
    from its `task_view_allowlist` rows (design §11: "the L_typed head_ids
    come from the committed policies/s4_stage_policy.json") -- never
    restated as a literal here."""
    heads: set[str] = set()
    for row in stage_policy["task_view_allowlist"]:
        if row["loss_family"] == _L_TYPED_FAMILY:
            heads.update(row["head_ids"])
    return heads


def _verify_census(census: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Verify the presented `s4_typed_census` payload's own self-hash and
    return `(census_hash, cache_identity_hash, expectation_hash,
    typed_head_config_hash)`.

    `typed_head_config_hash` is required INSIDE the payload, under
    `census_hash` (design §7/§9 R13, T8): without it, a palette or
    cardinality-cap change would leave a stale census self-consistent and
    valid -- `census_hash` only ever binds `cache_identity_hash` /
    `expectation_hash` / `counts` on its own.
    """
    if not isinstance(census, Mapping):
        raise _refuse(f"census must be an object, got {type(census).__name__}")
    required = {
        "cache_identity_hash", "expectation_hash", "typed_head_config_hash",
        "counts", "census_hash",
    }
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
    return (
        census["census_hash"],
        census["cache_identity_hash"],
        census["expectation_hash"],
        census["typed_head_config_hash"],
    )


def _verify_census_against_companion(
    census: Mapping[str, Any], companion: Mapping[str, Any]
) -> str:
    """Cross-check: the census's own `cache_identity_hash` /
    `expectation_hash` must equal the companion's -- a census counted over a
    different cache than the run's own companion binds is refused, not
    silently trusted. Also (design §9 R13): the census's own
    `typed_head_config_hash` must equal the CURRENT
    `typed_head_config.typed_head_config_hash()` -- a census computed
    under a since-changed palette/cap configuration is refused, not
    silently trusted either."""
    census_hash, cache_identity_hash, expectation_hash, typed_head_config_hash_value = (
        _verify_census(census)
    )
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
    current = _current_thc_hash()
    if typed_head_config_hash_value != current:
        raise _refuse(
            f"census typed_head_config_hash ({typed_head_config_hash_value}) does "
            f"not match the current typed_head_config.typed_head_config_hash() "
            f"({current}) -- a census computed under a different palette/cap "
            "configuration is refused, not silently trusted (design §9 R13)"
        )
    return census_hash


def _validate_typed_supervision_floor(floor: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(floor, Mapping):
        raise _refuse(
            f"typed_supervision_floor must be an object, got {type(floor).__name__}"
        )
    required = (
        "min_supervised_per_head", "min_lineages_per_head", "min_supervised_per_pointer",
    )
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

    Public and named for reuse: `s4_train`'s admission gate (design §7.3
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


def new_pointer_supervised_counts(census: Mapping[str, Any]) -> dict[str, int]:
    """`{pointer_supervision_key: supervised_count}` for every T2/T4 "new"
    pointer (`typed_head_config.new_supervised_pointers`, design §9 R8):
    `0` for a pointer the census's `counts.per_pointer` never recorded at
    all -- never absent, so a pointer with genuinely zero supervision
    cannot silently read as "unmeasured" instead of "below floor"."""
    per_pointer = census["counts"]["per_pointer"]
    return {
        pointer_supervision_key(head_id, node_type, pointer): (
            per_pointer.get(head_id, {}).get(pointer, {}).get("supervised", 0)
        )
        for head_id, node_type, pointer in new_supervised_pointers()
    }


def pointer_meets_floor(supervised: int, floor: Mapping[str, int]) -> bool:
    """True iff a pointer's census supervised count meets
    `typed_supervision_floor.min_supervised_per_pointer` (design §9 R8) --
    the per-POINTER analogue of `head_meets_floor`, public and named for
    reuse by `s4_train`'s own admission-time re-check."""
    return supervised >= floor["min_supervised_per_pointer"]


def _validate_low_supervision_pointers_disclosed(
    low_supervision_pointers_disclosed: Sequence[str],
    *,
    census: Mapping[str, Any],
    floor: Mapping[str, int],
) -> list[str]:
    """The per-POINTER analogue of `_validate_heads_frozen_for_no_
    supervision` (design §9 R8, §10.1 T8 point 4) -- but, unlike that
    field, this one gates NO parameter at training time (fix loop 2: a
    `PaletteHead` pointer's `head_id` is a whole group shared by many
    pointers, so freezing at pointer granularity is architecturally
    impossible without splitting head groups; that is why this field is
    named "disclosed", not "frozen"). It is a pure DISCLOSURE requirement:
    the existing head-level floor is computed over `expert_adapter.
    dense`'s AGGREGATE, so a single new pointer with almost no supervision
    (the design's own example: `font_family`'s 3 supervised targets, by
    analogy -- `font_family` itself is not census-measurable this way,
    §9 R8's fix loop 2 note) would otherwise pass silently inside it.
    Every `new_supervised_pointers` entry below `min_supervised_per_
    pointer` must be named here by exactly this module's `pointer_
    supervision_key` string, or the manifest refuses to freeze; a named
    pointer whose count actually meets the floor is a STALE disclosure,
    refused too (the same DEC-027 D2 "an unused freeze is a lie" spirit,
    applied to a claim about coverage rather than a frozen parameter)."""
    if not isinstance(low_supervision_pointers_disclosed, Sequence) or isinstance(
        low_supervision_pointers_disclosed, (str, bytes)
    ):
        raise _refuse(
            "low_supervision_pointers_disclosed must be a sequence of pointer "
            f"keys, got {type(low_supervision_pointers_disclosed).__name__}"
        )
    disclosed = list(low_supervision_pointers_disclosed)
    counts = new_pointer_supervised_counts(census)
    unknown = sorted(set(disclosed) - set(counts))
    if unknown:
        raise _refuse(
            f"low_supervision_pointers_disclosed names unknown pointer key(s): "
            f"{unknown!r} -- not one of typed_head_config.new_supervised_pointers() "
            "(design §9 R8)"
        )
    for key in disclosed:
        supervised = counts[key]
        if pointer_meets_floor(supervised, floor):
            raise _refuse(
                f"{key!r} is listed in low_supervision_pointers_disclosed but its "
                f"census supervised count ({supervised}) meets "
                f"min_supervised_per_pointer ({floor['min_supervised_per_pointer']}) "
                "-- a stale disclosure is a lie too (DEC-027 D2 rule, design §9 R8)"
            )
    missing = sorted(
        key for key, supervised in counts.items()
        if not pointer_meets_floor(supervised, floor) and key not in disclosed
    )
    if missing:
        raise _refuse(
            f"pointer(s) below min_supervised_per_pointer are not listed in "
            f"low_supervision_pointers_disclosed: {missing!r} -- a pointer with "
            "almost no supervision must be disclosed by name, never left silent "
            "inside its head_id aggregate (design §9 R8)"
        )
    return sorted(disclosed)


def init_from_block_from_checkpoint(
    checkpoint_path: Path | str, parent_companion: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive the S4 manifest's `init_from` block (design §6) from the real
    parent CHECKPOINT artefacts -- never accepted as a caller-typed JSON
    document. `s4_train.load_init_weights`'s `pinned` argument reads
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


def _resolve_selection_policy_hash(
    selection_policy: Mapping[str, Any] | None, *, run_class: str
) -> str | None:
    """RFC005-ERRATA-005 E3: bind `selection_policy_hash`.

    `run_class="promotion"` REQUIRES a `selection_policy`, validates it
    (`s4_selection_policy.validate_s4_selection_policy`, via
    `s4_selection_policy_hash`), and refuses freezing while any gate still
    carries a placeholder floor (`has_placeholder_floor`) -- a promotion
    run may not freeze before its selection thresholds are ratified.
    `run_class="discovery"` never requires it; a presented policy is still
    validated and bound, never floor-checked.
    """
    if selection_policy is None:
        if run_class == "promotion":
            raise _refuse(
                "run_class is 'promotion' but no selection_policy was presented -- "
                "RFC005-ERRATA-005 E3 requires the selection policy frozen before "
                "the first gradient"
            )
        return None
    policy = dict(selection_policy)
    try:
        policy_hash = s4_selection_policy_hash(policy)
    except S4SelectionPolicyInvalid as exc:
        raise _refuse(f"selection_policy refused: {exc}") from exc
    if run_class == "promotion" and has_placeholder_floor(policy):
        raise _refuse(
            "selection_policy has a placeholder floor (is_placeholder=true) -- a "
            "promotion run may not freeze without ratified selection floors "
            "(RFC005-ERRATA-005 E3)"
        )
    return policy_hash


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
    low_supervision_pointers_disclosed: Sequence[str],
    lr_reduced_micro: int,
    init_from: Mapping[str, Any] | None,
    parent_companion: Mapping[str, Any] | None,
    selection_policy: Mapping[str, Any] | None = None,
    mask_inactive_union_arms: bool = False,
    consistency_lambda_milli: int = 0,
    level_router_policy_hash: str | None = None,
    structural_table_region: str = _STRUCTURAL_TABLE_REGION_DEFAULT,
    explicit_zero_bin_pointers: Sequence[Sequence[str]] = (),
    visual_input: Mapping[str, Any] | None = None,
    data_quality_gate_hash: str | None = None,
    lr_schedule: Mapping[str, Any] | None = None,
    collapsed_region_rule: str = _COLLAPSED_REGION_RULE_DEFAULT,
    story_decision_weight_milli: int = _STORY_DECISION_WEIGHT_MILLI_DEFAULT,
    alignment_type_weights_milli: Mapping[str, int] | None = None,
    page_uniform_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every input, re-derive every hash from the presented
    documents, and return the manifest payload with `run_id` and the
    self-hash OMITTED (the addendum A projection).

    The v16 keys (`S4_RUN_MANIFEST_SCHEMA_VERSION_V16`) enter the payload
    only when selected; at their defaults the payload is exactly the 1.3.0
    one. A non-empty `explicit_zero_bin_pointers` (R1) is refused
    (`R1_REFUSAL`)."""
    if run_class not in _RUN_CLASSES:
        raise _refuse(f"run_class must be one of {_RUN_CLASSES}, got {run_class!r}")
    if type(gradient_bearing) is not bool:
        raise _refuse(
            f"gradient_bearing must be a declared boolean, got {gradient_bearing!r}"
        )
    if type(mask_inactive_union_arms) is not bool:
        raise _refuse(
            "mask_inactive_union_arms must be a declared boolean, got "
            f"{mask_inactive_union_arms!r}"
        )
    if type(consistency_lambda_milli) is not int or consistency_lambda_milli < 0:
        raise _refuse(
            "consistency_lambda_milli must be an integer >= 0, got "
            f"{consistency_lambda_milli!r}"
        )
    v17_keys = _validated_v17_keys(
        collapsed_region_rule=collapsed_region_rule,
        story_decision_weight_milli=story_decision_weight_milli,
        alignment_type_weights_milli=alignment_type_weights_milli,
        page_uniform_meta=page_uniform_meta,
    )
    v16_keys = _validated_v16_keys(
        structural_table_region=structural_table_region,
        explicit_zero_bin_pointers=explicit_zero_bin_pointers,
        visual_input=visual_input,
        init_from=init_from,
        level_router_policy_hash=level_router_policy_hash,
        data_quality_gate_hash=data_quality_gate_hash,
        lr_schedule=lr_schedule,
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
    # module re-raises everything it reuses as `S4RunManifestInvalid` so a
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
    disclosed_pointers = _validate_low_supervision_pointers_disclosed(
        low_supervision_pointers_disclosed, census=census, floor=floor,
    )

    resolved_init_from = _resolve_init_from(
        init_from, parent_companion, run_class=run_class
    )
    selection_policy_hash_value = _resolve_selection_policy_hash(
        selection_policy, run_class=run_class
    )

    return {
        "s4_run_manifest_schema_version": (
            S4_RUN_MANIFEST_SCHEMA_VERSION_V17
            if v17_keys
            else S4_RUN_MANIFEST_SCHEMA_VERSION_V16
            if v16_keys
            else S4_RUN_MANIFEST_SCHEMA_VERSION
        ),
        "run_class": run_class,
        "gradient_bearing": gradient_bearing,
        "stage_id": "S4",
        "run_seed": run_seed,
        "binding_s1_run_manifest_hash": binding_hash,
        "s4_stage_policy_hash": sp.stage_policy_hash(stage_policy),
        "stage_data_matrix_hash": stage_data_matrix_hash(matrix),
        "s2_sampler_policy_hash": s2_sampler_policy_hash(sampler_policy),
        "s2_exposure_policy_hash": s2_exposure_policy_hash(exposure_policy),
        "ordered_exposure_plan_hash": plan_hash,
        "typed_supervision_census_hash": census_hash,
        "typed_supervision_floor": floor,
        "heads_frozen_for_no_supervision": sorted(frozen_heads),
        "low_supervision_pointers_disclosed": sorted(disclosed_pointers),
        "typed_integer_codec": TYPED_INTEGER_CODEC,
        "typed_head_config_hash": _current_thc_hash(),
        "lr_reduced_micro": lr_reduced_micro,
        "init_from": resolved_init_from,
        "selection_policy_hash": selection_policy_hash_value,
        "mask_inactive_union_arms": mask_inactive_union_arms,
        "consistency_lambda_milli": consistency_lambda_milli,
        **v16_keys,
        **v17_keys,
    }


def _validated_v16_keys(
    *,
    structural_table_region: str,
    explicit_zero_bin_pointers: Sequence[Sequence[str]],
    visual_input: Mapping[str, Any] | None,
    init_from: Mapping[str, Any] | None,
    level_router_policy_hash: str | None = None,
    data_quality_gate_hash: str | None = None,
    lr_schedule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The 1.4.0 keys to write -- only the selected ones (module constant
    `S4_RUN_MANIFEST_SCHEMA_VERSION_V16`)."""
    keys: dict[str, Any] = {}
    if level_router_policy_hash is not None:
        if not _is_sha256(level_router_policy_hash):
            raise _refuse(
                "level_router_policy_hash must be a sha256:<64 hex> string or None, got "
                f"{level_router_policy_hash!r}"
            )
        # RFC007 (PROPOSED, v16 only): written only when selected.
        keys["level_router_policy_hash"] = level_router_policy_hash
    if structural_table_region not in _STRUCTURAL_TABLE_REGION_VALUES:
        raise _refuse(
            f"structural_table_region must be one of {_STRUCTURAL_TABLE_REGION_VALUES}, "
            f"got {structural_table_region!r}"
        )
    if structural_table_region != _STRUCTURAL_TABLE_REGION_DEFAULT:
        keys["structural_table_region"] = structural_table_region
    if explicit_zero_bin_pointers:
        raise _refuse(
            R1_REFUSAL
            + (
                "; it would also need a run without init_from (it widens two typed "
                "heads' state_dict)"
                if init_from is not None
                else ""
            )
        )
    if visual_input is not None:
        optional = set(visual_input) - _VISUAL_INPUT_KEYS
        if (
            not _VISUAL_INPUT_KEYS <= set(visual_input)
            or not optional <= _VISUAL_INPUT_OPTIONAL_KEYS
            or visual_input["mode"] != "onfly_vit"
        ):
            raise _refuse(
                f"visual_input must carry exactly {sorted(_VISUAL_INPUT_KEYS)} (plus optional "
                f"{sorted(_VISUAL_INPUT_OPTIONAL_KEYS)}) with mode 'onfly_vit', got "
                f"{dict(visual_input)!r}"
            )
        augmentation = visual_input["augmentation"]
        if augmentation is not None and set(augmentation) != _VISUAL_INPUT_AUGMENTATION_KEYS:
            raise _refuse(
                f"visual_input.augmentation must carry {sorted(_VISUAL_INPUT_AUGMENTATION_KEYS)}"
            )
        keys["visual_input"] = json.loads(json.dumps(dict(visual_input)))
    if data_quality_gate_hash is not None:
        if not isinstance(data_quality_gate_hash, str) or not data_quality_gate_hash:
            raise _refuse(
                f"data_quality_gate_hash must be a hash string, got {data_quality_gate_hash!r}"
            )
        keys["data_quality_gate_hash"] = data_quality_gate_hash
    if lr_schedule is not None:
        try:
            schedule = lrs.from_manifest_block(lr_schedule)
        except (KeyError, TypeError, ValueError) as exc:
            raise _refuse(f"lr_schedule is not a valid block: {exc!r}") from exc
        # Written only when selected: absent == the constant LR (see `lr_schedule`).
        keys["lr_schedule"] = lrs.to_manifest_block(schedule)
    return keys


def _milli_weight(name: str, value: object) -> int:
    # `type is int` on purpose: `bool` is an `int` subclass.
    if type(value) is not int or value < 1:
        raise _refuse(f"{name} must be an integer >= 1 (zero is refused), got {value!r}")
    return value


def _validated_v17_keys(
    *,
    collapsed_region_rule: str,
    story_decision_weight_milli: int,
    alignment_type_weights_milli: Mapping[str, int] | None,
    page_uniform_meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The 1.5.0 keys to write -- only the selected ones (module constant
    `S4_RUN_MANIFEST_SCHEMA_VERSION_V17`)."""
    keys: dict[str, Any] = {}
    if collapsed_region_rule not in _COLLAPSED_REGION_RULE_VALUES:
        raise _refuse(
            f"collapsed_region_rule must be one of {_COLLAPSED_REGION_RULE_VALUES}, "
            f"got {collapsed_region_rule!r}"
        )
    if collapsed_region_rule != _COLLAPSED_REGION_RULE_DEFAULT:
        keys["collapsed_region_rule"] = collapsed_region_rule
    story = _milli_weight("story_decision_weight_milli", story_decision_weight_milli)
    if story != _STORY_DECISION_WEIGHT_MILLI_DEFAULT:
        keys["story_decision_weight_milli"] = story
    if alignment_type_weights_milli:
        known = events_vocab.node_types("word-visual-v1.1")
        for node_type in alignment_type_weights_milli:
            if node_type not in known:
                raise _refuse(
                    f"alignment_type_weights_milli key {node_type!r} is not an "
                    "events_vocab node type"
                )
        keys["alignment_type_weights_milli"] = {
            t: _milli_weight(f"alignment_type_weights_milli[{t!r}] (milli)", w)
            for t, w in sorted(alignment_type_weights_milli.items())
        }
    boost = None if page_uniform_meta is None else page_uniform_meta.get("boost")
    if boost is not None:
        missing = [k for k in _PAGE_UNIFORM_BOOST_KEYS if k not in boost]
        if missing:
            raise _refuse(f"page-uniform sidecar boost block lacks {missing}")
        keys["page_uniform_boost"] = {k: boost[k] for k in _PAGE_UNIFORM_BOOST_KEYS}
    return keys


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(c in "0123456789abcdef" for c in value[7:])
    )


def build_s4_run_manifest(
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
    low_supervision_pointers_disclosed: Sequence[str],
    lr_reduced_micro: int,
    init_from: Mapping[str, Any] | None = None,
    parent_companion: Mapping[str, Any] | None = None,
    selection_policy: Mapping[str, Any] | None = None,
    mask_inactive_union_arms: bool = False,
    consistency_lambda_milli: int = 0,
    level_router_policy_hash: str | None = None,
    structural_table_region: str = _STRUCTURAL_TABLE_REGION_DEFAULT,
    explicit_zero_bin_pointers: Sequence[Sequence[str]] = (),
    visual_input: Mapping[str, Any] | None = None,
    data_quality_gate_hash: str | None = None,
    lr_schedule: Mapping[str, Any] | None = None,
    collapsed_region_rule: str = _COLLAPSED_REGION_RULE_DEFAULT,
    story_decision_weight_milli: int = _STORY_DECISION_WEIGHT_MILLI_DEFAULT,
    alignment_type_weights_milli: Mapping[str, int] | None = None,
    page_uniform_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the hash-frozen S4 run manifest for the pinned inputs.

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
        low_supervision_pointers_disclosed=low_supervision_pointers_disclosed,
        lr_reduced_micro=lr_reduced_micro,
        init_from=init_from,
        parent_companion=parent_companion,
        selection_policy=selection_policy,
        mask_inactive_union_arms=mask_inactive_union_arms,
        consistency_lambda_milli=consistency_lambda_milli,
        level_router_policy_hash=level_router_policy_hash,
        structural_table_region=structural_table_region,
        explicit_zero_bin_pointers=explicit_zero_bin_pointers,
        visual_input=visual_input,
        data_quality_gate_hash=data_quality_gate_hash,
        lr_schedule=lr_schedule,
        collapsed_region_rule=collapsed_region_rule,
        story_decision_weight_milli=story_decision_weight_milli,
        alignment_type_weights_milli=alignment_type_weights_milli,
        page_uniform_meta=page_uniform_meta,
    )
    payload: dict[str, Any] = {**base, "run_id": run_id}
    payload["s4_run_manifest_hash"] = domain_hash(
        S4_RUN_MANIFEST_DOMAIN,
        {k: v for k, v in payload.items() if k != "s4_run_manifest_hash"},
    )
    errors = list(_manifest_validator().iter_errors(payload))
    if errors:
        raise _refuse(
            f"manifest schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )
    return payload


def verify_s4_run_manifest(manifest: Mapping[str, Any]) -> None:
    """Schema validation PLUS the self-hash check: a manifest whose
    `s4_run_manifest_hash` does not match its payload was tampered with (or
    never produced by this module) and is refused."""
    errors = list(_manifest_validator().iter_errors(manifest))
    if errors:
        raise _refuse(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )
    payload = {k: v for k, v in manifest.items() if k != "s4_run_manifest_hash"}
    if domain_hash(S4_RUN_MANIFEST_DOMAIN, payload) != manifest["s4_run_manifest_hash"]:
        raise _refuse("s4_run_manifest_hash does not verify against the payload")


def load_s4_run_manifest(path: Path | str) -> dict[str, Any]:
    """Read a frozen S4 run manifest from disk, verified on the way in
    (schema + self-hash)."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise S4RunManifestInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise S4RunManifestInvalid(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise S4RunManifestInvalid(f"{path} must hold a JSON object")
    verify_s4_run_manifest(payload)
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
    low_supervision_pointers_disclosed: Sequence[str],
    run_class: str,
    gradient_bearing: bool,
    run_seed: int,
    lr_reduced_micro: int,
    out_dir: Path,
    init_from: Mapping[str, Any] | None = None,
    parent_companion: Mapping[str, Any] | None = None,
    selection_policy: Mapping[str, Any] | None = None,
    mask_inactive_union_arms: bool = False,
    consistency_lambda_milli: int = 0,
    level_router_policy_hash: str | None = None,
    structural_table_region: str = _STRUCTURAL_TABLE_REGION_DEFAULT,
    explicit_zero_bin_pointers: Sequence[Sequence[str]] = (),
    visual_input: Mapping[str, Any] | None = None,
    data_quality_gate_hash: str | None = None,
    lr_schedule: Mapping[str, Any] | None = None,
    collapsed_region_rule: str = _COLLAPSED_REGION_RULE_DEFAULT,
    story_decision_weight_milli: int = _STORY_DECISION_WEIGHT_MILLI_DEFAULT,
    alignment_type_weights_milli: Mapping[str, int] | None = None,
    page_uniform_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the S4 run manifest for the pinned inputs, writing
    `s4_run_manifest.json` under `out_dir` (refusing to overwrite).

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
        low_supervision_pointers_disclosed=low_supervision_pointers_disclosed,
        lr_reduced_micro=lr_reduced_micro,
        init_from=init_from,
        parent_companion=parent_companion,
        selection_policy=selection_policy,
        mask_inactive_union_arms=mask_inactive_union_arms,
        consistency_lambda_milli=consistency_lambda_milli,
        level_router_policy_hash=level_router_policy_hash,
        structural_table_region=structural_table_region,
        explicit_zero_bin_pointers=explicit_zero_bin_pointers,
        visual_input=visual_input,
        data_quality_gate_hash=data_quality_gate_hash,
        lr_schedule=lr_schedule,
        collapsed_region_rule=collapsed_region_rule,
        story_decision_weight_milli=story_decision_weight_milli,
        alignment_type_weights_milli=alignment_type_weights_milli,
        page_uniform_meta=page_uniform_meta,
    )
    run_id = make_id(RUN_ID_PREFIX, RUN_ID_DOMAIN, dict(base))
    manifest = build_s4_run_manifest(
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
        low_supervision_pointers_disclosed=low_supervision_pointers_disclosed,
        lr_reduced_micro=lr_reduced_micro,
        init_from=init_from,
        parent_companion=parent_companion,
        selection_policy=selection_policy,
        mask_inactive_union_arms=mask_inactive_union_arms,
        consistency_lambda_milli=consistency_lambda_milli,
        level_router_policy_hash=level_router_policy_hash,
        structural_table_region=structural_table_region,
        explicit_zero_bin_pointers=explicit_zero_bin_pointers,
        visual_input=visual_input,
        data_quality_gate_hash=data_quality_gate_hash,
        lr_schedule=lr_schedule,
        collapsed_region_rule=collapsed_region_rule,
        story_decision_weight_milli=story_decision_weight_milli,
        alignment_type_weights_milli=alignment_type_weights_milli,
        page_uniform_meta=page_uniform_meta,
    )

    out_dir = Path(out_dir)
    manifest_path = out_dir / S4_RUN_MANIFEST_FILENAME
    if manifest_path.exists():
        raise S4RunManifestInvalid(
            f"{manifest_path} already exists -- refusing to overwrite a frozen "
            "run manifest"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    return {
        "run_id": manifest["run_id"],
        "s4_run_manifest_hash": manifest["s4_run_manifest_hash"],
        "out_dir": str(out_dir),
        "typed_supervision_census_hash": manifest["typed_supervision_census_hash"],
        "ordered_exposure_plan_hash": manifest["ordered_exposure_plan_hash"],
        "selection_policy_hash": manifest["selection_policy_hash"],
    }


def data_quality_gate_hash_for(
    cache_manifests: Mapping[str, Mapping[str, Any]], gate_path: Path | None
) -> str | None:
    """DEC-057 point 6 for S4: `{gate split name: cache manifest}` (`train`,
    `val_iid`) through `quality_gate.gate.require_data_quality_gate` --
    `None` for pre-v16 caches (the manifest keeps its 1.3.0 identity), the
    passing result's hash for v16 caches, a refusal otherwise. Shared by the
    freeze CLI and `s4_train.main`, so both derive the same value."""
    from unflatten.dataset.quality_gate.gate import (
        DataQualityGateRefused,
        require_data_quality_gate,
    )

    try:
        return require_data_quality_gate(dict(cache_manifests), gate_path)
    except DataQualityGateRefused as exc:
        raise _refuse(f"DEC-057 data-quality gate: {exc}") from exc


def _split_manifests(specs: Sequence[str]) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for spec in specs:
        split, _, path = spec.partition("=")
        if not split or not path or split in manifests:
            raise _refuse(f"--cache-manifest expects distinct SPLIT=PATH, got {spec!r}")
        manifests[split] = json.loads(Path(path).read_text())
    return manifests


def _alignment_weights(pairs: Sequence[str]) -> dict[str, int]:
    """`--alignment-type-weight-milli TYPE=INT` (repeated) -> `{type: int}`."""
    weights: dict[str, int] = {}
    for pair in pairs:
        node_type, sep, raw = pair.partition("=")
        if not sep or not node_type or not raw.isdecimal() or int(raw) < 1:
            raise _refuse(
                f"--alignment-type-weight-milli expects TYPE=<integer >= 1>, got {pair!r}"
            )
        if node_type in weights:
            raise _refuse(f"--alignment-type-weight-milli repeats type {node_type!r}")
        weights[node_type] = int(raw)
    return weights


def _r1_pointers() -> frozenset[tuple[str, str]]:
    from unflatten.model.candidate_model import R1_EXPLICIT_ZERO_BIN_POINTERS

    return R1_EXPLICIT_ZERO_BIN_POINTERS


def _level_router_policy_hash(path: Path) -> str:
    from unflatten.model.level_router.policy import load_level_router_policy

    return load_level_router_policy(path).policy_hash()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze an s4_run_manifest for the pinned S4 inputs."
    )
    parser.add_argument(
        "--binding-run-manifest",
        type=Path,
        required=True,
        help="the companion S1-format run manifest this S4 run is bound through",
    )
    parser.add_argument("--stage-policy", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--sampler-policy", type=Path, required=True)
    parser.add_argument("--exposure-policy", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--min-supervised-per-head", type=int, default=1000)
    parser.add_argument("--min-lineages-per-head", type=int, default=100)
    parser.add_argument("--min-supervised-per-pointer", type=int, default=1000)
    parser.add_argument(
        "--head-frozen-for-no-supervision",
        dest="heads_frozen_for_no_supervision",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--low-supervision-pointer-disclosed",
        dest="low_supervision_pointers_disclosed",
        action="append",
        default=[],
        help="a typed_head_config.pointer_supervision_key() string (design "
        "§9 R8) for a new T2/T4 pointer whose census supervised count is "
        "below --min-supervised-per-pointer",
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
    parser.add_argument(
        "--selection-policy",
        type=Path,
        default=None,
        help="RFC005 §40 checkpoint selection policy JSON (RFC005-ERRATA-005 "
        "E3); required for --run-class promotion, optional for discovery",
    )
    parser.add_argument(
        "--structural-table-region",
        choices=list(_STRUCTURAL_TABLE_REGION_VALUES),
        default=_STRUCTURAL_TABLE_REGION_DEFAULT,
        help="DEC-055 V6b (v16): 'exclude' drops boxless structural table/row/cell "
        "records from the region channel. Written to the manifest only when not the "
        "default; s4_train must be launched with the same value.",
    )
    parser.add_argument(
        "--explicit-zero-bin-r1",
        action="store_true",
        help="RFC004-ERRATA-006 E1 (R1, v16): REFUSED for S4 -- the explicit zero bin "
        "exists only on bins_plus_residual heads and S4 pins direct_regression (D5).",
    )
    parser.add_argument(
        "--visual-input",
        type=Path,
        default=None,
        help="DEC-056 addendum A: a JSON visual_input block "
        "(onfly_visual.visual_input_manifest_block) -- the on-the-fly ViT policy hash, "
        "DeepStack tap and augmentation seed/policy hash this run trains under.",
    )
    parser.add_argument(
        "--cache-manifest", action="append", default=[], metavar="SPLIT=PATH",
        help="DEC-057: an exact-text cache manifest this run trains/evaluates on, "
        "keyed by the gate's split name (train, val_iid). A cache built under a v16 "
        "target policy requires --data-quality-gate; pre-v16 caches record nothing.",
    )
    parser.add_argument(
        "--data-quality-gate", type=Path, default=None,
        help="DEC-057: the passing data-quality-gate.json binding the caches; its "
        "gate_result_hash enters the manifest as data_quality_gate_hash.",
    )
    parser.add_argument(
        "--mask-inactive-union-arms",
        action="store_true",
        help="RFC005-ERRATA-008 E1 / T1 (ratified for v16, DEC-048; on in the next run, "
        "DEC-049): bind union-arm masking; s4_train must then pass the same flag.",
    )
    parser.add_argument(
        "--consistency-lambda-milli",
        type=int,
        default=0,
        help="RFC004-ERRATA-006 E2 / R3 weight in milli-units (DEC-049 point 3: 0 = off).",
    )
    parser.add_argument(
        "--lr-schedule-start-step", type=int, default=None,
        help="Dormant LR tail (model.lr_schedule): bind a linear_tail schedule that starts "
        "decaying after this step; needs --lr-schedule-floor-milli. Omit both for the "
        "constant LR. s4_train must then pass the same --lr-schedule flags.",
    )
    parser.add_argument(
        "--lr-schedule-floor-milli", type=int, default=None,
        help="LR multiplier on the last step, in thousandths (100 = decay to 10 %%).",
    )
    parser.add_argument(
        "--level-router-policy",
        type=Path,
        default=None,
        help="RFC007 variant B (PROPOSED): bind this LevelRouterPolicy JSON by its hash; "
        "s4_train refuses it without the owner's ratification record.",
    )
    parser.add_argument(
        "--collapsed-region-rule",
        choices=list(_COLLAPSED_REGION_RULE_VALUES),
        default=_COLLAPSED_REGION_RULE_DEFAULT,
        help="RFC004-ERRATA-012 E1 (v17): 'exclude' drops nodes a view collapsed off the page "
        "from the region channel. Written only when not the default.",
    )
    parser.add_argument(
        "--story-decision-weight-milli",
        type=int,
        default=_STORY_DECISION_WEIGHT_MILLI_DEFAULT,
        help="RFC004-ERRATA-012 E3 (v17): weight of the story-decision positions in "
        "L_structure, in thousandths (>= 1; 1000 = v16, not written).",
    )
    parser.add_argument(
        "--alignment-type-weight-milli",
        action="append",
        default=[],
        metavar="TYPE=INT",
        help="RFC004-ERRATA-012 E2 (v17): L_alignment weight of an events_vocab node type, in "
        "thousandths (>= 1); repeatable.",
    )
    parser.add_argument(
        "--page-uniform-meta",
        type=Path,
        default=None,
        help="the page-uniform plan sidecar (s4_page_uniform_plan.json); its boost block "
        "(if any) is bound as page_uniform_boost.",
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
    selection_policy = (
        load_s4_selection_policy(args.selection_policy)
        if args.selection_policy
        else None
    )

    if (args.lr_schedule_start_step is None) != (args.lr_schedule_floor_milli is None):
        raise _refuse(
            "--lr-schedule-start-step and --lr-schedule-floor-milli must be supplied "
            "together or not at all"
        )
    lr_schedule = (
        None
        if args.lr_schedule_start_step is None
        else {
            "kind": "linear_tail",
            "start_step": args.lr_schedule_start_step,
            "floor_milli": args.lr_schedule_floor_milli,
        }
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
            "min_supervised_per_pointer": args.min_supervised_per_pointer,
        },
        heads_frozen_for_no_supervision=args.heads_frozen_for_no_supervision,
        low_supervision_pointers_disclosed=args.low_supervision_pointers_disclosed,
        run_class=args.run_class,
        gradient_bearing=args.gradient_bearing,
        run_seed=args.run_seed,
        lr_reduced_micro=args.lr_reduced_micro,
        init_from=init_from,
        parent_companion=parent_companion,
        selection_policy=selection_policy,
        out_dir=args.out_dir,
        mask_inactive_union_arms=args.mask_inactive_union_arms,
        consistency_lambda_milli=args.consistency_lambda_milli,
        level_router_policy_hash=(
            _level_router_policy_hash(args.level_router_policy)
            if args.level_router_policy is not None
            else None
        ),
        structural_table_region=args.structural_table_region,
        explicit_zero_bin_pointers=(
            sorted(_r1_pointers()) if args.explicit_zero_bin_r1 else ()
        ),
        visual_input=(
            json.loads(args.visual_input.read_text()) if args.visual_input is not None else None
        ),
        data_quality_gate_hash=data_quality_gate_hash_for(
            _split_manifests(args.cache_manifest), args.data_quality_gate
        ),
        lr_schedule=lr_schedule,
        collapsed_region_rule=args.collapsed_region_rule,
        story_decision_weight_milli=args.story_decision_weight_milli,
        alignment_type_weights_milli=_alignment_weights(args.alignment_type_weight_milli),
        page_uniform_meta=(
            json.loads(args.page_uniform_meta.read_text())
            if args.page_uniform_meta is not None
            else None
        ),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

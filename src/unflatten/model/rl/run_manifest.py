"""S4.5 run manifest (RFC005-ERRATA-010 E2/E7.1; dormant, builds no model).

The manifest binds, by hash, the artefacts a run executes under. None of the
bindings is trusted from the manifest itself: :func:`verify_rl_run_manifest`
re-derives every one from the real artefact it is handed (each artefact's own
hash is re-verified first, the state map is re-derived from S4, the parent
run class comes from the policy-parent identity object, the ratification hash
from the record). A manifest whose self-hash was recomputed over a tampered
binding therefore still fails.

Parent ancestry (RFC005 s12.1, ERRATA-005, ERRATA-010 E2) is
``reference.check_reference_compatible``: a ``promotion`` run on a
``discovery`` parent is refused, a ``discovery`` run may sit on either.

``composite_policy_hash`` is the hash a ratification record's ``policy_hash``
must equal: reward, rollout, training, dataset, stage-policy and state-map
hashes together (six keys). A record also requires every policy to be
``RATIFIED``.

Companion shape (1.1.0, D4). The manifest is itself the S1-format companion
``checkpoint_ancestry._parent_entry`` asks a parent to present: it exposes a
top-level ``run_class`` and an ``s1_run_manifest_hash`` that a checkpoint
sidecar's slot of the same name must equal. Unlike an S1 manifest (whose
self-hash is over the payload minus that one field) the RL manifest also holds
``run_manifest_hash``, so both are taken over the body minus *both* fields and
are equal by construction; the RL domain, not S1's ``rfc005.s1-run-manifest``,
is used because ``run_binding`` owns the latter.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from unflatten.model.hashing import domain_hash, make_id
from unflatten.model.rl.gate_policy import POLICY_HASH_FIELD, verify_rl_stage_gate_policy
from unflatten.model.rl.ids import RUN_ID_PREFIX, RUN_MANIFEST_DOMAIN, STAGE_ID
from unflatten.model.rl.policies import (
    MANIFEST_HASH_FIELD,
    REWARD_HASH_FIELD,
    ROLLOUT_HASH_FIELD,
    _schema_check,
    verify_dataset_manifest,
    verify_reward_policy,
    verify_rollout_policy,
)
from unflatten.model.rl.ratification import RlRatificationRecord
from unflatten.model.rl.reference import (
    RUN_CLASSES,
    ReferencePolicyIdentity,
    check_reference_compatible,
)
from unflatten.model.rl.state_map import s45_state_map_hash
from unflatten.model.rl.training_policy import (
    TRAINING_HASH_FIELD,
    check_execution_profile_matches,
    check_lr_groups,
    verify_rl_training_policy,
)

SCHEMA_VERSION: Final = "1.2.0"
# Equals ``loss.LOSS_SCOPE`` (pinned by a test); a literal avoids importing torch here.
_LOSS_SCOPE_CLASS: Final = "rollout_reward"
_LOSS_OWNER_SCOPE: Final = "example_local"
MANIFEST_HASH_FIELD_NAME: Final = "run_manifest_hash"
S1_SLOT_FIELD_NAME: Final = "s1_run_manifest_hash"
_SCHEMA: Final = "rl_run_manifest"
_IDENTITY_FIELDS: Final = (
    "ancestry_hash",
    "checkpoint_id",
    "checkpoint_sha256",
    "typed_head_config_hash",
    "run_class",
)
_SELF_HASH_FIELDS: Final = (S1_SLOT_FIELD_NAME, MANIFEST_HASH_FIELD_NAME)


class RlRunManifestInvalid(ValueError):
    """The run manifest is malformed, unbound, or disagrees with a real artefact."""


@dataclass(frozen=True, slots=True)
class RunArtefacts:
    """The real artefacts a manifest binds; ``verify`` re-derives from these."""

    reward_policy: Mapping[str, Any]
    rollout_policy: Mapping[str, Any]
    dataset_manifest: Mapping[str, Any]
    stage_policy: Mapping[str, Any]
    training_policy: Mapping[str, Any]
    reference: ReferencePolicyIdentity
    policy_parent: ReferencePolicyIdentity
    cache_identity: str
    visual_feature_tap: str
    ratification: RlRatificationRecord | None = None


def policy_status(arts: RunArtefacts) -> str:
    """``RATIFIED`` only when reward, rollout, training and stage policy all are."""
    statuses = {
        arts.reward_policy["status"],
        arts.rollout_policy["status"],
        arts.training_policy["status"],
        arts.stage_policy["status"],
    }
    return "RATIFIED" if statuses == {"RATIFIED"} else "PROPOSED"


def _identity_block(identity: ReferencePolicyIdentity) -> dict[str, str]:
    return {k: getattr(identity, k) for k in _IDENTITY_FIELDS}


def _record_hash(record: RlRatificationRecord) -> str:
    return domain_hash(
        RUN_MANIFEST_DOMAIN,
        {
            "kind": "ratification_record",
            "decision_id": record.decision_id,
            "stage": record.stage,
            "child": record.child,
            "policy_hash": record.policy_hash,
            "ratified": record.ratified,
            "ratified_by": record.ratified_by,
            "ratified_at": record.ratified_at,
            "parent_checkpoint_id": record.parent_checkpoint_id,
        },
    )


def composite_policy_hash(
    arts: RunArtefacts, run_class: str, selected_typed_heads: Iterable[str] = ()
) -> str:
    return domain_hash(
        RUN_MANIFEST_DOMAIN,
        {
            "kind": "composite_policy",
            "reward_policy_hash": arts.reward_policy[REWARD_HASH_FIELD],
            "rollout_policy_hash": arts.rollout_policy[ROLLOUT_HASH_FIELD],
            "rl_training_policy_hash": arts.training_policy[TRAINING_HASH_FIELD],
            "dataset_manifest_hash": arts.dataset_manifest[MANIFEST_HASH_FIELD],
            "stage_policy_hash": arts.stage_policy[POLICY_HASH_FIELD],
            "state_map_hash": s45_state_map_hash(run_class, selected_typed_heads),
        },
    )


def _check_artefacts(arts: RunArtefacts, run_class: str, selected: Iterable[str] = ()) -> None:
    verify_reward_policy(arts.reward_policy)
    verify_rollout_policy(arts.rollout_policy)
    verify_rl_training_policy(arts.training_policy)
    check_execution_profile_matches(arts.training_policy, arts.rollout_policy)
    check_lr_groups(arts.training_policy, run_class, selected)
    verify_dataset_manifest(arts.dataset_manifest)
    verify_rl_stage_gate_policy(arts.stage_policy)
    check_reference_compatible(arts.reference, arts.policy_parent, run_class=run_class)
    for name in ("cache_identity", "visual_feature_tap"):
        value = getattr(arts, name)
        if not isinstance(value, str) or not value:
            raise RlRunManifestInvalid(f"{name} must be a non-empty string, got {value!r}")


def _check_ratification(arts: RunArtefacts, composite: str) -> str | None:
    record = arts.ratification
    if record is None:
        return None
    if policy_status(arts) != "RATIFIED":
        raise RlRunManifestInvalid(
            "a ratification record is bound but a policy is not RATIFIED "
            f"(status {policy_status(arts)!r})"
        )
    if record.policy_hash != composite:
        raise RlRunManifestInvalid(
            f"ratification record policy_hash {record.policy_hash!r} != composite {composite!r}"
        )
    if (
        record.parent_checkpoint_id is not None
        and record.parent_checkpoint_id != arts.policy_parent.checkpoint_id
    ):
        raise RlRunManifestInvalid(
            f"ratification record binds parent {record.parent_checkpoint_id!r}, "
            f"policy parent is {arts.policy_parent.checkpoint_id!r}"
        )
    return _record_hash(record)


def _loss_scope(arts: RunArtefacts, selected: list[str]) -> dict[str, Any]:
    """Closed ``loss_scope`` block (RFC005-ERRATA-013 E2): what the loss may touch."""
    refs = list(arts.dataset_manifest["rights_privacy_refs"])
    return {
        "class": _LOSS_SCOPE_CLASS,
        "owner_scope": _LOSS_OWNER_SCOPE,
        "dataset_manifest_hash": arts.dataset_manifest[MANIFEST_HASH_FIELD],
        "rights_privacy_refs_hash": domain_hash(
            RUN_MANIFEST_DOMAIN, {"kind": "rights_privacy_refs", "refs": refs}
        ),
        "reward_policy_hash": arts.reward_policy[REWARD_HASH_FIELD],
        "rollout_policy_hash": arts.rollout_policy[ROLLOUT_HASH_FIELD],
        "reference": _identity_block(arts.reference),
        "output_locator": ["composite_head", *selected],
    }


def _derive_body(
    run_class: str, run_seed: int, arts: RunArtefacts, selected: list[str]
) -> dict[str, Any]:
    composite = composite_policy_hash(arts, run_class, selected)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "run_class": run_class,
        "run_seed": run_seed,
        "reward_policy_hash": arts.reward_policy[REWARD_HASH_FIELD],
        "rollout_policy_hash": arts.rollout_policy[ROLLOUT_HASH_FIELD],
        "rl_training_policy_hash": arts.training_policy[TRAINING_HASH_FIELD],
        "dataset_manifest_hash": arts.dataset_manifest[MANIFEST_HASH_FIELD],
        "stage_policy_hash": arts.stage_policy[POLICY_HASH_FIELD],
        "state_map_hash": s45_state_map_hash(run_class, selected),
        "selected_typed_heads": selected,
        "reference": _identity_block(arts.reference),
        "policy_parent": _identity_block(arts.policy_parent),
        "cache_identity": arts.cache_identity,
        "typed_head_config_hash": arts.policy_parent.typed_head_config_hash,
        "visual_feature_tap": arts.visual_feature_tap,
        "composite_policy_hash": composite,
        "ratification_record_hash": _check_ratification(arts, composite),
        "loss_scope": _loss_scope(arts, selected),
    }


def build_rl_run_manifest(
    *,
    run_class: str,
    run_seed: int,
    artefacts: RunArtefacts,
    selected_typed_heads: Iterable[str] = (),
) -> dict[str, Any]:
    if run_class not in RUN_CLASSES:
        raise RlRunManifestInvalid(f"run_class {run_class!r} not in {RUN_CLASSES}")
    if type(run_seed) is not int or run_seed < 0:
        raise RlRunManifestInvalid(f"run_seed must be an int >= 0, got {run_seed!r}")
    selected = sorted(set(selected_typed_heads))
    _check_artefacts(artefacts, run_class, selected)
    body = _derive_body(run_class, run_seed, artefacts, selected)
    sealed = {**body, "run_id": make_id(RUN_ID_PREFIX, RUN_MANIFEST_DOMAIN, body)}
    self_hash = domain_hash(RUN_MANIFEST_DOMAIN, sealed)
    doc = {**sealed, S1_SLOT_FIELD_NAME: self_hash, MANIFEST_HASH_FIELD_NAME: self_hash}
    _schema_check(_SCHEMA, doc)
    return doc


def verify_rl_run_manifest(manifest: Mapping[str, Any], artefacts: RunArtefacts) -> None:
    _schema_check(_SCHEMA, manifest)
    body = {k: v for k, v in manifest.items() if k not in _SELF_HASH_FIELDS}
    if domain_hash(RUN_MANIFEST_DOMAIN, body) != manifest[MANIFEST_HASH_FIELD_NAME]:
        raise RlRunManifestInvalid("run_manifest_hash does not verify against the payload")
    if manifest[S1_SLOT_FIELD_NAME] != manifest[MANIFEST_HASH_FIELD_NAME]:
        raise RlRunManifestInvalid(
            f"s1_run_manifest_hash {manifest[S1_SLOT_FIELD_NAME]!r} != run_manifest_hash "
            f"{manifest[MANIFEST_HASH_FIELD_NAME]!r}: the companion slot would not name "
            "this manifest"
        )
    run_class = manifest["run_class"]
    _check_artefacts(artefacts, run_class, manifest["selected_typed_heads"])
    derived = _derive_body(
        run_class, manifest["run_seed"], artefacts, list(manifest["selected_typed_heads"])
    )
    for key, value in derived.items():
        if manifest[key] != value:
            raise RlRunManifestInvalid(
                f"{key} {manifest[key]!r} does not match the one re-derived from the "
                f"real artefact {value!r}"
            )
    expected_id = make_id(RUN_ID_PREFIX, RUN_MANIFEST_DOMAIN, derived)
    if manifest["run_id"] != expected_id:
        raise RlRunManifestInvalid(f"run_id {manifest['run_id']!r} != derived {expected_id!r}")

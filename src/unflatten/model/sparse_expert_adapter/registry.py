"""Sparse trainable-group registry, capability-profile draft, lock kinds.

W3-3 / DEC-047 condition 3: a **new, separately versioned** artefact for the
sparse Word Expert Adapter's `expert_adapter.router` / `expert_adapter.shared`
/ `expert_adapter.specialists.<id>` groups. This module never imports
`trainable_group_registry.py` or `capability_profile.py`, and nothing in the
production path imports this module -- so `TRAINABLE_GROUP_REGISTRY_HASH` and
every existing `*_CAPABILITY_PROFILE.profile_hash` stay byte-identical
(pinned as literals in
`tests/model/sparse_expert_adapter/test_registry.py
::test_existing_registry_and_profile_hashes_unchanged`, captured before this
module existed).

Unlike the closed dense-V0 `trainable_group_registry`, the sparse registry is
**built from a manifest**, not a fixed module-level constant: DEC-047
condition 4 requires expert count/topology to stay a parameter, and a sparse
registry with a hardcoded specialist count would smuggle a topology choice
back in as a constant.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from unflatten.model.hashing import domain_hash
from unflatten.model.sparse_expert_adapter.adapter import MissingExpertWeightError
from unflatten.model.sparse_expert_adapter.manifest import SparseExpertAdapterManifest

_TRAINABLE_GROUP_DOMAIN = "rfc004.sparse-expert-adapter-trainable-group-registry.v0"
_CAPABILITY_PROFILE_DOMAIN = "rfc004.sparse-expert-adapter-capability-profile-draft.v0"

#: The four Component Lock kinds this draft artefact reserves for sparse
#: components. Deliberately NOT added to `schemas/component_lock.schema.json`
#: -- adding them there is a separate, ratified change; until then no real
#: lock document can validate against these kinds, so keeping them here only
#: cannot move `component_lock_hash` for any existing lock.
SPARSE_EXPERT_COMPONENT_KINDS: frozenset[str] = frozenset(
    {
        "sparse_expert_router",
        "sparse_expert_shared",
        "sparse_expert_specialist",
    }
)

ROUTER_GROUP_ID = "expert_adapter.router"
SHARED_GROUP_ID = "expert_adapter.shared"


def specialist_group_id(expert_id: str) -> str:
    return f"expert_adapter.specialists.{expert_id}"


@dataclass(frozen=True)
class SparseTrainableGroup:
    """One row of the draft sparse trainable-group registry."""

    group_id: str
    kind: str  # "trainable_sparse_draft"
    lock_kind: str  # one of SPARSE_EXPERT_COMPONENT_KINDS
    owned_tensor_identities: tuple[str, ...]


def build_sparse_trainable_group_registry(
    manifest: SparseExpertAdapterManifest,
) -> dict[str, SparseTrainableGroup]:
    """Derive the sparse registry rows for one concrete manifest's topology.

    Row count and specialist ids follow `manifest.specialist_roles`
    (DEC-047 condition 4: the topology is the manifest's, not this
    function's).
    """
    registry: dict[str, SparseTrainableGroup] = {
        ROUTER_GROUP_ID: SparseTrainableGroup(
            group_id=ROUTER_GROUP_ID,
            kind="trainable_sparse_draft",
            lock_kind="sparse_expert_router",
            owned_tensor_identities=(f"{ROUTER_GROUP_ID}/base_weights",),
        ),
        SHARED_GROUP_ID: SparseTrainableGroup(
            group_id=SHARED_GROUP_ID,
            kind="trainable_sparse_draft",
            lock_kind="sparse_expert_shared",
            owned_tensor_identities=(f"{SHARED_GROUP_ID}/base_weights",),
        ),
    }
    for role in manifest.specialist_roles:
        group_id = specialist_group_id(role.expert_id)
        registry[group_id] = SparseTrainableGroup(
            group_id=group_id,
            kind="trainable_sparse_draft",
            lock_kind="sparse_expert_specialist",
            owned_tensor_identities=(f"{group_id}/base_weights",),
        )
    return registry


def sparse_trainable_group_registry_projection(
    registry: Mapping[str, SparseTrainableGroup],
) -> dict[str, Any]:
    return {
        "sparse_trainable_group_registry_schema_version": "0.1.0",
        "groups": [
            {
                "group_id": g.group_id,
                "kind": g.kind,
                "lock_kind": g.lock_kind,
                "owned_tensor_identities": list(g.owned_tensor_identities),
            }
            for g in sorted(registry.values(), key=lambda g: g.group_id)
        ],
    }


def sparse_trainable_group_registry_hash(registry: Mapping[str, SparseTrainableGroup]) -> str:
    return domain_hash(
        _TRAINABLE_GROUP_DOMAIN, sparse_trainable_group_registry_projection(registry)
    )


@dataclass(frozen=True)
class SparseCapabilityProfileDraft:
    """A PROPOSED capability-profile payload for the sparse expert adapter.

    Never inserted into `capability_profile.KNOWN_CAPABILITY_PROFILES`: this
    is a draft artefact for the eventual S6/G2 ratification, not a resolvable
    profile any consumer can look up today.
    """

    profile_id: str
    status: str
    expert_adapter_mode: str
    router_group_id: str
    shared_group_id: str
    specialist_group_ids: tuple[str, ...]
    profile_hash: str


def _sparse_capability_profile_payload(
    profile_id: str, specialist_group_ids: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "sparse_capability_profile_schema_version": "0.1.0",
        "profile_id": profile_id,
        "status": "PROPOSED",
        "expert_adapter_mode": "sparse",
        "router_group_id": ROUTER_GROUP_ID,
        "shared_group_id": SHARED_GROUP_ID,
        "specialist_group_ids": sorted(specialist_group_ids),
    }


def build_sparse_capability_profile_draft(
    profile_id: str, manifest: SparseExpertAdapterManifest
) -> SparseCapabilityProfileDraft:
    """Build a draft profile for `manifest`'s topology. Status is always PROPOSED."""
    specialist_group_ids = tuple(
        specialist_group_id(role.expert_id) for role in manifest.specialist_roles
    )
    payload = _sparse_capability_profile_payload(profile_id, specialist_group_ids)
    return SparseCapabilityProfileDraft(
        profile_id=profile_id,
        status="PROPOSED",
        expert_adapter_mode="sparse",
        router_group_id=ROUTER_GROUP_ID,
        shared_group_id=SHARED_GROUP_ID,
        specialist_group_ids=specialist_group_ids,
        profile_hash=domain_hash(_CAPABILITY_PROFILE_DOMAIN, payload),
    )


def verify_sparse_component_weights(
    declared_group_ids: Iterable[str], available_tensor_identities: Iterable[str]
) -> None:
    """§23 hard load error: a declared sparse group with no available weight.

    `declared_group_ids` names the groups an active sparse Component Lock
    declares (router/shared/specialist ids); `available_tensor_identities`
    is whatever the caller's weight store can produce bytes for. Does no I/O
    itself, mirroring `reasoning.lock.verify_reasoning_component_weights`.
    """
    available = frozenset(available_tensor_identities)
    for group_id in declared_group_ids:
        identity = f"{group_id}/base_weights"
        if identity not in available:
            raise MissingExpertWeightError(
                f"sparse component {group_id!r} is declared active but no "
                f"weight artifact is available for {identity!r} -- refusing "
                "to load rather than leave it at its random init"
            )

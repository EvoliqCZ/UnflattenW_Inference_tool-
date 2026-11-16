"""RFC005 §11/§44 Candidate Track (plan `spec-candidate-track-c4`).

A `CandidateTrack` is a closed, cross-checked manifest that binds one
admission track name (`admission_track_name`) to one clean_pretrained base
checkpoint ancestry (`unflatten.model.checkpoint_ancestry`), one Model
Component Lock and one Architecture Manifest. It is owned by the Stage Gate
Report (C5), not `s1_run_manifest` (R1).

Everything a caller might be tempted to declare about rights, privacy or
provenance is instead COPIED, verbatim, from the underlying checkpoint
ancestry document -- and only after that document passes
`checkpoint_ancestry.verify_base`. This module never imports or calls
`derive_rights_status` / `derive_privacy_status` / `derive_distribution_allowed`
(`tests/model/test_candidate_track.py::TestNoModelSurface` source-scans for
this): re-deriving those statuses here would let a candidate track disagree
with the ancestry it claims to summarize without either side noticing.

`adapter_policy` is produced here, not copied: RFC005 §14.3's adaptation
policy is the closed trainable-group registry
(`unflatten.model.trainable_group_registry.TRAINABLE_GROUP_REGISTRY_HASH`)
with all `lora.*` deltas disabled (RFC005-ERRATA-001 E1,
`adaptation_kind = "adapter_plus_new_heads_no_lora"`).

Torch-free by construction, mirroring `checkpoint_ancestry.py`: nothing here
imports torch.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.model import checkpoint_ancestry
from unflatten.model.capability_profile import CapabilityProfile
from unflatten.model.checkpoint_ancestry import normalise_sha256
from unflatten.model.hashing import make_id
from unflatten.model.manifests import (
    component_lock_hash,
    validate_exact_text_s1_architecture_manifest,
)
from unflatten.model.trainable_group_registry import TRAINABLE_GROUP_REGISTRY_HASH

_SCHEMA_DIR = Path(__file__).parent / "schemas"

#: Owned here: no other module computes a `candidate_track_id`.
_CANDIDATE_TRACK_DOMAIN = "rfc005.candidate-track/v1"

CANDIDATE_TRACK_ID_PREFIX = "track"

#: The `candidate_track` schema revision this module produces.
CANDIDATE_TRACK_SCHEMA_VERSION = "1.0.0"

#: RFC004 ratified only for `dense_contract_canary_ready` /
#: `dense_vertical_slice_ready` under DEC-005 -- the one admission track
#: this module may bind a candidate to today. Pinned equal (by test) to
#: `unflatten.dataset.admission.report.DENSE_CONTRACT_CANARY_TRACK` and the
#: identically-named literal in `unflatten.model.trainability_smoke`.
ADMISSION_TRACK_NAME = "dense_contract_canary"

#: The one backbone profile a candidate track may declare in v1.
BACKBONE_PROFILE = "qwen3-vl-word-adapter-v0"

#: The one AST profile a candidate track may declare in v1 (RFC001-ERRATA
#: series word-core-v0 successor for visual grounding).
AST_PROFILE = "word-visual-v1.1"

#: DEC-007 D3 pin, repeated here as a literal (not imported from
#: `candidate_loader`, which pulls torch): `tests/model/test_candidate_track.py`
#: asserts by subprocess that this literal equals
#: `candidate_loader.QWEN_CANDIDATE_MODEL_ID`.
QWEN_BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

#: RFC005-ERRATA-001 E1: the one adaptation kind v1 produces.
ADAPTATION_KIND = "adapter_plus_new_heads_no_lora"

_RFC_REF = "RFC005 §14.3"


class CandidateTrackInvalid(Exception):
    """Raised on any refusal in this module: a malformed document, a
    mutated hash, or a cross-check disagreement between the ancestry,
    component lock, capability profile and architecture manifest supplied
    to :func:`build_candidate_track`. Never raised for a merely
    surprising-but-consistent document."""


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / "candidate_track.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _validate_schema(doc: dict[str, Any], *, what: str) -> None:
    errors = sorted(_validator().iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        pointer = "/" + "/".join(str(p) for p in first.absolute_path)
        raise CandidateTrackInvalid(f"{what} invalid at {pointer or '/'}: {first.message}")


def candidate_track_projection(doc: dict[str, Any]) -> dict[str, Any]:
    """The payload `candidate_track_id` is computed over: `doc` with its own
    `candidate_track_id` removed. Refuses when `doc` has no
    `candidate_track_id` key at all -- every caller must explicitly carry
    the self-id slot, even as `None`, before hashing."""
    if "candidate_track_id" not in doc:
        raise CandidateTrackInvalid(
            "doc has no 'candidate_track_id' key -- the self-id slot must be "
            "present (even as None) before it can be projected away"
        )
    return {k: v for k, v in doc.items() if k != "candidate_track_id"}


def compute_candidate_track_id(doc: dict[str, Any]) -> str:
    """`candidate_track_id` for `doc` -- `make_id` over
    :func:`candidate_track_projection`, so the id can never be computed over
    a payload that still carries a (possibly stale) id of its own."""
    return make_id(
        CANDIDATE_TRACK_ID_PREFIX, _CANDIDATE_TRACK_DOMAIN, candidate_track_projection(doc)
    )


def _architecture_manifest_sha256(architecture_manifest: dict[str, Any]) -> str:
    """The raw SHA-256 of `architecture_manifest`'s canonical JCS bytes --
    the same canonical serialization `manifests.component_lock_hash` uses,
    so this equals a lock's `architecture_manifest` component
    `artifact_sha256` iff the lock was bound against these exact bytes."""
    manifest_bytes = to_canonical_json(architecture_manifest)
    return f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}"


def _verify_base(ancestry: dict[str, Any]) -> None:
    """`checkpoint_ancestry.verify_base`, re-raised as `CandidateTrackInvalid`
    so every refusal from this module's public functions shares one
    exception type."""
    try:
        checkpoint_ancestry.verify_base(ancestry)
    except checkpoint_ancestry.CheckpointAncestryInvalid as exc:
        raise CandidateTrackInvalid(f"ancestry failed verify_base: {exc}") from exc


def _lock_component(component_lock: dict[str, Any], kind: str) -> dict[str, Any]:
    for entry in component_lock.get("components", []):
        if entry.get("kind") == kind:
            return entry
    raise CandidateTrackInvalid(f"component lock has no {kind!r} component")


def _map_transitive_exposure_manifest(ancestry_field: dict[str, Any]) -> dict[str, Any]:
    """Map `checkpoint_ancestry`'s `transitive_exposure_manifest` exposure_state
    into this document's closed union: `not_produced` copies verbatim, a
    `root` branch becomes `{kind: hash, sha256: root_hash}` (design decision
    2 -- a candidate track exposes only the hash, not the root-object
    shape)."""
    kind = ancestry_field.get("kind")
    if kind == "not_produced":
        return {"kind": "not_produced", "reason": ancestry_field["reason"]}
    if kind == "root":
        return {"kind": "hash", "sha256": ancestry_field["root_hash"]}
    raise CandidateTrackInvalid(
        f"ancestry transitive_exposure_manifest has unknown kind {kind!r}"
    )


def _cross_check_build_inputs(
    *,
    component_lock: dict[str, Any],
    ancestry: dict[str, Any],
    capability_profile: CapabilityProfile,
    architecture_manifest: dict[str, Any],
) -> str:
    """The nine cross-checks of :func:`build_candidate_track`, in order,
    refusing (`CandidateTrackInvalid`) on any of the following
    disagreements:

    - `ancestry` is not a schema-valid, self-consistent `document_kind:
      "base"` checkpoint ancestry document (`checkpoint_ancestry.verify_base`).
    - `ancestry["component_lock_hash"]` does not match
      `manifests.component_lock_hash(component_lock)`.
    - `ancestry["base_model"]["weights_set_sha256"]` does not match the
      lock's `base_model_weights` component `artifact_sha256`.
    - `capability_profile.profile_hash` does not match the lock's
      `model_capability_profile` component `semantic_hash`.
    - `architecture_manifest`'s canonical-JCS SHA-256 does not match the
      lock's `architecture_manifest` component `artifact_sha256`.
    - `component_lock["architecture_id"]` does not match
      `architecture_manifest["architecture_id"]`.
    - `component_lock["profiles"]` does not match
      `architecture_manifest["profiles"]` (the AST profile list, never the
      capability profile id).
    - `architecture_manifest` fails
      `manifests.validate_exact_text_s1_architecture_manifest`.
    - `ancestry["base_model"]["model_id"]` is not :data:`QWEN_BASE_MODEL_ID`.

    Returns `architecture_manifest`'s canonical-JCS SHA-256, which the
    caller stores as `architecture_manifest_sha256` instead of recomputing.
    """
    _verify_base(ancestry)

    lock_hash = component_lock_hash(component_lock)
    if ancestry["component_lock_hash"] != lock_hash:
        raise CandidateTrackInvalid(
            f"ancestry component_lock_hash {ancestry['component_lock_hash']!r} != "
            f"component_lock_hash(component_lock) {lock_hash!r}"
        )

    weights_entry = _lock_component(component_lock, "base_model_weights")
    lock_weights_sha256 = normalise_sha256(weights_entry["artifact_sha256"])
    ancestry_weights_sha256 = normalise_sha256(ancestry["base_model"]["weights_set_sha256"])
    if lock_weights_sha256 != ancestry_weights_sha256:
        raise CandidateTrackInvalid(
            f"lock base_model_weights artifact_sha256 {lock_weights_sha256!r} != "
            f"ancestry base_model.weights_set_sha256 {ancestry_weights_sha256!r}"
        )

    capability_entry = _lock_component(component_lock, "model_capability_profile")
    lock_capability_hash = normalise_sha256(capability_entry["semantic_hash"])
    if lock_capability_hash != normalise_sha256(capability_profile.profile_hash):
        raise CandidateTrackInvalid(
            f"lock model_capability_profile semantic_hash {lock_capability_hash!r} != "
            f"capability_profile.profile_hash {capability_profile.profile_hash!r}"
        )

    manifest_entry = _lock_component(component_lock, "architecture_manifest")
    lock_manifest_sha256 = normalise_sha256(manifest_entry["artifact_sha256"])
    computed_manifest_sha256 = _architecture_manifest_sha256(architecture_manifest)
    if lock_manifest_sha256 != computed_manifest_sha256:
        raise CandidateTrackInvalid(
            f"lock architecture_manifest artifact_sha256 {lock_manifest_sha256!r} != "
            f"canonical-JCS SHA-256 of the supplied architecture_manifest "
            f"{computed_manifest_sha256!r}"
        )

    if component_lock["architecture_id"] != architecture_manifest["architecture_id"]:
        raise CandidateTrackInvalid(
            f"component lock architecture_id {component_lock['architecture_id']!r} != "
            f"architecture manifest architecture_id "
            f"{architecture_manifest['architecture_id']!r}"
        )

    if component_lock["profiles"] != architecture_manifest["profiles"]:
        raise CandidateTrackInvalid(
            f"component lock profiles {component_lock['profiles']!r} != "
            f"architecture manifest profiles {architecture_manifest['profiles']!r}"
        )

    issues = validate_exact_text_s1_architecture_manifest(architecture_manifest)
    if issues:
        raise CandidateTrackInvalid(
            "architecture manifest failed the exact-text S1 profile: "
            + "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        )

    ancestry_model_id = ancestry["base_model"]["model_id"]
    if ancestry_model_id != QWEN_BASE_MODEL_ID:
        raise CandidateTrackInvalid(
            f"ancestry base_model.model_id {ancestry_model_id!r} != "
            f"{QWEN_BASE_MODEL_ID!r}"
        )
    return computed_manifest_sha256


def build_candidate_track(
    *,
    component_lock: dict[str, Any],
    ancestry: dict[str, Any],
    capability_profile: CapabilityProfile,
    architecture_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build one Candidate Track document. All input disagreements are
    refused by :func:`_cross_check_build_inputs` (`CandidateTrackInvalid`);
    this function assembles the document, freezes its self-id and
    schema-validates the result."""
    computed_manifest_sha256 = _cross_check_build_inputs(
        component_lock=component_lock,
        ancestry=ancestry,
        capability_profile=capability_profile,
        architecture_manifest=architecture_manifest,
    )

    doc: dict[str, Any] = {
        "candidate_track_schema_version": CANDIDATE_TRACK_SCHEMA_VERSION,
        "candidate_track_id": None,
        "admission_track_name": ADMISSION_TRACK_NAME,
        "backbone_profile": BACKBONE_PROFILE,
        "ast_profile": AST_PROFILE,
        "capability_profile_id": capability_profile.profile_id,
        "capability_profile_hash": capability_profile.profile_hash,
        "architecture_manifest_sha256": computed_manifest_sha256,
        "architecture_id": architecture_manifest["architecture_id"],
        "base_checkpoint": {
            "kind": "base_weights_only",
            "weights_set_sha256": ancestry["base_model"]["weights_set_sha256"],
        },
        "initialization_kind": ancestry["initialization_kind"],
        "adaptation_kind": ADAPTATION_KIND,
        "legacy_ancestry": ancestry["legacy_ancestry"]["value"],
        "ancestry_hash": ancestry["ancestry_hash"],
        "checkpoint_rights_status": ancestry["checkpoint_rights_status"],
        "checkpoint_privacy_status": ancestry["checkpoint_privacy_status"],
        "transitive_exposure_manifest": _map_transitive_exposure_manifest(
            ancestry["transitive_exposure_manifest"]
        ),
        "adapter_policy": {
            "kind": "registry",
            "trainable_group_registry_hash": TRAINABLE_GROUP_REGISTRY_HASH,
            "lora": "disabled",
            "rfc_ref": _RFC_REF,
        },
        "model_component_lock_hash": ancestry["component_lock_hash"],
    }
    doc["candidate_track_id"] = compute_candidate_track_id(doc)

    _validate_schema(doc, what="built candidate track document")
    return doc


def _verify_copied_fields_against_ancestry(
    doc: dict[str, Any], ancestry: dict[str, Any]
) -> None:
    """Re-compare every field :func:`build_candidate_track` copies (or maps)
    verbatim from `ancestry` against the (already `verify_base`-checked)
    ancestry itself, so a candidate track cannot silently drift from the
    ancestry it claims to summarize."""
    expected_transitive_exposure_manifest = _map_transitive_exposure_manifest(
        ancestry["transitive_exposure_manifest"]
    )
    checks: tuple[tuple[str, Any, Any], ...] = (
        ("checkpoint_rights_status", doc["checkpoint_rights_status"],
         ancestry["checkpoint_rights_status"]),
        ("checkpoint_privacy_status", doc["checkpoint_privacy_status"],
         ancestry["checkpoint_privacy_status"]),
        ("initialization_kind", doc["initialization_kind"], ancestry["initialization_kind"]),
        ("legacy_ancestry", doc["legacy_ancestry"], ancestry["legacy_ancestry"]["value"]),
        ("ancestry_hash", doc["ancestry_hash"], ancestry["ancestry_hash"]),
        (
            "transitive_exposure_manifest",
            doc["transitive_exposure_manifest"],
            expected_transitive_exposure_manifest,
        ),
        (
            "base_checkpoint.weights_set_sha256",
            normalise_sha256(doc["base_checkpoint"]["weights_set_sha256"]),
            normalise_sha256(ancestry["base_model"]["weights_set_sha256"]),
        ),
    )
    for field, doc_value, ancestry_value in checks:
        if doc_value != ancestry_value:
            raise CandidateTrackInvalid(
                f"candidate track {field} {doc_value!r} != ancestry-derived "
                f"{ancestry_value!r} -- the candidate track no longer agrees "
                f"with the ancestry it claims to summarize"
            )


def verify_candidate_track(
    doc: dict[str, Any],
    *,
    component_lock: dict[str, Any] | None = None,
    ancestry: dict[str, Any] | None = None,
) -> None:
    """Verify a previously-built candidate track document: schema, self-id,
    and -- for each of `component_lock` / `ancestry` that is supplied -- that
    the document still agrees with it (design §"Čo číta C5"). When `ancestry`
    is supplied, every field :func:`build_candidate_track` copies or maps
    from it is re-compared, not just `ancestry_hash`."""
    _validate_schema(doc, what="candidate track document")

    expected_id = compute_candidate_track_id(doc)
    if doc["candidate_track_id"] != expected_id:
        raise CandidateTrackInvalid(
            "candidate_track_id does not match its own payload -- the "
            "document was mutated after freezing"
        )

    if component_lock is not None:
        lock_hash = component_lock_hash(component_lock)
        if doc["model_component_lock_hash"] != lock_hash:
            raise CandidateTrackInvalid(
                f"candidate track model_component_lock_hash "
                f"{doc['model_component_lock_hash']!r} != "
                f"component_lock_hash(component_lock) {lock_hash!r}"
            )

    if ancestry is not None:
        _verify_base(ancestry)

        if doc["model_component_lock_hash"] != ancestry["component_lock_hash"]:
            raise CandidateTrackInvalid(
                "candidate track model_component_lock_hash != "
                "ancestry component_lock_hash -- the ancestry supplied does not "
                "belong to this candidate track"
            )
        _verify_copied_fields_against_ancestry(doc, ancestry)

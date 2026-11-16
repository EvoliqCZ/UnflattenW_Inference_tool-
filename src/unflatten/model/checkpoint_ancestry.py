"""RFC005 checkpoint ancestry manifest (design §35.1/§11, plan
`spec-checkpoint-ancestry-c6`).

Rights/privacy status are DERIVED from typed decision references
(`base_model_decision`, each `dataset_exposures[i].decision`), never
declared: :func:`derive_rights_status` / :func:`derive_privacy_status`
compute them the same way both the producer (:func:`build_base_ancestry`)
and the verifiers (:func:`verify_base`, :func:`verify_sidecar`) check them,
so a caller cannot hand-write an `approved` status past the decisions that
would actually justify it. `distribution_allowed` (DEC-011) is derived the
same way: :func:`derive_distribution_allowed` is true only when a non-null
`distribution_decision` is present AND both rights and privacy already
derive `approved`.

Generation 0 (`document_kind: "base"`) is written once, before any
checkpoint exists, by `freeze_checkpoint_ancestry.py`. Generation 1
(`document_kind: "checkpoint"`) extends it once per checkpoint
(:func:`extend_for_checkpoint`) and is only completed
(:func:`finalize_checkpoint_document`) after the checkpoint's bytes exist,
because `ancestry_hash`'s projection (:func:`ancestry_projection`, design
decision 4) deliberately excludes `checkpoint.checkpoint_sha256` and
`checkpoint.checkpoint_id` -- so the very same `ancestry_hash` can be
embedded inside a checkpoint file *before* that file's own bytes (and
therefore its sha256) exist, and still match the final, filled-in sidecar.

Torch-free by construction: nothing here imports torch, so importing this
module never pulls it into `sys.modules` (mirrors
`tests/model/test_run_binding.py`'s no-model-surface guard). The actual
`torch.save(...)` call, and the write-order in design decision 5 (doc
without sha -> ancestry_hash -> torch.save -> replace -> file sha256 ->
sidecar `.tmp` -> replace), belong to `s1_train.py` (plan tasks 3-4), not
here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.model.hashing import domain_hash, make_id
from unflatten.model.manifests import component_lock_hash

_SCHEMA_DIR = Path(__file__).parent / "schemas"

#: `ancestry_hash = domain_hash(_ANCESTRY_DOMAIN, ancestry_projection(doc))`.
#: Owned here: no other module computes an `ancestry_hash`.
_ANCESTRY_DOMAIN = "rfc005.checkpoint-ancestry/v1"

#: `checkpoint_id = make_id(CHECKPOINT_ID_PREFIX, _CHECKPOINT_ID_DOMAIN,
#: {ancestry_hash, checkpoint_sha256})`.
_CHECKPOINT_ID_DOMAIN = "rfc005.checkpoint-id/v1"

CHECKPOINT_ID_PREFIX = "ckpt"

#: The `checkpoint_ancestry` schema revision this module produces.
CHECKPOINT_ANCESTRY_SCHEMA_VERSION = "1.1.0"

#: Sidecar file naming: `<checkpoint_path><SIDECAR_SUFFIX>`.
SIDECAR_SUFFIX = ".ancestry.json"

_CHUNK_BYTES = 1024 * 1024


class CheckpointAncestryInvalid(Exception):
    """Raised on any refusal in this module: a malformed document, a
    mutated hash, a disagreeing derived status, or a mismatched binding.
    Never raised for a merely surprising-but-consistent document."""


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / "checkpoint_ancestry.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _validate_schema(doc: dict[str, Any], *, what: str) -> None:
    errors = sorted(_validator().iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        pointer = "/" + "/".join(str(p) for p in first.absolute_path)
        raise CheckpointAncestryInvalid(f"{what} invalid at {pointer or '/'}: {first.message}")


def sidecar_path(checkpoint_path: Path) -> Path:
    """`<checkpoint_path>.ancestry.json` -- the sidecar this checkpoint carries."""
    return Path(str(checkpoint_path) + SIDECAR_SUFFIX)


def ancestry_projection(doc: dict[str, Any]) -> dict[str, Any]:
    """The payload `ancestry_hash` is computed over (design decision 4):
    `doc` with its own `ancestry_hash` removed, and -- when `checkpoint` is
    an object -- with `checkpoint.checkpoint_sha256` and
    `checkpoint.checkpoint_id` also removed, so a `.pt` can carry the
    sidecar's hash before its own bytes (and therefore its sha256) exist.

    Refuses (rather than silently projecting a differently-shaped document)
    when `doc` has no `ancestry_hash` key at all: every caller must
    explicitly carry the self-hash slot, even as `None`, before hashing.
    """
    if "ancestry_hash" not in doc:
        raise CheckpointAncestryInvalid(
            "doc has no 'ancestry_hash' key -- the self-hash slot must be "
            "present (even as None) before it can be projected away"
        )
    projection = {k: v for k, v in doc.items() if k != "ancestry_hash"}
    checkpoint = projection.get("checkpoint")
    if isinstance(checkpoint, dict):
        projection["checkpoint"] = {
            k: v for k, v in checkpoint.items() if k not in ("checkpoint_sha256", "checkpoint_id")
        }
    return projection


def compute_ancestry_hash(doc: dict[str, Any]) -> str:
    """`ancestry_hash` for `doc` (design decision 3)."""
    return domain_hash(_ANCESTRY_DOMAIN, ancestry_projection(doc))


def compute_checkpoint_id(ancestry_hash: str, checkpoint_sha256: str) -> str:
    """`checkpoint_id` binding one checkpoint's bytes to the ancestry that
    authorized it (design decision 3)."""
    return make_id(
        CHECKPOINT_ID_PREFIX,
        _CHECKPOINT_ID_DOMAIN,
        {"ancestry_hash": ancestry_hash, "checkpoint_sha256": checkpoint_sha256},
    )


def file_sha256(path: Path) -> str:
    """`sha256:<64 lowercase hex>` of `path`'s bytes, read in fixed-size
    chunks so this never loads a multi-gigabyte checkpoint whole."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def normalise_sha256(value: str) -> str:
    """Normalise a sha256 given as `sha256:<hex>` or bare `<hex>` (any
    case) to the canonical `sha256:<64 lowercase hex>` form. Refuses
    anything that is not exactly 64 hex characters once the optional
    prefix is stripped."""
    raw = value[len("sha256:") :] if value.startswith("sha256:") else value
    raw = raw.lower()
    if len(raw) != 64 or any(c not in "0123456789abcdef" for c in raw):
        raise CheckpointAncestryInvalid(f"not a sha256 hex digest: {value!r}")
    return f"sha256:{raw}"


def derive_rights_status(doc: dict[str, Any]) -> str:
    """`approved` iff `base_model_decision` is non-null AND at least one
    `dataset_exposures` entry exists AND every `dataset_exposures[i].decision`
    is non-null; else `unresolved` (design decision 2). An empty
    `dataset_exposures` is never treated as vacuously satisfying the "every
    decision is non-null" clause. Never returns `prohibited` -- v1 has no
    refusal path that produces it; the schema keeps the value in its closed
    enum for a future one."""
    if doc.get("base_model_decision") is None:
        return "unresolved"
    exposures = doc.get("dataset_exposures", [])
    if not exposures:
        return "unresolved"
    if any(exposure.get("decision") is None for exposure in exposures):
        return "unresolved"
    return "approved"


def derive_privacy_status(doc: dict[str, Any]) -> str:
    """`approved` iff at least one `dataset_exposures` entry exists AND
    every `dataset_exposures[i].decision` is non-null; else `unresolved`
    (design decision 2). An empty `dataset_exposures` is never treated as
    vacuously satisfying the "every decision is non-null" clause."""
    exposures = doc.get("dataset_exposures", [])
    if not exposures:
        return "unresolved"
    if any(exposure.get("decision") is None for exposure in exposures):
        return "unresolved"
    return "approved"


def derive_distribution_allowed(doc: dict[str, Any]) -> bool:
    """DEC-011: true iff `distribution_decision` is non-null AND both
    :func:`derive_rights_status` and :func:`derive_privacy_status` derive
    `approved`. Never a declared boolean -- callers pass
    `distribution_decision`, never `distribution_allowed` itself."""
    if doc.get("distribution_decision") is None:
        return False
    return derive_rights_status(doc) == "approved" and derive_privacy_status(doc) == "approved"


def _lock_component_artifact_sha256(component_lock: dict[str, Any], kind: str) -> str:
    for entry in component_lock.get("components", []):
        if entry.get("kind") == kind:
            sha = entry.get("artifact_sha256")
            if sha is None:
                raise CheckpointAncestryInvalid(
                    f"component lock entry {kind!r} has no artifact_sha256"
                )
            return normalise_sha256(sha)
    raise CheckpointAncestryInvalid(f"component lock has no {kind!r} component")


def _parent_entry(parent: dict[str, Any], *, doc: dict[str, Any]) -> dict[str, Any]:
    """One `parents[0]` entry (schema 1.1.0) from `parent = {"sidecar":
    ..., "companion": ...}` (design §6). `doc` is the base document being
    built so far (already carrying its own `dataset_exposures`), used only
    for the DEC-006 transitive-exposure subset refusal."""
    sidecar = parent["sidecar"]
    companion = parent["companion"]
    checkpoint = sidecar["checkpoint"]

    if companion.get("s1_run_manifest_hash") != checkpoint.get("s1_run_manifest_hash"):
        raise CheckpointAncestryInvalid(
            "parent companion s1_run_manifest_hash "
            f"{companion.get('s1_run_manifest_hash')!r} != parent sidecar "
            f"checkpoint.s1_run_manifest_hash {checkpoint.get('s1_run_manifest_hash')!r} -- "
            "the presented companion is not the manifest that bound the parent's run"
        )

    parent_exposure_hashes = {
        exposure["release_root_hash"] for exposure in sidecar.get("dataset_exposures", [])
    }
    doc_exposure_hashes = {
        exposure["release_root_hash"] for exposure in doc.get("dataset_exposures", [])
    }
    if not parent_exposure_hashes <= doc_exposure_hashes:
        raise CheckpointAncestryInvalid(
            f"parent dataset_exposures release_root_hash set {sorted(parent_exposure_hashes)} "
            f"is not a subset of this document's declared exposures {sorted(doc_exposure_hashes)} "
            "-- DEC-006 transitive exposure refuses a warm start from a checkpoint exposed to "
            "data this document does not itself declare"
        )

    return {
        "checkpoint_id": checkpoint["checkpoint_id"],
        "ancestry_hash": sidecar["ancestry_hash"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "run_id": checkpoint["run_id"],
        "s1_run_manifest_hash": checkpoint["s1_run_manifest_hash"],
        "step": checkpoint["step"],
        "run_class": companion["run_class"],
    }


def build_base_ancestry(
    *,
    base_model: dict[str, Any],
    component_lock: dict[str, Any],
    s0_report: dict[str, Any],
    s0_report_path: Path,
    release_manifest: dict[str, Any],
    release_name: str,
    store_name: str,
    base_model_decision: dict[str, Any] | None,
    dataset_decision: dict[str, Any] | None,
    expected_revision: str,
    distribution_decision: dict[str, Any] | None = None,
    parent: dict[str, Any] | None = None,
    also_exposed: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Freeze one generation-0 (`document_kind: "base"`) checkpoint
    ancestry document.

    Refuses (`CheckpointAncestryInvalid`) when: the S0 report's own
    `legacy_ancestry.value` is not `False` (an unaudited or positive legacy
    checkpoint finding never reaches a `clean_pretrained` base ancestry);
    the S0 report's `inputs.release_root_hash` does not match the supplied
    `release_manifest`'s `release_root_hash` (the audit ran against a
    different release); the component lock's `base_model_weights`
    `artifact_sha256` does not match `base_model["weights_set_sha256"]`
    once normalised; or `base_model["revision"]` does not match
    `expected_revision`.

    `checkpoint_rights_status`, `checkpoint_privacy_status` and
    `distribution_allowed` are always derived here, never accepted as
    arguments -- see the module docstring.

    `parent` (DEC-028 D1, ancestry schema 1.1.0): `None` for a
    `clean_pretrained` base (unchanged v1 behaviour, `parents: []`), or
    `{"sidecar": <parent's finalized document_kind='checkpoint' sidecar
    document>, "companion": <parent's companion S1-format run manifest>}`
    for a `warm_start` base. When present: `initialization_kind` becomes
    `"warm_start"`, `parents` becomes a single entry taken from the sidecar
    (`checkpoint_id`, `ancestry_hash`, `checkpoint_sha256`, `run_id`,
    `s1_run_manifest_hash`, `step`) and the companion (`run_class` --
    never a caller argument). Refuses when the companion's own
    `s1_run_manifest_hash` disagrees with the sidecar's `checkpoint`
    object (the presented companion is not the manifest that bound the
    parent's run), or when the parent's `dataset_exposures` (by
    `release_root_hash`) is not a subset of this document's own declared
    exposures (DEC-006 transitive exposure). `also_exposed` declares further
    releases (`release_name`, `release_root_hash`, `store_name`) under the
    same `dataset_decision` -- how a cross-release warm start declares the
    parent's data instead of inheriting it silently.
    """
    legacy = s0_report.get("legacy_ancestry", {})
    legacy_value = legacy.get("value")
    if legacy_value is not False:
        raise CheckpointAncestryInvalid(
            f"S0 report legacy_ancestry.value must be False, got {legacy_value!r} -- "
            "an unaudited (null) or positive (true) legacy checkpoint finding refuses "
            "a clean_pretrained base ancestry"
        )

    s0_release_root_hash = s0_report.get("inputs", {}).get("release_root_hash")
    release_root_hash = release_manifest.get("release_root_hash")
    if s0_release_root_hash != release_root_hash:
        raise CheckpointAncestryInvalid(
            f"S0 report inputs.release_root_hash {s0_release_root_hash!r} != release "
            f"manifest release_root_hash {release_root_hash!r} -- the S0 audit was run "
            "against a different release than the one being bound"
        )

    lock_weights_hash = _lock_component_artifact_sha256(component_lock, "base_model_weights")
    normalised_weights_hash = normalise_sha256(base_model["weights_set_sha256"])
    if lock_weights_hash != normalised_weights_hash:
        raise CheckpointAncestryInvalid(
            f"component lock base_model_weights artifact_sha256 {lock_weights_hash!r} "
            f"!= base_model weights_set_sha256 {normalised_weights_hash!r} -- the lock "
            "and the pinned base model weights disagree"
        )

    if base_model["revision"] != expected_revision:
        raise CheckpointAncestryInvalid(
            f"base_model revision {base_model['revision']!r} != expected_revision "
            f"{expected_revision!r} -- refused rather than silently trusting whatever "
            "revision the caller happened to supply"
        )

    audited_roots = list(legacy.get("evidence", {}).get("data_roots", []))

    doc: dict[str, Any] = {
        "checkpoint_ancestry_schema_version": CHECKPOINT_ANCESTRY_SCHEMA_VERSION,
        "document_kind": "base",
        "base_model": base_model,
        "component_lock_hash": component_lock_hash(component_lock),
        "initialization_kind": "clean_pretrained" if parent is None else "warm_start",
        "parents": [],
        "adapters": [],
        "merge_history": [],
        "legacy_ancestry": {
            "value": False,
            "evidence": {
                "s0_report": {
                    "path": str(s0_report_path),
                    "sha256": file_sha256(s0_report_path),
                },
                "s0_report_value": legacy_value,
                "release_root_hash": release_root_hash,
                "audited_roots": audited_roots,
            },
        },
        "dataset_exposures": [
            {
                "release_name": release_name,
                "release_root_hash": release_root_hash,
                "store_name": store_name,
                "decision": dataset_decision,
            },
            *(
                {
                    "release_name": entry["release_name"],
                    "release_root_hash": entry["release_root_hash"],
                    "store_name": entry["store_name"],
                    "decision": dataset_decision,
                }
                for entry in also_exposed
            ),
        ],
        "exposure_ledger": {
            "kind": "not_produced",
            "reason": "no released checkpoint distribution root exists in v1",
        },
        "transitive_exposure_manifest": {
            "kind": "not_produced",
            "reason": "no transitive exposure manifest exists in v1",
        },
        "unknown_project_exposure": {
            "value": False,
            "reason": {
                "code": "clean_pretrained_audited",
                "detail": "S0 legacy_ancestry audit found no model checkpoints under audited_roots",
            },
        },
        "base_model_decision": base_model_decision,
        "distribution_decision": distribution_decision,
        "checkpoint": None,
        "ancestry_hash": None,
    }
    if parent is not None:
        doc["parents"] = [_parent_entry(parent, doc=doc)]
    doc["checkpoint_rights_status"] = derive_rights_status(doc)
    doc["checkpoint_privacy_status"] = derive_privacy_status(doc)
    doc["distribution_allowed"] = derive_distribution_allowed(doc)
    doc["ancestry_hash"] = compute_ancestry_hash(doc)

    _validate_schema(doc, what="built base ancestry document")
    return doc


def extend_for_checkpoint(
    base: dict[str, Any],
    *,
    step: int,
    run_id: str | None,
    s1_run_manifest_hash: str | None,
) -> dict[str, Any]:
    """Extend a verified `document_kind: "base"` document into a
    `document_kind: "checkpoint"` document for one checkpoint step.

    `run_id` and `s1_run_manifest_hash` must be null together or non-null
    together (a diagnostic run with no run id cannot carry a run manifest
    hash, and a bound run id must carry its manifest hash) -- refused
    otherwise.

    The returned document deliberately does NOT yet carry
    `checkpoint.checkpoint_sha256` / `checkpoint.checkpoint_id` (design
    decision 5): those are filled in by :func:`finalize_checkpoint_document`
    once the checkpoint's bytes exist, and this intermediate shape is not
    itself schema-valid -- callers must finalize before persisting or
    verifying it against the schema.
    """
    if (run_id is None) != (s1_run_manifest_hash is None):
        raise CheckpointAncestryInvalid(
            "run_id and s1_run_manifest_hash must both be null or both be "
            "non-null -- refused rather than allowing a bound run without "
            "its manifest hash or vice versa"
        )
    verify_base(base)

    doc = copy.deepcopy(base)
    doc["document_kind"] = "checkpoint"
    doc["checkpoint"] = {
        "step": step,
        "run_id": run_id,
        "s1_run_manifest_hash": s1_run_manifest_hash,
        "base_ancestry_hash": base["ancestry_hash"],
    }
    doc["ancestry_hash"] = compute_ancestry_hash(doc)
    return doc


def finalize_checkpoint_document(doc: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    """Fill in `checkpoint.checkpoint_sha256` / `checkpoint.checkpoint_id`
    now that `checkpoint_path`'s bytes exist, and validate the now-complete
    document against the schema.

    `ancestry_hash` is left untouched: :func:`ancestry_projection` already
    excludes both fields it fills in here, so the hash computed by
    :func:`extend_for_checkpoint` (and embedded inside the checkpoint file
    itself, design decision 5) still matches.
    """
    checkpoint_sha256 = file_sha256(checkpoint_path)
    checkpoint_id = compute_checkpoint_id(doc["ancestry_hash"], checkpoint_sha256)

    finalized = copy.deepcopy(doc)
    finalized["checkpoint"] = {
        **finalized["checkpoint"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": checkpoint_id,
    }
    _validate_schema(finalized, what="finalized checkpoint document")
    return finalized


def write_sidecar(doc: dict[str, Any], checkpoint_path: Path) -> Path:
    """Write `doc` to `checkpoint_path`'s sidecar via a `.tmp` file plus
    `os.replace` (design decision 5's last step) -- a crash mid-write
    leaves the old sidecar (or none) in place, never a half-written one."""
    path = sidecar_path(checkpoint_path)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def verify_base(doc: dict[str, Any]) -> None:
    """Verify one `document_kind: "base"` document: schema, kind, self-hash,
    and that both derived statuses and `distribution_allowed` agree with
    what is stored."""
    _validate_schema(doc, what="checkpoint ancestry base document")
    if doc["document_kind"] != "base":
        raise CheckpointAncestryInvalid(
            f"expected document_kind 'base', got {doc['document_kind']!r}"
        )
    expected_hash = compute_ancestry_hash(doc)
    if doc["ancestry_hash"] != expected_hash:
        raise CheckpointAncestryInvalid(
            "ancestry_hash does not match its own payload -- the base "
            "document was mutated after freezing"
        )
    if derive_rights_status(doc) != doc["checkpoint_rights_status"]:
        raise CheckpointAncestryInvalid(
            "checkpoint_rights_status does not match its derivation from "
            "base_model_decision and dataset_exposures"
        )
    if derive_privacy_status(doc) != doc["checkpoint_privacy_status"]:
        raise CheckpointAncestryInvalid(
            "checkpoint_privacy_status does not match its derivation from dataset_exposures"
        )
    if derive_distribution_allowed(doc) != doc["distribution_allowed"]:
        raise CheckpointAncestryInvalid(
            "distribution_allowed does not match its derivation from "
            "distribution_decision, rights and privacy"
        )


def verify_sidecar(
    checkpoint_path: Path,
    *,
    expect_base_ancestry_hash: str,
    expect_s1_run_manifest_hash: str | None,
    embedded_ancestry_hash: str,
) -> dict[str, Any]:
    """Verify a checkpoint's sidecar against the checkpoint file on disk
    and the caller's expectations, in this order: schema, kind, self-hash,
    derived statuses (including `distribution_allowed`), `checkpoint_sha256`
    against the real file bytes, the embedded (in-file) `ancestry_hash`
    against the sidecar's own, `s1_run_manifest_hash` against expectation,
    `base_ancestry_hash` against expectation, and finally `checkpoint_id`
    recomputed from the (now verified) ancestry hash and file sha256.

    Returns the verified sidecar document on success.
    """
    path = sidecar_path(checkpoint_path)
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise CheckpointAncestryInvalid(
            f"checkpoint ancestry sidecar {path} could not be read: {exc}"
        ) from exc

    _validate_schema(doc, what="checkpoint ancestry sidecar document")

    if doc["document_kind"] != "checkpoint":
        raise CheckpointAncestryInvalid(
            f"expected document_kind 'checkpoint', got {doc['document_kind']!r}"
        )

    expected_self_hash = compute_ancestry_hash(doc)
    if doc["ancestry_hash"] != expected_self_hash:
        raise CheckpointAncestryInvalid(
            "ancestry_hash does not match its own payload -- the sidecar was mutated after writing"
        )

    if derive_rights_status(doc) != doc["checkpoint_rights_status"]:
        raise CheckpointAncestryInvalid(
            "checkpoint_rights_status does not match its derivation from "
            "base_model_decision and dataset_exposures"
        )
    if derive_privacy_status(doc) != doc["checkpoint_privacy_status"]:
        raise CheckpointAncestryInvalid(
            "checkpoint_privacy_status does not match its derivation from dataset_exposures"
        )
    if derive_distribution_allowed(doc) != doc["distribution_allowed"]:
        raise CheckpointAncestryInvalid(
            "distribution_allowed does not match its derivation from "
            "distribution_decision, rights and privacy"
        )

    checkpoint = doc["checkpoint"]
    actual_checkpoint_sha256 = file_sha256(checkpoint_path)
    if checkpoint["checkpoint_sha256"] != actual_checkpoint_sha256:
        raise CheckpointAncestryInvalid(
            f"sidecar checkpoint_sha256 {checkpoint['checkpoint_sha256']!r} != actual "
            f"checkpoint file sha256 {actual_checkpoint_sha256!r} -- the checkpoint "
            "bytes were tampered with after the sidecar was frozen"
        )

    if embedded_ancestry_hash != doc["ancestry_hash"]:
        raise CheckpointAncestryInvalid(
            "ancestry_hash embedded in the checkpoint file does not match "
            "the sidecar's ancestry_hash -- the two were written from "
            "different documents"
        )

    if checkpoint["s1_run_manifest_hash"] != expect_s1_run_manifest_hash:
        raise CheckpointAncestryInvalid(
            f"sidecar s1_run_manifest_hash {checkpoint['s1_run_manifest_hash']!r} != "
            f"expected {expect_s1_run_manifest_hash!r}"
        )

    if checkpoint["base_ancestry_hash"] != expect_base_ancestry_hash:
        raise CheckpointAncestryInvalid(
            f"sidecar base_ancestry_hash {checkpoint['base_ancestry_hash']!r} != "
            f"expected {expect_base_ancestry_hash!r}"
        )

    expected_checkpoint_id = compute_checkpoint_id(
        doc["ancestry_hash"], checkpoint["checkpoint_sha256"]
    )
    if checkpoint["checkpoint_id"] != expected_checkpoint_id:
        raise CheckpointAncestryInvalid(
            f"sidecar checkpoint_id {checkpoint['checkpoint_id']!r} != recomputed "
            f"{expected_checkpoint_id!r}"
        )

    return doc

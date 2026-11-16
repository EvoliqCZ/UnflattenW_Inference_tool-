"""RFC004 §46 Model Capability Manifest + §39.6 full Release Manifest (W2-2).

Additive only: `manifests.model_release_id`/`_release_projection` keep
pinning `model_capability_manifest_hash = None` exactly as before (§39.6
A0, capability manifest "deferred"). This module adds the capability
manifest itself (schema `model-capability-manifest.schema.json`), its
§3.1 semantic hash, and a *separate* full §39.6 release-manifest builder
(schema `model-release-manifest.schema.json`) that a caller may use once it
actually has a capability manifest to reference -- it does not replace or
feed back into `manifests.model_release_id`.

§46.1's cross-capability gate truth table lives in
:mod:`unflatten.model.capability_gate`, not here, since it needs only the
enum values (`strict`), not this module's builders.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jsonschema.validators

from unflatten.ast.issues import Issue
from unflatten.model.hashing import domain_hash, make_id

_SCHEMA_DIR = Path(__file__).parent / "schemas"

CAPABILITY_MANIFEST_SCHEMA_VERSION = "0.1.0"
RELEASE_MANIFEST_SCHEMA_VERSION = "0.1.0"

#: §46 "referenčný status" -- treated as closed pending an errata that adds
#: a fifth value.
CAPABILITY_STATUS = frozenset({"strict", "partial", "unsupported", "experimental"})


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads((_SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))


def _validate(name: str, document: Mapping) -> list[Issue]:
    schema = _load_schema(name)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    validator = validator_cls(schema)
    return [
        Issue(
            code=f"{name.upper().replace('-', '_')}_SCHEMA_INVALID",
            severity="error",
            node_id=None,
            path="".join(f"/{p}" for p in err.absolute_path),
            message=err.message,
        )
        for err in validator.iter_errors(document)
    ]


def validate_capability_manifest(document: Mapping) -> list[Issue]:
    """JSON-Schema-only validation (§46 has no cross-field invariant beyond
    the schema itself)."""
    return _validate("model-capability-manifest", document)


def build_capability_manifest(
    *,
    node_types: Mapping,
    property_groups: Mapping,
    max_structural_sizes: Mapping,
    languages_scripts: list[str],
    input_kinds: list[str],
    dpi_range: Mapping,
    document_modes: Mapping,
    resource_recovery_modes: Mapping,
    optional_modules: Mapping,
    confidence_calibration_available: bool,
) -> dict[str, Any]:
    """Build a §46 Model Capability Manifest document. Raises `ValueError`
    if the assembled document fails its own schema (fail-closed, mirrors
    `manifests.model_release_id`'s validate-before-hash order)."""
    document = {
        "model_capability_manifest_schema_version": CAPABILITY_MANIFEST_SCHEMA_VERSION,
        "node_types": dict(node_types),
        "property_groups": dict(property_groups),
        "max_structural_sizes": dict(max_structural_sizes),
        "languages_scripts": list(languages_scripts),
        "input_kinds": list(input_kinds),
        "dpi_range": dict(dpi_range),
        "document_modes": dict(document_modes),
        "resource_recovery_modes": dict(resource_recovery_modes),
        "optional_modules": dict(optional_modules),
        "confidence_calibration_available": confidence_calibration_available,
    }
    issues = validate_capability_manifest(document)
    if issues:
        raise ValueError(
            "capability manifest invalid: "
            + "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        )
    return document


#: §3.1 domain-separated semantic hash for the capability manifest content
#: (not an ArtifactRef raw byte hash -- callers that also need the raw file
#: hash of their serialized copy compute it separately, same split as
#: `ast.ast_hash` vs `ast.ast_file_hash`, §35.2).
_CAPABILITY_DOMAIN = "rfc004.capability-manifest.v0"


def capability_manifest_hash(document: Mapping) -> str:
    """§3.1 semantic hash of a *valid* capability manifest. Does not
    validate; call :func:`validate_capability_manifest` first."""
    return domain_hash(_CAPABILITY_DOMAIN, document)


def validate_release_manifest(document: Mapping) -> list[Issue]:
    issues = _validate("model-release-manifest", document)
    if issues:
        return issues
    issues = []
    if document["component_lock"]["sha256"] != document["model_component_lock_hash"]:
        issues.append(
            Issue(
                code="RELEASE_MANIFEST_COMPONENT_LOCK_HASH_MISMATCH",
                severity="error",
                node_id=None,
                path="/component_lock/sha256",
                message="component_lock ArtifactRef.sha256 must equal model_component_lock_hash",
            )
        )
    if document["capability_manifest"]["sha256"] != document["model_capability_manifest_hash"]:
        issues.append(
            Issue(
                code="RELEASE_MANIFEST_CAPABILITY_HASH_MISMATCH",
                severity="error",
                node_id=None,
                path="/capability_manifest/sha256",
                message=(
                    "capability_manifest ArtifactRef.sha256 must equal "
                    "model_capability_manifest_hash"
                ),
            )
        )
    return issues


_RELEASE_DOMAIN = "rfc004.model-release.v0"


def build_release_manifest(
    *,
    architecture_id: str,
    model_version: str,
    model_component_lock_hash: str,
    model_capability_manifest_hash: str,
    supported_profiles: list[str],
    release_channel: str,
    component_lock_artifact: Mapping,
    capability_manifest_artifact: Mapping,
) -> dict[str, Any]:
    """Build a full §39.6 Model Release Manifest with a non-null capability
    hash. Independent of `manifests.model_release_id`'s A0 provisional
    projection: the two must never be confused, so the `model_release_id`
    computed here uses its own §39.6 CBOR projection (same formula, but
    `model_capability_manifest_hash` is real, not pinned `None`)."""
    projection = {
        "model_release_manifest_schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "architecture_id": architecture_id,
        "model_version": model_version,
        "model_component_lock_hash": model_component_lock_hash,
        "model_capability_manifest_hash": model_capability_manifest_hash,
        "supported_profiles": sorted(supported_profiles),
        "release_channel": release_channel,
    }
    release_id = make_id("model", _RELEASE_DOMAIN, projection)
    document = {
        "model_release_manifest_schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "model_release_id": release_id,
        "architecture_id": architecture_id,
        "model_version": model_version,
        "model_component_lock_hash": model_component_lock_hash,
        "model_capability_manifest_hash": model_capability_manifest_hash,
        "supported_profiles": supported_profiles,
        "release_channel": release_channel,
        "component_lock": dict(component_lock_artifact),
        "capability_manifest": dict(capability_manifest_artifact),
    }
    issues = validate_release_manifest(document)
    if issues:
        raise ValueError(
            "release manifest invalid: "
            + "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        )
    return document

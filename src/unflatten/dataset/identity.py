"""RFC003 §12 identity/hash layer.

Every identity is `H(domain, payload) = SHA-256(UTF-8(JCS([domain, payload])))`
with RFC 8785 JSON Canonicalization (RFC003 §12.1), reusing the RFC001 JCS
serializer `unflatten.ast.canonical_json.to_canonical_json`. That serializer
rejects floats by design — RFC003 identity payloads are integer-only.

Serializations:
- hash fields: `sha256:<64 lowercase hex>` (full digest)
- IDs: `<prefix>_<64 lowercase hex>` (same digest, no `sha256:` inside IDs)

Hash-leaf closure discipline (§12.1): every `*_hash` / `*_id` argument is
validated for exact `sha256:<64hex>` / `<prefix>_<64hex>` shape; a malformed
opaque digest is a `ValueError`. Full closure verification (that the digest is
recomputable from a registered projection or a pinned artifact) lives with the
caller registry, not here.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.dataset.enums import TASK_KIND, VIEW_KIND

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_RESOURCE_INPUT_FIELDS = frozenset({"role", "sha256", "size_bytes", "mime_type"})


def domain_hash(domain: str, payload: Any) -> str:
    """`H(domain, payload)` serialized as `sha256:<64 lowercase hex>`."""
    return f"sha256:{_digest_hex(domain, payload)}"


def make_id(prefix: str, domain: str, payload: Any) -> str:
    """`<prefix>_<64hex>` from the same digest as :func:`domain_hash`."""
    return f"{prefix}_{_digest_hex(domain, payload)}"


def _digest_hex(domain: str, payload: Any) -> str:
    if not isinstance(domain, str) or not domain:
        raise ValueError(f"domain must be a non-empty string, got {domain!r}")
    return hashlib.sha256(to_canonical_json([domain, payload])).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be 'sha256:<64 lowercase hex>', got {value!r}")
    return value


def _require_domain_id(name: str, value: Any, prefix: str) -> str:
    marker = f"{prefix}_"
    if (
        not isinstance(value, str)
        or not value.startswith(marker)
        or not _HEX64_RE.fullmatch(value[len(marker) :])
    ):
        raise ValueError(f"{name} must be '{prefix}_<64 lowercase hex>', got {value!r}")
    return value


def _require_nonempty_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_index(name: str, value: Any) -> int:
    # bool is an int subclass but canonicalizes as JSON true/false, silently
    # yielding a different digest — reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative int (bool rejected), got {value!r}")
    return value


def lineage_root_id_real(*, source_set_hash: str) -> str:
    """§12.2 real-source lineage root: `lin_` from `rfc003.lineage-real/v1`."""
    _require_sha256("source_set_hash", source_set_hash)
    return make_id("lin", "rfc003.lineage-real/v1", {"source_set_hash": source_set_hash})


def lineage_root_id_synthetic(*, synthetic_root_key: str) -> str:
    """§12.2 synthetic lineage root: `lin_` from `rfc003.lineage-synthetic/v1`."""
    _require_sha256("synthetic_root_key", synthetic_root_key)
    return make_id(
        "lin", "rfc003.lineage-synthetic/v1", {"synthetic_root_key": synthetic_root_key}
    )


def synthetic_root_key(
    *,
    dataset_namespace: str,
    generator_version: str,
    recipe_id: str,
    recipe_hash: str,
    root_instance_index: int,
) -> str:
    """§12.2 RNG-independent synthetic root key (a hash field, not an ID)."""
    _require_nonempty_str("dataset_namespace", dataset_namespace)
    _require_nonempty_str("generator_version", generator_version)
    _require_nonempty_str("recipe_id", recipe_id)
    _require_sha256("recipe_hash", recipe_hash)
    _require_index("root_instance_index", root_instance_index)
    return domain_hash(
        "rfc003.synthetic-root/v1",
        {
            "dataset_namespace": dataset_namespace,
            "generator_version": generator_version,
            "recipe_id": recipe_id,
            "recipe_hash": recipe_hash,
            "root_instance_index": root_instance_index,
        },
    )


def recipe_hash(
    *,
    recipe_schema_version: str,
    recipe_id: str,
    recipe_body: dict[str, Any],
    resource_inputs: list[dict[str, Any]],
    dependency_recipe_hashes: list[str],
) -> str:
    """§12.2 recipe hash over `rfc003.synthetic-recipe/v1`.

    `resource_inputs` and `dependency_recipe_hashes` are sorted per the
    normative projection. Duplicate dependency edges and (post-compute)
    self-reference are rejected here; the full DAG/cycle check over the
    dependency graph lives with the caller's recipe registry, which alone
    can resolve dependency hashes to recipe artifacts.
    """
    for index, resource in enumerate(resource_inputs):
        if not isinstance(resource, dict) or set(resource) != _RESOURCE_INPUT_FIELDS:
            raise ValueError(
                f"resource_inputs[{index}] must be a record with exactly the keys "
                f"{sorted(_RESOURCE_INPUT_FIELDS)} (RFC003 §12.2; transport 'ref' "
                f"never enters identity), got {resource!r}"
            )
        _require_sha256(f"resource_inputs[{index}].sha256", resource["sha256"])
    for dep in dependency_recipe_hashes:
        _require_sha256("dependency_recipe_hash", dep)
    if len(set(dependency_recipe_hashes)) != len(dependency_recipe_hashes):
        raise ValueError("dependency_recipe_hashes contains duplicate edges")
    result = domain_hash(
        "rfc003.synthetic-recipe/v1",
        {
            "recipe_schema_version": recipe_schema_version,
            "recipe_id": recipe_id,
            "recipe_body": recipe_body,
            "resource_inputs": sorted(
                resource_inputs,
                key=lambda r: to_canonical_json(r),
            ),
            "dependency_recipe_hashes": sorted(dependency_recipe_hashes),
        },
    )
    if result in dependency_recipe_hashes:
        raise ValueError("recipe_hash self-reference in dependency_recipe_hashes")
    return result


def document_bundle_id(
    *,
    lineage_root_id: str,
    ast_hash: str,
    ast_file_hash: str,
    resource_manifest_hash: str,
    label_revision_hash: str,
) -> str:
    """§12.3 authoritative DocumentBundle ID (`bundle_`)."""
    _require_domain_id("lineage_root_id", lineage_root_id, "lin")
    _require_sha256("ast_hash", ast_hash)
    _require_sha256("ast_file_hash", ast_file_hash)
    _require_sha256("resource_manifest_hash", resource_manifest_hash)
    _require_sha256("label_revision_hash", label_revision_hash)
    return make_id(
        "bundle",
        "rfc003.document-bundle/v1",
        {
            "lineage_root_id": lineage_root_id,
            "ast_hash": ast_hash,
            "ast_file_hash": ast_file_hash,
            "resource_manifest_hash": resource_manifest_hash,
            "label_revision_hash": label_revision_hash,
        },
    )


def render_view_id(
    *,
    document_bundle_id: str | None,
    candidate_id: str | None,
    hypothesis_hash: str | None,
    view_kind: str,
    input_artifact_set_hash: str,
    render_profile_hash: str,
    transform_chain_hash: str,
    output_artifact_set_hash: str,
) -> str:
    """§12.4 RenderView ID (`render_`).

    Exactly one of `document_bundle_id` / `candidate_id` is non-null.
    A DocumentBundle authority must carry `hypothesis_hash = None`;
    `view_kind == "hypothesis_preview"` requires a `hypothesis_hash`.
    """
    if (document_bundle_id is None) == (candidate_id is None):
        raise ValueError(
            "authority_ref requires exactly one of document_bundle_id/candidate_id non-null"
        )
    if document_bundle_id is not None:
        _require_domain_id("document_bundle_id", document_bundle_id, "bundle")
        if hypothesis_hash is not None:
            raise ValueError("hypothesis_hash must be null for a DocumentBundle authority")
    else:
        _require_domain_id("candidate_id", candidate_id, "candidate")
        if hypothesis_hash is not None:
            _require_sha256("hypothesis_hash", hypothesis_hash)
    if view_kind not in VIEW_KIND:
        raise ValueError(f"unknown view_kind {view_kind!r}")
    if view_kind == "hypothesis_preview" and hypothesis_hash is None:
        raise ValueError("view_kind 'hypothesis_preview' requires hypothesis_hash")
    if view_kind == "canonical_render" and document_bundle_id is None:
        raise ValueError("view_kind 'canonical_render' requires a DocumentBundle authority")
    if view_kind in ("source_render", "human_reference") and hypothesis_hash is not None:
        raise ValueError(f"view_kind {view_kind!r} must carry hypothesis_hash null")
    _require_sha256("input_artifact_set_hash", input_artifact_set_hash)
    _require_sha256("render_profile_hash", render_profile_hash)
    _require_sha256("transform_chain_hash", transform_chain_hash)
    _require_sha256("output_artifact_set_hash", output_artifact_set_hash)
    return make_id(
        "render",
        "rfc003.render-view/v1",
        {
            "authority_ref": {
                "document_bundle_id": document_bundle_id,
                "candidate_id": candidate_id,
                "hypothesis_hash": hypothesis_hash,
            },
            "view_kind": view_kind,
            "input_artifact_set_hash": input_artifact_set_hash,
            "render_profile_hash": render_profile_hash,
            "transform_chain_hash": transform_chain_hash,
            "output_artifact_set_hash": output_artifact_set_hash,
        },
    )


def task_view_id(
    *,
    render_view_id: str,
    task_kind: str,
    input_selection_hash: str,
    projection_hash: str,
    mask_policy_hash: str,
    mask_projection_hash: str,
    reasoning_spec_id: str | None = None,
    reasoning_derivation_policy_hash: str | None = None,
) -> str:
    """§12.5 TaskView ID (`task_`).

    `reasoning_spec_id` / `reasoning_derivation_policy_hash` are the
    RFC003-ERRATA-003 E2 rule 6 identity extension for
    `task_kind == "reasoning_constructive_v0"` only. Both are required
    together exactly for that `task_kind` and forbidden (must stay `None`)
    for every other `task_kind`; when `None` they are omitted from the
    hashed payload entirely, so every pre-ERRATA-003 `task_view_id` is
    byte-for-byte unchanged.
    """
    _require_domain_id("render_view_id", render_view_id, "render")
    if task_kind not in TASK_KIND:
        raise ValueError(f"unknown task_kind {task_kind!r}")
    _require_sha256("input_selection_hash", input_selection_hash)
    _require_sha256("projection_hash", projection_hash)
    _require_sha256("mask_policy_hash", mask_policy_hash)
    _require_sha256("mask_projection_hash", mask_projection_hash)
    is_reasoning_constructive = task_kind == "reasoning_constructive_v0"
    if is_reasoning_constructive:
        _require_nonempty_str("reasoning_spec_id", reasoning_spec_id)
        _require_sha256("reasoning_derivation_policy_hash", reasoning_derivation_policy_hash)
    elif reasoning_spec_id is not None or reasoning_derivation_policy_hash is not None:
        raise ValueError(
            "reasoning_spec_id/reasoning_derivation_policy_hash must be None "
            f"for task_kind {task_kind!r} (only reasoning_constructive_v0 carries them)"
        )
    payload: dict[str, Any] = {
        "render_view_id": render_view_id,
        "task_kind": task_kind,
        "input_selection_hash": input_selection_hash,
        "projection_hash": projection_hash,
        "mask_policy_hash": mask_policy_hash,
        "mask_projection_hash": mask_projection_hash,
    }
    if is_reasoning_constructive:
        payload["reasoning_spec_id"] = reasoning_spec_id
        payload["reasoning_derivation_policy_hash"] = reasoning_derivation_policy_hash
    return make_id("task", "rfc003.task-view/v1", payload)


_CONTENT_REF_FIELDS = frozenset({"sha256", "size_bytes", "mime_type"})
_ANCESTRY_SOURCE_KINDS = frozenset(
    {"render_page", "task_input", "fragment_page", "model_candidate"}
)
_ANCESTRY_DERIVATION_KINDS = frozenset({"identity", "transform_chain"})
_ANCESTRY_CANDIDATE_ORIGINS = frozenset({"frozen_model", "controlled_corruption"})
_ANCESTRY_PRODUCER_EXPOSURES = frozenset(
    {"in_sample", "out_of_fold", "earlier_checkpoint"}
)


def _require_content_ref(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CONTENT_REF_FIELDS:
        raise ValueError(
            f"{name} must be a record with exactly the keys "
            f"{sorted(_CONTENT_REF_FIELDS)}, got {value!r}"
        )
    _require_sha256(f"{name}.sha256", value["sha256"])
    return dict(value)


def input_ancestry_hash(
    *,
    document_bundle_id: str | None,
    candidate_id: str | None,
    render_view_id: str,
    parent_task_view_id: str | None,
    bindings: list[dict[str, Any]],
) -> str:
    """§30.10 Input Ancestry hash, extended by RFC003-ERRATA-003 E3
    (the `model_candidate` binding branch).

    Enforces the structural invariants of §30.10 (opaque-id shapes, closed
    `source_kind`/`derivation_kind` vocabularies, ordinal contiguity,
    `derivation_kind` <-> transform/lineage nullability) and the E3
    `model_candidate` field rules (rule "derivation_kind must be
    transform_chain", candidate_origin-conditional producer/corruption
    fields). Cross-artefact closure against a parent TaskView/RenderView
    (source_page_index membership, fragment/lineage existence) is the
    caller's/`envelope_validator.validate_input_ancestry`'s job, not this
    pure projection.
    """
    if (document_bundle_id is None) == (candidate_id is None):
        raise ValueError(
            "authority_ref requires exactly one of document_bundle_id/candidate_id non-null"
        )
    if document_bundle_id is not None:
        _require_domain_id("document_bundle_id", document_bundle_id, "bundle")
    else:
        _require_domain_id("candidate_id", candidate_id, "candidate")
    _require_domain_id("render_view_id", render_view_id, "render")
    if parent_task_view_id is not None:
        _require_domain_id("parent_task_view_id", parent_task_view_id, "task")
    if not bindings:
        raise ValueError("bindings must be non-empty")

    normalized: list[dict[str, Any]] = []
    for index, binding in enumerate(bindings):
        prefix = f"bindings[{index}]"
        if binding.get("input_ordinal") != index:
            raise ValueError(
                f"{prefix}.input_ordinal must equal its position {index}, "
                f"got {binding.get('input_ordinal')!r}"
            )
        source_kind = binding.get("source_kind")
        if source_kind not in _ANCESTRY_SOURCE_KINDS:
            raise ValueError(f"{prefix}.source_kind unknown: {source_kind!r}")
        derivation_kind = binding.get("derivation_kind")
        if derivation_kind not in _ANCESTRY_DERIVATION_KINDS:
            raise ValueError(f"{prefix}.derivation_kind unknown: {derivation_kind!r}")
        input_artifact = _require_content_ref(
            f"{prefix}.input_artifact", binding.get("input_artifact")
        )
        source_artifact = _require_content_ref(
            f"{prefix}.source_artifact", binding.get("source_artifact")
        )
        source_page_index = _require_index(
            f"{prefix}.source_page_index", binding.get("source_page_index")
        )

        source_input_ordinal = binding.get("source_input_ordinal")
        source_fragment_id = binding.get("source_fragment_id")
        transform_chain_hash = binding.get("transform_chain_hash")
        lineage_edge_id = binding.get("lineage_edge_id")
        candidate_origin = binding.get("candidate_origin")
        producer_checkpoint_sha256 = binding.get("producer_checkpoint_sha256")
        producer_component_lock_hash = binding.get("producer_component_lock_hash")
        producer_decode_policy_hash = binding.get("producer_decode_policy_hash")
        producer_exposure = binding.get("producer_exposure")
        corruption_policy_hash = binding.get("corruption_policy_hash")

        if source_fragment_id is not None:
            _require_domain_id(f"{prefix}.source_fragment_id", source_fragment_id, "fragment")
        if lineage_edge_id is not None:
            _require_domain_id(f"{prefix}.lineage_edge_id", lineage_edge_id, "edge")
        for label, value in (
            ("transform_chain_hash", transform_chain_hash),
            ("producer_checkpoint_sha256", producer_checkpoint_sha256),
            ("producer_component_lock_hash", producer_component_lock_hash),
            ("producer_decode_policy_hash", producer_decode_policy_hash),
            ("corruption_policy_hash", corruption_policy_hash),
        ):
            if value is not None:
                _require_sha256(f"{prefix}.{label}", value)
        if candidate_origin is not None and candidate_origin not in _ANCESTRY_CANDIDATE_ORIGINS:
            raise ValueError(f"{prefix}.candidate_origin unknown: {candidate_origin!r}")
        if producer_exposure is not None and producer_exposure not in _ANCESTRY_PRODUCER_EXPOSURES:
            raise ValueError(f"{prefix}.producer_exposure unknown: {producer_exposure!r}")

        if derivation_kind == "identity" and (
            transform_chain_hash is not None or lineage_edge_id is not None
        ):
            raise ValueError(
                f"{prefix}: derivation_kind 'identity' requires transform_chain_hash "
                "and lineage_edge_id null (§30.10 rule 5)"
            )
        if derivation_kind == "transform_chain" and (
            transform_chain_hash is None or lineage_edge_id is None
        ):
            raise ValueError(
                f"{prefix}: derivation_kind 'transform_chain' requires "
                "transform_chain_hash and lineage_edge_id non-null (§30.10 rule 5)"
            )

        if source_kind == "render_page":
            if (
                source_input_ordinal is not None
                or source_fragment_id is not None
                or candidate_origin is not None
            ):
                raise ValueError(
                    f"{prefix}: source_kind 'render_page' forbids source_input_ordinal/"
                    "source_fragment_id/candidate_origin (§30.10 rule 2)"
                )
        elif source_kind == "task_input":
            if (
                source_input_ordinal is None
                or source_fragment_id is not None
                or candidate_origin is not None
            ):
                raise ValueError(
                    f"{prefix}: source_kind 'task_input' requires source_input_ordinal "
                    "non-null and forbids source_fragment_id/candidate_origin (§30.10 rule 3)"
                )
        elif source_kind == "fragment_page":
            if (
                source_input_ordinal is not None
                or source_fragment_id is None
                or candidate_origin is not None
            ):
                raise ValueError(
                    f"{prefix}: source_kind 'fragment_page' requires source_fragment_id "
                    "non-null and forbids source_input_ordinal/candidate_origin (§30.10 rule 3)"
                )
        else:  # model_candidate (RFC003-ERRATA-003 E3)
            if source_input_ordinal is not None or source_fragment_id is not None:
                raise ValueError(
                    f"{prefix}: source_kind 'model_candidate' forbids "
                    "source_input_ordinal/source_fragment_id (ERRATA-003 E3)"
                )
            if candidate_origin is None:
                raise ValueError(
                    f"{prefix}: source_kind 'model_candidate' requires candidate_origin "
                    "(ERRATA-003 E3)"
                )
            if derivation_kind != "transform_chain":
                raise ValueError(
                    f"{prefix}: source_kind 'model_candidate' requires derivation_kind "
                    "'transform_chain' (ERRATA-003 E3)"
                )
            if candidate_origin == "frozen_model":
                if (
                    producer_checkpoint_sha256 is None
                    or producer_component_lock_hash is None
                    or producer_decode_policy_hash is None
                ):
                    raise ValueError(
                        f"{prefix}: candidate_origin 'frozen_model' requires "
                        "producer_checkpoint_sha256, producer_component_lock_hash and "
                        "producer_decode_policy_hash non-null (ERRATA-003 E3)"
                    )
                if corruption_policy_hash is not None:
                    raise ValueError(
                        f"{prefix}: candidate_origin 'frozen_model' forbids "
                        "corruption_policy_hash (ERRATA-003 E3)"
                    )
                if producer_exposure is None:
                    raise ValueError(
                        f"{prefix}: candidate_origin 'frozen_model' requires "
                        "producer_exposure (ERRATA-003 E3 rule 5)"
                    )
            else:  # controlled_corruption
                if corruption_policy_hash is None:
                    raise ValueError(
                        f"{prefix}: candidate_origin 'controlled_corruption' requires "
                        "corruption_policy_hash non-null (ERRATA-003 E3)"
                    )
                if (
                    producer_checkpoint_sha256 is not None
                    or producer_component_lock_hash is not None
                    or producer_decode_policy_hash is not None
                    or producer_exposure is not None
                ):
                    raise ValueError(
                        f"{prefix}: candidate_origin 'controlled_corruption' forbids "
                        "producer_* fields (ERRATA-003 E3)"
                    )

        normalized.append(
            {
                "input_ordinal": index,
                "input_artifact": input_artifact,
                "source_kind": source_kind,
                "source_page_index": source_page_index,
                "source_input_ordinal": source_input_ordinal,
                "source_fragment_id": source_fragment_id,
                "source_artifact": source_artifact,
                "derivation_kind": derivation_kind,
                "transform_chain_hash": transform_chain_hash,
                "lineage_edge_id": lineage_edge_id,
                "candidate_origin": candidate_origin,
                "producer_checkpoint_sha256": producer_checkpoint_sha256,
                "producer_component_lock_hash": producer_component_lock_hash,
                "producer_decode_policy_hash": producer_decode_policy_hash,
                "producer_exposure": producer_exposure,
                "corruption_policy_hash": corruption_policy_hash,
            }
        )

    return domain_hash(
        "rfc003.input-ancestry/v1",
        {
            "authority_ref": {
                "document_bundle_id": document_bundle_id,
                "candidate_id": candidate_id,
            },
            "render_view_id": render_view_id,
            "parent_task_view_id": parent_task_view_id,
            "bindings": normalized,
        },
    )


def task_example_id(
    *,
    task_view_id: str,
    selection_projection_hash: str,
    transform_chain_hash: str,
) -> str:
    """§12.6 optional crop/window TaskExample ID (`example_`)."""
    _require_domain_id("task_view_id", task_view_id, "task")
    _require_sha256("selection_projection_hash", selection_projection_hash)
    _require_sha256("transform_chain_hash", transform_chain_hash)
    return make_id(
        "example",
        "rfc003.task-example/v1",
        {
            "task_view_id": task_view_id,
            "selection_projection_hash": selection_projection_hash,
            "transform_chain_hash": transform_chain_hash,
        },
    )


def operation_seed(
    *,
    synthetic_root_key: str,
    recipe_cohort_id: str,
    operation_id: str,
    variant_index: int,
) -> int:
    """§20.3 deterministic RNG seed for one generative operation.

    Returns an unsigned 64-bit integer: the first 8 bytes of the
    `rfc003.operation-seed/v1` digest interpreted as big-endian
    (network byte order), i.e. `int.from_bytes(digest[:8], "big")`.
    """
    _require_sha256("synthetic_root_key", synthetic_root_key)
    _require_index("variant_index", variant_index)
    digest = hashlib.sha256(
        to_canonical_json(
            [
                "rfc003.operation-seed/v1",
                {
                    "synthetic_root_key": synthetic_root_key,
                    "recipe_cohort_id": recipe_cohort_id,
                    "operation_id": operation_id,
                    "variant_index": variant_index,
                },
            ]
        )
    ).digest()
    return int.from_bytes(digest[:8], "big")

"""RFC003 D0 envelope validators (§14, §22.3, §30.8 subset).

Each validator runs a JSON Schema pass (Draft 2020-12, closed envelopes) and,
only when that pass is clean, a semantic pass. The core D0 guarantee is ID
integrity: the envelope's own ID is recomputed from its fields through the
:mod:`unflatten.dataset.identity` projections and must match byte-for-byte.

Validators never raise on invalid documents — they return ``list[Issue]``
(empty means valid), reusing the RFC001 Issue model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema

from unflatten.ast.issues import Issue
from unflatten.dataset import identity
from unflatten.dataset.artifact_ref import ArtifactRef

_SCHEMA_DIR = Path(__file__).parent / "schemas"


@cache
def _validator(name: str) -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{name}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _pointer(path: list[Any]) -> str:
    # RFC 6901: the root pointer is the empty string.
    return "".join(f"/{p}" for p in path)


def _schema_issues(name: str, doc: Any) -> list[Issue]:
    return [
        Issue(
            code="SCHEMA_VALIDATION_ERROR",
            severity="error",
            node_id=None,
            path=_pointer(list(error.absolute_path)),
            message=error.message,
            details={"validator": error.validator},
        )
        for error in _validator(name).iter_errors(doc)
    ]


def _issue(code: str, path: str, message: str, **details: Any) -> Issue:
    return Issue(
        code=code, severity="error", node_id=None, path=path, message=message, details=details
    )


def _float_issues(node: Any, path: list[Any] | None = None) -> list[Issue]:
    """RFC003 envelopes are integer-only; JSON Schema 'integer' accepts 300.0."""
    path = path or []
    if isinstance(node, float):
        return [
            _issue(
                "FLOAT_IN_ENVELOPE",
                _pointer(path),
                f"float {node!r} is forbidden: RFC003 envelope payloads are integer-only",
            )
        ]
    if isinstance(node, dict):
        return [i for key, value in node.items() for i in _float_issues(value, [*path, key])]
    if isinstance(node, list):
        return [i for idx, value in enumerate(node) for i in _float_issues(value, [*path, idx])]
    return []


def _check_artifact_ref(data: dict[str, Any], path: str, issues: list[Issue]) -> None:
    """Semantic ArtifactRef rules (traversal, URI scheme, backslash, ...)."""
    try:
        ArtifactRef.from_dict(data)
    except ValueError as exc:
        issues.append(_issue("ARTIFACT_REF_INVALID", path, str(exc)))


def _check_id_integrity(
    actual: str, expected: str, path: str, issues: list[Issue]
) -> None:
    if actual != expected:
        issues.append(
            _issue(
                "ID_INTEGRITY_MISMATCH",
                path,
                f"stored ID {actual!r} does not match the ID recomputed "
                f"from the envelope's own fields",
                expected=expected,
            )
        )


def validate_bundle(doc: Any) -> list[Issue]:
    """Validate a DocumentBundle envelope (RFC003 §14 D0 subset)."""
    issues = _schema_issues("document_bundle", doc) + _float_issues(doc)
    if issues:
        return issues

    _check_artifact_ref(doc["ast"], "/ast", issues)
    _check_artifact_ref(doc["resources_manifest"], "/resources_manifest", issues)

    supersedes = doc.get("supersedes_revision_id")
    if supersedes is not None and supersedes >= doc["revision_id"]:
        issues.append(
            _issue(
                "SUPERSEDES_NOT_OLDER",
                "/supersedes_revision_id",
                f"supersedes_revision_id {supersedes!r} must be strictly older "
                f"than revision_id {doc['revision_id']!r}",
            )
        )

    expected = identity.document_bundle_id(
        lineage_root_id=doc["lineage_root_id"],
        ast_hash=doc["ast_hash"],
        ast_file_hash=doc["ast_file_hash"],
        resource_manifest_hash=doc["resource_manifest_hash"],
        label_revision_hash=doc["label_revision_hash"],
    )
    _check_id_integrity(doc["document_bundle_id"], expected, "/document_bundle_id", issues)
    return issues


def validate_render_view(doc: Any) -> list[Issue]:
    """Validate a RenderView envelope (RFC003 §22.3 D0 subset)."""
    issues = _schema_issues("render_view", doc) + _float_issues(doc)
    if issues:
        return issues

    _check_artifact_ref(doc["outputs"]["pages_index"], "/outputs/pages_index", issues)

    authority = doc["authority_ref"]
    bundle_id = authority["document_bundle_id"]
    candidate_id = authority["candidate_id"]
    hypothesis_hash = authority["hypothesis_hash"]
    view_kind = doc["view_kind"]

    # §22.3 authority/view contract (the hypothesis_hash null/non-null half is
    # already enforced by the schema's allOf; here the authority pairing).
    if view_kind == "canonical_render" and bundle_id is None:
        issues.append(
            _issue(
                "VIEW_KIND_AUTHORITY_MISMATCH",
                "/authority_ref",
                "view_kind 'canonical_render' requires a DocumentBundle authority",
            )
        )
    elif view_kind == "hypothesis_preview" and candidate_id is None:
        issues.append(
            _issue(
                "VIEW_KIND_AUTHORITY_MISMATCH",
                "/authority_ref",
                "view_kind 'hypothesis_preview' requires a Candidate authority",
            )
        )
    elif view_kind in ("source_render", "human_reference") and hypothesis_hash is not None:
        issues.append(
            _issue(
                "VIEW_KIND_AUTHORITY_MISMATCH",
                "/authority_ref",
                f"view_kind {view_kind!r} must carry hypothesis_hash null",
            )
        )

    # The identity constructor enforces the same §12.4 contract and would
    # raise on the combos flagged above — skip ID recomputation once the
    # authority/view_kind pairing is already reported broken.
    if not any(issue.code == "VIEW_KIND_AUTHORITY_MISMATCH" for issue in issues):
        expected = identity.render_view_id(
            document_bundle_id=bundle_id,
            candidate_id=candidate_id,
            hypothesis_hash=hypothesis_hash,
            view_kind=view_kind,
            input_artifact_set_hash=doc["input_artifact_set_hash"],
            render_profile_hash=doc["render_profile"]["render_profile_hash"],
            transform_chain_hash=doc["transform_chain_hash"],
            output_artifact_set_hash=doc["outputs"]["output_artifact_set_hash"],
        )
        _check_id_integrity(doc["render_view_id"], expected, "/render_view_id", issues)
    return issues


def validate_task_view(doc: Any) -> list[Issue]:
    """Validate a TaskView envelope (RFC003 §30.8 D0 subset)."""
    issues = _schema_issues("task_view", doc) + _float_issues(doc)
    if issues:
        return issues

    selection = doc["input_selection"]
    for index, artifact in enumerate(selection["input_artifacts"]):
        _check_artifact_ref(artifact, f"/input_selection/input_artifacts/{index}", issues)
    _check_artifact_ref(doc["masks"]["mask_artifact"], "/masks/mask_artifact", issues)

    pages = selection["page_indices"]
    if pages != sorted(pages):
        issues.append(
            _issue(
                "PAGE_INDICES_NOT_ASCENDING",
                "/input_selection/page_indices",
                f"page_indices must be unique ascending, got {pages!r}",
            )
        )

    crop = selection["crop_bbox_ppm"]
    if crop is not None and not (crop[0] < crop[2] and crop[1] < crop[3]):
        issues.append(
            _issue(
                "CROP_BBOX_INVALID",
                "/input_selection/crop_bbox_ppm",
                f"crop_bbox_ppm requires x1<x2 and y1<y2, got {crop!r}",
            )
        )

    projection = doc["target_projection"]
    canonical_target = projection["canonical_target_bundle"]
    candidate_target = projection["candidate_target_bundle"]
    chosen_target = canonical_target if canonical_target is not None else candidate_target
    if chosen_target is not None:
        _check_artifact_ref(
            chosen_target,
            "/target_projection/canonical_target_bundle"
            if canonical_target is not None
            else "/target_projection/candidate_target_bundle",
            issues,
        )
        if projection["authority_target_hash"] != chosen_target["sha256"]:
            issues.append(
                _issue(
                    "AUTHORITY_TARGET_HASH_MISMATCH",
                    "/target_projection/authority_target_hash",
                    "authority_target_hash must equal the raw sha256 of the "
                    "selected target bundle ArtifactRef",
                    expected=chosen_target["sha256"],
                )
            )

    task_kind = doc["task_kind"]
    if task_kind == "full_ast_exact_v0" and canonical_target is None:
        issues.append(
            _issue(
                "TASK_KIND_TARGET_MISMATCH",
                "/target_projection",
                "task_kind 'full_ast_exact_v0' requires a DocumentBundle target "
                "(canonical_target_bundle non-null)",
            )
        )
    elif task_kind == "partial_ast_v0" and candidate_target is None:
        issues.append(
            _issue(
                "TASK_KIND_TARGET_MISMATCH",
                "/target_projection",
                "task_kind 'partial_ast_v0' requires a Candidate target "
                "(candidate_target_bundle non-null)",
            )
        )

    eligibility = doc["eligibility"]
    if eligibility["training_allowed"] and eligibility["status"] not in (
        "train_ready",
        "partial_ready",
    ):
        issues.append(
            _issue(
                "ELIGIBILITY_INVARIANT_VIOLATION",
                "/eligibility",
                "training_allowed requires status 'train_ready' or 'partial_ready', "
                f"got {eligibility['status']!r}",
            )
        )
    if eligibility["verification_tier"] == "human_gold" and eligibility["training_allowed"]:
        issues.append(
            _issue(
                "HUMAN_GOLD_TRAINING_FORBIDDEN",
                "/eligibility",
                "verification_tier 'human_gold' is eval-only (§33.4): "
                "training_allowed must be false",
            )
        )

    _check_reasoning_fields(doc, task_kind, issues)

    # identity.task_view_id() unconditionally requires reasoning_spec_id/
    # reasoning_derivation_policy_hash once task_kind is reasoning_constructive_v0
    # (§12.5); once REASONING_FIELDS_REQUIRED already reported one missing, ID
    # recomputation cannot proceed (mirrors validate_render_view's skip on an
    # already-reported broken authority/view_kind pairing).
    if any(issue.code == "REASONING_FIELDS_REQUIRED" for issue in issues):
        return issues

    reasoning_kwargs: dict[str, Any] = {}
    if task_kind == "reasoning_constructive_v0":
        reasoning_kwargs = {
            "reasoning_spec_id": doc["reasoning_spec_id"],
            "reasoning_derivation_policy_hash": doc["reasoning_derivation_policy_hash"],
        }
    expected = identity.task_view_id(
        render_view_id=doc["render_view_id"],
        task_kind=task_kind,
        input_selection_hash=selection["input_selection_hash"],
        projection_hash=projection["projection_hash"],
        mask_policy_hash=doc["masks"]["mask_policy_hash"],
        mask_projection_hash=doc["masks"]["mask_projection_hash"],
        **reasoning_kwargs,
    )
    _check_id_integrity(doc["task_view_id"], expected, "/task_view_id", issues)
    return issues


#: RFC003-ERRATA-003 E2 rule 3: the reasoning_spec_id family banned while
#: DEC-033 (no cross-page continuation) stands. `reasoning_spec_id` has the
#: shape `reasoning.<family>.<version>`.
_FORBIDDEN_REASONING_FAMILY = "continuation"


def _check_reasoning_fields(doc: dict[str, Any], task_kind: str, issues: list[Issue]) -> None:
    """RFC003-ERRATA-003 E2: §30.11 envelope fields, present only on the schema's
    0.2.0 arm. `reasoning_spec_id`/`reasoning_derivation_policy_hash`/
    `reasoning_aux_target_ref` must be non-null iff `task_kind ==
    "reasoning_constructive_v0"`, and null for every other task_kind
    (including `reasoning_corrective_v0`, which carries no aux targets)."""
    spec_id = doc.get("reasoning_spec_id")
    policy_hash = doc.get("reasoning_derivation_policy_hash")
    aux_target = doc.get("reasoning_aux_target_ref")

    if task_kind == "reasoning_constructive_v0":
        if spec_id is None or policy_hash is None or aux_target is None:
            issues.append(
                _issue(
                    "REASONING_FIELDS_REQUIRED",
                    "/reasoning_spec_id",
                    "task_kind 'reasoning_constructive_v0' requires non-null "
                    "reasoning_spec_id, reasoning_derivation_policy_hash and "
                    "reasoning_aux_target_ref (RFC003-ERRATA-003 E2)",
                )
            )
            return
        _check_artifact_ref(aux_target, "/reasoning_aux_target_ref", issues)
        family = spec_id.split(".")[1] if spec_id.count(".") >= 2 else None
        if family == _FORBIDDEN_REASONING_FAMILY:
            issues.append(
                _issue(
                    "REASONING_CONTINUATION_FORBIDDEN",
                    "/reasoning_spec_id",
                    f"reasoning_spec_id family {_FORBIDDEN_REASONING_FAMILY!r} is "
                    "forbidden while DEC-033 stands (RFC003-ERRATA-003 E2 rule 3)",
                )
            )
    elif spec_id is not None or policy_hash is not None or aux_target is not None:
        issues.append(
            _issue(
                "REASONING_FIELDS_FORBIDDEN",
                "/reasoning_spec_id",
                f"task_kind {task_kind!r} must carry reasoning_spec_id, "
                "reasoning_derivation_policy_hash and reasoning_aux_target_ref "
                "null (RFC003-ERRATA-003 E2)",
            )
        )


def validate_input_ancestry(doc: Any) -> list[Issue]:
    """Validate an Input Ancestry envelope (RFC003 §30.10 D0 subset,
    extended by RFC003-ERRATA-003 E3's `model_candidate` binding branch).

    Schema pass first (closed vocabularies, per-field nullable shapes), then
    `input_ancestry_hash` recomputation, which also carries the
    `source_kind`/`derivation_kind`/candidate cross-field rules (see
    `identity.input_ancestry_hash`).
    """
    issues = _schema_issues("input_ancestry", doc) + _float_issues(doc)
    if issues:
        return issues

    authority = doc["authority_ref"]
    try:
        expected = identity.input_ancestry_hash(
            document_bundle_id=authority["document_bundle_id"],
            candidate_id=authority["candidate_id"],
            render_view_id=doc["render_view_id"],
            parent_task_view_id=doc["parent_task_view_id"],
            bindings=doc["bindings"],
        )
    except ValueError as exc:
        issues.append(_issue("INPUT_ANCESTRY_INVALID", "/bindings", str(exc)))
        return issues
    _check_id_integrity(doc["input_ancestry_hash"], expected, "/input_ancestry_hash", issues)
    return issues


def validate_reasoning_corrective_rules(
    task_view: dict[str, Any],
    ancestry: dict[str, Any],
    *,
    split_name: str,
    producer_touches_val_test: bool,
) -> list[Issue]:
    """RFC003-ERRATA-003 E3 build-time refusal rules for `reasoning_corrective_v0`.

    These are release/build-context facts no schema can see on its own
    (`split_name` — release membership per §40.6 lives outside the envelope;
    `producer_touches_val_test` — a checkpoint's training-lineage exposure,
    established by the release's lineage graph), so the caller supplies them
    explicitly rather than this function inferring them.
    """
    issues: list[Issue] = []
    if task_view.get("task_kind") != "reasoning_corrective_v0":
        return issues

    if split_name != "train":
        issues.append(
            _issue(
                "REASONING_CORRECTIVE_TRAIN_ONLY",
                "/task_kind",
                f"task_kind 'reasoning_corrective_v0' may only be materialized in the "
                f"'train' split, got {split_name!r} (RFC003-ERRATA-003 E3 rule 3)",
            )
        )

    if producer_touches_val_test:
        issues.append(
            _issue(
                "MODEL_CANDIDATE_SPLIT_EXPOSURE",
                "/bindings",
                "the frozen_model producer's checkpoint has non-zero exposure to "
                "this release's val/test lineages: fatal, not a warning "
                "(RFC003-ERRATA-003 E3 rule 4)",
            )
        )

    page_indices = task_view["input_selection"]["page_indices"]
    if len(page_indices) != 1:
        issues.append(
            _issue(
                "MODEL_CANDIDATE_MULTI_PAGE_FORBIDDEN",
                "/input_selection/page_indices",
                "reasoning_corrective_v0 concerns exactly one standalone page while "
                f"DEC-033 stands, got {len(page_indices)} pages "
                "(RFC003-ERRATA-003 E3 rule 8)",
            )
        )

    for index, binding in enumerate(ancestry["bindings"]):
        if binding["source_kind"] != "model_candidate":
            continue
        if binding.get("privacy_class") is not None:
            issues.append(
                _issue(
                    "MODEL_CANDIDATE_PRIVACY_NOT_INHERITED",
                    f"/bindings/{index}/privacy_class",
                    "a model_candidate binding must inherit privacy_class from the "
                    "authority; it may not declare its own (RFC003-ERRATA-003 E3 rule 7)",
                )
            )

    return issues


def validate_reasoning_corrective_rules_from_ancestry(
    task_view: dict[str, Any],
    ancestry: dict[str, Any],
    *,
    split_name: str,
    producer_ancestry: Mapping[str, Any] | None,
    release_name: str,
    val_test_ids: Iterable[str],
) -> list[Issue]:
    """Same E3 rules as `validate_reasoning_corrective_rules`, but computes
    `producer_touches_val_test` itself from the producer checkpoint's own
    ancestry instead of trusting a caller-supplied bool
    (`docs/plans/reasoning-data-readiness-2026-09-28.md` §2 row e/step 2).

    `producer_ancestry=None` is treated exactly like an ancestry document
    with no split-level exposure detail at all: `"unknown"`, which refuses
    (fail-closed — a missing ancestry is never evidence of a clean one).
    """
    from unflatten.dataset.reasoning.exposure import (
        compute_producer_split_exposure,
        producer_touches_val_test,
    )

    exposure = compute_producer_split_exposure(
        producer_ancestry or {}, release_name=release_name, val_test_ids=val_test_ids
    )
    return validate_reasoning_corrective_rules(
        task_view,
        ancestry,
        split_name=split_name,
        producer_touches_val_test=producer_touches_val_test(exposure),
    )

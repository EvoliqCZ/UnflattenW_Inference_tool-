"""RFC004 §35 Model Result Envelope, boundary-safe subset (W2-1).

Wraps today's `promotion.PromotionEnvelope` (`ast.status` restricted to
`{strict_valid, failed}`, `Literal` in `promotion.py`) into the public
`model-result.schema.json` shape, WITHOUT changing anything
`promotion.promote`/`PromotionEnvelope` return. `review_candidate` and
`partial` AST branches and the full §35.6 result-identity projection
(component-lock-derived `renderer_profile_hash`, resource-manifest
recomputation, etc.) are out of scope -- `promotion.py` itself has no
review_candidate/partial status to wrap, and the full identity projection
needs component records this module has no access to. What IS implemented:

- `build_model_result_envelope`: strict_valid/failed AST, and
  not_run/render_ready/approximation_required/failed renderer, each
  projected from `PromotionEnvelope` fields into closed `ArtifactRef`s.
- `result_id`: a documented PROVISIONAL A0 projection (mirrors
  `manifests.model_release_id`'s own "A0 provisional projection" precedent)
  over the envelope's own already-closed identity fields. This is NOT the
  full §35.6 `rfc001-hash-projection-v0`-anchored identity; a future A-phase
  must ratify the real projection before this id is treated as
  cross-implementation-stable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.model import promotion as pm
from unflatten.model.hashing import make_id

MODEL_RESULT_SCHEMA_VERSION = "0.1.0"
_RESULT_DOMAIN = "rfc004.model-result.v0-provisional-a0"


def _artifact_ref_from_bytes(data: bytes, *, ref: str, mime_type: str) -> dict[str, Any]:
    return {
        "ref": ref,
        "sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
        "size_bytes": len(data),
        "mime_type": mime_type,
    }


def _ast_branch(ast: pm.AstBranch) -> dict[str, Any]:
    canonical_ast_ref = None
    if ast.status == "strict_valid":
        assert ast.canonical_ast is not None
        canonical_ast_ref = _artifact_ref_from_bytes(
            to_canonical_json(ast.canonical_ast),
            ref="canonical_ast.json",
            mime_type="application/vnd.canonical-document-ast+json",
        )
    return {
        "status": ast.status,
        "canonical_ast": canonical_ast_ref,
        "ast_hash": ast.ast_hash,
        "ast_file_hash": ast.ast_file_hash,
        "review_candidate": None,
        "partial_hypothesis": None,
        "failure_report": ast.failure_report,
    }


def _renderer_branch(renderer: pm.RendererBranch) -> dict[str, Any]:
    docx_ref = None
    if renderer.status == "render_ready":
        assert renderer.docx_path is not None
        data = Path(renderer.docx_path).read_bytes()
        docx_ref = _artifact_ref_from_bytes(
            data,
            ref="package.docx",
            mime_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
        )
    return {
        "status": renderer.status,
        "reason": renderer.reason,
        "docx": docx_ref,
        "failure_report": renderer.failure_report,
    }


def build_model_result_envelope(
    promotion: pm.PromotionEnvelope,
    *,
    architecture_id: str,
    model_release_id: str,
    model_component_lock_hash: str,
    runtime_lock_hash: str,
    inference_config_hash: str,
    input_fingerprint: str,
    profile: str,
    inference_mode: str,
) -> dict[str, Any]:
    """Build a closed `model-result.schema.json` document from an already
    `.validate()`-clean `PromotionEnvelope`. Raises `ValueError` if the
    envelope itself is invalid -- this module never guesses at a broken
    input's intent."""
    issues = promotion.validate()
    if issues:
        raise ValueError(
            "promotion envelope invalid: "
            + "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        )

    document: dict[str, Any] = {
        "model_result_schema_version": MODEL_RESULT_SCHEMA_VERSION,
        "result_id": "",  # filled in below, once the rest of the doc exists
        "architecture_id": architecture_id,
        "model_release_id": model_release_id,
        "model_component_lock_hash": model_component_lock_hash,
        "runtime_lock_hash": runtime_lock_hash,
        "inference_config_hash": inference_config_hash,
        "input_fingerprint": input_fingerprint,
        "profile": profile,
        "inference_mode": inference_mode,
        "ast": _ast_branch(promotion.ast),
        "renderer": _renderer_branch(promotion.renderer),
        "acceptance": {
            "status": promotion.axes.acceptance_status,
            "reason_codes": [],
        },
        "confidence_report": None,
        "prediction_ir": None,
        "issues": [],
    }
    document["result_id"] = result_id(document)
    return document


def result_id(document: Mapping[str, Any]) -> str:
    """PROVISIONAL A0 §35.6 identity projection: every field of the
    envelope except `result_id` itself (excluding it prevents a hash
    cycle, same rule §35.3 states for `failure_report`)."""
    projection = {k: v for k, v in document.items() if k != "result_id"}
    return make_id("result", _RESULT_DOMAIN, projection)

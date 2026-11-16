"""Deterministic bytes and independent identities for diagnostic Prediction IR."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.model.prediction_ir import (
    prediction_ir_hash,
    validate_model_prediction_ir,
)


class PredictionIrArtifactInvalid(ValueError):
    """Prediction IR bytes, schema, semantic identity, or byte identity disagree."""


@dataclass(frozen=True, slots=True)
class PredictionIrArtifact:
    raw_bytes: bytes
    semantic_hash: str
    raw_sha256: str


def _raw_hash(raw_bytes: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}"


def build_prediction_ir_artifact(payload: Mapping[str, Any]) -> PredictionIrArtifact:
    """Validate first, then compute semantic-CBOR and raw-JCS hashes separately."""
    validate_model_prediction_ir(payload)
    raw_bytes = to_canonical_json(dict(payload))
    return PredictionIrArtifact(
        raw_bytes=raw_bytes,
        semantic_hash=prediction_ir_hash(payload),
        raw_sha256=_raw_hash(raw_bytes),
    )


def verify_prediction_ir_artifact(
    artifact: PredictionIrArtifact,
    *,
    raw_bytes: bytes | None = None,
    expected_raw_sha256: str | None = None,
) -> None:
    """Re-read and independently verify exact bytes and projected meaning."""
    actual_bytes = artifact.raw_bytes if raw_bytes is None else raw_bytes
    expected_raw = artifact.raw_sha256 if expected_raw_sha256 is None else expected_raw_sha256
    if _raw_hash(actual_bytes) != expected_raw:
        raise PredictionIrArtifactInvalid("raw Prediction IR byte hash mismatch")
    try:
        payload = json.loads(actual_bytes)
        validate_model_prediction_ir(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PredictionIrArtifactInvalid(f"Prediction IR payload invalid: {exc}") from exc
    if prediction_ir_hash(payload) != artifact.semantic_hash:
        raise PredictionIrArtifactInvalid("semantic Prediction IR hash mismatch")
    if to_canonical_json(payload) != actual_bytes:
        raise PredictionIrArtifactInvalid("raw Prediction IR bytes are not deterministic JCS")

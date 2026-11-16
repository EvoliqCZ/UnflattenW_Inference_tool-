from __future__ import annotations

import hashlib
import json

import pytest

from unflatten.model.node_lifecycle import NodeState
from unflatten.model.prediction_ir import RawDocumentHypothesis
from unflatten.service.model_runtime import seal_diagnostic_prediction
from unflatten.service.prediction_ir_artifact import (
    PredictionIrArtifactInvalid,
    build_prediction_ir_artifact,
    verify_prediction_ir_artifact,
)


def _payload():
    raw = RawDocumentHypothesis(
        accepted_tokens=(1, 2),
        node_states=(
            NodeState(
                local_node_id="n0",
                slot_kind="opened_node",
                node_type="document",
                parent_local_node_id=None,
            ),
        ),
        terminated=True,
    )
    return seal_diagnostic_prediction(raw)


def test_prediction_ir_artifact_has_deterministic_projection_and_independent_hashes():
    artifact_a = build_prediction_ir_artifact(_payload())
    artifact_b = build_prediction_ir_artifact(_payload())

    assert artifact_a == artifact_b
    assert json.loads(artifact_a.raw_bytes) == _payload()
    assert artifact_a.raw_sha256 == f"sha256:{hashlib.sha256(artifact_a.raw_bytes).hexdigest()}"
    assert artifact_a.semantic_hash != artifact_a.raw_sha256
    verify_prediction_ir_artifact(artifact_a)


def test_raw_and_semantic_hashes_verify_independently():
    artifact = build_prediction_ir_artifact(_payload())
    with pytest.raises(PredictionIrArtifactInvalid, match="raw"):
        verify_prediction_ir_artifact(artifact, raw_bytes=artifact.raw_bytes + b" ")

    changed = json.loads(artifact.raw_bytes)
    changed["document_hypotheses"][0]["fragments"][0]["accepted_tokens"].append(99)
    changed_bytes = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
    changed_raw_hash = f"sha256:{hashlib.sha256(changed_bytes).hexdigest()}"
    with pytest.raises(PredictionIrArtifactInvalid, match="semantic"):
        verify_prediction_ir_artifact(
            artifact, raw_bytes=changed_bytes, expected_raw_sha256=changed_raw_hash
        )

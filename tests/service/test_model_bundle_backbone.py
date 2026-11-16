"""DEC-050 §3: a model bundle CAN carry its own copy of the backbone in
`backbone/`. Absent -> today's behaviour byte-identical (`test_model_bundle.
py`'s existing coverage already proves that side); these tests cover the
new `add_backbone_to_bundle` / `verify_bundle` backbone path only.
"""

from __future__ import annotations

import json

import pytest
from tests.service.test_model_bundle import _build_bundle

from unflatten.model.candidate_loader import QWEN_CANDIDATE_MODEL_ID, QWEN_CANDIDATE_REVISION
from unflatten.model.qwen_tokenizer_artifacts import tokenizer_artifact_set_hash
from unflatten.service.model_bundle import (
    BACKBONE_DIRNAME,
    MANIFEST_FILENAME,
    ModelBundleInvalid,
    add_backbone_to_bundle,
    verify_bundle,
)


def _write_snapshot(snapshot_dir) -> None:
    snapshot_dir.mkdir(parents=True)
    for name, content in {
        "config.json": b"{}",
        "model.safetensors": b"weights",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
    }.items():
        (snapshot_dir / name).write_bytes(content)


class TestAddBackboneToBundle:
    def test_carries_the_backbone_and_verify_bundle_accepts_it(self, tmp_path):
        out_dir = _build_bundle(tmp_path)
        snapshot_dir = tmp_path / "snapshot"
        _write_snapshot(snapshot_dir)

        manifest_path = out_dir / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        manifest["identities"]["tokenizer_artifact_set_hash"] = tokenizer_artifact_set_hash(
            snapshot_dir, model_id=QWEN_CANDIDATE_MODEL_ID, revision=QWEN_CANDIDATE_REVISION
        )
        manifest_path.write_text(json.dumps(manifest))

        add_backbone_to_bundle(
            out_dir,
            snapshot_dir,
            model_id=QWEN_CANDIDATE_MODEL_ID,
            revision=QWEN_CANDIDATE_REVISION,
        )
        assert (out_dir / BACKBONE_DIRNAME / "model.safetensors").is_file()
        result = verify_bundle(out_dir)
        assert result["identities"]["backbone"]["source_model_id"] == QWEN_CANDIDATE_MODEL_ID

    def test_refuses_model_id_mismatch(self, tmp_path):
        out_dir = _build_bundle(tmp_path)
        snapshot_dir = tmp_path / "snapshot"
        _write_snapshot(snapshot_dir)
        with pytest.raises(ModelBundleInvalid, match="model_id/revision"):
            add_backbone_to_bundle(
                out_dir, snapshot_dir, model_id="someone/else", revision=QWEN_CANDIDATE_REVISION
            )

    def test_refuses_tokenizer_hash_mismatch(self, tmp_path):
        out_dir = _build_bundle(tmp_path)
        snapshot_dir = tmp_path / "snapshot"
        _write_snapshot(snapshot_dir)
        with pytest.raises(ModelBundleInvalid, match="tokenizer_artifact_set_hash"):
            add_backbone_to_bundle(
                out_dir,
                snapshot_dir,
                model_id=QWEN_CANDIDATE_MODEL_ID,
                revision=QWEN_CANDIDATE_REVISION,
            )

    def test_refuses_overwriting_an_existing_backbone_dir(self, tmp_path):
        out_dir = _build_bundle(tmp_path)
        (out_dir / BACKBONE_DIRNAME).mkdir()
        snapshot_dir = tmp_path / "snapshot"
        _write_snapshot(snapshot_dir)
        with pytest.raises(ModelBundleInvalid, match="already exists"):
            add_backbone_to_bundle(
                out_dir,
                snapshot_dir,
                model_id=QWEN_CANDIDATE_MODEL_ID,
                revision=QWEN_CANDIDATE_REVISION,
            )


class TestVerifyBundleBackbone:
    def _bundle_with_backbone(self, tmp_path):
        out_dir = _build_bundle(tmp_path)
        snapshot_dir = tmp_path / "snapshot"
        _write_snapshot(snapshot_dir)
        manifest_path = out_dir / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        manifest["identities"]["tokenizer_artifact_set_hash"] = tokenizer_artifact_set_hash(
            snapshot_dir, model_id=QWEN_CANDIDATE_MODEL_ID, revision=QWEN_CANDIDATE_REVISION
        )
        manifest_path.write_text(json.dumps(manifest))
        add_backbone_to_bundle(
            out_dir,
            snapshot_dir,
            model_id=QWEN_CANDIDATE_MODEL_ID,
            revision=QWEN_CANDIDATE_REVISION,
        )
        return out_dir

    def test_verify_bundle_never_touches_the_hf_cache_when_backbone_present(self, tmp_path):
        out_dir = self._bundle_with_backbone(tmp_path)
        # hf_cache_dir points nowhere -- if verify_bundle still consulted
        # the HF cache for base weights it would refuse here.
        verify_bundle(out_dir, hf_cache_dir=tmp_path / "nonexistent-cache")

    def test_verify_bundle_refuses_a_tampered_backbone_file(self, tmp_path):
        out_dir = self._bundle_with_backbone(tmp_path)
        (out_dir / BACKBONE_DIRNAME / "model.safetensors").write_bytes(b"tampered")
        with pytest.raises(ModelBundleInvalid, match="sha256 does not match"):
            verify_bundle(out_dir)

    def test_verify_bundle_refuses_a_missing_declared_backbone_file(self, tmp_path):
        out_dir = self._bundle_with_backbone(tmp_path)
        (out_dir / BACKBONE_DIRNAME / "model.safetensors").unlink()
        with pytest.raises(ModelBundleInvalid, match="missing file"):
            verify_bundle(out_dir)

    def test_verify_bundle_refuses_an_extra_undeclared_backbone_file(self, tmp_path):
        out_dir = self._bundle_with_backbone(tmp_path)
        (out_dir / BACKBONE_DIRNAME / "extra.bin").write_bytes(b"sneaky")
        with pytest.raises(ModelBundleInvalid, match="unaccounted-for"):
            verify_bundle(out_dir)

    def test_verify_bundle_refuses_backbone_identity_naming_a_different_model(self, tmp_path):
        out_dir = self._bundle_with_backbone(tmp_path)
        manifest_path = out_dir / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text())
        manifest["identities"]["backbone"]["source_model_id"] = "someone/else"
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ModelBundleInvalid, match="source_model_id"):
            verify_bundle(out_dir)

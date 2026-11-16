"""DEC-050 §3 for the native worker: a bundle that carries `backbone/` is
loaded from it (HF cache view), never from the host/container HF cache."""

from __future__ import annotations

from pathlib import Path

import huggingface_hub.constants as hf_constants
import pytest
from huggingface_hub import try_to_load_from_cache

from unflatten.service import worker_cli

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
_BACKBONE_IDENTITY = {"source_model_id": MODEL_ID, "source_revision": REVISION, "files": {}}


@pytest.fixture(autouse=True)
def _restore_hf_cache(monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", hf_constants.HF_HUB_CACHE)


def _bundle_with_backbone(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "backbone").mkdir(parents=True)
    (bundle / "backbone" / "tokenizer.json").write_text("{}", encoding="utf-8")
    return bundle


def test_bind_bundle_backbone_resolves_snapshot_to_bundle_backbone(tmp_path):
    bundle = _bundle_with_backbone(tmp_path)
    identities = {"backbone": _BACKBONE_IDENTITY}

    worker_cli.bind_bundle_backbone(bundle, identities, model_id=MODEL_ID, revision=REVISION)

    cached = try_to_load_from_cache(MODEL_ID, "tokenizer.json", revision=REVISION)
    assert isinstance(cached, str)
    assert Path(cached).read_text(encoding="utf-8") == "{}"
    assert Path(cached).resolve() == (bundle / "backbone" / "tokenizer.json").resolve()


def test_bind_bundle_backbone_without_carried_backbone_leaves_hf_cache_alone(tmp_path):
    before = hf_constants.HF_HUB_CACHE
    worker_cli.bind_bundle_backbone(tmp_path, {}, model_id=MODEL_ID, revision=REVISION)
    assert hf_constants.HF_HUB_CACHE == before


def test_bind_bundle_backbone_missing_directory_raises(tmp_path):
    identities = {"backbone": _BACKBONE_IDENTITY}
    with pytest.raises(FileNotFoundError, match="backbone"):
        worker_cli.bind_bundle_backbone(tmp_path, identities, model_id=MODEL_ID, revision=REVISION)

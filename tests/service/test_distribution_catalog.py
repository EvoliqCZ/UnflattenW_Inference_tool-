from pathlib import Path

from unflatten.service.model_variants import resolve_default_variant, resolve_variant_bundle
from unflatten.service.models_config import load_models_config

ROOT = Path(__file__).resolve().parents[2]


def test_distribution_catalog_download_binds_the_released_model(monkeypatch, tmp_path):
    config = load_models_config(ROOT / "deploy/distribution/models.json")
    selected, origin = resolve_default_variant(
        config,
        user_default=None,
        model_state={
            "total_vram_mib": 100_000,
            "compute_capability": [12, 1],
        },
    )
    assert selected == "dense-1.0-bf16" and origin == "auto"
    requested = {}

    def download(**kwargs):
        requested.update(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr("unflatten.service.model_variants._snapshot_download", download)
    path, pin = resolve_variant_bundle(config["variants"][0], cache_dir=tmp_path / "cache")
    assert requested == {
        "repo_id": "Evoliq/Unflatten-W-Dense-1.0",
        "revision": "9ebf7df1d6b01ae5d3a396f379fe454378a38ae7",
        "cache_dir": tmp_path / "cache",
    }
    assert path == tmp_path / "snapshot"
    assert pin == "sha256:0ee3a3c68e1290aeb7eaf72b9d28371c249f4d75f82851e9745ac221b0920c03"


def test_distribution_catalog_small_gpu_does_not_substitute_a_model():
    config = load_models_config(ROOT / "deploy/distribution/models.json")
    selected, _ = resolve_default_variant(
        config,
        user_default=None,
        model_state={
            "total_vram_mib": 8_000,
            "compute_capability": [8, 9],
        },
    )
    assert selected is None

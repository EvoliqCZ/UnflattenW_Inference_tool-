from __future__ import annotations

import json
from pathlib import Path

import pytest

from unflatten.service.models_config import (
    ModelsConfigInvalid,
    load_models_config,
    validate_models_config,
)

_SHA = "sha256:" + "a" * 64
_REVISION = "b" * 40


def _config(**overrides: object) -> dict:
    base = {
        "schema_version": "1.0",
        "default_variant": "fp8",
        "variants": [
            {
                "variant_id": "fp8",
                "label": "FP8 (fast)",
                "precision": "fp8",
                "kv_cache": "fp8",
                "bundle": "/data/model-bundles/v15b-fp8-fp8",
            },
            {
                "variant_id": "remote-int4",
                "label": "INT4 (remote)",
                "precision": "int4",
                "kv_cache": "bf16",
                "source": {
                    "kind": "hf",
                    "repo_id": "unflatten/v15b",
                    "revision": _REVISION,
                    "bundle_manifest_sha256": _SHA,
                },
            },
        ],
    }
    base.update(overrides)
    return base


def test_validate_models_config_accepts_well_formed_config() -> None:
    result = validate_models_config(_config())
    assert result["default_variant"] == "fp8"
    assert len(result["variants"]) == 2


def test_validate_models_config_rejects_unknown_default_variant() -> None:
    with pytest.raises(ModelsConfigInvalid):
        validate_models_config(_config(default_variant="does-not-exist"))


def test_validate_models_config_rejects_duplicate_variant_ids() -> None:
    config = _config()
    config["variants"][1]["variant_id"] = "fp8"
    with pytest.raises(ModelsConfigInvalid):
        validate_models_config(config)


def test_validate_models_config_rejects_branch_name_revision() -> None:
    config = _config()
    config["variants"][1]["source"]["revision"] = "main"
    with pytest.raises(ModelsConfigInvalid, match="revision"):
        validate_models_config(config)


def test_validate_models_config_rejects_malformed_bundle_manifest_sha256() -> None:
    config = _config()
    config["variants"][1]["source"]["bundle_manifest_sha256"] = "not-a-hash"
    with pytest.raises(ModelsConfigInvalid):
        validate_models_config(config)


def test_validate_models_config_rejects_both_bundle_and_source() -> None:
    config = _config()
    config["variants"][0]["source"] = config["variants"][1]["source"]
    with pytest.raises(ModelsConfigInvalid, match="exactly one"):
        validate_models_config(config)


def test_validate_models_config_rejects_neither_bundle_nor_source() -> None:
    config = _config()
    del config["variants"][0]["bundle"]
    with pytest.raises(ModelsConfigInvalid, match="exactly one"):
        validate_models_config(config)


def test_validate_models_config_rejects_unknown_precision() -> None:
    config = _config()
    config["variants"][0]["precision"] = "fp32"
    with pytest.raises(ModelsConfigInvalid):
        validate_models_config(config)


def test_validate_models_config_rejects_unknown_top_level_field() -> None:
    config = _config()
    config["extra"] = True
    with pytest.raises(ModelsConfigInvalid):
        validate_models_config(config)


def test_validate_models_config_rejects_empty_variants() -> None:
    with pytest.raises(ModelsConfigInvalid):
        validate_models_config(_config(variants=[]))


def test_validate_models_config_rejects_wrong_schema_version() -> None:
    with pytest.raises(ModelsConfigInvalid):
        validate_models_config(_config(schema_version="2.0"))


def test_load_models_config_reads_and_validates_json_file(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")

    result = load_models_config(path)

    assert result["default_variant"] == "fp8"


def test_load_models_config_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(ModelsConfigInvalid):
        load_models_config(path)


def test_load_models_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ModelsConfigInvalid):
        load_models_config(tmp_path / "does-not-exist.json")


def test_default_variant_auto_accepted() -> None:
    result = validate_models_config(_config(default_variant="auto"))
    assert result["default_variant"] == "auto"


def test_variant_id_auto_rejected() -> None:
    config = _config()
    config["variants"][0]["variant_id"] = "auto"
    config["default_variant"] = "auto"
    with pytest.raises(ModelsConfigInvalid, match="auto"):
        validate_models_config(config)


def test_engine_defaults_native() -> None:
    result = validate_models_config(_config())
    assert all(variant["engine"] == "native" for variant in result["variants"])


def test_engine_unknown_rejected() -> None:
    config = _config()
    config["variants"][0]["engine"] = "tensorrt"
    with pytest.raises(ModelsConfigInvalid, match="engine"):
        validate_models_config(config)


# Owner request 2026-09-29: catalogue variants that are not downloadable yet.
_UNPUBLISHED = {
    "variant_id": "int8-hf",
    "label": "INT8",
    "precision": "int8",
    "kv_cache": "bf16",
    "availability": "unpublished",
}
_PLANNED = {"variant_id": "moe", "label": "Unflatten W MoE", "availability": "planned"}


def test_validate_models_config_availability_omitted_defaults_to_published() -> None:
    result = validate_models_config(_config())
    assert all(variant["availability"] == "published" for variant in result["variants"])


def test_validate_models_config_explicit_published_accepted() -> None:
    config = _config()
    config["variants"][0]["availability"] = "published"
    result = validate_models_config(config)
    assert result["variants"][0]["availability"] == "published"


def test_validate_models_config_unpublished_without_origin_accepted() -> None:
    config = _config()
    config["variants"].append(dict(_UNPUBLISHED))

    result = validate_models_config(config)

    assert result["variants"][2] == {
        "variant_id": "int8-hf",
        "label": "INT8",
        "precision": "int8",
        "kv_cache": "bf16",
        "engine": "native",
        "availability": "unpublished",
    }


@pytest.mark.parametrize("origin", ["bundle", "source"])
def test_validate_models_config_unpublished_with_origin_rejected(origin: str) -> None:
    config = _config()
    unpublished = dict(_UNPUBLISHED)
    unpublished[origin] = config["variants"][0 if origin == "bundle" else 1][origin]
    config["variants"].append(unpublished)
    with pytest.raises(ModelsConfigInvalid, match="unpublished"):
        validate_models_config(config)


def test_validate_models_config_unpublished_without_precision_rejected() -> None:
    config = _config()
    unpublished = dict(_UNPUBLISHED)
    del unpublished["precision"]
    config["variants"].append(unpublished)
    with pytest.raises(ModelsConfigInvalid):
        validate_models_config(config)


def test_validate_models_config_planned_without_precision_or_origin_accepted() -> None:
    config = _config()
    config["variants"].append(dict(_PLANNED))

    result = validate_models_config(config)

    assert result["variants"][2] == {
        "variant_id": "moe",
        "label": "Unflatten W MoE",
        "precision": None,
        "kv_cache": None,
        "engine": "native",
        "availability": "planned",
    }


@pytest.mark.parametrize(
    "extra",
    [
        {"precision": "bf16", "kv_cache": "bf16"},
        {"bundle": "/data/model-bundles/moe"},
    ],
)
def test_validate_models_config_planned_with_precision_or_origin_rejected(extra: dict) -> None:
    config = _config()
    config["variants"].append({**_PLANNED, **extra})
    with pytest.raises(ModelsConfigInvalid, match="planned"):
        validate_models_config(config)


def test_validate_models_config_unknown_availability_rejected() -> None:
    config = _config()
    config["variants"][0]["availability"] = "beta"
    with pytest.raises(ModelsConfigInvalid, match="availability"):
        validate_models_config(config)


@pytest.mark.parametrize("variant", [_UNPUBLISHED, _PLANNED])
def test_validate_models_config_default_names_non_published_rejected(variant: dict) -> None:
    config = _config(default_variant=variant["variant_id"])
    config["variants"].append(dict(variant))
    with pytest.raises(ModelsConfigInvalid, match="published"):
        validate_models_config(config)


def test_validate_models_config_normalized_output_revalidates_unchanged() -> None:
    config = _config()
    config["variants"] += [dict(_UNPUBLISHED), dict(_PLANNED)]
    normalized = validate_models_config(config)
    assert validate_models_config(normalized) == normalized

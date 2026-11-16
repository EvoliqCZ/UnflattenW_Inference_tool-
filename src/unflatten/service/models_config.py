"""DEC-044 Dodatok D §1: the operator-supplied model variants config.

`UNFLATTEN_MODELS_CONFIG` / `--models-config` names a JSON file listing the
model variants the API and worker may serve. Each variant is either a local
bundle directory (already on disk) or a pinned Hugging Face source (an
immutable 40-hex commit SHA, never a branch name, plus the expected bundle
manifest SHA-256 the worker refuses to load a mismatch against). Absent the
flag, the service keeps today's single-bundle behaviour untouched -- this
module is never invoked in that path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unflatten.model.quantization.spec import KV_CACHE_DTYPES, WEIGHT_PRECISIONS
from unflatten.service.contracts import VARIANT_AVAILABILITIES

SCHEMA_VERSION = "1.0"

_TOP_LEVEL_FIELDS = frozenset({"schema_version", "default_variant", "variants"})
_VARIANT_REQUIRED_FIELDS = frozenset({"variant_id", "label", "precision", "kv_cache"})
_VARIANT_ORIGIN_FIELDS = frozenset({"bundle", "source"})
_VARIANT_OPTIONAL_FIELDS = frozenset({"engine", "availability"})
#: Owner request 2026-09-29: a catalogue entry may be listed before it can be
#: downloaded. `published` (the default, so every existing file keeps its
#: meaning) carries exactly one origin; `unpublished` is a known precision
#: variant whose HF repo is not public yet (DEC-050: publishing is the
#: owner's action) and carries no origin at all; `planned` is a future model
#: (Unflatten W MoE) with neither an origin nor a precision -- nothing about
#: it has been measured. Only `published` variants can ever be downloaded,
#: loaded, selected or resolved as a default.
PUBLISHED = "published"
_SOURCE_REQUIRED_FIELDS = frozenset({"kind", "repo_id", "revision", "bundle_manifest_sha256"})
_SOURCE_OPTIONAL_FIELDS = frozenset({"subfolder"})
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ENGINES = frozenset({"native"})
_DEFAULT_ENGINE = "native"
_AUTO_DEFAULT_VARIANT = "auto"


class ModelsConfigInvalid(ValueError):
    """The models variants config is malformed or violates a closed constraint."""


def load_models_config(path: Path | str) -> dict[str, Any]:
    """Read and validate the models config JSON file at `path`."""
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelsConfigInvalid(f"cannot read models config {config_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ModelsConfigInvalid(f"models config {config_path} is not valid JSON: {exc}") from exc
    return validate_models_config(payload)


def validate_models_config(config: object) -> dict[str, Any]:
    """Validate a parsed models config mapping and return a normalized copy."""
    if not isinstance(config, dict) or set(config) != _TOP_LEVEL_FIELDS:
        raise ModelsConfigInvalid(
            f"models config must be a closed object with exactly {sorted(_TOP_LEVEL_FIELDS)}"
        )
    if config["schema_version"] != SCHEMA_VERSION:
        raise ModelsConfigInvalid(
            f"models config schema_version must be {SCHEMA_VERSION!r}, "
            f"got {config['schema_version']!r}"
        )
    variants = config["variants"]
    if not isinstance(variants, list) or not variants:
        raise ModelsConfigInvalid("models config variants must be a non-empty array")
    normalized_variants = [_validate_variant(variant) for variant in variants]
    ids = [variant["variant_id"] for variant in normalized_variants]
    if len(ids) != len(set(ids)):
        raise ModelsConfigInvalid("models config variant_id values must be unique")
    default_variant = config["default_variant"]
    published_ids = {
        variant["variant_id"] for variant in normalized_variants if is_published(variant)
    }
    if not isinstance(default_variant, str) or (
        default_variant != _AUTO_DEFAULT_VARIANT and default_variant not in published_ids
    ):
        raise ModelsConfigInvalid(
            f"models config default_variant {default_variant!r} names no known published "
            f"variant and is not {_AUTO_DEFAULT_VARIANT!r}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "default_variant": default_variant,
        "variants": normalized_variants,
    }


def is_published(variant: Mapping[str, Any]) -> bool:
    """Whether a (normalized) variant can be downloaded, loaded or selected.
    A mapping without `availability` is published (pre-2026-09-29 shape)."""
    return variant.get("availability", PUBLISHED) == PUBLISHED


def _validate_variant(variant: object) -> dict[str, Any]:
    if not isinstance(variant, dict):
        raise ModelsConfigInvalid("each models config variant must be an object")
    availability = variant.get("availability", PUBLISHED)
    if availability not in VARIANT_AVAILABILITIES:
        raise ModelsConfigInvalid(
            f"models config availability must be one of {sorted(VARIANT_AVAILABILITIES)}, "
            f"got {availability!r}"
        )
    origins = _VARIANT_ORIGIN_FIELDS & set(variant)
    if availability == PUBLISHED and len(origins) != 1:
        raise ModelsConfigInvalid(
            "each models config variant must carry exactly one of 'bundle' or 'source'"
        )
    if availability != PUBLISHED and origins:
        raise ModelsConfigInvalid(
            f"models config variant with availability {availability!r} has nothing to "
            "download and must carry neither 'bundle' nor 'source'"
        )
    if availability == "planned":
        return _validate_planned_variant(variant)
    optional = _VARIANT_OPTIONAL_FIELDS & set(variant)
    allowed = _VARIANT_REQUIRED_FIELDS | origins | optional
    if set(variant) != allowed:
        raise ModelsConfigInvalid(f"models config variant has unexpected fields: {set(variant)}")
    variant_id = variant["variant_id"]
    label = variant["label"]
    precision = variant["precision"]
    kv_cache = variant["kv_cache"]
    _validate_identity(variant_id, label)
    if precision not in WEIGHT_PRECISIONS:
        raise ModelsConfigInvalid(
            f"models config precision must be one of {sorted(WEIGHT_PRECISIONS)}, got {precision!r}"
        )
    if kv_cache not in KV_CACHE_DTYPES:
        raise ModelsConfigInvalid(
            f"models config kv_cache must be one of {sorted(KV_CACHE_DTYPES)}, got {kv_cache!r}"
        )
    normalized: dict[str, Any] = {
        "variant_id": variant_id,
        "label": label,
        "precision": precision,
        "kv_cache": kv_cache,
        "engine": _validate_engine(variant),
        "availability": availability,
    }
    if "bundle" in variant:
        bundle = variant["bundle"]
        if not isinstance(bundle, str) or not bundle:
            raise ModelsConfigInvalid("models config bundle must be a non-empty string path")
        normalized["bundle"] = bundle
    elif "source" in variant:
        normalized["source"] = _validate_source(variant["source"])
    return normalized


def _validate_planned_variant(variant: dict[str, Any]) -> dict[str, Any]:
    """A `planned` variant names a model that does not exist yet: it may
    carry `precision`/`kv_cache` only as `null` (the normalized form), since
    no precision of it has been built or measured."""
    allowed = {"variant_id", "label", "availability", "engine", "precision", "kv_cache"}
    if not {"variant_id", "label"} <= set(variant) <= allowed:
        raise ModelsConfigInvalid(
            f"models config planned variant has unexpected fields: {set(variant)}"
        )
    if variant.get("precision") is not None or variant.get("kv_cache") is not None:
        raise ModelsConfigInvalid(
            "models config variant with availability 'planned' has no measured precision; "
            "'precision' and 'kv_cache' must be omitted"
        )
    _validate_identity(variant["variant_id"], variant["label"])
    return {
        "variant_id": variant["variant_id"],
        "label": variant["label"],
        "precision": None,
        "kv_cache": None,
        "engine": _validate_engine(variant),
        "availability": "planned",
    }


def _validate_identity(variant_id: object, label: object) -> None:
    if (
        not isinstance(variant_id, str)
        or not variant_id
        or variant_id == _AUTO_DEFAULT_VARIANT
    ):
        raise ModelsConfigInvalid(
            f"models config variant_id must be a non-empty string other than "
            f"{_AUTO_DEFAULT_VARIANT!r}"
        )
    if not isinstance(label, str) or not label:
        raise ModelsConfigInvalid("models config label must be a non-empty string")


def _validate_engine(variant: Mapping[str, Any]) -> str:
    engine = variant.get("engine", _DEFAULT_ENGINE)
    if engine not in _ENGINES:
        raise ModelsConfigInvalid(
            f"models config engine must be one of {sorted(_ENGINES)}, got {engine!r}"
        )
    return engine


def _validate_source(source: object) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ModelsConfigInvalid("models config source must be an object")
    fields = set(source)
    if not (_SOURCE_REQUIRED_FIELDS <= fields <= _SOURCE_REQUIRED_FIELDS | _SOURCE_OPTIONAL_FIELDS):
        raise ModelsConfigInvalid(
            f"models config source must carry exactly {sorted(_SOURCE_REQUIRED_FIELDS)} "
            f"plus optionally {sorted(_SOURCE_OPTIONAL_FIELDS)}"
        )
    if source["kind"] != "hf":
        raise ModelsConfigInvalid("models config source.kind must be 'hf'")
    repo_id = source["repo_id"]
    if not isinstance(repo_id, str) or not repo_id:
        raise ModelsConfigInvalid("models config source.repo_id must be a non-empty string")
    revision = source["revision"]
    if not isinstance(revision, str) or _REVISION_PATTERN.match(revision) is None:
        raise ModelsConfigInvalid(
            "models config source.revision must be a pinned 40-hex commit SHA, never a "
            f"branch name; got {revision!r}"
        )
    bundle_manifest_sha256 = source["bundle_manifest_sha256"]
    if not isinstance(bundle_manifest_sha256, str) or _SHA256_PATTERN.match(
        bundle_manifest_sha256
    ) is None:
        raise ModelsConfigInvalid(
            "models config source.bundle_manifest_sha256 must be 'sha256:<64 hex>'"
        )
    normalized: dict[str, Any] = {
        "kind": "hf",
        "repo_id": repo_id,
        "revision": revision,
        "bundle_manifest_sha256": bundle_manifest_sha256,
    }
    if "subfolder" in source:
        subfolder = source["subfolder"]
        if not isinstance(subfolder, str) or not subfolder:
            raise ModelsConfigInvalid("models config source.subfolder must be a non-empty string")
        normalized["subfolder"] = subfolder
    return normalized

"""Strict local safetensors bundles; integrity is not quality promotion."""

import json
import shutil
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from .config import DecisionHeadConfig
from .head import DecisionHead

_SCHEMA = "unflatten-decision-bundle.v1"
_DTYPES = {
    str(dtype): dtype for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64)
}


@dataclass(frozen=True)
class FeatureContract:
    visual_backbone: str
    visual_tap: str
    visual_pooling: str
    text_embedding: str

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"decision feature contract: {field.name} must be nonempty text")


def _json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _hash(path: Path) -> str:
    h = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"decision bundle: cannot read JSON at {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"decision bundle: expected JSON object at {path}")
    return value


def _validate_state(state: dict[str, torch.Tensor]) -> str:
    dtypes = {str(t.dtype) for t in state.values()}
    if len(dtypes) != 1 or not dtypes <= _DTYPES.keys():
        raise ValueError("decision bundle: state must have one supported floating storage dtype")
    if any(not torch.isfinite(t).all() for t in state.values()):
        raise ValueError("decision bundle: weights or buffers contain nonfinite values")
    if "network.sd" in state and (state["network.sd"] <= 0).any():
        raise ValueError("decision bundle: normalizer sd must be positive")
    return next(iter(dtypes))


def export_bundle(
    head: DecisionHead, destination: Path, *, feature_contract: FeatureContract
) -> None:
    """Write a new local experimental bundle; never overwrite an existing path."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"decision bundle: destination already exists: {destination}")
    if not isinstance(feature_contract, FeatureContract):
        raise ValueError("decision bundle: feature_contract must be FeatureContract")
    state = {key: value.detach().cpu().contiguous() for key, value in head.state_dict().items()}
    dtype = _validate_state(state)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        (destination / "config.json").write_text(_json(asdict(head.config)), encoding="utf-8")
        save_file(state, destination / "model.safetensors")
        manifest = {
            "schema": _SCHEMA,
            "checkpoint_status": "experimental",
            "head_config": asdict(head.config),
            "weight_dtype": dtype,
            "feature_contract": asdict(feature_contract),
            "config_sha256": _hash(destination / "config.json"),
            "weights_sha256": _hash(destination / "model.safetensors"),
        }
        # Written last: its presence marks a complete bundle.
        (destination / "manifest.json").write_text(_json(manifest), encoding="utf-8")
    except (OSError, ValueError, SafetensorError):
        shutil.rmtree(destination)
        raise


def load_bundle(
    path: Path, *, expected_feature_contract: FeatureContract, device: str = "cpu"
) -> DecisionHead:
    """Load only a strict local bundle matching the caller's feature semantics.

    Hashes detect corruption, not hostile replacement/authenticity. The caller
    remains responsible for actually producing the declared feature contract.
    No remote code/downloads or quality promotion occurs.
    """
    path = Path(path)
    if not isinstance(expected_feature_contract, FeatureContract):
        raise ValueError("decision bundle: expected_feature_contract must be FeatureContract")
    manifest = _read_json(path / "manifest.json")
    required = {
        "schema",
        "checkpoint_status",
        "head_config",
        "weight_dtype",
        "feature_contract",
        "config_sha256",
        "weights_sha256",
    }
    if set(manifest) != required or manifest["schema"] != _SCHEMA:
        raise ValueError("decision bundle: unsupported manifest schema or fields")
    if manifest["checkpoint_status"] != "experimental":
        raise ValueError("decision bundle: unsupported checkpoint status; no quality promotion")
    if manifest["feature_contract"] != asdict(expected_feature_contract):
        raise ValueError("decision bundle: feature contract mismatch")
    try:
        if _hash(path / "config.json") != manifest["config_sha256"]:
            raise ValueError("decision bundle: config SHA256 mismatch")
        if _hash(path / "model.safetensors") != manifest["weights_sha256"]:
            raise ValueError("decision bundle: weights SHA256 mismatch")
    except OSError as exc:
        raise ValueError(f"decision bundle: cannot read files at {path}") from exc
    config_data = _read_json(path / "config.json")
    if set(config_data) != {field.name for field in fields(DecisionHeadConfig)}:
        raise ValueError("decision bundle: unsupported config fields")
    if config_data != manifest["head_config"]:
        raise ValueError("decision bundle: manifest and config disagree")
    try:
        config = DecisionHeadConfig(**config_data)
        state = load_file(path / "model.safetensors", device="cpu")
        dtype = _validate_state(state)
        if dtype != manifest["weight_dtype"]:
            raise ValueError("weights and manifest storage dtype disagree")
        head = DecisionHead(config).to(dtype=_DTYPES[dtype])
        head.load_state_dict(state, strict=True)
    except (ValueError, RuntimeError, SafetensorError) as exc:
        raise ValueError(f"decision bundle: invalid config or weights at {path}: {exc}") from exc
    return head.to(device=device).eval()

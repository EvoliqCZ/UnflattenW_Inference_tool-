"""Opt-in, inference-only Dense-to-MoE overlays; never a production default.

The full verified Dense model remains the immutable base. An overlay binds its
entire state and checkpoint identity and stores only the added adapter weights.
It authorizes no training, promotion, optimizer restoration or product claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from unflatten.model.candidate_architecture_variant import ExpertAdapterVariant
from unflatten.model.candidate_loader import QwenCandidate
from unflatten.model.candidate_model import QwenDenseCandidateModel
from unflatten.model.hashing import domain_hash
from unflatten.model.quantization.quantize import decoder_module, vision_module
from unflatten.model.sparse_expert_adapter.adapter import SparseExpertAdapter
from unflatten.model.sparse_expert_adapter.manifest import ExpertRole, SparseExpertAdapterManifest

_DOMAIN = "unflatten.moe-step0-conversion/v1"
_WEIGHTS = "adapter.safetensors"
_DOCUMENT = "conversion.json"
_FIELDS = {
    "schema_version",
    "kind",
    "source_identity",
    "source_model_state_sha256",
    "topology",
    "seed",
    "initialization",
    "weights_file",
    "weights_sha256",
    "training_ready",
    "artifact_hash",
}


class Step0ConversionError(ValueError):
    """An overlay cannot establish exact, immutable-source step-zero conversion."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def tensor_state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    """Bind names, shapes, dtypes and exact bytes, including frozen model state."""
    digest = hashlib.sha256(b"unflatten.moe-step0-source-model/v1\0")
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise Step0ConversionError(f"source model state {name!r} is not a tensor")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":"))
        digest.update(header.encode() + b"\0")
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return "sha256:" + digest.hexdigest()


def dense_base_fingerprint(model: QwenDenseCandidateModel, *, candidate: QwenCandidate) -> str:
    """Include frozen nonpersistent references and behavior-changing Dense settings."""
    if model._decode is not candidate.decode or model._inject_visual is not candidate.inject_visual:
        raise Step0ConversionError("source model is not bound to the supplied frozen candidate")
    excluded = ("sparse_adapter.", "sparse_reasoning_adapter.")
    state = {k: v for k, v in model.state_dict().items() if not k.startswith(excluded)}
    for name, buffer in model.named_buffers():
        if not name.startswith(excluded):
            state["@buffer:" + name] = buffer
    for name, parameter in candidate.named_parameters.items():
        state["@backbone:" + name] = parameter
    backbone_config = {}
    for label, module in (
        ("decoder", decoder_module(candidate)),
        ("vision", vision_module(candidate)),
    ):
        for name, parameter in module.named_parameters():
            state[f"@backbone-live:{label}:{name}"] = parameter
        for name, buffer in module.named_buffers():
            state[f"@backbone-buffer:{label}:{name}"] = buffer
        config = getattr(module, "config", None)
        backbone_config[label] = config.to_dict() if config is not None else {}
        # Paths and package versions are packaging facts, not neural configuration.
        for key in ("_name_or_path", "transformers_version"):
            backbone_config[label].pop(key, None)
    config = {
        "deepstack": model.deepstack,
        "image_token_id": model.image_token_id,
        "ranges": dataclasses.asdict(model.ranges),
        "variant": dataclasses.asdict(
            dataclasses.replace(model.architecture_variant, expert_adapter=ExpertAdapterVariant())
        ),
        "typed_integer_config": dataclasses.asdict(model.typed_integer_config),
        "explicit_zero_bin_pointers": sorted(model.explicit_zero_bin_pointers),
        "backbone_config": backbone_config,
        "candidate": {
            key: getattr(candidate, key)
            for key in (
                "hidden_size",
                "visual_hidden_size",
                "vocab_size",
                "attn_implementation",
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
                "deepstack_visual_indexes",
                "spatial_merge_size",
            )
        },
    }
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    state["@config"] = torch.tensor(list(config_bytes), dtype=torch.uint8)
    return tensor_state_fingerprint(state)


def _source_identity(value: Mapping[str, Any]) -> dict[str, str]:
    if set(value) != {"checkpoint_id", "checkpoint_sha256", "visual_feature_tap"}:
        raise Step0ConversionError("conversion source identity has missing or extra fields")
    if not all(isinstance(v, str) and v for v in value.values()):
        raise Step0ConversionError("conversion source identity values must be nonempty strings")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value["checkpoint_sha256"]):
        raise Step0ConversionError("conversion source identity checkpoint SHA-256 is invalid")
    return dict(value)


def _require_dense(model: QwenDenseCandidateModel) -> None:
    if (
        model.architecture_variant.expert_adapter.mode != "dense"
        or model.sparse_adapter is not None
    ):
        raise Step0ConversionError("step-zero conversion requires an unconverted Dense model")
    if model.architecture_variant.reasoning.enabled or (
        model.architecture_variant.post_encoder_compressor.profile != "bypass"
        or model.architecture_variant.pre_encoder_compressor.profile != "bypass"
    ):
        raise Step0ConversionError("step-zero conversion requires reasoning and compression off")


def _require_visual_source(
    model: QwenDenseCandidateModel, candidate: QwenCandidate, identity: Mapping[str, str]
) -> None:
    expected = "pooler_output"
    if model.deepstack:
        expected += "+deepstack:" + ",".join(map(str, candidate.deepstack_visual_indexes))
    if identity["visual_feature_tap"] != expected:
        raise Step0ConversionError("conversion source model visual feature tap mismatch")


def _topology_payload(manifest: SparseExpertAdapterManifest) -> dict[str, Any]:
    payload = dataclasses.asdict(manifest)
    # This conversion permits exactly these constants. The project's canonical
    # CBOR identity profile admits integers, never binary floating-point values.
    payload["dropout"] = 0
    payload["alpha_shared"] = 1
    payload["specialist_roles"] = [dataclasses.asdict(role) for role in manifest.specialist_roles]
    payload["eligible_logical_roles_by_context"] = {
        k: sorted(v) for k, v in manifest.eligible_logical_roles_by_context.items()
    }
    return payload


def _read_topology(payload: Mapping[str, Any]) -> SparseExpertAdapterManifest:
    try:
        fields = dict(payload)
        fields["specialist_roles"] = tuple(ExpertRole(**row) for row in fields["specialist_roles"])
        fields["eligible_logical_roles_by_context"] = {
            k: frozenset(v) for k, v in fields["eligible_logical_roles_by_context"].items()
        }
        return SparseExpertAdapterManifest(**fields)
    except (KeyError, TypeError, AttributeError) as exc:
        raise Step0ConversionError(f"conversion topology is malformed: {exc}") from exc


def _build_adapter(
    model: QwenDenseCandidateModel, manifest: SparseExpertAdapterManifest, seed: int
) -> SparseExpertAdapter:
    dense = model.dense_adapter
    if (
        manifest.d_model != dense.d_model
        or manifest.layers != len(dense.layers)
        or manifest.hidden_multiplier != dense.layers[0].up.out_features // dense.d_model
        or manifest.alpha_shared != 1.0
        or manifest.dropout != 0.0
        or manifest.insertion_point != "adapter_stack"
        or manifest.eligible_logical_roles_by_context
    ):
        raise Step0ConversionError("conversion topology must match the source Dense adapter slot")
    if type(seed) is not int or seed < 0:
        raise Step0ConversionError("conversion seed must be a nonnegative integer")
    reference = next(dense.parameters())
    with torch.random.fork_rng(devices=[]):
        # Seed CPU construction only; do not reset another CUDA user's RNG.
        torch.random.default_generator.manual_seed(seed)
        adapter = SparseExpertAdapter(manifest).to(device=reference.device, dtype=reference.dtype)
    adapter.shared_expert.blocks.load_state_dict(dense.layers.state_dict(), strict=True)
    for specialist in adapter.specialists:
        specialist.blocks.load_state_dict(dense.layers.state_dict(), strict=True)
        specialist.zero_output_()
    adapter.requires_grad_(False).eval()
    return adapter


def _validate_weights(
    model: QwenDenseCandidateModel,
    adapter: SparseExpertAdapter,
    weights: Mapping[str, torch.Tensor],
) -> None:
    expected = adapter.state_dict()
    if set(weights) != set(expected):
        raise Step0ConversionError("conversion expert weights have missing or extra keys")
    for name, value in weights.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise Step0ConversionError(f"conversion expert weight shape/dtype mismatch: {name}")
        if not bool(torch.isfinite(value).all()):
            raise Step0ConversionError(f"conversion expert weight is nonfinite: {name}")
    for name, value in model.dense_adapter.layers.state_dict().items():
        shared = weights[f"shared_expert.blocks.{name}"]
        if not torch.equal(shared.cpu(), value.detach().cpu()):
            raise Step0ConversionError(f"conversion shared expert differs from source: {name}")
        for index in range(len(adapter.specialists)):
            specialist = weights[f"specialists.{index}.blocks.{name}"]
            if ".down." in name:
                if bool(torch.count_nonzero(specialist)):
                    raise Step0ConversionError(
                        "conversion specialist output must be zero at step 0"
                    )
            elif not torch.equal(specialist.cpu(), value.detach().cpu()):
                raise Step0ConversionError(f"conversion specialist differs from source: {name}")


def save_step0_conversion(
    model: QwenDenseCandidateModel,
    output_dir: Path,
    *,
    source_identity: Mapping[str, str],
    candidate: QwenCandidate,
    manifest: SparseExpertAdapterManifest,
    seed: int,
) -> dict[str, Any]:
    """Write a new overlay without changing the Dense model or existing files."""
    _require_dense(model)
    identity = _source_identity(source_identity)
    _require_visual_source(model, candidate, identity)
    adapter = _build_adapter(model, manifest, seed)
    source_fingerprint = dense_base_fingerprint(model, candidate=candidate)
    weights = {k: v.detach().cpu().contiguous() for k, v in adapter.state_dict().items()}
    _validate_weights(model, adapter, weights)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    save_file(weights, str(output_dir / _WEIGHTS))
    document = {
        "schema_version": "1.0.0",
        "kind": "unflatten-moe-step0-overlay",
        "source_identity": identity,
        "source_model_state_sha256": source_fingerprint,
        "topology": _topology_payload(manifest),
        "seed": seed,
        "initialization": "copy-dense-internals-zero-specialist-output.v1",
        "weights_file": _WEIGHTS,
        "weights_sha256": file_sha256(output_dir / _WEIGHTS),
        "training_ready": False,
    }
    document["artifact_hash"] = domain_hash(_DOMAIN, document)
    (output_dir / _DOCUMENT).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def load_step0_conversion(
    model: QwenDenseCandidateModel,
    output_dir: Path,
    *,
    source_identity: Mapping[str, str],
    candidate: QwenCandidate,
) -> dict[str, Any]:
    """Validate all content before activating a frozen sparse adapter in this model."""
    _require_dense(model)
    identity = _source_identity(source_identity)
    _require_visual_source(model, candidate, identity)
    output_dir = Path(output_dir)
    try:
        document = json.loads((output_dir / _DOCUMENT).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Step0ConversionError(f"conversion manifest cannot be read: {exc}") from exc
    if not isinstance(document, dict) or set(document) != _FIELDS:
        raise Step0ConversionError("conversion manifest has missing or extra fields")
    payload = {k: v for k, v in document.items() if k != "artifact_hash"}
    if domain_hash(_DOMAIN, payload) != document["artifact_hash"]:
        raise Step0ConversionError("conversion manifest content hash mismatch")
    if (
        document["schema_version"] != "1.0.0"
        or document["kind"] != "unflatten-moe-step0-overlay"
        or document["weights_file"] != _WEIGHTS
        or document["training_ready"] is not False
        or document["initialization"] != "copy-dense-internals-zero-specialist-output.v1"
    ):
        raise Step0ConversionError("conversion manifest has unsupported step-zero semantics")
    if document["source_identity"] != identity:
        raise Step0ConversionError("conversion source identity does not match loaded Dense base")
    if file_sha256(output_dir / _WEIGHTS) != document["weights_sha256"]:
        raise Step0ConversionError("conversion weights checksum mismatch")
    if dense_base_fingerprint(model, candidate=candidate) != document["source_model_state_sha256"]:
        raise Step0ConversionError("conversion source model state does not match the Dense base")
    adapter = _build_adapter(model, _read_topology(document["topology"]), document["seed"])
    try:
        weights = load_file(str(output_dir / _WEIGHTS), device="cpu")
    except (OSError, SafetensorError) as exc:
        raise Step0ConversionError(f"conversion weights cannot be read: {exc}") from exc
    _validate_weights(model, adapter, weights)
    adapter.load_state_dict(weights, strict=True)
    # Activation occurs only after every manifest, source and weight check passed.
    model.architecture_variant = dataclasses.replace(
        model.architecture_variant,
        expert_adapter=ExpertAdapterVariant(mode="sparse", sparse_manifest=adapter.manifest),
    )
    model.sparse_adapter = adapter
    model.requires_grad_(False).eval()
    return document

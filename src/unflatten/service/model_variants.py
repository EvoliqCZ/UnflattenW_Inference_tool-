"""DEC-044 Dodatok D §1: per-variant VRAM estimate, fit, and download status.

Reuses the SAME measured/derived tier table as `quantization.auto_policy`
(`service/policies/precision_auto_policy.v1.json`) so `GET /v1/models`'s
`fits` field can never disagree with what the worker's own `--precision
auto` would resolve.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unflatten.model.quantization.auto_policy import load_auto_policy
from unflatten.service.models_config import is_published

_MIB = 1024 * 1024
#: `int8` and `fp8` share one measured tier (design note in auto_policy.py:
#: `Fp8Linear` dequantizes on this decode shape, so the peak is the same).
_PRECISION_TIER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "bf16": ("bf16",),
    "int8": ("int8_or_fp8",),
    "fp8": ("int8_or_fp8",),
    "int4": ("int4",),
}


#: Owner decision 2026-09-29 (`docs/runs/vc-eviction-prep-2026-09-28.md`,
#: `bypass_static_kv`: -15 % per decode step, identical output on 20/20
#: pages): the worker decodes on a preallocated `StaticKVCache` unless
#: `--kv-cache-mode`/`UNFLATTEN_KV_CACHE_MODE` says otherwise. The API
#: process cannot see the worker's override, so its estimates always assume
#: this default -- conservative when the worker was switched to
#: `bf16_dynamic`. Owner decision later the same day
#: (`docs/runs/cuda-graphs-decode-2026-09-29.md`): `static_graph` -- the same
#: static cache plus piecewise CUDA graphs for the q=1 step, bit-identical
#: tokens, -11 % page wall against `static`, same VRAM reserve.
SERVICE_DEFAULT_KV_CACHE_MODE = "static_graph"


def static_kv_reserve_mib(kv_cache_mode: str, policy: Mapping[str, Any]) -> float:
    """The policy's `static_kv_reserve_mib` for a `static` KV cache, else 0:
    every tier's measured peak was taken on a dynamic cache, and the static
    buffer preallocates all `max_steps` positions up front. `static_graph`
    uses the same buffer (its graph pool is small, q=1 activations only)."""
    if kv_cache_mode in ("static", "static_graph"):
        return float(policy["static_kv_reserve_mib"])
    return 0.0


class ModelVariantsInvalid(ValueError):
    """No auto-policy tier matches a requested precision."""


def _find_tier(precision: str, policy: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = _PRECISION_TIER_CANDIDATES.get(precision)
    if candidates is None:
        raise ModelVariantsInvalid(f"unknown precision {precision!r}")
    for tier in policy["tiers"]:
        if tier["decoder_precision"] in candidates:
            return tier
    raise ModelVariantsInvalid(f"auto policy names no tier for precision {precision!r}")


def estimated_peak_vram_mib(
    precision: str,
    kv_cache: str,
    *,
    policy: Mapping[str, Any] | None = None,
    kv_cache_mode: str = SERVICE_DEFAULT_KV_CACHE_MODE,
) -> float:
    """Margin-inflated VRAM estimate for one (precision, kv_cache) combination,
    identical to the requirement `quantization.auto_policy.resolve_auto_quantization`
    checks against free VRAM for that tier (with the same static KV reserve
    the worker passes it as `extra_reserve_mib`)."""
    policy = policy or load_auto_policy()
    tier = _find_tier(precision, policy)
    required_mib = tier["measured_peak_mib"] * policy["safety_margin_ratio"]
    if tier["kv_capable_dtype"] == "fp8" and kv_cache == "bf16":
        # Never-measured fallback combination: auto_policy costs it higher.
        required_mib *= policy["non_fp8_kv_overhead_ratio"]
    return required_mib + static_kv_reserve_mib(kv_cache_mode, policy)


def variant_fit_reason(
    precision: str,
    kv_cache: str,
    *,
    free_vram_mib: float,
    compute_capability: tuple[int, int] | None = None,
    policy: Mapping[str, Any] | None = None,
    kv_cache_mode: str = SERVICE_DEFAULT_KV_CACHE_MODE,
) -> str | None:
    """Why a variant would not fit, or `None` if it fits.

    Checks in the same order as `variant_fits`: FP8 compute capability first,
    then measured VRAM -- so a card too old for FP8 is reported as
    `"compute_capability"` even if it also lacks the VRAM.
    """
    policy = policy or load_auto_policy()
    if precision == "fp8" and compute_capability is not None:
        if tuple(compute_capability) < tuple(policy["fp8_min_compute_capability"]):
            return "compute_capability"
    if free_vram_mib < estimated_peak_vram_mib(
        precision, kv_cache, policy=policy, kv_cache_mode=kv_cache_mode
    ):
        return "vram"
    return None


def variant_fits(
    precision: str,
    kv_cache: str,
    *,
    free_vram_mib: float,
    compute_capability: tuple[int, int] | None = None,
    policy: Mapping[str, Any] | None = None,
    kv_cache_mode: str = SERVICE_DEFAULT_KV_CACHE_MODE,
) -> bool:
    """Whether a variant's estimated peak fits the measured free VRAM.

    `fp8` additionally requires `compute_capability >= fp8_min_compute_capability`
    (Ada/Hopper, SM 8.9+) when a capability is known -- an older card cannot run
    FP8 weights regardless of how much VRAM it has free.
    """
    return (
        variant_fit_reason(
            precision,
            kv_cache,
            free_vram_mib=free_vram_mib,
            compute_capability=compute_capability,
            policy=policy,
            kv_cache_mode=kv_cache_mode,
        )
        is None
    )


def resolve_default_variant(
    config: Mapping[str, Any],
    *,
    user_default: str | None,
    model_state: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Resolve the effective default variant and where it came from.

    Precedence: an existing user override (`service_settings`) wins; else the
    config's own `default_variant` when it names a variant; else, for
    `default_variant == "auto"`, the largest variant (by estimated peak VRAM)
    that fits `model_state`'s measured device facts, with `fp8` preferred over
    `int8` on a tie (it is faster) and config order breaking any other tie.
    Returns `(None, None)` when nothing can be resolved. Only `published`
    variants take part: an unpublished or planned catalogue entry is never a
    default, whichever of the three sources names it.
    """
    variants = [variant for variant in config["variants"] if is_published(variant)]
    ids = [variant["variant_id"] for variant in variants]
    if user_default is not None and user_default in ids:
        return user_default, "user"
    default_variant = config["default_variant"]
    if default_variant != "auto":
        return default_variant, "config"
    if model_state is None:
        return None, None
    policy = policy or load_auto_policy()
    # Cold startup supplies free memory; public capacity selection intentionally
    # supplies only total memory because an already-resident model owns VRAM.
    free_vram_mib = model_state.get("free_vram_mib", model_state["total_vram_mib"])
    compute_capability = model_state.get("compute_capability")
    candidates = [
        variant
        for variant in variants
        if variant_fits(
            variant["precision"],
            variant["kv_cache"],
            free_vram_mib=free_vram_mib,
            compute_capability=compute_capability,
            policy=policy,
        )
    ]
    if not candidates:
        return None, None
    best = max(
        range(len(candidates)),
        key=lambda index: (
            estimated_peak_vram_mib(
                candidates[index]["precision"], candidates[index]["kv_cache"], policy=policy
            ),
            1 if candidates[index]["precision"] == "fp8" else 0,
            -index,
        ),
    )
    return candidates[best]["variant_id"], "auto"


def variant_download_state(
    variant: Mapping[str, Any],
    *,
    cache_dir: Path,
    active_command: Mapping[str, Any] | None,
    last_command: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Derive a variant's `download` status from the active/last model command.

    Order: an active (`queued`/`running`) command wins -- `queued` reports as
    `downloading` with no byte progress yet; else an already-downloaded
    snapshot is `ready`; else a `failed` last command is reported with its
    error; else the variant is `absent`.
    """
    if active_command is not None:
        phase = active_command.get("phase")
        if active_command["state"] == "queued" or phase is None:
            state = "downloading"
            bytes_done = None
        else:
            state = phase
            bytes_done = active_command.get("bytes_done")
        return {
            "state": state,
            "bytes_done": bytes_done,
            "bytes_total": active_command.get("bytes_total"),
            "error_code": None,
            "command_id": active_command["command_id"],
        }
    if (
        last_command is not None
        and last_command.get("kind") == "load"
        and last_command["state"] == "failed"
    ):
        return {
            "state": "failed",
            "bytes_done": None,
            "bytes_total": None,
            "error_code": last_command.get("error_code"),
            "command_id": last_command.get("command_id"),
        }
    downloaded, _ = is_variant_downloaded(variant, cache_dir=cache_dir)
    if downloaded:
        return {
            "state": "ready",
            "bytes_done": None,
            "bytes_total": None,
            "error_code": None,
            "command_id": None,
        }
    if last_command is not None and last_command["state"] == "failed":
        return {
            "state": "failed",
            "bytes_done": None,
            "bytes_total": None,
            "error_code": last_command.get("error_code"),
            "command_id": last_command.get("command_id"),
        }
    return {
        "state": "absent",
        "bytes_done": None,
        "bytes_total": None,
        "error_code": None,
        "command_id": None,
    }


def delete_variant_snapshot(variant: Mapping[str, Any], *, cache_dir: Path) -> None:
    """Delete only the variant's pinned revision from the local HF cache.

    A local `bundle` variant has no downloaded snapshot to delete -- the
    caller (worker command executor) must never reach this for one."""
    if "bundle" in variant:
        raise ModelVariantsInvalid("cannot delete a local bundle variant's snapshot")
    source = variant["source"]
    info = _scan_cache_dir(cache_dir)
    info.delete_revisions(source["revision"]).execute()


def _snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def resolve_variant_bundle(
    variant: Mapping[str, Any], *, cache_dir: Path
) -> tuple[Path, str | None]:
    """Resolve a models config variant to a local bundle directory.

    A local `bundle` path variant is returned unchanged, with no pinned
    manifest sha256 (the operator already trusts the path). An `hf` source
    variant is downloaded (or reused from cache) at its pinned commit SHA via
    `huggingface_hub.snapshot_download`, and its `bundle_manifest_sha256` is
    returned so the caller can pass it to `model_bundle.verify_bundle` --
    this function never verifies the bundle itself.
    """
    if "bundle" in variant:
        return Path(variant["bundle"]), None
    source = variant["source"]
    snapshot_dir = Path(
        _snapshot_download(
            repo_id=source["repo_id"], revision=source["revision"], cache_dir=cache_dir
        )
    )
    bundle_dir = snapshot_dir / source["subfolder"] if "subfolder" in source else snapshot_dir
    return bundle_dir, source["bundle_manifest_sha256"]


def _scan_cache_dir(cache_dir: Path) -> Any:
    from huggingface_hub import scan_cache_dir

    return scan_cache_dir(cache_dir)


def is_variant_downloaded(
    variant: Mapping[str, Any], *, cache_dir: Path
) -> tuple[bool, int | None]:
    """Whether a variant's bundle is already present locally, and its size.

    A local `bundle` path variant is always considered downloaded (it is
    already on disk by construction). An `hf` source variant is downloaded
    when its pinned revision is present in the local HF cache under
    `cache_dir` -- the same cache the worker's `snapshot_download` uses.
    """
    from huggingface_hub.errors import CacheNotFound

    if "bundle" in variant:
        return True, None
    source = variant["source"]
    try:
        info = _scan_cache_dir(cache_dir)
    except CacheNotFound:
        return False, None
    for repo in info.repos:
        if repo.repo_id != source["repo_id"]:
            continue
        for revision in repo.revisions:
            if revision.commit_hash == source["revision"]:
                if not _snapshot_files_complete(revision, source):
                    return False, None
                return True, int(revision.size_on_disk)
    return False, None


def _snapshot_files_complete(revision: Any, source: Mapping[str, Any]) -> bool:
    """A cache revision can exist while HF is still downloading its blobs.

    This is a cheap completeness check, not weight verification. The worker
    still verifies all hashes before loading the actual model.
    """
    try:
        root = Path(revision.snapshot_path)
        if "subfolder" in source:
            root /= source["subfolder"]
        raw = (root / "manifest.json").read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != source["bundle_manifest_sha256"]:
            return False
        manifest = json.loads(raw)
        if not manifest["files"]:
            return False
        for name, info in manifest["files"].items():
            if (root / name).stat().st_size != info["size_bytes"]:
                return False
        backbone_files = manifest.get("identities", {}).get("backbone", {}).get("files", {})
        return all((root / "backbone" / name).is_file() for name in backbone_files)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False

"""Resolve `--precision auto` (owner requirement 2026-09-27, "sprav to tak,
že to nebude len pre RTX 4070, ale všeobecne"): pick a `QuantizationSpec`
from MEASURED VRAM facts at worker startup (`torch.cuda.mem_get_info`,
`torch.cuda.get_device_capability`), not a hardcoded card name.

The policy table (`service/policies/precision_auto_policy.v1.json`) holds
every threshold this module uses; only the RTX 4070 SUPER row is a real
measurement (`docs/runs/quantization-rtx4070-2026-09-27.md`) -- every other
card's outcome is DERIVED from that single measurement plus a stated safety
margin, never a second measurement. FP8 weights and the FP8 KV cache are
only offered at or above `fp8_min_compute_capability` (Ada/Hopper, SM 8.9+);
below that this module falls back to INT8 weights (same measured peak as
FP8 on the one measured card, because `Fp8Linear` itself dequantizes on
this decode shape -- see the measurement doc's finding 2) and a bf16 KV
cache, inflated by `non_fp8_kv_overhead_ratio` because that combination was
never measured directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from unflatten.model.quantization.spec import QuantizationSpec

_MIB = 1024 * 1024

DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "service" / "policies" / "precision_auto_policy.v1.json"
)


class QuantizationAutoPolicyInvalid(ValueError):
    """The auto-precision policy file is malformed."""


class QuantizationAutoResolutionRefused(ValueError):
    """No tier in the policy fits the measured free VRAM -- fail closed
    rather than let a chosen precision OOM mid-job."""


def load_auto_policy(path: Path | str = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    try:
        policy = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuantizationAutoPolicyInvalid(f"cannot read auto policy {path}: {exc}") from exc
    for key in ("fp8_min_compute_capability", "safety_margin_ratio", "tiers"):
        if key not in policy:
            raise QuantizationAutoPolicyInvalid(f"auto policy is missing {key!r}")
    if not policy["tiers"]:
        raise QuantizationAutoPolicyInvalid("auto policy names no tiers")
    return policy


def resolve_auto_quantization(
    *,
    free_vram_bytes: int,
    compute_capability: tuple[int, int],
    total_vram_bytes: int | None = None,
    policy: dict[str, Any] | None = None,
    extra_reserve_mib: float = 0.0,
) -> QuantizationSpec:
    """Pick the highest-quality tier (bf16 first, then 8-bit, then int4)
    whose margin-inflated requirement fits `free_vram_bytes`.

    `extra_reserve_mib` (default 0, today's behaviour) is added unmargined to
    every tier's requirement -- the service worker passes the policy's
    `static_kv_reserve_mib` when it decodes on a static KV cache.

    `total_vram_bytes` is accepted only for the caller's own logging --
    gating uses `free_vram_bytes` alone, since that is what a real page
    decode actually has left after whatever else already holds the card
    (desktop compositor, other processes)."""
    del total_vram_bytes
    policy = policy or load_auto_policy()
    fp8_ok = tuple(compute_capability) >= tuple(policy["fp8_min_compute_capability"])
    margin = policy["safety_margin_ratio"]
    overhead = policy["non_fp8_kv_overhead_ratio"]

    for tier in policy["tiers"]:
        kv_capable = tier["kv_capable_dtype"]
        kv_dtype = "fp8" if (kv_capable == "fp8" and fp8_ok) else "bf16"
        required_bytes = tier["measured_peak_mib"] * _MIB * margin
        if kv_capable == "fp8" and kv_dtype == "bf16":
            # Fell back from fp8 to bf16 KV: this combination was never
            # measured, so treat it as costing more, not the same.
            required_bytes *= overhead
        required_bytes += extra_reserve_mib * _MIB
        if free_vram_bytes < required_bytes:
            continue
        decoder = tier["decoder_precision"]
        if decoder == "int8_or_fp8":
            decoder = "fp8" if fp8_ok else "int8"
        return QuantizationSpec(decoder_precision=decoder, kv_cache_dtype=kv_dtype)

    raise QuantizationAutoResolutionRefused(
        f"no precision tier fits {free_vram_bytes / _MIB:.1f} MiB free VRAM "
        f"(compute capability {tuple(compute_capability)}) -- refusing to guess "
        "a precision that would OOM mid-job"
    )

"""RFC005-ERRATA-007 E5 rule 2: the `vc0_run` CLI over the VC-0 harness.

VC-0 "moze ukoncit program" -- it measures the bypass path's own token
count, timing and per-slice quality with no gradient and no compressor, so
an owner can decide whether compression has anything to save before VC-1
spends any training budget.

**VC-0 rehearsal on S4 v15b, not the formal VC-0** -- runs on a local
checkpoint (not S7 which does not exist yet), discovery only, no gradient.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from unflatten.model.visual_compression.measurement import (
    BaselineMeasurement,
    MeasuredCandidate,
    measure_compression_baseline,
)

REHEARSAL_TAG = "VC-0 rehearsal on S4 v15b, not the formal VC-0"
DEFAULT_BUNDLE = Path.home() / "unflatten-runs" / "export-v15b-140000-bundle"
DEFAULT_SELECTION_DIR = Path.home() / "unflatten-runs" / "dspark-rehearsal-tf"


class Vc0RealRunNotAvailable(NotImplementedError):
    """A non-dry-run VC-0 measurement needs a real, loaded primary candidate."""


class IdealRegionMapUnavailable(ValueError):
    """The rehearsal cannot prove an AST/aligner-backed ideal PRE map."""


@dataclass
class RehearsalPageResult:
    page_index: int
    bundle_id: str
    split: str
    arm: str
    encoder_s: float
    prefill_s: float
    total_decode_s: float
    decode_steps: int
    mean_step_ms: float
    visual_tokens: int
    peak_vram_mib: float
    wall_s: float
    original_visual_tokens: int | None = None
    gold_tokens: int | None = None
    decoded_tokens: int | None = None
    text_similarity: float | None = None
    protected_region_count: int | None = None
    error: str | None = None
    note: str | None = None


def load_runtime(bundle_dir, device, decoder_precision, kv_cache_dtype):
    from scripts.profile_decode import load_runtime as _load
    return _load(
        bundle_dir=bundle_dir, device=device,
        decoder_precision=decoder_precision, kv_cache_dtype=kv_cache_dtype,
        typed_decode_override=None, region_decode_override=None,
    )


def build_model_safe_input(png_path, dpi=300):
    from scripts.profile_decode import build_model_safe_input as _build
    return _build(png_path, dpi)


def gold_token_index(gold_dir, split):
    path = gold_dir / f"gold-tokens-{split}.jsonl"
    if not path.exists():
        return {}
    index = {}
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index[(row["document_bundle_id"], row["page_index"])] = row["packed_token_ids"]
    return index


def text_similarity(runtime: Any, decoded: tuple[int, ...], gold: list[int]) -> float:
    """Existing S5-style text similarity over assembled token-stream text."""
    from scripts.quantized_quality_report import text_similarity_ratio

    from unflatten.eval.s2_gate.score import assemble_hypothesis_text

    decoded_text = assemble_hypothesis_text(
        decoded, vocabulary=runtime.vocabulary, tokenizer=runtime.tokenizer
    )
    gold_text = assemble_hypothesis_text(
        gold, vocabulary=runtime.vocabulary, tokenizer=runtime.tokenizer
    )
    return text_similarity_ratio(decoded_text, gold_text)


def _refuse_if_gpu_busy(device_index, max_wait_s=300.0):
    from scripts.profile_decode import _refuse_if_gpu_busy
    _refuse_if_gpu_busy(device_index, max_wait_s)


def _decode_with_phase_timings(runtime: Any, prepared: Any) -> tuple[Any, str | None, dict]:
    """Decode once while timing the real prefill and every model step."""
    from unflatten.model.free_running_decode import DecodeMaxStepsExceeded

    model = runtime.model
    prefix_name = (
        "encode_prefix_with_visual_hidden"
        if runtime.decode_policy.region_decode
        else "encode_prefix"
    )
    original_prefix = getattr(model, prefix_name)
    original_step = model.encode_step
    prefill_timings: list[float] = []
    step_timings: list[float] = []

    @functools.wraps(original_prefix)
    def timed_prefix(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_prefix(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_timings.append(time.perf_counter() - started)
        return result

    @functools.wraps(original_step)
    def timed_step(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_step(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_timings.append(time.perf_counter() - started)
        return result

    setattr(model, prefix_name, timed_prefix)
    model.encode_step = timed_step
    started = time.perf_counter()
    error = None
    try:
        hypothesis = runtime.decode_features(prepared)
    except DecodeMaxStepsExceeded as exc:
        hypothesis = exc
        error = f"DecodeMaxStepsExceeded at {len(exc.accepted_tokens)} tokens"
    finally:
        setattr(model, prefix_name, original_prefix)
        model.encode_step = original_step
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    wall_s = time.perf_counter() - started
    prefill_s = sum(prefill_timings)
    return hypothesis, error, {
        "wall_s": wall_s,
        "prefill_s": prefill_s,
        "decode_after_prefill_s": max(0.0, wall_s - prefill_s),
        "model_step_total_s": sum(step_timings),
        "model_step_mean_ms": (
            1000 * statistics.mean(step_timings) if step_timings else 0.0
        ),
    }


def measure_bypass(runtime, png_path, gold_tokens, *, label):
    model_input = build_model_safe_input(png_path, 300)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    wall_start = time.perf_counter()
    t0 = time.perf_counter()
    prepared = runtime.extract(model_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    encoder_s = time.perf_counter() - t0

    hypothesis, error, phases = _decode_with_phase_timings(runtime, prepared)
    wall_s = time.perf_counter() - wall_start

    decode_steps = len(hypothesis.accepted_tokens)
    decoded = hypothesis.accepted_tokens
    visual_tokens = int(prepared.visual_features.shape[1])
    peak_vram_mib = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available() else 0.0
    )
    similarity = (
        text_similarity(runtime, decoded, gold_tokens)
        if gold_tokens is not None and error is None
        else None
    )

    return RehearsalPageResult(
        page_index=0, bundle_id="", split="", arm=label,
        encoder_s=encoder_s, prefill_s=phases["prefill_s"],
        total_decode_s=phases["decode_after_prefill_s"], decode_steps=decode_steps,
        mean_step_ms=phases["model_step_mean_ms"],
        visual_tokens=visual_tokens, peak_vram_mib=peak_vram_mib,
        wall_s=wall_s,
        gold_tokens=len(gold_tokens) if gold_tokens is not None else None,
        decoded_tokens=decode_steps, text_similarity=similarity, error=error,
    )


def crop_prepared_visual_prefix(prepared: Any, ratio: float) -> Any:
    """Crop a valid rectangular Qwen visual prefix to approximately ``ratio``."""
    from unflatten.model.candidate_runtime import PreparedCandidateInput
    grid_t, grid_h, grid_w = (int(v) for v in prepared.image_grid_thw[0].tolist())
    full_tokens = int(prepared.visual_features.shape[1])
    merge_area, remainder = divmod(grid_t * grid_h * grid_w, full_tokens)
    merge = math.isqrt(merge_area)
    if remainder or merge * merge != merge_area or grid_h % merge or grid_w % merge:
        raise ValueError(
            "visual features/grid_thw do not form a rectangular spatial-merge grid: "
            f"tokens={full_tokens}, grid={(grid_t, grid_h, grid_w)}"
        )
    merged_h, merged_w = grid_h // merge, grid_w // merge
    feature_grid = prepared.visual_features.reshape(
        1, grid_t, merged_h, merged_w, prepared.visual_features.shape[-1]
    )

    target = max(1, round(full_tokens * ratio))
    candidates = (
        (h, w)
        for h in range(1, merged_h + 1)
        for w in range(1, merged_w + 1)
    )
    crop_h, crop_w = min(
        candidates,
        key=lambda shape: (
            abs(shape[0] * shape[1] - target),
            abs(shape[0] / shape[1] - merged_h / merged_w),
        ),
    )
    features = feature_grid[:, :, :crop_h, :crop_w, :].reshape(
        1, grid_t * crop_h * crop_w, prepared.visual_features.shape[-1]
    )
    grid = torch.tensor(
        [[grid_t, crop_h * merge, crop_w * merge]],
        dtype=prepared.image_grid_thw.dtype,
        device=prepared.image_grid_thw.device,
    )
    return PreparedCandidateInput(features, grid)


def measure_kv_sensitivity(runtime, png_path, ratios, *, label_prefix):
    """Timing-only prefill and decode-step cost for shortened visual prefixes.

    Qwen's ``grid_thw`` describes pre-spatial-merge cells; the prepared visual
    feature count is ``h * w / merge**2``. Each arm crops a valid rectangular
    merged grid and writes the corresponding ``grid_thw`` instead of inventing
    a flat ``[1, 1, tokens]`` grid. No arm makes a quality claim.
    """
    from unflatten.model import events_vocab
    from unflatten.model.candidate_model import TeacherForcedBatch
    from unflatten.model.composite_vocab import QwenRef, StructuralRef

    prepared = runtime.extract(build_model_safe_input(png_path, 300))
    full_tokens = int(prepared.visual_features.shape[1])

    results = []
    model = runtime.model
    device = next(model.parameters()).device
    placeholder_id = runtime.vocabulary.pack(QwenRef(runtime.candidate.image_token_id))
    bos_id = runtime.vocabulary.pack(
        StructuralRef(local_id=events_vocab.TOKEN_IDS[events_vocab.END_DOCUMENT])
    )
    for ratio in ratios:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cropped = crop_prepared_visual_prefix(prepared, ratio)
        visual_tokens = int(cropped.visual_features.shape[1])
        prefix_ids = [placeholder_id] * visual_tokens + [bos_id]
        prefix_batch = TeacherForcedBatch(
            visual_features=cropped.visual_features,
            composite_input_ids=torch.tensor([prefix_ids], dtype=torch.long, device=device),
            composite_attention_mask=torch.ones(
                1, len(prefix_ids), dtype=torch.long, device=device
            ),
            image_grid_thw=cropped.image_grid_thw,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            state, _hidden, _visual = model.encode_prefix_with_visual_hidden(prefix_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_s = time.perf_counter() - started

        step_timings = []
        with torch.inference_mode():
            for _ in range(5):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_started = time.perf_counter()
                state, _hidden = model.encode_step(state, bos_id)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step_timings.append(time.perf_counter() - step_started)
        step_total = sum(step_timings)
        results.append(RehearsalPageResult(
            page_index=0, bundle_id="", split="",
            arm=f"{label_prefix}_kv_{int(ratio * 100)}",
            encoder_s=0.0,
            prefill_s=prefill_s,
            total_decode_s=step_total,
            decode_steps=len(step_timings),
            mean_step_ms=1000 * statistics.mean(step_timings),
            visual_tokens=visual_tokens,
            peak_vram_mib=(
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available() else 0.0
            ),
            wall_s=prefill_s + step_total,
            note=(
                "timing-only valid rectangular prefix; "
                f"requested={ratio:.2f}, actual={visual_tokens / full_tokens:.4f}"
            ),
        ))
        del state, _hidden, _visual, prefix_batch, cropped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def release_artifact_index(release_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        row["document_bundle_id"]: row["artifacts"]
        for row in _jsonl(release_dir / "artifacts" / "index.jsonl")
    }


def ideal_region_proposals(
    *,
    geometry_store: Path,
    bundle_id: str,
    page_index: int,
    png_sha256: str,
    expected_target_ref: str,
    page_width_px: int,
    page_height_px: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Load an exact-raster, exact-target AST/aligner ideal map fail-closed."""
    from unflatten.model.region_targets import RegionTarget
    from unflatten.model.visual_compression.region_importance_targets import (
        build_region_importance_targets,
        importance_targets_to_pixel_region_proposals,
    )

    match = None
    for row in _jsonl(geometry_store / "renders.jsonl"):
        if row.get("document_bundle_id") != bundle_id:
            continue
        if row.get("render_status") != "rendered":
            continue
        artifacts = row.get("artifacts") or {}
        pages = artifacts.get("pages") or []
        if page_index >= len(pages) or pages[page_index].get("sha256") != png_sha256:
            continue
        target = artifacts.get("target_bundle") or {}
        if target.get("ref") != expected_target_ref:
            continue
        if artifacts.get("observation_sidecar") is None:
            continue
        match = artifacts
        break
    if match is None:
        raise IdealRegionMapUnavailable(
            f"{bundle_id} page {page_index}: {geometry_store} has no rendered row "
            "matching both the selected raster SHA-256 and release target_bundle ref"
        )

    target_ref = match["target_bundle"]["ref"]
    sidecar_ref = match["observation_sidecar"]["ref"]
    target_path = geometry_store / target_ref
    sidecar_path = geometry_store / sidecar_ref
    if not target_path.is_file() or not sidecar_path.is_file():
        raise IdealRegionMapUnavailable(
            f"{bundle_id}: ideal-map artifacts are missing under {geometry_store}"
        )
    target = json.loads(target_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    pages = sidecar.get("pages") or []
    if page_index >= len(pages) or pages[page_index].get("page_index") != page_index:
        raise IdealRegionMapUnavailable(
            f"{bundle_id}: observation sidecar has no page {page_index}"
        )
    page = pages[page_index]
    sidecar_width = page.get("width_px")
    sidecar_height = page.get("height_px")
    if (
        not isinstance(sidecar_width, int)
        or not isinstance(sidecar_height, int)
        or abs(sidecar_width - page_width_px) > 1
        or abs(sidecar_height - page_height_px) > 1
    ):
        raise IdealRegionMapUnavailable(
            f"{bundle_id}: sidecar page size differs from selected raster by more than 1 px"
        )

    node_types = {
        event["node_id"]: event["node_type"]
        for event in target["structural"]["events"]
        if event.get("op") == "open"
    }
    by_node: dict[str, list[dict[str, Any]]] = {}
    for observation in page.get("observations", []):
        by_node.setdefault(observation["node_id"], []).append(observation)
    region_targets = []
    for node_id, observations in sorted(by_node.items()):
        node_type = node_types.get(node_id)
        if node_type is None:
            raise IdealRegionMapUnavailable(
                f"{bundle_id}: sidecar node {node_id!r} has no target-bundle node type"
            )
        observations.sort(key=lambda item: item["fragment_index"])
        region_targets.append(RegionTarget(
            node_id=node_id,
            node_type=node_type,
            anchor_token_index=0,
            page_evidence="observed",
            observable_mask=True,
            supervision_mask=True,
            regions=tuple(tuple(item["bbox_ppm"]) for item in observations),
            continuation=tuple(
                (bool(item["continuation_prev"]), bool(item["continuation_next"]))
                for item in observations
            ),
            mapping_method=observations[0].get("mapping_method", "text"),
            chain_scope=observations[0].get("chain_scope", "document"),
        ))
    importance = build_region_importance_targets(tuple(region_targets))
    proposals = importance_targets_to_pixel_region_proposals(
        importance,
        page_width_px=page_width_px,
        page_height_px=page_height_px,
        protect_all_observed=True,
    )
    if not proposals:
        raise IdealRegionMapUnavailable(f"{bundle_id}: ideal map contains no observed region")
    return proposals, {
        "geometry_store": str(geometry_store),
        "target_bundle_ref": target_ref,
        "observation_sidecar_ref": sidecar_ref,
        "sidecar_page_size": [sidecar_width, sidecar_height],
        "raster_page_size": [page_width_px, page_height_px],
        "protected_region_count": len(proposals),
    }


def select_ideal_map_valid_pages(
    pages,
    *,
    pages_dir,
    geometry_store,
    release_artifacts,
):
    """Deterministically preflight PRE evidence before choosing a run limit."""
    from PIL import Image

    eligible = []
    excluded = []
    maps = {}
    for page_info in pages:
        bundle_id = page_info["bundle_id"]
        page_index = page_info.get("page_index", 0)
        try:
            target_ref = (
                (release_artifacts.get(bundle_id) or {}).get("target_bundle") or {}
            ).get("ref")
            if target_ref is None:
                raise IdealRegionMapUnavailable(
                    f"{bundle_id}: release artifact index has no target_bundle ref"
                )
            with Image.open(pages_dir / page_info["png_ref"]) as image:
                page_width_px, page_height_px = image.size
            proposals, provenance = ideal_region_proposals(
                geometry_store=geometry_store,
                bundle_id=bundle_id,
                page_index=page_index,
                png_sha256=page_info["png_sha256"],
                expected_target_ref=target_ref,
                page_width_px=page_width_px,
                page_height_px=page_height_px,
            )
        except (IdealRegionMapUnavailable, FileNotFoundError) as exc:
            excluded.append({
                "bundle_id": bundle_id,
                "page_index": page_index,
                "reason": str(exc),
            })
            continue
        eligible.append(page_info)
        maps[(bundle_id, page_index)] = (proposals, provenance)
    return eligible, excluded, maps


def measure_pre_region(
    runtime,
    png_path,
    gold_tokens,
    *,
    label_prefix,
    budgets,
    proposals,
    provenance,
):
    from unflatten.model.input_contract import ModelSafeInput
    from unflatten.model.visual_compression.pre_encoder import apply_region_resolution_plan
    from unflatten.model.visual_compression.region_plan import plan_region_resolution

    results = []
    base_msi = build_model_safe_input(png_path, 300)
    original_plane = base_msi.planes[0]

    for budget in budgets:
        plan = plan_region_resolution(
            original_plane.width_px,
            original_plane.height_px,
            proposals,
            global_downsample=budget,
        )
        compressed_plane = apply_region_resolution_plan(original_plane, plan)
        compressed = ModelSafeInput(
            planes=(compressed_plane,),
            masks=base_msi.masks,
            transforms=base_msi.transforms,
            scale=base_msi.scale,
            profile=base_msi.profile,
            input_mode=base_msi.input_mode,
            locale_hint=base_msi.locale_hint,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        wall_start = time.perf_counter()
        t0 = time.perf_counter()
        prepared = runtime.extract(compressed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        encoder_s = time.perf_counter() - t0

        hypothesis, error, phases = _decode_with_phase_timings(runtime, prepared)
        wall_s = time.perf_counter() - wall_start

        decode_steps = len(hypothesis.accepted_tokens)
        visual_tokens = int(prepared.visual_features.shape[1])
        peak_vram_mib = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available() else 0.0
        )
        similarity = (
            text_similarity(runtime, hypothesis.accepted_tokens, gold_tokens)
            if gold_tokens is not None and error is None else None
        )

        label = f"{label_prefix}_pre_budget_{budget}"
        results.append(RehearsalPageResult(
            page_index=0, bundle_id="", split="", arm=label,
            encoder_s=encoder_s, prefill_s=phases["prefill_s"],
            total_decode_s=phases["decode_after_prefill_s"], decode_steps=decode_steps,
            mean_step_ms=phases["model_step_mean_ms"],
            visual_tokens=visual_tokens, peak_vram_mib=peak_vram_mib,
            wall_s=wall_s,
            gold_tokens=len(gold_tokens) if gold_tokens is not None else None,
            decoded_tokens=decode_steps, text_similarity=similarity,
            protected_region_count=len(proposals),
            error=error,
            note=(
                "ideal AST/aligner map; pixel-only PRE keeps plane dimensions and "
                f"therefore token count; source={provenance['observation_sidecar_ref']}"
            ),
        ))
    return results


def build_uniform_post_encoder_compressor(model, ratio):
    """Build a frozen POST stress arm assigning every token to ``low``.

    This is not a learned adaptive policy. It exercises the real
    ``adaptive-document-visual-memory-v0`` merge and model wiring at a fixed
    compression ratio without using AST labels or running a gradient.
    """
    from unflatten.model.visual_compression.adaptive_v0 import (
        RISK_CLASSES,
        AdaptiveVisualMemoryCompressor,
        AdaptiveVisualMemoryPolicy,
    )

    if ratio < 1:
        raise ValueError(f"POST compression ratio must be >= 1, got {ratio}")
    policy = AdaptiveVisualMemoryPolicy(
        policy_version=f"rehearsal-uniform-low-{ratio}x",
        retention_ratio={
            "protected": 1,
            "high": 1,
            "medium": ratio,
            "low": ratio,
            "uncertain": 1,
        },
    )
    compressor = AdaptiveVisualMemoryCompressor(model.d_model, policy=policy)
    reference = model.dense_adapter.layers[0].norm.weight
    compressor.to(device=reference.device, dtype=reference.dtype)
    compressor.requires_grad_(False)
    compressor.eval()
    with torch.no_grad():
        low_index = RISK_CLASSES.index("low")
        compressor.risk_head.proj.bias[low_index] = 10.0
    return compressor


def measure_post_encoder_uniform(runtime, png_path, gold_tokens, ratios):
    """Measure full decode through real POST merges at fixed stress ratios."""
    results = []
    model = runtime.model
    original_compressor = model.post_encoder_compressor_module
    original_decode_policy = runtime.decode_policy
    try:
        runtime.decode_policy = replace(original_decode_policy, region_decode=False)
        for ratio in ratios:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            model.post_encoder_compressor_module = build_uniform_post_encoder_compressor(
                model, ratio
            )
            wall_start = time.perf_counter()
            encoder_start = time.perf_counter()
            prepared = runtime.extract(build_model_safe_input(png_path, 300))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            encoder_s = time.perf_counter() - encoder_start

            hypothesis, error, phases = _decode_with_phase_timings(runtime, prepared)
            wall_s = time.perf_counter() - wall_start
            original_tokens = int(prepared.visual_features.shape[1])
            compressed_tokens = math.ceil(original_tokens / ratio)
            decode_steps = len(hypothesis.accepted_tokens)
            similarity = (
                text_similarity(runtime, hypothesis.accepted_tokens, gold_tokens)
                if gold_tokens is not None and error is None else None
            )
            results.append(RehearsalPageResult(
                page_index=0,
                bundle_id="",
                split="",
                arm=f"post_uniform_ratio_{ratio}",
                encoder_s=encoder_s,
                prefill_s=phases["prefill_s"],
                total_decode_s=phases["decode_after_prefill_s"],
                decode_steps=decode_steps,
                mean_step_ms=phases["model_step_mean_ms"],
                visual_tokens=compressed_tokens,
                original_visual_tokens=original_tokens,
                peak_vram_mib=(
                    torch.cuda.max_memory_allocated() / (1024 * 1024)
                    if torch.cuda.is_available() else 0.0
                ),
                wall_s=wall_s,
                gold_tokens=len(gold_tokens) if gold_tokens is not None else None,
                decoded_tokens=decode_steps,
                text_similarity=similarity,
                error=error,
                note=(
                    "uniform POST stress curve through adaptive-document-visual-memory-v0; "
                    "all tokens forced low; not a learned adaptive compressor; PRE=bypass; "
                    "region_decode=false because the POST retention/protected-detail path "
                    "is not yet wired into Node Visual Aligner; compare only with ratio 1"
                ),
            ))
            model.post_encoder_compressor_module = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        model.post_encoder_compressor_module = original_compressor
        runtime.decode_policy = original_decode_policy
    return results


def result_to_dict(r):
    d = {
        "page_index": r.page_index,
        "bundle_id": r.bundle_id,
        "split": r.split,
        "arm": r.arm,
        "encoder_s": round(r.encoder_s, 4),
        "prefill_s": round(r.prefill_s, 4),
        "total_decode_s": round(r.total_decode_s, 4),
        "decode_steps": r.decode_steps,
        "mean_step_ms": round(r.mean_step_ms, 4),
        "visual_tokens": r.visual_tokens,
        "peak_vram_mib": round(r.peak_vram_mib, 1),
        "wall_s": round(r.wall_s, 4),
    }
    if r.original_visual_tokens is not None:
        d["original_visual_tokens"] = r.original_visual_tokens
    if r.gold_tokens is not None:
        d["gold_tokens"] = r.gold_tokens
    if r.decoded_tokens is not None:
        d["decoded_tokens"] = r.decoded_tokens
    if r.text_similarity is not None:
        d["text_similarity"] = round(r.text_similarity, 6)
    if r.protected_region_count is not None:
        d["protected_region_count"] = r.protected_region_count
    if r.error:
        d["error"] = r.error
    if r.note:
        d["note"] = r.note
    return d


def failed_page_result(*, arm, bundle_id, split, page_index, error):
    return RehearsalPageResult(
        page_index=page_index,
        bundle_id=bundle_id,
        split=split,
        arm=arm,
        encoder_s=0.0,
        prefill_s=0.0,
        total_decode_s=0.0,
        decode_steps=0,
        mean_step_ms=0.0,
        visual_tokens=0,
        peak_vram_mib=0.0,
        wall_s=0.0,
        error=error,
    )


def compute_summary(results):
    by_arm = {}
    for r in results:
        by_arm.setdefault(r["arm"], []).append(r)

    summary = {}
    for arm, arm_results in sorted(by_arm.items()):
        valid_results = [row for row in arm_results if not row.get("error")]
        n = len(valid_results)
        medians = {}
        for key in ("visual_tokens", "wall_s", "encoder_s", "prefill_s", "total_decode_s",
                     "decode_steps", "mean_step_ms", "peak_vram_mib"):
            vals = [r[key] for r in valid_results if r[key] is not None]
            if vals:
                medians[key] = round(statistics.median(vals), 4)
        similarities = [
            r["text_similarity"] for r in valid_results
            if r.get("text_similarity") is not None
        ]
        if similarities:
            medians["text_similarity"] = round(statistics.median(similarities), 6)
        summary[arm] = {
            "n": n,
            "failed": len(arm_results) - n,
            "median": medians,
        }

    if "bypass" in by_arm:
        bypass_med = summary["bypass"]["median"]
        bypass_by_page = {
            (row["bundle_id"], row["page_index"]): row
            for row in by_arm["bypass"]
            if not row.get("error")
        }
        for arm, arm_sum in summary.items():
            if arm == "bypass":
                continue
            if arm.startswith("bypass_kv_") or arm.startswith("post_uniform_ratio_"):
                # KV arms contain prefill plus five steps, not a full decode,
                # so page-time deltas against bypass would be meaningless.
                continue
            arm_sum["vs_bypass"] = {}
            for key in ("wall_s", "visual_tokens", "mean_step_ms"):
                bv = bypass_med.get(key)
                av = arm_sum["median"].get(key)
                if bv and av and bv > 0:
                    arm_sum["vs_bypass"][f"{key}_pct_change"] = round(
                        (av - bv) / bv * 100, 2
                    )
            paired = [
                (row, bypass_by_page[(row["bundle_id"], row["page_index"])])
                for row in by_arm[arm]
                if not row.get("error")
                and (row["bundle_id"], row["page_index"]) in bypass_by_page
            ]
            if paired:
                arm_sum["paired_vs_bypass"] = {
                    "n": len(paired),
                    "wall_s_delta_median": round(
                        statistics.median(row["wall_s"] - base["wall_s"] for row, base in paired),
                        4,
                    ),
                    "visual_tokens_delta_median": round(
                        statistics.median(
                            row["visual_tokens"] - base["visual_tokens"] for row, base in paired
                        ),
                        4,
                    ),
                }
                quality_pairs = [
                    row["text_similarity"] - base["text_similarity"]
                    for row, base in paired
                    if row.get("text_similarity") is not None
                    and base.get("text_similarity") is not None
                ]
                if quality_pairs:
                    arm_sum["paired_vs_bypass"]["text_similarity_delta_median"] = round(
                        statistics.median(quality_pairs), 6
                    )
    post_control = by_arm.get("post_uniform_ratio_1")
    if post_control:
        control_by_page = {
            (row["bundle_id"], row["page_index"]): row
            for row in post_control
            if not row.get("error")
        }
        for arm, arm_results in by_arm.items():
            if not arm.startswith("post_uniform_ratio_") or arm == "post_uniform_ratio_1":
                continue
            paired = [
                (row, control_by_page[(row["bundle_id"], row["page_index"])])
                for row in arm_results
                if not row.get("error")
                and (row["bundle_id"], row["page_index"]) in control_by_page
            ]
            if not paired:
                continue
            comparison = {
                "n": len(paired),
                "wall_s_delta_median": round(
                    statistics.median(row["wall_s"] - base["wall_s"] for row, base in paired),
                    4,
                ),
                "visual_tokens_delta_median": round(
                    statistics.median(
                        row["visual_tokens"] - base["visual_tokens"] for row, base in paired
                    ),
                    4,
                ),
            }
            quality_pairs = [
                row["text_similarity"] - base["text_similarity"]
                for row, base in paired
                if row.get("text_similarity") is not None
                and base.get("text_similarity") is not None
            ]
            if quality_pairs:
                comparison["text_similarity_delta_median"] = round(
                    statistics.median(quality_pairs), 6
                )
            summary[arm]["paired_vs_post_ratio_1"] = comparison
    return summary


def phase_statuses(requested_arms, results, n_pages):
    statuses = {
        "model_and_inputs": {"status": "DONE", "detail": f"{n_pages} selected page(s)"},
        "bypass": {"status": "NOT_RUN", "detail": "not requested"},
        "kv_sensitivity": {"status": "NOT_RUN", "detail": "not requested"},
        "pre_ideal_map": {"status": "NOT_RUN", "detail": "not requested"},
        "post_uniform_stress": {"status": "NOT_RUN", "detail": "not requested"},
        "summary": {"status": "DONE", "detail": "JSON and markdown generated"},
    }
    arm_matchers = {
        "bypass": lambda arm: arm == "bypass",
        "kv_sensitivity": lambda arm: arm.startswith("bypass_kv_"),
        "pre_ideal_map": lambda arm: arm.startswith("bypass_pre_"),
        "post_uniform_stress": lambda arm: arm.startswith("post_uniform_ratio_"),
    }
    for requested, status_key in (
        ("bypass", "bypass"),
        ("kv_sensitivity", "kv_sensitivity"),
        ("pre_region", "pre_ideal_map"),
        ("post_encoder", "post_uniform_stress"),
    ):
        if requested not in requested_arms:
            continue
        matcher = arm_matchers[status_key]
        rows = [row for row in results if matcher(row["arm"])]
        errors = [row for row in rows if row.get("error")]
        statuses[status_key] = {
            "status": "DONE" if rows and not errors else "FAILED",
            "detail": f"{len(rows)} result row(s), {len(errors)} error(s)",
        }
    return statuses


def generate_markdown_report(summary, n_pages, tag, phases=None):
    def formatted(medians, key, spec):
        value = medians.get(key)
        return format(value, spec) if isinstance(value, int | float) else "N/A"

    lines = [
        f"# {tag}",
        "",
        f"Pages measured: {n_pages}",
        "",
        "## Phase status",
        "",
        "| Phase | Status | Detail |",
        "|---|---|---|",
    ]
    for phase, status in (phases or {}).items():
        lines.append(f"| {phase} | **{status['status']}** | {status['detail']} |")
    lines.extend([
        "",
        "## Per-arm medians",
        "",
        "| Arm | n | Wall (s) | Visual tokens | Mean step (ms) | "
        "Decode steps | Encoder (s) | Prefill (s) | Peak VRAM (MiB) | Text similarity |",
        "|-----|---|----------|---------------|----------------|"
        "--------------|-------------|-------------|-----------------|-----------------|",
    ])
    for arm, arm_sum in sorted(summary.items()):
        m = arm_sum["median"]
        wall = formatted(m, "wall_s", ".1f")
        vt = formatted(m, "visual_tokens", ".1f")
        step = formatted(m, "mean_step_ms", ".1f")
        steps = formatted(m, "decode_steps", ".1f")
        enc = formatted(m, "encoder_s", ".2f")
        prefill = formatted(m, "prefill_s", ".2f")
        vram = formatted(m, "peak_vram_mib", ".0f")
        similarity_value = m.get("text_similarity")
        similarity = f"{similarity_value:.4f}" if similarity_value is not None else "N/A"
        lines.append(
            f"| {arm} | {arm_sum['n']} | {wall} | {vt} | {step} | "
            f"{steps} | {enc} | {prefill} | {vram} | {similarity} |"
        )

    lines.extend(["", "## Paired changes vs bypass", "",
        "| Arm | paired n | Wall Δ (s) | Visual tokens Δ | Text similarity Δ |",
        "|---|---:|---:|---:|---:|"])
    for arm, arm_sum in sorted(summary.items()):
        paired = arm_sum.get("paired_vs_bypass")
        if not paired:
            continue
        lines.append(
            f"| {arm} | {paired['n']} | {paired['wall_s_delta_median']:+.4f} | "
            f"{paired['visual_tokens_delta_median']:+.1f} | "
            f"{paired.get('text_similarity_delta_median', 'N/A')} |"
        )

    lines.extend(["", "## Paired POST changes vs POST ratio 1 control", "",
        "| Arm | paired n | Wall Δ (s) | Visual tokens Δ | Text similarity Δ |",
        "|---|---:|---:|---:|---:|"])
    for arm, arm_sum in sorted(summary.items()):
        paired = arm_sum.get("paired_vs_post_ratio_1")
        if not paired:
            continue
        lines.append(
            f"| {arm} | {paired['n']} | {paired['wall_s_delta_median']:+.4f} | "
            f"{paired['visual_tokens_delta_median']:+.1f} | "
            f"{paired.get('text_similarity_delta_median', 'N/A')} |"
        )

    verdict = "insufficient evidence"
    bypass = summary.get("bypass", {}).get("median", {})
    pre_arms = [
        arm for name, arm in summary.items()
        if name.startswith("bypass_pre_")
        and arm.get("paired_vs_bypass", {}).get("text_similarity_delta_median") is not None
    ]
    verdict_parts = []
    if bypass.get("wall_s") and pre_arms:
        best = min(pre_arms, key=lambda arm: arm["median"]["wall_s"])
        saving = max(0.0, 100 * (bypass["wall_s"] - best["median"]["wall_s"]) / bypass["wall_s"])
        quality_delta = best["paired_vs_bypass"]["text_similarity_delta_median"]
        verdict_parts.append(
            f"PRE may save at most {saving:.1f}% page time at text-similarity "
            f"difference {quality_delta:+.4f}"
        )
    post_control = summary.get("post_uniform_ratio_1", {}).get("median", {})
    post_arms = [
        arm for name, arm in summary.items()
        if name.startswith("post_uniform_ratio_") and name != "post_uniform_ratio_1"
        and arm.get("paired_vs_post_ratio_1", {}).get("text_similarity_delta_median") is not None
    ]
    if post_control.get("wall_s") and post_arms:
        best = min(post_arms, key=lambda arm: arm["median"]["wall_s"])
        saving = max(
            0.0,
            100 * (post_control["wall_s"] - best["median"]["wall_s"])
            / post_control["wall_s"],
        )
        quality_delta = best["paired_vs_post_ratio_1"]["text_similarity_delta_median"]
        verdict_parts.append(
            f"uniform POST stress may save at most {saving:.1f}% page time at "
            f"text-similarity difference {quality_delta:+.4f} versus POST ratio 1"
        )
    if verdict_parts:
        verdict = "; ".join(verdict_parts) + f" on this {n_pages}-page rehearsal cohort"
    lines.extend([
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "This is a rehearsal/discovery measurement. No formal VC-0 decision is made here.",
        "",
        f"({tag})",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stub candidate (for --dry-run, unchanged from before)
# ---------------------------------------------------------------------------


class _StubCandidate:
    """A tiny, deterministic, CPU-only candidate for  tests."""

    def __init__(self, *, tokens: int = 4, d_model: int = 8) -> None:
        self._tokens = tokens
        self._d_model = d_model

    def encode(self, page: object) -> torch.Tensor:
        return torch.zeros(self._tokens, self._d_model)

    def encode_prefix(self, encoded_pages: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(encoded_pages, dim=0)

    def encode_step(self, prefix_state: torch.Tensor) -> torch.Tensor:
        return prefix_state[:1]


def build_stub_candidate() -> MeasuredCandidate:
    return _StubCandidate()


def run_vc0_cohort(
    candidate: MeasuredCandidate, pages: list[object]
) -> list[BaselineMeasurement]:
    return [measure_compression_baseline(candidate, [page]) for page in pages]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"VC-0 rehearsal on S4 v15b (discovery, no gradient). {REHEARSAL_TAG}"
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory for results")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use an in-process stub candidate over synthetic pages (original mode)",
    )
    parser.add_argument(
        "--mode",
        choices=("rehearsal", "dry-run"),
        default=None,
        help="rehearsal: real model, real pages. dry-run: stub (default if --dry-run passed)",
    )
    parser.add_argument("--pages", type=int, default=3, help="number of pages under --dry-run")

    # Rehearsal options
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument(
        "--geometry-store",
        type=Path,
        default=Path.home() / "unflatten-data" / "m2a-v7",
        help="local store supplying the exact-raster/target AST and observation sidecar",
    )
    parser.add_argument("--split", default="val_iid", help="split name in selection.json")
    parser.add_argument("--limit", type=int, default=None, help="max pages to process")
    parser.add_argument(
        "--ideal-map-valid-only",
        action="store_true",
        help=(
            "before --limit, retain only pages with exact raster/target PRE evidence; "
            "record every exclusion in the report"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "int8", "fp8", "int4"), default="fp8")
    parser.add_argument("--kv-cache", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--skip-gpu-wait", action="store_true")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("bypass", "kv_sensitivity", "pre_region", "post_encoder"),
        default=("bypass",),
        help="which measurement arms to run",
    )
    parser.add_argument(
        "--kv-ratios",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.50, 0.25],
        help="KV sensitivity ratios (default: 1.0 0.75 0.50 0.25)",
    )
    parser.add_argument(
        "--post-ratios",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="uniform POST stress ratios (default: 1 2 3 4; not a learned policy)",
    )
    parser.add_argument(
        "--pre-region-budgets",
        type=int,
        nargs="+",
        default=[4, 8],
        help="global_downsample budgets for PRE region arm",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.dry_run or args.mode == "dry-run" or args.mode is None:
        # Dry-run (original) path -- default when no mode is specified
        candidate = build_stub_candidate()
        pages_list = list(range(args.pages))
        measurements = run_vc0_cohort(candidate, pages_list)
        report = {
            "vc0_report_schema_version": "0.1.0",
            "mode": "dry_run",
            "tag": REHEARSAL_TAG,
            "page_count": len(measurements),
            "measurements": [
                {
                    "token_count": m.token_count,
                    "peak_vram_bytes": m.peak_vram_bytes,
                    "timing": {
                        "encoder_s": m.timing.encoder_s,
                        "encode_prefix_s": m.timing.encode_prefix_s,
                        "encode_step_s": m.timing.encode_step_s,
                    },
                    "slice_quality": [
                        {"slice_name": s.slice_name, "score": s.score}
                        for s in m.slice_quality
                    ],
                }
                for m in measurements
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    # Rehearsal mode: bind the cohort before allocating the model.
    selection = json.loads((args.selection_dir / "selection.json").read_text(encoding="utf-8"))
    pages = selection.get(args.split, [])
    if not pages:
        print(f"No pages found for split {args.split!r} in selection.json", flush=True)
        return 1
    selection_page_count = len(pages)
    arms = list(args.arms)
    gold_index = gold_token_index(args.selection_dir / "gold", args.split)
    release_artifacts = release_artifact_index(args.selection_dir / "release-v14-r6")
    pages_dir = args.selection_dir / "pages"
    ideal_map_exclusions = []
    ideal_maps = {}
    if args.ideal_map_valid_only:
        if "pre_region" not in arms:
            raise ValueError("--ideal-map-valid-only requires the pre_region arm")
        pages, ideal_map_exclusions, ideal_maps = select_ideal_map_valid_pages(
            pages,
            pages_dir=pages_dir,
            geometry_store=args.geometry_store,
            release_artifacts=release_artifacts,
        )
        print(
            f"[{REHEARSAL_TAG}] PRE evidence eligibility: "
            f"{len(pages)}/{selection_page_count} pages; "
            f"excluded={len(ideal_map_exclusions)}",
            flush=True,
        )
    if args.limit is not None:
        pages = pages[:args.limit]
    if not pages:
        print("No eligible pages remain after evidence preflight", flush=True)
        return 1

    print(f"[{REHEARSAL_TAG}] loading runtime from {args.bundle}...", flush=True)
    if not args.skip_gpu_wait:
        _refuse_if_gpu_busy(args.gpu_index)
    runtime = load_runtime(
        bundle_dir=args.bundle,
        device=args.device,
        decoder_precision=args.precision,
        kv_cache_dtype=args.kv_cache,
    )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for page_idx, page_info in enumerate(pages):
        bundle_id = page_info["bundle_id"]
        png_ref = page_info["png_ref"]
        png_path = pages_dir / png_ref
        if not png_path.exists():
            print(
                f"  [{page_idx + 1}/{len(pages)}] {bundle_id[:20]}... PNG not found, skip",
                flush=True,
            )
            continue

        selected_page_index = page_info.get("page_index", 0)
        gold = gold_index.get((bundle_id, selected_page_index))
        print(f"  [{page_idx + 1}/{len(pages)}] {bundle_id[:20]}... ", end="", flush=True)

        if "bypass" in arms:
            result = measure_bypass(runtime, png_path, gold, label="bypass")
            result.bundle_id = bundle_id
            result.split = args.split
            result.page_index = selected_page_index
            all_results.append(result_to_dict(result))
            print(f"bypass: {result.wall_s:.1f}s tok={result.visual_tokens} ", end="", flush=True)

        if "kv_sensitivity" in arms:
            kv_results = measure_kv_sensitivity(
                runtime, png_path, args.kv_ratios, label_prefix="bypass"
            )
            for kr in kv_results:
                kr.bundle_id = bundle_id
                kr.split = args.split
                kr.page_index = selected_page_index
                all_results.append(result_to_dict(kr))
            print(f"kv({len(kv_results)} ratios) ", end="", flush=True)

        if "post_encoder" in arms:
            post_results = measure_post_encoder_uniform(
                runtime, png_path, gold, args.post_ratios
            )
            for post_result in post_results:
                post_result.bundle_id = bundle_id
                post_result.split = args.split
                post_result.page_index = selected_page_index
                all_results.append(result_to_dict(post_result))
            print(f"post({len(post_results)} ratios) ", end="", flush=True)

        if "pre_region" in arms:
            try:
                map_key = (bundle_id, selected_page_index)
                if map_key in ideal_maps:
                    proposals, provenance = ideal_maps[map_key]
                else:
                    release_row = release_artifacts.get(bundle_id) or {}
                    target_ref = (release_row.get("target_bundle") or {}).get("ref")
                    if target_ref is None:
                        raise IdealRegionMapUnavailable(
                            f"{bundle_id}: release artifact index has no target_bundle ref"
                        )
                    model_input = build_model_safe_input(png_path, 300)
                    plane = model_input.planes[0]
                    proposals, provenance = ideal_region_proposals(
                        geometry_store=args.geometry_store,
                        bundle_id=bundle_id,
                        page_index=selected_page_index,
                        png_sha256=page_info["png_sha256"],
                        expected_target_ref=target_ref,
                        page_width_px=plane.width_px,
                        page_height_px=plane.height_px,
                    )
                pre_results = measure_pre_region(
                    runtime, png_path, gold,
                    label_prefix="bypass", budgets=args.pre_region_budgets,
                    proposals=proposals, provenance=provenance,
                )
                for pr in pre_results:
                    pr.bundle_id = bundle_id
                    pr.split = args.split
                    pr.page_index = selected_page_index
                    all_results.append(result_to_dict(pr))
                print(f"pre({len(pre_results)} budgets) ", end="", flush=True)
            except IdealRegionMapUnavailable as exc:
                all_results.append(result_to_dict(failed_page_result(
                    arm="bypass_pre_ideal_map",
                    bundle_id=bundle_id,
                    split=args.split,
                    page_index=selected_page_index,
                    error=str(exc),
                )))
                print("pre(FAILED) ", end="", flush=True)

        print(flush=True)

    # Summary
    summary = compute_summary(all_results)
    phases = phase_statuses(arms, all_results, len(pages))
    md_report = generate_markdown_report(summary, len(pages), REHEARSAL_TAG, phases)

    report = {
        "tag": REHEARSAL_TAG,
        "bundle": str(args.bundle),
        "selection_dir": str(args.selection_dir),
        "geometry_store": str(args.geometry_store),
        "split": args.split,
        "selection_page_count": selection_page_count,
        "ideal_map_valid_only": args.ideal_map_valid_only,
        "ideal_map_exclusions": ideal_map_exclusions,
        "n_pages": len(pages),
        "arms": arms,
        "precision": args.precision,
        "kv_cache": args.kv_cache,
        "results": all_results,
        "summary": summary,
        "phase_status": phases,
        "markdown_report": md_report,
    }

    (out_dir / "vc0-rehearsal-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_dir / "vc0-rehearsal-report.md").write_text(md_report, encoding="utf-8")
    print(f"\nReport written to {out_dir / 'vc0-rehearsal-report.json'}", flush=True)
    print(md_report)
    return 1 if any(row.get("error") for row in all_results) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

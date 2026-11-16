"""v16 pilot: DEC-049 point 8 + DEC-056 point 3, on the on-the-fly ViT path.

Before the multi-day v16 run, on the SAME pages:

1. **Step-0 DeepStack check** (DEC-056 point 3, lesson "random projection
   destroys native signal"): the v15b checkpoint's weights, loaded strictly
   into two models that differ only in `deepstack` (DeepStack adds no
   parameter), scored teacher-forced on the same pages -- pooler-only vs
   pooler + DeepStack levels, both with the frozen ViT run on the fly.
   Per page: every loss family and the mean target log-prob. A material
   degradation stops the long run and returns the decision to the owner;
   the threshold is the owner's (`--max-mean-text-loss-increase`), absent
   means the verdict is `owner_review`, never an invented number.
2. **Optional feature parity**: on-the-fly `pooler_output` vs the v15b
   cache's stored fp16 features for the same pages (max abs, cosine).
3. **Short fine-tune with the on-the-fly ViT** through `run_pilot.run_pilot`
   (the existing DEC-049 harness): N steps, a checkpoint, a resume, worst-
   page forward+backward memory (ViT forward included), unbuffered log,
   fresh keys.
4. **s/step** read back from the trainer's own per-step log lines
   (`step_s`, `data_wait_s`), first step excluded (warm-up).

This module takes optimizer steps in (3). Its CLI therefore refuses to run
without `--owner-go <file>` naming an existing decision record; tests and
the CPU dry run call `run_v16_pilot` directly on tiny models.
"""

from __future__ import annotations

import dataclasses
import json
import statistics
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from unflatten.model import onfly_visual as ov
from unflatten.model import run_pilot, s1_train
from unflatten.model import s4_typed_loss as stl

PILOT_REPORT_SCHEMA = "unflatten.v16-pilot-report/2"


class V16PilotRefused(RuntimeError):
    """The pilot cannot run as asked (no owner go, mismatched inputs)."""


# ---------------------------------------------------------------------------
# 1. step 0: v15b weights without vs with DeepStack
# ---------------------------------------------------------------------------


def _score(
    model: Any,
    sample: Any,
    *,
    encoder: ov.OnTheFlyVisualEncoder,
    image_token_id: int,
    stage_policy: Mapping[str, Any],
    allowed_head_ids: Collection[str],
    device: torch.device | str,
    structural_table_region: str,
    cache_refs: Sequence[str | None] | None = None,
) -> dict[str, Any]:
    """Teacher-forced S4 loss families + mean target log-prob for one
    pixel-attached sample, exactly the `run_s4_training` step recipe."""
    batch = s1_train.collate_training_batch(
        [sample.base], ranges=model.ranges, image_token_id=image_token_id, device=device
    )
    batch = encoder.materialize(batch, cache_refs=cache_refs)
    typed_batch = stl.collate_typed([sample], allowed_head_ids=allowed_head_ids)
    region_batch = stl.collate_regions([sample], **stl.region_kwargs(structural_table_region))
    hidden, prefix_len = model.encode_hidden(batch.model_batch)
    logits = model.composite_head(hidden[:, prefix_len + batch.visual_steps :, :])
    total, report = stl.s4_total_loss(
        logits=logits,
        target_ids=batch.target_ids,
        loss_mask=batch.loss_mask,
        loss_family_ids=batch.loss_family_ids,
        model=model,
        hidden=hidden,
        prefix_len=prefix_len,
        visual_steps=batch.visual_steps,
        typed_batch=typed_batch,
        stage_policy=stage_policy,
        region_batch=region_batch,
        image_grid_thw=batch.model_batch.image_grid_thw,
    )
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    target_lp = log_probs.gather(-1, batch.target_ids.unsqueeze(-1)).squeeze(-1)
    scored = batch.loss_mask
    return {
        "total_loss": float(total.detach()),
        "families": {name: entry["mean"] for name, entry in report["families"].items()},
        "mean_target_logprob": (
            float(target_lp[scored].mean()) if bool(scored.any()) else None
        ),
        "scored_tokens": int(scored.sum()),
    }


def step0_deepstack_comparison(
    *,
    v15b_state_dict: Mapping[str, torch.Tensor] | None,
    build_model: Callable[[bool], Any],
    candidate: Any,
    samples: Sequence[Any],
    policy: ov.VisualInputPolicy,
    stage_policy: Mapping[str, Any],
    allowed_head_ids: Collection[str],
    device: torch.device | str = "cpu",
    structural_table_region: str = "abstain_not_visible",
    max_mean_text_loss_increase: float | None = None,
) -> dict[str, Any]:
    """The same v15b weights, strictly loaded into a plain and a DeepStack
    model (`build_model(deepstack)`), scored on the same pixel-attached
    samples. `policy` must name DeepStack taps; the plain arm uses the same
    policy without them. Returns per-page rows and a summary whose verdict
    is `pass` / `material_degradation` against the owner's threshold, or
    `owner_review` when no threshold was given.

    `v15b_state_dict=None` (fresh S4, DEC-058 clean pretrained start): the
    plain model is built once and ITS weights are loaded into both arms, so
    the comparison is the native Qwen3-VL path with vs without DeepStack on
    identical weights ("random projection destroys native signal")."""
    if not policy.deepstack:
        raise V16PilotRefused("the step-0 check needs a DeepStack policy (taps to compare)")
    init = "v15b" if v15b_state_dict is not None else "clean_pretrained"
    if v15b_state_dict is None:
        fresh = build_model(False).state_dict()
        v15b_state_dict = {k: v.detach().clone() for k, v in fresh.items()}
    plain_policy = dataclasses.replace(policy, deepstack_visual_indexes=())
    arms = {}
    for name, deepstack, arm_policy in (
        ("pooler_only", False, plain_policy),
        ("deepstack", True, policy),
    ):
        model = build_model(deepstack)
        model.load_state_dict(dict(v15b_state_dict), strict=True)
        model.eval()
        arms[name] = (
            model,
            ov.OnTheFlyVisualEncoder(candidate, policy=arm_policy, device=device),
        )
    rows = []
    with torch.no_grad():
        for sample in samples:
            row: dict[str, Any] = {
                "document_bundle_id": sample.base.document_bundle_id,
                "page_index": sample.base.page_index,
            }
            for name, (model, encoder) in arms.items():
                row[name] = _score(
                    model,
                    sample,
                    encoder=encoder,
                    image_token_id=candidate.image_token_id,
                    stage_policy=stage_policy,
                    allowed_head_ids=allowed_head_ids,
                    device=device,
                    structural_table_region=structural_table_region,
                )
            rows.append(row)
    return {
        "init": init,
        "rows": rows,
        "summary": summarize_step0(rows, max_mean_text_loss_increase),
    }


def summarize_step0(
    rows: Sequence[Mapping[str, Any]], max_mean_text_loss_increase: float | None
) -> dict[str, Any]:
    """Mean per-family deltas (DeepStack - pooler-only; positive = worse)
    and the verdict. Pages without a family's value do not enter its mean."""
    families = sorted(
        {f for row in rows for f in row["pooler_only"]["families"]}
        | {f for row in rows for f in row["deepstack"]["families"]}
    )
    deltas: dict[str, float | None] = {}
    for family in families:
        pairs = [
            (row["deepstack"]["families"].get(family), row["pooler_only"]["families"].get(family))
            for row in rows
        ]
        values = [a - b for a, b in pairs if a is not None and b is not None]
        deltas[family] = statistics.fmean(values) if values else None
    total_delta = statistics.fmean(
        row["deepstack"]["total_loss"] - row["pooler_only"]["total_loss"] for row in rows
    )
    text_delta = deltas.get("L_text")
    pages_worse = sum(
        1
        for row in rows
        if row["deepstack"]["families"].get("L_text") is not None
        and row["pooler_only"]["families"].get("L_text") is not None
        and row["deepstack"]["families"]["L_text"] > row["pooler_only"]["families"]["L_text"]
    )
    if max_mean_text_loss_increase is None:
        verdict = "owner_review"
    elif text_delta is None:
        verdict = "insufficient_evidence"
    elif text_delta > max_mean_text_loss_increase:
        verdict = "material_degradation"
    else:
        verdict = "pass"
    return {
        "pages": len(rows),
        "mean_delta_total_loss": total_delta,
        "mean_delta_by_family": deltas,
        "pages_text_loss_worse": pages_worse,
        "max_mean_text_loss_increase": max_mean_text_loss_increase,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 2. on-the-fly pooler_output vs the v15b cache's stored features
# ---------------------------------------------------------------------------


def feature_parity(
    encoder: ov.OnTheFlyVisualEncoder, pairs: Sequence[tuple[Any, torch.Tensor]]
) -> dict[str, Any]:
    """`pairs` = `(pixel-attached sample, stored fp16 features)` for the same
    page. On-the-fly `pooler_output` against the stored tensor: max abs
    difference and mean per-token cosine similarity per page."""
    rows = []
    for sample, stored in pairs:
        pooled, _levels = encoder.encode(sample.base.pixel_values, sample.base.grid_thw)
        ours = pooled.float().cpu()
        ref = stored.float().cpu()
        if ours.shape != ref.shape:
            rows.append(
                {
                    "document_bundle_id": sample.base.document_bundle_id,
                    "page_index": sample.base.page_index,
                    "shape_mismatch": [list(ours.shape), list(ref.shape)],
                }
            )
            continue
        rows.append(
            {
                "document_bundle_id": sample.base.document_bundle_id,
                "page_index": sample.base.page_index,
                "max_abs_diff": float((ours - ref).abs().max()),
                "mean_cosine": float(
                    torch.nn.functional.cosine_similarity(ours, ref, dim=-1).mean()
                ),
            }
        )
    return {"rows": rows}


# ---------------------------------------------------------------------------
# 4. s/step from the trainer's own step log
# ---------------------------------------------------------------------------


def step_timing(log_paths: Sequence[Path]) -> dict[str, Any]:
    """Median / p90 of `step_s` and `data_wait_s` over every per-step JSON
    line `run_s4_training` printed into `log_paths`, the first step of
    each log excluded (model warm-up, first ViT call)."""
    step_s: list[float] = []
    wait_s: list[float] = []
    for path in log_paths:
        if not Path(path).exists():
            continue
        first = True
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step_s" not in record:
                continue
            if first:
                first = False
                continue
            step_s.append(float(record["step_s"]))
            wait_s.append(float(record["data_wait_s"]))

    def _stats(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"n": 0, "median": None, "p90": None}
        ordered = sorted(values)
        return {
            "n": len(values),
            "median": statistics.median(ordered),
            "p90": ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))],
        }

    return {"step_s": _stats(step_s), "data_wait_s": _stats(wait_s)}


# ---------------------------------------------------------------------------
# 5. Fresh S4 speed knobs (owner 2026-09-30): ViT-only time, overlap A/B,
#    fp8 write-through cache precision. Each is decided by these numbers.
# ---------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_reset(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak(device: torch.device) -> int | None:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None


def _ms_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median_ms": None, "p90_ms": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))],
    }


def vit_timing(
    encoder: ov.OnTheFlyVisualEncoder, samples: Sequence[Any], *, repeats: int = 3
) -> dict[str, Any]:
    """ViT-only milliseconds per page on the encoder's device: every sample
    `repeats` times one page per tower call, then pairs through one call
    (`batch_pages`). Also: distinct page grids (a CUDA graph would need one
    capture per grid), tokens per page, compute dtype, the SDPA backends
    PyTorch may pick, and peak memory of the tower alone."""
    device = encoder.device
    batched = ov.OnTheFlyVisualEncoder(
        encoder.candidate, policy=encoder.policy, device=device, batch_pages=True
    )
    rows = [(s.base.pixel_values, s.base.grid_thw) for s in samples]
    encoder.encode(*rows[0])  # warm-up: kernels, allocator
    _sync(device)
    _peak_reset(device)
    single: list[float] = []
    for _ in range(repeats):
        for pixel_values, grid in rows:
            started = time.perf_counter()
            encoder.encode(pixel_values, grid)
            _sync(device)
            single.append(1000.0 * (time.perf_counter() - started))
    pairs: list[float] = []
    for _ in range(repeats):
        for index in range(0, len(rows) - 1, 2):
            started = time.perf_counter()
            batched.encode_rows(rows[index : index + 2])
            _sync(device)
            pairs.append(1000.0 * (time.perf_counter() - started) / 2)
    grids = {tuple(int(v) for v in grid.reshape(3).tolist()) for _pv, grid in rows}
    sdpa = None
    if device.type == "cuda":
        sdpa = {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
        }
    pair_stats = _ms_stats(pairs)
    return {
        "pages": len(rows),
        "single": _ms_stats(single),
        "batched_2": {
            "n": pair_stats["n"],
            "median_ms_per_page": pair_stats["median_ms"],
            "p90_ms_per_page": pair_stats["p90_ms"],
        },
        "grid_buckets": len(grids),
        "tokens_per_page": sorted(encoder.policy.visual_tokens_for(g) for g in grids),
        "compute_dtype": encoder.policy.compute_dtype,
        "sdpa_backends": sdpa,
        "peak_memory_bytes": _peak(device),
    }


def overlap_ab(
    run_arm: Callable[[bool, Path], Any], *, log_dir: Path, device: torch.device | str
) -> dict[str, Any]:
    """The same short training run twice -- `visual_prefetch` off, then on
    -- each into its own run dir with its stdout (the trainer's per-step
    JSON lines) captured, timed by `step_timing` (first step excluded) and
    with its own peak memory. `run_arm(prefetch, run_dir)` runs the steps."""
    import contextlib

    device = torch.device(device)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for name, prefetch in (("plain", False), ("overlap", True)):
        run_dir = log_dir / f"ab-{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"ab-{name}.log"
        _peak_reset(device)
        with open(log_path, "w") as log, contextlib.redirect_stdout(log):
            run_arm(prefetch, run_dir)
        report[name] = {
            "timing": step_timing([log_path]),
            "peak_memory_bytes": _peak(device),
        }
    plain = report["plain"]["timing"]["step_s"]["median"]
    overlap = report["overlap"]["timing"]["step_s"]["median"]
    report["median_step_speedup"] = plain / overlap if plain and overlap else None
    return report


def _feature_error(fresh: torch.Tensor, cached: torch.Tensor) -> dict[str, float]:
    a, b = fresh.float().cpu(), cached.float().cpu()
    return {
        "max_abs": float((a - b).abs().max()),
        "max_rel_to_row_absmax": float(
            ((a - b).abs() / a.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)).max()
        ),
        "mean_cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()),
    }


def feature_cache_precision(
    model: Any,
    samples: Sequence[Any],
    *,
    encoder: ov.OnTheFlyVisualEncoder,
    stage_policy: Mapping[str, Any],
    allowed_head_ids: Collection[str],
    cache_dir: Path,
    cache_pages: int,
    device: torch.device | str = "cpu",
    structural_table_region: str = "abstain_not_visible",
    max_mean_text_loss_increase: float | None = None,
) -> dict[str, Any]:
    """The epoch-1 write-through fallback's cost: per page, teacher-forced
    losses on fresh features vs the fp8 round trip the cache hands out
    (`ViTFeatureCache` under `cache_dir`, written on this first touch),
    plus the feature error of the pooler output and every DeepStack level.
    The summary carries the mean per-family loss delta (fp8 - fresh), the
    verdict against the owner's threshold (`owner_review` without one) and
    the disk estimate for `cache_pages` pages at the largest page seen."""
    cache = ov.ViTFeatureCache(cache_dir, policy=encoder.policy)
    cached_encoder = ov.OnTheFlyVisualEncoder(
        encoder.candidate, policy=encoder.policy, device=encoder.device, feature_cache=cache
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    tokens_max = width = 0
    with torch.no_grad():
        for sample in samples:
            ref = str(sample.base.page_image["sha256"])
            common = dict(
                image_token_id=encoder.candidate.image_token_id, stage_policy=stage_policy,
                allowed_head_ids=allowed_head_ids, device=device,
                structural_table_region=structural_table_region,
            )
            fresh = _score(model, sample, encoder=encoder, **common)
            fp8 = _score(model, sample, encoder=cached_encoder, cache_refs=[ref], **common)
            pooled, levels = encoder.encode(sample.base.pixel_values, sample.base.grid_thw)
            back_pooled, back_levels = cache.round_trip(pooled, levels, dtype=pooled.dtype)
            tokens_max = max(tokens_max, int(pooled.shape[0]))
            width = int(pooled.shape[-1])
            rows.append(
                {
                    "document_bundle_id": sample.base.document_bundle_id,
                    "page_index": sample.base.page_index,
                    "fresh": fresh,
                    "fp8": fp8,
                    "feature_error": {
                        "pooler": _feature_error(pooled, back_pooled),
                        "levels": [
                            _feature_error(a, b) for a, b in zip(levels, back_levels, strict=True)
                        ],
                    },
                }
            )
    families = sorted({f for row in rows for f in row["fresh"]["families"]})
    deltas: dict[str, float | None] = {}
    for family in families:
        values = [
            row["fp8"]["families"][family] - row["fresh"]["families"][family]
            for row in rows
            if row["fp8"]["families"].get(family) is not None
            and row["fresh"]["families"].get(family) is not None
        ]
        deltas[family] = statistics.fmean(values) if values else None
    text_delta = deltas.get("L_text")
    if max_mean_text_loss_increase is None:
        verdict = "owner_review"
    elif text_delta is None:
        verdict = "insufficient_evidence"
    elif text_delta > max_mean_text_loss_increase:
        verdict = "material_degradation"
    else:
        verdict = "pass"
    bytes_per_page = ov.fp8_cache_bytes_per_page(
        tokens=tokens_max, width=width, levels=cache.levels
    )
    return {
        "rows": rows,
        "summary": {
            "pages": len(rows),
            "mean_delta_total_loss": (
                statistics.fmean(r["fp8"]["total_loss"] - r["fresh"]["total_loss"] for r in rows)
                if rows
                else None
            ),
            "mean_delta_by_family": deltas,
            "max_mean_text_loss_increase": max_mean_text_loss_increase,
            "verdict": verdict,
            "format": ov.VIT_FEATURE_CACHE_FORMAT,
            "tokens_max": tokens_max,
            "width": width,
            "bytes_per_page_max": bytes_per_page,
            "cache_pages": int(cache_pages),
            "disk_estimate_bytes": bytes_per_page * int(cache_pages),
        },
    }


# ---------------------------------------------------------------------------
# The whole pilot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V16PilotInputs:
    """Everything `run_v16_pilot` needs, already built by the caller (the
    CLI on prod, a test on tiny models)."""

    run_dir: Path
    cache_dir: Path
    candidate: Any
    build_model: Callable[[bool], Any]
    v15b_state_dict: Mapping[str, torch.Tensor] | None
    step0_samples: Sequence[Any]
    policy: ov.VisualInputPolicy
    stage_policy: Mapping[str, Any]
    allowed_head_ids: Collection[str]
    pilot_config_factory: Callable[[Path], run_pilot.PilotConfig]
    device: str = "cpu"
    max_mean_text_loss_increase: float | None = None
    parity_pairs: Sequence[tuple[Any, torch.Tensor]] = ()
    structural_table_region: str = "abstain_not_visible"
    #: Fresh S4 speed sections (0 / None = not run): ViT-only timing repeats
    #: over the step-0 pages, the overlap A/B (`overlap_ab`'s `run_arm`), and
    #: the fp8 cache precision check on the first N step-0 pages with the
    #: disk estimate for `fp8_cache_pages` pages.
    vit_timing_repeats: int = 0
    overlap_ab_run: Callable[[bool, Path], Any] | None = None
    fp8_precision_pages: int = 0
    fp8_cache_pages: int = 0
    max_fp8_text_loss_increase: float | None = None


def run_v16_pilot(inputs: V16PilotInputs) -> dict[str, Any]:
    """Step-0 check, optional feature parity, the DEC-049 harness on the
    on-the-fly path, and s/step. The overall verdict is `PASS` only if the
    harness passed AND the step-0 verdict is `pass`; `owner_review` when no
    threshold was given; `FAIL` otherwise."""
    step0 = step0_deepstack_comparison(
        v15b_state_dict=inputs.v15b_state_dict,
        build_model=inputs.build_model,
        candidate=inputs.candidate,
        samples=inputs.step0_samples,
        policy=inputs.policy,
        stage_policy=inputs.stage_policy,
        allowed_head_ids=inputs.allowed_head_ids,
        device=inputs.device,
        structural_table_region=inputs.structural_table_region,
        max_mean_text_loss_increase=inputs.max_mean_text_loss_increase,
    )
    parity = None
    if inputs.parity_pairs:
        parity = feature_parity(
            ov.OnTheFlyVisualEncoder(
                inputs.candidate,
                policy=dataclasses.replace(inputs.policy, deepstack_visual_indexes=()),
                device=inputs.device,
            ),
            inputs.parity_pairs,
        )
    encoder = ov.OnTheFlyVisualEncoder(inputs.candidate, policy=inputs.policy, device=inputs.device)
    vit = (
        vit_timing(encoder, inputs.step0_samples, repeats=inputs.vit_timing_repeats)
        if inputs.vit_timing_repeats
        else None
    )
    fp8 = None
    if inputs.fp8_precision_pages:
        fp8_model = inputs.build_model(True)
        if inputs.v15b_state_dict is not None:
            fp8_model.load_state_dict(dict(inputs.v15b_state_dict), strict=True)
        fp8 = feature_cache_precision(
            fp8_model,
            inputs.step0_samples[: inputs.fp8_precision_pages],
            encoder=encoder,
            stage_policy=inputs.stage_policy,
            allowed_head_ids=inputs.allowed_head_ids,
            cache_dir=inputs.run_dir / "fp8-precision-cache",
            cache_pages=inputs.fp8_cache_pages,
            device=inputs.device,
            structural_table_region=inputs.structural_table_region,
            max_mean_text_loss_increase=inputs.max_fp8_text_loss_increase,
        )
        del fp8_model
    log_dir = inputs.run_dir / "pilot-logs"
    harness = run_pilot.run_pilot(inputs.pilot_config_factory(log_dir))
    timing = step_timing([log_dir / "pilot_run.log", log_dir / "resume_run.log"])
    overlap = (
        overlap_ab(
            inputs.overlap_ab_run, log_dir=inputs.run_dir / "overlap-ab", device=inputs.device
        )
        if inputs.overlap_ab_run is not None
        else None
    )
    step0_verdict = step0["summary"]["verdict"]
    if harness.verdict != "PASS" or step0_verdict in ("material_degradation",):
        verdict = "FAIL"
    elif step0_verdict == "pass":
        verdict = "PASS"
    else:
        verdict = "owner_review"
    return {
        "schema": PILOT_REPORT_SCHEMA,
        "verdict": verdict,
        "visual_input_policy_hash": inputs.policy.policy_hash(),
        "visual_feature_tap": inputs.policy.visual_feature_tap(),
        "step0_deepstack": step0,
        "feature_parity": parity,
        "harness": harness.to_json(),
        "timing": timing,
        "vit_timing": vit,
        "overlap_ab": overlap,
        "fp8_cache_precision": fp8,
    }


def build_onfly_augmentation(
    *, seed: int | None, policy_path: Path | None
) -> ov.OnTheFlyAugmentation | None:
    """The fine-tune arm's augmentation: none without a seed, the shipped v1 default
    with a seed alone, the policy file's (v1 or v2) with `--onfly-augmentation-policy`."""
    if seed is None:
        if policy_path is not None:
            raise V16PilotRefused("--onfly-augmentation-policy needs --onfly-augmentation-seed")
        return None
    if policy_path is None:
        return ov.OnTheFlyAugmentation(seed=seed)
    from unflatten.dataset.augment import policy as aug_policy

    return ov.OnTheFlyAugmentation(policy=aug_policy.load_policy(policy_path), seed=seed)


def require_owner_go(path: Path | None) -> None:
    """The pilot takes optimizer steps on the production GPU: no run
    without a named, existing owner decision record."""
    if path is None or not Path(path).is_file():
        raise V16PilotRefused(
            "the v16 pilot takes optimizer steps; it needs --owner-go <existing "
            "decision record> (the owner's go for the prod pilot)"
        )


# ---------------------------------------------------------------------------
# CLI (prod, owner's go only)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs the 4B pin + GPU
    """Build the real inputs and run the pilot. Everything the pilot reads
    is named on the command line; nothing is inferred from a run dir it did
    not write itself. `--init v15b` (the original v16 plan): the fine-tune
    arm warm-starts the DeepStack model from the v15b weights; `--init
    clean_pretrained` (DEC-058): it starts from the pretrained weights like
    the fresh run. Its checkpoints are stamped with the DeepStack tap and
    are pilot artefacts, never a production parent. The overlap A/B runs on
    the fine-tune arm's model after the harness (timing only)."""
    import argparse
    import hashlib

    from transformers import AutoImageProcessor

    from unflatten.model import composite_vocab as cv
    from unflatten.model import pilot_hooks_s4 as ph
    from unflatten.model import s4_train
    from unflatten.model import stage_policy as spolicy
    from unflatten.model.candidate_loader import (
        QWEN_CANDIDATE_MODEL_ID,
        QWEN_CANDIDATE_REVISION,
        load_qwen_candidate,
    )
    from unflatten.model.candidate_model import QwenDenseCandidateModel
    from unflatten.model.exact_text_cache import CacheExpectation

    parser = argparse.ArgumentParser(prog="python -m unflatten.model.v16_pilot")
    parser.add_argument("--owner-go", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="page-image (7.0.0) cache")
    parser.add_argument("--expectation", type=Path, required=True)
    parser.add_argument("--page-image-store", type=Path, required=True)
    parser.add_argument("--visual-input-policy", type=Path, required=True)
    parser.add_argument(
        "--init", choices=("v15b", "clean_pretrained"), default="v15b",
        help="fresh S4 (DEC-058): clean_pretrained = step 0 on the pretrained weights and a "
        "fine-tune arm from them, exactly like the run; --v15b-checkpoint then optional",
    )
    parser.add_argument("--v15b-checkpoint", type=Path, default=None)
    parser.add_argument("--v15b-checkpoint-sha256", default=None)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="prefetch workers of the fine-tune arm (use the launcher's value: s/step and "
        "data_wait are only meaningful with the real worker count)",
    )
    parser.add_argument(
        "--onfly-augmentation-seed", type=int, default=None,
        help="the run's DEC-040 augmentation (~20 %% views, ~3.9 s CPU each) in the fine-tune arm",
    )
    parser.add_argument(
        "--onfly-augmentation-policy", type=Path, default=None,
        help="augmentation policy file (schema 1.0.0 or 2.0.0) for --onfly-augmentation-seed; "
        "absent = the shipped v1 default",
    )
    parser.add_argument("--vit-prefetch-overlap", action="store_true")
    parser.add_argument(
        "--overlap-ab-steps", type=int, default=0,
        help="> 0: also run this many steps with overlap off, then on (s/step + memory each)",
    )
    parser.add_argument("--vit-timing-repeats", type=int, default=0)
    parser.add_argument("--fp8-precision-pages", type=int, default=0)
    parser.add_argument("--max-fp8-text-loss-increase", type=float, default=None)
    parser.add_argument("--step0-pages", type=int, default=50)
    parser.add_argument("--pilot-steps", type=int, default=20)
    parser.add_argument("--resume-steps", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--memory-budget-bytes", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lr-reduced", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-mean-text-loss-increase", type=float, default=None)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    require_owner_go(args.owner_go)

    v15b_state: dict[str, Any] | None = None
    if args.init == "v15b" or args.v15b_checkpoint is not None:
        if args.v15b_checkpoint is None or args.v15b_checkpoint_sha256 is None:
            raise V16PilotRefused("--init v15b needs --v15b-checkpoint and its sha256 pin")
        digest = hashlib.sha256(args.v15b_checkpoint.read_bytes()).hexdigest()
        if f"sha256:{digest}" != args.v15b_checkpoint_sha256:
            raise V16PilotRefused("the v15b checkpoint does not match its sha256 pin")
        v15b_state = torch.load(args.v15b_checkpoint, map_location="cpu", weights_only=False)
        ov.require_checkpoint_visual_feature_tap(
            v15b_state, expected=ov.POOLER_OUTPUT_TAP, what="--v15b-checkpoint"
        )
    warm = v15b_state["model_state_dict"] if args.init == "v15b" and v15b_state else None
    companion_hash = args.v15b_checkpoint_sha256 or "sha256:" + "0" * 64
    policy = ov.VisualInputPolicy.from_payload(json.loads(args.visual_input_policy.read_text()))
    expectation = CacheExpectation.from_payload(json.loads(args.expectation.read_text()))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "exact_text_cache_expectation.json").write_text(
        json.dumps(expectation.payload())
    )
    candidate = load_qwen_candidate(device=args.device)
    processor = AutoImageProcessor.from_pretrained(
        QWEN_CANDIDATE_MODEL_ID, revision=QWEN_CANDIDATE_REVISION
    )
    stage_policy = spolicy.load_stage_policy(s4_train._POLICIES_DIR / "s4_stage_policy.json")
    allowed = s4_train.l_typed_head_ids(stage_policy)

    def build_model(deepstack: bool) -> QwenDenseCandidateModel:
        extra = {"deepstack": True, "image_token_id": candidate.image_token_id}
        model = QwenDenseCandidateModel(
            decode=candidate.decode,
            inject_visual=candidate.inject_visual,
            visual_dim=candidate.visual_hidden_size,
            d_model=candidate.hidden_size,
            ranges=cv.packed_ranges(candidate.vocab_size),
            qwen_embedding_weight=candidate.embed_tokens.weight,
            qwen_lm_head_weight=candidate.lm_head_weight,
            **(extra if deepstack else {}),
        )
        return model.to(args.device)

    typed = stl.S4TrainingDataset(
        s1_train.ExactTextTrainingDataset(
            args.cache, gradient_bearing=True, expectation=expectation
        )
    )
    pages = list(range(min(args.step0_pages, len(typed))))
    samples = [
        ov.attach_pixels(
            typed[i], store_dir=args.page_image_store, processor=processor, policy=policy
        )
        for i in pages
    ]
    plan_order = [pages[i % len(pages)] for i in range(args.pilot_steps + args.resume_steps)]
    keys = expectation.target_keys()

    augmentation = build_onfly_augmentation(
        seed=args.onfly_augmentation_seed, policy_path=args.onfly_augmentation_policy
    )
    shared: dict[str, Any] = {}

    def trainer_for(run_dir: Path, *, prefetch: bool, order: Sequence[int]) -> Any:
        dataset = ov.OnTheFlyDrawDataset(
            typed, plan_order=order, batch_size=1, store_dir=args.page_image_store,
            processor=processor, policy=policy, augmentation=augmentation,
        )
        return ph.build_trainer_entrypoint(
            model=shared["model"], optimizer=shared["optimizer"], dataset=dataset,
            image_token_id=candidate.image_token_id,
            plan_pages=[keys[i] for i in order],
            plan_lineages=[f"pilot:{keys[i][0]}" for i in order],
            plan_cells=[("train", "pilot", "structure_v0", "pilot")] * len(order),
            plan_order=list(order), companion_hash=companion_hash,
            s4_run_id="s4run_v16_pilot", stage_policy=stage_policy, allowed_head_ids=allowed,
            grad_clip=1.0, run_dir=run_dir,
            integrity={"pilot_init": args.init, "v15b_checkpoint": str(companion_hash)},
            visual_encoder=shared["encoder"], device=args.device,
            num_workers=args.num_workers, visual_prefetch=prefetch,
        )

    def pilot_config_factory(log_dir: Path) -> run_pilot.PilotConfig:
        model = build_model(policy.deepstack)
        if warm is not None:
            model.load_state_dict(warm, strict=True)
        s4_train.unfreeze_and_refreeze_typed_heads(model, [])
        optimizer = s4_train.build_s4_optimizer(
            model, lr_standard=args.lr, lr_reduced=args.lr_reduced, algorithm="adamw",
            run_class="promotion",
        )
        encoder = ov.OnTheFlyVisualEncoder(candidate, policy=policy, device=args.device)
        shared.update(model=model, optimizer=optimizer, encoder=encoder)
        trainer_main = trainer_for(
            args.run_dir, prefetch=args.vit_prefetch_overlap, order=plan_order
        )
        return run_pilot.PilotConfig(
            runner=run_pilot.in_process_runner(trainer_main),
            base_argv=("--out", str(args.run_dir / "pilot-checkpoint.pt")),
            run_dir=args.run_dir, cache_dir=args.cache, log_dir=log_dir,
            pilot_steps=args.pilot_steps, resume_steps=args.resume_steps,
            checkpoint_every=args.checkpoint_every,
            worst_page_and_batch=ph.build_worst_page_and_batch(
                run_dir=args.run_dir, model=model, image_token_id=candidate.image_token_id,
                allowed_head_ids=allowed, device=args.device, visual_encoder=encoder,
                page_image_store=args.page_image_store, processor=processor,
            ),
            forward_backward=ph.build_forward_backward(
                model=model, optimizer=optimizer, stage_policy=stage_policy,
                visual_encoder=encoder,
            ),
            memory_budget_bytes=args.memory_budget_bytes,
        )

    ab_order = [pages[i % len(pages)] for i in range(args.overlap_ab_steps)]

    def overlap_arm(prefetch: bool, run_dir: Path) -> None:
        trainer_for(run_dir, prefetch=prefetch, order=ab_order)(
            ["--steps", str(args.overlap_ab_steps)]
        )

    report = run_v16_pilot(
        V16PilotInputs(
            run_dir=args.run_dir, cache_dir=args.cache, candidate=candidate,
            build_model=build_model, v15b_state_dict=warm,
            step0_samples=samples, policy=policy, stage_policy=stage_policy,
            allowed_head_ids=allowed, pilot_config_factory=pilot_config_factory,
            device=args.device, max_mean_text_loss_increase=args.max_mean_text_loss_increase,
            vit_timing_repeats=args.vit_timing_repeats,
            overlap_ab_run=overlap_arm if args.overlap_ab_steps > 0 else None,
            fp8_precision_pages=args.fp8_precision_pages,
            fp8_cache_pages=len(keys),
            max_fp8_text_loss_increase=args.max_fp8_text_loss_increase,
        )
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"verdict": report["verdict"], "report": str(args.report)}))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

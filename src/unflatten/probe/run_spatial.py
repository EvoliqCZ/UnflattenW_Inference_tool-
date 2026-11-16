"""Stage-0 spatial probe run.

Streams pages through a frozen backbone, labels every visual token from
the Observation Sidecar geometry, and fits one linear head per label on
the tokens themselves — no pooling.

Verdict is AUROC against chance with a significance threshold, the same
rule the pooled probe uses. The unit of evidence is the PAGE, not the
token: a page's thousands of tokens are strongly correlated, so the
sigma is bootstrapped over pages rather than derived from token counts
(see `evaluate_spatial`).

A label with no eval geometry is INCONCLUSIVE, never a PASS on the
remaining one: two labels of which one is unscored is not a verdict.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import torch

from unflatten.dataset.items import build_item
from unflatten.dataset.samples import iter_page_samples
from unflatten.probe.heads import predict, standardize
from unflatten.probe.spatial import (
    SPATIAL_LABELS,
    MergeSizeError,
    RidgeAccumulator,
    evaluate_spatial,
    page_spatial_labels,
)


def _pages(backbone: Any, release_dir: Path, store_dir: Path, split: str,
           task_kind: str, limit: int, progress_every: int = 100,
           use_multiscale: bool = False, at_merged: bool = False,
           require_privacy_review: bool = False):
    """Yield (token features, per-label targets) for pages with geometry.

    `require_privacy_review` is the RFC005 §9.2 admission boundary: off
    by default because this is a diagnostic, but askable (it previously
    was not).
    """
    from PIL import Image

    done = 0
    for sample in iter_page_samples(
        release_dir,
        store_dir,
        split,
        task_kind,
        require_privacy_review=require_privacy_review,
    ):
        if done >= limit:
            break
        item = build_item(store_dir, sample, with_geometry=True)
        image = Image.open(io.BytesIO(item.image_bytes))
        image.load()
        features = backbone.extract(image)
        # Kept on the accelerator: at 4 M pixels a page is ~92 MB of
        # features, and round-tripping it to CPU to standardise made the
        # probe CPU-bound with the GPU idle at 1 %.
        rows = (
            features.dense(at_merged=at_merged, include_levels=use_multiscale)
            if use_multiscale or at_merged
            else features.visual_features
        ).float()
        try:
            labels = page_spatial_labels(
                item, sample.page_index, features.grid_thw, rows.shape[0],
                task_kind=task_kind,
            )
        except MergeSizeError:
            # a tower whose row count matches no merge of its reported grid
            # cannot be aligned to page geometry; skip rather than guess.
            # Typed, because the other ValueErrors here — a multiscale level
            # that does not tile the grid, a task kind without full geometry
            # coverage — say the run is wrong, not the page.
            continue
        if labels is None:
            continue
        done += 1
        if progress_every and done % progress_every == 0:
            print(f"[spatial] {done}/{limit} strán", flush=True)
        yield rows, labels


def run(backbone: Any, release_dir: Path, store_dir: Path, task_kind: str,
        train_pages: int, eval_pages: int, alpha: float = 10.0,
        use_multiscale: bool = False,
        at_merged: bool = False,
        require_privacy_review: bool = False) -> dict[str, Any]:
    dim = None
    accumulator: RidgeAccumulator | None = None
    shift = scale = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for rows, labels in _pages(backbone, release_dir, store_dir, "train",
                               task_kind, train_pages,
                               use_multiscale=use_multiscale,
                               at_merged=at_merged,
                               require_privacy_review=require_privacy_review):
        if dim is None:
            dim = rows.shape[1]
            # One Gram matrix for every label: XᵀX does not depend on y.
            accumulator = RidgeAccumulator(dim, len(SPATIAL_LABELS), device=device)
            # The first page's statistics are a PROVISIONAL scaler only:
            # they keep the accumulated values near unit scale (raw
            # features would be summed over millions of rows in float32),
            # and `solve_standardized` then recovers the exact statistics
            # of every page seen. Fitting on them directly is what made
            # the head depend on which document sorted first.
            _, shift, scale = standardize(rows)
            shift, scale = shift.detach(), scale.detach()
        z = (rows - shift.to(rows.device)) / scale.to(rows.device)
        accumulator.add(
            z,
            torch.stack([labels[n] for n in SPATIAL_LABELS], dim=1).to(rows.device),
        )
    if dim is None or accumulator is None:
        return {"verdict": "INCONCLUSIVE", "reason": "no page carried geometry"}

    solved, mean, std = accumulator.solve_standardized(alpha=alpha)
    weights = {name: solved[:, i] for i, name in enumerate(SPATIAL_LABELS)}
    scores: dict[str, list[torch.Tensor]] = {n: [] for n in SPATIAL_LABELS}
    truth: dict[str, list[torch.Tensor]] = {n: [] for n in SPATIAL_LABELS}
    for rows, labels in _pages(backbone, release_dir, store_dir, "val_iid",
                               task_kind, eval_pages,
                               use_multiscale=use_multiscale,
                               at_merged=at_merged,
                               require_privacy_review=require_privacy_review):
        provisional = (rows - shift.to(rows.device)) / scale.to(rows.device)
        z = (provisional - mean.to(rows.device)) / std.to(rows.device)
        for name in SPATIAL_LABELS:
            scores[name].append(
                predict(weights[name].to(rows.device), z).cpu()
            )
            truth[name].append(labels[name])

    results = {}
    for name in SPATIAL_LABELS:
        if not scores[name]:
            results[name] = {"auroc": None, "reason": "no eval geometry"}
            continue
        # NOT concatenated: the page grouping is the denominator of the
        # significance estimate (see `evaluate_spatial`).
        results[name] = evaluate_spatial(scores[name], truth[name])
    return {
        "verdict": spatial_verdict(results),
        "train_rows": accumulator.rows,
        "labels": results,
    }


def spatial_verdict(results: dict[str, dict[str, Any]]) -> str:
    """PASS only when EVERY declared spatial label was scored and clears
    3 sigma; INCONCLUSIVE when a label could not be scored at all.

    A label whose AUROC exists but whose sigma does not is unscored, not
    failed: an undefined significance is a measurement nobody made.
    """
    scored = [
        r
        for r in results.values()
        if r.get("auroc") is not None
        and r["auroc"] == r["auroc"]
        and r.get("sigma") is not None
    ]
    if len(scored) < len(SPATIAL_LABELS):
        return "INCONCLUSIVE"
    return "PASS" if all(r["sigma"] > 3.0 for r in scored) else "FAIL"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.probe.run_spatial", description=__doc__
    )
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("store_dir", type=Path)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--task-kind", default="layout_v0")
    parser.add_argument("--train-pages", type=int, default=300)
    parser.add_argument("--eval-pages", type=int, default=100)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tap",
        default=None,
        help="feature tap, for a backbone that exposes several "
             "(Granite Vision: 'siglip4' / 'projected8')",
    )
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="concatenate the backbone's multiscale levels at patch "
             "resolution; without it a DeepStack backbone is measured "
             "with its main advantage switched off",
    )
    parser.add_argument(
        "--merged-grid",
        action="store_true",
        help="pool the final layer onto the backbone's coarse merged-token "
             "grid: a quarter of the rows, a quarter of the cache. Works "
             "with or without --multiscale, so grid resolution and feature "
             "richness can be measured one at a time",
    )
    parser.add_argument(
        "--require-privacy-review",
        action="store_true",
        help="demand the RFC005 §9.2 training admission boundary (off by "
             "default — this probe is a diagnostic, not a training run)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from unflatten.probe.features import DEFAULT_BACKBONE, FrozenVisionBackbone

    model_id = args.backbone or DEFAULT_BACKBONE
    backbone = FrozenVisionBackbone(
        model_id, device=args.device, max_pixels=args.max_pixels, tap=args.tap
    )
    tap_tag = f" tap {backbone.tap}" if backbone.tap else ""
    print(
        f"[spatial] backbone {model_id}{tap_tag}, dim {backbone.feature_dim}",
        flush=True,
    )
    print(f"[spatial] pixel budget: {backbone.pixel_budget_applied or 'native'}"
          f"  measured: {backbone.pixel_budget_measured or 'not capped'}",
          flush=True)
    summary = run(backbone, args.release_dir, args.store_dir, args.task_kind,
                  args.train_pages, args.eval_pages,
                  use_multiscale=args.multiscale, at_merged=args.merged_grid,
                  require_privacy_review=args.require_privacy_review)
    summary["backbone"] = model_id
    summary["feature_tap"] = backbone.tap
    summary["multiscale"] = args.multiscale
    summary["merged_grid"] = args.merged_grid
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())

"""Run the scaling curve and report where more data stops helping.

Trains the same decoder from scratch on doubling prefixes of the cache
and evaluates each on the same held-out set, with the same seed,
architecture and optimizer at every point.

**Consecutive points do not differ only in how much data they saw.**
`--epochs` is fixed, so doubling the pages also doubles the optimizer
steps: data and compute move together, and the curve measures the pair.
The prefixes are also cache order, not a frozen lineage permutation, so
a point can be enriched in whatever the build happened to write first.
Read the curve as a lower bound with those two confounds named, not as a
data-scaling law; removing them is P2 work.

The number that matters is the gap to the null model, not the accuracy:
the grammar alone reaches ~0.95, so a run at 0.96 has learned almost
nothing from the pixels. `--random-features` on `train.py` measures that
floor for the same cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from unflatten.model.decoder import StructureDecoder
from unflatten.model.feature_cache import cache_manifest
from unflatten.model.scaling import subset_sizes, take_subset
from unflatten.model.train import (
    CacheDataset,
    evaluate,
    feature_channel_stats,
    require_shared_lineage,
    require_train_split,
    train_epoch,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.run_scaling", description=__doc__
    )
    parser.add_argument("train_cache", type=Path)
    parser.add_argument("val_cache", type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--smallest", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = cache_manifest(args.train_cache)
    require_train_split(manifest)
    require_shared_lineage(manifest, cache_manifest(args.val_cache))
    backbone = manifest["identity"]["backbone"]
    train_full = CacheDataset(args.train_cache, expect_backbone=backbone)
    val_set = CacheDataset(args.val_cache, expect_backbone=backbone)
    sizes = subset_sizes(len(train_full), smallest=args.smallest)
    print(
        f"[scaling] backbone {backbone}, visual_dim "
        f"{manifest['visual_dim']}, body {sizes}",
        flush=True,
    )

    # Conditioning is held FIXED across the curve: every point gets the
    # full cache's channel statistics, not its own subset's. A scaling
    # curve must vary in data quantity alone, and a scale estimate is not
    # label information — letting the 32-page point be standardised by 32
    # pages would put estimator noise into the axis being measured.
    stats = feature_channel_stats(train_full)

    rows = []
    for size in sizes:
        # Same seed and same architecture for every point, but NOT the
        # same compute: `--epochs` is fixed, so a point with 8x the pages
        # also takes 8x the optimizer steps. Data size and compute vary
        # together and the curve cannot separate them. Holding compute
        # fixed (and drawing prefixes from a frozen lineage permutation
        # rather than cache order) is P2, not something this loop does.
        torch.manual_seed(args.seed)
        model = StructureDecoder(
            visual_dim=manifest["visual_dim"],
            d_model=args.d_model,
            layers=args.layers,
            heads=args.heads,
        )
        model.set_feature_stats(*stats)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        subset = take_subset(train_full, size)
        for _ in range(args.epochs):
            train_epoch(
                model, subset, optimizer,
                batch_size=args.batch_size, device=args.device,
            )
        metrics = evaluate(
            model, val_set, batch_size=args.batch_size, device=args.device
        )
        row = {"train_pages": size, **metrics}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    print()
    print(f"{'strán':>9}{'val loss':>11}{'presnosť':>11}{'zlepšenie':>12}")
    previous = None
    for row in rows:
        delta = "—" if previous is None else f"{row['token_accuracy'] - previous:+.4f}"
        print(
            f"{row['train_pages']:>9,}{row['loss']:>11.4f}"
            f"{row['token_accuracy']:>11.4f}{delta:>12}"
        )
        previous = row["token_accuracy"]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())

"""Stage-1 training: the structural decoder over cached features.

The backbone is frozen, so this loop never loads it — it reads the
feature cache and trains the decoder alone. That is what puts Stage 1
inside a 12 GB card.

Batching pads two ways, and the difference matters:

- **targets** pad with `IGNORE_INDEX`, so padded steps are not scored;
- **the grammar mask** pads permissively. Those steps are never scored,
  but an all-False mask would leave a row of `-inf` logits, and a softmax
  over nothing is NaN. Padding a mask the same way as a target is the
  kind of symmetry that looks right and poisons a run.

**The grammar is a strong baseline, and the number to beat is not 0.5.**
A run with random features still reaches ~0.95 next-token accuracy,
because the §18.4 mask leaves only a handful of legal tokens at most
steps: the decoder learns the structural prior without reading the page
at all. Any real backbone must be scored against that null model, not
against chance — `--random-features` reproduces it.

For that comparison to mean anything the two arms must differ only in
information, so the noise arm is standardised by its own statistics
exactly as the real one is by the cache's: `feature_channel_stats` runs
on the dataset, after `randomize`, and the decoder holds the result.
Before that, real features arrived at cross-attention 14x louder than
the noise, and the null model was winning on conditioning.

Visual token counts differ per page too (pages are not all the same
size), so features pad to the batch's longest — and **that padding is
content unless it is masked.** `visual_projection` is a `nn.Linear` with
a bias, so a zero pad row projects to the learned bias vector, not to
zero, and cross-attends as a fully-fledged key. Measured: max |Δ logits|
= 0.176 for the same page alone versus co-batched with a longer one.
`collate` therefore returns `feature_padding_mask` and both loops pass it
as `memory_key_padding_mask`. With it, the pad rows cannot reach the
output at all — replacing their contents with noise leaves the logits
bitwise identical — and what is left across batch compositions is float
non-associativity at 4e-7, which two equal-length pages show as well.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from unflatten.model.decoder import masked_token_loss
from unflatten.model.feature_cache import DEFAULT_CACHE_SHARDS, FeatureCacheReader
from unflatten.model.shard_sampler import ShardWindowSampler
from unflatten.model.targets import BOS_TOKEN_ID, IGNORE_INDEX

#: Manifest identity fields a val cache must share with the train cache.
#: Differ on any of these and the two are not measuring the same model.
SHARED_LINEAGE_FIELDS = (
    "backbone",
    "backbone_revision",
    "processor",
    # A backbone can expose several feature taps, and Granite's two differ
    # in width, grid and what they are read from while sharing a model id
    # and a processor config. `visual_dim` happens to catch a cross-tap
    # pair today, but by DIMENSION — two taps of equal width would pass it.
    "feature_tap",
    "max_pixels",
    "multiscale",
    "merged_grid",
    "token_map_version",
    "grammar_hash",
    "grammar_profile",
    # DEC-024: which capability profile's grammar/exclusion set built this
    # cache -- `None` (the historical reproduction) and an explicit profile
    # are not the same measurement, even when `grammar_profile` happens to
    # already differ; this catches the identity directly rather than
    # relying on that as a proxy.
    "capability_profile_hash",
    "target_projector_version",
    "task_kind",
    # Different release: different documents, so the val number is not
    # measuring the model the train cache produced.
    "release_root_hash",
    # RFC005 §9.2: evaluating on inventory training is blocked from is
    # not the same measurement, in either direction.
    "admission",
)

#: Manifest fields outside `identity` that must also agree. `visual_dim`
#: is a hard incompatibility — one decoder has one input width — and
#: catching it here beats a shape error minutes into an epoch.
SHARED_MANIFEST_FIELDS = ("visual_dim", "storage_dtype")


class NonFiniteStepError(RuntimeError):
    """A training step produced a value that is not a finite number."""


def require_train_split(manifest: dict[str, Any]) -> None:
    """Refuse a cache that is not the training split.

    A `--split test` cache builds exactly like a train one and nothing
    downstream would notice; leakage has to be refused where it is
    stated, which is the manifest.
    """
    split = manifest.get("identity", {}).get("split")
    if split != "train":
        raise ValueError(
            f"train cache records split {split!r}, not 'train'; training on "
            "eval or test lineage is leakage, and a cache with no recorded "
            "split cannot prove it is not"
        )


def require_shared_lineage(
    train_manifest: dict[str, Any], val_manifest: dict[str, Any]
) -> None:
    """Refuse a val cache built under a different identity."""
    for field in SHARED_MANIFEST_FIELDS:
        if train_manifest.get(field) != val_manifest.get(field):
            raise ValueError(
                f"val cache disagrees with the train cache on {field}: "
                f"{val_manifest.get(field)!r} vs {train_manifest.get(field)!r}"
            )
    train_id = train_manifest.get("identity", {})
    val_id = val_manifest.get("identity", {})
    for field in SHARED_LINEAGE_FIELDS:
        if train_id.get(field) != val_id.get(field):
            raise ValueError(
                f"val cache disagrees with the train cache on {field}: "
                f"{val_id.get(field)!r} vs {train_id.get(field)!r}; the two "
                "would not be measuring the same model"
            )
    if val_id.get("split") == "train":
        raise ValueError("val cache records split 'train'")


class CacheDataset(Dataset):
    """A feature cache as a torch dataset."""

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        expect_backbone: str | None = None,
        randomize: bool = False,
        cache_shards: int = DEFAULT_CACHE_SHARDS,
    ) -> None:
        self.reader = FeatureCacheReader(
            cache_dir, expect_backbone=expect_backbone, cache_shards=cache_shards
        )
        #: Replace features with noise to measure the grammar-only floor.
        self.randomize = randomize
        #: Shard geometry, so `train_epoch` can shuffle without asking the
        #: cache for a different 512 MB file on every sample.
        self.shard_size = self.reader.shard_size
        self.shard_window = self.reader.cache_shards

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.reader[index]
        features = entry.features.float()
        if self.randomize:
            features = torch.randn_like(features)
        return {
            "features": features,
            "target_ids": entry.target_ids,
            "input_ids": entry.input_ids,
            "allowed": entry.allowed,
        }


def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a batch of pages to their longest sequence and token count."""
    steps = max(int(i["target_ids"].shape[0]) for i in items)
    tokens = max(int(i["features"].shape[0]) for i in items)
    dim = int(items[0]["features"].shape[1])
    vocab = int(items[0]["allowed"].shape[1])
    batch = len(items)

    features = torch.zeros(batch, tokens, dim)
    # True where a visual token is padding: the projection has a bias, so
    # an unmasked pad row is a key with content, not an empty one.
    feature_padding_mask = torch.ones(batch, tokens, dtype=torch.bool)
    target_ids = torch.full((batch, steps), IGNORE_INDEX, dtype=torch.long)
    input_ids = torch.full((batch, steps), BOS_TOKEN_ID, dtype=torch.long)
    # permissive, not all-False: a padded row of -inf logits softmaxes to NaN
    allowed = torch.ones(batch, steps, vocab, dtype=torch.bool)

    for row, item in enumerate(items):
        n_tokens = int(item["features"].shape[0])
        n_steps = int(item["target_ids"].shape[0])
        features[row, :n_tokens] = item["features"]
        feature_padding_mask[row, :n_tokens] = False
        target_ids[row, :n_steps] = item["target_ids"]
        input_ids[row, :n_steps] = item["input_ids"]
        allowed[row, :n_steps] = item["allowed"]
    return {
        "features": features,
        "feature_padding_mask": feature_padding_mask,
        "target_ids": target_ids,
        "input_ids": input_ids,
        "allowed": allowed,
    }


def feature_channel_stats(
    dataset: Dataset, *, pages: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and sd of a cache's features, from a page sample.

    The decoder standardises its frozen input by these (see
    `decoder.StructureDecoder`), and it has to get them from somewhere:
    the backbone's channel scales are a property of the cache, not of the
    decoder, and the cache stores raw features on purpose — they are what
    the backbone actually emitted, and re-deriving them costs GPU hours.
    Conditioning belongs to the consumer.

    Sampled, not exhaustive: the pages are spread evenly over the cache
    (so the estimate is not one document's), and a few dozen pages is
    already tens of thousands of token rows per channel — far more than a
    scale estimate needs, and orders of magnitude cheaper than a pass
    over a 39 GB cache. Ascending order keeps each shard loaded once.

    Accumulated in float64: these are sums of squares over millions of
    values, some of them in the thousands.
    """
    total = len(dataset)  # type: ignore[arg-type]
    if total == 0:
        raise ValueError("cannot take feature statistics of an empty dataset")
    if pages < 1:
        raise ValueError(f"pages must be positive, got {pages}")
    step = max(1, total // min(pages, total))
    indices = list(range(0, total, step))[:pages]

    count = 0
    summed = squared = None
    for index in indices:
        rows = dataset[index]["features"].double()  # type: ignore[index]
        if summed is None:
            summed = torch.zeros(rows.shape[1], dtype=torch.float64)
            squared = torch.zeros(rows.shape[1], dtype=torch.float64)
        summed += rows.sum(dim=0)
        squared += (rows * rows).sum(dim=0)
        count += int(rows.shape[0])
    if summed is None or count < 2:
        raise ValueError(
            f"feature statistics need at least two token rows, got {count}"
        )
    mean = summed / count
    variance = (squared - count * mean**2) / (count - 1.0)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    return mean.float(), std.float()


def shard_aware_sampler(dataset: Dataset) -> ShardWindowSampler | None:
    """A shuffling sampler that keeps shard locality, if the dataset has any.

    A dataset that does not describe its shards (a plain in-memory one,
    or a test double) gets `None` and falls back to `shuffle=True`, which
    is correct there because there is nothing to thrash.
    """
    shard_size = getattr(dataset, "shard_size", None)
    window = getattr(dataset, "shard_window", None)
    if not shard_size or not window:
        return None
    return ShardWindowSampler(
        len(dataset),  # type: ignore[arg-type]
        shard_size=int(shard_size),
        window=int(window),
    )


def train_epoch(
    model: torch.nn.Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 4,
    device: str = "cpu",
    grad_clip: float = 1.0,
    log_every: int = 0,
) -> float:
    """One pass over `dataset`; returns the mean loss."""
    from torch.utils.data import DataLoader

    sampler = shard_aware_sampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=collate,
    )
    model.train().to(device)
    total = 0.0
    steps = 0
    for index, batch in enumerate(loader, start=1):
        features = batch["features"].to(device)
        target_ids = batch["target_ids"].to(device)
        input_ids = batch["input_ids"].to(device)
        allowed = batch["allowed"].to(device)

        logits = model(
            features,
            input_ids,
            memory_key_padding_mask=batch["feature_padding_mask"].to(device),
        )
        loss = masked_token_loss(logits, target_ids, allowed)
        if not bool(torch.isfinite(loss)):
            # Skip the step, then stop: an optimizer step on a non-finite
            # loss poisons every parameter it touches, and continuing past
            # it turns a broken run into a run that merely looks bad.
            raise NonFiniteStepError(
                f"batch {index}: loss is {float(loss.detach())}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not bool(torch.isfinite(norm)):
                optimizer.zero_grad(set_to_none=True)
                raise NonFiniteStepError(
                    f"batch {index}: gradient norm is {float(norm)}"
                )
        optimizer.step()

        total += float(loss.detach())
        steps += 1
        if log_every and index % log_every == 0:
            print(f"[train] {index} dávok, loss {total / steps:.4f}", flush=True)
    return total / max(1, steps)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 4,
    device: str = "cpu",
) -> dict[str, float]:
    """Token-weighted loss and next-token accuracy over `dataset`.

    Weighted by scored tokens, not averaged over batches: pages differ in
    length by an order of magnitude, so a mean of batch means reports a
    number that no token has.
    """
    from torch.utils.data import DataLoader

    if len(dataset) == 0:  # type: ignore[arg-type]
        raise ValueError(
            "cannot evaluate an empty dataset; a zeroed metric reads as a "
            "measurement and there is nothing here to measure"
        )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    model.eval().to(device)
    total = correct = scored = 0.0
    for batch in loader:
        features = batch["features"].to(device)
        target_ids = batch["target_ids"].to(device)
        allowed = batch["allowed"].to(device)
        logits = model(
            features,
            batch["input_ids"].to(device),
            memory_key_padding_mask=batch["feature_padding_mask"].to(device),
        )
        mask = target_ids != IGNORE_INDEX
        in_batch = float(mask.sum())
        total += float(masked_token_loss(logits, target_ids, allowed)) * in_batch
        predicted = logits.masked_fill(~allowed, float("-inf")).argmax(dim=-1)
        correct += float((predicted[mask] == target_ids[mask]).sum())
        scored += in_batch
    if scored == 0.0:
        raise ValueError("no scored tokens in the dataset; nothing to evaluate")
    return {
        "loss": total / scored,
        "token_accuracy": correct / scored,
        "scored_tokens": scored,
    }


def main(argv: list[str] | None = None) -> int:
    """Train the structural decoder over a feature cache."""
    import argparse
    import json
    from pathlib import Path as _Path

    from unflatten.model.decoder import StructureDecoder
    from unflatten.model.feature_cache import cache_manifest

    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.train", description=__doc__
    )
    parser.add_argument("train_cache", type=_Path)
    parser.add_argument("--val-cache", type=_Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=_Path, default=None, help="checkpoint path")
    parser.add_argument(
        "--random-features",
        action="store_true",
        help="replace cached features with noise: the grammar-only null "
             "model a real backbone has to beat (~0.95 token accuracy)",
    )
    parser.add_argument(
        "--stat-pages",
        type=int,
        default=64,
        help="pages sampled for the per-channel feature statistics the "
             "decoder standardises its frozen input by",
    )
    args = parser.parse_args(argv)

    manifest = cache_manifest(args.train_cache)
    require_train_split(manifest)
    backbone = manifest["identity"]["backbone"]
    val_manifest = None
    if args.val_cache:
        val_manifest = cache_manifest(args.val_cache)
        require_shared_lineage(manifest, val_manifest)
    # The decoder's only backbone-dependent number comes from the cache
    # that produced it, never from a constant.
    model = StructureDecoder(
        visual_dim=manifest["visual_dim"],
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
    )
    params = sum(p.numel() for p in model.parameters())
    print(
        f"[train] backbone {backbone}, visual_dim "
        f"{manifest['visual_dim']}, decoder {params / 1e6:.1f} M param",
        flush=True,
    )

    # expect_backbone is always passed: the check exists precisely for
    # the case where nobody remembered to ask for it.
    train_set = CacheDataset(
        args.train_cache, expect_backbone=backbone,
        randomize=args.random_features,
    )
    val_set = (
        CacheDataset(
            args.val_cache, expect_backbone=backbone,
            randomize=args.random_features,
        )
        if args.val_cache
        else None
    )
    if args.random_features:
        print("[train] NULL MODEL: features are noise, grammar only", flush=True)

    # Taken from the training cache only — val is conditioned by train
    # statistics, like any other fitted normalisation — and from the
    # dataset rather than the reader, so the null model is standardised
    # by its own noise and both arms reach cross-attention alike.
    mean, std = feature_channel_stats(train_set, pages=args.stat_pages)
    model.set_feature_stats(mean, std)
    print(
        f"[train] feature stats over {args.stat_pages} pages: "
        f"channel sd min {float(std.min()):.4f} median "
        f"{float(std.median()):.4f} max {float(std.max()):.3f}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model, train_set, optimizer,
            batch_size=args.batch_size, device=args.device, log_every=50,
        )
        row = {"epoch": epoch, "train_loss": loss}
        if val_set is not None:
            row.update(
                {
                    f"val_{k}": v
                    for k, v in evaluate(
                        model, val_set, batch_size=args.batch_size, device=args.device
                    ).items()
                }
            )
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "visual_dim": manifest["visual_dim"],
                "backbone": backbone,
                "cache_identity": manifest["identity"],
                "d_model": args.d_model,
                "layers": args.layers,
                "heads": args.heads,
            },
            args.out,
        )
        print(f"[train] checkpoint -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())

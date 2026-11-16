"""Diagnostic: is paragraph spacing readable once features keep position?

The Stage-0 pooled probe scored seven of its eight signals and failed
the spacing regression outright (rho -0.008, R² -1.122, scored then as
`mean_space_after_mpt` — the page MEAN this label no longer is). The hypothesis
under test is that the failure is the REPRESENTATION, not the backbone:
`BackboneFeatures.pooled` averages every token of the page into one
vector, and paragraph spacing is a vertical distance between blocks —
a distance cannot survive an average, while the two regressions that did
work (`bold_run_ratio`, the font-size label) are ratios over content,
which averaging preserves.

This module runs the controlled comparison. One extraction pass stores,
per page, BOTH `pooled()` — the production vector, unchanged — and
`band_pooled(k)`, the same features with the vertical axis kept. Same
pages, same labels, same head, one variable.

Two things it deliberately does NOT do:

- **It does not tune until something passes.** The penalty is chosen by
  cross-validation on the TRAIN split, per arm and per label, from one
  shared grid; the eval number is then reported once. The full eval
  curve over that grid is reported too, so nothing is hidden behind the
  selection.
- **It does not touch the production probe.** `unflatten.probe.run` and
  its cache are untouched, and its numbers stay reproducible; this
  writes its own cache under its own identity.

A negative result is the point as much as a positive one: if spacing is
unreadable with position kept, the information is lost inside the frozen
vision tower rather than in our pooling, and RFC004 has to answer for
the feature tap rather than for the adapter.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import torch

from unflatten.probe.bands import BAND_COUNTS, compare, report
from unflatten.probe.run import (
    ProbeCacheIdentity,
    _check_cache_identity,
    _require_an_identifiable_release,
    _supervised_targets,
    probe_cache_identity,
)

#: Bumped when the stored payload's meaning changes.
CACHE_VERSION = 1


class BandCacheError(RuntimeError):
    """A band-feature cache cannot prove it is what this run asked for."""


# ---------------------------------------------------------------- extraction --


def _representation(feature_dim: int) -> dict[str, Any]:
    """What the stored payload IS, recorded beside the run identity.

    `ProbeCacheIdentity` is the production probe's and is deliberately
    left alone; the extra facts that distinguish this cache — which band
    counts it holds, how wide the features are — ride alongside it.
    """
    return {
        "version": CACHE_VERSION,
        "bands": list(BAND_COUNTS),
        "feature_dim": feature_dim,
    }


def extract_split(
    backbone: Any,
    release_dir: Path | str,
    store_dir: Path | str,
    split: str,
    task_kind: str,
    *,
    limit: int,
    cache_path: Path | None = None,
    identity: ProbeCacheIdentity | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Pooled AND banded features plus probe labels for one split.

    A page enters both arms or neither: a grid too short to split into
    the finest band count is skipped entirely rather than contributed to
    the pooled arm alone, which would make the two arms differ by their
    page set as well as by their representation.
    """
    from PIL import Image

    from unflatten.dataset.items import build_item
    from unflatten.dataset.samples import iter_page_samples
    from unflatten.probe.labels import extract_labels

    if cache_path is not None and identity is not None:
        _require_an_identifiable_release(identity, Path(cache_path))
    if cache_path is not None and Path(cache_path).exists():
        blob = torch.load(cache_path, weights_only=True)
        if identity is not None:
            _check_cache_identity(blob, identity, Path(cache_path))
        recorded = blob.get("representation")
        expected = _representation(int(blob["pooled"].shape[1]))
        if recorded != expected:
            raise BandCacheError(
                f"{cache_path}: it holds {recorded!r}, this run compares "
                f"{expected!r}; delete it and re-extract"
            )
        return blob

    pooled: list[torch.Tensor] = []
    banded: dict[int, list[torch.Tensor]] = {k: [] for k in BAND_COUNTS}
    labels: list[dict[str, Any]] = []
    # which document each page came from: pages of one document share a
    # document-level label, so significance is estimated over documents
    documents: list[str] = []
    skipped_unlabelled = skipped_geometry = 0
    pack_cache: dict[str, Any] = {}
    for sample in iter_page_samples(release_dir, store_dir, split, task_kind):
        if len(pooled) >= limit:
            break
        item = build_item(Path(store_dir), sample)
        if not item.fragment:
            skipped_unlabelled += 1
            continue
        image = Image.open(io.BytesIO(item.image_bytes))
        image.load()
        features = backbone.extract(image)
        try:
            strips = {k: features.band_pooled(k).cpu() for k in BAND_COUNTS}
        except ValueError:
            # a grid this run cannot band is skipped from BOTH arms; the
            # comparison is only controlled if the page sets are equal
            skipped_geometry += 1
            continue
        pooled.append(features.pooled().cpu())
        documents.append(str(getattr(sample, "document_bundle_id", "")))
        for count, strip in strips.items():
            banded[count].append(strip)
        labels.append(
            extract_labels(
                item.fragment, _supervised_targets(item, Path(store_dir), pack_cache)
            ).as_dict()
        )
        if progress_every and len(pooled) % progress_every == 0:
            print(f"[bands] {len(pooled)}/{limit} strán", flush=True)

    if skipped_unlabelled or skipped_geometry:
        print(
            f"[bands] preskočené: {skipped_unlabelled} bez fragmentu, "
            f"{skipped_geometry} bez použiteľnej mriežky",
            flush=True,
        )
    payload = {
        "pooled": torch.stack(pooled) if pooled else torch.empty(0),
        "bands": {k: torch.stack(v) for k, v in banded.items() if v},
        "labels": labels,
        "documents": documents,
    }
    payload["representation"] = _representation(
        int(payload["pooled"].shape[1]) if pooled else -1
    )
    print(
        f"[bands] {len(pooled)} strán z {len(set(documents))} dokumentov",
        flush=True,
    )
    if identity is not None:
        payload["identity"] = identity.as_dict()
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
    return payload


# ----------------------------------------------------------------------- CLI --


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.probe.run_bands", description=__doc__
    )
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("store_dir", type=Path)
    parser.add_argument("--task-kind", default="structure_v0")
    parser.add_argument("--train-pages", type=int, default=1500)
    parser.add_argument("--eval-pages", type=int, default=500)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument(
        "--tap",
        default=None,
        help="feature tap, for a backbone that exposes several "
             "(Granite Vision: 'siglip4' / 'projected8')",
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
        f"[bands] backbone {model_id}{tap_tag}, dim {backbone.feature_dim}",
        flush=True,
    )
    print(
        f"[bands] pixel budget: {backbone.pixel_budget_applied or 'native'}"
        f"  measured: {backbone.pixel_budget_measured or 'not capped'}",
        flush=True,
    )

    def cache(name: str) -> Path | None:
        if args.cache_dir is None:
            return None
        tag = model_id.replace("/", "_")
        budget = args.max_pixels or "native"
        # the tap changes the features, so it changes the cache key too
        tap = f"-{backbone.tap}" if backbone.tap else ""
        return Path(args.cache_dir) / (
            f"{tag}{tap}-{args.task_kind}-{budget}-bands-{name}.pt"
        )

    def identity(split: str, pages: int) -> ProbeCacheIdentity:
        return probe_cache_identity(
            backbone,
            args.release_dir,
            split=split,
            task_kind=args.task_kind,
            max_pixels=args.max_pixels,
            page_limit=pages,
        )

    train = extract_split(
        backbone, args.release_dir, args.store_dir, "train", args.task_kind,
        limit=args.train_pages, cache_path=cache("train"),
        identity=identity("train", args.train_pages),
    )
    evaluation = extract_split(
        backbone, args.release_dir, args.store_dir, "val_iid", args.task_kind,
        limit=args.eval_pages, cache_path=cache("val"),
        identity=identity("val_iid", args.eval_pages),
    )
    summary = compare(train, evaluation)
    summary["backbone"] = model_id
    summary["feature_tap"] = backbone.tap
    summary["task_kind"] = args.task_kind
    print(report(summary))
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())

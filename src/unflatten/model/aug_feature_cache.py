"""T5b — the augmented feature store (DEC-041 point 2/3, T1b/T5a follow-on).

`m2b_augment` (T5a) writes `augmented-views.jsonl` in the M2a/M2b store:
one row per augmented PNG, naming the source `(document_bundle_id,
page_index)`, the deterministic `render_view_id`, the pixel transform
(`transform_matrix`) and the output PNG's store ref. This module reads
that ledger plus the MAIN exact-text cache (for the source page's
`region_targets`) and writes a small, SEPARATE store,
`cache-aug-features-<rev>/`, holding per `render_view_id`: the pinned
vision backbone's `features`/`grid_thw` on the augmented PNG, and the
source page's `region_targets` TRANSFORMED by the SAME matrix
(`unflatten.dataset.augment.geometry.transform_bbox_ppm`).

**Never touches `ExactTextCacheRow`/`EXACT_TEXT_CACHE_VERSION`/
`ExactTextCacheIdentity`.** The main v15 cache is being built under
6.0.0 on the production box; a shape change there would make it
unreadable. This store is its own format with its own manifest, bound to
the main cache by `main_cache_identity_hash` (`run_binding.
compute_exact_text_cache_identity_hash`) plus the exact
`backbone_artifact_hash`/`processor_artifact_hash` pair the main cache
carries -- an augmented feature from a DIFFERENT backbone than the clean
rows it sits next to in a training batch would be silently incomparable.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch

from unflatten.dataset.augment.geometry import Matrix3x3, transform_bbox_ppm
from unflatten.dataset.gates.store import read_object
from unflatten.model.exact_text_cache import ExactTextCacheReader, PageKey
from unflatten.model.region_targets import RegionTarget
from unflatten.model.run_binding import compute_exact_text_cache_identity_hash

AUG_FEATURE_CACHE_VERSION = "0.1.0"

#: A ledger view whose source page the main cache does not hold -- counted
#: and skipped (it can never be served), never a crash of the run.
SOURCE_PAGE_NOT_IN_MAIN_CACHE = "SOURCE_PAGE_NOT_IN_MAIN_CACHE"
_MANIFEST_FILENAME = "manifest.json"
_ROWS_DIRNAME = "rows"


class AugFeatureCacheError(RuntimeError):
    """The augmentation ledger, the main cache, or an existing aug store
    could not be read/bound truthfully. A refusal, never a repair."""


class _BackboneLike(Protocol):
    """The one method this module needs (`candidate_backbone.
    CandidateVisionBackbone.extract` shape) -- tests inject a fake with
    this surface, never the real GPU-backed loader."""

    def extract(self, image: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def ledger_sha256(ledger_path: Path | str) -> str:
    """`sha256:<hex>` of the ledger file's raw bytes -- the manifest's own
    identity for "which ledger this store was built from"."""
    digest = hashlib.sha256()
    with open(ledger_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def load_ledger_rows(ledger_path: Path | str) -> list[dict[str, Any]]:
    """Every `augmented-views.jsonl` row, last-wins per `render_view_id`
    (mirrors `m2b_augment`'s own last-wins convention, keyed the way THIS
    consumer needs: one row per output view, not per source page)."""
    rows: dict[str, dict[str, Any]] = {}
    with open(ledger_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["render_view_id"]] = row
    return [rows[key] for key in sorted(rows)]


# ---------------------------------------------------------------------------
# Region-target transform
# ---------------------------------------------------------------------------


def transform_region_targets(
    region_targets: Sequence[RegionTarget],
    matrix: Matrix3x3,
    *,
    width_px: int,
    height_px: int,
) -> tuple[RegionTarget, ...]:
    """`transform_region_targets_with_collapsed` without the collapsed ids
    (the DEC-040 store and `retarget` need only the targets)."""
    return transform_region_targets_with_collapsed(
        region_targets, matrix, width_px=width_px, height_px=height_px
    )[0]


def transform_region_targets_with_collapsed(
    region_targets: Sequence[RegionTarget],
    matrix: Matrix3x3,
    *,
    width_px: int,
    height_px: int,
) -> tuple[tuple[RegionTarget, ...], frozenset[str]]:
    """Every `region_targets` entry with its `regions` boxes pushed
    through `matrix` (`geometry.transform_bbox_ppm`). A box `matrix` moves
    off the page or collapses to zero area is DROPPED from `regions`
    (never fabricated back in) -- the node's OTHER fields (`page_evidence`,
    `observable_mask`, `supervision_mask`) keep their named status
    unchanged; only `mapping_method`/`chain_scope` fall back to `None`
    when EVERY region collapsed, matching `RegionTarget`'s own documented
    invariant (empty `regions` implies `None`).

    Also returns the `node_id`s for which AT LEAST ONE box collapsed (a node
    with two regions, one collapsed, keeps k == 1 and would otherwise look
    like a node that was always single-region -- RFC004-ERRATA-012 E1)."""
    transformed: list[RegionTarget] = []
    collapsed_ids: set[str] = set()
    for target in region_targets:
        kept_regions: list[tuple[int, int, int, int]] = []
        kept_continuation: list[tuple[bool, bool]] = []
        for bbox, cont in zip(target.regions, target.continuation, strict=True):
            status, new_bbox = transform_bbox_ppm(bbox, matrix, width_px, height_px)
            if status == "collapsed":
                collapsed_ids.add(target.node_id)
                continue
            kept_regions.append(new_bbox)  # type: ignore[arg-type]
            kept_continuation.append(cont)
        all_collapsed = bool(target.regions) and not kept_regions
        transformed.append(
            RegionTarget(
                node_id=target.node_id,
                node_type=target.node_type,
                anchor_token_index=target.anchor_token_index,
                page_evidence=target.page_evidence,
                observable_mask=target.observable_mask,
                supervision_mask=target.supervision_mask,
                regions=tuple(kept_regions),
                continuation=tuple(kept_continuation),
                mapping_method=None if all_collapsed else target.mapping_method,
                chain_scope=None if all_collapsed else target.chain_scope,
            )
        )
    return tuple(transformed), frozenset(collapsed_ids)


def collapse_region_targets(
    region_targets: Sequence[RegionTarget],
) -> tuple[tuple[RegionTarget, ...], frozenset[str]]:
    """`region_targets` for a view whose pixels are NOT aligned with the page
    reference (an unaligned external capture): every box of every node that
    HAS observed regions is dropped, exactly as if `transform_region_targets_
    with_collapsed` had collapsed them all (empty `regions`, `mapping_method`
    and `chain_scope` `None`). Returns those nodes' ids. A node with no regions
    to begin with is returned untouched and is not named."""
    out: list[RegionTarget] = []
    collapsed_ids: set[str] = set()
    for target in region_targets:
        if not target.regions:
            out.append(target)
            continue
        collapsed_ids.add(target.node_id)
        out.append(
            RegionTarget(
                node_id=target.node_id,
                node_type=target.node_type,
                anchor_token_index=target.anchor_token_index,
                page_evidence=target.page_evidence,
                observable_mask=target.observable_mask,
                supervision_mask=target.supervision_mask,
                regions=(),
                continuation=(),
                mapping_method=None,
                chain_scope=None,
            )
        )
    return tuple(out), frozenset(collapsed_ids)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _row_path(out_dir: Path, view_id: str) -> Path:
    return out_dir / _ROWS_DIRNAME / f"{view_id}.pt"


def _meta_path(out_dir: Path, view_id: str) -> Path:
    return out_dir / _ROWS_DIRNAME / f"{view_id}.meta.json"


def _already_written(out_dir: Path) -> dict[str, tuple[str, int]]:
    """`{render_view_id: (document_bundle_id, page_index)}` for every row
    THIS out_dir already carries -- read from the small meta sidecar, not
    by re-loading the tensor file, so resume never pays a heavy load just
    to learn what is already done."""
    rows_dir = out_dir / _ROWS_DIRNAME
    if not rows_dir.is_dir():
        return {}
    done: dict[str, tuple[str, int]] = {}
    for meta_path in rows_dir.glob("*.meta.json"):
        view_id = meta_path.name.removesuffix(".meta.json")
        if not (out_dir / _ROWS_DIRNAME / f"{view_id}.pt").exists():
            continue  # a meta sidecar without its tensor is a partial write
        meta = json.loads(meta_path.read_text())
        done[view_id] = (meta["document_bundle_id"], meta["page_index"])
    return done



def _shard_of(view_id: str, count: int) -> int:
    """Which of `count` shards owns `view_id` -- a stable content hash, so
    every process computes the same split without coordinating."""
    return int(hashlib.sha256(view_id.encode()).hexdigest(), 16) % count

def build_aug_feature_store(
    *,
    ledger_path: Path | str,
    store_dir: Path | str,
    main_cache_dir: Path | str,
    out_dir: Path | str,
    backbone: _BackboneLike,
    augmentation_policy_hash: str,
    expect_backbone_artifact_hash: str,
    expect_processor_artifact_hash: str,
    limit: int | None = None,
    progress_every: int = 200,
    shard: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build (or resume) the aug feature store at `out_dir`.

    `expect_backbone_artifact_hash`/`expect_processor_artifact_hash` have
    no default: the caller must name which backbone it believes
    `main_cache_dir` was built under, and this function refuses rather
    than silently trusting a DIFFERENT main cache's identity -- the
    augmented rows would then sit next to clean rows from an incomparable
    backbone in the same training batch with nothing to say so.
    """
    ledger_path = Path(ledger_path)
    store_dir = Path(store_dir)
    main_cache_dir = Path(main_cache_dir)
    out_dir = Path(out_dir)

    reader = ExactTextCacheReader(main_cache_dir, gradient_bearing=False)
    identity = reader.identity
    if identity.backbone_artifact_hash != expect_backbone_artifact_hash:
        raise AugFeatureCacheError(
            f"main cache {main_cache_dir} backbone_artifact_hash "
            f"{identity.backbone_artifact_hash!r} != expected "
            f"{expect_backbone_artifact_hash!r} -- refusing to build augmented "
            "features that would sit next to clean rows from a different backbone"
        )
    if identity.processor_artifact_hash != expect_processor_artifact_hash:
        raise AugFeatureCacheError(
            f"main cache {main_cache_dir} processor_artifact_hash "
            f"{identity.processor_artifact_hash!r} != expected "
            f"{expect_processor_artifact_hash!r}"
        )
    from dataclasses import asdict

    main_cache_identity_hash = compute_exact_text_cache_identity_hash(asdict(identity))
    page_index_of: dict[PageKey, int] = {key: i for i, key in enumerate(reader.page_keys)}

    ledger_hash = ledger_sha256(ledger_path)
    rows = load_ledger_rows(ledger_path)
    if limit is not None:
        rows = rows[:limit]

    if shard is not None:
        index, count = shard
        if count < 1 or not 0 <= index < count:
            raise ValueError(f"shard {index}/{count}: need 0 <= index < count")
        # Split by the source page's MAIN-CACHE shard, not by view id: each
        # source read loads a whole ~1.3 GB main-cache shard (sha256, unpickle,
        # 32 row checks), and a view-hash split made every process load nearly
        # every shard once per view -- ~29 views/min over four processes on
        # prod (2026-09-24). A page the main cache lacks is skipped below;
        # the view hash only decides who counts that skip. Either way the
        # split depends on the cache and view ids alone, so shards stay
        # disjoint and complete however the ledger is ordered or resumed.
        def owner(row: dict[str, Any]) -> int:
            position = page_index_of.get((row["document_bundle_id"], row["page_index"]))
            if position is None:
                return _shard_of(row["render_view_id"], count)
            return (position // reader.shard_size) % count

        rows = [row for row in rows if owner(row) == index]

    done = _already_written(out_dir)
    pending = [row for row in rows if row["render_view_id"] not in done]
    # Cache order, so consecutive views reuse the main-cache shard just loaded.
    pending.sort(
        key=lambda row: page_index_of.get(
            (row["document_bundle_id"], row["page_index"]), len(page_index_of)
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _ROWS_DIRNAME).mkdir(exist_ok=True)

    written = 0
    skipped: dict[str, int] = {}
    for position, row in enumerate(pending, start=1):
        view_id = row["render_view_id"]
        bundle = row["document_bundle_id"]
        page_index = row["page_index"]
        source_key: PageKey = (bundle, page_index)
        cache_position = page_index_of.get(source_key)
        if cache_position is None:
            # Augmentation selects over every rendered train page; the main
            # cache holds only the pages it could write (~13 % drop on
            # release-v14-r5: no source features, grammar refusals). Such a
            # variant can never be served -- the plan enumerates cache pages
            # only -- so count it and move on, before any backbone call.
            # This used to RAISE and would have stopped the real GPU run
            # within its first few hundred views.
            skipped[SOURCE_PAGE_NOT_IN_MAIN_CACHE] = (
                skipped.get(SOURCE_PAGE_NOT_IN_MAIN_CACHE, 0) + 1
            )
            continue
        source_row = reader[cache_position]

        from PIL import Image

        image_bytes = read_object(store_dir, row["image"])
        import io

        image = Image.open(io.BytesIO(image_bytes))
        features = backbone.extract(image)

        matrix: Matrix3x3 = tuple(
            tuple(float(v) for v in triple) for triple in row["transform_matrix"]
        )  # type: ignore[assignment]
        region_targets = transform_region_targets(
            source_row.region_targets,
            matrix,
            width_px=row["width_px"],
            height_px=row["height_px"],
        )

        grid_thw = torch.tensor(list(features.grid_thw), dtype=torch.int64)
        torch.save(
            {
                "features": features.visual_features.detach().to(torch.float16).cpu(),
                "grid_thw": grid_thw,
                "region_targets": region_targets,
                "document_bundle_id": bundle,
                "page_index": page_index,
            },
            _row_path(out_dir, view_id),
        )
        _meta_path(out_dir, view_id).write_text(
            json.dumps({"document_bundle_id": bundle, "page_index": page_index})
        )
        written += 1
        if progress_every and position % progress_every == 0:
            print(f"[aug_feature_cache] {position}/{len(pending)} written={written}", flush=True)

    all_done = _already_written(out_dir)
    manifest = {
        "aug_feature_cache_version": AUG_FEATURE_CACHE_VERSION,
        "main_cache_identity_hash": main_cache_identity_hash,
        "augmentation_policy_hash": augmentation_policy_hash,
        "ledger_sha256": ledger_hash,
        "backbone_artifact_hash": identity.backbone_artifact_hash,
        "processor_artifact_hash": identity.processor_artifact_hash,
        "row_count": len(all_done),
        "skipped": dict(sorted(skipped.items())),
        "rows": {view_id: list(key) for view_id, key in sorted(all_done.items())},
    }
    # Atomic: with `--shard`, several processes finish and rewrite the manifest
    # from the rows on disk; a torn write would leave invalid JSON behind.
    tmp = out_dir / f"{_MANIFEST_FILENAME}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(tmp, out_dir / _MANIFEST_FILENAME)
    return {
        "out_dir": str(out_dir),
        "candidates": len(rows),
        "already_done": len(done),
        "written": written,
        "skipped": dict(sorted(skipped.items())),
        "row_count": len(all_done),
    }


def retarget_aug_feature_store(
    *,
    src_dir: Path | str,
    ledger_path: Path | str,
    main_cache_dir: Path | str,
    out_dir: Path | str,
    shard: tuple[int, int] | None = None,
    progress_every: int = 500,
) -> dict[str, Any]:
    """Rewrite a built aug store's `region_targets` from a REBUILT main cache,
    into a new `out_dir`; features are copied unchanged.

    The augmented features depend only on the augmented PNG and the pinned
    backbone, so a main cache rebuilt for new observations (e.g. image
    positions) changes the targets, never the features -- this avoids the
    GPU pass. Refuses a different backbone/processor or ledger than the
    source store was built with, and never writes into `src_dir`.
    """
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    if out_dir.resolve() == src_dir.resolve():
        raise AugFeatureCacheError(f"{out_dir}: retarget never writes into its source store")
    src = AugFeatureCacheReader(src_dir)
    ledger_hash = ledger_sha256(ledger_path)
    if ledger_hash != src.manifest["ledger_sha256"]:
        raise AugFeatureCacheError(
            f"ledger {ledger_path} hashes to {ledger_hash}, but {src_dir} was built "
            f"from {src.manifest['ledger_sha256']}"
        )
    reader = ExactTextCacheReader(main_cache_dir, gradient_bearing=False)
    identity = reader.identity
    for field_name in ("backbone_artifact_hash", "processor_artifact_hash"):
        if getattr(identity, field_name) != src.manifest[field_name]:
            raise AugFeatureCacheError(
                f"main cache {main_cache_dir} {field_name} {getattr(identity, field_name)!r} "
                f"!= the source store's {src.manifest[field_name]!r} -- its features "
                "would sit next to clean rows from a different backbone"
            )
    from dataclasses import asdict

    main_cache_identity_hash = compute_exact_text_cache_identity_hash(asdict(identity))
    page_index_of: dict[PageKey, int] = {key: i for i, key in enumerate(reader.page_keys)}
    ledger = {row["render_view_id"]: row for row in load_ledger_rows(ledger_path)}

    views = sorted(src.manifest["rows"])
    if shard is not None:
        index, count = shard
        if count < 1 or not 0 <= index < count:
            raise ValueError(f"shard {index}/{count}: need 0 <= index < count")

        def owner(view_id: str) -> int:
            key = tuple(src.manifest["rows"][view_id])
            position = page_index_of.get(key)  # type: ignore[arg-type]
            if position is None:
                return _shard_of(view_id, count)
            return (position // reader.shard_size) % count

        views = [v for v in views if owner(v) == index]

    done = _already_written(out_dir)
    (out_dir / _ROWS_DIRNAME).mkdir(parents=True, exist_ok=True)
    skipped: dict[str, int] = {}
    pending = []
    for view_id in views:
        key = tuple(src.manifest["rows"][view_id])
        if key not in page_index_of:
            reason = SOURCE_PAGE_NOT_IN_MAIN_CACHE
            skipped[reason] = skipped.get(reason, 0) + 1
        elif view_id not in done:
            pending.append(view_id)
    # Cache order, so consecutive views reuse the main-cache shard just loaded.
    pending.sort(key=lambda v: page_index_of[tuple(src.manifest["rows"][v])])  # type: ignore[index]

    written = 0
    for position, view_id in enumerate(pending, start=1):
        row = ledger[view_id]
        bundle, page_index = src.manifest["rows"][view_id]
        source_row = reader[page_index_of[(bundle, page_index)]]
        matrix: Matrix3x3 = tuple(
            tuple(float(v) for v in triple) for triple in row["transform_matrix"]
        )  # type: ignore[assignment]
        stored = src.get(view_id)
        stored["region_targets"] = transform_region_targets(
            source_row.region_targets, matrix, width_px=row["width_px"], height_px=row["height_px"]
        )
        torch.save(stored, _row_path(out_dir, view_id))
        _meta_path(out_dir, view_id).write_text(
            json.dumps({"document_bundle_id": bundle, "page_index": page_index})
        )
        written += 1
        if progress_every and position % progress_every == 0:
            print(f"[aug_feature_cache retarget] {position}/{len(pending)}", flush=True)

    all_done = _already_written(out_dir)
    manifest = {
        **{k: src.manifest[k] for k in (
            "aug_feature_cache_version", "augmentation_policy_hash", "ledger_sha256",
            "backbone_artifact_hash", "processor_artifact_hash",
        )},
        "main_cache_identity_hash": main_cache_identity_hash,
        "retargeted_from_manifest_sha256": ledger_sha256(src_dir / _MANIFEST_FILENAME),
        "row_count": len(all_done),
        "skipped": dict(sorted(skipped.items())),
        "rows": {view_id: list(key) for view_id, key in sorted(all_done.items())},
    }
    tmp = out_dir / f"{_MANIFEST_FILENAME}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(tmp, out_dir / _MANIFEST_FILENAME)
    return {"out_dir": str(out_dir), "written": written, "skipped": dict(sorted(skipped.items())),
            "row_count": len(all_done)}


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


class AugFeatureCacheReader:
    """Random access into a built aug feature store by `render_view_id`."""

    def __init__(self, out_dir: Path | str) -> None:
        self.out_dir = Path(out_dir)
        manifest_path = self.out_dir / _MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise AugFeatureCacheError(f"{self.out_dir}: no {_MANIFEST_FILENAME}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("aug_feature_cache_version") != AUG_FEATURE_CACHE_VERSION:
            raise AugFeatureCacheError(
                f"{self.out_dir}: aug_feature_cache_version "
                f"{self.manifest.get('aug_feature_cache_version')!r} != "
                f"{AUG_FEATURE_CACHE_VERSION!r}"
            )

    def require_bound_to(self, main_cache_identity_hash: str) -> None:
        """Refuse unless this store's manifest was built against exactly
        the presented main cache identity -- the check `s4_train` runs
        before ever serving an augmented row into a batch."""
        if self.manifest["main_cache_identity_hash"] != main_cache_identity_hash:
            raise AugFeatureCacheError(
                f"{self.out_dir}: bound to main_cache_identity_hash "
                f"{self.manifest['main_cache_identity_hash']!r}, not the presented "
                f"{main_cache_identity_hash!r} -- refusing to serve augmented rows "
                "next to clean rows from a different main cache"
            )

    def __contains__(self, view_id: str) -> bool:
        return view_id in self.manifest["rows"]

    def get(self, view_id: str) -> dict[str, Any]:
        if view_id not in self.manifest["rows"]:
            raise AugFeatureCacheError(
                f"{self.out_dir}: no row for render_view_id {view_id!r}"
            )
        return torch.load(_row_path(self.out_dir, view_id), weights_only=False)


def iter_ledger_rows(ledger_path: Path | str) -> Iterable[Mapping[str, Any]]:
    """Public re-export of the ledger reader for callers that only need
    to inspect rows (e.g. a launch script deciding `--limit`)."""
    return load_ledger_rows(ledger_path)


# ---------------------------------------------------------------------------
# CLI -- the only place the real, GPU-backed backbone loader is reachable.
# Tests inject a fake `_BackboneLike` into `build_aug_feature_store` directly
# and never exercise this function.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # pragma: no cover -- needs a GPU
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.aug_feature_cache",
        description="Build the T5b augmented feature store from an m2b_augment ledger.",
    )
    parser.add_argument("ledger", type=Path, help="augmented-views.jsonl")
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--main-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--augmentation-policy-hash", required=True)
    parser.add_argument("--expect-backbone-artifact-hash", required=True)
    parser.add_argument("--expect-processor-artifact-hash", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--vram-cap-gb", type=float, default=20.0,
        help="hard cap via torch.cuda.set_per_process_memory_fraction, so this "
        "runs alongside production services without starving them; applied "
        "after the backbone loads (the load itself briefly needs ~8 GB)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="kept for the old interface; must be 1 -- run one process per "
        "--shard instead, each with its own --vram-cap-gb",
    )
    parser.add_argument(
        "--shard", default=None, metavar="INDEX/COUNT",
        help="process only the views whose stable hash falls in shard INDEX of "
        "COUNT; run COUNT processes on the same --out-dir, then one pass "
        "without --shard to rewrite the complete manifest",
    )
    args = parser.parse_args(argv)
    if args.workers != 1:
        parser.error("--workers > 1 is not implemented; use --shard INDEX/COUNT per process")
    shard = None
    if args.shard is not None:
        index, _, count = args.shard.partition("/")
        shard = (int(index), int(count))

    import torch

    from unflatten.model.candidate_backbone import load_candidate_vision_backbone

    backbone = load_candidate_vision_backbone(device="cuda" if torch.cuda.is_available() else "cpu")
    # The cap binds AFTER loading: the loader moves the whole candidate (~8 GB)
    # to the GPU before offloading its language half, so a cap sized for
    # extraction (2.34 GB peak) killed every shard at load (prod 2026-09-24).
    # Callers stagger process starts so the load transients never overlap.
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        fraction = min(1.0, args.vram_cap_gb / total_gb)
        torch.cuda.set_per_process_memory_fraction(fraction)
    print("== backbone loaded, vram cap applied", flush=True)
    summary = build_aug_feature_store(
        ledger_path=args.ledger,
        store_dir=args.store_dir,
        main_cache_dir=args.main_cache_dir,
        out_dir=args.out_dir,
        backbone=backbone,
        augmentation_policy_hash=args.augmentation_policy_hash,
        expect_backbone_artifact_hash=args.expect_backbone_artifact_hash,
        expect_processor_artifact_hash=args.expect_processor_artifact_hash,
        limit=args.limit,
        shard=shard,
    )
    print(json.dumps(summary, indent=1))
    return 0


def retarget_main(argv: list[str] | None = None) -> int:
    """`python -m unflatten.model.aug_feature_cache retarget ...` -- CPU only."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.aug_feature_cache retarget",
        description="Rewrite a built aug store's region targets from a rebuilt main cache.",
    )
    parser.add_argument("ledger", type=Path, help="augmented-views.jsonl the store was built from")
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--main-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard", default=None, metavar="INDEX/COUNT")
    args = parser.parse_args(argv)
    shard = None
    if args.shard is not None:
        index, _, count = args.shard.partition("/")
        shard = (int(index), int(count))
    summary = retarget_aug_feature_store(
        src_dir=args.src_dir, ledger_path=args.ledger, main_cache_dir=args.main_cache_dir,
        out_dir=args.out_dir, shard=shard,
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    if sys.argv[1:2] == ["retarget"]:
        sys.exit(retarget_main(sys.argv[2:]))
    sys.exit(main())

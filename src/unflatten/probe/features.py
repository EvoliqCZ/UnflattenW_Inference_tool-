"""Frozen vision-backbone feature extraction for the Stage-0 probe.

RFC004 requires the backbone to sit behind an interface exposing
`visual_features` and `multiscale_features`, with no hidden dimension
baked into anything downstream. This module is the first implementation
of that interface, over Qwen3-VL.

It deliberately loads **only the vision tower**. The language model is
~3.6 B of the 4 B parameters and contributes nothing to a visual-feature
probe; dropping it turns an 8 GB GPU residency into 0.83 GB, which is
what makes the probe runnable on a 12 GB development card.

Two environment quirks are handled here rather than at every call site:

- **The patch embedding is re-expressed as a matmul.** Some cuDNN builds
  raise `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` on the tower's
  `conv3d`, and disabling cuDNN makes PyTorch pick a fallback kernel that
  takes *23 seconds* for one page. The convolution's stride equals its
  kernel, so it is a non-overlapping patch projection — exactly a linear
  map — and the processor already hands over flattened patches. Rewriting
  it as a matmul is mathematically identical (bf16 rounding aside) and
  takes 0.2 ms. See `patch_embed_as_matmul`.

  **Any other loader of the same Qwen3-VL tower needs this too — it is
  not specific to this probe.** `candidate_loader.load_qwen_candidate`
  hit exactly this failure the first time it ran a real forward pass on a
  real device (2026-09-04); it now calls `probe.backbones.optimize_patch_embed`
  directly rather than carrying a second copy of the fix, but a THIRD
  loader of this tower — or a different tower with the same
  stride==kernel patch conv — needs to know this paragraph exists. If you
  are writing one, call `optimize_patch_embed` after moving the tower to
  its device; do not re-derive the matmul rewrite from scratch.
- **Native resolution.** Qwen3-VL has no fixed input size: the number of
  visual tokens follows the pixel count (a 16 px patch grid merged 2x2,
  so one token per 32x32 px). Page rasters are therefore passed at their
  own resolution and the caller controls detail through the render DPI,
  or caps it per run with `max_pixels`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

#: Default backbone. RFC004 OPEN-004-001 leaves the product choice to a
#: measured ablation; this is the Stage-0 candidate, not a decision.
DEFAULT_BACKBONE = "Qwen/Qwen3-VL-4B-Instruct"


@dataclass(frozen=True)
class BackboneFeatures:
    """One page's frozen features (RFC004 backbone interface, v0)."""

    #: Final vision-tower hidden states, one row per patch.
    visual_features: torch.Tensor
    #: Intermediate fused levels (Qwen3-VL "DeepStack"), one row per
    #: merged visual token. Empty when the backbone exposes none.
    multiscale_features: tuple[torch.Tensor, ...]
    #: (temporal, height, width) patch grid of the input raster.
    grid_thw: tuple[int, int, int]

    def pooled(self) -> torch.Tensor:
        """Mean-pool every level into one fixed-size vector.

        The probe fits linear heads on this: a fixed width regardless of
        page size, which a variable token count would not give. Pooling
        discards position, so it is a floor on what the features carry —
        a probe that succeeds on pooled features would only do better
        with the full grid.
        """
        parts = [self.visual_features.mean(dim=0).float()]
        parts.extend(level.mean(dim=0).float() for level in self.multiscale_features)
        return torch.cat(parts)

    def band_pooled(self, bands: int) -> torch.Tensor:
        """Mean-pool into `bands` horizontal strips of the page.

        `pooled` averages every token into one vector, which discards
        position by construction — a paragraph gap is a vertical distance
        between blocks, and a distance cannot survive an average. This
        keeps the vertical axis and nothing else: each level's token grid
        is averaged across its columns, then its rows are split into
        `bands` contiguous strips. The result is (bands, feature_dim),
        same parts in the same order as `pooled`, so a linear head over
        the flattened strips can form differences between page thirds
        that a pooled head has no term for.

        `bands=1` reproduces `pooled` exactly (up to float order): every
        grid row holds the same number of columns, so the mean of the row
        means IS the mean of the tokens. That equality is what makes the
        two representations comparable rather than merely adjacent.

        Rows come from the reported patch grid at the merge each level
        actually used, not at the configured one — see `infer_merge_size`.
        """
        from unflatten.probe.spatial import infer_merge_size

        if bands < 1:
            raise ValueError(f"bands must be at least 1, got {bands}")
        strips = []
        for part in (self.visual_features, *self.multiscale_features):
            merge = infer_merge_size(self.grid_thw, part.shape[0])
            rows = self.grid_thw[1] // merge
            cols = self.grid_thw[2] // merge
            if rows < bands:
                raise ValueError(
                    f"a {rows}-row grid cannot be split into {bands} bands; "
                    "an empty band would be a mean over no tokens"
                )
            by_row = part.reshape(rows, cols, -1).float().mean(dim=1)
            edges = [index * rows // bands for index in range(bands + 1)]
            strips.append(
                torch.stack(
                    [by_row[a:b].mean(dim=0) for a, b in zip(edges, edges[1:])]
                )
            )
        return torch.cat(strips, dim=1)

    def dense(
        self, *, at_merged: bool = False, include_levels: bool = True
    ) -> torch.Tensor:
        """Per-token features, optionally with the multiscale levels.

        `at_merged` returns the *coarse* grid instead: the final layer is
        mean-pooled over each merged cell and the levels are used as they
        are. That is a quarter of the rows for the same information
        content per cell — 16.7 MB per page instead of 67 — which is the
        difference between a cache that fits on disk and one that does
        not. Whether the coarser grid costs accuracy is a measurement,
        not an assumption.

        `include_levels=False` drops the levels but keeps the grid choice,
        so the coarse grid is reachable carrying the final layer alone.
        Grid resolution and feature richness are then two flags rather
        than one, and each can be measured against the other held fixed —
        which is the only way `at_merged` costs anything measurable.

        DeepStack levels sit on the *merged* grid — a quarter of the patch
        count — so they cannot simply be concatenated. Each merged token
        is repeated over the patches it covers, which is exactly the
        region it summarises.

        Without this a spatial head sees only the final layer, and a
        backbone whose whole advantage is multi-level fusion is compared
        with that advantage switched off.
        """
        rows = self.visual_features.shape[0]
        levels = self.multiscale_features if include_levels else ()
        if at_merged:
            return self._dense_merged(rows, levels)
        if not levels:
            return self.visual_features
        _, height, width = self.grid_thw
        parts = [self.visual_features]
        for level in levels:
            count = level.shape[0]
            if count == rows:
                parts.append(level)
                continue
            ratio = rows // count if count else 0
            merge = int(round(ratio**0.5))
            if (
                merge < 1
                or merge * merge * count != rows
                or height % merge
                or width % merge
            ):
                raise ValueError(
                    f"multiscale level of {count} rows does not tile a "
                    f"{height}x{width} patch grid of {rows} rows"
                )
            level_rows, level_cols = height // merge, width // merge
            grid = level.reshape(level_rows, level_cols, -1)
            grid = grid.repeat_interleave(merge, dim=0).repeat_interleave(merge, dim=1)
            parts.append(grid.reshape(rows, -1))
        return torch.cat(parts, dim=1)

    def _dense_merged(
        self, rows: int, levels: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Final layer pooled onto the multiscale grid, `levels` appended.

        The coarse grid is *defined* by the backbone's own merged tokens,
        so it needs `multiscale_features` for its geometry even when the
        levels themselves are not wanted. A backbone that exposes none is
        refused rather than served the fine grid: silently returning a
        different representation than the caller asked for is how a cache
        ends up mislabelled.

        A backbone whose levels sit on the SAME grid as the final layer is
        refused for the same reason, and it is not hypothetical: Granite's
        streams are assembled onto one shared grid, so the merge would come
        out 1 and the pooling would be the identity. The caller would get
        the fine grid back while `merged_grid=True` went into the cache
        identity — a false lineage claim, which is worse than a flag that
        does nothing.
        """
        if not self.multiscale_features:
            raise ValueError(
                "at_merged pools onto the backbone's merged-token grid, and "
                "this backbone exposes no multiscale levels to define one; "
                "the coarse grid cannot be produced"
            )
        _, height, width = self.grid_thw
        count = self.multiscale_features[0].shape[0]
        if count == rows:
            raise ValueError(
                f"at_merged pools onto a COARSER grid, and this backbone's "
                f"multiscale levels sit on the same {height}x{width} grid as "
                f"its final layer ({rows} rows each); there is no merged grid "
                "to pool onto, so the request cannot be honoured rather than "
                "silently answered with the fine grid"
            )
        ratio = rows // count if count else 0
        merge = int(round(ratio**0.5))
        if merge < 1 or merge * merge * count != rows or height % merge or width % merge:
            raise ValueError(
                f"multiscale level of {count} rows does not tile a "
                f"{height}x{width} patch grid of {rows} rows"
            )
        grid = self.visual_features.reshape(
            height // merge, merge, width // merge, merge, -1
        )
        pooled = grid.mean(dim=(1, 3)).reshape(count, -1)
        if not levels:
            return pooled
        return torch.cat([pooled, *levels], dim=1)


def _longest_edge(target: object) -> int | None:
    """`target.size["longest_edge"]`, or None if it has no such key.

    Duck-typed on purpose. The real processor's `size` is a `SizeDict`,
    which supports subscript and `in` but is NOT a dict subclass, so an
    `isinstance(size, dict)` test skipped a key that was plainly there
    and refused a budget it could have applied.
    """
    size = getattr(target, "size", None)
    if size is None:
        return None
    try:
        if "longest_edge" not in size:
            return None
        return int(size["longest_edge"])
    except (TypeError, KeyError, ValueError):
        return None


def apply_pixel_budget(processor: object, max_pixels: int) -> str:
    """Cap the processor's pixel budget and say what was actually set.

    Which attribute carries the budget depends on the transformers
    version — `size["longest_edge"]` on the newer image processors, a
    `max_pixels` attribute on the older ones — and a wrapper `Processor`
    keeps both on its nested `image_processor`. Setting the one that does
    not exist used to be a silent no-op: the probe then ran at native
    resolution and reported nothing either way, so `--max-pixels` could
    not be verified from its output. It is refused loudly instead.

    Every write is read back. A processor that accepts an assignment and
    keeps its old value would otherwise reproduce the original defect one
    layer down, and this is not hypothetical: `size` may be a property, a
    frozen container, or a copy handed out per read.

    Returns a description of what was set. That is what was CONFIGURED,
    not what the processor then produced — `verify_pixel_budget` is the
    evidence.
    """
    targets = [processor]
    nested = getattr(processor, "image_processor", None)
    if nested is not None:
        targets.append(nested)
    for target in targets:
        name = type(target).__name__
        if _longest_edge(target) is not None:
            size = target.size
            if hasattr(size, "longest_edge"):
                # a SizeDict is a dataclass: assign the field in place
                size.longest_edge = max_pixels
            else:
                target.size = {**dict(size), "longest_edge": max_pixels}
            if _longest_edge(target) != max_pixels:
                raise ValueError(
                    f"{name}.size['longest_edge'] did not take the pixel "
                    f"budget {max_pixels} (it still reads "
                    f"{_longest_edge(target)}); the write was discarded"
                )
            return f"{name}.size['longest_edge']={max_pixels}"
        if hasattr(target, "max_pixels"):
            target.max_pixels = max_pixels
            if getattr(target, "max_pixels", None) != max_pixels:
                raise ValueError(
                    f"{name}.max_pixels did not take the pixel budget "
                    f"{max_pixels}; the write was discarded"
                )
            return f"{name}.max_pixels={max_pixels}"
    raise ValueError(
        f"{type(processor).__name__} exposes neither size['longest_edge'] nor "
        f"max_pixels, so the pixel budget {max_pixels} cannot be applied; it "
        "would silently run at native resolution"
    )


#: Page raster the budget is verified against: A4 at 300 dpi, ~8.7 M
#: pixels — comfortably above any budget this probe runs at, so the
#: processor has to downscale it or the budget is not in force.
_BUDGET_PROBE_PAGE = (2480, 3508)


def verify_pixel_budget(
    processor: object, max_pixels: int, *, page: tuple[int, int] | None = None
) -> dict[str, int | tuple[int, int, int]]:
    """Measure what the processor actually produces under the budget.

    `apply_pixel_budget` reports what it configured, which is honest but
    is not evidence that the output changed. This feeds one oversized
    synthetic page through the processor — cheap, no model weights — and
    reads the patch grid back, so the run can state the resolution it
    measured rather than the one it asked for.
    """
    from PIL import Image as _Image

    width, height = page or _BUDGET_PROBE_PAGE
    batch = processor(
        images=_Image.new("RGB", (width, height), "white"), return_tensors="pt"
    )
    grid = batch["image_grid_thw"][0]
    thw = (int(grid[0]), int(grid[1]), int(grid[2]))
    patch = int(getattr(processor, "patch_size", 16))
    merge = int(getattr(processor, "merge_size", 2))
    patches = thw[1] * thw[2]
    pixels = patches * patch * patch
    if pixels > max_pixels:
        raise ValueError(
            f"the pixel budget {max_pixels} did not bound the processor: a "
            f"{width}x{height} page still produced a {thw[1]}x{thw[2]} patch "
            f"grid, i.e. {pixels} pixels ({patches // (merge * merge)} tokens)"
        )
    return {"tokens": patches // (merge * merge), "pixels": pixels, "grid": thw}


class FrozenVisionBackbone:
    """A document VLM's vision tower, frozen, as a feature extractor.

    Which tower and how to call it comes from `probe.backbones`; this
    class only owns the pixel budget and the pooling contract, so a new
    candidate is a registry entry rather than a change here.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_BACKBONE,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_pixels: int | None = None,
        tap: str | None = None,
    ) -> None:
        from unflatten.probe.backbones import load_vision_runtime, optimize_patch_embed

        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_pixels = max_pixels
        runtime = load_vision_runtime(model_id, dtype=dtype, tap=tap)
        self.runtime = runtime
        self.processor = runtime.processor
        # Store rasters high, feed the model what it can afford: one
        # visual token per merged patch, so the budget IS the token count.
        # Downscaling here is always possible; upscaling detail is not,
        # which is why the stored DPI is the high one.
        #: What the pixel budget configured, and what the processor was
        #: then MEASURED to produce under it. Both None when no budget
        #: was asked for. Printed by the runners, so a run's resolution
        #: is evidence in its own log rather than an assumption.
        self.pixel_budget_applied: str | None = None
        self.pixel_budget_measured: dict[str, Any] | None = None
        if max_pixels is not None:
            self.pixel_budget_applied = apply_pixel_budget(
                self.processor, max_pixels
            )
            self.pixel_budget_measured = verify_pixel_budget(
                self.processor, max_pixels
            )
        self.visual = runtime.module.to(device)
        #: Which feature tap this instance loaded, or None for a backbone
        #: with only one. Read by the cache identity.
        self.tap = runtime.tap
        # AFTER the device move: the rewrite closes over the weight tensor.
        # Skipped for towers fed whole rasters: there the rewrite is not a
        # slower-or-faster choice but a wrong one.
        if runtime.flattened_patch_input:
            optimize_patch_embed(self.visual)

    @property
    def feature_dim(self) -> int:
        """Width of `BackboneFeatures.pooled()` for this backbone."""
        return self.runtime.feature_dim

    @property
    def param_count(self) -> int:
        """Parameters resident for the frozen tower."""
        return self.runtime.param_count

    @torch.inference_mode()
    def extract(self, image: Image.Image) -> BackboneFeatures:
        """Frozen features for one page raster, at its native resolution."""
        batch = self.processor(images=image, return_tensors="pt")
        if self.runtime.prepare is not None:
            # AnyRes: the batch carries tiles and a page size rather than a
            # patch grid, and the grid comes from `grid_of` below.
            raw_pixels, aux = self.runtime.prepare(batch)
            pixel_values = raw_pixels.to(self.device, self.dtype)
            if isinstance(aux, torch.Tensor):
                aux = aux.to(self.device)
            last_hidden, levels = self.runtime.forward(pixel_values, aux)
            grid_thw = self.runtime.grid_of(batch, int(last_hidden.shape[0]))
            return BackboneFeatures(
                visual_features=last_hidden,
                multiscale_features=tuple(levels),
                grid_thw=grid_thw,
            )
        pixel_values = batch["pixel_values"].to(self.device, self.dtype)
        grid = batch.get("image_grid_thw")
        if grid is not None:
            grid = grid.to(self.device)
        last_hidden, levels = self.runtime.forward(pixel_values, grid)
        grid_thw = (
            (int(grid[0][0]), int(grid[0][1]), int(grid[0][2]))
            if grid is not None
            else (1, 0, 0)
        )
        return BackboneFeatures(
            visual_features=last_hidden,
            multiscale_features=tuple(levels),
            grid_thw=grid_thw,
        )

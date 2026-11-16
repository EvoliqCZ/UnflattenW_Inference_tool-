"""On-the-fly visual path for v16 (DEC-056 addendum A, variant c).

v15b read `pooler_output` features from a ~42 MB/page cache. v16 trains with
DeepStack, and the owner rejected caching the three extra levels (~15 TB):
the frozen Qwen3-VL ViT runs inside the training step instead. This module
is that path, default off -- nothing here is reached unless a trainer is
handed an `OnTheFlyVisualEncoder`:

- **`VisualInputPolicy`** -- what the rows of a page-image exact-text cache
  are bound to instead of a feature-cache id: the ViT revision
  (`backbone_artifact_hash`), which blocks feed DeepStack, the compute dtype,
  the processor artifact and its pixel budget / patch geometry. Its
  `policy_hash()` is the cache identity's `feature_tap`
  (`"onfly-visual:<hash>"`, `exact_text_cache.PAGE_IMAGE_FEATURE_TAP_PREFIX`)
  and a run-manifest field.
- **`page_pixels`** -- one PNG through the processor on CPU (a data worker),
  refused unless the grid equals what the policy derives from the image size.
- **`OnTheFlyAugmentation`** -- DEC-040's policy (families, recipes, the
  ~20 % selection share, the image ops and the box transform), re-drawn per
  `(seed, step, page)`: every epoch's slot of a page gets its own view.
- **`OnTheFlyDrawDataset`** -- the S4 dataset addressed by PLAN DRAW index,
  so the worker knows the step and can apply that draw's view.
- **`OnTheFlyVisualEncoder`** -- the ViT under `no_grad` on the GPU in the
  step, filling a collated batch's visual features and DeepStack levels.

Fresh S4 speed knobs (DEC-058, owner 2026-09-30), every one default off and
decided by the pilot:

- `OnTheFlyVisualEncoder(batch_pages=True)` -- one tower call for a
  multi-page batch (a no-op at the frozen batch size 1; used by eval and any
  batch > 1).
- **`ViTFeatureCache`** -- the epoch-1 write-through fallback: a clean
  (non-augmented) draw's pooler output + DeepStack levels stored fp8 e4m3
  with a float32 scale per token row, keyed by page-image sha256 + ViT
  revision + visual-input policy hash. Miss and hit hand the model the SAME
  fp8 round trip; augmented draws are always recomputed.
- **`prefetch_materialized`** -- the next step's ViT issued (on a side CUDA
  stream when the device is CUDA) before the current step's forward/backward
  runs; bit-identical to the plain path because the tower is frozen and the
  augmentation view is a pure function of `(seed, step, page)`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from unflatten.model.hashing import domain_hash

#: `hashing.domain_hash` domain of `VisualInputPolicy.policy_hash`.
VISUAL_INPUT_POLICY_DOMAIN = "unflatten.visual-input-policy/v1"
#: Domain of the per-draw augmentation salt (`OnTheFlyAugmentation`).
ONFLY_AUGMENTATION_DRAW_DOMAIN = "unflatten.onfly-augmentation-draw/v1"
#: Domain of the per-attempt salt a legibility redraw re-draws the RECIPE under
#: (RFC003-ERRATA-004 E3). Selection and family never see it.
ONFLY_AUGMENTATION_REDRAW_DOMAIN = "unflatten.onfly-augmentation-redraw/v1"
#: Explicit runtime profile: its policy id/hash bind the repaired guard and retry.
READABLE_RETRY_POLICY_ID = "augmentation-v2-readable-retry"
INK_FREE_REDRAW_DOMAIN = "unflatten.onfly-augmentation-occluder-redraw/v1"
ENVELOPE_REDRAW_DOMAIN = "unflatten.onfly-augmentation-envelope-redraw/v1"
BALANCED_SELECTION_DOMAIN = "unflatten.onfly-augmentation-balanced-selection/v1"
#: Visual token kinds a checkpoint was trained on (`visual_feature_tap`).
POOLER_OUTPUT_TAP = "pooler_output"

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class OnTheFlyVisualError(ValueError):
    """The on-the-fly visual input disagrees with what the row or the
    policy binds (grid, token count, DeepStack taps, image bytes)."""


def visual_feature_tap(deepstack_visual_indexes: Sequence[int]) -> str:
    """The checkpoint/bundle flag: `"pooler_output"` for every v15b model,
    `"pooler_output+deepstack:5,11,17"` for a v16 DeepStack model."""
    if not deepstack_visual_indexes:
        return POOLER_OUTPUT_TAP
    return f"{POOLER_OUTPUT_TAP}+deepstack:" + ",".join(str(i) for i in deepstack_visual_indexes)


def checkpoint_visual_feature_tap(state: Mapping[str, Any]) -> str:
    """The tap a checkpoint (or bundle manifest) was trained on. Absent --
    every v15b checkpoint -- means `"pooler_output"`."""
    return str(state.get("visual_feature_tap") or POOLER_OUTPUT_TAP)


def require_checkpoint_visual_feature_tap(
    state: Mapping[str, Any], *, expected: str, what: str
) -> None:
    """Refuse a checkpoint whose training tap is not the one this runtime/
    run computes: a v15b checkpoint never runs with DeepStack, a DeepStack
    checkpoint never without (DEC-056 point 2)."""
    actual = checkpoint_visual_feature_tap(state)
    if actual != expected:
        raise OnTheFlyVisualError(
            f"{what}: the checkpoint was trained with visual_feature_tap {actual!r} "
            f"but this run computes {expected!r} -- a checkpoint trained without "
            "DeepStack never runs with it, and one trained with it never without"
        )


def visual_input_manifest_block(
    policy: VisualInputPolicy,
    augmentation: OnTheFlyAugmentation | None,
    *,
    feature_cache: bool = False,
    batch_pages: bool = False,
    external_views: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The run manifest's `visual_input` block (S4 1.4.0 / S6): the policy
    hash, the checkpoint tap and the augmentation seed + policy hash.

    Two fresh-S4 knobs change what the model is fed and are bound here, each
    written ONLY when on (so every existing block keeps its exact bytes):
    `feature_cache` (epochs 2+ train on the fp8 round trip of clean views)
    and `batch_pages` (one tower call per multi-page batch), and
    `external_views` (the external-views table's `manifest_fields()`).
    Prefetch overlap
    is bit-identical and is not bound."""
    block: dict[str, Any] = {
        "mode": "onfly_vit",
        "visual_input_policy_hash": policy.policy_hash(),
        "visual_feature_tap": policy.visual_feature_tap(),
        "augmentation": None if augmentation is None else augmentation.manifest_fields(),
    }
    if feature_cache:
        block["feature_cache"] = {"format": VIT_FEATURE_CACHE_FORMAT}
    if batch_pages:
        block["batch_pages"] = True
    if external_views is not None:
        block["external_views"] = dict(external_views)
    return block


@dataclass(frozen=True)
class VisualInputPolicy:
    """The frozen visual front end a page-image cache and a run bind.

    `deepstack_visual_indexes` empty means pooler output only. The patch
    geometry and pixel budget are the processor's own (`patch_size`,
    `temporal_patch_size`, `merge_size`, `size.shortest_edge`/
    `longest_edge`); `grid_thw_for` reproduces its `smart_resize`, so the
    cache builder can write each row's grid from the PNG size alone and
    training can refuse a processor whose output disagrees.
    """

    backbone_artifact_hash: str
    processor_artifact_hash: str
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int
    deepstack_visual_indexes: tuple[int, ...] = ()
    compute_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.compute_dtype not in _DTYPES:
            raise ValueError(f"compute_dtype {self.compute_dtype!r} not in {sorted(_DTYPES)}")
        for name in ("patch_size", "temporal_patch_size", "merge_size", "min_pixels"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_pixels < self.min_pixels:
            raise ValueError("max_pixels must be >= min_pixels")

    @property
    def deepstack(self) -> bool:
        return bool(self.deepstack_visual_indexes)

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.compute_dtype]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> VisualInputPolicy:
        """The inverse of `payload`; unknown or missing keys are refused."""
        names = {f.name for f in dataclasses.fields(cls)}
        if set(payload) != names:
            raise ValueError(
                f"visual input policy keys {sorted(payload)} != {sorted(names)}"
            )
        values = dict(payload)
        values["deepstack_visual_indexes"] = tuple(
            int(i) for i in values["deepstack_visual_indexes"]
        )
        return cls(**values)

    def payload(self) -> dict[str, Any]:
        return {
            "backbone_artifact_hash": self.backbone_artifact_hash,
            "processor_artifact_hash": self.processor_artifact_hash,
            "patch_size": int(self.patch_size),
            "temporal_patch_size": int(self.temporal_patch_size),
            "merge_size": int(self.merge_size),
            "min_pixels": int(self.min_pixels),
            "max_pixels": int(self.max_pixels),
            "deepstack_visual_indexes": [int(i) for i in self.deepstack_visual_indexes],
            "compute_dtype": self.compute_dtype,
        }

    def policy_hash(self) -> str:
        return domain_hash(VISUAL_INPUT_POLICY_DOMAIN, self.payload())

    def feature_tap(self) -> str:
        """`ExactTextCacheIdentity.feature_tap` of a cache bound to this policy."""
        from unflatten.model.exact_text_cache import PAGE_IMAGE_FEATURE_TAP_PREFIX

        return PAGE_IMAGE_FEATURE_TAP_PREFIX + self.policy_hash()

    def visual_feature_tap(self) -> str:
        return visual_feature_tap(self.deepstack_visual_indexes)

    def grid_thw_for(self, width: int, height: int) -> tuple[int, int, int]:
        """The processor's patch grid for a still image of this size: one
        temporal patch, `smart_resize` to a multiple of patch*merge inside
        the pixel budget (transformers' own function, not a copy)."""
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        resized_h, resized_w = smart_resize(
            int(height),
            int(width),
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return 1, resized_h // self.patch_size, resized_w // self.patch_size

    def visual_tokens_for(self, grid_thw: Sequence[int]) -> int:
        temporal, height, width = (int(v) for v in grid_thw)
        return temporal * (height // self.merge_size) * (width // self.merge_size)


def pooler_only_policy(policy: VisualInputPolicy) -> VisualInputPolicy:
    """The same pixels, processor and tower without DeepStack taps -- how a
    checkpoint trained without DeepStack (v15b) reads a page-image cache
    built under a DeepStack policy."""
    return dataclasses.replace(policy, deepstack_visual_indexes=())


def visual_input_policy_from_processor(
    processor: Any,
    *,
    backbone_artifact_hash: str,
    processor_artifact_hash: str,
    deepstack_visual_indexes: Sequence[int] = (),
    compute_dtype: str = "bfloat16",
) -> VisualInputPolicy:
    """Read the geometry off a loaded Qwen image processor (the real pin's
    `patch_size=16, temporal_patch_size=2, merge_size=2`, `size` =
    `{shortest_edge: 65536, longest_edge: 16777216}`)."""
    size = getattr(processor, "size", None)
    # transformers 5.x hands a `SizeDict` (attributes), older versions a dict.
    if isinstance(size, Mapping):
        min_pixels, max_pixels = size.get("shortest_edge"), size.get("longest_edge")
    else:
        min_pixels = getattr(size, "shortest_edge", None)
        max_pixels = getattr(size, "longest_edge", None)
    if min_pixels is None or max_pixels is None:
        raise OnTheFlyVisualError("processor declares no pixel budget (size / min/max_pixels)")
    return VisualInputPolicy(
        backbone_artifact_hash=backbone_artifact_hash,
        processor_artifact_hash=processor_artifact_hash,
        patch_size=int(processor.patch_size),
        temporal_patch_size=int(processor.temporal_patch_size),
        merge_size=int(processor.merge_size),
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
        deepstack_visual_indexes=tuple(int(i) for i in deepstack_visual_indexes),
        compute_dtype=compute_dtype,
    )


# ---------------------------------------------------------------------------
# CPU side: PNG -> pixel_values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PagePixels:
    pixel_values: torch.Tensor  # float32 [patches, C*T*P*P]
    grid_thw: torch.Tensor  # int64 [3]
    visual_tokens: int


def open_png(png_bytes: bytes) -> Any:
    """The page raster exactly as the feature-cache build handed it to the
    processor (`build_cache.PreparedPage.image`): opened and loaded, mode
    untouched."""
    from PIL import Image

    image = Image.open(io.BytesIO(png_bytes))
    image.load()
    return image


def page_pixels(png_bytes: bytes, processor: Any, policy: VisualInputPolicy) -> PagePixels:
    """One page through the processor. Refused unless the processor's grid
    equals `policy.grid_thw_for(size)` -- a processor configured with
    another pixel budget than the policy the cache was built under would
    otherwise silently change the token count of every page."""
    image = open_png(png_bytes)
    width, height = image.size
    out = processor(images=image, return_tensors="pt")
    grid = torch.as_tensor(out["image_grid_thw"]).reshape(-1, 3)[0].to(torch.int64)
    expected = policy.grid_thw_for(width, height)
    if tuple(int(v) for v in grid) != expected:
        raise OnTheFlyVisualError(
            f"processor grid {grid.tolist()} != policy grid {list(expected)} for a "
            f"{width}x{height} page -- the processor is not the one the policy binds"
        )
    return PagePixels(
        pixel_values=torch.as_tensor(out["pixel_values"]).to(torch.float32),
        grid_thw=grid,
        visual_tokens=policy.visual_tokens_for(expected),
    )


# ---------------------------------------------------------------------------
# Augmentation: DEC-040's policy, re-drawn per (seed, step, page)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnTheFlyView:
    family: str
    ops: tuple[dict[str, Any], ...]
    seeds: tuple[int, ...]
    transform_chain_hash: str
    schema_major: int = 1


class OnTheFlyAugmentation:
    """DEC-040's augmentation policy with a per-draw salt.

    `m2b_augment` fixes ONE view per page (`selection.is_selected(bundle,
    page, policy_hash)`). Here the policy hash is salted with
    `(seed, step)`: `draw_hash = H(policy_hash, seed, step)` goes where
    `policy_hash` went, so selection (the policy's own numerator/denominator,
    ~20 %), family and every op parameter are re-drawn for each draw, yet
    are a pure function of `(seed, step, document_bundle_id, page_index)` --
    a resumed run, a second worker count or a rerun gets the same pixels.
    The ops and their geometry are `augment.ops`/`augment.geometry`
    unchanged.
    """

    def __init__(self, policy: Mapping[str, Any] | None = None, *, seed: int) -> None:
        from unflatten.dataset.augment import policy as aug_policy

        self.policy = dict(policy) if policy is not None else aug_policy.default_policy()
        aug_policy.validate_policy(self.policy)
        self.policy_hash = aug_policy.policy_hash(self.policy)
        self.schema_major = aug_policy.policy_schema_major(self.policy)
        self.seed = int(seed)
        self.readable_retry = self.schema_major == 2 and (
            self.policy["policy_id"] == READABLE_RETRY_POLICY_ID
        )

    def manifest_fields(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "policy_id": self.policy["policy_id"],
            "policy_hash": self.policy_hash,
            "draw_domain": ONFLY_AUGMENTATION_DRAW_DOMAIN,
            "selection": [
                self.policy["selection_numerator"],
                self.policy["selection_denominator"],
            ],
        }

    def draw_hash(self, step: int) -> str:
        from unflatten.dataset.identity import domain_hash as dataset_domain_hash

        return dataset_domain_hash(
            ONFLY_AUGMENTATION_DRAW_DOMAIN,
            {"policy_hash": self.policy_hash, "seed": self.seed, "step": int(step)},
        )

    def view(self, document_bundle_id: str, page_index: int, step: int) -> OnTheFlyView | None:
        from unflatten.dataset.augment import selection

        salted = self.draw_hash(step)
        selected = (
            self._balanced_selected(step) if self.readable_retry else
            selection.is_selected(document_bundle_id, page_index, salted, self.policy)
        )
        if not selected:
            return None
        family = selection.select_family(document_bundle_id, page_index, salted)
        ops, seeds = selection.resolve_recipe(
            document_bundle_id, page_index, family, self.policy, salted
        )
        return OnTheFlyView(
            family=family,
            ops=tuple(ops),
            seeds=tuple(seeds),
            transform_chain_hash=selection.transform_chain_hash(
                document_bundle_id, page_index, family, ops, salted
            ),
            schema_major=self.schema_major,
        )

    def _balanced_selected(self, step: int) -> bool:
        """Select fixed slots before pixels: no replacement after a rejected view.

        This profile requires batch_size=1, so step-1 is the global plan draw.
        Blocks continue across resume; a partial final block keeps its slots.
        """
        from unflatten.dataset.identity import domain_hash as dataset_domain_hash

        denominator = self.policy["selection_denominator"]
        block, slot = divmod(int(step) - 1, denominator)
        ranked = sorted(range(denominator), key=lambda position: (
            dataset_domain_hash(BALANCED_SELECTION_DOMAIN, {
                "policy_hash": self.policy_hash, "seed": self.seed,
                "block": block, "slot": position,
            }), position,
        ))
        return slot in ranked[:self.policy["selection_numerator"]]

    def redraw_recipe(
        self,
        view: OnTheFlyView,
        document_bundle_id: str,
        page_index: int,
        step: int,
        attempt: int,
    ) -> OnTheFlyView:
        """`view`'s family with a freshly drawn recipe (legibility redraw, `attempt >= 1`).

        Selection and family are NOT re-decided: only the recipe is drawn again,
        from `H(ONFLY_AUGMENTATION_REDRAW_DOMAIN, {policy_hash, seed, step,
        attempt})` and hashed into a new `transform_chain_hash`, so the share
        of augmented views and the family ratio stay what the policy declares.
        """
        from unflatten.dataset.augment import selection
        from unflatten.dataset.identity import domain_hash as dataset_domain_hash

        if attempt < 1:
            raise OnTheFlyVisualError(f"redraw_recipe: attempt must be >= 1, got {attempt}")
        salted = dataset_domain_hash(
            ONFLY_AUGMENTATION_REDRAW_DOMAIN,
            {
                "policy_hash": self.policy_hash,
                "seed": self.seed,
                "step": int(step),
                "attempt": int(attempt),
            },
        )
        ops, seeds = selection.resolve_recipe(
            document_bundle_id, page_index, view.family, self.policy, salted
        )
        return OnTheFlyView(
            family=view.family,
            ops=tuple(ops),
            seeds=tuple(seeds),
            transform_chain_hash=selection.transform_chain_hash(
                document_bundle_id, page_index, view.family, ops, salted
            ),
            schema_major=self.schema_major,
        )

    def redraw_ink_free_recipe(
        self, view: OnTheFlyView, document_bundle_id: str, page_index: int, step: int, attempt: int
    ) -> OnTheFlyView | None:
        """Same family/count/parameter limits, weighted over its ink-free ops.

        Used only after a retained-ink rejection in the named retry profile.
        Insufficient eligible ops leave the ordinary redraw in control.
        """
        from unflatten.dataset.augment import ops_v2

        eligible = [
            op for op in self.policy["families"][view.family]["ops"]
            if ops_v2.GEOMETRY_CLASS[op["op"]] == "occluder"
        ]
        return self._redraw_from_ops(
            view, document_bundle_id, page_index, step, attempt, eligible, INK_FREE_REDRAW_DOMAIN
        )

    def redraw_envelope_recipe(
        self, view: OnTheFlyView, document_bundle_id: str, page_index: int, step: int,
        attempt: int, reason: str | None, stats: Any,
    ) -> OnTheFlyView | None:
        """Address the rejected band using existing family ops and parameter ranges.

        Low contrast/sharpness avoids compounding blur or uneven illumination.
        A repeated low-band failure tries ink-free artifacts instead. Every
        result still passes the same full guard; no failed view is repaired.
        """
        envelope = self.policy["legibility"]["envelope"]
        if not reason or not reason.startswith("envelope:") or stats is None or envelope is None:
            return None
        metric = reason.removeprefix("envelope:")
        low = getattr(stats, metric) < envelope[metric]["low"]
        if low and attempt >= 3:
            return self.redraw_ink_free_recipe(view, document_bundle_id, page_index, step, attempt)
        gentle = attempt >= 2 and (low or metric == "ink_contrast")
        tone = "paper_tint" if view.family == "scan_simulation" else "uneven_light"
        if gentle and (not low or metric != "ink_contrast"):
            names = {tone, "soft_resample", "noise" if low else "toner_faded"}
        elif low:
            third = "noise"
            names = {"soft_resample", "page_offset", third}
        elif metric == "ink_sharpness":
            names = {"soft_resample", "page_offset", "blur"}
        elif metric == "ink_contrast":
            names = {"soft_resample", "page_offset", tone}
        else:
            return None
        eligible = [op for op in self.policy["families"][view.family]["ops"] if op["op"] in names]
        if gentle:
            eligible = _gentle_retry_ops(eligible)
        return self._redraw_from_ops(
            view, document_bundle_id, page_index, step, attempt, eligible, ENVELOPE_REDRAW_DOMAIN
        )

    def _redraw_from_ops(
        self, view: OnTheFlyView, document_bundle_id: str, page_index: int, step: int,
        attempt: int, eligible: list[Any], domain: str,
    ) -> OnTheFlyView | None:
        from unflatten.dataset.augment import selection
        from unflatten.dataset.identity import domain_hash as dataset_domain_hash

        if attempt < 1:
            raise OnTheFlyVisualError(f"redraw recipe: attempt must be >= 1, got {attempt}")
        if len(eligible) < self.policy["ops_per_view"][0]:
            return None
        scoped = {**self.policy, "families": {
            **self.policy["families"], view.family: {"ops": eligible}
        }}
        salted = dataset_domain_hash(domain, {
            "policy_hash": self.policy_hash, "seed": self.seed,
            "step": int(step), "attempt": int(attempt),
        })
        ops, seeds = selection.resolve_recipe(
            document_bundle_id, page_index, view.family, scoped, salted
        )
        return OnTheFlyView(
            family=view.family, ops=tuple(ops), seeds=tuple(seeds), schema_major=2,
            transform_chain_hash=selection.transform_chain_hash(
                document_bundle_id, page_index, view.family, ops, salted
            ),
        )


def _gentle_retry_ops(ops: list[Any]) -> list[Any]:
    """Keep physical limits; avoid double downsampling during contrast correction.

    Noise strength is never increased to satisfy the sharpness measurement.
    Higher DPI/scale preserves detail; stronger tone lowers excessive contrast
    without adding another geometric resample. Each candidate is still guarded.
    """
    result = []
    for op in ops:
        params = dict(op["params"])
        upper = {"soft_resample": ["dpi"], "page_offset": ["scale"],
                 "uneven_light": ["strength"]}.get(op["op"], [])
        lower = ["tint_r", "tint_g", "tint_b"] if op["op"] == "paper_tint" else []
        for name in upper:
            low, high = params[name]
            params[name] = [low + .75 * (high - low), high]
        for name in lower:
            low, high = params[name]
            params[name] = [low, low + .25 * (high - low)]
        result.append({**op, "params": params})
    return result


def apply_view(png_bytes: bytes, view: OnTheFlyView) -> tuple[bytes, Any]:
    """The view's image ops (`augment.ops.apply_recipe`) -> `(png, matrix)`."""
    from unflatten.dataset.augment import ops

    return ops.apply_recipe(png_bytes, list(view.ops), list(view.seeds))


@dataclass(frozen=True)
class ServedView:
    """What `serve_view` hands the dataset.

    `view` is the accepted view (the redrawn one after a legibility redraw) or
    None for a clean page. `matrix` is the forward pixel map (None when clean).
    `report` is None on a schema-1 policy, else the counters of design §1.7.
    """

    png: bytes
    matrix: Any | None
    view: OnTheFlyView | None
    report: dict[str, Any] | None
    #: External capture with no usable alignment: region boxes are dropped.
    collapse_regions: bool = False
    #: External capture lost the page's colour: the colour targets are unseen.
    color_lost: bool = False


def _empty_report(*, selected: int) -> dict[str, Any]:
    return {
        "selected": selected,
        "attempts": 0,
        "applied": [],
        "skipped": {},
        "legibility_rejected": {},
        "legibility_exhausted": 0,
    }


def serve_view(
    png: bytes,
    augmentation: OnTheFlyAugmentation,
    document_bundle_id: str,
    page_index: int,
    step: int,
) -> ServedView:
    """The page's served pixels under `augmentation` (RFC003-ERRATA-004 E3).

    Schema 1: `apply_view`, exactly today's path, no legibility guard. Schema 2:
    a selected page is tried up to `1 + max_redraws` times - attempt 0 is
    `augmentation.view`, later attempts `redraw_recipe` - and the first view
    inside the legibility limits wins; when none does, the clean page is served
    and `legibility_exhausted` is counted.
    """
    view = augmentation.view(document_bundle_id, page_index, step)
    if augmentation.schema_major == 1:
        if view is None:
            return ServedView(png, None, None, None)
        served_png, matrix = apply_view(png, view)
        return ServedView(served_png, matrix, view, None)
    if view is None:
        return ServedView(png, None, None, _empty_report(selected=0))
    return _serve_v2(png, augmentation, view, document_bundle_id, page_index, step)


def _serve_v2(
    png: bytes,
    augmentation: OnTheFlyAugmentation,
    view: OnTheFlyView,
    document_bundle_id: str,
    page_index: int,
    step: int,
) -> ServedView:
    policy = augmentation.policy
    limits = policy["legibility"]
    clean_ink, reference = _v2_clean_ink(png, augmentation)
    report = _empty_report(selected=1)
    if augmentation.readable_retry:
        report.update(
            ink_free_retry_attempts=0, ink_free_retry_applied=0,
            envelope_retry_attempts=0, envelope_retry_applied=0,
        )
    reason = stats = None
    for attempt in range(limits["max_redraws"] + 1):
        kind = None
        if attempt:
            view, kind = _v2_redraw(
                augmentation, view, document_bundle_id, page_index, step, attempt, reason, stats
            )
        if kind is not None:
            key = f"{kind}_retry_attempts"
            report[key] = report.get(key, 0) + 1
        report["attempts"] = attempt + 1
        out_png, matrix, recipe, ok, reason, stats = _v2_candidate(
            png, view, policy, clean_ink, reference
        )
        if ok:
            report["applied"] = list(recipe["applied"])
            report["skipped"] = dict(recipe["skipped"])
            if kind is not None:
                report[f"{kind}_retry_applied"] = 1
            return ServedView(out_png, matrix, view, report)
        rejected = report["legibility_rejected"]
        rejected[reason] = rejected.get(reason, 0) + 1
    report["legibility_exhausted"] = 1
    return ServedView(png, None, None, report)


def _v2_redraw(augmentation, view, bundle, page, step, attempt, reason, stats):
    if augmentation.readable_retry:
        if reason == "retained_ink":
            retry = augmentation.redraw_ink_free_recipe(view, bundle, page, step, attempt)
            kind = "ink_free"
        else:
            retry = augmentation.redraw_envelope_recipe(
                view, bundle, page, step, attempt, reason, stats
            )
            kind = "envelope"
        if retry is not None:
            return retry, kind
    return augmentation.redraw_recipe(view, bundle, page, step, attempt), None


def _v2_clean_ink(png: bytes, augmentation: OnTheFlyAugmentation) -> tuple[Any, Any]:
    import numpy as np

    from unflatten.dataset.augment import ink, legibility

    threshold = augmentation.policy["ink_free"]["ink_threshold"]
    if augmentation.readable_retry:
        reference = legibility.ink_reference(png, threshold)
        return reference.ink, reference
    # Existing frozen v2 policies keep their exact historical behavior.
    return ink.ink_mask(np.asarray(open_png(png).convert("RGB")), threshold), None


def _v2_candidate(png: bytes, view: OnTheFlyView, policy: Any, clean_ink: Any, reference: Any):
    from unflatten.dataset.augment import legibility, ops_v2

    ink_free = policy["ink_free"]
    out_png, matrix, recipe = ops_v2.apply_recipe_v2(
        png, list(view.ops), list(view.seeds), ink_free=ink_free
    )
    if reference is not None and not recipe["applied"]:
        return out_png, matrix, recipe, False, "no_applied_ops", None
    stats = legibility.view_statistics(
        out_png, clean_ink=clean_ink, matrix=matrix,
        ink_threshold=ink_free["ink_threshold"], reference=reference,
    )
    ok, reason = legibility.within_envelope(stats, policy["legibility"])
    return out_png, matrix, recipe, ok, reason, stats


# ---------------------------------------------------------------------------
# Dataset: rows -> pixels, addressed by plan draw index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnTheFlyDraw:
    """One served draw: the sample with pixels attached, plus what the
    exposure ledger records (`view_id`) and whether it was augmented."""

    sample: Any
    view_id: str
    augmented: bool
    aug_report: Mapping[str, Any] | None = None
    #: `"scan"`/`"photo"`/`"augraphy"` for an external capture, else None.
    external_kind: str | None = None


def read_page_png(store_dir: Path, page_image: Mapping[str, Any]) -> bytes:
    """The row's PNG from the store, sha256 + size verified (`read_object`)."""
    from unflatten.dataset.gates.store import read_object

    return read_object(Path(store_dir), dict(page_image))


def attach_pixels(
    sample: Any,
    *,
    store_dir: Path,
    processor: Any,
    policy: VisualInputPolicy,
    view: OnTheFlyView | None = None,
    served: ServedView | None = None,
    png: bytes | None = None,
) -> Any:
    """`sample` (an `S4TrainingSample`) with its page's `pixel_values`
    attached, after `view` (if any) was applied to the image AND to its
    region boxes by the same matrix (`aug_feature_cache.
    transform_region_targets`, the DEC-040 store's own box transform).
    `served` (from `serve_view`) replaces `view`: its pixels and matrix are
    used as they are; `png` is the already-read clean page (else it is read
    from the store). Refused if the row is not a page-image row or the processor's grid
    disagrees with the row's."""
    base = sample.base
    if base.page_image is None:
        raise OnTheFlyVisualError(
            f"{base.document_bundle_id} page {base.page_index}: a feature row reached "
            "the on-the-fly path -- it needs a page-image (7.0.0) cache"
        )
    if png is None:
        png = read_page_png(store_dir, base.page_image)
    region_targets = sample.region_targets
    collapsed_ids: frozenset[str] = frozenset()
    if served is None and view is not None:
        view_png, view_matrix = apply_view(png, view)
        served = ServedView(view_png, view_matrix, view, None)
    if served is not None:
        if served.collapse_regions:
            from unflatten.model.aug_feature_cache import collapse_region_targets

            region_targets, collapsed_ids = collapse_region_targets(region_targets)
        elif served.matrix is not None:
            from unflatten.model.aug_feature_cache import (
                transform_region_targets_with_collapsed,
            )

            width, height = open_png(png).size
            region_targets, collapsed_ids = transform_region_targets_with_collapsed(
                region_targets, served.matrix, width_px=width, height_px=height
            )
        png = served.png
    pixels = page_pixels(png, processor, policy)
    if not torch.equal(pixels.grid_thw, base.grid_thw.to(torch.int64)):
        raise OnTheFlyVisualError(
            f"{base.document_bundle_id} page {base.page_index}: processor grid "
            f"{pixels.grid_thw.tolist()} != the row's grid {base.grid_thw.tolist()}"
        )
    new_base = dataclasses.replace(
        base, pixel_values=pixels.pixel_values, visual_tokens=pixels.visual_tokens
    )
    out = dataclasses.replace(
        sample,
        base=new_base,
        region_targets=region_targets,
        collapsed_region_node_ids=collapsed_ids,
    )
    if served is not None and served.color_lost:
        out = dataclasses.replace(out, color_unseen=True)
    return out


class OnTheFlyDrawDataset(Dataset):
    """The S4 dataset addressed by plan DRAW index (`plan_order` position).

    `run_s4_training` fetches `dataset[i]` for `i` in the plan order; this
    dataset is fetched with the identity order instead, so `__getitem__`
    receives the draw index and can derive the step (`draw // batch_size +
    1`) the augmentation view is keyed on. Returns `OnTheFlyDraw`.
    Everything here is CPU (store read, augmentation, processor) and runs
    in the prefetch workers.
    """

    def __init__(
        self,
        base: Any,
        *,
        plan_order: Sequence[int],
        batch_size: int,
        store_dir: Path,
        processor: Any,
        policy: VisualInputPolicy,
        augmentation: OnTheFlyAugmentation | None = None,
        external: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if augmentation is not None and augmentation.readable_retry and batch_size != 1:
            raise OnTheFlyVisualError(
                "readable-retry requires batch_size=1: one page per global plan draw"
            )
        self.base = base
        self.plan_order = list(plan_order)
        self.batch_size = int(batch_size)
        self.store_dir = Path(store_dir)
        self.processor = processor
        self.policy = policy
        self.augmentation = augmentation
        self.external = external

    @property
    def reader(self) -> Any:
        """The exact-text reader, for `s2_train._clear_reader_lru`."""
        return self.base.base.reader

    def __len__(self) -> int:
        return len(self.plan_order)

    def __getitem__(self, draw_index: int) -> OnTheFlyDraw:
        sample = self.base[self.plan_order[draw_index]]
        bundle, page = sample.base.document_bundle_id, sample.base.page_index
        step = draw_index // self.batch_size + 1
        if self.external is not None and sample.base.page_image is not None:
            # `draw_index` is the PLAN draw index (the trainer fetches with the
            # identity order, and a resume starts at the global step), so the
            # external pick is a pure function of the plan draw, not of worker
            # count or of where an attempt started. Decided BEFORE augmentation.
            entry = self.external.select(draw_index, bundle, page)
            if entry is not None:
                return self._external_draw(sample, entry)
        png = served_view = None
        if self.augmentation is not None and sample.base.page_image is not None:
            png = read_page_png(self.store_dir, sample.base.page_image)
            served_view = serve_view(png, self.augmentation, bundle, page, step)
        served = attach_pixels(
            sample,
            store_dir=self.store_dir,
            processor=self.processor,
            policy=self.policy,
            served=served_view,
            png=png,
        )
        view = None if served_view is None else served_view.view
        view_id = f"{bundle}:{page}" if view is None else (
            f"{bundle}:{page}:onfly:{view.family}:{view.transform_chain_hash}"
        )
        return OnTheFlyDraw(
            sample=served,
            view_id=view_id,
            augmented=view is not None,
            aug_report=None if served_view is None else served_view.report,
        )

    def _external_draw(self, sample: Any, entry: Any) -> OnTheFlyDraw:
        png = read_page_png(self.store_dir, sample.base.page_image)
        external_png = self.external.read(entry)
        clean_size, served_size = open_png(png).size, open_png(external_png).size
        if clean_size != served_size:
            raise OnTheFlyVisualError(
                f"{entry.view_id}: the external image is {served_size[0]}x{served_size[1]}, "
                f"the clean page {clean_size[0]}x{clean_size[1]} -- an external view must "
                "keep the page size"
            )
        matrix = None if entry.matrix_ref_to_view is None else tuple(
            tuple(row) for row in entry.matrix_ref_to_view
        )
        served = ServedView(
            external_png,
            matrix,
            None,
            None,
            collapse_regions=not entry.alignment_usable,
            color_lost=entry.color_lost,
        )
        out = attach_pixels(
            sample,
            store_dir=self.store_dir,
            processor=self.processor,
            policy=self.policy,
            served=served,
            png=png,
        )
        return OnTheFlyDraw(
            sample=out,
            view_id=entry.view_id,
            augmented=False,
            aug_report=None,
            external_kind=entry.kind,
        )


# ---------------------------------------------------------------------------
# GPU side: the frozen ViT inside the step
# ---------------------------------------------------------------------------


class OnTheFlyVisualEncoder:
    """Runs the frozen tower on a collated page-image batch (`S1TrainingBatch.
    page_pixels`) under `torch.no_grad()` and returns the batch with real
    `visual_features` (float32, the dtype the cached path collates to) and,
    for a DeepStack policy, `deepstack_visual_embeds` -- level rows in the
    row-major (page, token) order the placeholder scatter consumes. No
    parameter of the tower is touched; nothing it computes carries grad.

    `batch_pages` (default off): the rows still to compute go through ONE
    tower call (pixel rows concatenated, grids stacked; the tower attends
    per image), split back by each page's token count. `feature_cache`
    (default off): see `ViTFeatureCache`; `materialize(cache_refs=...)` names
    the rows it may serve or write."""

    def __init__(
        self,
        candidate: Any,
        *,
        policy: VisualInputPolicy,
        device: torch.device | str = "cpu",
        feature_dtype: torch.dtype = torch.float32,
        batch_pages: bool = False,
        feature_cache: ViTFeatureCache | None = None,
    ) -> None:
        if policy.deepstack:
            if candidate.vision_forward_with_deepstack is None:
                raise OnTheFlyVisualError("the policy asks for DeepStack; the candidate has none")
            if tuple(candidate.deepstack_visual_indexes) != tuple(
                policy.deepstack_visual_indexes
            ):
                raise OnTheFlyVisualError(
                    f"policy DeepStack taps {policy.deepstack_visual_indexes} != the "
                    f"candidate's {candidate.deepstack_visual_indexes}"
                )
        if int(candidate.spatial_merge_size) != int(policy.merge_size):
            raise OnTheFlyVisualError(
                f"policy merge_size {policy.merge_size} != the tower's "
                f"{candidate.spatial_merge_size}"
            )
        if feature_cache is not None and feature_cache.policy != policy:
            raise OnTheFlyVisualError(
                "the feature cache is bound to another visual input policy than the encoder"
            )
        self.candidate = candidate
        self.policy = policy
        self.device = torch.device(device)
        self.feature_dtype = feature_dtype
        self.batch_pages = bool(batch_pages)
        self.feature_cache = feature_cache

    @torch.no_grad()
    def encode(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        pixel_values = pixel_values.to(self.device, self.policy.torch_dtype)
        grid = grid_thw.reshape(-1, 3).to(self.device)
        if self.policy.deepstack:
            pooled, levels = self.candidate.vision_forward_with_deepstack(
                pixel_values, grid_thw=grid
            )
            return pooled, tuple(levels)
        return self.candidate.vision_forward(pixel_values, grid_thw=grid), ()

    def encode_rows(
        self, rows: Sequence[tuple[torch.Tensor, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, tuple[torch.Tensor, ...]]]:
        """Per-row `(pooled, levels)`: one tower call per row, or one call
        for all rows under `batch_pages`."""
        if not self.batch_pages or len(rows) < 2:
            return [self.encode(pixel_values, grid) for pixel_values, grid in rows]
        pooled, levels = self.encode(
            torch.cat([pixel_values for pixel_values, _grid in rows], dim=0),
            torch.stack([grid.reshape(3) for _pixel_values, grid in rows]),
        )
        counts = [
            self.policy.visual_tokens_for(grid.reshape(3).tolist()) for _pv, grid in rows
        ]
        if sum(counts) != int(pooled.shape[0]):
            raise OnTheFlyVisualError(
                f"the batched tower produced {pooled.shape[0]} visual tokens, the grids "
                f"name {sum(counts)}"
            )
        pooled_rows = torch.split(pooled, counts, dim=0)
        level_rows = [torch.split(level, counts, dim=0) for level in levels]
        return [
            (pooled_rows[i], tuple(level[i] for level in level_rows)) for i in range(len(rows))
        ]

    @torch.no_grad()
    def materialize_deferred(
        self, batch: Any, *, cache_refs: Sequence[str | None] | None = None
    ) -> tuple[Any, list[tuple[str, Any]]]:
        """`materialize` without the cache file writes: returns the batch and
        the pending `(page_sha256, quantized)` writes, so the prefetch path
        can write them after the tower's stream finished instead of blocking
        on a device-to-host copy before the current step runs."""
        if batch.page_pixels is None:
            raise OnTheFlyVisualError("materialize() got a batch whose features are not pending")
        if cache_refs is not None and self.feature_cache is None:
            raise OnTheFlyVisualError("cache_refs given but the encoder has no feature cache")
        if cache_refs is not None and len(cache_refs) != len(batch.page_pixels):
            raise OnTheFlyVisualError(
                f"{len(cache_refs)} cache refs for {len(batch.page_pixels)} batch rows"
            )
        model_batch = batch.model_batch
        lengths = model_batch.visual_valid_lengths.tolist()
        refs = list(cache_refs) if cache_refs is not None else [None] * len(lengths)
        results: list[tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None] = [None] * len(refs)
        if self.feature_cache is not None:
            for index, ref in enumerate(refs):
                if ref is not None:
                    results[index] = self.feature_cache.get(
                        ref,
                        tokens=int(lengths[index]),
                        device=self.device,
                        dtype=self.policy.torch_dtype,
                    )
        todo = [i for i, got in enumerate(results) if got is None]
        writes: list[tuple[str, Any]] = []
        encoded = self.encode_rows([batch.page_pixels[i] for i in todo])
        for index, (pooled, levels) in zip(todo, encoded, strict=True):
            ref = refs[index]
            if ref is not None and self.feature_cache is not None:
                quantized = self.feature_cache.quantize(pooled, levels)
                writes.append((ref, quantized))
                pooled, levels = self.feature_cache.dequantize(quantized, dtype=pooled.dtype)
            results[index] = (pooled, levels)
        rows: list[torch.Tensor] = []
        level_rows: list[list[torch.Tensor]] = []
        for index, result in enumerate(results):
            if result is None:  # pragma: no cover - every row was either hit or encoded
                raise OnTheFlyVisualError(f"row {index}: no visual features produced")
            pooled, levels = result
            if int(pooled.shape[0]) != int(lengths[index]):
                raise OnTheFlyVisualError(
                    f"row {index}: the tower produced {pooled.shape[0]} visual tokens, "
                    f"the placeholder run has {lengths[index]}"
                )
            rows.append(pooled)
            if self.policy.deepstack:
                if len(levels) != len(self.policy.deepstack_visual_indexes):
                    raise OnTheFlyVisualError(
                        f"row {index}: {len(levels)} DeepStack levels, policy taps "
                        f"{len(self.policy.deepstack_visual_indexes)}"
                    )
                for level_index, level in enumerate(levels):
                    if len(level_rows) <= level_index:
                        level_rows.append([])
                    level_rows[level_index].append(level)
        width = int(rows[0].shape[-1])
        features = torch.zeros(
            len(rows),
            int(batch.visual_steps),
            width,
            dtype=self.feature_dtype,
            device=model_batch.composite_input_ids.device,
        )
        for index, pooled in enumerate(rows):
            features[index, : pooled.shape[0]] = pooled.to(self.feature_dtype)
        deepstack = (
            tuple(torch.cat(level, dim=0) for level in level_rows)
            if self.policy.deepstack
            else None
        )
        new_model_batch = dataclasses.replace(
            model_batch, visual_features=features, deepstack_visual_embeds=deepstack
        )
        done = dataclasses.replace(batch, model_batch=new_model_batch, page_pixels=None)
        return done, writes

    def write_cache(self, writes: Sequence[tuple[str, Any]]) -> None:
        """Flush `materialize_deferred`'s pending writes to the cache."""
        if writes and self.feature_cache is None:
            raise OnTheFlyVisualError("cache writes given but the encoder has no feature cache")
        for ref, (values, scales) in writes:
            self.feature_cache.put(ref, values, scales)  # type: ignore[union-attr]

    def materialize(self, batch: Any, *, cache_refs: Sequence[str | None] | None = None) -> Any:
        """`batch` (an `S1TrainingBatch` with `page_pixels`) with its visual
        features filled in; a feature batch is refused (nothing to do is a
        caller bug, not a no-op). `cache_refs` (feature cache only): per row
        the page image's sha256, or `None` for a row that must be computed
        fresh and never cached (an augmented view)."""
        done, writes = self.materialize_deferred(batch, cache_refs=cache_refs)
        self.write_cache(writes)
        return done

    def cache_refs(self, draws: Sequence[OnTheFlyDraw]) -> list[str | None] | None:
        """`draw_cache_refs(draws)` when this encoder has a feature cache."""
        return None if self.feature_cache is None else draw_cache_refs(draws)


def draw_cache_refs(draws: Sequence[OnTheFlyDraw]) -> list[str | None]:
    """Per draw: its page image's sha256 for a clean view, `None` for an
    augmented or external one (its pixels are not the stored PNG's)."""
    return [
        None
        if draw.augmented or draw.external_kind is not None
        else str(draw.sample.base.page_image["sha256"])
        for draw in draws
    ]


# ---------------------------------------------------------------------------
# Epoch-1 write-through fp8 feature cache (fallback, default off)
# ---------------------------------------------------------------------------

#: `hashing.domain_hash` domain of a cache entry's key.
VIT_FEATURE_CACHE_KEY_DOMAIN = "unflatten.vit-feature-cache-key/v1"
#: Storage format of one entry: fp8 e4m3 values, one float32 scale per row.
VIT_FEATURE_CACHE_FORMAT = "fp8_e4m3fn_rowscale_f32/v1"
_FP8 = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8).max)  # 448


def quantize_fp8_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8 e4m3 with one float32 scale per row (`absmax / 448`; an all-zero
    row gets scale 1). Rows are visual tokens, columns channels."""
    x = values.to(torch.float32)
    absmax = x.abs().amax(dim=-1)
    scale = torch.where(absmax > 0, absmax / _FP8_MAX, torch.ones_like(absmax))
    q = (x / scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    return q, scale


def dequantize_fp8_rows(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (q.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)).to(dtype)


def fp8_cache_bytes_per_page(*, tokens: int, width: int, levels: int) -> int:
    """On-disk payload of one page: `levels` tensors (pooler + DeepStack) of
    `tokens x width` fp8 bytes, plus a float32 scale per row per tensor."""
    return int(tokens) * int(width) * int(levels) + int(tokens) * 4 * int(levels)


def vit_feature_cache_key(
    *, page_image_sha256: str, backbone_artifact_hash: str, visual_input_policy_hash: str
) -> str:
    """The entry key: the page image, the ViT revision and the whole
    preprocessing/tap policy (which already binds the revision -- named
    twice so a key can be audited without the policy file), and the format."""
    return domain_hash(
        VIT_FEATURE_CACHE_KEY_DOMAIN,
        {
            "page_image_sha256": page_image_sha256,
            "backbone_artifact_hash": backbone_artifact_hash,
            "visual_input_policy_hash": visual_input_policy_hash,
            "format": VIT_FEATURE_CACHE_FORMAT,
        },
    )


Quantized = tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]


class ViTFeatureCache:
    """One file per page under `root/<2 hex>/<64 hex>.pt`, written atomically
    (temp file + rename) by the training process only. A file whose recorded
    key, token count or level count disagrees with the request is refused,
    never silently recomputed: it is foreign or corrupt, and recomputing
    would hide that."""

    def __init__(self, root: Path | str, *, policy: VisualInputPolicy) -> None:
        self.root = Path(root)
        self.policy = policy
        self.levels = 1 + len(policy.deepstack_visual_indexes)
        self.stats = {"hits": 0, "misses": 0, "writes": 0}

    def key(self, page_image_sha256: str) -> str:
        return vit_feature_cache_key(
            page_image_sha256=page_image_sha256,
            backbone_artifact_hash=self.policy.backbone_artifact_hash,
            visual_input_policy_hash=self.policy.policy_hash(),
        )

    def path_for(self, key: str) -> Path:
        digest = key.split(":", 1)[1]
        return self.root / digest[:2] / f"{digest}.pt"

    def quantize(self, pooled: torch.Tensor, levels: Sequence[torch.Tensor]) -> Quantized:
        tensors = (pooled, *levels)
        if len(tensors) != self.levels:
            raise OnTheFlyVisualError(
                f"{len(tensors)} tensors (pooler + levels) for a policy with {self.levels} levels"
            )
        pairs = [quantize_fp8_rows(t) for t in tensors]
        return tuple(q for q, _s in pairs), tuple(s for _q, s in pairs)

    @staticmethod
    def dequantize(
        quantized: Quantized, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        values, scales = quantized
        out = [dequantize_fp8_rows(q, s, dtype) for q, s in zip(values, scales, strict=True)]
        return out[0], tuple(out[1:])

    def round_trip(
        self, pooled: torch.Tensor, levels: Sequence[torch.Tensor], *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        return self.dequantize(self.quantize(pooled, levels), dtype=dtype)

    def put(
        self,
        page_image_sha256: str,
        values: Sequence[torch.Tensor],
        scales: Sequence[torch.Tensor],
    ) -> None:
        key = self.key(page_image_sha256)
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "key": key,
                "format": VIT_FEATURE_CACHE_FORMAT,
                "page_image_sha256": page_image_sha256,
                "backbone_artifact_hash": self.policy.backbone_artifact_hash,
                "visual_input_policy_hash": self.policy.policy_hash(),
                "tokens": int(values[0].shape[0]),
                "levels": len(values),
            },
            "values": [q.detach().to("cpu").contiguous().view(torch.uint8) for q in values],
            "scales": [s.detach().to("cpu", torch.float32).contiguous() for s in scales],
        }
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        self.stats["writes"] += 1

    def get(
        self,
        page_image_sha256: str,
        *,
        tokens: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
        key = self.key(page_image_sha256)
        path = self.path_for(key)
        if not path.exists():
            self.stats["misses"] += 1
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        meta = payload["meta"]
        if meta["key"] != key or meta["format"] != VIT_FEATURE_CACHE_FORMAT:
            raise OnTheFlyVisualError(
                f"{path}: the entry was written for another key ({meta['key']!r}), "
                f"not {key!r} -- a foreign or corrupt cache file"
            )
        if int(meta["tokens"]) != int(tokens):
            raise OnTheFlyVisualError(
                f"{path}: {meta['tokens']} cached tokens, the row has {tokens}"
            )
        if int(meta["levels"]) != self.levels:
            raise OnTheFlyVisualError(
                f"{path}: {meta['levels']} cached levels, the policy has {self.levels}"
            )
        values = tuple(v.view(_FP8).to(device) for v in payload["values"])
        scales = tuple(s.to(device) for s in payload["scales"])
        self.stats["hits"] += 1
        return self.dequantize((values, scales), dtype=dtype)


# ---------------------------------------------------------------------------
# Prefetch overlap: the next step's ViT while the current step trains
# ---------------------------------------------------------------------------


def _record_stream(value: Any, stream: Any) -> None:
    """`Tensor.record_stream` over every CUDA tensor of a (nested) batch, so
    the caching allocator does not hand memory made on the side stream back
    out while the main stream still reads it."""
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _record_stream(getattr(value, field.name), stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record_stream(item, stream)


def require_frozen_tower(candidate: Any) -> None:
    """Overlap computes step k+1's features before step k's optimizer step;
    that is the same computation only if no tower parameter can change."""
    named = getattr(candidate, "named_parameters", None) or {}
    trainable = [
        name for name, p in named.items() if name.startswith("visual.") and p.requires_grad
    ]
    if trainable:
        raise OnTheFlyVisualError(
            f"prefetch overlap needs a frozen tower; trainable: {trainable[:3]}"
        )


def prefetch_materialized(
    batches: Iterable[Sequence[OnTheFlyDraw]],
    *,
    encoder: OnTheFlyVisualEncoder,
    collate: Callable[[Sequence[OnTheFlyDraw]], Any],
) -> Iterator[tuple[Sequence[OnTheFlyDraw], Any]]:
    """Yield `(draws, materialized batch)` for every batch of `batches`,
    issuing batch k+1's collate + ViT BEFORE handing out batch k.

    On a CUDA encoder the issue runs on a side stream; handing out waits for
    that stream (`wait_stream`), marks every tensor for the consuming stream
    (`record_stream`) and then writes that batch's cache entries (their
    device-to-host copy no longer stalls the next step). On CPU the same
    order runs synchronously -- the CPU bit-identity test proves the
    reordering, the CUDA test the stream handling. The augmentation view is
    chosen in the data worker from `(seed, step, page)`, never from call
    order, and the frozen tower in eval mode draws no random numbers."""
    require_frozen_tower(encoder.candidate)
    side = torch.cuda.Stream(device=encoder.device) if encoder.device.type == "cuda" else None

    def issue(draws: Sequence[OnTheFlyDraw]) -> tuple[Sequence[OnTheFlyDraw], Any, list[Any]]:
        context = torch.cuda.stream(side) if side is not None else contextlib.nullcontext()
        with context:
            batch, writes = encoder.materialize_deferred(
                collate(draws), cache_refs=encoder.cache_refs(draws)
            )
        return draws, batch, writes

    def take(item: tuple[Sequence[OnTheFlyDraw], Any, list[Any]]) -> tuple[Any, Any]:
        draws, batch, writes = item
        if side is not None:
            current = torch.cuda.current_stream(encoder.device)
            current.wait_stream(side)
            _record_stream(batch, current)
            for _ref, (values, scales) in writes:
                _record_stream([*values, *scales], current)
        encoder.write_cache(writes)
        return draws, batch

    iterator = iter(batches)
    try:
        pending = issue(next(iterator))
    except StopIteration:
        return
    for draws in iterator:
        upcoming = issue(draws)
        yield take(pending)
        pending = upcoming
    yield take(pending)

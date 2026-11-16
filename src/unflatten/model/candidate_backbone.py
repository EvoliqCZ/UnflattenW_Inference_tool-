"""Pixels-to-visual-features for the M2/M3 Qwen dense candidate.

The gap this module closes: `candidate_model.QwenDenseCandidateModel`
starts from an already-computed `(batch, tokens, visual_dim)` tensor
(`candidate_model.py`'s own docstring, assumption 1 — "pixel-to-visual-
feature batching is out of scope here"), and nothing in this repository
produced that tensor FOR THE CANDIDATE before this module. Without it the
S1 canary has no input.

**Why this is not `probe.features.FrozenVisionBackbone`, reused as-is.**
The probe loads only the vision tower (`probe/backbones.py`,
`_drop_language_half` is deliberate there) through a SEPARATE loading path
from `candidate_loader.load_qwen_candidate`'s full-candidate load. DEC-007
D3 pins an exact checkpoint revision, dtype and attention implementation as
Component-Lock material; a probe-extracted feature is not provably the same
tensor `candidate.vision_forward` would produce for the same pixels unless
it goes through the SAME loaded weights and the SAME code path. So this
module wraps `candidate_loader.QwenCandidate` directly, not the probe.

**Why `build_cache.py`/`feature_cache.py` are extended, not duplicated.**
Their sharding/manifest/transactional-write machinery
(`FeatureCacheWriter`/`FeatureCacheReader`) is backbone-agnostic — it does
not care whose features it is storing, only that they are finite float16
tensors bound to a `CacheIdentity`. `build_cache.build()` already calls
`backbone.extract(page.image()) -> BackboneFeatures` and reads
`backbone.model_id`/`.processor`/`.tap` for the manifest — exactly the
surface `CandidateVisionBackbone` below provides — so `build()` runs
unmodified against this backbone (see `feature_cache.py`'s `CacheEntry.
grid_thw` for the one field that DID need adding: the candidate track
needs `image_grid_thw`, which the probe/decoder-training track never did).
Duplicating ~250 lines of tested transactional-cache machinery for one
extra field would have been the wrong trade under CLAUDE.md's simplicity
principle.

**Why no `.dense(at_merged=...)` call.** The probe's fine-grained patch
grid needs `BackboneFeatures.dense(at_merged=True)` to reach the coarse,
post-spatial-merge token count `inject_visual`'s placeholder scatter
expects. `QwenCandidate.vision_forward` (the real backend) already returns
POST-MERGE tokens directly — Qwen3-VL's own vision tower performs the
`spatial_merge_size=2` merge internally before returning `image_embeds`
(`candidate_loader.py`'s own docstring; measured: a 4x4 patch grid yields 4
placeholder tokens, not 16) — so `visual_features` here IS already at the
placeholder-token grain. Calling `.dense(...)` on it would double-merge.

This module authorizes no gradient update: `extract` runs under
`torch.inference_mode()` and nothing here builds an optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from PIL import Image
from torch import nn

from unflatten.model.candidate_loader import (
    QWEN_CANDIDATE_MODEL_ID,
    QWEN_CANDIDATE_REVISION,
    QwenCandidate,
    build_stub_qwen_candidate,
    group_id_of_qwen_parameter,
    load_qwen_candidate,
)
from unflatten.probe.features import BackboneFeatures


class _ImageProcessorLike(Protocol):
    """Minimal duck-typed surface this module needs from an HF image
    processor: turn a PIL image into `pixel_values` + `image_grid_thw`."""

    def __call__(self, *, images: Image.Image, return_tensors: str) -> Any: ...


@dataclass
class CandidateVisionBackbone:
    """`FrozenVisionBackbone`-shaped adapter over `QwenCandidate`.

    Exposes exactly what `build_cache.build()`/`build_cache.cache_identity()`
    read off a backbone (`model_id`, `processor`, `tap`, `extract`), so the
    existing cache build path runs against it unmodified.
    """

    candidate: QwenCandidate
    processor: _ImageProcessorLike | None
    device: str = "cpu"
    dtype: torch.dtype = torch.bfloat16
    #: Candidate track has no alternate feature tap (unlike Granite Vision,
    #: the probe's other backend) — always `None`, read by `cache_identity`.
    tap: str | None = None

    @property
    def model_id(self) -> str:
        return self.candidate.model_id

    @torch.inference_mode()
    def extract(self, image: Image.Image) -> BackboneFeatures:
        """One page's candidate-native visual features, already at the
        post-merge placeholder-token grain (module docstring)."""
        if self.processor is None:
            raise ValueError(
                "CandidateVisionBackbone has no processor configured; use "
                "load_candidate_vision_backbone for a real extraction path"
            )
        batch = self.processor(images=image, return_tensors="pt")
        pixel_values = batch["pixel_values"].to(self.device, self.dtype)
        grid = batch["image_grid_thw"][0].to(self.device)
        image_embeds = self.candidate.vision_forward(pixel_values, grid_thw=grid.unsqueeze(0))
        return BackboneFeatures(
            visual_features=image_embeds,
            multiscale_features=(),
            grid_thw=(int(grid[0]), int(grid[1]), int(grid[2])),
        )

    @torch.inference_mode()
    def extract_with_deepstack(
        self, image: Image.Image
    ) -> tuple[BackboneFeatures, tuple[torch.Tensor, ...]]:
        """DEC-056: `extract` plus the DeepStack levels from the SAME tower
        call (`QwenCandidate.vision_forward_with_deepstack`). Only a
        DeepStack runtime calls this; `extract` is unchanged."""
        if self.processor is None:
            raise ValueError("CandidateVisionBackbone has no processor configured")
        if self.candidate.vision_forward_with_deepstack is None:
            raise ValueError("this candidate has no DeepStack path")
        batch = self.processor(images=image, return_tensors="pt")
        pixel_values = batch["pixel_values"].to(self.device, self.dtype)
        grid = batch["image_grid_thw"][0].to(self.device)
        pooled, levels = self.candidate.vision_forward_with_deepstack(
            pixel_values, grid_thw=grid.unsqueeze(0)
        )
        features = BackboneFeatures(
            visual_features=pooled,
            multiscale_features=(),
            grid_thw=(int(grid[0]), int(grid[1]), int(grid[2])),
        )
        return features, tuple(levels)


def offload_language_parameters(
    named_parameters: dict[str, nn.Parameter], *, device: str | torch.device
) -> int:
    """Move every `backbone.language` parameter (by `group_id_of_qwen_parameter`)
    to `device`, in place, and return how many moved. Vision tower and
    projector parameters are never touched: they are what produces the
    features. Idempotent -- a parameter already on `device` is not counted.
    """
    target = torch.device(device)
    moved = 0
    for name, param in named_parameters.items():
        if group_id_of_qwen_parameter(name) != "backbone.language":
            continue
        if param.data.device == target:
            continue
        param.data = param.data.to(target)
        moved += 1
    return moved


def load_candidate_vision_backbone(
    *,
    model_id: str = QWEN_CANDIDATE_MODEL_ID,
    revision: str = QWEN_CANDIDATE_REVISION,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> CandidateVisionBackbone:
    """The real, full-candidate-backed extraction path. Heavy (network +
    ~8 GB VRAM, DEC-007 D3's exact pin); callers exercising this in tests
    must mark the test `slow`.
    """
    from transformers import AutoImageProcessor

    candidate = load_qwen_candidate(
        model_id=model_id, revision=revision, dtype=dtype, device=device
    )
    # Feature extraction runs only the vision tower, but DEC-007 D3 wants the
    # features to come from the SAME loaded weights and code path as
    # `candidate.vision_forward` -- so the full candidate is loaded, then the
    # idle language half is moved off the accelerator. Where an unused tensor
    # lives does not change which weights produce the features; keeping ~7 GB
    # of language model resident did change whether an 11.6 GiB card could
    # hold a 300 DPI page at all (measured 2026-09-06: hard OOM on the first
    # page larger than A4).
    offload_language_parameters(candidate.named_parameters, device="cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    return CandidateVisionBackbone(
        candidate=candidate, processor=processor, device=device, dtype=dtype
    )


class _StubImageProcessor:
    """Deterministic, offline stand-in: downsamples to a fixed small grid
    and broadcasts each cell's grayscale intensity across the stub
    candidate's `visual_hidden_size` — enough to exercise this module's
    plumbing without a real Qwen processor or network access.
    """

    def __init__(self, *, visual_hidden_size: int, grid: tuple[int, int] = (2, 2)) -> None:
        self.visual_hidden_size = visual_hidden_size
        self.grid = grid

    def __call__(self, *, images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
        import numpy as np

        del return_tensors
        rows, cols = self.grid
        small = images.convert("L").resize((cols, rows))
        values = torch.from_numpy(np.array(small, dtype=np.float32).reshape(-1)) / 255.0
        pixel_values = values.unsqueeze(-1).expand(rows * cols, self.visual_hidden_size).clone()
        grid_thw = torch.tensor([[1, rows, cols]], dtype=torch.long)
        return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}


def build_stub_candidate_vision_backbone(
    *,
    visual_hidden_size: int = 6,
    hidden_size: int = 8,
    vocab_size: int = 12,
    seed: int = 0,
) -> CandidateVisionBackbone:
    """Fast, offline stand-in with the real path's exact interface. Default
    path for every non-`slow` test."""
    candidate = build_stub_qwen_candidate(
        visual_hidden_size=visual_hidden_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        seed=seed,
    )
    processor = _StubImageProcessor(visual_hidden_size=visual_hidden_size)
    return CandidateVisionBackbone(
        candidate=candidate, processor=processor, device="cpu", dtype=torch.float32
    )

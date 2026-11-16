"""Torch `Dataset` over a release.

Deliberately thin. Everything that decides *what* a sample is lives in
`samples.py` and `items.py`, which are framework-free and tested without
an ML stack; this module only decodes the PNG and hands over tensors.

Requires the optional `ml` dependency group (`uv sync --group ml`).

Labels are returned as plain dicts, not collated tensors: the target
shape is task-specific and belongs to the model code, not here. That
also means the default `torch.utils.data` collate function will not
handle a batch of these — pass a task-specific `collate_fn`, or use
`batch_size=None` for the Stage-0 probe.

Thin does NOT mean permissive. This is a documented entry point
(`docs/RUNBOOK.md`), and an entry point that quietly withdraws less
supervision than `model/build_cache.py` is how the strict path gets
bypassed. So the same two withdrawals are applied here, in the same
order, and admission defaults to the same strictness.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from unflatten.dataset.gates.store import read_object
from unflatten.dataset.items import ItemBuildError, TrainingItem, build_item
from unflatten.dataset.samples import iter_page_samples
from unflatten.dataset.targets.masks import apply_masks


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """`PIL.Image` -> float32 CHW tensor in [0, 1]."""
    array = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    return (
        array.view(image.size[1], image.size[0], 3).permute(2, 0, 1).float().div_(255.0)
    )


@lru_cache(maxsize=64)
def _sidecar_font_pack_hash(store_dir: str, ref_json: str) -> str | None:
    """The font pack the pixels of this page were rendered with.

    Read from the Observation Sidecar's `renderer` block and cached per
    document — a sidecar is large and a document averages several pages.
    Deliberately a copy of `model.build_cache.render_font_pack_hash`:
    the dataset layer must not import the model layer, and the model
    layer's version carries its own cache-summary concerns.
    """
    sidecar = json.loads(read_object(Path(store_dir), json.loads(ref_json)))
    return (sidecar.get("renderer") or {}).get("font_pack_hash")


def _withdrawn_targets(store_dir: Path, item: TrainingItem) -> dict[str, Any] | None:
    """The page's labels with every withdrawal this page is subject to.

    `targets.masks.apply_masks` owns the load-bearing order (the font
    projection re-derives `supervision_mask` from `known AND observable`,
    so the placement withdrawal must run second) and applies both for one
    deep copy and one `target_bundle_hash`. Same call as
    `model.build_cache.withdraw_placements`, which additionally reports
    what it flipped for its cache summary.

    A page with no fragment cannot state which of its nodes were
    actually observed, so it cannot be handed out: unwithdrawn labels
    for an imputed placement are exactly the fabricated positives this
    guards against.
    """
    if item.targets is None:
        return None
    if item.fragment is None:
        raise ItemBuildError(
            f"{item.sample.document_bundle_id} page {item.sample.page_index}: "
            "no fragment, so no page_evidence — supervision cannot be "
            "withdrawn and the labels must not be used"
        )
    # the projection is only meaningful against the render it was derived
    # for; naming that render is required, not optional
    pack = None
    if item.mask_projection is not None and item.sample.observation_sidecar is not None:
        pack = _sidecar_font_pack_hash(
            str(store_dir),
            json.dumps(item.sample.observation_sidecar, sort_keys=True),
        )
    masked, _flipped = apply_masks(
        item.targets, item.mask_projection, [item.fragment], render_font_pack_hash=pack
    )
    return masked


class PageDataset(Dataset):
    """One rendered page per item, selected by split and task view.

    The sample list is materialised in `__init__` so `__len__` is exact
    and workers can index it; only the object bytes are read lazily.
    """

    def __init__(
        self,
        release_dir: Path | str,
        store_dir: Path | str,
        split: str,
        task_kind: str,
        *,
        eligibilities: Sequence[str] = ("train_ready",),
        transform: Callable[[Image.Image], Image.Image] | None = None,
        require_privacy_review: bool = True,
    ) -> None:
        """`require_privacy_review` defaults to True, matching
        `model.build_cache.build()`: this dataset feeds gradients, and
        RFC005 §9.2 blocks an incomplete privacy decision from training.
        Pass False to enumerate the dev/research inventory (§9.3)."""
        self.store_dir = Path(store_dir)
        self.task_kind = task_kind
        self.transform = transform
        self.samples = list(
            iter_page_samples(
                release_dir,
                store_dir,
                split,
                task_kind,
                eligibilities=eligibilities,
                require_privacy_review=require_privacy_review,
            )
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        item = build_item(self.store_dir, sample)
        image = Image.open(io.BytesIO(item.image_bytes))
        image.load()
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": _to_tensor(image),
            "targets": _withdrawn_targets(self.store_dir, item),
            "fragment": item.fragment,
            "mask_projection": item.mask_projection,
            "document_bundle_id": sample.document_bundle_id,
            "lineage_root_id": sample.lineage_root_id,
            "page_index": sample.page_index,
            "page_count": sample.page_count,
        }

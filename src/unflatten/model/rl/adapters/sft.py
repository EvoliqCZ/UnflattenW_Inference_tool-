"""``build_sft_batch`` port: the S4 batch recipe for ONE page (design sec. 3 ``sft.py``,
inv. 4; real-paths D7; WP6).

The auxiliary term of the S4.5 update (``rl.auxiliary``) needs an S4 batch over the
GOLD token stream, separate from the rollout batch. This builds it for one page with
the same calls ``s4_train`` makes, in the same order::

    S4TrainingDataset[i]            (key -> index from the pinned expectation)
    -> onfly_visual.attach_pixels   (PNG -> processor, view=None, never augmented)
    -> s1_train.collate_training_batch([base], ranges, image_token_id, device)
    -> visual_encoder.materialize   (frozen ViT, float32)
    -> s4_typed_loss.collate_typed  (allowed_head_ids, mask_inactive_union_arms)
    -> s4_typed_loss.collate_regions(structural_table_region="exclude")

Guards, each fail-closed:

* strict admission of the reader, as on the rollout path (``admission``);
* the page must be in the pinned expectation, and the row served must be that page;
* the cache feature tap must be the one the policy derives;
* policy-trained head ids must be a subset of ``allowed_head_ids`` (a mistyped id
  would silently exclude nothing), and their keys are DROPPED from the typed batch
  here, so no sampled head is ever under both the PG and the CE family (E4.3);
  ``auxiliary_loss`` masks the same heads again from ``selected_typed_heads``;
* every tensor of the result is a detached CLONE, and with ``rollout_visual``
  injected the result is checked against that page's rollout ``PageVisual`` tensors
  by identity and storage (``assert_disjoint_from_visual``).

``allowed_head_ids`` is the stage policy's full ``L_typed`` set: ``collate_typed``
treats it as a REFUSAL list (a target on a head outside it raises), so the policy
heads are removed from the batch after collation, not from that set.

``torch`` and ``onfly_visual`` are imported lazily; the module is import-light.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from unflatten.model.rl.adapters.admission import require_strict_admission
from unflatten.model.rl.adapters.visual import CacheFeatureTapMismatch, PageKeyRefused

if TYPE_CHECKING:
    from unflatten.model.rl.auxiliary import SftAuxBatch
    from unflatten.model.rl.data import RlPage
    from unflatten.model.rl.real_run_ports import PageVisual


class SftBatchRefused(ValueError):
    """The SFT auxiliary batch of a page cannot be built within the D7 contract."""


def _clone_value(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple) and any(isinstance(v, torch.Tensor) for v in value):
        return tuple(_clone_value(v) for v in value)
    return value


def _clone_dataclass(obj: Any) -> Any:
    changes = {f.name: _clone_value(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    return dataclasses.replace(obj, **changes)


def _clone_batch(model_batch: Any, typed_batch: Any, region_batch: Any) -> tuple[Any, Any, Any]:
    from unflatten.model import s4_typed_loss as stl

    keys = {k: _clone_dataclass(v) for k, v in typed_batch.keys.items()}
    typed = stl.TypedBatch(keys=keys, excluded=dict(typed_batch.excluded))
    return _clone_dataclass(model_batch), typed, _clone_dataclass(region_batch)


def assert_disjoint_from_visual(sft: SftAuxBatch, visual: PageVisual) -> None:
    """Refuse an SFT batch that shares a tensor (identity or storage) with the page's
    rollout ``PageVisual``, DeepStack levels included. ``rl.auxiliary.assert_disjoint``
    covers ``visual_features`` / ``image_grid_thw`` (the allow-listed rollout keys);
    the DeepStack levels are not rollout-batch keys, so they are checked here."""
    from unflatten.model.rl.auxiliary import AuxiliaryRefused, assert_disjoint

    assert_disjoint(
        sft,
        {
            "visual_features": visual.visual_features,
            "image_grid_thw": visual.image_grid_thw,
        },
    )
    levels = tuple(visual.deepstack_visual_embeds or ())
    ptrs = {t.untyped_storage().data_ptr() for t in levels if t.numel()}
    ids = {id(t) for t in levels}
    for t in sft.tensors():
        if id(t) in ids or (t.numel() and t.untyped_storage().data_ptr() in ptrs):
            raise AuxiliaryRefused(
                "SFT batch shares a DeepStack level with the rollout PageVisual "
                "(identity or storage); clone it"
            )


def _require_policy_heads(
    allowed_head_ids: Collection[str], policy_head_ids: Collection[str]
) -> frozenset[str]:
    policy = frozenset(policy_head_ids)
    unknown = sorted(policy - frozenset(allowed_head_ids))
    if unknown:
        raise SftBatchRefused(
            f"policy-trained head ids {unknown} are not in allowed_head_ids: a mistyped or "
            "stale id would exclude nothing from the auxiliary term (D7, E4.3)"
        )
    return policy


def make_build_sft_batch(
    *,
    dataset: Any,
    store_dir: Path | str,
    processor: Any,
    policy_payload: Mapping[str, Any],
    visual_encoder: Any,
    ranges: Any,
    image_token_id: int,
    allowed_head_ids: Collection[str],
    policy_head_ids: Collection[str],
    mask_inactive_union_arms: bool = False,
    structural_table_region: str = "exclude",
    device: Any = "cpu",
    rollout_visual: Callable[[RlPage], PageVisual] | None = None,
) -> Callable[[RlPage], SftAuxBatch]:
    """``page -> SftAuxBatch`` over a 7.0.0 page-image cache.

    ``dataset`` is an ``S4TrainingDataset`` whose reader was opened
    ``gradient_bearing=True`` with a pinned expectation; ``dataset[i]`` addresses
    ``expectation.target_keys()[i]``. Construction-time refusals fire before any
    page is read."""
    from unflatten.model import onfly_visual as ov

    reader = dataset.base.reader
    require_strict_admission(reader)
    policy = ov.VisualInputPolicy.from_payload(policy_payload)
    tap = reader.identity.feature_tap
    if tap != policy.feature_tap():
        raise CacheFeatureTapMismatch(
            f"cache feature_tap {tap!r} != the policy's {policy.feature_tap()!r}: the "
            "SFT batch would see another visual input than the rollout"
        )
    policy_heads = _require_policy_heads(allowed_head_ids, policy_head_ids)
    keys = tuple(reader.expectation.target_keys())
    index_of = {key: i for i, key in enumerate(keys)}
    store = Path(store_dir)
    allowed = frozenset(allowed_head_ids)

    def build_sft_batch(page: RlPage) -> SftAuxBatch:
        key = (page.page_key[0], page.page_key[1])
        index = index_of.get(key)
        if index is None:
            raise PageKeyRefused(f"page {key!r} is not in the pinned expectation target keys")
        sample = dataset[index]
        if (sample.base.document_bundle_id, sample.base.page_index) != key:
            raise PageKeyRefused(
                f"cache index {index} is page "
                f"{(sample.base.document_bundle_id, sample.base.page_index)!r}, "
                f"the page asked for is {key!r}"
            )
        sft = _assemble(
            sample,
            store=store,
            processor=processor,
            policy=policy,
            visual_encoder=visual_encoder,
            ranges=ranges,
            image_token_id=image_token_id,
            allowed=allowed,
            policy_heads=policy_heads,
            mask_inactive_union_arms=mask_inactive_union_arms,
            structural_table_region=structural_table_region,
            device=device,
        )
        if rollout_visual is not None:
            assert_disjoint_from_visual(sft, rollout_visual(page))
        return sft

    return build_sft_batch


def _assemble(
    sample: Any,
    *,
    store: Path,
    processor: Any,
    policy: Any,
    visual_encoder: Any,
    ranges: Any,
    image_token_id: int,
    allowed: frozenset[str],
    policy_heads: frozenset[str],
    mask_inactive_union_arms: bool,
    structural_table_region: str,
    device: Any,
) -> SftAuxBatch:
    from unflatten.model import onfly_visual as ov
    from unflatten.model import s1_train
    from unflatten.model import s4_typed_loss as stl
    from unflatten.model.rl.auxiliary import SftAuxBatch

    sample = ov.attach_pixels(
        sample, store_dir=store, processor=processor, policy=policy, view=None
    )
    batch = s1_train.collate_training_batch(
        [sample.base], ranges=ranges, image_token_id=image_token_id, device=device
    )
    batch = visual_encoder.materialize(batch)
    typed = stl.collate_typed(
        [sample],
        allowed_head_ids=allowed,
        mask_inactive_union_arms=mask_inactive_union_arms,
    )
    typed = stl.TypedBatch(
        keys={k: v for k, v in typed.keys.items() if k[0] not in policy_heads},
        excluded=dict(typed.excluded),
    )
    regions = stl.collate_regions([sample], **stl.region_kwargs(structural_table_region))
    model_batch, typed, regions = _clone_batch(batch.model_batch, typed, regions)
    return SftAuxBatch(
        model_batch=model_batch,
        visual_steps=int(batch.visual_steps),
        typed_batch=typed,
        region_batch=regions,
    )

"""``build_visual`` port + per-page raster adapter (design DP4/DP6, inv. 4/5/11; WP4).

The v16 cache is a 7.0.0 page-image cache: rows carry ``page_image`` (a PNG ref) and
``grid_thw`` but no features, so a page's visual input is *computed*: PNG -> processor
-> frozen ViT (``OnTheFlyVisualEncoder``), float32 like S4 training. Nothing here
reads a gold field of the cache row (``packed_token_ids``, ``effective_loss_mask``,
``typed_targets``, ``region_targets``): only ``document_bundle_id``, ``page_index``,
``page_image`` and ``grid_thw``.

Six named refusals, each a :class:`VisualRefused` subclass:

1. :class:`CacheFeatureTapMismatch` - the cache identity's ``feature_tap`` is not the
   one the policy derives. ``compute_dtype`` is part of ``policy_hash`` and therefore of
   that tap, so a policy that differs only in dtype is refused here as well.
2. :class:`CheckpointVisualTapMismatch` - the checkpoint's ``visual_feature_tap``.
3. :class:`ModelDeepstackMismatch` - ``model.deepstack`` against ``policy.deepstack``.
4. :class:`RowGridMismatch` - the row's ``grid_thw`` against the grid of the PNG.
5. :class:`FeatureSourceMismatch` - ``ports.feature_source`` against the cache tap.
6. :class:`EncoderPolicyMismatch` - the encoder was built under another policy (dtype
   included) than the one the cache is bound to.

Heavy imports (``torch``, ``onfly_visual``) are lazy; this module is import-light.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Container, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unflatten.model.rl.data import RlPage
    from unflatten.model.rl.real_run_ports import PageVisual


class VisualRefused(ValueError):
    """The visual input of a page cannot be tied to the frozen policy."""


class CacheFeatureTapMismatch(VisualRefused):
    """The cache was not built under this visual input policy (refusal 1)."""


class CheckpointVisualTapMismatch(VisualRefused):
    """The checkpoint was trained on another visual feature tap (refusal 2)."""


class ModelDeepstackMismatch(VisualRefused):
    """``model.deepstack`` disagrees with the policy (refusal 3)."""


class RowGridMismatch(VisualRefused):
    """The cache row's grid is not the grid of its PNG under the policy (refusal 4)."""


class FeatureSourceMismatch(VisualRefused):
    """``ports.feature_source`` is not the cache's feature tap (refusal 5)."""


class EncoderPolicyMismatch(VisualRefused):
    """The encoder computes under another policy than the cache's (refusal 6)."""


class PageKeyRefused(VisualRefused):
    """A page the pinned expectation does not target, or a row that is not the page."""


class VisualShapeMismatch(VisualRefused):
    """The encoder's output disagrees with the grid / policy it was asked for."""


class RasterRefused(VisualRefused):
    """A raster that is not the verified PNG of page ``k`` of its bundle."""


def _require_construction(
    *,
    policy: Any,
    reader: Any,
    model: Any,
    encoder: Any,
    checkpoint_state: Mapping[str, Any],
    feature_source: str,
) -> None:
    from unflatten.model import onfly_visual as ov

    tap = reader.identity.feature_tap
    if tap != policy.feature_tap():
        raise CacheFeatureTapMismatch(
            f"cache feature_tap {tap!r} != the policy's {policy.feature_tap()!r}: the "
            "cache was not built under this visual input policy (dtype included)"
        )
    try:
        ov.require_checkpoint_visual_feature_tap(
            checkpoint_state, expected=policy.visual_feature_tap(), what="RL build_visual"
        )
    except ov.OnTheFlyVisualError as exc:
        raise CheckpointVisualTapMismatch(str(exc)) from exc
    if bool(model.deepstack) != policy.deepstack:
        raise ModelDeepstackMismatch(
            f"model.deepstack={bool(model.deepstack)} but the policy's DeepStack taps are "
            f"{policy.deepstack_visual_indexes!r}"
        )
    if feature_source != tap:
        raise FeatureSourceMismatch(
            f"ports.feature_source {feature_source!r} != the cache identity's feature_tap {tap!r}"
        )
    if encoder.policy.policy_hash() != policy.policy_hash():
        raise EncoderPolicyMismatch(
            "the encoder was built under a different visual input policy than the cache "
            f"binds (encoder compute_dtype {encoder.policy.compute_dtype!r}, policy "
            f"{policy.compute_dtype!r}; hash {encoder.policy.policy_hash()!r} != "
            f"{policy.policy_hash()!r})"
        )


class PageRasterAdapter:
    """The builder chain's ``raster_source`` for ONE page ``(bundle, k)``.

    ``read(bundle, 0)`` is what ``process_model_page`` calls (it knows only page 0 of
    a model page); the adapter returns the bytes of page ``k`` of the bound bundle,
    sha256-checked against the cache row's ``page_image``. ``PinnedRasterSource`` is
    deliberately not used: it answers page 0 of the release for every ``k``.
    """

    def __init__(
        self,
        document_bundle_id: str,
        page_index: int,
        page_image: Mapping[str, Any],
        *,
        store_dir: Path | str,
        expectation_keys: Container[tuple[str, int]],
        fetch: Callable[[Path, Mapping[str, Any]], bytes] | None = None,
    ) -> None:
        if (document_bundle_id, page_index) not in expectation_keys:
            raise PageKeyRefused(
                f"page {(document_bundle_id, page_index)!r} is not in the pinned expectation keys"
            )
        self.document_bundle_id = document_bundle_id
        self.page_index = page_index
        self._page_image = dict(page_image)
        self._store_dir = Path(store_dir)
        self._fetch = fetch
        self._bytes: bytes | None = None

    def read(self, document_bundle_id: str, page_index: int) -> bytes:
        if document_bundle_id != self.document_bundle_id:
            raise RasterRefused(
                f"raster read for bundle {document_bundle_id!r}, adapter is bound to "
                f"{self.document_bundle_id!r}"
            )
        if isinstance(page_index, bool) or page_index != 0:
            raise RasterRefused(
                f"raster read for page {page_index!r}: the chain asks for page 0 of the "
                f"bound page ({self.document_bundle_id!r}, {self.page_index})"
            )
        if self._bytes is None:
            self._bytes = self._verified_bytes()
        return self._bytes

    def _verified_bytes(self) -> bytes:
        from unflatten.dataset.gates.store import StoreError
        from unflatten.model import onfly_visual as ov

        fetch = self._fetch or ov.read_page_png
        try:
            data = fetch(self._store_dir, self._page_image)
        except StoreError as exc:
            raise RasterRefused(
                f"{self.document_bundle_id}/pages/{self.page_index}: {exc.code}"
            ) from exc
        actual = f"sha256:{hashlib.sha256(data).hexdigest()}"
        if actual != self._page_image["sha256"]:
            raise RasterRefused(
                f"{self.document_bundle_id}/pages/{self.page_index}: raster sha256 {actual} "
                f"!= the cache row's page_image {self._page_image['sha256']}"
            )
        return data


def make_build_visual(
    *,
    store_dir: Path | str,
    policy_payload: Mapping[str, Any],
    processor: Any,
    encoder: Any,
    reader: Any,
    expectation: Any,
    model: Any,
    checkpoint_state: Mapping[str, Any],
    feature_source: str,
) -> Callable[[RlPage], PageVisual]:
    """``RealRunPorts.build_visual`` over a 7.0.0 page-image cache.

    ``reader[i]`` addresses ``expectation.target_keys()[i]`` (never the physical page
    order), so the key -> index map is built from the expectation, as ``s2_train``
    does. The five construction-time refusals fire here, before any page is read.
    """
    from unflatten.model import onfly_visual as ov

    policy = ov.VisualInputPolicy.from_payload(policy_payload)
    _require_construction(
        policy=policy,
        reader=reader,
        model=model,
        encoder=encoder,
        checkpoint_state=checkpoint_state,
        feature_source=feature_source,
    )
    keys = tuple(expectation.target_keys())
    index_of = {key: i for i, key in enumerate(keys)}
    key_set = frozenset(keys)
    store = Path(store_dir)

    def build_visual(page: RlPage) -> PageVisual:
        return _build_page_visual(
            page,
            policy=policy,
            processor=processor,
            encoder=encoder,
            reader=reader,
            index_of=index_of,
            keys=key_set,
            store=store,
        )

    return build_visual


def _build_page_visual(
    page: RlPage,
    *,
    policy: Any,
    processor: Any,
    encoder: Any,
    reader: Any,
    index_of: Mapping[tuple[str, int], int],
    keys: frozenset[tuple[str, int]],
    store: Path,
) -> PageVisual:
    import torch

    from unflatten.dataset.gates.store import StoreError
    from unflatten.model import onfly_visual as ov
    from unflatten.model.rl.real_run_ports import PageVisual

    key = (page.page_key[0], page.page_key[1])
    index = index_of.get(key)
    if index is None:
        raise PageKeyRefused(f"page {key!r} is not in the pinned expectation target keys")
    row = reader[index]
    if (row.document_bundle_id, row.page_index) != key:
        raise PageKeyRefused(
            f"cache index {index} is page {(row.document_bundle_id, row.page_index)!r}, "
            f"the page asked for is {key!r}"
        )
    if row.page_image is None:
        raise PageKeyRefused(f"page {key!r}: a feature row reached the page-image path")
    try:
        png = ov.read_page_png(store, row.page_image)
    except StoreError as exc:
        raise RasterRefused(f"{key[0]}/pages/{key[1]}: {exc.code}") from exc
    width, height = ov.open_png(png).size
    expected_grid = policy.grid_thw_for(width, height)
    row_grid = tuple(int(v) for v in row.grid_thw.reshape(-1).tolist())
    if row_grid != expected_grid:
        raise RowGridMismatch(
            f"page {key!r}: the row's grid_thw {list(row_grid)} != {list(expected_grid)} "
            f"computed from its {width}x{height} PNG under the policy"
        )
    pixels = ov.page_pixels(png, processor, policy)
    pooled, levels = encoder.encode(pixels.pixel_values, pixels.grid_thw)
    tokens = policy.visual_tokens_for(expected_grid)
    if pooled.ndim != 2 or int(pooled.shape[0]) != tokens:
        raise VisualShapeMismatch(
            f"page {key!r}: encoder returned {tuple(pooled.shape)}, expected ({tokens}, D)"
        )
    taps = len(policy.deepstack_visual_indexes)
    if len(levels) != taps or any(tuple(lv.shape) != tuple(pooled.shape) for lv in levels):
        raise VisualShapeMismatch(
            f"page {key!r}: {len(levels)} DeepStack levels, policy taps {taps}"
        )
    # one row per visual placeholder, no batch dim (candidate_model's DeepStack check,
    # rollout._checked_levels, S4's OnTheFlyVisualEncoder); visual_features keeps its batch dim
    deepstack = tuple(lv.detach().to(torch.float32).clone() for lv in levels) if taps else None
    return PageVisual(
        visual_features=pooled.detach().to(torch.float32).unsqueeze(0).clone(),
        image_grid_thw=torch.tensor([list(expected_grid)], dtype=torch.int64),
        deepstack_visual_embeds=deepstack,
        raster_source=PageRasterAdapter(
            key[0], key[1], row.page_image, store_dir=store, expectation_keys=keys
        ),
    )

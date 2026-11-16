"""`StaticKVCache`: a preallocated, fixed-capacity KV cache for the
speculative-decode verify path (`docs/runs/dspark-rehearsal-2026-09-27.md`
§19-§20, remaining `encode_steps` cost -- `aten::cat` 8,8 ms/round, 15 % of
the measured 59,4 ms self-CUDA, because HF's `DynamicCache.update`
concatenates K/V on every call).

Subclasses `transformers.cache_utils.DynamicCache` for the same reason
`quantization/kv_cache.Float8KVCache` does: it duck-types (and
`isinstance`-satisfies) the HF `Cache` API every real Qwen attention layer
calls, and never populates the base class's own `key_cache`/`value_cache`,
so `candidate_model._crop_kv_cache` (which special-cases anything with a
`.crop` method) and the stub's plain-tuple path both keep working
unchanged.

Allocates one `(batch, heads, capacity, head_dim)` buffer per layer, lazily,
the first time that layer's `update()` is called (shape/dtype/device are
not known before then). `update()` writes the new chunk into the next free
slice IN PLACE and returns a VIEW over `[:current_length]` -- no
`torch.cat`, no reallocation, for the whole life of the cache as long as
the total token count never exceeds `capacity`. `crop()` only moves the
per-layer length pointer back; the data already written past the new
length is left in the buffer (dead until the next `update()` overwrites
it), exactly the "no dequant/requant, no data movement" property
`Float8KVCacheIncremental`'s design note independently arrived at for the
same reason (cheap rollback matters for speculative decode's per-round
reject path)."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache


class StaticKVCacheCapacityExceeded(ValueError):
    """Raised by `StaticKVCache.update` when a layer would grow past the
    capacity fixed at construction time. A specific type, not a bare
    `ValueError`, so a caller that wants to catch "ran out of preallocated
    room" (e.g. to fall back to a `DynamicCache`) can do so without also
    catching an unrelated `ValueError` from elsewhere in the update path."""


class StaticKVCache(DynamicCache):
    """An HF-`Cache`-compatible KV cache backed by one fixed-size,
    preallocated buffer per layer.

    `capacity` must cover the whole decode this cache will be used for:
    the KV prefix length plus every token appended afterwards (design's
    own guidance: `prefix + max_steps + gamma`, where `gamma` is the
    largest speculative block size that can be appended before a verify
    round's rollback -- `crop()` only moves the pointer back, it never
    frees buffer space for reuse by a longer continuation)."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        super().__init__()
        self._capacity = capacity
        self._keys: dict[int, torch.Tensor] = {}
        self._values: dict[int, torch.Tensor] = {}
        self._lengths: dict[int, int] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        new_len = key_states.shape[-2]
        if layer_idx not in self._keys:
            batch, heads, _, head_dim = key_states.shape
            self._keys[layer_idx] = key_states.new_zeros(batch, heads, self._capacity, head_dim)
            self._values[layer_idx] = value_states.new_zeros(
                batch, heads, self._capacity, head_dim
            )
            self._lengths[layer_idx] = 0
        start = self._lengths[layer_idx]
        end = start + new_len
        if end > self._capacity:
            raise StaticKVCacheCapacityExceeded(
                f"layer {layer_idx} needs {end} positions, capacity is {self._capacity}"
            )
        self._keys[layer_idx][:, :, start:end, :] = key_states
        self._values[layer_idx][:, :, start:end, :] = value_states
        self._lengths[layer_idx] = end
        return self._keys[layer_idx][:, :, :end, :], self._values[layer_idx][:, :, :end, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._lengths.get(layer_idx, 0)

    def crop(self, max_length: int) -> None:
        """Same sign convention as `transformers`' own `DynamicLayer.crop`
        (and `Float8KVCache.crop`): negative removes `abs(max_length)`
        trailing positions, non-negative truncates to that absolute
        length. Moves only `self._lengths[layer_idx]` -- the buffer itself
        is untouched, so this is O(1) regardless of how much was cropped."""
        for layer_idx, seq_len in list(self._lengths.items()):
            new_len = seq_len + max_length if max_length < 0 else min(max_length, seq_len)
            self._lengths[layer_idx] = max(new_len, 0)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for layer_idx, seq_len in self._lengths.items():
            key = self._keys[layer_idx][:, :, :seq_len, :].index_select(
                0, beam_idx.to(self._keys[layer_idx].device)
            )
            value = self._values[layer_idx][:, :, :seq_len, :].index_select(
                0, beam_idx.to(self._values[layer_idx].device)
            )
            self._keys[layer_idx][:, :, :seq_len, :] = key
            self._values[layer_idx][:, :, :seq_len, :] = value

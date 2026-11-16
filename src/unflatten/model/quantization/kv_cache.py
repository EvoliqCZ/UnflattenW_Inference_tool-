"""`Float8KVCache` (owner requirement §2): an incremental-decode KV cache
storing K/V in `torch.float8_e4m3fn` with a per-layer, per-update scale,
dequantizing on read. Subclasses `transformers.cache_utils.DynamicCache` so
it duck-types (and `isinstance`-satisfies) the HF `Cache` API real Qwen
attention layers call -- but never populates the base class's own
`key_cache`/`value_cache`, so it works with `candidate_model.py`'s
`_crop_kv_cache` (which special-cases anything with a `.crop` method) and
with the stub's plain-tuple `past_key_values` path unaffected (a
`Float8KVCache` instance is never a plain tuple, so the stub's
`isinstance`-free `if past_key_values is not None` branch treats it exactly
like a real HF cache object).

Growth strategy: every `update()` call dequantizes what is already stored,
concatenates the new (unquantized) chunk, and re-quantizes the WHOLE
per-layer cache with a single fresh scale. This is `O(n)` work per step
(quadratic over a full decode) rather than `O(1)`, in exchange for a
correctness property that matters more for a first PTQ pass: one scale per
layer that always reflects the CURRENT max magnitude, so an early small-
magnitude prefix does not later clip a much larger value seen mid-decode.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

_FP8_E4M3_MAX = 448.0


def quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric FP8 quantize: `(quantized, scale)` such that
    `quantized.to(dtype) * scale ≈ x`."""
    scale = x.detach().abs().amax().clamp(min=1e-12) / _FP8_E4M3_MAX
    quantized = (x.float() / scale).to(torch.float8_e4m3fn)
    return quantized, scale


def dequantize_fp8(
    quantized: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    return (quantized.to(torch.float32) * scale).to(dtype)


class Float8KVCache(DynamicCache):
    """An HF-`Cache`-compatible KV cache that stores every layer's K/V as
    FP8 + a scalar scale instead of the base class's full-precision
    tensors."""

    def __init__(self) -> None:
        super().__init__()
        self._fp8_keys: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._fp8_values: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._dtype: torch.dtype | None = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        self._dtype = key_states.dtype
        if layer_idx in self._fp8_keys:
            prev_key = dequantize_fp8(*self._fp8_keys[layer_idx], self._dtype)
            prev_value = dequantize_fp8(*self._fp8_values[layer_idx], self._dtype)
            full_key = torch.cat([prev_key, key_states], dim=-2)
            full_value = torch.cat([prev_value, value_states], dim=-2)
        else:
            full_key, full_value = key_states, value_states
        self._fp8_keys[layer_idx] = quantize_fp8(full_key)
        self._fp8_values[layer_idx] = quantize_fp8(full_value)
        return full_key, full_value

    def get_seq_length(self, layer_idx: int = 0) -> int:
        entry = self._fp8_keys.get(layer_idx)
        return 0 if entry is None else entry[0].shape[-2]

    def crop(self, max_length: int) -> None:
        """Same sign convention as `transformers`' own `DynamicLayer.crop`:
        negative removes `abs(max_length)` trailing positions; non-negative
        truncates to that absolute length."""
        for layer_idx in list(self._fp8_keys):
            key = dequantize_fp8(*self._fp8_keys[layer_idx], self._dtype)
            value = dequantize_fp8(*self._fp8_values[layer_idx], self._dtype)
            seq_len = key.shape[-2]
            new_len = seq_len + max_length if max_length < 0 else min(max_length, seq_len)
            new_len = max(new_len, 0)
            self._fp8_keys[layer_idx] = quantize_fp8(key[..., :new_len, :])
            self._fp8_values[layer_idx] = quantize_fp8(value[..., :new_len, :])

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for layer_idx in list(self._fp8_keys):
            key = dequantize_fp8(*self._fp8_keys[layer_idx], self._dtype)
            value = dequantize_fp8(*self._fp8_values[layer_idx], self._dtype)
            key = key.index_select(0, beam_idx.to(key.device))
            value = value.index_select(0, beam_idx.to(value.device))
            self._fp8_keys[layer_idx] = quantize_fp8(key)
            self._fp8_values[layer_idx] = quantize_fp8(value)


#: `Float8KVCacheChunked.update()`/`.crop()` never quantize more than one
#: chunk of this many positions at a time -- the bound that keeps their
#: per-call cost independent of the total cache length (see class
#: docstring). Chosen as "a handful of decode rounds' worth of tokens", not
#: measured/tuned; the owner's speculative block sizes top out at 9.
_CHUNK_TOKENS = 256

_Fp8Chunk = tuple[torch.Tensor, torch.Tensor]


class Float8KVCacheChunked(Float8KVCache):
    """Opt-in per-chunk-scale variant of `Float8KVCache` (owner requirement,
    2026-09-28). `Float8KVCache` itself stays byte-for-byte unchanged -- it
    backs the deployed fp8 service -- this subclass is never constructed
    unless a caller explicitly asks for it.

    Root cause this fixes: the base class re-quantizes the WHOLE per-layer
    cache with one fresh scale on every `update()`/`crop()` call --
    `O(current length)` work per call, ~50 ms/call at `kv_len~8.6k` measured
    in `docs/runs/dspark-rehearsal-2026-09-27.md` §14.2/§16.3. A prior
    attempt (`Float8KVCacheIncremental`, unmerged) kept ONE running scale per
    layer instead, which is `O(new)` for storage but never shrinks the scale
    after a high-magnitude chunk is superseded -- a rejected speculative
    candidate that spiked the scale keeps degrading precision for every
    later, smaller-magnitude token for the rest of the sequence.

    This class instead keeps each layer's K/V as a LIST of independently
    scaled fp8 chunks (each at most `_CHUNK_TOKENS` positions): every
    chunk's scale reflects only that chunk's own contents, so a spike in one
    chunk can never affect another chunk's precision, and there is nothing
    to "shrink" because a stale chunk is simply replaced or dropped, never
    carried forward with a loosened bound.

    - `update()` quantizes only the NEW tokens (merged into the current
      open chunk while it still fits under `_CHUNK_TOKENS`, else started as
      a fresh chunk) -- `O(new)`, never `O(total)`.
    - `crop()` drops whole trailing chunks outright and re-quantizes AT MOST
      the one chunk straddling the new boundary -- `O(one chunk)`, never
      `O(total)`, and unlike the base class needs no dequant/requant of
      chunks that survive untouched.
    - Reads (the tensor `update()`/`crop()` return for the attention call)
      dequantize every surviving chunk fresh each call into a scratch
      buffer -- see the class's `_dequantize_all` docstring for why this,
      not a persistent bf16 mirror, is the right tradeoff here."""

    def __init__(self) -> None:
        super().__init__()
        self._key_chunks: dict[int, list[_Fp8Chunk]] = {}
        self._value_chunks: dict[int, list[_Fp8Chunk]] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        self._dtype = key_states.dtype
        keys = self._key_chunks.setdefault(layer_idx, [])
        values = self._value_chunks.setdefault(layer_idx, [])
        _append_chunk(keys, key_states)
        _append_chunk(values, value_states)
        return self._dequantize_all(keys), self._dequantize_all(values)

    def _dequantize_all(self, chunks: list[_Fp8Chunk]) -> torch.Tensor:
        """Every read re-dequantizes each stored chunk into a fresh scratch
        tensor rather than maintaining a persistent bf16 mirror that grows
        alongside the fp8 chunks.

        Tradeoff, stated explicitly (owner requirement): a persistent bf16
        mirror would make THIS call `O(new)` too, but it would then hold a
        full-precision copy of the entire cache in memory ALONGSIDE the fp8
        chunks for the whole decode -- exactly the memory `Float8KVCache`
        exists to avoid, so it would defeat the point of quantizing at all.
        Re-dequantizing on every read keeps the persistent memory footprint
        fp8-only (chunks' fp8 bytes + one scalar scale each); the transient
        scratch tensor this method allocates is freed once the caller's
        attention forward is done with it, same as the base class's own
        dequantize-on-every-call reads always were. This is `O(total
        chunks)` per call, same order as the base class's own dequant half
        (see module docstring) -- the fix is removing the base class's
        REQUANTIZE-the-whole-cache half, not the unavoidable dequant-for-
        attention half every backend pays."""
        return torch.cat([dequantize_fp8(q, s, self._dtype) for q, s in chunks], dim=-2)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        chunks = self._key_chunks.get(layer_idx)
        return 0 if not chunks else sum(q.shape[-2] for q, _ in chunks)

    def crop(self, max_length: int) -> None:
        """Same sign convention as the base class: negative removes
        `abs(max_length)` trailing positions, non-negative truncates to
        that absolute length."""
        for layer_idx in list(self._key_chunks):
            seq_len = self.get_seq_length(layer_idx)
            new_len = seq_len + max_length if max_length < 0 else min(max_length, seq_len)
            new_len = max(new_len, 0)
            self._key_chunks[layer_idx] = _crop_chunks(
                self._key_chunks[layer_idx], new_len, self._dtype
            )
            self._value_chunks[layer_idx] = _crop_chunks(
                self._value_chunks[layer_idx], new_len, self._dtype
            )

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for layer_idx in list(self._key_chunks):
            key = self._dequantize_all(self._key_chunks[layer_idx])
            value = self._dequantize_all(self._value_chunks[layer_idx])
            key = key.index_select(0, beam_idx.to(key.device))
            value = value.index_select(0, beam_idx.to(value.device))
            self._key_chunks[layer_idx] = [quantize_fp8(key)]
            self._value_chunks[layer_idx] = [quantize_fp8(value)]


def _append_chunk(chunks: list[_Fp8Chunk], new_states: torch.Tensor) -> None:
    """Merge `new_states` into the last chunk when it still fits under
    `_CHUNK_TOKENS` (re-quantizing only that merged chunk, bounded size),
    else start a fresh chunk (quantizing only `new_states` itself)."""
    if chunks:
        last_q, last_scale = chunks[-1]
        if last_q.shape[-2] + new_states.shape[-2] <= _CHUNK_TOKENS:
            merged = torch.cat(
                [dequantize_fp8(last_q, last_scale, new_states.dtype), new_states], dim=-2
            )
            chunks[-1] = quantize_fp8(merged)
            return
    chunks.append(quantize_fp8(new_states))


def _crop_chunks(chunks: list[_Fp8Chunk], new_len: int, dtype: torch.dtype) -> list[_Fp8Chunk]:
    """Keep whole chunks up to `new_len` untouched, drop whole trailing
    chunks past it, and re-quantize (bounded by one chunk's size) only the
    single chunk straddling the boundary, if any."""
    kept: list[_Fp8Chunk] = []
    remaining = new_len
    for q, scale in chunks:
        if remaining <= 0:
            break
        length = q.shape[-2]
        if length <= remaining:
            kept.append((q, scale))
            remaining -= length
        else:
            sliced = dequantize_fp8(q, scale, dtype)[..., :remaining, :]
            kept.append(quantize_fp8(sliced))
            remaining = 0
    return kept

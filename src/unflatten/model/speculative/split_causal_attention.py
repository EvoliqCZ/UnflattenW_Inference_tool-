"""Split-causal flash attention for HF's q_len>1 verify-block forward.

`docs/runs/dspark-rehearsal-2026-09-27.md` §19-§20: under installed
`transformers==5.16.1`, `Qwen3VLTextModel.forward` builds its mask via
`create_causal_mask`, which only skips materialization (`_ignore_causal_
mask_sdpa`) when `q_length == 1`, `kv_length == q_length`, or `q_offset ==
0`. A speculative-decode VERIFY block is `q = k > 1` over a non-empty KV
cache -- none of those hold, so an explicit 4D bool mask is built and
handed to every attention layer regardless of what `attention_mask`
argument the caller passed in (§16's fix, passing `attention_mask=None`
into `self._decode`, is therefore a no-op on this path -- confirmed by
kernel name in §19.3). With an explicit mask, PyTorch's SDPA dispatcher
sets `is_causal=False` and falls onto the ~10x slower memory-efficient
kernel instead of flash (measured: `fmha_cutlassF_bf16_aligned_64x128...`
484us/call vs `flash_fwd_splitkv_kernel` 48us/call, same shape).

This module bypasses `create_causal_mask` entirely for that one shape by
registering a new HF attention implementation, `unflatten_split_causal`,
through `AttentionInterface`/`AttentionMaskInterface`: the mask side always
skips (returns `None`), and the attention side derives causality
structurally instead of from a mask tensor -- the cached prefix is fully
visible to every new query row (it is strictly in the past), and the new
block is top-left causal among itself (query row i sees block key j iff j
<= i), which is exactly `is_causal=True`'s native alignment when q_len ==
kv_len. Two mask-free flash calls, one per region, combined exactly via
log-sum-exp rescaling (the same identity FlashAttention's own block-
parallel/ring-attention forms use).

A caller opts a specific `decode`/`_decode` call into this path by setting
`language_model.config._attn_implementation = ATTN_IMPLEMENTATION_NAME` for
the duration of that one call (see `candidate_loader.py`'s `decode`
closure) -- every attention layer reads `self.config._attn_implementation`
fresh on each forward (confirmed by reading the installed
`modeling_qwen3_vl.py`), so this is safe to toggle per call without
touching the DEC-007 D3 production `"sdpa"` pin anywhere else.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

#: HF `_attn_implementation` name. A caller must set this on a REAL
#: backbone's `language_model.config` (never the stub, which has no HF
#: config or real attention kernel) to route one `decode` call through this
#: module instead of SDPA/eager.
ATTN_IMPLEMENTATION_NAME = "unflatten_split_causal"


def _skip_mask(*args: Any, **kwargs: Any) -> None:
    """`AttentionMaskInterface` entry for `ATTN_IMPLEMENTATION_NAME`: always
    skip mask materialization. Safe because `split_causal_attention_forward`
    never reads the mask it is handed -- causality is derived structurally
    -- and every caller that requests this implementation has already
    established there is no padding (single-page decode only, see
    `candidate_model.encode_steps`'s own docstring)."""
    return None


def split_causal_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """`(batch, heads, seq, head_dim)` in, `(batch, seq, heads, head_dim)`
    out -- same contract as `transformers.integrations.sdpa_attention.
    sdpa_attention_forward`, which this delegates to for every shape/dtype/
    device it was not written for (non-CUDA, dropout>0, fp32, or an actual
    mask a caller passed anyway -- defensive; `_skip_mask` means none of
    our own callers ever produce one, but a future direct caller of this
    function might).

    `query`: `(batch, heads_q, k, head_dim)`, the k new (candidate) block
    positions only. `key`/`value`: `(batch, heads_kv, N, head_dim)`, the
    FULL cache -- the caller's cache update has already concatenated the
    cached prefix (length `N - k`) with the new block's own keys/values
    before this is called. Split back into prefix (`[:N-k]`, fully visible
    to every query row) and block (`[N-k:]`, top-left causal). When there
    is no prefix (`N == k`, e.g. a from-empty prefill), falls back to one
    plain causal flash call over the whole thing instead of the split.

    The two flash calls' outputs are combined exactly via log-sum-exp
    rescaling, in fp32 regardless of `query.dtype`, to avoid a second bf16
    rounding on top of each call's own:
    `m = max(lse_a, lse_b)`; `out = (out_a*e^{lse_a-m} + out_b*e^{lse_b-m})
    / (e^{lse_a-m} + e^{lse_b-m})`.
    """
    if (
        not query.is_cuda
        or dropout > 0.0
        or attention_mask is not None
        or query.dtype not in (torch.float16, torch.bfloat16)
    ):
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs
        )

    # GQA without `repeat_kv`: materializing the FULL cache n_rep times was
    # ~200 us of copy per call, per layer, per round (profile 2026-09-28,
    # 14.4 of 55.7 ms CUDA per verify). The flash op takes fewer KV heads
    # natively (bit-identical output), and for the fully visible prefix the
    # n_rep query heads sharing one KV head are folded into one head with
    # n_rep*k query rows (measured 707 -> 48 us per call at kv 8 346).
    batch, heads_q, k, head_dim = query.shape
    heads_kv = key.shape[1]
    n_rep = heads_q // heads_kv
    kv_length = key.shape[2]
    prefix_length = kv_length - k
    if prefix_length <= 0:
        out, _lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            query, key, value, dropout_p=0.0, is_causal=True, scale=scaling
        )
        return out.transpose(1, 2).contiguous(), None

    key_prefix, key_block = key[..., :prefix_length, :], key[..., prefix_length:, :]
    value_prefix, value_block = value[..., :prefix_length, :], value[..., prefix_length:, :]

    folded = query.reshape(batch, heads_kv, n_rep * k, head_dim)
    out_a, lse_a, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        folded, key_prefix, value_prefix, dropout_p=0.0, is_causal=False, scale=scaling
    )
    out_a = out_a.reshape(batch, heads_q, k, head_dim)
    lse_a = lse_a.reshape(batch, heads_q, k)
    out_b, lse_b, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        query, key_block, value_block, dropout_p=0.0, is_causal=True, scale=scaling
    )

    lse_a = lse_a.float().unsqueeze(-1)
    lse_b = lse_b.float().unsqueeze(-1)
    m = torch.maximum(lse_a, lse_b)
    weight_a = torch.exp(lse_a - m)
    weight_b = torch.exp(lse_b - m)
    combined = (out_a.float() * weight_a + out_b.float() * weight_b) / (weight_a + weight_b)
    return combined.to(query.dtype).transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS.register(ATTN_IMPLEMENTATION_NAME, split_causal_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register(ATTN_IMPLEMENTATION_NAME, _skip_mask)

"""Piecewise CUDA graphs for the single-token (q=1) decode step
(owner speed item 1, 2026-09-29, `docs/runs/cuda-graphs-decode-2026-09-29.md`).

Opt-in through `CandidateRuntime.kv_cache_mode = "static_graph"`, which seeds
the decode with a `GraphedStaticKVCache`. Every other mode, and every call
this module declines (`graphed_decode_step_or_none`), runs exactly as before.

**Why piecewise, not one graph over the whole step.** A single graph needs
every shape fixed, so attention would have to run over the full
preallocated capacity with a length mask (HF `StaticCache` style, or
`torch.compile(mode="reduce-overhead")` on top of it). That changes the
attention kernel and its reduction order -- the tokens would no longer be
guaranteed identical to `kv_cache_mode="static"` -- and it reads the unused
tail of the cache on every step (capacity is `prefix + max_steps`, ~13 k
positions for a ~8.6 k page). `torch.compile` additionally fuses/reorders
the elementwise tail through Inductor, which again moves the numbers.
Instead, each decoder layer is split at attention:

    segment 0      rotary + layer 0 input norm, q/k/v, q/k norm, rope,
                   K/V write at a device-side position
    attention l    EAGER: HF's `sdpa_attention_forward` over the exact
                   `[:length]` view -- the very call `"static"` makes
    segment l+1    layer l o_proj, residual, post norm, MLP, residual,
                   then layer l+1's pre-attention part (as segment 0)
    segment n      last layer's post-attention part + final norm

Segments have fixed shapes (q=1, hidden width) and no data-dependent host
logic, so each is one CUDA graph; they share one memory pool and replay in
capture order. The eager attention between them sees exactly the tensors
and view shapes the `"static"` path sees, so the result is bit-identical by
construction, not by tolerance (`tests/model/test_decode_graph.py` proves
the decomposition against HF's forward on CPU, and replay against it on
CUDA). The cost is 37 graph replays + 36 SDPA launches per step instead of
one replay -- host work that stays far below the ~20 ms of GPU work per
step, so the GPU is no longer waiting on Python.

Specific to Qwen3-VL's text decoder (`Qwen3VLTextDecoderLayer`), under the
DEC-007 D3 `"sdpa"` attention pin; anything else is refused, not guessed.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    apply_rotary_pos_emb,
)

from unflatten.model.speculative.static_kv_cache import (
    StaticKVCache,
    StaticKVCacheCapacityExceeded,
)


class GraphedStaticKVCache(StaticKVCache):
    """A `StaticKVCache` whose plain q=1 decode steps run through
    `PiecewiseDecodeGraph` (captured on the first such step, per page).

    Prefill, verify blocks (q>1) and any q=1 call the graph path declines go
    through the inherited `update()` unchanged, so this is a drop-in
    `StaticKVCache` for every caller. `crop()` stays O(1): the graphed step
    reads the write position from the host length before each replay."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.graph_runner: PiecewiseDecodeGraph | None = None

    def graphed_step(
        self,
        language_model: Any,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One q=1 step: writes this token's K/V at the current length in
        every layer and returns the final-norm hidden state `(1, 1, hidden)`
        (a fresh tensor, safe to keep across later steps)."""
        layers = len(language_model.layers)
        if len(self._keys) != layers:
            raise ValueError(
                f"graphed decode step needs a prefilled cache ({len(self._keys)} of "
                f"{layers} layers allocated)"
            )
        lengths = set(self._lengths.values())
        if len(lengths) != 1:
            raise ValueError(f"graphed decode step needs equal layer lengths, got {lengths}")
        length = lengths.pop()
        if length + 1 > self._capacity:
            raise StaticKVCacheCapacityExceeded(
                f"graphed step needs {length + 1} positions, capacity is {self._capacity}"
            )
        if self.graph_runner is None:
            self.graph_runner = PiecewiseDecodeGraph(
                language_model,
                keys=[self._keys[i] for i in range(layers)],
                values=[self._values[i] for i in range(layers)],
            )
        hidden = self.graph_runner.step(inputs_embeds, position_ids, length)
        for layer_idx in self._lengths:
            self._lengths[layer_idx] = length + 1
        return hidden


def graphed_decode_step_or_none(
    language_model: Any,
    *,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor | None,
    past_key_values: Any,
    use_cache: bool,
    output_hidden_states: bool,
    attn_implementation_override: str | None,
) -> BaseModelOutputWithPast | None:
    """The `candidate_loader.decode` hook: the graphed step's output for a
    plain q=1 cached step over a `GraphedStaticKVCache`, else `None` (the
    caller then runs HF's forward exactly as before). Declined: any other
    cache, q>1 (verify blocks), `use_cache=False`, `output_hidden_states`
    (DSpark taps) and an attention override (split-causal verify)."""
    if (
        not isinstance(past_key_values, GraphedStaticKVCache)
        or inputs_embeds.shape[:2] != (1, 1)
        or position_ids is None
        or not use_cache
        or output_hidden_states
        or attn_implementation_override is not None
    ):
        return None
    hidden = past_key_values.graphed_step(language_model, inputs_embeds, position_ids)
    return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values)


class PiecewiseDecodeGraph:
    """The segment decomposition described in the module docstring, bound to
    one page's KV buffers. On CUDA the segments are captured as CUDA graphs
    on the first `step()`; on CPU (tests) the same segments run eagerly."""

    def __init__(
        self,
        language_model: Any,
        *,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> None:
        if language_model.config._attn_implementation != "sdpa":
            raise ValueError(
                "graphed decode step supports only the 'sdpa' attention pin, got "
                f"{language_model.config._attn_implementation!r}"
            )
        self._layers = list(language_model.layers)
        if not all(isinstance(layer, Qwen3VLTextDecoderLayer) for layer in self._layers):
            raise ValueError("graphed decode step supports only Qwen3VLTextDecoderLayer")
        self._rotary = language_model.rotary_emb
        self._norm = language_model.norm
        self._keys = keys
        self._values = values
        device, dtype = keys[0].device, keys[0].dtype
        hidden_size = language_model.config.hidden_size
        attn = self._layers[0].self_attn
        attn_width = language_model.config.num_attention_heads * attn.head_dim
        # Static inputs: filled before every replay, read inside the graphs.
        self._embeds = torch.zeros(1, 1, hidden_size, device=device, dtype=dtype)
        self._position_ids = torch.zeros(3, 1, 1, device=device, dtype=torch.long)
        self._cache_position = torch.zeros(1, device=device, dtype=torch.long)
        self._attn_out = torch.zeros(1, 1, attn_width, device=device, dtype=dtype)
        self._use_graphs = device.type == "cuda"
        self._graphs: list[torch.cuda.CUDAGraph] = []
        # Segment outputs: (residual, query) per layer, then the final hidden.
        self._residual: list[torch.Tensor] = []
        self._query: list[torch.Tensor] = []
        self._rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self._out: torch.Tensor | None = None

    @property
    def captured(self) -> bool:
        return bool(self._graphs)

    def step(
        self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, length: int
    ) -> torch.Tensor:
        self._embeds.copy_(inputs_embeds)
        self._position_ids.copy_(position_ids.reshape(3, 1, 1))
        self._cache_position.fill_(length)
        if not self._use_graphs:
            self._run(length, segment_runner=self._segment)
        else:
            if not self._graphs:
                self._capture(length)
            self._run(length, segment_runner=lambda index: self._graphs[index].replay())
        assert self._out is not None
        return self._out.clone()

    # -- the decomposition ------------------------------------------------

    def _pre_attention(self, layer_idx: int, residual: torch.Tensor) -> torch.Tensor:
        """`Qwen3VLTextDecoderLayer.forward` up to the attention call, plus
        the K/V write `StaticKVCache.update` would do (a copy either way)."""
        layer = self._layers[layer_idx]
        attn = layer.self_attn
        hidden = layer.input_layernorm(residual)
        shape = (1, 1, -1, attn.head_dim)
        query = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
        key = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
        value = attn.v_proj(hidden).view(shape).transpose(1, 2)
        assert self._rope is not None
        cos, sin = self._rope
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        self._keys[layer_idx].index_copy_(2, self._cache_position, key)
        self._values[layer_idx].index_copy_(2, self._cache_position, value)
        return query

    def _post_attention(self, layer_idx: int, residual: torch.Tensor) -> torch.Tensor:
        layer = self._layers[layer_idx]
        hidden = residual + layer.self_attn.o_proj(self._attn_out)
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        return residual + hidden

    def _segment(self, index: int) -> None:
        """Segment `index` in `0..n_layers`; stores its outputs on `self`
        (during capture those are the graph's static output tensors)."""
        n_layers = len(self._layers)
        if index == 0:
            self._rope = self._rotary(self._embeds, self._position_ids)
            residual = self._embeds
        else:
            residual = self._post_attention(index - 1, self._residual[index - 1])
        if index == n_layers:
            self._out = self._norm(residual)
            return
        query = self._pre_attention(index, residual)
        if len(self._residual) == index:
            self._residual.append(residual)
            self._query.append(query)
        else:
            self._residual[index] = residual
            self._query[index] = query

    def _attention(self, layer_idx: int, length: int) -> None:
        """EAGER: the exact call HF's `Qwen3VLTextAttention` makes for q=1
        under the `"static"` cache -- same function, same view shapes, no
        mask (`test_reference_q1_step_hands_sdpa_no_mask`)."""
        attn = self._layers[layer_idx].self_attn
        end = length + 1
        output, _ = sdpa_attention_forward(
            attn,
            self._query[layer_idx],
            self._keys[layer_idx][:, :, :end, :],
            self._values[layer_idx][:, :, :end, :],
            None,
            dropout=0.0,
            scaling=attn.scaling,
        )
        self._attn_out.copy_(output.reshape(1, 1, -1))

    def _run(self, length: int, *, segment_runner: Any) -> None:
        for index in range(len(self._layers) + 1):
            segment_runner(index)
            if index < len(self._layers):
                self._attention(index, length)

    def _capture(self, length: int) -> None:
        """Warm up once eagerly on a side stream (cuBLAS/cuBLASLt handles,
        lazy module state), then capture every segment into its own graph
        on that stream, sharing one pool. The warm-up writes this step's K/V
        at `length`; the replay that follows rewrites the same slot with the
        same values. `capture_begin`/`capture_end` directly, not
        `torch.cuda.graph`, which empties the caching allocator on every
        entry -- 37 times per page here."""
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._run(length, segment_runner=self._segment)
            pool = torch.cuda.graph_pool_handle()
            for index in range(len(self._layers) + 1):
                graph = torch.cuda.CUDAGraph()
                graph.capture_begin(pool=pool)
                try:
                    self._segment(index)
                finally:
                    graph.capture_end()
                self._graphs.append(graph)
        torch.cuda.current_stream().wait_stream(stream)

"""`DocumentLevelRouter` -- RFC007 variant B, query-side level routing.

Why query-side (RFC007 §3.1): native DeepStack adds its levels to the
decoder hidden state at the visual positions, once, during prefill; those
positions then sit frozen in the KV cache. Nothing that happens there can
depend on what the decoder is generating later. So the per-step,
grammar-conditioned choice has to be made on the QUERY side, after the
frozen decoder:

```text
levels (final, deepstack.0..2), each (B, N, visual_dim)       -- per page, once
  -> per-level LayerNorm + Linear -> memory (B, L, N, memory_dim)
hidden_t (after the expert adapter) + grammar state s_t        -- per position
  -> router: softmax(W_r [LN(h_t); E(s_t)] / T) -> w_t (L,)
  -> query:  q_t = W_q [LN(h_t); E(s_t)]
  -> read_l = attention(q_t, memory_l)            (multi-head, masked padding)
  -> delta_t = W_o sum_l w_t,l read_l             (W_o zero-initialised)
hidden_t' = hidden_t + delta_t
```

Per-position and batch-independent: position t reads only its own hidden
state, its own grammar state and the fixed per-page memory, so a full
teacher-forced pass and a KV-cache incremental decode agree exactly, and
nothing here touches the KV cache (RFC007 §9). `W_o` starts at exactly
zero, so a freshly built router adds exactly nothing (DEC-047 condition 2);
the gradient reaches `W_o` first and the router/query afterwards, the same
warm-start shape LoRA's zero `B` uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from unflatten.model.level_router.grammar_state import num_grammar_states
from unflatten.model.level_router.policy import FINAL_LEVEL, LevelRouterPolicy

#: The trainable-group id this module's parameters map to
#: (`candidate_model.parameter_group_ids`). Deliberately NOT a row in
#: `trainable_group_registry` or any v15b stage-state policy (DEC-047
#: condition 3): until a ratified v16 stage policy names it, `build_s4_optimizer`
#: refuses these parameters by name.
LEVEL_ROUTER_GROUP_ID = "visual_level_router"


class LevelRouterInputMissing(ValueError):
    """A level the policy reads is absent or misshaped in the batch."""


@dataclass(frozen=True)
class LevelMemory:
    """Projected per-level visual memory for one batch of pages.

    `values` is `(batch, levels, tokens, memory_dim)`; `key_padding_mask`
    is `(batch, tokens)`, True == padding (the `VisualMemory` convention)."""

    values: torch.Tensor
    key_padding_mask: torch.Tensor


@dataclass(frozen=True)
class LevelRouterOutput:
    delta: torch.Tensor  # (batch, positions, d_model)
    level_weights: torch.Tensor  # (batch, positions, levels)


def gather_level_features(
    policy: LevelRouterPolicy,
    *,
    visual_features: torch.Tensor,
    visual_valid_lengths: torch.Tensor | None,
    deepstack_visual_embeds: Sequence[torch.Tensor] | None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """The policy's levels as padded `(batch, tokens, dim)` tensors plus the
    shared key-padding mask.

    `"final"` is `visual_features` as given. `"deepstack.<k>"` is
    `deepstack_visual_embeds[k]`, whose rows arrive flat in row-major
    (page, token) order -- the v16 on-the-fly ViT contract
    (`TeacherForcedBatch.deepstack_visual_embeds`, DEC-056) -- and are
    scattered back onto the same padded page grid `visual_features` uses.
    """
    batch, tokens, _ = visual_features.shape
    if visual_valid_lengths is None:
        valid = torch.ones(batch, tokens, dtype=torch.bool, device=visual_features.device)
    else:
        positions = torch.arange(tokens, device=visual_features.device)
        valid = positions.unsqueeze(0) < visual_valid_lengths.to(visual_features.device).view(-1, 1)
    total_valid = int(valid.sum())
    levels: list[torch.Tensor] = []
    for name in policy.level_names:
        if name == FINAL_LEVEL:
            levels.append(visual_features)
            continue
        index = int(name.split(".", 1)[1])
        if deepstack_visual_embeds is None:
            raise LevelRouterInputMissing(
                f"level router policy reads {name!r} but the batch carries no "
                "deepstack_visual_embeds (DEC-056 on-the-fly ViT levels)"
            )
        if index >= len(deepstack_visual_embeds):
            raise LevelRouterInputMissing(
                f"level router policy reads {name!r} but the batch carries only "
                f"{len(deepstack_visual_embeds)} DeepStack level(s)"
            )
        flat = deepstack_visual_embeds[index]
        if flat.dim() != 2 or flat.shape[0] != total_valid:
            raise LevelRouterInputMissing(
                f"{name!r} has shape {tuple(flat.shape)}; expected ({total_valid}, dim) rows, "
                "one per valid visual token in row-major (page, token) order"
            )
        padded = flat.new_zeros(batch, tokens, flat.shape[1])
        padded[valid] = flat
        levels.append(padded)
    return tuple(levels), ~valid


class DocumentLevelRouter(nn.Module):
    def __init__(self, *, d_model: int, visual_dim: int, policy: LevelRouterPolicy) -> None:
        super().__init__()
        self.policy = policy
        self.num_states = num_grammar_states(policy)
        levels = len(policy.level_names)
        self.level_projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, policy.memory_dim))
            for _ in range(levels)
        )
        self.state_embedding = nn.Embedding(self.num_states, policy.state_embedding_dim)
        self.query_norm = nn.LayerNorm(d_model)
        condition_dim = d_model + policy.state_embedding_dim
        self.router = nn.Linear(condition_dim, levels)
        self.query = nn.Linear(condition_dim, policy.memory_dim)
        self.output = nn.Linear(policy.memory_dim, d_model)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def build_memory(
        self, level_features: Sequence[torch.Tensor], key_padding_mask: torch.Tensor
    ) -> LevelMemory:
        if len(level_features) != len(self.level_projections):
            raise LevelRouterInputMissing(
                f"expected {len(self.level_projections)} level tensors, got {len(level_features)}"
            )
        dtype = self.output.weight.dtype
        # Frozen-side features are cast UP to the trainable dtype (candidate_
        # model's dtype rule): each level is (tokens x visual_dim) per page,
        # a small tensor next to the frozen decoder's weights.
        values = torch.stack(
            [
                projection(features.to(dtype))
                for projection, features in zip(self.level_projections, level_features)
            ],
            dim=1,
        )
        return LevelMemory(values=values, key_padding_mask=key_padding_mask)

    def forward(
        self, hidden: torch.Tensor, state_ids: torch.Tensor, memory: LevelMemory
    ) -> LevelRouterOutput:
        dtype = self.output.weight.dtype
        condition = torch.cat(
            [self.query_norm(hidden.to(dtype)), self.state_embedding(state_ids)], dim=-1
        )
        temperature = self.policy.router_temperature_milli / 1000.0
        weights = torch.softmax(self.router(condition) / temperature, dim=-1)

        batch, positions, _ = hidden.shape
        heads = self.policy.num_heads
        head_dim = self.policy.memory_dim // heads
        query = self.query(condition).view(batch, positions, heads, head_dim).transpose(1, 2)
        attend = (~memory.key_padding_mask).view(batch, 1, 1, -1)
        reads = []
        for level in range(memory.values.shape[1]):
            kv = memory.values[:, level]
            kv = kv.view(batch, kv.shape[1], heads, head_dim).transpose(1, 2)
            read = F.scaled_dot_product_attention(query, kv, kv, attn_mask=attend)
            reads.append(read.transpose(1, 2).reshape(batch, positions, -1))
        combined = (weights.unsqueeze(-1) * torch.stack(reads, dim=2)).sum(dim=2)
        return LevelRouterOutput(delta=self.output(combined), level_weights=weights)

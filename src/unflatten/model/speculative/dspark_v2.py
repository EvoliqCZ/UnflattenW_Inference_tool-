"""`DflashConfig(variant="dspark_v2")`: LiquidAI DSpark, architecture per the
paper/model cards (2026-09-28 owner instruction), not the simplified
attention-only recipe `dspark.py`'s `DSparkDrafter` (`"dspark"` variant)
implements. Dormant, opt-in, DEC-047-shaped exactly like every other variant
in this package -- nothing here is imported by any decode entry point.

**Architecture (paper: arXiv:2607.05147v1, "DSpark: Confidence-Scheduled
Speculative Decoding with Semi-Autoregressive Generation"; model cards
LiquidAI/LFM2.5-2.6B-DSpark, LFM2.5-1.2B-Instruct-DSpark).**

- **Context injection**: ``H_ctx = RMSNorm(W_c [H^(l_1); ...; H^(l_m)])`` --
  the target's tapped hidden-decoder-layer states (:data:`DflashConfig.
  dspark_hidden_taps`, plus the final dense-adapted state when
  ``dspark_final_tap``) concatenated, one shared linear projection, one
  RMSNorm. Prepended as the first token of the drafter's own short
  sequence, so every draft layer's self-attention (over its own KV) sees it
  -- the paper's "prepended/concatenated into each draft layer's KV
  sequence" (this module concatenates once, at the input, rather than
  re-injecting into every layer's KV cache separately -- the two are
  equivalent for a same-length forward with no incremental KV reuse across
  blocks, which is what a single block-state forward is).
- **Block layout**: ``block_size`` doubles as the paper's block width
  ``gamma`` -- slot 0 is the ANCHOR (the primary's own next token, known
  exactly, never drafted -- :data:`DflashConfig.dspark_seed_first_token`,
  required for this variant), slots ``1..block_size-1`` are ``gamma - 1``
  mask-embedded positions, refined together in ONE transformer forward
  (:meth:`DSparkV2Drafter._block_states_batched`) -- ``h_1..h_{gamma-1}``
  all come out of that single pass, none of them re-running the 5-layer
  stack.
- **5 full pre-norm transformer layers** (:class:`_DSparkV2Layer`):
  multi-head self-attention with grouped-query attention (``dspark_v2_
  n_heads`` query heads sharing ``dspark_v2_n_kv_heads`` key/value heads),
  RMSNorm, and a DENSE SwiGLU feedforward (the paper's own backbone uses an
  MoE FFN in its setting; ours is dense per the owner's explicit
  instruction here).
- **Low-rank Markov head** (:class:`_LowRankMarkovHead`): ``B(x_{k-1}, .) =
  W1[x_{k-1}] @ W2``, rank ``dspark_v2_markov_rank`` (paper: 256), ADDED to
  slot ``k``'s base logits (``output_head(h_k)``). Sequential AT INFERENCE
  (:meth:`DSparkV2Drafter.draft_blocks` conditions on the just-drafted
  token), teacher-forced (the target/sampled previous token) in
  :meth:`DSparkV2Drafter.forward_training_blocks` -- the same
  parallel-training/sequential-inference asymmetry `dspark.py`'s
  ``DSparkDrafter`` already uses, named in that module's own docstring.
- **Confidence head**: ``c_k = sigmoid(w^T [h_k ; W1[x_{k-1}]])`` -- a
  per-slot survival probability, trained with BCE against the actual
  accept/reject label a training batch's teacher-forced logits already
  imply (slot ``k`` "survives" iff its own AND every earlier slot's argmax
  matches the target -- :func:`survival_labels`). Consumed by
  ``decode_integration``'s confidence-scheduled verification
  (``drafter.choose_verify_length``) through
  :meth:`DSparkV2SpeculativeDrafter.draft_with_confidence`.

**Not implemented here (see the 2026-09-28 rehearsal doc §18 for why):**
the paper's total-variation loss against the target's own probability
distribution (``s7_dflash_train.py``'s ``DSparkTapsRecord`` stores only the
target's greedy TOKEN stream, not its logits -- there is nothing to take a
TV distance against without re-exporting taps with full distributions, a
separate, larger change out of this task's scope). CE + confidence BCE are
implemented and mandatory, as instructed.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from unflatten.model.speculative.dflash import DflashConfig
from unflatten.model.speculative.drafter import DecoderSnapshot

logger = __import__("logging").getLogger(__name__)


class RMSNorm(nn.Module):
    """Standard RMSNorm (no mean-centering, unlike LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight


class _SwiGLU(nn.Module):
    """Dense SwiGLU FFN: ``down(silu(gate(x)) * up(x))``. No MoE routing --
    the owner's explicit instruction for this drafter, unlike the paper's
    own backbone setting."""

    def __init__(self, d_model: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _GQAAttention(nn.Module):
    """Grouped-query causal self-attention: `n_heads` query heads share
    `n_kv_heads` key/value heads (`n_heads % n_kv_heads == 0`)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        groups = self.n_heads // self.n_kv_heads

        def split(t: torch.Tensor, heads: int) -> torch.Tensor:
            return t.view(batch, seq, heads, self.head_dim).transpose(1, 2)

        q = split(self.q_proj(x), self.n_heads)
        k = split(self.k_proj(x), self.n_kv_heads).repeat_interleave(groups, dim=1)
        v = split(self.v_proj(x), self.n_kv_heads).repeat_interleave(groups, dim=1)
        attn_bias = torch.zeros(seq, seq, dtype=q.dtype, device=q.device)
        attn_bias.masked_fill_(causal_mask, float("-inf"))
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).reshape(batch, seq, self.n_heads * self.head_dim)
        return self.o_proj(out)


class _DSparkV2Layer(nn.Module):
    """One pre-norm transformer layer: GQA self-attention + dense SwiGLU FFN."""

    def __init__(self, config: DflashConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = _GQAAttention(
            config.d_model,
            config.dspark_v2_n_heads,
            config.dspark_v2_n_kv_heads,
            config.dspark_v2_head_dim,
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = _SwiGLU(config.d_model, config.dspark_v2_ffn_intermediate)

    def forward(self, x: torch.Tensor, *, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), causal_mask=causal_mask)
        return x + self.ffn(self.ffn_norm(x))


class DSparkV2Drafter(nn.Module):
    """See module docstring. `config.variant` must be `"dspark_v2"`."""

    def __init__(
        self,
        config: DflashConfig,
        *,
        frozen_token_embedding_weight: torch.Tensor | None = None,
        frozen_output_head_weight: torch.Tensor | None = None,
        frozen_output_head_bias: torch.Tensor | None = None,
    ) -> None:
        """Same frozen-table option as `DflashDrafter`/`DSparkDrafter`
        (see their docstrings) -- `None` (default) keeps trainable tables."""
        super().__init__()
        if config.variant != "dspark_v2":
            raise ValueError(
                f"DSparkV2Drafter requires config.variant='dspark_v2', got {config.variant!r}"
            )
        self.config = config
        n_taps = config.n_tap_inputs
        self.context_proj = nn.Linear(config.hidden_dim * n_taps, config.d_model, bias=False)
        self.context_norm = RMSNorm(config.d_model)
        if frozen_token_embedding_weight is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                frozen_token_embedding_weight, freeze=True
            )
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_embedding = nn.Parameter(torch.zeros(config.d_model))
        self.layers = nn.ModuleList(
            _DSparkV2Layer(config) for _ in range(config.dspark_v2_n_layers)
        )
        if frozen_output_head_weight is not None:
            self.output_head = nn.Linear(
                config.d_model, config.vocab_size, bias=frozen_output_head_bias is not None
            )
            with torch.no_grad():
                self.output_head.weight.copy_(frozen_output_head_weight)
                if frozen_output_head_bias is not None:
                    self.output_head.bias.copy_(frozen_output_head_bias)
            self.output_head.weight.requires_grad_(False)
            if frozen_output_head_bias is not None:
                self.output_head.bias.requires_grad_(False)
        else:
            self.output_head = nn.Linear(config.d_model, config.vocab_size)
        rank = config.dspark_v2_markov_rank
        self.markov_embed = nn.Embedding(config.vocab_size, rank)
        self.markov_out = nn.Linear(rank, config.vocab_size, bias=False)
        self.confidence_head = nn.Linear(config.d_model + rank, 1)

    def _flatten_taps(self, taps: torch.Tensor) -> torch.Tensor:
        """`(B, n_taps, hidden_dim) -> (B, n_taps * hidden_dim)`."""
        return taps.reshape(taps.shape[0], -1)

    def _block_states_batched(
        self,
        *,
        hidden_taps: torch.Tensor,
        prefix_windows: torch.Tensor,
        prefix_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """ONE forward through the 5-layer stack for a whole batch of
        blocks: context token (from `hidden_taps`) + left-aligned, padded
        prefix window + `block_size` block positions (anchor's own mask
        slot 0, mask slots 1..block_size-1) scattered right after each
        row's actual prefix length. Causal attention means the trailing
        padding after a row's block never reaches anything read out.
        Returns `(B, block_size, d_model)`, the block's OWN positions'
        hidden states (index 0 is the anchor's, `1..block_size-1` are the
        drafted slots')."""
        window = self.config.dspark_prefix_window
        if window is None:
            raise ValueError("DSparkV2Drafter requires config.dspark_prefix_window")
        if prefix_windows.shape[1] > window:
            raise ValueError(
                f"prefix_windows is {prefix_windows.shape[1]} wide, window is {window}"
            )
        block = self.config.block_size
        device = self.mask_embedding.device
        dtype = self.context_proj.weight.dtype
        batch = hidden_taps.shape[0]
        context = self.context_norm(
            self.context_proj(self._flatten_taps(hidden_taps.to(device=device, dtype=dtype)))
        )
        prefix_embed = self.token_embedding(prefix_windows.to(device))
        total_len = 1 + window + block
        sequence = torch.zeros(batch, total_len, self.config.d_model, device=device, dtype=dtype)
        sequence[:, 0, :] = context
        sequence[:, 1 : 1 + window, :] = prefix_embed
        rows = torch.arange(batch, device=device).unsqueeze(1)
        block_cols = 1 + prefix_lengths.to(device).unsqueeze(1) + torch.arange(
            block, device=device
        ).unsqueeze(0)
        block_embed = self.mask_embedding.view(1, 1, -1).expand(batch, block, -1).clone()
        sequence[rows, block_cols] = block_embed
        causal_mask = torch.triu(
            torch.ones(total_len, total_len, dtype=torch.bool, device=device), diagonal=1
        )
        for layer in self.layers:
            sequence = layer(sequence, causal_mask=causal_mask)
        return sequence[rows, block_cols]

    def forward_training_blocks(
        self,
        *,
        hidden_taps: torch.Tensor,
        prefix_windows: torch.Tensor,
        prefix_lengths: torch.Tensor,
        target_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced, batched: `target_blocks` is `(B, block_size)`
        (slot 0 is the anchor, matched by construction -- never predicted).
        Returns `(logits, confidence_logits)`: `logits` is `(B, block_size
        - 1, vocab_size)` for slots `1..block_size-1`; `confidence_logits`
        is `(B, block_size - 1)` (apply `sigmoid` for the probability, this
        module returns the pre-sigmoid logit so callers can use
        `binary_cross_entropy_with_logits`)."""
        block = self.config.block_size
        if target_blocks.shape[1] != block:
            raise ValueError(
                f"target_blocks must have config.block_size={block} columns, "
                f"got {target_blocks.shape[1]}"
            )
        states = self._block_states_batched(
            hidden_taps=hidden_taps, prefix_windows=prefix_windows, prefix_lengths=prefix_lengths
        )
        h = states[:, 1:, :]  # (B, block-1, d_model): slots 1..block-1
        prev_tokens = target_blocks[:, : block - 1].to(h.device)  # x_0..x_{block-2}
        low_rank = self.markov_embed(prev_tokens)  # (B, block-1, rank)
        logits = self.output_head(h) + self.markov_out(low_rank)
        confidence_logits = self.confidence_head(torch.cat([h, low_rank], dim=-1)).squeeze(-1)
        return logits, confidence_logits

    def draft_blocks(
        self,
        *,
        hidden_taps: torch.Tensor,
        prefix_windows: torch.Tensor,
        prefix_lengths: torch.Tensor,
        first_tokens: torch.Tensor,
        shortlist: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched sequential Markov-head draft: `first_tokens` `(B,)` is
        the ANCHOR (slot 0, e.g. the primary's own next token -- always
        "accepted" since it is not actually drafted). Returns `(tokens,
        confidences)`, both `(B, block_size)`: `tokens[:, 0] ==
        first_tokens`, `confidences[:, 0] == 1.0` (certain), and
        `confidences[:, 1:]` are `sigmoid(confidence_head(...))` per
        drafted slot.

        The base logits `output_head(h_k)` do not depend on any drafted
        token, so every slot is read out in ONE call before the walk
        (2026-09-28: one full vocabulary readout per slot was ~2 ms each,
        most of the ~14.8 ms eval draft round); only the rank-`dspark_v2_
        markov_rank` correction runs per slot.

        `shortlist > 0` (opt-in, APPROXIMATE): slot `k`'s Markov-corrected
        logits are scored only over its own top-`shortlist` base ids, so the
        per-slot `markov_out` product touches `shortlist` rows instead of
        the whole vocabulary. A Markov correction that would lift a token
        from outside that shortlist to the top is missed; the primary's
        verify still decides what is committed. `0` (default) is exact."""
        block = self.config.block_size
        with torch.no_grad():
            states = self._block_states_batched(
                hidden_taps=hidden_taps,
                prefix_windows=prefix_windows,
                prefix_lengths=prefix_lengths,
            )
            h = states[:, 1:, :]
            batch = h.shape[0]
            device = h.device
            base = self.output_head(h)  # (B, block - 1, vocab): the one full readout
            rows = None
            if shortlist > 0:
                # Sorted ids, so the argmax breaks ties toward the lower id
                # exactly like the full argmax.
                ids = base.topk(min(shortlist, base.shape[-1]), dim=-1).indices.sort(
                    dim=-1
                ).values
                rows = (ids, base.gather(-1, ids), self.markov_out.weight[ids])
            prev = first_tokens.to(device)
            tokens = [prev]
            confidences = [torch.ones(batch, device=device)]
            for k in range(block - 1):
                low_rank = self.markov_embed(prev)
                if rows is None:
                    logits = base[:, k, :] + self.markov_out(low_rank)
                    token = logits.argmax(dim=-1)
                else:
                    ids, base_rows, markov_rows = rows
                    short = base_rows[:, k, :] + torch.einsum(
                        "br,bkr->bk", low_rank, markov_rows[:, k]
                    ).to(base_rows.dtype)
                    token = ids[:, k, :].gather(-1, short.argmax(dim=-1, keepdim=True))[:, 0]
                confidence = torch.sigmoid(
                    self.confidence_head(torch.cat([h[:, k, :], low_rank], dim=-1)).squeeze(-1)
                )
                tokens.append(token)
                confidences.append(confidence)
                prev = token
            return torch.stack(tokens, dim=1), torch.stack(confidences, dim=1)

    def draft_block(
        self,
        *,
        hidden_taps: tuple[torch.Tensor, ...],
        prefix_tokens: tuple[int, ...],
        first_token: int,
        shortlist: int = 0,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Single-example convenience wrapper over :meth:`draft_blocks`
        (batch size 1), for :class:`DSparkV2SpeculativeDrafter`."""
        window = self.config.dspark_prefix_window
        if window is None:
            raise ValueError("DSparkV2Drafter requires config.dspark_prefix_window")
        device = self.mask_embedding.device
        taps = torch.stack([tap.reshape(-1).to(device=device) for tap in hidden_taps]).unsqueeze(0)
        windowed = prefix_tokens[-window:] if window is not None else prefix_tokens
        window_ids = torch.zeros(1, window, dtype=torch.long, device=device)
        if windowed:
            window_ids[0, : len(windowed)] = torch.tensor(windowed, dtype=torch.long, device=device)
        lengths = torch.tensor([len(windowed)], dtype=torch.long, device=device)
        first = torch.tensor([first_token], dtype=torch.long, device=device)
        tokens, confidences = self.draft_blocks(
            hidden_taps=taps, prefix_windows=window_ids, prefix_lengths=lengths, first_tokens=first,
            shortlist=shortlist,
        )
        return (
            tuple(int(t) for t in tokens[0].tolist()),
            tuple(float(c) for c in confidences[0].tolist()),
        )

    def _walk_markov_head_single(
        self, states: torch.Tensor, first_token: int
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Single-example (`states` is `(1, block_size, d_model)`) sequential
        Markov-head walk -- the same math as :meth:`draft_blocks`' inner loop,
        kept as its OWN small copy (not a refactor of that already-tested
        batched path) for :meth:`draft_block_context_last`/
        :meth:`draft_block_incremental` below."""
        h = states[:, 1:, :]
        device = h.device
        prev = torch.tensor([first_token], device=device)
        tokens = [prev]
        confidences = [torch.ones(1, device=device)]
        for k in range(self.config.block_size - 1):
            low_rank = self.markov_embed(prev)
            logits = self.output_head(h[:, k, :]) + self.markov_out(low_rank)
            token = logits.argmax(dim=-1)
            confidence = torch.sigmoid(
                self.confidence_head(torch.cat([h[:, k, :], low_rank], dim=-1)).squeeze(-1)
            )
            tokens.append(token)
            confidences.append(confidence)
            prev = token
        return (
            tuple(int(t.item()) for t in tokens),
            tuple(float(c.item()) for c in confidences),
        )

    def _block_states_context_last_full(
        self, *, hidden_taps: tuple[torch.Tensor, ...], prefix_tokens: tuple[int, ...]
    ) -> torch.Tensor:
        """Full (non-cached) recompute of the reordered `[prefix, context,
        block]` sequence used when `config.dspark_context_last=True` --
        single example (batch=1), prefix windowed to `config.
        dspark_prefix_window` with NO padding (unlike :meth:`_block_states_
        batched`, which pads to a fixed width for a whole batch of
        differently-lengthed examples -- irrelevant here, online decode is
        always one example at a time). Moving context from FIRST (the
        default, paper-faithful layout) to right after the (fixed) prefix
        window is NOT a pure reordering (corrected 2026-09-28): in the
        context-first layout every prefix position attends to the context
        row, here none does, so the block states differ -- a model trained
        through `_block_states_batched` (context first, what
        `forward_training_blocks` and `draft_blocks` use whatever
        `config.dspark_context_last` says) is a different function under
        this layout (`test_dspark_v2_draft_walk.py`; on the step-2000
        rehearsal checkpoint 27 of 30 steady-state drafts changed). This
        reordering is what makes :class:`DSparkV2IncrementalCache` possible
        (see its docstring) -- the default `dspark_context_last=False`
        layout is architecturally incompatible with any per-round KV cache,
        because every prefix position there attends back to context
        (position 0), which changes every round.

        Ground truth for :class:`DSparkV2IncrementalCache` correctness
        tests -- an INDEPENDENT construction, not sharing code with the
        cache path, exactly like `dspark.py`'s `_block_state` vs
        `_block_state_incremental`. Returns `(1, block_size, d_model)`,
        the block positions' own final hidden states -- same shape/meaning
        as `_block_states_batched`'s return value."""
        window = self.config.dspark_prefix_window
        if window is None:
            raise ValueError("dspark_context_last=True requires config.dspark_prefix_window")
        device = self.mask_embedding.device
        dtype = self.context_proj.weight.dtype
        windowed = prefix_tokens[-window:]
        if windowed:
            prefix_embed = self.token_embedding(
                torch.tensor(windowed, dtype=torch.long, device=device)
            ).unsqueeze(0)
        else:
            prefix_embed = torch.zeros(1, 0, self.config.d_model, device=device, dtype=dtype)
        stacked = torch.stack(
            [tap.reshape(-1).to(device=device, dtype=dtype) for tap in hidden_taps]
        ).unsqueeze(0)
        context = self.context_norm(self.context_proj(self._flatten_taps(stacked))).unsqueeze(1)
        block = self.config.block_size
        block_embed = self.mask_embedding.view(1, 1, -1).expand(1, block, -1).clone()
        sequence = torch.cat([prefix_embed, context, block_embed], dim=1)
        steps = sequence.shape[1]
        causal_mask = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=device), diagonal=1
        )
        for layer in self.layers:
            sequence = layer(sequence, causal_mask=causal_mask)
        return sequence[:, -block:, :]

    def draft_block_context_last(
        self,
        *,
        hidden_taps: tuple[torch.Tensor, ...],
        prefix_tokens: tuple[int, ...],
        first_token: int,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """`draft_block`'s result under `config.dspark_context_last=True`
        (full, non-cached recompute -- see :meth:`_block_states_context_
        last_full`). `config.dspark_context_last=False` (default) keeps
        using `draft_block` unchanged; this is an additive sibling, not a
        replacement."""
        if not self.config.dspark_context_last:
            raise ValueError("draft_block_context_last requires config.dspark_context_last=True")
        with torch.no_grad():
            states = self._block_states_context_last_full(
                hidden_taps=hidden_taps, prefix_tokens=prefix_tokens
            )
            return self._walk_markov_head_single(states, first_token)

    def draft_block_incremental(
        self,
        cache: DSparkV2IncrementalCache,
        *,
        hidden_taps: tuple[torch.Tensor, ...],
        committed_tokens: tuple[int, ...],
        first_token: int,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """`draft_block_context_last`'s result, using `cache` (see
        :class:`DSparkV2IncrementalCache`) instead of re-attending the
        whole prefix window every round. `committed_tokens` is the FULL
        committed prefix (windowed internally, like `draft_block_context_
        last`'s own `prefix_tokens`)."""
        with torch.no_grad():
            states = cache.block_states(
                self, hidden_taps=hidden_taps, committed_tokens=committed_tokens
            )
            return self._walk_markov_head_single(states, first_token)


def _split_heads(t: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    return t.view(1, t.shape[1], heads, head_dim).transpose(1, 2)


def _merge_heads(t: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    return t.transpose(1, 2).reshape(1, t.shape[2], heads * head_dim)


def _v2_layer_attend_new_over_full(
    layer: _DSparkV2Layer,
    x_new: torch.Tensor,
    k_old: torch.Tensor | None,
    v_old: torch.Tensor | None,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One `_DSparkV2Layer` (GQA attention + SwiGLU FFN, its own `forward`'s
    exact math) applied to `x_new` (the NEW positions, `(1, n_new, d_model)`)
    attending causally over `(k_old, v_old)` (ALL visible, unmasked -- they
    are all earlier) plus `x_new`'s own freshly-projected K/V (causal among
    themselves). Returns `(layer_output_new, k_new, v_new)`: `k_new`/`v_new`
    are PRE-repeat-interleave (`n_kv_heads` heads), for the caller to cache
    and reuse forever (§21.1's `dflash.py` `DSparkIncrementalCache` already
    established this pattern; the only difference here is an extra SwiGLU
    sublayer per layer)."""
    n_old = 0 if k_old is None else k_old.shape[2]
    n_new = x_new.shape[1]
    device = x_new.device
    normed = layer.attn_norm(x_new)
    q = _split_heads(layer.attn.q_proj(normed), n_heads, head_dim)
    k_new = _split_heads(layer.attn.k_proj(normed), n_kv_heads, head_dim)
    v_new = _split_heads(layer.attn.v_proj(normed), n_kv_heads, head_dim)
    k_full = k_new if k_old is None else torch.cat([k_old, k_new], dim=2)
    v_full = v_new if v_old is None else torch.cat([v_old, v_new], dim=2)
    groups = n_heads // n_kv_heads
    k_rep = k_full.repeat_interleave(groups, dim=1)
    v_rep = v_full.repeat_interleave(groups, dim=1)
    total = n_old + n_new
    mask = torch.zeros(n_new, total, dtype=torch.bool, device=device)
    mask[:, n_old:] = torch.triu(
        torch.ones(n_new, n_new, dtype=torch.bool, device=device), diagonal=1
    )
    attn_bias = torch.zeros(n_new, total, dtype=q.dtype, device=device)
    attn_bias.masked_fill_(mask, float("-inf"))
    attn_out = nn.functional.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_bias)
    attn_out = layer.attn.o_proj(_merge_heads(attn_out, n_heads, head_dim))
    x_after_attn = x_new + attn_out
    ffn_out = layer.ffn(layer.ffn_norm(x_after_attn))
    x_final = x_after_attn + ffn_out
    return x_final, k_new, v_new


class DSparkV2IncrementalCache:
    """Per-round incremental key/value cache for `DSparkV2Drafter`'s 5-layer
    transformer, valid only when `config.dspark_context_last=True` (see
    `_block_states_context_last_full`'s docstring for why the DEFAULT
    context-first layout cannot support any cross-round cache at all: every
    prefix position there attends back to context, which changes every
    round, so NOTHING about the prefix is stable across rounds in that
    layout).

    Under `dspark_context_last=True` -- `[prefix, context, block]` -- the
    prefix window's per-layer key/value projections never depend on context
    or the block (both come after it, causally invisible to it), so they
    are computed ONCE per prefix token and reused for every later round
    until that token slides out of the window. `context` and `block`
    themselves are NEVER cached (context changes every round with the
    target's latest hidden taps; block is always freshly re-initialized
    mask embeddings) -- they are treated as a SINGLE `1 + block_size`
    "new chunk" each round (see :meth:`block_states`), attending over the
    cached prefix.

    Same correctness discipline as `dflash.py`'s `DSparkIncrementalCache`:
    pure prefix APPEND reuses the cache; any front eviction (a token sliding
    out of the window changes what's causally visible to the tokens that
    used to sit after it) forces a safe full cold rebuild rather than an
    incorrect reuse.
    """

    def __init__(self, config: DflashConfig) -> None:
        if config.variant != "dspark_v2":
            raise ValueError(
                f"DSparkV2IncrementalCache requires variant='dspark_v2', got {config.variant!r}"
            )
        if not config.dspark_context_last:
            raise ValueError(
                "DSparkV2IncrementalCache requires config.dspark_context_last=True "
                "(only that layout keeps the prefix window's per-layer state "
                "independent of the ever-changing context row and block)"
            )
        self._window = config.dspark_prefix_window
        n_layers = config.dspark_v2_n_layers
        #: `k[layer_index]`/`v[layer_index]`, `layer_index` in `0..n_layers-1`:
        #: `(1, n_kv_heads, n_window, head_dim)` projected keys/values of the
        #: prefix window's INPUT to that layer, or `None` before the first
        #: `update()`.
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers
        self._tokens: tuple[int, ...] = ()

    def reset(self) -> None:
        self.k = [None] * len(self.k)
        self.v = [None] * len(self.v)
        self._tokens = ()

    def update(self, drafter: DSparkV2Drafter, committed_tokens: tuple[int, ...]) -> None:
        """Bring the cache up to date with `committed_tokens` (the FULL
        committed prefix so far, windowed internally exactly like
        `_block_states_context_last_full` windows `prefix_tokens`).
        Idempotent: calling it again with the same `committed_tokens` is a
        no-op."""
        window = self._window
        windowed = committed_tokens[-window:] if window is not None else committed_tokens
        if windowed == self._tokens:
            return
        # Only a PURE APPEND is safe to reuse -- see the class docstring.
        if self._tokens and windowed[: len(self._tokens)] != self._tokens:
            self.reset()
        new_tokens = windowed[len(self._tokens):]
        if not new_tokens:
            self._tokens = windowed
            return
        config = drafter.config
        n_heads, n_kv_heads, head_dim = (
            config.dspark_v2_n_heads, config.dspark_v2_n_kv_heads, config.dspark_v2_head_dim
        )
        device = drafter.mask_embedding.device
        hidden = drafter.token_embedding(
            torch.tensor(new_tokens, dtype=torch.long, device=device)
        ).unsqueeze(0)
        for layer_index in range(len(self.k)):
            layer = drafter.layers[layer_index]
            hidden, k_new, v_new = _v2_layer_attend_new_over_full(
                layer, hidden, self.k[layer_index], self.v[layer_index],
                n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
            )
            self.k[layer_index] = k_new if self.k[layer_index] is None else torch.cat(
                [self.k[layer_index], k_new], dim=2
            )
            self.v[layer_index] = v_new if self.v[layer_index] is None else torch.cat(
                [self.v[layer_index], v_new], dim=2
            )
        self._tokens = windowed

    def block_states(
        self,
        drafter: DSparkV2Drafter,
        *,
        hidden_taps: tuple[torch.Tensor, ...],
        committed_tokens: tuple[int, ...],
    ) -> torch.Tensor:
        """`_block_states_context_last_full`'s result, computed via this
        cache: updates the prefix cache, then runs the `[context, block]`
        "new chunk" (`1 + block_size` positions, always fresh -- see class
        docstring) through all layers attending over the cached prefix.
        Returns `(1, block_size, d_model)`."""
        self.update(drafter, committed_tokens)
        config = drafter.config
        device = drafter.mask_embedding.device
        dtype = drafter.context_proj.weight.dtype
        stacked = torch.stack(
            [tap.reshape(-1).to(device=device, dtype=dtype) for tap in hidden_taps]
        ).unsqueeze(0)
        context = drafter.context_norm(
            drafter.context_proj(drafter._flatten_taps(stacked))
        ).unsqueeze(1)
        block = config.block_size
        block_embed = drafter.mask_embedding.view(1, 1, -1).expand(1, block, -1).clone()
        new_chunk = torch.cat([context, block_embed], dim=1)
        n_heads, n_kv_heads, head_dim = (
            config.dspark_v2_n_heads, config.dspark_v2_n_kv_heads, config.dspark_v2_head_dim
        )
        for layer_index in range(len(self.k)):
            layer = drafter.layers[layer_index]
            new_chunk, _k, _v = _v2_layer_attend_new_over_full(
                layer, new_chunk, self.k[layer_index], self.v[layer_index],
                n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
            )
        return new_chunk[:, 1:, :]


def survival_labels(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """`(B, block-1, vocab)` teacher-forced logits + `(B, block-1)` targets
    -> `(B, block-1)` float 0/1 labels: slot `k` (0-indexed here, i.e. the
    paper's slot `k+1`) "survives" iff its own argmax AND every earlier
    slot's argmax in this row already matched the target -- exactly what a
    real block verify would accept. Used as the confidence head's BCE
    target."""
    matches = (logits.argmax(dim=-1) == targets).float()
    return matches.cumprod(dim=1)


def position_weights(block_minus_one: int, gamma: int, *, device: torch.device) -> torch.Tensor:
    """`w_k = exp(-(k - 1) / gamma)` for slots `k = 1..block_minus_one`
    (1-indexed in the paper), returned in the same `0..block_minus_one-1`
    order :meth:`DSparkV2Drafter.forward_training_blocks` uses."""
    k = torch.arange(1, block_minus_one + 1, dtype=torch.float32, device=device)
    return torch.exp(-(k - 1) / gamma)


class DSparkV2SpeculativeDrafter:
    """`drafter.SpeculativeDrafter`-compatible adapter over a
    :class:`DSparkV2Drafter`. Provides both `draft` (protocol-compatible,
    ignores confidence) and `draft_with_confidence` (read by
    `drafter.run_speculative_decode`'s optional confidence-scheduled
    truncation, mirroring `DSparkSpeculativeDrafter`'s fail-closed
    contract)."""

    def __init__(
        self,
        *,
        model: DSparkV2Drafter,
        hidden_state_provider: Callable[[tuple[int, ...]], tuple[torch.Tensor, ...]],
        anchor_provider: Callable[[tuple[int, ...]], int | None],
        incremental_cache: DSparkV2IncrementalCache | None = None,
    ) -> None:
        """`incremental_cache` (2026-09-28, opt-in): when given, `draft_
        with_confidence` calls `model.draft_block_incremental` against it
        instead of `model.draft_block` -- requires `model.config.
        dspark_context_last=True` (see `DSparkV2IncrementalCache`'s
        docstring). `None` (default, every pre-existing caller) is the
        unchanged `draft_block` path -- no behavior change."""
        self._model = model
        self._hidden_state_provider = hidden_state_provider
        self._anchor_provider = anchor_provider
        self._incremental_cache = incremental_cache
        #: See `DflashSpeculativeDrafter.last_error`'s identical docstring.
        self.last_error: BaseException | None = None

    def draft_with_confidence(
        self, snapshot: DecoderSnapshot
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        try:
            anchor = self._anchor_provider(snapshot.committed_tokens)
            if anchor is None:
                return (), ()
            hidden_taps = self._hidden_state_provider(snapshot.committed_tokens)
            if self._incremental_cache is not None:
                return self._model.draft_block_incremental(
                    self._incremental_cache,
                    hidden_taps=hidden_taps,
                    committed_tokens=snapshot.committed_tokens,
                    first_token=anchor,
                )
            return self._model.draft_block(
                hidden_taps=hidden_taps,
                prefix_tokens=snapshot.committed_tokens,
                first_token=anchor,
            )
        except Exception as exc:  # noqa: BLE001 -- fail-closed, see DflashSpeculativeDrafter.draft
            self.last_error = exc
            logger.exception("dspark_v2 drafter draft failed; fail-closed to primary decode")
            return (), ()

    def draft(self, snapshot: DecoderSnapshot) -> tuple[int, ...]:
        tokens, _confidences = self.draft_with_confidence(snapshot)
        return tokens

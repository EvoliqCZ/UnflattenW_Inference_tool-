"""RFC004 §37.2, §37.4-5: `dflash_speculator`, a block-diffusion drafter.

**Scope.** This is the same token/grammar-level harness scope as
`drafter.py`'s module docstring: a real ``nn.Module`` drafter proven against
:mod:`drafter`'s `SpeculativeDrafter` protocol and (via
``TestModelBackedDflash``) against a real ``QwenDenseCandidateModel`` KV
cache, but never wired into the production decode path
(`free_running_decode.py`). Installing this module never activates it
(§37.5) -- nothing here is imported by any decode entry point.

**Design (§37.2 pinned surface).**

- *Conditioning inputs*: the target's pinned hidden-state tap(s)
  (:data:`DflashConfig.hidden_taps`, today only ``"final"`` -- the one
  per-step hidden state `candidate_model.encode_step` already exposes,
  matching what `drafter.py`'s own model-backed harness reads) plus the
  embedded committed prefix tokens.
- *Denoising loop*: :meth:`DflashDrafter.draft_block` runs
  :data:`DflashConfig.num_steps` forward passes over
  ``[hidden; prefix; block]``, where ``block`` starts as
  ``num_steps`` copies of a learned mask embedding and is re-embedded from
  each step's own argmax prediction for the next step -- a block-diffusion
  refinement, not autoregression (all ``block_size`` positions are proposed
  and refined together every step). The pinned anchor policy
  ``"all_positions"`` is the only one implemented: every position is
  re-predicted at every step, with the final step's argmax taken as the
  committed proposal (no partial fix-and-carry ordering).
- *Determinism*: ``TransformerEncoderLayer`` is built with ``dropout=0.0``,
  so a fixed seed and fixed weights give a bit-identical draft regardless of
  ``self.training``; :meth:`draft_block` still runs under
  ``torch.no_grad()`` because a draft must never leave a gradient graph
  behind (no side effect on any decoder state, §37.1).
- *Side effects*: none. ``draft_block`` reads its module's own parameters
  and its two inputs; it does not touch any KV cache, grammar stack, or
  `DecoderSnapshot`.

**Fail-closed (§37.2 "pri runtime chybe sa musí fail-closed prepnúť na
primary decode").** :class:`DflashSpeculativeDrafter` is the
`drafter.SpeculativeDrafter` adapter used with `drafter.run_speculative_decode`
(unmodified). Its `draft` catches any exception raised while producing a
block and returns ``()`` instead of propagating -- `run_speculative_decode`
already treats an empty proposal as "nothing verified, primary decode
continues alone" (its own module docstring), so this is a fail-closed
degrade to primary-only decode with no new control path in `drafter.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn

from unflatten.model.hashing import domain_hash
from unflatten.model.speculative.drafter import DecoderSnapshot

logger = logging.getLogger(__name__)

_COMPATIBILITY_DOMAIN: Final = "unflatten.speculative.dflash-compatibility/v1"

#: §37.2: "block size, anchors a acceptance algorithm musia byť pinned" --
#: the only anchor policy this module implements.
_ANCHOR_POLICY = "all_positions"

#: The pre-existing block-diffusion drafter (`DflashDrafter`, unchanged) is
#: `"block_diffusion"`. `"dspark"` selects `DSparkDrafter` below (LiquidAI
#: DSpark recipe, 2026-09-27) -- a variant field rather than a replacement,
#: since `"block_diffusion"` is a pinned surface (§37.2) other code and tests
#: already depend on. `"dspark_v2"` (2026-09-28) selects `DSparkV2Drafter`
#: (`dspark_v2.py`) -- the paper/model-card-faithful architecture (5 full
#: transformer layers, GQA + SwiGLU, low-rank Markov head, confidence head),
#: a further variant rather than a change to `"dspark"`'s pinned surface.
_VARIANTS: Final = ("block_diffusion", "dspark", "dspark_v2")


@dataclass(frozen=True, slots=True)
class DflashConfig:
    """§37.2's pinned surface: block size, step count, hidden-state taps,
    anchor policy and the architecture dims the drafter's weights assume.
    Every field here (plus target/tokenizer identity) feeds
    :func:`drafter_target_compatibility_hash`.

    `variant` (2026-09-27): `"block_diffusion"` (default) is exactly
    today's `DflashDrafter` -- every field below it (`dspark_hidden_taps`,
    `markov_head`) is inert for that variant. `"dspark"` instead builds a
    :class:`DSparkDrafter`, conditioned on `dspark_hidden_taps` (a FIXED SET
    of frozen-decoder layer indices, not only the final layer -- see
    `candidate_model.QwenDenseCandidateModel.encode_step`'s `hidden_taps`
    parameter) rather than the single string-tagged `hidden_taps` tap the
    block-diffusion variant reads."""

    hidden_dim: int
    d_model: int
    vocab_size: int
    block_size: int = 4
    num_steps: int = 2
    hidden_taps: tuple[str, ...] = ("final",)
    anchor_policy: str = _ANCHOR_POLICY
    seed: int = 0
    variant: str = "block_diffusion"
    #: DSpark-only: frozen-decoder layer indices to tap (e.g. `(4, 12, 20)`
    #: for a 24-ish-layer decoder, spread over depth -- a policy field, not
    #: a code literal; see `s7_dflash_policy.json`'s `dspark` block).
    dspark_hidden_taps: tuple[int, ...] = ()
    #: DSpark-only: whether position `i`'s logits condition on position
    #: `i - 1`'s drafted token (intra-block dependency -> higher acceptance,
    #: DSpark's own stated motivation). `False` drops that conditioning,
    #: kept only as the ablation arm `test_dspark.py` mutates against.
    markov_head: bool = False
    #: DSpark-only: how many of the most recent committed tokens the
    #: 4-layer decoder attends over (`None`, the default, keeps the whole
    #: prefix -- unchanged from before this field existed). The rehearsal
    #: on real pages (2026-09-27) found the whole-prefix form re-encodes
    #: thousands of tokens on EVERY draft; the target's tapped hidden
    #: states already carry the long-range context, so the drafter only
    #: needs a short local window. Hashed only when set, so an unset
    #: window leaves every pre-existing compatibility hash unchanged.
    dspark_prefix_window: int | None = None
    #: DSpark-only: LayerNorm each tapped layer before the shared
    #: projection. The rehearsal (2026-09-27) measured the frozen layers'
    #: mean norms at ~38 / 43 / 135 / 550 (layers 9 / 18 / 27 / 35), so a
    #: raw concatenation is dominated by the last tap. `False` (default)
    #: is the pre-existing drafter; hashed only when set.
    dspark_tap_norm: bool = False
    #: DSpark-only: append the target's dense-adapted FINAL hidden state
    #: (what its composite head reads) as one more tap, after the frozen
    #: layers. The rehearsal (2026-09-27) found the frozen layers alone
    #: leave the first drafted token wrong on ~92 % of held-out blocks.
    #: `False` (default) is the pre-existing drafter; hashed only when set.
    dspark_final_tap: bool = False
    #: DSpark-only: put the tapped context AFTER the prefix and read the
    #: block state at the context position (its residual stream then
    #: carries the target state), instead of context first and reading at
    #: the last prefix token. `False` (default) is the pre-existing layout;
    #: hashed only when set.
    dspark_context_last: bool = False
    #: DSpark-only: the drafted block starts with the PRIMARY's own next
    #: token (known exactly from its cached hidden state, no extra forward)
    #: and the Markov head continues from it, instead of the drafter
    #: guessing position 0 from its start embedding. Requires
    #: `markov_head`. `False` (default) is the pre-existing drafter;
    #: hashed only when set.
    dspark_seed_first_token: bool = False
    #: dspark_v2-only (LiquidAI DSpark paper/model-card architecture,
    #: 2026-09-28): full transformer layer count. Paper: 5. `block_size`
    #: doubles as the paper's block width `gamma` for this variant (anchor
    #: slot + `block_size - 1` drafted slots) -- kept as one field rather
    #: than a second one, since both mean "how wide is one drafted block".
    dspark_v2_n_layers: int = 5
    #: dspark_v2-only: query heads (GQA). Paper: 32 at hidden 2048; we scale
    #: to `hidden_dim` via `head_dim` below.
    dspark_v2_n_heads: int = 40
    #: dspark_v2-only: key/value heads (GQA, paper ratio 4:1).
    dspark_v2_n_kv_heads: int = 10
    #: dspark_v2-only: per-head width. `n_heads * head_dim` must equal
    #: `d_model` (paper: 64, unchanged here).
    dspark_v2_head_dim: int = 64
    #: dspark_v2-only: SwiGLU FFN hidden width, dense (no MoE -- the paper's
    #: backbone uses an MoE FFN in its own setting; ours is dense per the
    #: owner's instruction). Sized so the drafter totals ~250-330M trainable
    #: parameters at `hidden_dim=2560` (see `dspark_v2.py`'s module
    #: docstring for the count) -- not copied from the paper's own 6144,
    #: since their low-rank Markov head is far smaller relative to our
    #: `vocab_size`.
    dspark_v2_ffn_intermediate: int = 2688
    #: dspark_v2-only: rank of the low-rank Markov bigram
    #: `B(x_{k-1}, .) = W1[x_{k-1}] @ W2`. Paper: 256.
    dspark_v2_markov_rank: int = 256

    @property
    def n_tap_inputs(self) -> int:
        """How many tapped states the DSpark drafter reads per position."""
        return len(self.dspark_hidden_taps) + int(self.dspark_final_tap)

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.anchor_policy != _ANCHOR_POLICY:
            raise ValueError(
                f"anchor_policy must be {_ANCHOR_POLICY!r} (the only pinned, "
                f"implemented policy), got {self.anchor_policy!r}"
            )
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}, got {self.variant!r}")
        if self.variant == "dspark" and not self.dspark_hidden_taps:
            raise ValueError("variant='dspark' requires a non-empty dspark_hidden_taps")
        if self.dspark_seed_first_token and not self.markov_head:
            raise ValueError("dspark_seed_first_token requires markov_head=True")
        if self.dspark_prefix_window is not None and self.dspark_prefix_window < 1:
            raise ValueError(
                f"dspark_prefix_window must be >= 1 or None, got {self.dspark_prefix_window}"
            )
        if self.variant == "dspark_v2":
            if not self.markov_head:
                raise ValueError("variant='dspark_v2' requires markov_head=True")
            if not self.dspark_seed_first_token:
                raise ValueError("variant='dspark_v2' requires dspark_seed_first_token=True")
            if self.block_size < 2:
                raise ValueError(
                    "variant='dspark_v2' requires block_size >= 2 (anchor + >=1 drafted slot), "
                    f"got {self.block_size}"
                )
            if self.dspark_v2_n_heads * self.dspark_v2_head_dim != self.d_model:
                raise ValueError(
                    "variant='dspark_v2' requires dspark_v2_n_heads * dspark_v2_head_dim == "
                    f"d_model, got {self.dspark_v2_n_heads} * {self.dspark_v2_head_dim} != "
                    f"{self.d_model}"
                )
            if self.dspark_v2_n_heads % self.dspark_v2_n_kv_heads != 0:
                raise ValueError(
                    "variant='dspark_v2' requires dspark_v2_n_heads to be a multiple of "
                    f"dspark_v2_n_kv_heads, got {self.dspark_v2_n_heads} / "
                    f"{self.dspark_v2_n_kv_heads}"
                )


class DflashDrafter(nn.Module):
    """A tiny block-diffusion drafter over pinned hidden-state taps + prefix tokens.

    Not autoregressive: :meth:`draft_block` proposes and iteratively
    refines all `config.block_size` positions together over
    `config.num_steps` forward passes.
    """

    def __init__(
        self,
        config: DflashConfig,
        *,
        frozen_token_embedding_weight: torch.Tensor | None = None,
        frozen_output_head_weight: torch.Tensor | None = None,
        frozen_output_head_bias: torch.Tensor | None = None,
    ) -> None:
        """`frozen_token_embedding_weight`/`frozen_output_head_*` (owner note,
        offline S7 trainer): the target's own frozen embedding and output-head
        tables, loaded alone -- never the full target model -- so the drafter
        ties into the target's vocabulary instead of learning its own copy.
        `None` (default, all existing callers) keeps the drafter's own
        trainable embedding/head, unchanged from before this option existed.
        """
        super().__init__()
        self.config = config
        self.hidden_proj = nn.Linear(config.hidden_dim, config.d_model)
        if frozen_token_embedding_weight is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                frozen_token_embedding_weight, freeze=True
            )
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_embedding = nn.Parameter(torch.zeros(config.d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=1,
            dim_feedforward=config.d_model * 2,
            dropout=0.0,
            batch_first=True,
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

    def _forward_logits(
        self, *, hidden_tap: torch.Tensor, prefix_tokens: tuple[int, ...], block_embed: torch.Tensor
    ) -> torch.Tensor:
        device = self.mask_embedding.device
        hidden_embed = self.hidden_proj(hidden_tap.reshape(1, 1, -1).to(device))
        if prefix_tokens:
            prefix_ids = torch.tensor(prefix_tokens, dtype=torch.long, device=device).unsqueeze(0)
            prefix_embed = self.token_embedding(prefix_ids)
        else:
            prefix_embed = hidden_embed.new_zeros((1, 0, self.config.d_model))
        sequence = torch.cat([hidden_embed, prefix_embed, block_embed], dim=1)
        encoded = self.encoder_layer(sequence)
        block_encoded = encoded[:, -self.config.block_size :, :]
        return self.output_head(block_encoded)

    def draft_block(
        self, *, hidden_tap: torch.Tensor, prefix_tokens: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Iteratively denoise `config.block_size` masked positions over
        `config.num_steps` steps; returns the final step's argmax tokens.
        No side effect on any decoder state -- reads only this module's own
        parameters and the two arguments."""
        block_embed = self.mask_embedding.view(1, 1, -1).expand(
            1, self.config.block_size, -1
        )
        with torch.no_grad():
            for _ in range(self.config.num_steps):
                logits = self._forward_logits(
                    hidden_tap=hidden_tap, prefix_tokens=prefix_tokens, block_embed=block_embed
                )
                predicted = logits.argmax(dim=-1)
                block_embed = self.token_embedding(predicted)
            return tuple(int(token) for token in predicted[0].tolist())

    def forward_training_step(
        self, *, hidden_tap: torch.Tensor, prefix_tokens: tuple[int, ...]
    ) -> torch.Tensor:
        """Single-pass, gradient-carrying block logits from the initial mask
        embedding -- the offline trainer's (:mod:`s7_dflash_train`) training
        surrogate for the `num_steps`-round denoise :meth:`draft_block` runs
        at inference. A full multi-step training curriculum is future work;
        this scaffold trains only the one-step case, matching the existing
        S7 VC-1/VC-2 "stub step" level of implementation. Returns
        `(1, block_size, vocab_size)` logits."""
        block_embed = self.mask_embedding.view(1, 1, -1).expand(1, self.config.block_size, -1)
        return self._forward_logits(
            hidden_tap=hidden_tap, prefix_tokens=prefix_tokens, block_embed=block_embed
        )


class _AttentionOnlyLayer(nn.Module):
    """One self-attention-plus-residual block, deliberately WITHOUT a
    feedforward sublayer -- DSpark's "4-layer attention-only decoder"
    (module docstring). `dropout=0.0` for the same determinism reason
    `DflashDrafter`'s `TransformerEncoderLayer` uses it (fixed weights + a
    fixed seed give a bit-identical result regardless of `self.training`)."""

    def __init__(self, d_model: int, nhead: int = 1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, *, attn_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _weights = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        return self.norm(x + attn_out)


class DSparkDrafter(nn.Module):
    """LiquidAI DSpark recipe (2026-09-27): conditions on the target's
    hidden states from :data:`DflashConfig.dspark_hidden_taps` -- a FIXED
    SET of frozen-decoder layers, not only the final one -- through a
    4-layer attention-only decoder (:class:`_AttentionOnlyLayer` x4) that
    summarizes the tapped layers plus the committed prefix into one "block
    state" vector. Drafting a `config.block_size` block is then SEQUENTIAL
    over the block (unlike `DflashDrafter`'s parallel denoise), but cheap:
    each of the `block_size` steps is one small "Markov head" projection,
    not a full decoder forward pass -- only the ONE-TIME block-state
    summary runs the 4-layer decoder.

    `config.markov_head=True` (DSpark's headline mechanism): position `i`'s
    logits condition on the block state AND position `i - 1`'s own drafted
    token embedding, so within-block tokens are no longer independent given
    the block state -- DSpark's stated reason this recipe reaches higher
    acceptance than a position-independent drafter. `markov_head=False` is
    kept only as the ablation arm: every position then reads the block
    state alone, with no intra-block dependency at all.
    """

    def __init__(
        self,
        config: DflashConfig,
        *,
        frozen_token_embedding_weight: torch.Tensor | None = None,
        frozen_output_head_weight: torch.Tensor | None = None,
        frozen_output_head_bias: torch.Tensor | None = None,
    ) -> None:
        """Same frozen-table option as `DflashDrafter.__init__` (see its
        docstring) -- `None` (default) keeps the drafter's own trainable
        embedding/head."""
        super().__init__()
        if config.variant != "dspark":
            raise ValueError(
                f"DSparkDrafter requires config.variant='dspark', got {config.variant!r}"
            )
        self.config = config
        n_taps = config.n_tap_inputs
        self.tap_norms = (
            nn.ModuleList(nn.LayerNorm(config.hidden_dim) for _ in range(n_taps))
            if config.dspark_tap_norm
            else None
        )
        self.hidden_proj = nn.Linear(config.hidden_dim * n_taps, config.d_model)
        if frozen_token_embedding_weight is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                frozen_token_embedding_weight, freeze=True
            )
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.start_embedding = nn.Parameter(torch.zeros(config.d_model))
        self.decoder_layers = nn.ModuleList(
            [_AttentionOnlyLayer(config.d_model) for _ in range(4)]
        )
        self.markov_proj = nn.Linear(config.d_model * 2, config.d_model)
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

    def _flatten_taps(self, taps: torch.Tensor) -> torch.Tensor:
        """`(B, n_taps, hidden_dim) -> (B, n_taps * hidden_dim)`, each tap
        LayerNormed first when `config.dspark_tap_norm`."""
        if self.tap_norms is not None:
            taps = torch.stack(
                [norm(taps[:, i]) for i, norm in enumerate(self.tap_norms)], dim=1
            )
        return taps.reshape(taps.shape[0], -1)

    def _block_state(
        self, *, hidden_taps: tuple[torch.Tensor, ...], prefix_tokens: tuple[int, ...]
    ) -> torch.Tensor:
        """The ONE-TIME 4-layer attention-only decoder pass: tapped layers
        (concatenated, projected to `d_model`) followed by the embedded
        committed prefix, causal self-attention over that whole short
        sequence, and the LAST position's output as the block's summary
        state -- `(1, d_model)`."""
        device = self.start_embedding.device
        dtype = self.hidden_proj.weight.dtype
        stacked = torch.stack(
            [tap.reshape(-1).to(device=device, dtype=dtype) for tap in hidden_taps]
        ).unsqueeze(0)  # (1, n_taps, hidden_dim)
        context_embed = self.hidden_proj(self._flatten_taps(stacked)).unsqueeze(1)
        window = self.config.dspark_prefix_window
        if window is not None:
            prefix_tokens = prefix_tokens[-window:]
        if prefix_tokens:
            prefix_ids = torch.tensor(prefix_tokens, dtype=torch.long, device=device).unsqueeze(0)
            prefix_embed = self.token_embedding(prefix_ids)
        else:
            prefix_embed = context_embed.new_zeros((1, 0, self.config.d_model))
        if self.config.dspark_context_last:
            sequence = torch.cat([prefix_embed, context_embed], dim=1)
        else:
            sequence = torch.cat([context_embed, prefix_embed], dim=1)
        steps = sequence.shape[1]
        causal_mask = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=device), diagonal=1
        )
        for layer in self.decoder_layers:
            sequence = layer(sequence, attn_mask=causal_mask)
        return sequence[:, -1, :]

    def draft_block(
        self,
        *,
        hidden_taps: tuple[torch.Tensor, ...],
        prefix_tokens: tuple[int, ...],
        first_token: int | None = None,
        shortlist: int = 0,
    ) -> tuple[int, ...]:
        """`config.block_size` tokens, one small Markov-head step at a
        time (or `config.markov_head=False`'s position-independent
        ablation), all downstream of the single `_block_state` summary.
        `first_token` (`config.dspark_seed_first_token`): the block starts
        with it and the Markov head drafts the remaining `block_size - 1`.
        `shortlist` (opt-in, approximate): see `_markov_walk`.
        No side effect on any decoder state."""
        with torch.no_grad():
            seed = self._seed_tensor(first_token)
            block_state = self._block_state(hidden_taps=hidden_taps, prefix_tokens=prefix_tokens)
            return self._markov_walk(block_state, seed, shortlist=shortlist)

    def _seed_tensor(self, first_token: int | None) -> torch.Tensor | None:
        """`first_token` as a `(1,)` device tensor, built BEFORE the block's
        device work is queued: a host-to-device copy waits for the stream,
        so building it after the block state would be a hidden sync."""
        if first_token is None:
            return None
        return torch.tensor([int(first_token)], device=self.start_embedding.device)

    def _markov_walk(
        self, block_state: torch.Tensor, seed: torch.Tensor | None, *, shortlist: int = 0
    ) -> tuple[int, ...]:
        """The sequential greedy walk shared by `draft_block` and
        `draft_block_incremental`. Every drafted token stays a `(1,)` device
        tensor and feeds the next step's embedding directly; the block is
        read back with ONE `.tolist()` at the end (2026-09-28: the old
        `int(argmax.item())` per step was a host sync per drafted token).

        `shortlist > 0` (2026-09-28, opt-in, APPROXIMATE): only the first
        computed step runs the full vocabulary readout (~152k rows, ~2 ms
        memory-bound per call on the RTX 4070); its top-`shortlist` ids are
        sliced out of the head once (`_shortlist_rows`) and every later
        step scores only those rows. A later step whose full-readout argmax
        falls outside the shortlist drafts a different token -- the
        primary's verify still decides what is committed, so only
        acceptance, never the decoded output, can change. `0` (default) is
        the exact full walk."""
        prev_embed = self.start_embedding.view(1, -1)
        drafted: list[torch.Tensor] = []
        if seed is not None:
            drafted.append(seed)
            prev_embed = self.token_embedding(seed)
        rows: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None
        for _ in range(self.config.block_size - len(drafted)):
            if self.config.markov_head:
                step_repr = self.markov_proj(torch.cat([block_state, prev_embed], dim=-1))
            else:
                step_repr = block_state
            if rows is None:
                logits = self.output_head(step_repr)
                token = logits.argmax(dim=-1)
                if shortlist > 0:
                    rows = _shortlist_rows(self.output_head, logits[0], shortlist)
            else:
                ids, weight, bias = rows
                # Rounded to the full readout's dtype so ties break like it.
                short = nn.functional.linear(step_repr.to(weight.dtype), weight, bias)
                token = ids[short.to(logits.dtype).argmax(dim=-1)]
            drafted.append(token)
            if self.config.markov_head:
                prev_embed = self.token_embedding(token)
        return tuple(torch.cat(drafted).tolist())

    def _block_state_incremental(
        self,
        cache: DSparkIncrementalCache,
        *,
        hidden_taps: tuple[torch.Tensor, ...],
        committed_tokens: tuple[int, ...],
    ) -> torch.Tensor:
        """`_block_state`'s result, computed via `cache` instead of
        re-attending the whole window every round -- see
        :class:`DSparkIncrementalCache`'s docstring for why this is exact
        only under `config.dspark_context_last=True`."""
        cache.update(self, committed_tokens)
        device = self.start_embedding.device
        dtype = self.hidden_proj.weight.dtype
        stacked = torch.stack(
            [tap.reshape(-1).to(device=device, dtype=dtype) for tap in hidden_taps]
        ).unsqueeze(0)
        context_hidden = self.hidden_proj(self._flatten_taps(stacked)).unsqueeze(1)
        for layer_index in range(4):
            layer = self.decoder_layers[layer_index]
            wq, wk, wv, bq, bk, bv = _split_in_proj(layer)
            q = torch.nn.functional.linear(context_hidden, wq, bq)
            k_ctx = torch.nn.functional.linear(context_hidden, wk, bk)
            v_ctx = torch.nn.functional.linear(context_hidden, wv, bv)
            k_full = torch.cat([cache.k[layer_index], k_ctx], dim=1)
            v_full = torch.cat([cache.v[layer_index], v_ctx], dim=1)
            scores = torch.matmul(q, k_full.transpose(-2, -1)) / (k_full.shape[-1] ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = layer.attn.out_proj(torch.matmul(attn, v_full))
            context_hidden = layer.norm(context_hidden + out)
        return context_hidden[:, -1, :]

    def draft_block_incremental(
        self,
        cache: DSparkIncrementalCache,
        *,
        hidden_taps: tuple[torch.Tensor, ...],
        committed_tokens: tuple[int, ...],
        first_token: int | None = None,
        shortlist: int = 0,
    ) -> tuple[int, ...]:
        """`draft_block`'s result, using `cache` (see
        :class:`DSparkIncrementalCache`) instead of `_block_state` so the
        committed prefix window's already-cached positions are not
        re-attended every round. `committed_tokens` must be the FULL
        committed prefix (the cache windows it internally, exactly like
        `draft_block`'s own `prefix_tokens` argument). Same sequential
        Markov-head walk as `draft_block` from there -- only the one-time
        block-state summary changes."""
        with torch.no_grad():
            seed = self._seed_tensor(first_token)
            block_state = self._block_state_incremental(
                cache, hidden_taps=hidden_taps, committed_tokens=committed_tokens
            )
            return self._markov_walk(block_state, seed, shortlist=shortlist)

    def forward_training_block(
        self, *, hidden_taps: tuple[torch.Tensor, ...], prefix_tokens: tuple[int, ...],
        target_block: tuple[int, ...],
    ) -> torch.Tensor:
        """Gradient-carrying, TEACHER-FORCED training surrogate for
        :meth:`draft_block`: the `block_size` positions are computed in
        parallel (one `_block_state` pass, then one batched Markov-head
        projection per position using the TARGET's own previous token
        instead of the drafter's own prediction), matching
        `DflashDrafter.forward_training_step`'s "single-pass surrogate"
        level of implementation. Returns `(1, block_size, vocab_size)`
        logits. `len(target_block)` must equal `config.block_size`."""
        if len(target_block) != self.config.block_size:
            raise ValueError(
                f"target_block must have config.block_size={self.config.block_size} "
                f"tokens, got {len(target_block)}"
            )
        block_state = self._block_state(hidden_taps=hidden_taps, prefix_tokens=prefix_tokens)
        if not self.config.markov_head:
            step_reprs = block_state.expand(self.config.block_size, -1)
            return self.output_head(step_reprs).unsqueeze(0)
        device = block_state.device
        previous_tokens = (
            torch.tensor(target_block[:-1], dtype=torch.long, device=device)
            if self.config.block_size > 1
            else torch.empty(0, dtype=torch.long, device=device)
        )
        previous_embeds = torch.cat(
            [self.start_embedding.view(1, -1), self.token_embedding(previous_tokens)], dim=0
        )
        step_reprs = self.markov_proj(
            torch.cat([block_state.expand(self.config.block_size, -1), previous_embeds], dim=-1)
        )
        return self.output_head(step_reprs).unsqueeze(0)

    def _block_states_batched(
        self,
        *,
        hidden_taps: torch.Tensor,
        prefix_windows: torch.Tensor,
        prefix_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Batched :meth:`_block_state` over `B` independent blocks, for the
        offline trainer and acceptance measurement. `hidden_taps` is
        `(B, n_taps, hidden_dim)`; `prefix_windows` is `(B, W)` holding each
        block's last `prefix_lengths[b] <= W` committed tokens LEFT-aligned
        (right-padded with any id). Causal attention means the padding after
        position `prefix_lengths[b]` cannot reach that position, so reading
        the output there equals the one-block form exactly. Requires
        `config.dspark_prefix_window` (the one-block form's whole-prefix
        mode has no fixed width to batch over). Returns `(B, d_model)`."""
        window = self.config.dspark_prefix_window
        if window is None:
            raise ValueError("batched DSpark forms require config.dspark_prefix_window")
        if prefix_windows.shape[1] > window:
            raise ValueError(
                f"prefix_windows is {prefix_windows.shape[1]} wide, window is {window}"
            )
        device = self.start_embedding.device
        dtype = self.hidden_proj.weight.dtype
        batch = hidden_taps.shape[0]
        context = self.hidden_proj(
            self._flatten_taps(hidden_taps.to(device=device, dtype=dtype))
        )
        prefix_embed = self.token_embedding(prefix_windows.to(device))
        rows = torch.arange(batch, device=device)
        lengths = prefix_lengths.to(device)
        if self.config.dspark_context_last:
            # Prefix left-aligned at 0..n-1, context written at n; anything
            # after n is padding the causal mask keeps away from position n.
            sequence = torch.cat([prefix_embed, context.new_zeros(batch, 1, context.shape[-1])],
                                 dim=1)
            sequence[rows, lengths] = context.to(sequence.dtype)
        else:
            sequence = torch.cat([context.unsqueeze(1), prefix_embed], dim=1)
        steps = sequence.shape[1]
        causal_mask = torch.triu(
            torch.ones(steps, steps, dtype=torch.bool, device=device), diagonal=1
        )
        for layer in self.decoder_layers:
            sequence = layer(sequence, attn_mask=causal_mask)
        return sequence[rows, lengths]

    def forward_training_blocks(
        self,
        *,
        hidden_taps: torch.Tensor,
        prefix_windows: torch.Tensor,
        prefix_lengths: torch.Tensor,
        target_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """Batched :meth:`forward_training_block` (same teacher forcing):
        `target_blocks` is `(B, block_size)`; returns `(B, block_size,
        vocab_size)` logits. Input layout as :meth:`_block_states_batched`."""
        block = self.config.block_size
        if target_blocks.shape[1] != block:
            raise ValueError(
                f"target_blocks must have config.block_size={block} columns, "
                f"got {target_blocks.shape[1]}"
            )
        states = self._block_states_batched(
            hidden_taps=hidden_taps, prefix_windows=prefix_windows, prefix_lengths=prefix_lengths
        )
        batch = states.shape[0]
        if not self.config.markov_head:
            return self.output_head(states).unsqueeze(1).expand(batch, block, -1)
        previous = torch.cat(
            [
                self.start_embedding.view(1, 1, -1).expand(batch, 1, -1),
                self.token_embedding(target_blocks[:, :-1].to(states.device)),
            ],
            dim=1,
        )
        step_reprs = self.markov_proj(
            torch.cat([states.unsqueeze(1).expand(-1, block, -1), previous], dim=-1)
        )
        return self.output_head(step_reprs)

    def draft_blocks(
        self,
        *,
        hidden_taps: torch.Tensor,
        prefix_windows: torch.Tensor,
        prefix_lengths: torch.Tensor,
        first_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Batched :meth:`draft_block` (same sequential Markov-head greedy
        walk, `first_tokens` `(B,)` as its `first_token`): returns
        `(B, block_size)` drafted token ids. Input layout as
        :meth:`_block_states_batched`."""
        with torch.no_grad():
            states = self._block_states_batched(
                hidden_taps=hidden_taps,
                prefix_windows=prefix_windows,
                prefix_lengths=prefix_lengths,
            )
            batch = states.shape[0]
            prev_embed = self.start_embedding.view(1, -1).expand(batch, -1)
            drafted: list[torch.Tensor] = []
            if first_tokens is not None:
                seeds = first_tokens.to(states.device)
                drafted.append(seeds)
                prev_embed = self.token_embedding(seeds)
            for _ in range(self.config.block_size - len(drafted)):
                if self.config.markov_head:
                    step_repr = self.markov_proj(torch.cat([states, prev_embed], dim=-1))
                else:
                    step_repr = states
                tokens = self.output_head(step_repr).argmax(dim=-1)
                drafted.append(tokens)
                if self.config.markov_head:
                    prev_embed = self.token_embedding(tokens)
            return torch.stack(drafted, dim=1)


def _shortlist_rows(
    output_head: nn.Module, logits: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """`(ids, weight, bias)` for `DSparkDrafter._markov_walk`'s shortlist:
    the top-`k` ids of one step's full `logits` (`(vocab,)`), sorted
    ascending so an argmax over the rows breaks ties toward the lower id
    exactly like the full readout's argmax, and the head's rows for them.
    A head that is not a plain `nn.Linear` over the id space (the eval
    `TiedOutputHead`, whose ids span three sub-tables) supplies its rows
    through `shortlist_rows(ids) -> (weight, bias)`."""
    ids = logits.topk(min(k, logits.shape[-1])).indices.sort().values
    rows = getattr(output_head, "shortlist_rows", None)
    if rows is not None:
        weight, bias = rows(ids)
        return ids, weight, bias
    bias = None if output_head.bias is None else output_head.bias[ids]
    return ids, output_head.weight[ids], bias


def _split_in_proj(layer: _AttentionOnlyLayer) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None,
    torch.Tensor | None,
]:
    """`nn.MultiheadAttention`'s combined `in_proj_weight`/`in_proj_bias`
    split into per-projection `(wq, wk, wv, bq, bk, bv)`, matching exactly
    what `nn.MultiheadAttention.forward` does internally for the
    `nhead=1`, `kdim == vdim == embed_dim` self-attention case
    `_AttentionOnlyLayer` always builds -- required to project queries and
    keys/values separately for the incremental cache below."""
    d_model = layer.attn.embed_dim
    w = layer.attn.in_proj_weight
    b = layer.attn.in_proj_bias
    wq, wk, wv = w.split(d_model, dim=0)
    if b is None:
        return wq, wk, wv, None, None, None
    bq, bk, bv = b.split(d_model, dim=0)
    return wq, wk, wv, bq, bk, bv


class DSparkIncrementalCache:
    """Per-round incremental key/value cache for `DSparkDrafter`'s 4-layer
    attention-only block-state decoder (`_block_state`), valid only when
    `config.dspark_context_last=True`. Under that layout the causal mask
    guarantees the committed prefix window's per-layer projections never
    depend on the context row (which changes every round with the target's
    latest hidden taps) -- so a prefix token's key/value at each layer can
    be computed exactly ONCE, when it first enters the window, and reused
    for every later round, AS LONG AS THE WINDOW ONLY GROWS (pure append,
    nothing evicted from its front yet).

    Without this cache, every `draft()` call re-projects and re-attends the
    WHOLE window (`config.dspark_prefix_window` tokens, or the whole
    committed prefix when unset) through all 4 layers even though only the
    positions accepted since the previous round are new. With it, a round
    that only appends `n` new tokens does `O(n)` work on the prefix side
    (plus a fixed `O(1)`-in-window-length cost for the context row), not
    `O(window)`.

    **Why eviction cannot reuse the cache.** `_block_state` does not
    implement a genuine sliding-window attention span (where an old
    position's own hidden state, once computed, is fixed forever regardless
    of what later gets evicted from OTHERS' view of it) -- it re-derives a
    fresh short sequence from `prefix_tokens[-window:]` every round. A token
    that used to sit in the MIDDLE of a full window (seeing everything
    before it) can become the window's FIRST position next round once
    earlier tokens are evicted -- its causal-attention output over a
    narrower window is a genuinely different number, not a cheap reuse of
    what was cached. So the moment eviction would occur, `update()` does a
    full cold rebuild of the current window instead of attempting to keep
    a stale, now-incorrect cached value. `reset()` (or feeding a
    `committed_tokens` sequence with no prefix overlap, e.g. a decode
    rollback that diverges) forces the same cold-rebuild path -- correct in
    all cases, at the cost of losing that one round's savings.
    """

    def __init__(self, config: DflashConfig) -> None:
        if config.variant != "dspark":
            raise ValueError(
                f"DSparkIncrementalCache requires variant='dspark', got {config.variant!r}"
            )
        if not config.dspark_context_last:
            raise ValueError(
                "DSparkIncrementalCache requires config.dspark_context_last=True "
                "(only that layout keeps the prefix window's per-layer state "
                "independent of the ever-changing context row)"
            )
        self._window = config.dspark_prefix_window
        #: `k[layer_index]`/`v[layer_index]` for `layer_index` in `0..3`:
        #: `(1, n_window, d_model)` projected keys/values of the prefix
        #: window's INPUT to that `decoder_layers[layer_index]`, or `None`
        #: before the first `update()`.
        self.k: list[torch.Tensor | None] = [None, None, None, None]
        self.v: list[torch.Tensor | None] = [None, None, None, None]
        self._tokens: tuple[int, ...] = ()

    def reset(self) -> None:
        self.k = [None, None, None, None]
        self.v = [None, None, None, None]
        self._tokens = ()

    def update(self, drafter: DSparkDrafter, committed_tokens: tuple[int, ...]) -> None:
        """Bring the cache up to date with `committed_tokens` (the FULL
        committed prefix so far -- windowed internally exactly like
        `_block_state` windows `prefix_tokens`). Idempotent: calling it
        again with the same `committed_tokens` is a no-op."""
        windowed = (
            committed_tokens[-self._window :]
            if self._window is not None
            else committed_tokens
        )
        if windowed == self._tokens:
            return
        # Only a PURE APPEND (every already-cached token retained, some new
        # ones added at the end) is safe to reuse -- see the class
        # docstring for why any front eviction invalidates the cache.
        if self._tokens and windowed[: len(self._tokens)] != self._tokens:
            self.reset()
        new_tokens = windowed[len(self._tokens):]
        if not new_tokens:
            self._tokens = windowed
            return
        device = drafter.start_embedding.device
        dtype = drafter.hidden_proj.weight.dtype
        ids = torch.tensor(new_tokens, dtype=torch.long, device=device)
        hidden = drafter.token_embedding(ids).unsqueeze(0).to(dtype)  # (1, n_new, d_model)
        n_new = hidden.shape[1]
        for layer_index in range(4):
            layer = drafter.decoder_layers[layer_index]
            wq, wk, wv, bq, bk, bv = _split_in_proj(layer)
            k_old = self.k[layer_index]
            v_old = self.v[layer_index]
            n_old = 0 if k_old is None else k_old.shape[1]
            k_new = torch.nn.functional.linear(hidden, wk, bk)
            v_new = torch.nn.functional.linear(hidden, wv, bv)
            self.k[layer_index] = k_new if k_old is None else torch.cat([k_old, k_new], dim=1)
            self.v[layer_index] = v_new if v_old is None else torch.cat([v_old, v_new], dim=1)
            if layer_index == 3:
                break  # prefix's own layer-4 output is never read (see module docstring math)
            q_new = torch.nn.functional.linear(hidden, wq, bq)
            k_full = self.k[layer_index]
            v_full = self.v[layer_index]
            scores = torch.matmul(q_new, k_full.transpose(-2, -1)) / (k_full.shape[-1] ** 0.5)
            new_mask = torch.triu(
                torch.ones(n_new, n_new, dtype=torch.bool, device=device), diagonal=1
            )
            mask = torch.cat(
                [torch.zeros(n_new, n_old, dtype=torch.bool, device=device), new_mask], dim=1
            )
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out = layer.attn.out_proj(torch.matmul(attn, v_full))
            hidden = layer.norm(hidden + out)
        self._tokens = windowed


class DSparkSpeculativeDrafter:
    """`drafter.SpeculativeDrafter` adapter over a :class:`DSparkDrafter`,
    mirroring :class:`DflashSpeculativeDrafter` exactly (same fail-closed
    contract) except `hidden_state_provider` returns a TUPLE of tapped
    layers (one per `config.dspark_hidden_taps` entry) instead of a single
    tensor."""

    def __init__(
        self,
        *,
        model: DSparkDrafter,
        hidden_state_provider: Callable[[tuple[int, ...]], tuple[torch.Tensor, ...]],
        incremental_cache: DSparkIncrementalCache | None = None,
    ) -> None:
        """`incremental_cache` (2026-09-28, opt-in): when given, `draft`
        calls `model.draft_block_incremental` against it instead of
        `model.draft_block`, computing only the prefix positions accepted
        since the previous call (see `DSparkIncrementalCache`). `None`
        (default, every pre-existing caller) is the unchanged full-recompute
        path -- no behavior change, no compatibility-hash change (the cache
        is a runtime adapter detail, not part of `DflashConfig`)."""
        self._model = model
        self._hidden_state_provider = hidden_state_provider
        self._incremental_cache = incremental_cache
        #: See `DflashSpeculativeDrafter.last_error`'s identical docstring.
        self.last_error: BaseException | None = None

    def draft(self, snapshot: DecoderSnapshot) -> tuple[int, ...]:
        try:
            hidden_taps = self._hidden_state_provider(snapshot.committed_tokens)
            if self._incremental_cache is not None:
                return self._model.draft_block_incremental(
                    self._incremental_cache,
                    hidden_taps=hidden_taps,
                    committed_tokens=snapshot.committed_tokens,
                )
            return self._model.draft_block(
                hidden_taps=hidden_taps, prefix_tokens=snapshot.committed_tokens
            )
        except Exception as exc:  # noqa: BLE001 -- see `DflashSpeculativeDrafter.draft`
            self.last_error = exc
            logger.exception("dspark drafter draft failed; fail-closed to primary decode")
            return ()


def drafter_target_compatibility_hash(
    *,
    target_checkpoint_hash: str,
    tokenizer_identity: str,
    structural_vocabulary_hash: str,
    config: DflashConfig,
) -> str:
    """§37.2: "DFlash checkpoint musí byť viazaný na exact target checkpoint,
    tokenizer a structural token map musia byť identické, ... block size,
    anchors a acceptance algorithm musia byť pinned". One hash over all of
    it; loading a drafter against a target that produces a different hash
    must refuse (:func:`assert_drafter_compatible`)."""
    payload = {
        "target_checkpoint_hash": target_checkpoint_hash,
        "tokenizer_identity": tokenizer_identity,
        "structural_vocabulary_hash": structural_vocabulary_hash,
        "hidden_taps": list(config.hidden_taps),
        "block_size": config.block_size,
        "num_steps": config.num_steps,
        "anchor_policy": config.anchor_policy,
        "hidden_dim": config.hidden_dim,
        "d_model": config.d_model,
        "vocab_size": config.vocab_size,
        "variant": config.variant,
        "dspark_hidden_taps": list(config.dspark_hidden_taps),
        "markov_head": config.markov_head,
    }
    if config.dspark_prefix_window is not None:
        payload["dspark_prefix_window"] = config.dspark_prefix_window
    if config.dspark_tap_norm:
        payload["dspark_tap_norm"] = True
    if config.dspark_final_tap:
        payload["dspark_final_tap"] = True
    if config.dspark_context_last:
        payload["dspark_context_last"] = True
    if config.dspark_seed_first_token:
        payload["dspark_seed_first_token"] = True
    if config.variant == "dspark_v2":
        payload["dspark_v2_n_layers"] = config.dspark_v2_n_layers
        payload["dspark_v2_n_heads"] = config.dspark_v2_n_heads
        payload["dspark_v2_n_kv_heads"] = config.dspark_v2_n_kv_heads
        payload["dspark_v2_head_dim"] = config.dspark_v2_head_dim
        payload["dspark_v2_ffn_intermediate"] = config.dspark_v2_ffn_intermediate
        payload["dspark_v2_markov_rank"] = config.dspark_v2_markov_rank
    return domain_hash(_COMPATIBILITY_DOMAIN, payload)


_TAPS_BINDING_DOMAIN: Final = "unflatten.speculative.dflash-taps-binding/v1"


def taps_compatibility_hash(
    *,
    target_checkpoint_hash: str,
    tokenizer_identity: str,
    structural_vocabulary_hash: str,
    hidden_taps: tuple[int, ...],
    final_tap: bool,
    dtype: str,
    export_mode: str,
) -> str:
    """What an offline taps export (`dflash_taps_export.py`) actually
    depends on: which frozen target produced the hidden states, which
    layers were tapped, whether the final (dense-adapted) hidden state rides
    along, at what precision, and whether the walk was free-running or
    teacher-forced. Deliberately excludes drafter-architecture fields
    (`block_size`, `variant`, `num_steps`, head counts, ...) that
    :func:`drafter_target_compatibility_hash` also covers -- those describe
    the model trained ON the taps, not the taps themselves, so a `dspark`
    and a `dspark_v2` drafter can train from the exact same export (2026-09
    -28 fix, see `docs/runs/drafter-training-runbook-2026-09-28.md` §0.2)."""
    payload = {
        "target_checkpoint_hash": target_checkpoint_hash,
        "tokenizer_identity": tokenizer_identity,
        "structural_vocabulary_hash": structural_vocabulary_hash,
        "hidden_taps": list(hidden_taps),
        "final_tap": final_tap,
        "dtype": dtype,
        "export_mode": export_mode,
    }
    return domain_hash(_TAPS_BINDING_DOMAIN, payload)


class DflashDrafterStale(RuntimeError):
    """§37.2: "po zmene target weights/tokenizer/structural vocabulary je
    starý drafter nekompatibilný" -- a mismatching compatibility hash."""


def assert_drafter_compatible(*, expected_hash: str, current_hash: str) -> None:
    """Raise :class:`DflashDrafterStale` unless the drafter's recorded
    compatibility hash still matches the target it is being loaded against."""
    if expected_hash != current_hash:
        raise DflashDrafterStale(
            f"drafter_target_compatibility_hash mismatch: drafter was bound to "
            f"{expected_hash!r}, target now resolves to {current_hash!r} -- "
            "the target checkpoint, tokenizer or structural vocabulary changed"
        )


def assert_taps_compatible(*, expected_hashes: frozenset[str], current_hash: str) -> None:
    """Like :func:`assert_drafter_compatible`, but against a *set* of hashes
    a taps export may legitimately carry: the current
    :func:`taps_compatibility_hash` plus any legacy hash existing on-disk
    records were bound to before this function existed (2026-09-28,
    `docs/runs/drafter-training-runbook-2026-09-28.md` §0.2). Raises
    :class:`DflashDrafterStale` only when the record matches none of them."""
    if current_hash not in expected_hashes:
        raise DflashDrafterStale(
            f"drafter_target_compatibility_hash mismatch: taps record was bound to "
            f"{current_hash!r}, none of the accepted bindings {sorted(expected_hashes)!r} "
            "match -- the target checkpoint, tokenizer or structural vocabulary changed"
        )


class DflashSpeculativeDrafter:
    """`drafter.SpeculativeDrafter` adapter over a :class:`DflashDrafter`.

    `hidden_state_provider(prefix) -> Tensor` supplies the pinned hidden-state
    tap for a given committed prefix (e.g. re-derived from the target's own
    KV cache, matching `test_drafter.py`'s `_ModelPrimaryDecoder._sync_to`
    pattern) -- this adapter has no opinion on how that tap is produced.
    """

    def __init__(
        self,
        *,
        model: DflashDrafter,
        hidden_state_provider: Callable[[tuple[int, ...]], torch.Tensor],
    ) -> None:
        self._model = model
        self._hidden_state_provider = hidden_state_provider
        #: Last caught exception, kept only so tests can assert the
        #: fail-closed path actually fired (not observable via the token
        #: stream alone once the primary re-derives the same tokens).
        self.last_error: BaseException | None = None

    def draft(self, snapshot: DecoderSnapshot) -> tuple[int, ...]:
        try:
            hidden_tap = self._hidden_state_provider(snapshot.committed_tokens)
            return self._model.draft_block(
                hidden_tap=hidden_tap, prefix_tokens=snapshot.committed_tokens
            )
        except Exception as exc:  # noqa: BLE001 -- §37.2 fail-closed contract: ANY
            # runtime error here must degrade to primary-only decode, not
            # propagate. `run_speculative_decode` already treats an empty
            # draft as "nothing verified, fall back to primary" (its own
            # docstring) -- this adapter's only job is to never let a drafter
            # failure reach that loop as an exception.
            self.last_error = exc
            logger.exception("dflash_speculator draft failed; fail-closed to primary decode")
            return ()

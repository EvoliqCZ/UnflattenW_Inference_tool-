"""RFC004 §37.2 / DEC-047: dormant S7 trainer for `dflash_speculator`.

**Offline-only, by design (owner note 2026-09-27).** The full target model
cannot run a full 300 DPI page prefill on the local RTX 4070 (12 GiB):
weights alone are ~9 GiB and even chunked prefill OOMs. This trainer never
instantiates the full target at all -- it reads:

1. :class:`~unflatten.model.speculative.dflash_taps_export.DflashTapsRecord`
   files (built by that module elsewhere, e.g. the H200, from the target's
   own frozen greedy decode), and
2. the target's frozen token-embedding and output-head *tensors* alone
   (`DflashDrafter`'s `frozen_token_embedding_weight`/`frozen_output_head_*`
   constructor args) -- never the backbone, decoder or adapter that produced
   them.

Every record is checked against the caller's expected
`drafter_target_compatibility_hash` before it is used (`assert_drafter_
compatible`); a record from a stale target refuses. Like every other S7
child (:mod:`unflatten.model.visual_compression.s7_ratification`), a *real*
gradient step additionally requires an explicit ratification record for
child `"dflash_speculator"`; only `--dry-run` runs today.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from unflatten.model.hashing import domain_hash
from unflatten.model.speculative.dflash import (
    DflashConfig,
    DflashDrafter,
    DSparkDrafter,
    assert_drafter_compatible,
)
from unflatten.model.speculative.dflash_taps_export import DflashTapsRecord, DSparkTapsRecord
from unflatten.model.speculative.dspark_v2 import DSparkV2Drafter, position_weights, survival_labels
from unflatten.model.speculative.s7_dflash_policy import (
    load_s7_dflash_policy,
    s7_dflash_policy_hash,
)
from unflatten.model.visual_compression.s7_ratification import (
    S7RatificationRecord,
    require_ratification,
)

_S7_DFLASH_MANIFEST_DOMAIN = "rfc005.s7-dflash-run-manifest/v1"
_CHILD = "dflash_speculator"


def build_offline_drafter(
    config: DflashConfig,
    *,
    frozen_token_embedding_weight: torch.Tensor,
    frozen_output_head_weight: torch.Tensor,
    frozen_output_head_bias: torch.Tensor | None = None,
) -> DflashDrafter:
    """The only drafter constructor this trainer uses: embedding and output
    head are the target's own frozen tables, loaded alone (never the full
    target). Only `hidden_proj`, `encoder_layer` and `mask_embedding` train."""
    return DflashDrafter(
        config,
        frozen_token_embedding_weight=frozen_token_embedding_weight,
        frozen_output_head_weight=frozen_output_head_weight,
        frozen_output_head_bias=frozen_output_head_bias,
    )


def run_dflash_offline_step(
    *, drafter: DflashDrafter, taps_record: DflashTapsRecord, optimizer: torch.optim.Optimizer
) -> dict[str, float]:
    """One optimizer step of block self-distillation over one page's cached
    record: slide non-overlapping `block_size` windows over the target's own
    greedy `token_stream`, using the hidden tap recorded just before each
    block as the drafter's conditioning input, and cross-entropy the
    drafter's one-step logits (`forward_training_step` -- the training
    surrogate for the multi-step denoise; see that method's docstring)
    against the target's own tokens for that block.
    """
    block_size = drafter.config.block_size
    token_stream = taps_record.token_stream
    hidden_taps = taps_record.hidden_taps
    optimizer.zero_grad()
    losses: list[torch.Tensor] = []
    for start in range(0, len(token_stream) - block_size + 1, block_size):
        prefix = token_stream[:start]
        target_block = token_stream[start : start + block_size]
        hidden_tap = hidden_taps[max(start - 1, 0)].float()
        logits = drafter.forward_training_step(hidden_tap=hidden_tap, prefix_tokens=prefix)
        loss = nn.functional.cross_entropy(
            logits.view(block_size, -1), torch.tensor(target_block, dtype=torch.long)
        )
        losses.append(loss)
    if not losses:
        return {"loss": 0.0, "n_blocks": 0}
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    optimizer.step()
    return {"loss": float(total_loss.detach()), "n_blocks": len(losses)}


def estimate_offline_memory_bytes(
    config: DflashConfig, *, optimizer_state_multiplier: int = 2, bytes_per_param: int = 4
) -> dict[str, int]:
    """Order-of-magnitude estimate only -- NOT a profiler measurement.
    `trainable_param_bytes`/`optimizer_state_bytes` cover `hidden_proj`,
    `encoder_layer` and `mask_embedding` (the only trainable pieces under
    :func:`build_offline_drafter`); `frozen_table_bytes` is the target's
    embedding + output-head tables, loaded once and excluded from optimizer
    state; `activations_per_page_bytes` is a rough per-page forward/backward
    footprint for one block."""
    d_model, vocab, hidden_dim, block = (
        config.d_model,
        config.vocab_size,
        config.hidden_dim,
        config.block_size,
    )
    encoder_layer_params = 4 * d_model * d_model + 2 * d_model * (d_model * 2)
    hidden_proj_params = hidden_dim * d_model + d_model
    trainable_params = encoder_layer_params + hidden_proj_params + d_model
    frozen_table_params = vocab * d_model * 2
    activations_per_page = block * d_model * bytes_per_param * 8
    return {
        "trainable_param_bytes": trainable_params * bytes_per_param,
        "optimizer_state_bytes": trainable_params * bytes_per_param * optimizer_state_multiplier,
        "frozen_table_bytes": frozen_table_params * bytes_per_param,
        "activations_per_page_bytes": activations_per_page,
    }


def build_offline_dspark_drafter(
    config: DflashConfig,
    *,
    frozen_token_embedding_weight: torch.Tensor,
    frozen_output_head_weight: torch.Tensor,
    frozen_output_head_bias: torch.Tensor | None = None,
) -> DSparkDrafter:
    """DSpark variant of :func:`build_offline_drafter`: only `hidden_proj`,
    `decoder_layers` and `markov_proj` train -- embedding and output head
    are the target's own frozen tables, same as the block-diffusion path."""
    return DSparkDrafter(
        config,
        frozen_token_embedding_weight=frozen_token_embedding_weight,
        frozen_output_head_weight=frozen_output_head_weight,
        frozen_output_head_bias=frozen_output_head_bias,
    )


def run_dspark_offline_step(
    *, drafter: DSparkDrafter, taps_record: DSparkTapsRecord, optimizer: torch.optim.Optimizer
) -> dict[str, float]:
    """Self-distillation over the target's own greedy `token_stream`, one
    non-overlapping `block_size` window at a time, mirroring
    :func:`run_dflash_offline_step` but reading `taps_record.hidden_taps`'s
    `(T, n_taps, hidden_dim)` stack and cross-entropying
    `DSparkDrafter.forward_training_block`'s TEACHER-FORCED, per-position
    logits against the target's own tokens for that block."""
    block_size = drafter.config.block_size
    token_stream = taps_record.token_stream
    hidden_taps = taps_record.hidden_taps
    optimizer.zero_grad()
    losses: list[torch.Tensor] = []
    for start in range(0, len(token_stream) - block_size + 1, block_size):
        prefix = token_stream[:start]
        target_block = token_stream[start : start + block_size]
        tap_index = max(start - 1, 0)
        taps = tuple(hidden_taps[tap_index, i].float() for i in range(hidden_taps.shape[1]))
        logits = drafter.forward_training_block(
            hidden_taps=taps, prefix_tokens=prefix, target_block=target_block
        )
        loss = nn.functional.cross_entropy(
            logits.view(block_size, -1), torch.tensor(target_block, dtype=torch.long)
        )
        losses.append(loss)
    if not losses:
        return {"loss": 0.0, "n_blocks": 0}
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    optimizer.step()
    return {"loss": float(total_loss.detach()), "n_blocks": len(losses)}


def estimate_dspark_offline_memory_bytes(
    config: DflashConfig, *, optimizer_state_multiplier: int = 2, bytes_per_param: int = 4
) -> dict[str, int]:
    """DSpark variant of :func:`estimate_offline_memory_bytes`. Trainable
    pieces under :func:`build_offline_dspark_drafter`: `hidden_proj`, the
    4 `_AttentionOnlyLayer`s (`nn.MultiheadAttention`'s `in_proj_weight`
    `3*d^2` + `out_proj` `d^2+d` + `LayerNorm`'s `2d`, per layer) and
    `markov_proj` (`2*d^2+d`)."""
    d_model, vocab, hidden_dim, n_taps = (
        config.d_model,
        config.vocab_size,
        config.hidden_dim,
        len(config.dspark_hidden_taps),
    )
    per_layer_params = (3 * d_model * d_model) + (d_model * d_model + d_model) + 2 * d_model
    decoder_params = 4 * per_layer_params
    hidden_proj_params = hidden_dim * n_taps * d_model + d_model
    markov_proj_params = 2 * d_model * d_model + d_model
    trainable_params = decoder_params + hidden_proj_params + markov_proj_params + d_model
    frozen_table_params = vocab * d_model * 2
    activations_per_page = config.block_size * d_model * bytes_per_param * 8
    return {
        "trainable_param_bytes": trainable_params * bytes_per_param,
        "optimizer_state_bytes": trainable_params * bytes_per_param * optimizer_state_multiplier,
        "frozen_table_bytes": frozen_table_params * bytes_per_param,
        "activations_per_page_bytes": activations_per_page,
    }


def build_dflash_run_manifest(
    *,
    policy_hash: str,
    seed: int,
    taps_record_page_id: str,
    ratification_record: S7RatificationRecord | None,
) -> dict[str, Any]:
    """This trainer's own manifest identity -- stage `S7`, child
    `dflash_speculator` -- never borrowed from the VC program's schemas."""
    manifest: dict[str, Any] = {
        "s7_dflash_run_manifest_schema_version": "0.1.0",
        "stage": "S7",
        "child": _CHILD,
        "policy_hash": policy_hash,
        "seed": seed,
        "taps_record_page_id": taps_record_page_id,
        "ratification_decision_id": (
            ratification_record.decision_id if ratification_record is not None else None
        ),
    }
    manifest["s7_dflash_run_manifest_hash"] = domain_hash(_S7_DFLASH_MANIFEST_DOMAIN, manifest)
    return manifest


def train_dflash_step_offline(
    *,
    drafter: DflashDrafter,
    taps_record: DflashTapsRecord,
    optimizer: torch.optim.Optimizer,
    expected_compatibility_hash: str,
    dry_run: bool,
    ratification_record: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """The gated, checked entry point: refuses a record bound to a different
    target (`assert_drafter_compatible`) and refuses a real gradient step
    without an explicit, matching ratification record for `dflash_speculator`
    (`dry_run=True` runs the stub step for tests without claiming to be
    authorized). Never constructs or references the full target model."""
    assert_drafter_compatible(
        expected_hash=expected_compatibility_hash,
        current_hash=taps_record.drafter_target_compatibility_hash,
    )
    record = require_ratification(
        child=_CHILD, dry_run=dry_run, ratification_record=ratification_record
    )
    policy = load_s7_dflash_policy()
    result = run_dflash_offline_step(
        drafter=drafter, taps_record=taps_record, optimizer=optimizer
    )
    manifest = build_dflash_run_manifest(
        policy_hash=s7_dflash_policy_hash(policy),
        seed=seed,
        taps_record_page_id=taps_record.page_id,
        ratification_record=record,
    )
    memory_estimate = estimate_offline_memory_bytes(drafter.config)
    return {
        **result,
        "manifest": manifest,
        "dry_run": dry_run,
        "memory_estimate_bytes": memory_estimate,
    }


def train_dspark_step_offline(
    *,
    drafter: DSparkDrafter,
    taps_record: DSparkTapsRecord,
    optimizer: torch.optim.Optimizer,
    expected_compatibility_hash: str,
    dry_run: bool,
    ratification_record: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """DSpark variant of :func:`train_dflash_step_offline` -- identical
    gating (compatibility check, ratification gate for `dflash_speculator`)
    and manifest shape, over :func:`run_dspark_offline_step` and
    :func:`estimate_dspark_offline_memory_bytes` instead of the
    block-diffusion pair."""
    assert_drafter_compatible(
        expected_hash=expected_compatibility_hash,
        current_hash=taps_record.drafter_target_compatibility_hash,
    )
    record = require_ratification(
        child=_CHILD, dry_run=dry_run, ratification_record=ratification_record
    )
    policy = load_s7_dflash_policy()
    result = run_dspark_offline_step(
        drafter=drafter, taps_record=taps_record, optimizer=optimizer
    )
    manifest = build_dflash_run_manifest(
        policy_hash=s7_dflash_policy_hash(policy),
        seed=seed,
        taps_record_page_id=taps_record.page_id,
        ratification_record=record,
    )
    memory_estimate = estimate_dspark_offline_memory_bytes(drafter.config)
    return {
        **result,
        "manifest": manifest,
        "dry_run": dry_run,
        "memory_estimate_bytes": memory_estimate,
    }


def build_offline_dspark_v2_drafter(
    config: DflashConfig,
    *,
    frozen_token_embedding_weight: torch.Tensor,
    frozen_output_head_weight: torch.Tensor,
    frozen_output_head_bias: torch.Tensor | None = None,
) -> DSparkV2Drafter:
    """`dspark_v2` variant of :func:`build_offline_dspark_drafter`: only the
    5-layer stack, context injection, Markov table and confidence head
    train -- embedding and output head are the target's own frozen tables."""
    return DSparkV2Drafter(
        config,
        frozen_token_embedding_weight=frozen_token_embedding_weight,
        frozen_output_head_weight=frozen_output_head_weight,
        frozen_output_head_bias=frozen_output_head_bias,
    )


def dspark_v2_loss(
    drafter: DSparkV2Drafter,
    taps: torch.Tensor,
    windows: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor,
    *,
    pages: list[DSparkPage] | None = None,
    picks: list[tuple[int, int]] | None = None,
    label_source: str = "gold",
    tv_weight: float = 0.0,
) -> torch.Tensor:
    """Position-weighted CE (`w_k = exp(-(k-1)/gamma)`, :func:`position_
    weights`) over slots `1..block_size-1`, plus confidence-head BCE, plus
    (owner 2026-09-28, `tv_weight != 0`) the paper's third, previously
    unimplemented loss: total variation against the target's own
    distribution restricted to its top-32 logits (:func:`top_k_tv_loss`).

    `label_source` (default `"gold"` here, backward compatible with every
    caller that never exported `final_hidden`): `"target"` teacher-forces
    AND supervises on the target's own on-policy argmax
    (`final_hidden_blocks_for_picks` + `drafter.output_head`) instead of
    the gold/exact-text token, and requires `pages`/`picks` -- a drafter
    trained on gold proposes gold at exactly the hard positions where the
    target would reject it. Confidence-head survival labels come from the
    SAME resolved reference as the CE term (target argmax under
    `label_source="target"`, gold under `"gold"`): "what real decoding
    accepts" has no target-argmax signal to fall back to in a pure-gold
    scenario with no `final_hidden` at all (e.g. a synthetic page with no
    real target output)."""
    if label_source not in ("target", "gold"):
        raise ValueError(f"unknown label_source {label_source!r}, expected 'target' or 'gold'")
    target_logits_full = None
    resolved = targets
    # Only resolved for label_source="target": a pure-gold ablation with no
    # exported final_hidden at all (e.g. a synthetic scenario with no real
    # target output) must stay fully gold, including the confidence head's
    # survival labels -- there is no on-policy signal to fall back to.
    if label_source == "target":
        if pages is None or picks is None:
            raise ValueError("label_source='target' requires pages and picks")
        target_hidden = final_hidden_blocks_for_picks(pages, picks, block_size=targets.shape[1])
        target_logits_full = target_logits_for_positions(target_hidden, drafter.output_head)
        resolved = target_logits_full.argmax(dim=-1)

    logits, confidence_logits = drafter.forward_training_blocks(
        hidden_taps=taps, prefix_windows=windows, prefix_lengths=lengths, target_blocks=resolved
    )
    ce_label_tail = resolved[:, 1:].to(logits.device)
    weights = position_weights(
        logits.shape[1], drafter.config.block_size, device=logits.device
    )
    ce_per_slot = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        ce_label_tail.reshape(-1),
        reduction="none",
    ).view(ce_label_tail.shape)
    ce = (ce_per_slot * weights).mean()
    survival_reference_tail = (
        target_logits_full.argmax(dim=-1)[:, 1:].to(logits.device)
        if target_logits_full is not None
        else targets[:, 1:].to(logits.device)
    )
    labels = survival_labels(logits.detach(), survival_reference_tail)
    bce = nn.functional.binary_cross_entropy_with_logits(confidence_logits, labels)
    loss = ce + bce
    if tv_weight and target_logits_full is not None:
        target_tail_logits = target_logits_full[:, 1:, :].to(logits.device)
        loss = loss + tv_weight * top_k_tv_loss(logits, target_tail_logits, k=32)
    return loss


def run_dspark_v2_offline_step(
    *, drafter: DSparkV2Drafter, taps_record: DSparkTapsRecord, optimizer: torch.optim.Optimizer
) -> dict[str, float]:
    """One optimizer step of :func:`dspark_v2_loss` over one page's cached
    record, one non-overlapping `block_size` window at a time -- the
    `dspark_v2` counterpart of :func:`run_dspark_offline_step`. Requires
    `drafter.config.dspark_prefix_window` (this variant has no whole-prefix
    form, unlike `DSparkDrafter`)."""
    window = drafter.config.dspark_prefix_window
    if window is None:
        raise ValueError("run_dspark_v2_offline_step requires config.dspark_prefix_window")
    block = drafter.config.block_size
    page = DSparkPage(
        page_id=taps_record.page_id,
        taps=taps_record.hidden_taps,
        tokens=torch.tensor(taps_record.token_stream, dtype=torch.long),
        final_hidden=taps_record.final_hidden,
    )
    optimizer.zero_grad()
    losses: list[torch.Tensor] = []
    for start in page.block_starts(block):
        taps, windows, lengths, targets = build_dspark_batch(
            [page], [(0, start)], window=window, block_size=block
        )
        losses.append(dspark_v2_loss(drafter, taps, windows, lengths, targets))
    if not losses:
        return {"loss": 0.0, "n_blocks": 0}
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    optimizer.step()
    return {"loss": float(total_loss.detach()), "n_blocks": len(losses)}


def estimate_dspark_v2_offline_memory_bytes(
    config: DflashConfig, *, optimizer_state_multiplier: int = 2, bytes_per_param: int = 4
) -> dict[str, int]:
    """Order-of-magnitude estimate only (module docstring's discipline,
    same as :func:`estimate_dspark_offline_memory_bytes`). Trainable
    pieces under :func:`build_offline_dspark_v2_drafter`: `context_proj` +
    `context_norm`, `mask_embedding`, `dspark_v2_n_layers` full transformer
    layers (GQA attention + dense SwiGLU FFN + 2 RMSNorms each), the
    low-rank Markov table (`markov_embed` + `markov_out`, `2 * vocab *
    rank`), and the confidence head."""
    d_model, vocab, hidden_dim = config.d_model, config.vocab_size, config.hidden_dim
    n_taps = config.n_tap_inputs
    n_heads, n_kv, head_dim = (
        config.dspark_v2_n_heads,
        config.dspark_v2_n_kv_heads,
        config.dspark_v2_head_dim,
    )
    intermediate, rank = config.dspark_v2_ffn_intermediate, config.dspark_v2_markov_rank
    attn_per_layer = (
        d_model * (n_heads * head_dim)
        + 2 * d_model * (n_kv * head_dim)
        + (n_heads * head_dim) * d_model
    )
    ffn_per_layer = 3 * d_model * intermediate
    per_layer = attn_per_layer + ffn_per_layer + 2 * d_model
    layers_params = config.dspark_v2_n_layers * per_layer
    context_params = hidden_dim * n_taps * d_model + d_model
    markov_params = 2 * vocab * rank
    confidence_params = (d_model + rank) + 1
    trainable_params = (
        context_params + layers_params + markov_params + confidence_params + d_model
    )
    frozen_table_params = vocab * d_model * 2
    activations_per_page = config.block_size * d_model * bytes_per_param * 8
    return {
        "trainable_param_bytes": trainable_params * bytes_per_param,
        "trainable_params": trainable_params,
        "optimizer_state_bytes": trainable_params * bytes_per_param * optimizer_state_multiplier,
        "frozen_table_bytes": frozen_table_params * bytes_per_param,
        "activations_per_page_bytes": activations_per_page,
    }


def train_dspark_v2_step_offline(
    *,
    drafter: DSparkV2Drafter,
    taps_record: DSparkTapsRecord,
    optimizer: torch.optim.Optimizer,
    expected_compatibility_hash: str,
    dry_run: bool,
    ratification_record: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """`dspark_v2` variant of :func:`train_dspark_step_offline` -- identical
    gating and manifest shape, over :func:`run_dspark_v2_offline_step` and
    :func:`estimate_dspark_v2_offline_memory_bytes`."""
    assert_drafter_compatible(
        expected_hash=expected_compatibility_hash,
        current_hash=taps_record.drafter_target_compatibility_hash,
    )
    record = require_ratification(
        child=_CHILD, dry_run=dry_run, ratification_record=ratification_record
    )
    policy = load_s7_dflash_policy()
    result = run_dspark_v2_offline_step(
        drafter=drafter, taps_record=taps_record, optimizer=optimizer
    )
    manifest = build_dflash_run_manifest(
        policy_hash=s7_dflash_policy_hash(policy),
        seed=seed,
        taps_record_page_id=taps_record.page_id,
        ratification_record=record,
    )
    memory_estimate = estimate_dspark_v2_offline_memory_bytes(drafter.config)
    return {
        **result,
        "manifest": manifest,
        "dry_run": dry_run,
        "memory_estimate_bytes": memory_estimate,
    }


def measure_dspark_v2_acceptance(
    drafter: DSparkV2Drafter,
    pages: list[DSparkPage],
    *,
    batch_size: int = 256,
    max_blocks: int | None = None,
    seed: int = 0,
    label_source: str = "gold",
) -> dict[str, Any]:
    """`dspark_v2` variant of :func:`measure_dspark_acceptance`: i.i.d.
    offline acceptance over every valid block start, drafting with
    `DSparkV2Drafter.draft_blocks`'s own sequential Markov-head walk (slot 0
    seeded from the target's own stream, since this variant always requires
    `dspark_seed_first_token`). `label_source` (owner 2026-09-28) -- same
    contract as :func:`measure_dspark_acceptance`, default `"gold"` here
    for backward compatibility with callers that never exported
    `final_hidden`."""
    block = drafter.config.block_size
    window = drafter.config.dspark_prefix_window
    if window is None:
        raise ValueError("measure_dspark_v2_acceptance requires config.dspark_prefix_window")
    picks = all_block_picks(pages, block)
    if max_blocks is not None and len(picks) > max_blocks:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(picks), generator=generator)[:max_blocks]
        picks = [picks[i] for i in sorted(order.tolist())]
    histogram = [0] * (block + 1)
    was_training = drafter.training
    drafter.eval()
    for offset in range(0, len(picks), batch_size):
        batch_picks = picks[offset : offset + batch_size]
        taps, windows, lengths, gold_targets = build_dspark_batch(
            pages, batch_picks, window=window, block_size=block
        )
        targets = resolve_block_targets(
            pages, batch_picks, gold_targets,
            label_source=label_source, output_head=drafter.output_head,
        )
        drafted, _confidences = drafter.draft_blocks(
            hidden_taps=taps, prefix_windows=windows, prefix_lengths=lengths,
            first_tokens=targets[:, 0],
        )
        for value in accepted_lengths(drafted.cpu(), targets).tolist():
            histogram[value] += 1
    drafter.train(was_training)
    n_blocks = sum(histogram)
    mean = sum(k * c for k, c in enumerate(histogram)) / n_blocks if n_blocks else 0.0
    return {"n_blocks": n_blocks, "mean_accepted": mean, "histogram": histogram}


def measure_round_boundary_acceptance(
    drafter: DSparkDrafter | DSparkV2Drafter,
    page: DSparkPage,
    *,
    seed: int = 0,
    label_source: str = "gold",
) -> dict[str, Any]:
    """Online-style REPLAY acceptance (rehearsal doc §13, 2026-09-28): draft
    a fresh block only at each round's start, advancing `start` by
    `max(accepted, 1)` -- exactly `drafter.run_speculative_decode`'s own
    loop shape -- instead of `measure_dspark_acceptance`/`measure_dspark_v2_
    acceptance`'s i.i.d. sample over every valid position. The rehearsal
    found this systematically lower than the i.i.d. number (harder,
    correlated starting contexts), not a bug -- see that doc's §13 for the
    replay this mirrors. Single page (a diagnostic replay over one already
    -exported stream); works for either drafter class since both share the
    same `draft_blocks` calling convention (`DSparkV2Drafter`'s additionally
    returns confidences, discarded here)."""
    block = drafter.config.block_size
    window = drafter.config.dspark_prefix_window
    if window is None:
        raise ValueError("measure_round_boundary_acceptance requires config.dspark_prefix_window")
    tokens = page.tokens
    histogram = [0] * (block + 1)
    was_training = drafter.training
    drafter.eval()
    start = 1  # row 0's tap conditions block [0, block), excluded (DSparkPage.block_starts)
    with torch.no_grad():
        while start + block <= len(tokens) and start - 1 < page.tap_rows:
            taps, windows, lengths, gold_targets = build_dspark_batch(
                [page], [(0, start)], window=window, block_size=block
            )
            targets = resolve_block_targets(
                [page], [(0, start)], gold_targets,
                label_source=label_source, output_head=drafter.output_head,
            )
            output = drafter.draft_blocks(
                hidden_taps=taps, prefix_windows=windows, prefix_lengths=lengths,
                first_tokens=targets[:, 0],
            )
            drafted = output[0] if isinstance(output, tuple) else output
            accepted = int(accepted_lengths(drafted.cpu(), targets).item())
            histogram[accepted] += 1
            start += max(accepted, 1)
    drafter.train(was_training)
    n_rounds = sum(histogram)
    mean = sum(k * c for k, c in enumerate(histogram)) / n_rounds if n_rounds else 0.0
    return {"n_rounds": n_rounds, "mean_accepted": mean, "histogram": histogram}


def select_epoch_by_acceptance(curve_path: Path) -> dict[str, Any] | None:
    """The row of a `train_dspark_offline`/`train_dspark_v2_offline`
    `curve.jsonl` with the HIGHEST `eval_mean_accepted` -- "select epoch by
    acceptance, not loss" (owner instruction, matching the paper's own
    stated selection criterion). `None` if the curve has no scored rows."""
    rows = [
        json.loads(line)
        for line in curve_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored = [row for row in rows if row.get("eval_mean_accepted") is not None]
    if not scored:
        return None
    return max(scored, key=lambda row: row["eval_mean_accepted"])


def train_dspark_v2_offline(
    *,
    drafter: DSparkV2Drafter,
    train_pages: list[DSparkPage],
    eval_pages: list[DSparkPage],
    steps: int,
    batch_blocks: int,
    lr: float,
    warmup_steps: int,
    eval_every: int,
    eval_max_blocks: int,
    checkpoint_dir: Path,
    ratification_record: Path | None,
    seed: int = 0,
    log: Any = print,
    label_source: str = "target",
    tv_weight: float = 1.0,
) -> dict[str, Any]:
    """`dspark_v2` variant of :func:`train_dspark_offline`: same AdamW/
    warmup-cosine schedule and resumable checkpointing, over
    :func:`dspark_v2_loss` (CE + confidence BCE + top-32 TV) and
    :func:`measure_dspark_v2_acceptance`. Selects the best step by
    held-out acceptance at the end (:func:`select_epoch_by_acceptance`),
    not by final-step loss. Refuses before any gradient without a matching
    ratification record for `dflash_speculator` (dspark_v2 is a variant of
    the same S7 child, not a new one).

    `label_source`/`tv_weight` (owner 2026-09-28, defaults `"target"`/
    `1.0`): see :func:`dspark_v2_loss`. `label_source="target"` requires
    every page's `final_hidden`. The held-out `target_gold_mismatch_rate`
    is computed once up front and returned in the report regardless."""
    import math
    import time

    record = require_ratification(
        child=_CHILD, dry_run=False, ratification_record=ratification_record
    )
    block = drafter.config.block_size
    window = drafter.config.dspark_prefix_window
    if window is None:
        raise ValueError("train_dspark_v2_offline requires config.dspark_prefix_window")
    picks = all_block_picks(train_pages, block)
    if not picks:
        raise ValueError("no trainable blocks in train_pages")
    params = [p for p in drafter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    generator = torch.Generator().manual_seed(seed)
    start_step = 0
    if last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        drafter.load_state_dict(payload["trainable_state"], strict=False)
        optimizer.load_state_dict(payload["optimizer_state"])
        generator.set_state(payload["generator_state"])
        start_step = payload["step"]
        log(f"[train] resumed from {last_path} at step {start_step}")
    eval_sample = all_block_picks(eval_pages, block)
    fixed_generator = torch.Generator().manual_seed(seed + 1)
    eval_loss_picks = [
        eval_sample[i]
        for i in torch.randperm(len(eval_sample), generator=fixed_generator)[:2048].tolist()
    ]
    config_payload = {
        field: getattr(drafter.config, field) for field in drafter.config.__dataclass_fields__
    }

    def save(step: int, name: str, *, resume_state: bool) -> None:
        payload: dict[str, Any] = {
            "step": step,
            "config": config_payload,
            "trainable_state": trainable_state(drafter),
            "ratification_decision_id": record.decision_id if record else None,
        }
        if resume_state:
            payload["optimizer_state"] = optimizer.state_dict()
            payload["generator_state"] = generator.get_state()
        tmp = checkpoint_dir / f"{name}.tmp"
        torch.save(payload, tmp)
        tmp.replace(checkpoint_dir / name)

    mismatch_rate = (
        target_gold_mismatch_rate(
            eval_pages, eval_loss_picks, block_size=block, output_head=drafter.output_head
        )
        if label_source == "target"
        else None
    )
    if mismatch_rate is not None:
        log(f"[train] held-out target-argmax vs gold mismatch rate: {mismatch_rate:.4f}")

    # Held-out loss chunks no larger than a training batch: fixed 256-block
    # chunks OOMed the 12 GiB card once AdamW state was resident (2026-09-28).
    eval_chunk = min(256, batch_blocks)

    def evaluate(step: int, running: list[float], started: float) -> dict[str, Any]:
        drafter.eval()
        with torch.no_grad():
            losses = []
            for offset in range(0, len(eval_loss_picks), eval_chunk):
                eval_picks = eval_loss_picks[offset : offset + eval_chunk]
                batch = build_dspark_batch(
                    eval_pages, eval_picks, window=window, block_size=block,
                )
                losses.append(
                    float(
                        dspark_v2_loss(
                            drafter, *batch,
                            pages=eval_pages, picks=eval_picks,
                            label_source=label_source, tv_weight=tv_weight,
                        )
                    )
                )
        acceptance = measure_dspark_v2_acceptance(
            drafter, eval_pages, max_blocks=eval_max_blocks, seed=seed, label_source=label_source
        )
        drafter.train()
        row = {
            "step": step,
            "train_loss": sum(running) / len(running) if running else None,
            "eval_loss": sum(losses) / len(losses) if losses else None,
            "eval_mean_accepted": acceptance["mean_accepted"],
            "eval_histogram": acceptance["histogram"],
            "label_source": label_source,
            "lr": lr_at(step),
            "elapsed_s": round(time.perf_counter() - started, 1),
        }
        with (checkpoint_dir / "curve.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        log(f"[train] {json.dumps(row)}")
        return row

    drafter.train()
    started = time.perf_counter()
    running: list[float] = []
    last_row: dict[str, Any] = {}
    if start_step == 0:
        last_row = evaluate(0, running, started)
    for step in range(start_step, steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        chosen = torch.randint(len(picks), (batch_blocks,), generator=generator).tolist()
        step_picks = [picks[i] for i in chosen]
        batch = build_dspark_batch(train_pages, step_picks, window=window, block_size=block)
        optimizer.zero_grad(set_to_none=True)  # before the forward: frees last step's grads
        loss = dspark_v2_loss(
            drafter, *batch,
            pages=train_pages, picks=step_picks, label_source=label_source, tv_weight=tv_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        running.append(float(loss.detach()))
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            last_row = evaluate(step + 1, running, started)
            running = []
            save(step + 1, "last.pt", resume_state=True)
            save(step + 1, f"step-{step + 1}.pt", resume_state=False)
    selected = select_epoch_by_acceptance(checkpoint_dir / "curve.jsonl")
    return {
        "steps": steps,
        "train_blocks": len(picks),
        "last_eval": last_row,
        "selected_by_acceptance": selected,
        "label_source": label_source,
        "target_gold_mismatch_rate": mismatch_rate,
        "ratification_decision_id": record.decision_id if record else None,
    }


#: Checkpoint keys that are the target's frozen tables, never saved with the
#: drafter (they are reloaded from the exported tables file).
_FROZEN_TABLE_KEYS = frozenset({"token_embedding.weight", "output_head.weight", "output_head.bias"})


@dataclass(frozen=True)
class DSparkPage:
    """One page's taps record in the trainer's working form: `taps` stays
    on the CPU in its stored dtype, `tokens` is a long tensor."""

    page_id: str
    taps: torch.Tensor  # (tap_rows, n_taps, hidden_dim)
    tokens: torch.Tensor  # (T,)
    #: `(tap_rows, d_model)` final hidden state per fed token (MTP input),
    #: `None` when the record was exported without it.
    final_hidden: torch.Tensor | None = None
    #: Set by :func:`with_final_tap`: `final_hidden` is joined as the last
    #: tap per row (:meth:`tap_row`), never concatenated into `taps` up front.
    append_final_tap: bool = False

    @property
    def tap_rows(self) -> int:
        if self.append_final_tap:
            return min(self.taps.shape[0], self.final_hidden.shape[0])
        return self.taps.shape[0]

    def tap_row(self, row: int) -> torch.Tensor:
        if not self.append_final_tap:
            return self.taps[row]
        final = self.final_hidden[row].to(self.taps.dtype).unsqueeze(0)
        return torch.cat([self.taps[row], final], dim=0)

    def block_starts(self, block_size: int) -> range:
        """Every start whose conditioning row (`start - 1`) exists and
        whose whole target block fits in the stream. Start 0 is excluded:
        the prefix pass's own taps are never recorded, and row 0 already
        saw token 0 (conditioning block `[0, block)` on it would leak)."""
        last = min(len(self.tokens) - block_size, self.tap_rows)
        return range(1, last + 1)


def resolve_holdout_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    """`selection.json`'s held-out split, preferring the trainers' own
    `holdout` key and falling back to `val_iid` -- `export-tf` writes the
    held-out split under `val_iid` and never renames it, so this removes the
    manual `jq '. + {holdout: .val_iid}'` copy step from the runbook
    (2026-09-28). Logs which key was used so a silent split swap is never
    invisible."""
    if "holdout" in selection:
        print("[selection] held-out split: 'holdout' key", flush=True)
        return selection["holdout"]
    if "val_iid" in selection:
        print("[selection] held-out split: no 'holdout' key, falling back to 'val_iid'", flush=True)
        return selection["val_iid"]
    raise KeyError("selection.json has neither a 'holdout' nor a 'val_iid' key")


def load_dspark_pages(
    taps_dir: Path,
    page_ids: list[str],
    *,
    expected_compatibility_hash: str | frozenset[str],
) -> list[DSparkPage]:
    """Load and compatibility-check every listed page's taps record; a
    page without a record is skipped (export still running), a record
    bound to another target refuses. `expected_compatibility_hash` accepts
    either a single hash (`assert_drafter_compatible`) or a set of
    acceptable hashes (`assert_taps_compatible`, `taps_binding`/`taps_
    binding_v2`'s `accepted_hashes`, 2026-09-28) -- a record must match at
    least one.

    `load_dspark_taps_record` now mmaps by default (2026-09-28 OOM fix):
    `taps`/`final_hidden` stay memory-mapped, reclaimable OS page-cache
    pages rather than anonymous RAM, until :func:`build_dspark_batch`
    slices out one batch's rows at a time -- the whole corpus is never
    materialized at once (see `s7_mtp_train.load_mtp_pages` for the MTP
    trainer's own, leaner loader, which needs only `final_hidden`)."""
    from unflatten.model.speculative.dflash import assert_taps_compatible
    from unflatten.model.speculative.dflash_taps_export import load_dspark_taps_record

    accepted_hashes = (
        frozenset({expected_compatibility_hash})
        if isinstance(expected_compatibility_hash, str)
        else frozenset(expected_compatibility_hash)
    )
    pages: list[DSparkPage] = []
    for page_id in page_ids:
        path = taps_dir / f"{page_id}.pt"
        if not path.exists():
            continue
        record = load_dspark_taps_record(path)
        assert_taps_compatible(
            expected_hashes=accepted_hashes,
            current_hash=record.drafter_target_compatibility_hash,
        )
        pages.append(
            DSparkPage(
                page_id=record.page_id,
                taps=record.hidden_taps,
                tokens=torch.tensor(record.token_stream, dtype=torch.long),
                final_hidden=record.final_hidden,
            )
        )
    return pages


def with_final_tap(pages: list[DSparkPage]) -> list[DSparkPage]:
    """Append each page's `final_hidden` row as the last tap
    (`DflashConfig.dspark_final_tap`), keeping the taps' dtype. Joined per
    row by :meth:`DSparkPage.tap_row`: concatenating up front copied the
    whole mmapped export into anonymous RAM (2026-09-28, v1 OOM-killed at
    31 GB anon-rss)."""
    out = []
    for page in pages:
        if page.final_hidden is None:
            raise ValueError(f"page {page.page_id} was exported without final_hidden")
        out.append(replace(page, append_final_tap=True))
    return out


def build_dspark_batch(
    pages: list[DSparkPage], picks: list[tuple[int, int]], *, window: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(taps, prefix_windows, prefix_lengths, target_blocks)` for
    `DSparkDrafter.forward_training_blocks`/`draft_blocks`, one row per
    `(page_index, start)` pick: taps row `start - 1`, the last `window`
    tokens before `start` (left-aligned, zero-padded), and the `block_size`
    tokens from `start` on -- exactly what the runtime drafter sees after
    `start` committed tokens."""
    taps = torch.stack([pages[p].tap_row(start - 1) for p, start in picks])
    windows = torch.zeros(len(picks), window, dtype=torch.long)
    lengths = torch.zeros(len(picks), dtype=torch.long)
    targets = torch.empty(len(picks), block_size, dtype=torch.long)
    for row, (p, start) in enumerate(picks):
        tokens = pages[p].tokens
        prefix = tokens[max(0, start - window) : start]
        windows[row, : len(prefix)] = prefix
        lengths[row] = len(prefix)
        targets[row] = tokens[start : start + block_size]
    return taps, windows, lengths, targets


def final_hidden_blocks_for_picks(
    pages: list[DSparkPage], picks: list[tuple[int, int]], *, block_size: int
) -> torch.Tensor:
    """`(n_picks, block_size, d_model)` final-hidden slices aligned with
    :func:`build_dspark_batch`'s own `targets`: slot `j` reads
    `final_hidden[start - 1 + j]`, the SAME hidden state the tap row for
    that slot already conditions on (row `i` slot `j`'s target label is
    "what would the target choose right after committing the tokens up to
    `start - 1 + j`", i.e. `argmax(output_head(final_hidden[start-1+j]))`)."""
    rows = []
    for p, start in picks:
        page = pages[p]
        if page.final_hidden is None:
            raise ValueError(f"page {page.page_id} has no final_hidden -- re-export with it")
        rows.append(page.final_hidden[start - 1 : start - 1 + block_size])
    return torch.stack(rows)


def target_logits_for_positions(final_hidden: torch.Tensor, output_head: nn.Module) -> torch.Tensor:
    """Composite-head logits at each position of `final_hidden`, through
    the SAME frozen output table the drafter itself embeds
    (`drafter.output_head`) -- the DSpark paper's on-policy teacher signal,
    computed with the exact tied weights rather than a second copy of the
    vocab table. `final_hidden`: `(..., d_model)`. No gradient: the target
    head is frozen and this is a label/loss TARGET, never a training input
    to backprop through."""
    with torch.no_grad():
        weight = next(output_head.parameters())
        return output_head(final_hidden.to(device=weight.device, dtype=weight.dtype))


def build_target_argmax_blocks(
    pages: list[DSparkPage],
    picks: list[tuple[int, int]],
    *,
    block_size: int,
    output_head: nn.Module,
) -> torch.Tensor:
    """The target's own greedy choice at every block slot (owner
    2026-09-28: the DSpark paper trains on the target's own on-policy
    regeneration, not the gold/exact-text token this teacher-forced export
    carries) -- aligned element-for-element with
    :func:`build_dspark_batch`'s own `targets`, so either can be swapped in
    as the CE label without touching the batch's taps/windows/lengths."""
    hidden = final_hidden_blocks_for_picks(pages, picks, block_size=block_size)
    return target_logits_for_positions(hidden, output_head).argmax(dim=-1)


def target_gold_mismatch_rate(
    pages: list[DSparkPage],
    picks: list[tuple[int, int]],
    *,
    block_size: int,
    output_head: nn.Module,
    chunk_size: int = 256,
) -> float:
    """Fraction of block-slot positions where the target's own argmax
    differs from the gold (exact-text) token -- how much a drafter trained
    on gold labels alone would have been steered wrong (owner 2026-09-28).
    `0.0` for an empty pick list (never divides by zero).

    Chunked `chunk_size` picks at a time (2026-09-28 OOM fix): the caller's
    own `picks` can be a 2,048-pick held-out sample, and
    `build_target_argmax_blocks` materializes one `(n_picks * block_size,
    vocab_size)` logits tensor -- at a real ~150k vocabulary that is
    several GB in one unchunked call. Chunking keeps the transient logits
    tensor batch-sized without changing the returned rate."""
    if not picks:
        return 0.0
    mismatches, total = 0, 0
    for offset in range(0, len(picks), chunk_size):
        chunk = picks[offset : offset + chunk_size]
        target = build_target_argmax_blocks(
            pages, chunk, block_size=block_size, output_head=output_head
        )
        gold = torch.stack([pages[p].tokens[start : start + block_size] for p, start in chunk])
        mismatches += int((target.cpu() != gold).sum())
        total += target.numel()
    return mismatches / total if total else 0.0


def top_k_tv_loss(
    student_logits: torch.Tensor, target_logits: torch.Tensor, k: int = 32
) -> torch.Tensor:
    """Total-variation distance between the student's and the target's
    softmax distributions, restricted to the target's own top-`k` logits
    (the DSpark paper's third loss term, previously not implemented here --
    see `dspark_v2.py`'s module docstring for why). `0.5 * sum(|p_student -
    p_target|)` over the `k` indices the TARGET ranks highest -- the
    student's mass on tokens the target considers irrelevant is never
    penalized, only its calibration on the target's own likely set.
    `student_logits`/`target_logits`: `(..., vocab)`, same shape."""
    target_probs = target_logits.float().softmax(dim=-1)
    student_probs = student_logits.float().softmax(dim=-1)
    top_values, top_indices = target_probs.topk(k, dim=-1)
    student_top = student_probs.gather(-1, top_indices)
    return 0.5 * (student_top - top_values).abs().sum(dim=-1).mean()


def all_block_picks(pages: list[DSparkPage], block_size: int) -> list[tuple[int, int]]:
    """Every valid `(page_index, start)` over `pages`."""
    return [(p, s) for p, page in enumerate(pages) for s in page.block_starts(block_size)]


def accepted_lengths(drafted: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Leading matches per row: how many drafted tokens a greedy verify
    against the target's own stream would accept (0..block_size). Every
    caller already moves `drafted` to CPU first; `targets` is moved to
    match here (owner 2026-09-28: `label_source="target"` makes `targets`
    live on `output_head`'s device, e.g. cuda, unlike the CPU-resident gold
    tokens this always compared against before)."""
    matches = (drafted == targets.to(drafted.device)).long()
    return matches.cumprod(dim=1).sum(dim=1)


def resolve_block_targets(
    pages: list[DSparkPage],
    picks: list[tuple[int, int]],
    gold_targets: torch.Tensor,
    *,
    label_source: str,
    output_head: nn.Module,
) -> torch.Tensor:
    """`gold_targets` (`build_dspark_batch`'s own return) unchanged for
    `label_source="gold"`, or the target's own on-policy argmax
    (`build_target_argmax_blocks`) for `label_source="target"` -- owner
    2026-09-28: the DSpark paper trains and measures acceptance against
    the target's own regeneration, not the gold/exact-text token, because
    a drafter trained on gold proposes gold at exactly the hard positions
    where the target would reject it."""
    if label_source == "gold":
        return gold_targets
    if label_source == "target":
        return build_target_argmax_blocks(
            pages, picks, block_size=gold_targets.shape[1], output_head=output_head
        )
    raise ValueError(f"unknown label_source {label_source!r}, expected 'target' or 'gold'")


def measure_dspark_acceptance(
    drafter: DSparkDrafter,
    pages: list[DSparkPage],
    *,
    batch_size: int = 256,
    max_blocks: int | None = None,
    seed: int = 0,
    label_source: str = "gold",
) -> dict[str, Any]:
    """Offline acceptance on held-out taps: draft a block at every valid
    start (or a seeded sample of `max_blocks` of them) with the drafter's
    own sequential greedy walk (`draft_blocks`) and count how many leading
    tokens match the reference stream named by `label_source` -- `"gold"`
    (default here, for backward compatibility with callers that never
    exported `final_hidden`) is the exact-text token; `"target"` (owner
    2026-09-28, what `train_dspark_offline`'s real pipeline actually uses)
    is the target's own on-policy argmax, i.e. what a real verify step
    against the live target would actually accept. Needs only the taps --
    never the target model."""
    block = drafter.config.block_size
    window = drafter.config.dspark_prefix_window
    if window is None:
        raise ValueError("measure_dspark_acceptance requires config.dspark_prefix_window")
    picks = all_block_picks(pages, block)
    if max_blocks is not None and len(picks) > max_blocks:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(picks), generator=generator)[:max_blocks]
        picks = [picks[i] for i in sorted(order.tolist())]
    histogram = [0] * (block + 1)
    was_training = drafter.training
    drafter.eval()
    for offset in range(0, len(picks), batch_size):
        batch_picks = picks[offset : offset + batch_size]
        taps, windows, lengths, gold_targets = build_dspark_batch(
            pages, batch_picks, window=window, block_size=block
        )
        targets = resolve_block_targets(
            pages, batch_picks, gold_targets,
            label_source=label_source, output_head=drafter.output_head,
        )
        # `dspark_seed_first_token`: the primary's own next token -- for
        # `label_source="target"` this is literally the target's own
        # argmax at `start`, matching what real online decode seeds with
        # (`StoreBackedDSparkDrafter`'s `self._primary.next_token`).
        seeds = targets[:, 0] if drafter.config.dspark_seed_first_token else None
        drafted = drafter.draft_blocks(
            hidden_taps=taps, prefix_windows=windows, prefix_lengths=lengths, first_tokens=seeds
        )
        for value in accepted_lengths(drafted.cpu(), targets).tolist():
            histogram[value] += 1
    drafter.train(was_training)
    n_blocks = sum(histogram)
    mean = sum(k * c for k, c in enumerate(histogram)) / n_blocks if n_blocks else 0.0
    return {"n_blocks": n_blocks, "mean_accepted": mean, "histogram": histogram}


def teacher_forced_loss(
    drafter: DSparkDrafter,
    taps: torch.Tensor,
    windows: torch.Tensor,
    lengths: torch.Tensor,
    targets: torch.Tensor,
    *,
    pages: list[DSparkPage] | None = None,
    picks: list[tuple[int, int]] | None = None,
    label_source: str = "gold",
) -> torch.Tensor:
    """Mean cross-entropy of `forward_training_blocks` against the label
    named by `label_source`. `"gold"` (default here, backward compatible
    with every caller that never exported `final_hidden`) uses `targets`
    (`build_dspark_batch`'s exact-text tokens) as both the teacher-forced
    conditioning input AND the CE label -- unchanged prior behaviour.
    `"target"` (owner 2026-09-28: the DSpark paper's on-policy label)
    instead resolves the SAME tensor's role -- conditioning and label
    alike -- to the target's own argmax (`build_target_argmax_blocks`,
    needs `pages`/`picks` and `final_hidden`), the actual self-distillation
    objective: the drafter is teacher-forced on what the target would
    really have said, not the gold token it will be rejected for."""
    if label_source != "gold" and (pages is None or picks is None):
        raise ValueError(f"label_source={label_source!r} requires pages and picks")
    resolved = (
        targets
        if label_source == "gold"
        else resolve_block_targets(
            pages, picks, targets, label_source=label_source, output_head=drafter.output_head
        )
    )
    logits = drafter.forward_training_blocks(
        hidden_taps=taps, prefix_windows=windows, prefix_lengths=lengths, target_blocks=resolved
    )
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        resolved.to(logits.device).reshape(-1),
    )


def trainable_state(drafter: nn.Module) -> dict[str, torch.Tensor]:
    """The drafter's own weights, without the target's frozen tables."""
    return {k: v for k, v in drafter.state_dict().items() if k not in _FROZEN_TABLE_KEYS}


def train_dspark_offline(
    *,
    drafter: DSparkDrafter,
    train_pages: list[DSparkPage],
    eval_pages: list[DSparkPage],
    steps: int,
    batch_blocks: int,
    lr: float,
    warmup_steps: int,
    eval_every: int,
    eval_max_blocks: int,
    checkpoint_dir: Path,
    ratification_record: Path | None,
    seed: int = 0,
    log: Any = print,
    label_source: str = "target",
) -> dict[str, Any]:
    """The real (ratified) offline DSpark training loop: AdamW over the
    drafter's trainable weights, uniformly sampled `(page, start)` blocks,
    teacher-forced self-distillation loss, linear warmup then cosine decay
    to 10 % of `lr`. Every `eval_every` steps: held-out teacher-forced loss
    and offline acceptance, appended to `checkpoint_dir/curve.jsonl`, and a
    resumable `last.pt` (optimizer + sampler state) plus `step-<n>.pt`.
    Refuses before any gradient without a matching ratification record for
    `dflash_speculator`. Never touches the target model.

    `label_source` (owner 2026-09-28, default `"target"`): the DSpark
    paper trains on the target's own on-policy regeneration
    (`resolve_block_targets`), not the gold/exact-text token this
    teacher-forced export carries -- a drafter trained on gold proposes
    gold at exactly the hard positions where the target would reject it.
    `"target"` requires every page's `final_hidden` (raises otherwise, via
    `build_target_argmax_blocks`). The held-out `target_gold_mismatch_rate`
    -- how often the two would have disagreed -- is computed once up front
    and returned in the report regardless of which source trains."""
    import math
    import time

    record = require_ratification(
        child=_CHILD, dry_run=False, ratification_record=ratification_record
    )
    block = drafter.config.block_size
    window = drafter.config.dspark_prefix_window
    if window is None:
        raise ValueError("train_dspark_offline requires config.dspark_prefix_window")
    picks = all_block_picks(train_pages, block)
    if not picks:
        raise ValueError("no trainable blocks in train_pages")
    params = [p for p in drafter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt"
    generator = torch.Generator().manual_seed(seed)
    start_step = 0
    if last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        drafter.load_state_dict(payload["trainable_state"], strict=False)
        optimizer.load_state_dict(payload["optimizer_state"])
        generator.set_state(payload["generator_state"])
        start_step = payload["step"]
        log(f"[train] resumed from {last_path} at step {start_step}")
    eval_sample = all_block_picks(eval_pages, block)
    fixed_generator = torch.Generator().manual_seed(seed + 1)
    eval_loss_picks = [
        eval_sample[i]
        for i in torch.randperm(len(eval_sample), generator=fixed_generator)[:2048].tolist()
    ]
    config_payload = {
        field: getattr(drafter.config, field) for field in drafter.config.__dataclass_fields__
    }

    def save(step: int, name: str, *, resume_state: bool) -> None:
        payload: dict[str, Any] = {
            "step": step,
            "config": config_payload,
            "trainable_state": trainable_state(drafter),
            "ratification_decision_id": record.decision_id if record else None,
        }
        if resume_state:  # only `last.pt`: optimizer state is 2x the weights
            payload["optimizer_state"] = optimizer.state_dict()
            payload["generator_state"] = generator.get_state()
        tmp = checkpoint_dir / f"{name}.tmp"
        torch.save(payload, tmp)
        tmp.replace(checkpoint_dir / name)

    mismatch_rate = (
        target_gold_mismatch_rate(
            eval_pages, eval_loss_picks, block_size=block, output_head=drafter.output_head
        )
        if label_source == "target"
        else None
    )
    if mismatch_rate is not None:
        log(f"[train] held-out target-argmax vs gold mismatch rate: {mismatch_rate:.4f}")

    def evaluate(step: int, running: list[float], started: float) -> dict[str, Any]:
        drafter.eval()
        with torch.no_grad():
            losses = []
            for offset in range(0, len(eval_loss_picks), 256):
                eval_picks = eval_loss_picks[offset : offset + 256]
                batch = build_dspark_batch(
                    eval_pages, eval_picks, window=window, block_size=block,
                )
                losses.append(
                    float(
                        teacher_forced_loss(
                            drafter, *batch,
                            pages=eval_pages, picks=eval_picks, label_source=label_source,
                        )
                    )
                )
        acceptance = measure_dspark_acceptance(
            drafter, eval_pages, max_blocks=eval_max_blocks, seed=seed, label_source=label_source
        )
        drafter.train()
        row = {
            "step": step,
            "train_loss": sum(running) / len(running) if running else None,
            "eval_loss": sum(losses) / len(losses) if losses else None,
            "eval_mean_accepted": acceptance["mean_accepted"],
            "eval_histogram": acceptance["histogram"],
            "label_source": label_source,
            "lr": lr_at(step),
            "elapsed_s": round(time.perf_counter() - started, 1),
        }
        with (checkpoint_dir / "curve.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        log(f"[train] {json.dumps(row)}")
        return row

    drafter.train()
    started = time.perf_counter()
    running: list[float] = []
    last_row: dict[str, Any] = {}
    if start_step == 0:
        last_row = evaluate(0, running, started)
    for step in range(start_step, steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        chosen = torch.randint(len(picks), (batch_blocks,), generator=generator).tolist()
        step_picks = [picks[i] for i in chosen]
        batch = build_dspark_batch(train_pages, step_picks, window=window, block_size=block)
        loss = teacher_forced_loss(
            drafter, *batch, pages=train_pages, picks=step_picks, label_source=label_source
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        running.append(float(loss.detach()))
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            last_row = evaluate(step + 1, running, started)
            running = []
            save(step + 1, "last.pt", resume_state=True)
            save(step + 1, f"step-{step + 1}.pt", resume_state=False)
    return {
        "steps": steps,
        "train_blocks": len(picks),
        "last_eval": last_row,
        "label_source": label_source,
        "target_gold_mismatch_rate": mismatch_rate,
        "ratification_decision_id": record.decision_id if record else None,
    }


def _build_dspark_stub_scenario(
    *, hidden_dim: int = 6, d_model: int = 8, vocab_size: int = 16, n_tokens: int = 8
) -> tuple[DSparkDrafter, DSparkTapsRecord, str]:
    config = DflashConfig(
        hidden_dim=hidden_dim,
        d_model=d_model,
        vocab_size=vocab_size,
        block_size=2,
        variant="dspark",
        dspark_hidden_taps=(0, 1, 2),
        markov_head=True,
    )
    frozen_embedding = torch.randn(vocab_size, d_model)
    frozen_head_weight = torch.randn(vocab_size, d_model)
    compatibility_hash = "sha256:" + "1" * 64
    drafter = build_offline_dspark_drafter(
        config,
        frozen_token_embedding_weight=frozen_embedding,
        frozen_output_head_weight=frozen_head_weight,
    )
    token_stream = tuple(int(t) for t in torch.randint(0, vocab_size, (n_tokens,)))
    taps_record = DSparkTapsRecord(
        page_id="stub-page-dspark",
        hidden_taps=torch.randn(n_tokens, len(config.dspark_hidden_taps), hidden_dim),
        token_stream=token_stream,
        drafter_target_compatibility_hash=compatibility_hash,
        dtype="float32",
    )
    return drafter, taps_record, compatibility_hash


def _build_dspark_v2_stub_scenario(
    *, hidden_dim: int = 8, d_model: int = 8, vocab_size: int = 16, n_tokens: int = 10
) -> tuple[DSparkV2Drafter, DSparkTapsRecord, str]:
    config = DflashConfig(
        hidden_dim=hidden_dim,
        d_model=d_model,
        vocab_size=vocab_size,
        block_size=3,
        variant="dspark_v2",
        dspark_hidden_taps=(0, 1, 2),
        markov_head=True,
        dspark_seed_first_token=True,
        dspark_prefix_window=4,
        dspark_v2_n_layers=1,
        dspark_v2_n_heads=2,
        dspark_v2_n_kv_heads=1,
        dspark_v2_head_dim=4,
        dspark_v2_ffn_intermediate=6,
        dspark_v2_markov_rank=4,
    )
    frozen_embedding = torch.randn(vocab_size, d_model)
    frozen_head_weight = torch.randn(vocab_size, d_model)
    compatibility_hash = "sha256:" + "2" * 64
    drafter = build_offline_dspark_v2_drafter(
        config,
        frozen_token_embedding_weight=frozen_embedding,
        frozen_output_head_weight=frozen_head_weight,
    )
    token_stream = tuple(int(t) for t in torch.randint(0, vocab_size, (n_tokens,)))
    taps_record = DSparkTapsRecord(
        page_id="stub-page-dspark-v2",
        hidden_taps=torch.randn(n_tokens, len(config.dspark_hidden_taps), hidden_dim),
        token_stream=token_stream,
        drafter_target_compatibility_hash=compatibility_hash,
        dtype="float32",
    )
    return drafter, taps_record, compatibility_hash


def _build_stub_scenario(
    *, hidden_dim: int = 6, d_model: int = 8, vocab_size: int = 16, n_tokens: int = 6
) -> tuple[DflashDrafter, DflashTapsRecord, str]:
    config = DflashConfig(
        hidden_dim=hidden_dim, d_model=d_model, vocab_size=vocab_size, block_size=2, num_steps=1
    )
    frozen_embedding = torch.randn(vocab_size, d_model)
    frozen_head_weight = torch.randn(vocab_size, d_model)
    compatibility_hash = "sha256:" + "0" * 64
    drafter = build_offline_drafter(
        config,
        frozen_token_embedding_weight=frozen_embedding,
        frozen_output_head_weight=frozen_head_weight,
    )
    token_stream = tuple(int(t) for t in torch.randint(0, vocab_size, (n_tokens,)))
    taps_record = DflashTapsRecord(
        page_id="stub-page",
        hidden_taps=torch.randn(n_tokens, hidden_dim),
        token_stream=token_stream,
        drafter_target_compatibility_hash=compatibility_hash,
        dtype="float32",
    )
    return drafter, taps_record, compatibility_hash


def _taps_binding_hashes(
    tables: dict[str, torch.Tensor], identities: dict[str, Any], prefix_window: int
) -> tuple[str, frozenset[str]]:
    """The taps binding both `dspark` and `dspark_v2` must agree on, since
    the export step is shared and does not know about `--variant` (2026-09
    -28 fix, runbook §0.2): `taps_compatibility_hash` over target identity +
    tap layers only, plus -- for backward compatibility -- the legacy hash
    every record under an existing `taps/`/`taps-tf/` directory is actually
    bound to (`drafter_target_compatibility_hash` over the `dspark` v1
    `DflashConfig`, computed here regardless of which variant is asking).
    Chosen over re-exporting or widening the on-disk record schema: no taps
    file needs to change, only the two hashes a record is checked against.
    Returns `(primary_hash, accepted_hashes)` -- `primary_hash` is the new,
    variant-independent hash (for display/reporting); `accepted_hashes`
    additionally contains the legacy hash so existing records keep loading."""
    from unflatten.model import events_vocab
    from unflatten.model.speculative.dflash import (
        drafter_target_compatibility_hash,
        taps_compatibility_hash,
    )

    policy = load_s7_dflash_policy()
    dspark_taps = tuple(policy["dspark"]["dspark_hidden_taps"])
    v2_taps = tuple(policy["dspark_v2"]["dspark_hidden_taps"])
    if dspark_taps != v2_taps:
        raise ValueError(
            f"dspark ({dspark_taps}) and dspark_v2 ({v2_taps}) hidden_taps diverge -- "
            "the export step taps ONE set of layers for both variants; a divergent "
            "policy needs a real re-export, not a hash reconciliation"
        )
    primary_hash = taps_compatibility_hash(
        target_checkpoint_hash=identities["checkpoint_sha256"],
        tokenizer_identity=identities["tokenizer_artifact_set_hash"],
        structural_vocabulary_hash=events_vocab.TOKEN_MAP_HASH,
        hidden_taps=dspark_taps,
        final_tap=True,
        dtype="bfloat16",
        export_mode="teacher_forced",
    )
    legacy_config = DflashConfig(
        hidden_dim=tables["embedding"].shape[1],
        d_model=tables["embedding"].shape[1],
        vocab_size=tables["embedding"].shape[0],
        block_size=policy["dspark"]["block_size"],
        variant="dspark",
        dspark_hidden_taps=dspark_taps,
        markov_head=policy["dspark"]["markov_head"],
        dspark_prefix_window=prefix_window,
    )
    legacy_hash = drafter_target_compatibility_hash(
        target_checkpoint_hash=identities["checkpoint_sha256"],
        tokenizer_identity=identities["tokenizer_artifact_set_hash"],
        structural_vocabulary_hash=events_vocab.TOKEN_MAP_HASH,
        config=legacy_config,
    )
    return primary_hash, frozenset({primary_hash, legacy_hash})


def taps_binding(
    tables: dict[str, torch.Tensor], identities: dict[str, Any], prefix_window: int
) -> tuple[DflashConfig, str, frozenset[str]]:
    """The DSpark config an export binds its taps records to, and the
    hashes a record may match, from the frozen tables' shape, the bundle
    manifest's identities and `s7_dflash_policy.json`'s `dspark` block.
    Both offline trainers (DSpark and MTP) check records against
    `accepted_hashes`, so neither can train on taps from another target."""
    policy = load_s7_dflash_policy()["dspark"]
    config = DflashConfig(
        hidden_dim=tables["embedding"].shape[1],
        d_model=tables["embedding"].shape[1],
        vocab_size=tables["embedding"].shape[0],
        block_size=policy["block_size"],
        variant="dspark",
        dspark_hidden_taps=tuple(policy["dspark_hidden_taps"]),
        markov_head=policy["markov_head"],
        dspark_prefix_window=prefix_window,
    )
    primary_hash, accepted_hashes = _taps_binding_hashes(tables, identities, prefix_window)
    return config, primary_hash, accepted_hashes


def taps_binding_v2(
    tables: dict[str, torch.Tensor], identities: dict[str, Any], prefix_window: int
) -> tuple[DflashConfig, str, frozenset[str]]:
    """`dspark_v2` variant of :func:`taps_binding`, reading `s7_dflash_
    policy.json`'s `dspark_v2` block for the DRAFTER's own architecture.
    `dspark_v2` taps records are the same `DSparkTapsRecord` shape as
    `dspark` (`(T, n_taps, hidden_dim)` + token stream) written by the exact
    same export step -- `accepted_hashes` is identical to :func:`taps_
    binding`'s, so a `dspark_v2` trainer accepts the same on-disk records a
    `dspark` trainer already does (2026-09-28 fix, runbook §0.2)."""
    policy = load_s7_dflash_policy()["dspark_v2"]
    config = DflashConfig(
        hidden_dim=tables["embedding"].shape[1],
        d_model=tables["embedding"].shape[1],
        vocab_size=tables["embedding"].shape[0],
        block_size=policy["block_size"],
        variant="dspark_v2",
        dspark_hidden_taps=tuple(policy["dspark_hidden_taps"]),
        markov_head=policy["markov_head"],
        dspark_seed_first_token=policy["dspark_seed_first_token"],
        dspark_context_last=policy.get("dspark_context_last", False),
        dspark_prefix_window=prefix_window,
        dspark_v2_n_layers=policy["dspark_v2_n_layers"],
        dspark_v2_n_heads=policy["dspark_v2_n_heads"],
        dspark_v2_n_kv_heads=policy["dspark_v2_n_kv_heads"],
        dspark_v2_head_dim=policy["dspark_v2_head_dim"],
        dspark_v2_ffn_intermediate=policy["dspark_v2_ffn_intermediate"],
        dspark_v2_markov_rank=policy["dspark_v2_markov_rank"],
    )
    primary_hash, accepted_hashes = _taps_binding_hashes(tables, identities, prefix_window)
    return config, primary_hash, accepted_hashes


def _run_real_dspark_v2(args: argparse.Namespace) -> dict[str, Any]:
    """`dspark_v2` variant of :func:`_run_real_dspark`: same taps/frozen
    -tables/bundle-identities inputs, `train_dspark_v2_offline` (or
    :func:`measure_dspark_v2_acceptance`) instead. Architecture dims come
    from `s7_dflash_policy.json`'s `dspark_v2` block, not CLI flags -- DEC
    -047 condition 4, same discipline as `_run_real_dspark`."""
    if args.dry_run:
        raise ValueError("--dry-run runs the stub step only; drop --taps-dir or --dry-run")
    for name in ("pages_json", "frozen_tables", "bundle_manifest"):
        if getattr(args, name) is None:
            raise ValueError(f"real mode requires --{name.replace('_', '-')}")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tables = torch.load(args.frozen_tables, map_location="cpu", weights_only=True)
    identities = json.loads(args.bundle_manifest.read_text(encoding="utf-8"))["identities"]
    selection = json.loads(args.pages_json.read_text(encoding="utf-8"))
    config, expected_hash, accepted_hashes = taps_binding_v2(tables, identities, args.prefix_window)

    def pages_for(ids: list[str]) -> list[DSparkPage]:
        return load_dspark_pages(args.taps_dir, ids, expected_compatibility_hash=accepted_hashes)

    holdout_ids = [row["bundle_id"] for row in resolve_holdout_rows(selection)]
    eval_pages = pages_for(holdout_ids)
    drafter = build_offline_dspark_v2_drafter(
        config,
        frozen_token_embedding_weight=tables["embedding"],
        frozen_output_head_weight=tables["head_weight"],
        frozen_output_head_bias=tables["head_bias"],
    ).to(args.device)
    del tables  # copied into `drafter`'s own frozen tables above; never read after this (2026-09-28 OOM fix)
    if args.measure_acceptance is not None:
        payload = torch.load(args.measure_acceptance, map_location="cpu", weights_only=False)
        drafter.load_state_dict(payload["trainable_state"], strict=False)
        acceptance = measure_dspark_v2_acceptance(
            drafter, eval_pages, label_source=args.label_source
        )
        return {
            "checkpoint": str(args.measure_acceptance),
            "step": payload["step"],
            "eval_pages": len(eval_pages),
            "label_source": args.label_source,
            **acceptance,
        }
    train_ids = [row["bundle_id"] for row in selection["train"]]
    train_pages = pages_for(train_ids)
    trainable = sum(p.numel() for p in drafter.parameters() if p.requires_grad)
    print(
        f"[train] dspark_v2: {len(train_pages)} train pages, {len(eval_pages)} held-out pages, "
        f"{trainable} trainable parameters, compat {expected_hash}",
        flush=True,
    )
    result = train_dspark_v2_offline(
        drafter=drafter,
        train_pages=train_pages,
        eval_pages=eval_pages,
        steps=args.steps,
        batch_blocks=args.batch_blocks,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        eval_max_blocks=args.eval_max_blocks,
        checkpoint_dir=args.checkpoint_dir or args.out.parent / "checkpoints",
        ratification_record=args.ratification_record,
        seed=args.seed,
        log=lambda line: print(line, flush=True),
        label_source=args.label_source,
        tv_weight=args.tv_weight,
    )
    return {
        **result,
        "train_pages": len(train_pages),
        "eval_pages": len(eval_pages),
        "trainable_parameters": trainable,
        "compatibility_hash": expected_hash,
    }


def _run_real_dspark(args: argparse.Namespace) -> dict[str, Any]:
    """Real mode of :func:`main`: taps + frozen tables + bundle identities
    in, a trained (or measured) drafter out. Block size, taps and Markov
    head come from `s7_dflash_policy.json`'s `dspark` block."""
    if args.variant != "dspark":
        raise ValueError("real offline training via _run_real_dspark is for --variant dspark only")
    if args.dry_run:
        raise ValueError("--dry-run runs the stub step only; drop --taps-dir or --dry-run")
    for name in ("pages_json", "frozen_tables", "bundle_manifest"):
        if getattr(args, name) is None:
            raise ValueError(f"real mode requires --{name.replace('_', '-')}")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tables = torch.load(args.frozen_tables, map_location="cpu", weights_only=True)
    identities = json.loads(args.bundle_manifest.read_text(encoding="utf-8"))["identities"]
    selection = json.loads(args.pages_json.read_text(encoding="utf-8"))
    binding_config, expected_hash, accepted_hashes = taps_binding(
        tables, identities, args.prefix_window
    )
    # Records are bound to the export's config (target identity + taps);
    # `--tap-norm` changes only the drafter built on top of them.
    config = replace(
        binding_config,
        dspark_tap_norm=args.tap_norm,
        dspark_final_tap=args.final_tap,
        dspark_context_last=args.context_last,
        dspark_seed_first_token=args.seed_first_token,
    )

    def pages_for(ids: list[str]) -> list[DSparkPage]:
        loaded = load_dspark_pages(
            args.taps_dir, ids, expected_compatibility_hash=accepted_hashes
        )
        return with_final_tap(loaded) if args.final_tap else loaded
    holdout_ids = [row["bundle_id"] for row in resolve_holdout_rows(selection)]
    eval_pages = pages_for(holdout_ids)
    drafter = build_offline_dspark_drafter(
        config,
        frozen_token_embedding_weight=tables["embedding"],
        frozen_output_head_weight=tables["head_weight"],
        frozen_output_head_bias=tables["head_bias"],
    ).to(args.device)
    del tables  # copied into `drafter`'s own frozen tables above; never read after this (2026-09-28 OOM fix)
    if args.measure_acceptance is not None:
        payload = torch.load(args.measure_acceptance, map_location="cpu", weights_only=False)
        drafter.load_state_dict(payload["trainable_state"], strict=False)
        acceptance = measure_dspark_acceptance(
            drafter, eval_pages, label_source=args.label_source
        )
        return {
            "checkpoint": str(args.measure_acceptance),
            "step": payload["step"],
            "eval_pages": len(eval_pages),
            "label_source": args.label_source,
            **acceptance,
        }
    train_ids = [row["bundle_id"] for row in selection["train"]]
    train_pages = pages_for(train_ids)
    trainable = sum(p.numel() for p in drafter.parameters() if p.requires_grad)
    print(
        f"[train] {len(train_pages)} train pages, {len(eval_pages)} held-out pages, "
        f"{trainable} trainable parameters, compat {expected_hash}",
        flush=True,
    )
    result = train_dspark_offline(
        drafter=drafter,
        train_pages=train_pages,
        eval_pages=eval_pages,
        steps=args.steps,
        batch_blocks=args.batch_blocks,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        eval_max_blocks=args.eval_max_blocks,
        checkpoint_dir=args.checkpoint_dir or args.out.parent / "checkpoints",
        ratification_record=args.ratification_record,
        seed=args.seed,
        log=lambda line: print(line, flush=True),
        label_source=args.label_source,
    )
    return {
        **result,
        "train_pages": len(train_pages),
        "eval_pages": len(eval_pages),
        "trainable_parameters": trainable,
        "compatibility_hash": expected_hash,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RFC004 §37.2 dormant offline dflash_speculator trainer"
    )
    parser.add_argument("--out", type=Path, required=True, help="output JSON report path")
    parser.add_argument(
        "--dry-run", action="store_true", help="run one stub step (the only mode implemented)"
    )
    parser.add_argument("--ratification-record", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant",
        choices=("block_diffusion", "dspark", "dspark_v2"),
        default="block_diffusion",
        help="drafter architecture to train the stub step for",
    )
    real = parser.add_argument_group(
        "real DSpark training (ratified, offline taps only -- never the target model)"
    )
    real.add_argument("--taps-dir", type=Path, default=None, help="enables real mode")
    real.add_argument("--pages-json", type=Path, default=None)
    real.add_argument("--frozen-tables", type=Path, default=None)
    real.add_argument("--bundle-manifest", type=Path, default=None)
    real.add_argument("--prefix-window", type=int, default=64)
    real.add_argument(
        "--seed-first-token",
        action="store_true",
        help="block starts with the primary's own next token (dspark_seed_first_token)",
    )
    real.add_argument(
        "--context-last",
        action="store_true",
        help="context after the prefix, block state read there (dspark_context_last)",
    )
    real.add_argument(
        "--final-tap",
        action="store_true",
        help="append the exported final hidden state as a tap (dspark_final_tap)",
    )
    real.add_argument(
        "--tap-norm", action="store_true", help="LayerNorm each tap (DflashConfig.dspark_tap_norm)"
    )
    real.add_argument("--steps", type=int, default=2000)
    real.add_argument("--batch-blocks", type=int, default=64)
    real.add_argument("--lr", type=float, default=3e-4)
    real.add_argument("--warmup-steps", type=int, default=100)
    real.add_argument("--eval-every", type=int, default=250)
    real.add_argument("--eval-max-blocks", type=int, default=8192)
    real.add_argument("--checkpoint-dir", type=Path, default=None)
    real.add_argument("--device", default="cuda:0")
    real.add_argument(
        "--measure-acceptance",
        type=Path,
        default=None,
        help="checkpoint to measure on EVERY held-out block instead of training",
    )
    real.add_argument(
        "--label-source",
        choices=("target", "gold"),
        default="target",
        help="CE label / acceptance reference (owner 2026-09-28): 'target' (default) is the "
        "target's own on-policy argmax computed from the exported final_hidden; 'gold' is the "
        "teacher-forced export's exact-text token, the paper's known-worse label",
    )
    real.add_argument(
        "--tv-weight",
        type=float,
        default=1.0,
        help="dspark_v2 only: weight of the top-32 total-variation term against the target's "
        "own distribution (DSpark paper's third loss, additive to CE+confidence BCE)",
    )
    args = parser.parse_args(argv)

    if args.taps_dir is not None:
        result = (
            _run_real_dspark_v2(args)
            if args.variant == "dspark_v2"
            else _run_real_dspark(args)
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0

    torch.manual_seed(args.seed)
    if args.variant == "dspark_v2":
        drafter, taps_record, compatibility_hash = _build_dspark_v2_stub_scenario()
        train_step = train_dspark_v2_step_offline
    elif args.variant == "dspark":
        drafter, taps_record, compatibility_hash = _build_dspark_stub_scenario()
        train_step = train_dspark_step_offline
    else:
        drafter, taps_record, compatibility_hash = _build_stub_scenario()
        train_step = train_dflash_step_offline
    optimizer = torch.optim.SGD(
        [p for p in drafter.parameters() if p.requires_grad], lr=1e-3
    )
    result = train_step(
        drafter=drafter,
        taps_record=taps_record,
        optimizer=optimizer,
        expected_compatibility_hash=compatibility_hash,
        dry_run=args.dry_run,
        ratification_record=args.ratification_record,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

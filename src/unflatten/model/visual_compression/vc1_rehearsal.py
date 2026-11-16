"""VC-1 rehearsal under DEC-054, discovery, not the formal VC-1.

DEC-054 (`docs/decisions/DEC-054-vc1-skuska-na-lokalnej-gpu.md`) ratifies a
real gradient for the post-encoder `visual_compressor` ONLY as a local
discovery rehearsal over the frozen S4 v15b step-140000 bundle: primary frozen
(RFC005-ERRATA-007 E5 rule 4, `full_update` = `visual_compressor` only),
pre-encoder in bypass, speculators off, no promotion, no product.

**Why not :class:`adaptive_v0.AdaptiveVisualMemoryCompressor` itself.** As
written it cannot be trained toward a token budget: its only parameters are
the zero-init risk head, the merged token is always the plain group mean, the
budget term in `s7_vc1_train.run_vc1_stub_step` is a Python float outside the
autograd graph, and the zero-init head keeps every token "protected", so no
merge ever happens and the straight-through weight never sees a gradient. Its
irregular contiguous-run merges also had no valid Qwen mRoPE grid:
`candidate_model._reconcile_compressed_placeholders` wrote a `(1, 1, M)`
grid, which `get_vision_position_ids` floors to 0 rows for
`spatial_merge_size == 2` on the real backbone. Round 2 (2026-09-29) fixed
both in place: `candidate_model._inject_compressed_visual` gives merged
tokens their anchor positions from `retention_map` (the rule used here), and
`AdaptiveVisualMemoryCompressor.soft_compression_ratio` puts the budget term
in the graph. The merged token of that profile is still a plain mean.

This rehearsal instead answers the narrower question DEC-054 asks -- can a
LEARNED merge keep ~50 % of the visual tokens without a material quality
loss -- with :class:`GridMergeCompressor`: a fixed rectangular stride merge of
the post-`visual_memory` token grid (1x2 = 50 %, 2x2 = 25 %) whose merged
token is an attention-pooled mean plus a zero-initialized MLP residual over
the whole group. Each merged token keeps the exact mRoPE position of its
top-left source token (:func:`remap_positions`), so the page geometry the
frozen decoder was trained on is preserved, not rescaled. At stride 1x1 it is
bit-identical to bypass; at init with stride > 1 it is the plain group mean
(the untrained control arm). No risk classes, no protected regions -- a
documented deviation from the E5 profile, recorded in every report.
"""

from __future__ import annotations

import contextlib
import functools
import gc
import hashlib
import json
import math
import random
import statistics
import time
import types
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from unflatten.model.visual_compression.contracts import (
    BudgetReport,
    CompressedVisualMemory,
    OriginalTokenAlignment,
    VisualMemory,
    retention_map_to_alignment,
)

REHEARSAL_TAG = "VC-1 rehearsal under DEC-054, discovery, not the formal VC-1"
DECISION_ID = "DEC-054"
RUN_CLASS = "discovery"
DEFAULT_BUNDLE = Path.home() / "unflatten-runs" / "export-v15b-140000-bundle"
DEFAULT_SELECTION_DIR = Path.home() / "unflatten-runs" / "dspark-rehearsal-tf"


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridMergePlan:
    """`groups[g]` lists the original token indices merged into compressed
    token `g` (row-major over the merged grid; an odd edge replicates its
    last row/column so every group has the same size). `retention_map[n]` is
    the group original token `n` belongs to (RFC004-ERRATA-005 E5 semantics).
    `representative[g]` is the group's top-left original token, whose mRoPE
    position the compressed token inherits."""

    groups: Tensor
    retention_map: Tensor
    representative: Tensor
    out_hw: tuple[int, int]


def grid_merge_plan(merged_h: int, merged_w: int, stride: tuple[int, int]) -> GridMergePlan:
    sh, sw = stride
    if sh < 1 or sw < 1 or merged_h < 1 or merged_w < 1:
        raise ValueError(f"invalid grid {merged_h}x{merged_w} or stride {stride}")
    out_h, out_w = math.ceil(merged_h / sh), math.ceil(merged_w / sw)
    rows = torch.arange(out_h).view(-1, 1, 1, 1) * sh + torch.arange(sh).view(1, 1, -1, 1)
    cols = torch.arange(out_w).view(1, -1, 1, 1) * sw + torch.arange(sw).view(1, 1, 1, -1)
    rows = rows.clamp(max=merged_h - 1).expand(out_h, out_w, sh, sw)
    cols = cols.clamp(max=merged_w - 1).expand(out_h, out_w, sh, sw)
    groups = (rows * merged_w + cols).reshape(out_h * out_w, sh * sw)
    r = torch.arange(merged_h).view(-1, 1)
    c = torch.arange(merged_w).view(1, -1)
    retention_map = ((r // sh) * out_w + (c // sw)).reshape(-1)
    return GridMergePlan(
        groups=groups,
        retention_map=retention_map,
        representative=groups[:, 0].clone(),
        out_hw=(out_h, out_w),
    )


class GridMergeCompressor(nn.Module):
    """Rehearsal `visual_compressor`: fixed stride grid merge with a learned
    merge function. Output identical to bypass at stride 1x1; plain group
    mean at init for any stride (zero-init scorer and residual)."""

    def __init__(self, d_model: int, *, stride: tuple[int, int], hidden: int = 1024) -> None:
        super().__init__()
        if d_model <= 0 or hidden <= 0:
            raise ValueError(f"d_model and hidden must be positive, got {d_model}, {hidden}")
        self.d_model = d_model
        self.stride = (int(stride[0]), int(stride[1]))
        group = self.stride[0] * self.stride[1]
        self.score_norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)
        self.delta_norm = nn.LayerNorm(group * d_model)
        self.delta_in = nn.Linear(group * d_model, hidden)
        self.delta_out = nn.Linear(hidden, d_model)
        with torch.no_grad():
            for linear in (self.score, self.delta_out):
                linear.weight.zero_()
                linear.bias.zero_()
        self._plans: dict[tuple[int, int], GridMergePlan] = {}

    @property
    def policy_version(self) -> str:
        return f"vc1-rehearsal-grid-merge-{self.stride[0]}x{self.stride[1]}"

    def plan(self, merged_hw: tuple[int, int]) -> GridMergePlan:
        if merged_hw not in self._plans:
            self._plans[merged_hw] = grid_merge_plan(*merged_hw, self.stride)
        return self._plans[merged_hw]

    def merge(self, tokens: Tensor, plan: GridMergePlan) -> Tensor:
        """`tokens`: `(n_tokens, d_model)` -> `(n_groups, d_model)`, same dtype."""
        groups = plan.groups.to(tokens.device)
        grouped = tokens.float()[groups]  # (M, k, D)
        n_groups, k, d = grouped.shape
        weights = torch.softmax(self.score(self.score_norm(grouped)).squeeze(-1), dim=-1)
        pooled = (weights.unsqueeze(-1) * grouped).sum(dim=1)
        delta = self.delta_out(
            F.gelu(self.delta_in(self.delta_norm(grouped.reshape(n_groups, k * d))))
        )
        return (pooled + delta).to(tokens.dtype)

    def compress(
        self, memory: VisualMemory, *, merged_hw: tuple[int, int]
    ) -> CompressedVisualMemory:
        batch, n_tokens, _ = memory.tokens.shape
        if batch != 1:
            raise ValueError(f"GridMergeCompressor supports batch == 1, got {batch}")
        if merged_hw[0] * merged_hw[1] != n_tokens:
            raise ValueError(f"merged grid {merged_hw} does not hold {n_tokens} visual tokens")
        if bool(memory.key_padding_mask.any()):
            raise ValueError("GridMergeCompressor expects an unpadded single page")
        plan = self.plan(merged_hw)
        coarse = self.merge(memory.tokens[0], plan).unsqueeze(0)
        n_groups = coarse.shape[1]
        coarse_memory = VisualMemory(
            tokens=coarse,
            key_padding_mask=torch.zeros(1, n_groups, dtype=torch.bool, device=coarse.device),
            page_ordinals=memory.page_ordinals,
        )
        return CompressedVisualMemory(
            coarse_memory=coarse_memory,
            protected_detail=memory,
            risk_map=torch.zeros(1, n_tokens, dtype=torch.long),
            retention_map=plan.retention_map,
            budget_report=BudgetReport(
                original_token_count=n_tokens,
                compressed_token_count=n_groups,
                bypassed_region_count=0,
                policy_version=self.policy_version,
            ),
        )

    def expand_alignment(self, retention_map: Tensor) -> OriginalTokenAlignment:
        return retention_map_to_alignment(retention_map)


# ---------------------------------------------------------------------------
# Compressed prefix path through the frozen primary
# ---------------------------------------------------------------------------


def merged_grid_hw(image_grid_thw: Tensor, n_tokens: int) -> tuple[int, int]:
    """Post-spatial-merge `(h, w)` of one image, same derivation as
    `vc0_run.crop_prepared_visual_prefix`."""
    grid_t, grid_h, grid_w = (int(v) for v in image_grid_thw[0].tolist())
    merge_area, remainder = divmod(grid_t * grid_h * grid_w, n_tokens)
    merge = math.isqrt(merge_area)
    if grid_t != 1 or remainder or merge * merge != merge_area or grid_h % merge or grid_w % merge:
        raise ValueError(
            f"visual tokens={n_tokens} and grid_thw={(grid_t, grid_h, grid_w)} do not form "
            "a single-frame rectangular spatial-merge grid"
        )
    return grid_h // merge, grid_w // merge


def remap_positions(
    full_positions: Tensor, *, representative: Tensor, n_visual: int, rope_deltas: Tensor
) -> tuple[Tensor, Tensor]:
    """Positions of the compressed composite range: each compressed visual
    token takes its representative's full-sequence mRoPE id, every later
    position keeps its own full-sequence id. The continuation delta grows by
    the number of removed tokens, so `encode_step`'s `rope_deltas +
    composite_len` lands exactly where the uncompressed sequence would."""
    rep = representative.to(full_positions.device)
    positions = torch.cat(
        [full_positions[:, :, :n_visual][:, :, rep], full_positions[:, :, n_visual:]], dim=-1
    )
    return positions, rope_deltas + (n_visual - rep.shape[0])


def compressed_merge_inputs(
    model: Any, compressor: GridMergeCompressor, batch: Any, *, sink: dict[str, Any] | None = None
) -> tuple[Tensor, Tensor, Tensor | None, int, Tensor | None, int]:
    """Drop-in replacement for `QwenDenseCandidateModel._merge_inputs` with
    the compressor inserted after `visual_memory` (the §15 slot the model's
    own dormant `_apply_post_encoder_compressor` uses). Same 6-tuple.

    Positions come from ONE native `inject_visual` call over the ORIGINAL
    sequence and grid, then :func:`remap_positions`; the compressed visual
    block replaces the contiguous placeholder block at the front of the
    composite range (asserted, not assumed)."""
    from unflatten.model.candidate_model import _native_id_view
    from unflatten.model.visual_compression.contracts import VisualMemory as ContractMemory

    if batch.visual_features.shape[0] != 1:
        raise ValueError("compressed_merge_inputs supports a single page (batch == 1)")
    memory_batch = model.visual_memory(
        batch.visual_features, valid_lengths=batch.visual_valid_lengths
    )
    tokens = memory_batch.memory
    n_visual = tokens.shape[1]
    ids = batch.composite_input_ids
    placeholder = ids[0, 0]
    if not bool((ids[:, :n_visual] == placeholder).all()) or bool(
        (ids[:, n_visual:] == placeholder).any()
    ):
        raise ValueError("visual placeholders must be one contiguous block at the front")
    compressed = compressor.compress(
        ContractMemory(
            tokens=tokens,
            key_padding_mask=memory_batch.key_padding_mask,
            page_ordinals=torch.zeros(1, dtype=torch.long, device=tokens.device),
        ),
        merged_hw=merged_grid_hw(batch.image_grid_thw, n_visual),
    )
    coarse = compressed.coarse_memory.tokens
    n_kept = coarse.shape[1]
    if sink is not None:
        sink["retention_map"] = compressed.retention_map
        sink["budget_report"] = compressed.budget_report
    context = model.document_context_encoder(
        coarse, key_padding_mask=compressed.coarse_memory.key_padding_mask
    )
    full_embeds = model.composite_head.embed_packed(ids)
    injected = model._inject_visual(
        inputs_embeds=full_embeds,
        visual_embeds=tokens[~memory_batch.key_padding_mask],
        native_id_sequence=_native_id_view(ids, model.ranges),
        image_grid_thw=batch.image_grid_thw,
        attention_mask=batch.composite_attention_mask,
    )
    composite_embeds = torch.cat(
        [coarse.to(injected.inputs_embeds.dtype), injected.inputs_embeds[:, n_visual:]], dim=1
    )
    mask = batch.composite_attention_mask
    composite_mask = torch.cat([mask[:, :1].expand(-1, n_kept), mask[:, n_visual:]], dim=1)

    context_len = context.shape[1]
    merged_embeds = torch.cat([context, composite_embeds], dim=1)
    merged_mask = torch.cat(
        [torch.ones(context.shape[:2], dtype=mask.dtype, device=context.device), composite_mask],
        dim=1,
    )
    position_ids = None
    rope_deltas = None
    if injected.position_ids is not None:
        composite_positions, rope_deltas = remap_positions(
            injected.position_ids,
            representative=compressor.plan(
                merged_grid_hw(batch.image_grid_thw, n_visual)
            ).representative,
            n_visual=n_visual,
            rope_deltas=injected.rope_deltas,
        )
        context_positions = (
            torch.arange(context_len, device=merged_embeds.device).view(1, 1, -1).expand(3, 1, -1)
        )
        position_ids = torch.cat([context_positions, composite_positions + context_len], dim=-1)
    return merged_embeds, merged_mask, position_ids, context_len, rope_deltas, n_kept


@contextlib.contextmanager
def compressed_decoding(model: Any, compressor: GridMergeCompressor) -> Iterator[dict[str, Any]]:
    """Route every prefix build of `model` through the compressor for the
    duration: `_merge_inputs` is replaced, and the region-decode visual
    hidden state is expanded back to the ORIGINAL token count through
    `retention_map` (E5 `expand_alignment`), because `decode_regions` reads
    it against the original `image_grid_thw`. Instance attributes only,
    removed on exit."""
    sink: dict[str, Any] = {}
    original_prefix = model.encode_prefix_with_visual_hidden

    def merge_inputs(batch: Any) -> Any:
        return compressed_merge_inputs(model, compressor, batch, sink=sink)

    def prefix_with_visual_hidden(batch: Any, **kwargs: Any) -> Any:
        state, last_hidden, visual_hidden = original_prefix(batch, **kwargs)
        index = sink["retention_map"].to(visual_hidden.device)
        return state, last_hidden, visual_hidden[:, index]

    model._merge_inputs = merge_inputs
    model.encode_prefix_with_visual_hidden = prefix_with_visual_hidden
    try:
        yield sink
    finally:
        del model._merge_inputs
        del model.encode_prefix_with_visual_hidden


# ---------------------------------------------------------------------------
# Frozen-primary backward helpers
# ---------------------------------------------------------------------------


class _Fp8InputGrad(torch.autograd.Function):
    """Forward = the unchanged `Fp8Linear` fast path; backward = gradient
    w.r.t. the INPUT only (`grad @ dequantized W`). The weights are frozen
    buffers and never receive a gradient; activation fp8 rounding is treated
    as identity (straight-through)."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, module: nn.Module) -> Tensor:
        from unflatten.model.quantization.linear import Fp8Linear

        ctx.module = module
        return Fp8Linear.forward(module, x)

    @staticmethod
    def backward(ctx: Any, grad: Tensor) -> tuple[Tensor, None]:
        return grad @ ctx.module.dequantized_weight(grad.dtype), None


@contextlib.contextmanager
def fp8_input_grad(root: nn.Module) -> Iterator[int]:
    """Give every `Fp8Linear` under `root` an input gradient (the fused
    `_scaled_mm` path has none) while grad mode is on. Yields the count."""
    from unflatten.model.quantization.linear import Fp8Linear

    def forward(self: nn.Module, x: Tensor) -> Tensor:
        if torch.is_grad_enabled() and x.requires_grad:
            return _Fp8InputGrad.apply(x, self)
        return Fp8Linear.forward(self, x)

    patched = [m for m in root.modules() if isinstance(m, Fp8Linear)]
    for module in patched:
        module.forward = types.MethodType(forward, module)
    try:
        yield len(patched)
    finally:
        for module in patched:
            del module.forward


@contextlib.contextmanager
def decoder_layer_checkpointing(language_model: nn.Module) -> Iterator[int]:
    """Per-layer activation checkpointing of the frozen HF decoder for the
    duration (only its `GradientCheckpointingLayer`s switch to train mode;
    Qwen3-VL has no dropout, so numerics are unchanged)."""
    from transformers.modeling_layers import GradientCheckpointingLayer

    layers = [m for m in language_model.modules() if isinstance(m, GradientCheckpointingLayer)]
    for layer in layers:
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = functools.partial(checkpoint, use_reentrant=False)
        layer.train()
    try:
        yield len(layers)
    finally:
        for layer in layers:
            layer.gradient_checkpointing = False
            layer.eval()


def primary_fingerprint(modules: Sequence[nn.Module]) -> str:
    """sha256 over every parameter and buffer byte of `modules` (bit-exact
    before/after proof that the primary did not move)."""
    digest = hashlib.sha256()
    for index, module in enumerate(modules):
        for name, tensor in module.state_dict(keep_vars=False).items():
            digest.update(f"{index}:{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
            flat = tensor.detach().reshape(-1).contiguous()
            digest.update(flat.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------


def _target_hidden(model: Any, merged: tuple[Any, ...], n_targets: int) -> Tensor:
    """Dense-adapted hidden state at the `n_targets` teacher-forced target
    positions (composite index `n_visual` is BOS, predicting target 0)."""
    embeds, mask, positions, context_len, _rope, n_visual = merged
    decoded = model._decode(inputs_embeds=embeds, attention_mask=mask, position_ids=positions)
    start = context_len + n_visual
    hidden = decoded.last_hidden_state[:, start : start + n_targets]
    dtype = model.dense_adapter.layers[0].norm.weight.dtype
    adapted = model._apply_expert_adapter(
        hidden.to(dtype), visual_state=embeds[:, :context_len], attention_mask=mask
    )
    return adapted[0]


def _chunk_stats(head: nn.Module, student: Tensor, teacher: Tensor, targets: Tensor) -> Tensor:
    logits_s = head(student).float()
    with torch.no_grad():
        logits_t = head(teacher).float()
        logp_t = torch.log_softmax(logits_t, dim=-1)
    logp_s = torch.log_softmax(logits_s, dim=-1)
    kl = (logp_t.exp() * (logp_t - logp_s)).sum(dim=-1)
    ce = F.cross_entropy(logits_s, targets, reduction="none")
    with torch.no_grad():
        agree = (logits_s.argmax(-1) == logits_t.argmax(-1)).float()
        student_acc = (logits_s.argmax(-1) == targets).float()
        teacher_ce = F.cross_entropy(logits_t, targets, reduction="none")
        teacher_acc = (logits_t.argmax(-1) == targets).float()
    return torch.stack(
        [kl.sum(), ce.sum(), agree.sum(), student_acc.sum(), teacher_ce.sum(), teacher_acc.sum()]
    )


def distill_losses(
    head: nn.Module,
    student: Tensor,
    teacher: Tensor,
    targets: Tensor,
    weights: dict[str, float],
    *,
    chunk: int = 256,
) -> tuple[Tensor, dict[str, float]]:
    """`kl` = KL(teacher || student) over the composite vocabulary (primary
    signal: same output as bypass), `ce` = gold teacher-forced CE, `hidden` =
    relative squared error of the dense-adapted hidden state (what the typed
    heads read). Logits are recomputed per chunk in backward (checkpoint), so
    memory does not scale with `targets x vocabulary`."""
    n = targets.shape[0]
    totals = torch.zeros(6, device=student.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        args = (student[start:end], teacher[start:end], targets[start:end])
        if torch.is_grad_enabled():
            totals = totals + checkpoint(_chunk_stats, head, *args, use_reentrant=False)
        else:
            totals = totals + _chunk_stats(head, *args)
    kl, ce = totals[0] / n, totals[1] / n
    counts = totals.detach()
    hidden = (
        ((student.float() - teacher.float()) ** 2).sum(-1)
        / (teacher.float() ** 2).sum(-1).clamp_min(1e-6)
    ).mean()
    loss = weights["kl"] * kl + weights["ce"] * ce + weights["hidden"] * hidden
    stats = {
        "loss": float(loss.detach()),
        "kl": float(kl.detach()),
        "ce": float(ce.detach()),
        "hidden": float(hidden.detach()),
        "top1_agreement": float(counts[2] / n),
        "student_gold_acc": float(counts[3] / n),
        "teacher_ce": float(counts[4] / n),
        "teacher_gold_acc": float(counts[5] / n),
        "n_targets": n,
    }
    return loss, stats


def distill_step(
    *,
    model: Any,
    compressor: GridMergeCompressor,
    batch: Any,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    weights: dict[str, float],
    max_grad_norm: float = 1.0,
    chunk: int = 256,
) -> dict[str, float]:
    """One optimizer step over `compressor` only: teacher = the frozen
    primary with FULL visual memory (no grad), student = the same primary
    with the compressed memory. Refuses a primary with any trainable
    parameter."""
    from unflatten.model.visual_compression.s7_vc1_train import assert_primary_frozen

    assert_primary_frozen(model)
    n_targets = targets.shape[0]
    with torch.no_grad():
        teacher = _target_hidden(model, model._merge_inputs(batch), n_targets)
    student = _target_hidden(model, compressed_merge_inputs(model, compressor, batch), n_targets)
    loss, stats = distill_losses(
        model.composite_head, student, teacher, targets.to(student.device), weights, chunk=chunk
    )
    optimizer.zero_grad(set_to_none=True)
    if not loss.requires_grad:
        raise RuntimeError(
            "compressor received no gradient: the loss does not depend on any trainable "
            "parameter (a frozen decoder path without backward, e.g. fused fp8 _scaled_mm "
            "without fp8_input_grad)"
        )
    loss.backward()
    if all(p.grad is None for p in compressor.parameters()):
        raise RuntimeError("compressor received no gradient after backward")
    stats["grad_norm"] = float(
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), max_grad_norm)
    )
    optimizer.step()
    assert_primary_frozen(model)
    return stats


@torch.no_grad()
def teacher_forced_eval(
    *, model: Any, compressor: GridMergeCompressor, batch: Any, targets: Tensor, chunk: int = 256
) -> dict[str, float]:
    n_targets = targets.shape[0]
    teacher = _target_hidden(model, model._merge_inputs(batch), n_targets)
    student = _target_hidden(model, compressed_merge_inputs(model, compressor, batch), n_targets)
    _loss, stats = distill_losses(
        model.composite_head,
        student,
        teacher,
        targets.to(student.device),
        {"kl": 1.0, "ce": 0.0, "hidden": 0.0},
        chunk=chunk,
    )
    return stats


# ---------------------------------------------------------------------------
# Real local run (RTX 4070, bundle export-v15b-140000) -- GPU only
# ---------------------------------------------------------------------------


def _log(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def load_pages(selection_dir: Path, split: str, bundle: Path) -> list[dict[str, Any]]:
    """Selected page-0 rows of `split` that have a PNG and gold tokens whose
    extraction identity matches `bundle` (fail-closed, same gate as
    `scripts/dspark_rehearsal.py`)."""
    from scripts.dspark_rehearsal import verify_gold_tokens_identity

    verify_gold_tokens_identity(selection_dir / "gold" / f"identity-{split}.json", bundle)
    selection = json.loads((selection_dir / "selection.json").read_text(encoding="utf-8"))
    wanted = {row["bundle_id"] for row in selection[split]}
    gold: dict[str, list[int]] = {}
    with (selection_dir / "gold" / f"gold-tokens-{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if '"page_index": 0,' not in line:
                continue
            row = json.loads(line)
            if row["page_index"] == 0 and row["document_bundle_id"] in wanted:
                gold[row["document_bundle_id"]] = row["packed_token_ids"]
    pages = []
    for row in selection[split]:
        png = selection_dir / "pages" / row["png_ref"]
        if row["bundle_id"] in gold and png.is_file():
            pages.append({**row, "png_path": str(png), "gold": gold[row["bundle_id"]]})
    return pages


def prepare_page(runtime: Any, png_path: Path) -> Any:
    from unflatten.model.candidate_runtime import PreparedCandidateInput
    from unflatten.model.visual_compression.vc0_run import build_model_safe_input

    with torch.inference_mode():
        prepared = runtime.extract(build_model_safe_input(png_path, 300))
    # Inference tensors cannot enter an autograd graph; a clone outside
    # inference mode is an ordinary tensor with the same values.
    return PreparedCandidateInput(prepared.visual_features.clone(), prepared.image_grid_thw.clone())


def target_ids(gold: Sequence[int], max_targets: int) -> list[int]:
    """Teacher-forced targets of one page. `max_targets == 0` keeps the whole
    page (round 2: the ends of long pages must be trained, the 1 024 cut of
    round 1 never showed them); a positive value keeps the first N."""
    if max_targets < 0:
        raise ValueError(f"max_targets must be >= 0 (0 = full page), got {max_targets}")
    return list(gold) if max_targets == 0 else list(gold[:max_targets])


def train_page_at(order: Sequence[Any], step: int, *, offset: int) -> Any:
    """Page for local `step` of a run initialised from a checkpoint that
    already consumed `offset` pages of the same `order`: unseen pages first,
    then wrap around."""
    return order[(offset + step) % len(order)]


def teacher_forced_batch(
    runtime: Any, prepared: Any, gold: Sequence[int], max_targets: int
) -> tuple[Any, Tensor]:
    from unflatten.model import events_vocab
    from unflatten.model.candidate_model import TeacherForcedBatch
    from unflatten.model.composite_vocab import QwenRef, StructuralRef

    device = prepared.visual_features.device
    targets = target_ids(gold, max_targets)
    placeholder = runtime.vocabulary.pack(QwenRef(runtime.candidate.image_token_id))
    bos = runtime.vocabulary.pack(
        StructuralRef(local_id=events_vocab.TOKEN_IDS[events_vocab.END_DOCUMENT])
    )
    ids = [placeholder] * prepared.visual_features.shape[1] + [bos, *targets[:-1]]
    batch = TeacherForcedBatch(
        visual_features=prepared.visual_features,
        composite_input_ids=torch.tensor([ids], dtype=torch.long, device=device),
        composite_attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device),
        image_grid_thw=prepared.image_grid_thw,
    )
    return batch, torch.tensor(targets, dtype=torch.long, device=device)


def freeze_primary(runtime: Any) -> list[nn.Module]:
    """Freeze every primary parameter and return the modules whose bytes
    :func:`primary_fingerprint` covers (candidate model + Qwen decoder + tower)."""
    from unflatten.model.quantization.quantize import decoder_module, vision_module

    runtime.model.requires_grad_(False)
    for parameter in runtime.candidate.named_parameters.values():
        parameter.requires_grad_(False)
    return [runtime.model, decoder_module(runtime.candidate), vision_module(runtime.candidate)]


def _lr_at(step: int, *, base: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def _peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def run_tf_eval(
    runtime: Any, compressor: GridMergeCompressor, pages: list[dict[str, Any]], max_targets: int
) -> dict[str, Any]:
    rows = []
    for page in pages:
        prepared = prepare_page(runtime, Path(page["png_path"]))
        batch, targets = teacher_forced_batch(runtime, prepared, page["gold"], max_targets)
        rows.append(
            teacher_forced_eval(
                model=runtime.model, compressor=compressor, batch=batch, targets=targets
            )
        )
    keys = ("kl", "ce", "teacher_ce", "top1_agreement", "student_gold_acc", "teacher_gold_acc")
    return {key: round(statistics.mean(r[key] for r in rows), 5) for key in keys} | {
        "n_pages": len(rows)
    }


def save_checkpoint(
    path: Path, *, compressor: nn.Module, optimizer: Any, step: int, config: dict[str, Any]
) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "compressor": compressor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
        },
        tmp,
    )
    tmp.replace(path)


def run_train(args: Any, *, record: Any) -> int:
    """Real local VC-1 rehearsal training loop with progress, resume and
    periodic held-out teacher-forced eval. Caller has already passed the
    DEC-054 authorization gate (`s7_vc1_train.require_local_rehearsal_authorization`)."""
    from unflatten.model.quantization.quantize import decoder_module
    from unflatten.model.visual_compression.vc0_run import load_runtime

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train-log.jsonl"
    ckpt_path = out / "compressor.ckpt.pt"
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    config = {
        "tag": REHEARSAL_TAG,
        "decision_id": record.decision_id,
        "run_class": RUN_CLASS,
        "bundle": str(args.bundle),
        "precision": args.precision,
        "stride": list(args.stride),
        "hidden": args.hidden,
        "lr": args.lr,
        "warmup": args.warmup,
        "total_steps": args.total_steps,
        "max_targets": args.max_targets,
        "weights": {"kl": args.w_kl, "ce": args.w_ce, "hidden": args.w_hidden},
        "seed": args.seed,
        "train_limit": args.train_limit,
        "val_tf_pages": args.val_tf_pages,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
    }
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    runtime = load_runtime(args.bundle, args.device, args.precision, "bf16")
    fingerprint_modules = freeze_primary(runtime)
    fingerprint_before = primary_fingerprint(fingerprint_modules)
    model = runtime.model
    language_model = decoder_module(runtime.candidate)
    compressor = GridMergeCompressor(
        model.d_model, stride=tuple(args.stride), hidden=args.hidden
    ).to(args.device)
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=args.lr, weight_decay=0.0)
    step = 0
    order_offset = 0
    if args.init_checkpoint is not None and not ckpt_path.is_file():
        # Round 2: continue a previous run's compressor AND optimizer moments
        # in a fresh output directory with a fresh lr schedule; the page order
        # is the same seeded permutation, continued after the pages it saw.
        init = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        if list(init["config"]["stride"]) != list(args.stride):
            raise ValueError(
                f"init checkpoint stride {init['config']['stride']} != --stride {args.stride}"
            )
        compressor.load_state_dict(init["compressor"])
        optimizer.load_state_dict(init["optimizer"])
        order_offset = int(init["step"])
        config["order_offset"] = order_offset
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        compressor.load_state_dict(state["compressor"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        order_offset = int(state["config"].get("order_offset", 0))
        config["order_offset"] = order_offset
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        _log(log_path, {"event": "resume", "step": step})

    train_pages = load_pages(args.selection_dir, "train", args.bundle)
    order = random.Random(args.seed).sample(train_pages, len(train_pages))
    if args.train_limit:
        order = order[: args.train_limit]
    val_pages = load_pages(args.selection_dir, "val_iid", args.bundle)[: args.val_tf_pages]
    _log(
        log_path,
        {
            "event": "start",
            "step": step,
            "train_pages": len(order),
            "val_tf_pages": len(val_pages),
            "fingerprint": fingerprint_before,
            **config,
        },
    )
    weights = config["weights"]

    def evaluate(label: str) -> None:
        compressor.eval()
        started = time.perf_counter()
        result = run_tf_eval(runtime, compressor, val_pages, args.max_targets)
        compressor.train()
        _log(
            log_path,
            {
                "event": "val_tf",
                "label": label,
                "step": step,
                "wall_s": round(time.perf_counter() - started, 1),
                **result,
            },
        )

    if step == 0 and val_pages:
        evaluate("step0_untrained_mean_merge")
    compressor.train()
    skipped = 0
    with fp8_input_grad(language_model), decoder_layer_checkpointing(language_model):
        while step < args.total_steps:
            if deadline is not None and time.time() > deadline:
                _log(log_path, {"event": "deadline", "step": step})
                break
            page = train_page_at(order, step, offset=order_offset)
            for group in optimizer.param_groups:
                group["lr"] = _lr_at(step, base=args.lr, warmup=args.warmup, total=args.total_steps)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                prepared = prepare_page(runtime, Path(page["png_path"]))
                encoder_s = time.perf_counter() - started
                batch, targets = teacher_forced_batch(
                    runtime, prepared, page["gold"], args.max_targets
                )
                stats = distill_step(
                    model=model,
                    compressor=compressor,
                    batch=batch,
                    targets=targets,
                    optimizer=optimizer,
                    weights=weights,
                )
            except torch.cuda.OutOfMemoryError as exc:
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                _log(
                    log_path,
                    {
                        "event": "oom_skip",
                        "step": step,
                        "bundle_id": page["bundle_id"],
                        "gold_len": len(page["gold"]),
                        "error": str(exc)[:200],
                    },
                )
                step += 1
                continue
            step += 1
            n_visual = int(prepared.visual_features.shape[1])
            _log(
                log_path,
                {
                    "event": "step",
                    "step": step,
                    "gold_len": len(page["gold"]),
                    "lr": optimizer.param_groups[0]["lr"],
                    **{k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()},
                    "visual_tokens": n_visual,
                    "kept_tokens": compressor.plan(
                        merged_grid_hw(prepared.image_grid_thw, n_visual)
                    ).groups.shape[0],
                    "encoder_s": round(encoder_s, 2),
                    "step_s": round(time.perf_counter() - started, 2),
                    "peak_mib": round(_peak_mib()),
                },
            )
            if step % args.ckpt_every == 0:
                save_checkpoint(
                    ckpt_path, compressor=compressor, optimizer=optimizer, step=step, config=config
                )
            if args.eval_every and step % args.eval_every == 0 and val_pages:
                evaluate(f"step{step}")
    save_checkpoint(ckpt_path, compressor=compressor, optimizer=optimizer, step=step, config=config)
    if val_pages:
        evaluate(f"final_step{step}")
    fingerprint_after = primary_fingerprint(fingerprint_modules)
    if fingerprint_after != fingerprint_before:
        from unflatten.model.visual_compression.s7_vc1_train import PrimaryFrozenViolation

        raise PrimaryFrozenViolation("primary bytes changed during the VC-1 rehearsal")
    report = {
        **config,
        "final_step": step,
        "oom_skipped": skipped,
        "primary_fingerprint_before": fingerprint_before,
        "primary_fingerprint_after": fingerprint_after,
        "primary_unchanged": True,
    }
    (out / "train-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(log_path, {"event": "done", "step": step, "primary_unchanged": True})
    return 0


def chain_outcome(
    runtime: Any, raw: Any, prepared: Any, png_path: Path, work_dir: Path, item_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The diagnostic Word chain's page outcome and AST status for one
    terminated hypothesis -- same flags as `service.docx_stage` -- and the
    sealed page dict (`sealed.page_to_dict`) the paired metrics read."""
    from unflatten.eval.builder_chain.run import process_model_page
    from unflatten.eval.s2_gate.decode import seal_page
    from unflatten.eval.s2_gate.sealed import page_to_dict
    from unflatten.service.docx_stage import InMemoryRasterSource

    page = seal_page(
        document_bundle_id=item_id,
        hypothesis=raw,
        eval_cache_identity=runtime.eval_cache_identity,
        visual_features=prepared.visual_features,
        component_lock_hash=runtime.component_lock_hash,
        selection_policy_hash=runtime.selection_policy_hash,
        decode_fn_identity="decode_free_running",
        max_steps=runtime.decode_policy.max_steps,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    row = process_model_page(
        page,
        vocabulary=runtime.vocabulary,
        tokenizer=runtime.tokenizer,
        raster_source=InMemoryRasterSource(item_id, png_path.read_bytes()),
        render_dir=work_dir,
        headless_defaults=True,
        table_geometry=True,
        story_dedup=True,
        recovered_image_materialization=True,
    )
    envelope = row.envelope
    outcome = {
        "page_outcome": row.page_outcome,
        "ast_status": envelope.axes.ast_status if envelope is not None else None,
        "strict_valid": envelope is not None and envelope.axes.ast_status == "strict_valid",
    }
    return outcome, page_to_dict(page)


def _loop_guard_interventions(raw: Any) -> int | None:
    for issue in getattr(raw, "issues", ()) or ():
        if isinstance(issue, dict) and issue.get("kind") == "loop_guard_v2":
            interventions = issue.get("interventions")
            return len(interventions) if isinstance(interventions, list | tuple) else interventions
    return None


def select_eval_pages(
    pages: Sequence[dict[str, Any]],
    *,
    offset: int,
    limit: int,
    add_longest: int,
    max_gold: int | None = None,
) -> list[dict[str, Any]]:
    """`pages[offset:offset+limit]` plus the `add_longest` pages with the
    longest gold sequence among the rest (round 2: long pages are where the
    round-1 compressor failed, so the cohort must contain them). `max_gold`
    leaves out pages no arm can finish inside the decode step cap."""
    head = list(pages[offset : offset + limit])
    chosen = {p["bundle_id"] for p in head}
    rest = [
        p
        for p in pages
        if p["bundle_id"] not in chosen and (max_gold is None or len(p["gold"]) <= max_gold)
    ]
    longest = sorted(rest, key=lambda p: len(p["gold"]), reverse=True)[:add_longest]
    return head + longest


def parse_arm_spec(spec: str) -> tuple[str, tuple[int, int], Path | None]:
    """`NAME=HxW[:CHECKPOINT]` -> `(name, stride, checkpoint or None)`; no
    checkpoint = the untrained compressor (plain group mean)."""
    name, _, rest = spec.partition("=")
    stride_text, _, checkpoint = rest.partition(":")
    height, _, width = stride_text.partition("x")
    if not name or not height.isdigit() or not width.isdigit() or name == "bypass":
        raise ValueError(f"arm spec must be NAME=HxW[:CHECKPOINT] (not bypass), got {spec!r}")
    return name, (int(height), int(width)), Path(checkpoint) if checkpoint else None


def _eval_arms(args: Any, d_model: int) -> dict[str, GridMergeCompressor | None]:
    specs: list[tuple[str, tuple[int, int], Path | None]] = []
    if args.checkpoint is not None:  # round-1 CLI: untrained + trained at --stride
        stride = tuple(args.stride)
        specs += [("untrained", stride, None), ("trained", stride, Path(args.checkpoint))]
    specs += [parse_arm_spec(spec) for spec in args.arm or ()]
    arms: dict[str, GridMergeCompressor | None] = {"bypass": None}
    for name, stride, ckpt_path in specs:
        compressor = GridMergeCompressor(d_model, stride=stride, hidden=args.hidden)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            compressor.load_state_dict(state["compressor"])
        arms[name] = compressor.to(args.device).eval()
    return arms


def run_eval(args: Any, *, record: Any) -> int:
    """Free-running decode on val_iid pages: bypass and every requested
    compressor arm, same pages. Each row carries text similarity, strict
    validity, loops, timings and table shapes (decoded and gold); the sealed
    page of every terminated decode is kept under `pages/<arm>/` for the
    paired typed-property and region metrics vs bypass."""
    from unflatten.model.visual_compression.vc0_run import (
        _decode_with_phase_timings,
        load_runtime,
        text_similarity,
    )
    from unflatten.model.visual_compression.vc1_eval_metrics import table_shapes

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "eval-rows.jsonl"
    runtime = load_runtime(args.bundle, args.device, args.precision, "bf16")
    freeze_primary(runtime)
    model = runtime.model
    torch.manual_seed(args.seed)
    arms = _eval_arms(args, model.d_model)
    ranges = runtime.vocabulary.ranges
    pages = select_eval_pages(
        load_pages(args.selection_dir, "val_iid", args.bundle),
        offset=args.eval_offset,
        limit=args.limit,
        add_longest=args.add_longest,
        max_gold=runtime.decode_policy.max_steps * 4 // 5,
    )
    done = set()
    if rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["bundle_id"], row["arm"]))
    for index, page in enumerate(pages):
        if all((page["bundle_id"], arm) in done for arm in arms):
            continue
        png_path = Path(page["png_path"])
        started = time.perf_counter()
        prepared = prepare_page(runtime, png_path)
        encoder_s = time.perf_counter() - started
        gold_shapes = table_shapes(page["gold"], ranges)
        for arm, compressor in arms.items():
            if (page["bundle_id"], arm) in done:
                continue
            torch.cuda.reset_peak_memory_stats()
            ctx = (
                contextlib.nullcontext({})
                if compressor is None
                else compressed_decoding(model, compressor)
            )
            try:
                with torch.inference_mode(), ctx as sink:
                    raw, error, phases = _decode_with_phase_timings(runtime, prepared)
            except torch.cuda.OutOfMemoryError as exc:
                # One page that does not fit must not end the whole eval: a
                # counted error row (never a silent skip), memory released.
                _log(rows_path, {
                    "index": index, "bundle_id": page["bundle_id"], "arm": arm,
                    "gold_len": len(page["gold"]), "error": "CUDA_OOM",
                    "detail": str(exc)[:200], "gold_table_shapes": gold_shapes,
                    "table_shapes": None, "text_similarity": None,
                })
                gc.collect()
                torch.cuda.empty_cache()
                continue
            visual = int(prepared.visual_features.shape[1])
            kept = sink["budget_report"].compressed_token_count if sink else visual
            row = {
                "index": index,
                "bundle_id": page["bundle_id"],
                "arm": arm,
                "gold_len": len(page["gold"]),
                "visual_tokens": kept,
                "decode_steps": len(raw.accepted_tokens),
                "prefill_s": round(phases["prefill_s"], 3),
                "step_ms": round(phases["model_step_mean_ms"], 2),
                "encoder_s": round(encoder_s, 3),
                "wall_s": round(encoder_s + phases["wall_s"], 2),
                "peak_mib": round(_peak_mib()),
                # wall clock, to attribute a timing outlier to concurrent load
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": error,
                "loop_guard_interventions": _loop_guard_interventions(raw)
                if error is None
                else None,
                "text_similarity": (
                    round(text_similarity(runtime, raw.accepted_tokens, page["gold"]), 5)
                    if error is None
                    else None
                ),
                "gold_table_shapes": gold_shapes,
                "table_shapes": table_shapes(raw.accepted_tokens, ranges)
                if error is None
                else None,
            }
            if error is None:
                try:
                    outcome, sealed_page = chain_outcome(
                        runtime,
                        raw,
                        prepared,
                        png_path,
                        out / "chain" / arm,
                        f"{page['bundle_id']}-{arm}",
                    )
                    row.update(outcome)
                    page_path = out / "pages" / arm / f"{page['bundle_id']}.json"
                    page_path.parent.mkdir(parents=True, exist_ok=True)
                    page_path.write_text(json.dumps(sealed_page), encoding="utf-8")
                except (ValueError, RuntimeError, KeyError) as exc:
                    row["chain_error"] = f"{type(exc).__name__}: {exc}"[:300]
            _log(rows_path, row)
            # A `DecodeMaxStepsExceeded` hypothesis is the exception itself; its
            # traceback frames still hold the whole KV cache (measured: the next
            # page's bypass prefill OOMed). Release it before the next arm.
            del raw, phases
            gc.collect()
            torch.cuda.empty_cache()
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    summary = summarize_eval(rows, pages=load_sealed_pages(out / "pages"))
    report = {
        "tag": REHEARSAL_TAG,
        "decision_id": record.decision_id,
        "run_class": RUN_CLASS,
        "arms": {name: (None if c is None else list(c.stride)) for name, c in arms.items()},
        "n_pages": len(pages),
        "summary": summary,
    }
    (out / "eval-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def load_sealed_pages(pages_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """`(bundle_id, arm) -> sealed page dict` written by :func:`run_eval`."""
    pages: dict[tuple[str, str], dict[str, Any]] = {}
    if pages_dir.is_dir():
        for path in sorted(pages_dir.glob("*/*.json")):
            pages[(path.stem, path.parent.name)] = json.loads(path.read_text(encoding="utf-8"))
    return pages


#: A page whose text similarity drops more than this below bypass counts as
#: a content-loss page (report-only, not a VC-5 threshold -- none is frozen).
CONTENT_LOSS_DELTA = 0.05


def summarize_eval(
    rows: list[dict[str, Any]], *, pages: dict[tuple[str, str], dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Per-arm medians/rates plus paired deltas vs bypass: text similarity,
    content-loss pages, loops, wall time, table shapes (vs gold and vs
    bypass) and, from the sealed pages, typed-property and region agreement
    with the bypass decode of the same page."""
    from unflatten.model.visual_compression.vc1_eval_metrics import (
        region_agreement,
        typed_agreement,
    )

    pages = pages or {}
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    bypass = {r["bundle_id"]: r for r in by_arm.get("bypass", [])}
    summary: dict[str, Any] = {}
    for arm, arm_rows in sorted(by_arm.items()):
        ok = [r for r in arm_rows if not r.get("error")]
        oom = sum(1 for r in arm_rows if r.get("error") == "CUDA_OOM")
        entry: dict[str, Any] = {
            "n": len(arm_rows),
            "max_steps_exceeded": len(arm_rows) - len(ok) - oom,
            "cuda_oom": oom,
        }
        for key in (
            "visual_tokens",
            "decode_steps",
            "prefill_s",
            "step_ms",
            "wall_s",
            "text_similarity",
        ):
            values = [r[key] for r in ok if r.get(key) is not None]
            if values:
                entry[f"median_{key}"] = round(statistics.median(values), 4)
        sims = [r["text_similarity"] for r in ok if r.get("text_similarity") is not None]
        if sims:
            entry["mean_text_similarity"] = round(statistics.mean(sims), 4)
        entry["strict_valid"] = sum(1 for r in arm_rows if r.get("strict_valid"))
        entry["wall_s_sum"] = round(sum(r["wall_s"] for r in arm_rows if r.get("wall_s")), 2)
        entry["loop_guard_pages"] = sum(1 for r in ok if (r.get("loop_guard_interventions") or 0))
        entry["loop_guard_interventions_sum"] = sum(
            r.get("loop_guard_interventions") or 0 for r in ok
        )
        gold_table_rows = [r for r in arm_rows if r.get("gold_table_shapes")]
        entry["table_pages_gold"] = len(gold_table_rows)
        entry["table_shape_exact_vs_gold"] = sum(
            1 for r in gold_table_rows if r.get("table_shapes") == r["gold_table_shapes"]
        )
        entry["table_count_exact_vs_gold"] = sum(
            1
            for r in gold_table_rows
            if r.get("table_shapes") is not None
            and len(r["table_shapes"]) == len(r["gold_table_shapes"])
        )
        if arm != "bypass":
            _paired_entry(entry, arm_rows, arm, bypass, pages, typed_agreement, region_agreement)
        summary[arm] = entry
    return summary


def _paired_entry(
    entry: dict[str, Any],
    arm_rows: list[dict[str, Any]],
    arm: str,
    bypass: dict[str, dict[str, Any]],
    pages: dict[tuple[str, str], dict[str, Any]],
    typed_agreement: Any,
    region_agreement: Any,
) -> None:
    paired = [(r, bypass[r["bundle_id"]]) for r in arm_rows if r["bundle_id"] in bypass]
    deltas = [
        r["text_similarity"] - b["text_similarity"]
        for r, b in paired
        if not r.get("error")
        and r.get("text_similarity") is not None
        and b.get("text_similarity") is not None
    ]
    if deltas:
        entry["paired_text_similarity_delta_median"] = round(statistics.median(deltas), 4)
        entry["paired_text_similarity_delta_mean"] = round(statistics.mean(deltas), 4)
        entry["paired_text_similarity_delta_min"] = round(min(deltas), 4)
    # a loop (no hypothesis) against a bypass page with text is a content loss too
    entry["content_loss_pages"] = sum(1 for d in deltas if d < -CONTENT_LOSS_DELTA) + sum(
        1
        for r, b in paired
        if r.get("error") and r["error"] != "CUDA_OOM" and not b.get("error")
    )
    wall = [r["wall_s"] - b["wall_s"] for r, b in paired if r.get("wall_s") and b.get("wall_s")]
    if wall:
        entry["paired_wall_s_delta_median"] = round(statistics.median(wall), 3)
        entry["paired_wall_s_delta_sum"] = round(sum(wall), 2)
    table_rows = [(r, b) for r, b in paired if r.get("gold_table_shapes")]
    entry["table_shape_equal_bypass"] = sum(
        1 for r, b in table_rows if r.get("table_shapes") == b.get("table_shapes")
    )
    agree = total = status_agree = status_n = nodes_matched = nodes = 0
    ious: list[int] = []
    length_errors: list[int] = []
    categorical = [0, 0]
    length_exact = [0, 0]
    by_type: dict[str, list[int]] = {}
    empty_page: dict[str, Any] = {}
    for r, _b in paired:
        reference = pages.get((r["bundle_id"], "bypass"))
        if reference is None:
            continue
        candidate = pages.get((r["bundle_id"], arm), empty_page)
        typed = typed_agreement(reference, candidate)
        agree += typed["agree"]
        total += typed["n"]
        nodes_matched += typed["nodes_matched"]
        nodes += typed["nodes"]
        for pooled, key in ((categorical, "categorical"), (length_exact, "length_exact")):
            pooled[0] += typed[key][0]
            pooled[1] += typed[key][1]
        length_errors += typed["length_abs_error_mpt"]
        for node_type, (a, n) in typed["by_node_type"].items():
            bucket = by_type.setdefault(node_type, [0, 0])
            bucket[0] += a
            bucket[1] += n
        regions = region_agreement(reference, candidate)
        status_agree += regions["status_agree"]
        status_n += regions["status_n"]
        ious += regions["iou_ppm"]
    entry["typed_agreement_vs_bypass"] = {
        "agree": agree, "n": total, "rate": round(agree / total, 4) if total else None
    }
    for key, (a, n) in (("typed_categorical_vs_bypass", categorical),
                        ("typed_length_exact_vs_bypass", length_exact)):
        entry[key] = {"agree": a, "n": n, "rate": round(a / n, 4) if n else None}
    if length_errors:
        entry["typed_length_abs_error_mpt_median"] = int(statistics.median(length_errors))
        entry["typed_length_abs_error_mpt_p90"] = sorted(length_errors)[
            int(0.9 * (len(length_errors) - 1))
        ]
    entry["node_match_vs_bypass"] = {
        "matched": nodes_matched, "n": nodes,
        "rate": round(nodes_matched / nodes, 4) if nodes else None,
    }
    entry["typed_agreement_by_node_type"] = {
        t: {"agree": a, "n": n, "rate": round(a / n, 4)} for t, (a, n) in sorted(by_type.items())
    }
    entry["region_status_agreement"] = {
        "agree": status_agree, "n": status_n,
        "rate": round(status_agree / status_n, 4) if status_n else None,
    }
    entry["region_iou_n"] = len(ious)
    if ious:
        entry["region_iou_median_ppm"] = int(statistics.median(ious))
        entry["region_iou_ge_half"] = sum(1 for v in ious if v >= 500_000)

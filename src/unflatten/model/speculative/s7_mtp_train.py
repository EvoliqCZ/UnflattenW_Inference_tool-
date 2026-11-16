"""RFC004 §37.3 / RFC005-ERRATA-007-style dormant trainer: train ONLY
`MtpHeads`, the primary decoder frozen throughout -- same shape as
`visual_compression.s7_vc1_train` (`assert_primary_frozen`/
`assert_state_unchanged`, a ratification-gated entry point, `--dry-run`
the only mode implemented). This module never authorizes a real gradient
step: :func:`train_mtp_step` calls `s7_ratification.require_ratification`
with ``child="MTP"``, so a non-dry-run call refuses exactly like every
other S7 child until an owner-signed record exists.

Supervision is `mtp_aux_loss` against the primary's OWN greedy decode
(the caller supplies `(hidden, targets)` pairs already collected from an
off, acceleration-disabled decode -- this module does not run a decode
itself, only fits the heads on hidden states handed to it).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from unflatten.model.hashing import domain_hash
from unflatten.model.speculative.mtp import MtpHeads, mtp_aux_loss
from unflatten.model.visual_compression.s7_ratification import (
    S7RatificationRecord,
    require_ratification,
)

_S7_MTP_MANIFEST_DOMAIN = "rfc004.s7-mtp-run-manifest/v1"


class PrimaryFrozenViolation(RuntimeError):
    """The primary model is not (or is no longer) fully frozen."""


def assert_primary_frozen(primary: nn.Module) -> None:
    trainable = [name for name, p in primary.named_parameters() if p.requires_grad]
    if trainable:
        raise PrimaryFrozenViolation(
            f"primary model has {len(trainable)} trainable parameter(s), must be "
            f"fully frozen for an MTP step: {trainable[:5]}"
        )


def snapshot_state(primary: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in primary.named_parameters()}


def assert_state_unchanged(before: dict[str, torch.Tensor], primary: nn.Module) -> None:
    for name, p in primary.named_parameters():
        if not torch.equal(before[name], p.detach()):
            raise PrimaryFrozenViolation(
                f"primary parameter {name!r} changed during an MTP step -- the "
                "primary model must never receive a gradient update from MTP training"
            )


def run_mtp_stub_step(
    *,
    mtp_heads: MtpHeads,
    primary: nn.Module,
    hidden_target_pairs: list[tuple[torch.Tensor, tuple[int, ...]]],
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """One optimizer step over `mtp_heads` only, summing `mtp_aux_loss`
    across every `(hidden, targets)` pair given. `primary` is checked
    frozen before and after -- nothing here ever calls `primary.forward`
    or touches its parameters; it is passed in only so both assertions can
    run against the actual model this step is meant to accelerate."""
    assert_primary_frozen(primary)
    before = snapshot_state(primary)

    total_loss = mtp_heads.heads[0].weight.new_zeros(())
    for hidden, targets in hidden_target_pairs:
        logits = mtp_heads(hidden)
        total_loss = total_loss + mtp_aux_loss(logits, targets)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    assert_primary_frozen(primary)
    assert_state_unchanged(before, primary)

    return {"mtp_aux_loss": float(total_loss.detach())}


def build_mtp_run_manifest(
    *, k: int, seed: int, ratification_record: S7RatificationRecord | None
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "s7_mtp_run_manifest_schema_version": "0.1.0",
        "stage": "S7",
        "child": "MTP",
        "k": k,
        "seed": seed,
        "ratification_decision_id": (
            ratification_record.decision_id if ratification_record is not None else None
        ),
    }
    manifest["s7_mtp_run_manifest_hash"] = domain_hash(_S7_MTP_MANIFEST_DOMAIN, manifest)
    return manifest


def train_mtp_step(
    *,
    mtp_heads: MtpHeads,
    primary: nn.Module,
    hidden_target_pairs: list[tuple[torch.Tensor, tuple[int, ...]]],
    optimizer: torch.optim.Optimizer,
    dry_run: bool,
    ratification_record: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """The gated entry point -- refuses a real gradient step without an
    explicit, matching ratification record (`child="MTP"`); `dry_run=True`
    runs the stub step for tests without claiming to be authorized."""
    record = require_ratification(
        child="MTP", dry_run=dry_run, ratification_record=ratification_record
    )
    result = run_mtp_stub_step(
        mtp_heads=mtp_heads,
        primary=primary,
        hidden_target_pairs=hidden_target_pairs,
        optimizer=optimizer,
    )
    manifest = build_mtp_run_manifest(
        k=mtp_heads.k, seed=seed, ratification_record=record
    )
    return {**result, "manifest": manifest, "dry_run": dry_run}


def _build_stub_scenario(
    *, d_model: int = 8, vocab_size: int = 12, k: int = 2
) -> tuple[MtpHeads, nn.Module, list[tuple[torch.Tensor, tuple[int, ...]]]]:
    mtp_heads = MtpHeads(d_model=d_model, vocab_size=vocab_size, k=k)
    primary = nn.Linear(d_model, d_model)
    primary.requires_grad_(False)
    pairs = [
        (torch.randn(1, d_model), (3, 5)),
        (torch.randn(1, d_model), (1, 1)),
    ]
    return mtp_heads, primary, pairs


# --------------------------------------------------------------------------
# Real (ratified, DEC-051) offline mode: MTP heads fitted on the SAME taps
# export the DSpark drafter trains on (DSpark rehearsal 2026-09-27). The
# primary is never loaded here -- it cannot receive a gradient because it is
# not in the process; the export's `final_hidden` rows are its hidden states.
# --------------------------------------------------------------------------

#: Missing future position (end of stream): dropped from the loss, never a
#: numeric target (CLAUDE.md masking rule, same as `mtp_aux_loss`).
_IGNORE = -100


@dataclass(frozen=True, slots=True)
class MtpPage:
    """Lean per-page record for MTP training: MTP only ever reads
    `final_hidden` and the token stream (`mtp_picks`/`build_mtp_batch`/
    `target_argmax_labels_mtp`) -- unlike `s7_dflash_train.DSparkPage`, it
    never carries `hidden_taps` (DSpark's 4 tapped decoder layers). Built by
    :func:`load_mtp_pages`, which drops that tensor immediately after load
    instead of relying on it staying unreferenced."""

    page_id: str
    final_hidden: torch.Tensor
    tokens: torch.Tensor


def load_mtp_pages(
    taps_dir: Path, page_ids: list[str], *, expected_compatibility_hash: str | frozenset[str]
) -> list[MtpPage]:
    """MTP's own page loader (2026-09-28 OOM fix) -- NOT
    `s7_dflash_train.load_dspark_pages`, which keeps the full multi-layer
    `hidden_taps` tensor DSpark needs and MTP does not. Loads each record
    memory-mapped (`load_dspark_taps_record(..., mmap=True)`) so the
    untouched `hidden_taps` tensor is never copied into RAM, then keeps
    only `final_hidden` + token ids and drops the loaded record (its
    `hidden_taps` reference) before moving to the next page: resident size
    for the whole corpus is `final_hidden` alone, roughly a fifth of a full
    DSpark load. A page without a record is skipped (export still
    running); a record bound to another target refuses
    (`assert_taps_compatible`)."""
    from unflatten.model.speculative.dflash import assert_taps_compatible
    from unflatten.model.speculative.dflash_taps_export import load_dspark_taps_record

    accepted_hashes = (
        frozenset({expected_compatibility_hash})
        if isinstance(expected_compatibility_hash, str)
        else frozenset(expected_compatibility_hash)
    )
    pages: list[MtpPage] = []
    for page_id in page_ids:
        path = taps_dir / f"{page_id}.pt"
        if not path.exists():
            continue
        record = load_dspark_taps_record(path, mmap=True)
        assert_taps_compatible(
            expected_hashes=accepted_hashes,
            current_hash=record.drafter_target_compatibility_hash,
        )
        if record.final_hidden is None:
            raise ValueError(f"page {page_id} was exported without final_hidden")
        pages.append(
            MtpPage(
                page_id=record.page_id,
                final_hidden=record.final_hidden,
                tokens=torch.tensor(record.token_stream, dtype=torch.long),
            )
        )
        del record  # drop the reference to `hidden_taps`, MTP's own load never keeps it
    return pages


def mtp_picks(pages: list[Any]) -> list[tuple[int, int]]:
    """Every `(page_index, n)` with `n >= 1` committed tokens, a recorded
    final-hidden row `n - 1` and at least one future token. `n = 0` is
    excluded for the same reason as the DSpark trainer: the prefix pass's
    own hidden state is not in the export."""
    picks: list[tuple[int, int]] = []
    for index, page in enumerate(pages):
        if page.final_hidden is None:
            raise ValueError(f"page {page.page_id} was exported without final_hidden")
        last = min(len(page.tokens) - 1, page.final_hidden.shape[0])
        picks.extend((index, n) for n in range(1, last + 1))
    return picks


def build_mtp_batch(
    pages: list[Any], picks: list[tuple[int, int]], k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(hidden (B, d_model), targets (B, k))`: the final hidden state after
    `n` committed tokens and the next `k` tokens, `-100` past the end."""
    hidden = torch.stack([pages[p].final_hidden[n - 1] for p, n in picks])
    targets = torch.full((len(picks), k), _IGNORE, dtype=torch.long)
    for row, (p, n) in enumerate(picks):
        future = pages[p].tokens[n : n + k]
        targets[row, : len(future)] = future
    return hidden, targets


def target_argmax_labels_mtp(
    pages: list[Any],
    picks: list[tuple[int, int]],
    *,
    k: int,
    head_weight: torch.Tensor,
    head_bias: torch.Tensor | None,
) -> torch.Tensor:
    """`(len(picks), k)` target argmax labels, `_IGNORE` past the stream
    end -- the SAME convention `build_mtp_batch`'s gold `targets` use.
    Owner 2026-09-28 ("fix everything"): the DSpark paper (and this
    project's MTP trainer alike) trains on the target's own on-policy
    regeneration, not the gold/exact-text token a teacher-forced export
    carries. Head slot `j` (predicting position `n + j`) is conditioned on
    `final_hidden[n - 1 + j]`, computed through the target's OWN frozen
    output table (`head_weight`/`head_bias`) -- never `mtp_heads`, which is
    the student under training."""
    labels = torch.full((len(picks), k), _IGNORE, dtype=torch.long)
    hidden_rows: list[torch.Tensor] = []
    positions: list[tuple[int, int]] = []
    for row, (p, n) in enumerate(picks):
        page = pages[p]
        for j in range(k):
            idx = n - 1 + j
            if idx < page.final_hidden.shape[0]:
                hidden_rows.append(page.final_hidden[idx])
                positions.append((row, j))
    if hidden_rows:
        # On `head_weight`'s device: the trainer passes the GPU copy, so the
        # per-step (B*k, vocab) matmul never runs on the CPU (2026-09-28).
        stacked = torch.stack(hidden_rows).to(device=head_weight.device, dtype=torch.float32)
        with torch.no_grad():
            logits = stacked @ head_weight.float().t()
            if head_bias is not None:
                logits = logits + head_bias.float()
            argmax = logits.argmax(dim=-1).cpu()
        for (row, j), value in zip(positions, argmax.tolist()):
            labels[row, j] = value
    return labels


def resolve_mtp_targets(
    pages: list[Any],
    picks: list[tuple[int, int]],
    gold_targets: torch.Tensor,
    *,
    k: int,
    label_source: str,
    head_weight: torch.Tensor | None,
    head_bias: torch.Tensor | None,
) -> torch.Tensor:
    """`gold_targets` (`build_mtp_batch`'s own return) unchanged for
    `label_source="gold"`, or :func:`target_argmax_labels_mtp` for
    `label_source="target"` (requires `head_weight`)."""
    if label_source == "gold":
        return gold_targets
    if label_source == "target":
        if head_weight is None:
            raise ValueError("label_source='target' requires head_weight")
        return target_argmax_labels_mtp(
            pages, picks, k=k, head_weight=head_weight, head_bias=head_bias
        )
    raise ValueError(f"unknown label_source {label_source!r}, expected 'target' or 'gold'")


def mtp_target_gold_mismatch_rate(
    pages: list[Any],
    picks: list[tuple[int, int]],
    *,
    k: int,
    head_weight: torch.Tensor,
    head_bias: torch.Tensor | None,
    chunk_size: int = 256,
) -> float:
    """Fraction of valid (non-`_IGNORE`) head-slot positions where the
    target's own argmax differs from the gold (exact-text) token -- how
    much a drafter trained on gold labels alone would have been steered
    wrong (owner 2026-09-28). `0.0` for an empty pick list or when every
    slot is masked (never divides by zero).

    Chunked `chunk_size` picks at a time (2026-09-28 OOM fix, matching
    `s7_dflash_train.target_gold_mismatch_rate`): `target_argmax_labels_mtp`
    materializes one `(n_picks * k, vocab_size)` logits tensor through the
    target's real ~150k-vocab frozen head -- several GB for a 2,048-pick
    held-out sample in one unchunked call, which is what drove
    `s7_mtp_train --tie-output-head`'s real dry run to ~11 GB peak RSS.
    Chunking keeps the transient logits tensor batch-sized without changing
    the returned rate."""
    if not picks:
        return 0.0
    mismatches, total = 0, 0
    for offset in range(0, len(picks), chunk_size):
        chunk = picks[offset : offset + chunk_size]
        target = target_argmax_labels_mtp(
            pages, chunk, k=k, head_weight=head_weight, head_bias=head_bias
        )
        _hidden, gold = build_mtp_batch(pages, chunk, k)
        valid = gold != _IGNORE
        if not bool(valid.any()):
            continue
        mismatches += int((target[valid] != gold[valid]).sum())
        total += int(valid.sum())
    return mismatches / total if total else 0.0


def warm_start_mtp_heads(
    mtp_heads: MtpHeads, head_weight: torch.Tensor, head_bias: torch.Tensor
) -> None:
    """Initialize every head from the target's own post-bridge output
    projection (the frozen tables the export writes) instead of zero.
    Zero init exists so an installed-but-untrained head changes nothing;
    a head that is about to be TRAINED starts from the vocabulary map the
    target already has, which a 390M-parameter head cannot learn from a
    few hundred pages.

    Not applicable to `tie_output_head=True` heads (`mtp.MtpHeads`
    docstring): those heads are `d_model -> d_model` adapters, not
    `d_model -> vocab_size` projections, and already start at the
    equivalent warm point (the shared frozen output head applied to the
    raw hidden state) via zero-init alone."""
    if mtp_heads.tie_output_head:
        raise ValueError(
            "warm_start_mtp_heads does not apply to tie_output_head=True MtpHeads -- "
            "zero-initialized adapters already start at the shared frozen head's own "
            "readout of the raw hidden state"
        )
    with torch.no_grad():
        for head in mtp_heads.heads:
            head.weight.copy_(head_weight)
            head.bias.copy_(head_bias)


def measure_mtp_acceptance(
    mtp_heads: MtpHeads,
    pages: list[Any],
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    max_blocks: int | None = None,
    seed: int = 0,
    label_source: str = "gold",
    head_weight: torch.Tensor | None = None,
    head_bias: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Offline acceptance on held-out records: each head's argmax at every
    valid `n`, accepted = leading matches against the reference named by
    `label_source` -- `"gold"` (default here, backward compatible) is the
    exact-text token; `"target"` (owner 2026-09-28, requires `head_weight`)
    is the target's own on-policy argmax, i.e. what a real verify step
    against the live target would actually accept (a position past the
    end of the stream never matches, either way)."""
    k = mtp_heads.k
    picks = mtp_picks(pages)
    if max_blocks is not None and len(picks) > max_blocks:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(picks), generator=generator)[:max_blocks]
        picks = [picks[i] for i in sorted(order.tolist())]
    histogram = [0] * (k + 1)
    dtype = mtp_heads.heads[0].weight.dtype
    with torch.no_grad():
        for offset in range(0, len(picks), batch_size):
            batch_picks = picks[offset : offset + batch_size]
            hidden, gold_targets = build_mtp_batch(pages, batch_picks, k)
            targets = resolve_mtp_targets(
                pages, batch_picks, gold_targets,
                k=k, label_source=label_source, head_weight=head_weight, head_bias=head_bias,
            )
            logits = mtp_heads(hidden.to(device=device, dtype=dtype))
            drafted = logits.argmax(dim=-1).cpu()
            accepted = (drafted == targets).long().cumprod(dim=1).sum(dim=1)
            for value in accepted.tolist():
                histogram[value] += 1
    n_blocks = sum(histogram)
    mean = sum(v * c for v, c in enumerate(histogram)) / n_blocks if n_blocks else 0.0
    return {"n_blocks": n_blocks, "mean_accepted": mean, "histogram": histogram}


def train_mtp_heads_offline(
    *,
    mtp_heads: MtpHeads,
    train_pages: list[Any],
    eval_pages: list[Any],
    steps_per_head: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    eval_every: int,
    checkpoint_dir: Path,
    ratification_record: Path | None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    log: Any = print,
    label_source: str = "target",
    head_weight: torch.Tensor | None = None,
    head_bias: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Fit each head in turn on the device (the others wait on the CPU):
    the per-head losses are independent, so this equals joint training
    while holding only one head's optimizer state at a time. AdamW, linear
    warmup then cosine to 10 % of `lr`, `-100`-masked cross-entropy.
    Every `eval_every` steps: held-out loss and top-1 of the head in
    training, appended to `checkpoint_dir/curve.jsonl`. A finished head is
    saved as `head-<i>.pt` and skipped on a rerun (resume at head
    granularity). Refuses before any gradient without a ratification
    record for child `MTP`.

    `label_source` (owner 2026-09-28, default `"target"`): the DSpark
    paper's on-policy label applies here too -- see
    :func:`resolve_mtp_targets`. `"target"` requires `head_weight` (the
    target's own frozen output table, e.g. `tables["head_weight"]`), NOT
    the heads under training. The held-out `target_gold_mismatch_rate` is
    computed once up front and returned in the report regardless."""
    import math
    import time

    record = require_ratification(
        child="MTP", dry_run=False, ratification_record=ratification_record
    )
    k = mtp_heads.k
    picks = mtp_picks(train_pages)
    eval_all = mtp_picks(eval_pages)
    fixed = torch.Generator().manual_seed(seed + 1)
    eval_picks = [
        eval_all[i] for i in torch.randperm(len(eval_all), generator=fixed)[:2048].tolist()
    ]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if head_weight is not None:
        # One fp32 copy on the training device for the per-step target-argmax
        # labels; on the CPU they were a (B*k, d_model) x (d_model, vocab)
        # matmul every step and left the GPU idle (owner's run, 2026-09-28).
        head_weight = head_weight.to(device=device, dtype=torch.float32)
        head_bias = None if head_bias is None else head_bias.to(device=device, dtype=torch.float32)
    mismatch_rate = (
        mtp_target_gold_mismatch_rate(
            eval_pages, eval_picks, k=k, head_weight=head_weight, head_bias=head_bias
        )
        if label_source == "target"
        else None
    )
    if mismatch_rate is not None:
        log(f"[mtp] held-out target-argmax vs gold mismatch rate: {mismatch_rate:.4f}")

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps_per_head - warmup_steps)
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    def evaluate(index: int, head: nn.Module, step: int, running: list[float]) -> float | None:
        head.eval()
        losses, correct, counted = [], 0, 0
        with torch.no_grad():
            for offset in range(0, len(eval_picks), 256):
                batch_picks = eval_picks[offset : offset + 256]
                hidden, gold_targets = build_mtp_batch(eval_pages, batch_picks, k)
                targets = resolve_mtp_targets(
                    eval_pages, batch_picks, gold_targets,
                    k=k, label_source=label_source, head_weight=head_weight, head_bias=head_bias,
                )
                logits = mtp_heads.head_logits(index, hidden.to(device=device, dtype=torch.float32))
                column = targets[:, index].to(device)
                valid = column != _IGNORE
                if not bool(valid.any()):
                    continue
                losses.append(
                    float(nn.functional.cross_entropy(logits, column, ignore_index=_IGNORE))
                )
                correct += int((logits.argmax(dim=-1)[valid] == column[valid]).sum())
                counted += int(valid.sum())
        head.train()
        row = {
            "head": index,
            "step": step,
            "train_loss": sum(running) / len(running) if running else None,
            "eval_loss": sum(losses) / len(losses) if losses else None,
            "eval_top1": correct / counted if counted else None,
            "label_source": label_source,
            "lr": lr_at(step),
            "elapsed_s": round(time.perf_counter() - started, 1),
        }
        with (checkpoint_dir / "curve.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        log(f"[mtp] {json.dumps(row)}")
        return row["eval_top1"]

    if mtp_heads.tie_output_head:
        # Shared and frozen across every head's turn -- moved once, not
        # per-head like the small adapters below (`train_mtp_heads_offline`
        # docstring's "the others wait on the CPU" applies to the per-head
        # adapters, this is the one piece every head's forward needs).
        mtp_heads.output_head.to(device=device, dtype=torch.float32)
    selected_steps: dict[int, int] = {}
    for index, head in enumerate(mtp_heads.heads):
        head_path = checkpoint_dir / f"head-{index}.pt"
        if head_path.exists():
            head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
            log(f"[mtp] head {index} loaded from {head_path}")
            continue
        head.to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
        generator = torch.Generator().manual_seed(seed + 100 + index)
        running: list[float] = []
        # Keep the best held-out top-1 state, step 0 included: under
        # `--tie-output-head` head 0 starts as the target's own readout
        # (top-1 1.0 against target labels) and training only moves it away
        # (owner's run 2026-09-28: 1.0 -> 0.95 by step 250).
        best_top1 = evaluate(index, head, 0, running)
        best_step = 0
        best_state = {name: t.detach().clone() for name, t in head.state_dict().items()}
        for step in range(steps_per_head):
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            chosen = torch.randint(len(picks), (batch_size,), generator=generator).tolist()
            step_picks = [picks[i] for i in chosen]
            hidden, gold_targets = build_mtp_batch(train_pages, step_picks, k)
            targets = resolve_mtp_targets(
                train_pages, step_picks, gold_targets,
                k=k, label_source=label_source, head_weight=head_weight, head_bias=head_bias,
            )
            column = targets[:, index].to(device)
            if not bool((column != _IGNORE).any()):
                continue
            logits = mtp_heads.head_logits(index, hidden.to(device=device, dtype=torch.float32))
            loss = nn.functional.cross_entropy(logits, column, ignore_index=_IGNORE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running.append(float(loss.detach()))
            if (step + 1) % eval_every == 0 or step + 1 == steps_per_head:
                top1 = evaluate(index, head, step + 1, running)
                running = []
                if top1 is not None and (best_top1 is None or top1 > best_top1):
                    best_top1, best_step = top1, step + 1
                    best_state = {n: t.detach().clone() for n, t in head.state_dict().items()}
        head.load_state_dict(best_state)
        selected_steps[index] = best_step
        log(f"[mtp] head {index}: kept step {best_step} (held-out top-1 {best_top1})")
        del optimizer
        head.to("cpu")
        tmp = head_path.with_suffix(".tmp")
        torch.save(head.state_dict(), tmp)
        tmp.replace(head_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if mtp_heads.tie_output_head:
        mtp_heads.output_head.to("cpu")
    return {
        "k": k,
        "steps_per_head": steps_per_head,
        "train_positions": len(picks),
        "selected_steps": selected_steps,
        "label_source": label_source,
        "target_gold_mismatch_rate": mismatch_rate,
        "ratification_decision_id": record.decision_id if record else None,
    }


def _run_real_mtp(args: argparse.Namespace) -> dict[str, Any]:
    """Real mode of :func:`main`: the DSpark taps export's `final_hidden`
    rows + frozen tables + bundle identities in; `k` trained heads (bf16,
    `mtp-heads.pt`) and their held-out acceptance out."""
    from unflatten.model.speculative.s7_dflash_train import (
        resolve_holdout_rows,
        taps_binding,
    )

    if args.dry_run:
        raise ValueError("--dry-run runs the stub step only; drop --taps-dir or --dry-run")
    for name in ("pages_json", "frozen_tables", "bundle_manifest", "checkpoint_dir"):
        if getattr(args, name) is None:
            raise ValueError(f"real mode requires --{name.replace('_', '-')}")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tables = torch.load(args.frozen_tables, map_location="cpu", weights_only=True)
    identities = json.loads(args.bundle_manifest.read_text(encoding="utf-8"))["identities"]
    selection = json.loads(args.pages_json.read_text(encoding="utf-8"))
    _config, expected_hash, accepted_hashes = taps_binding(tables, identities, args.prefix_window)
    vocab, d_model = tables["head_weight"].shape

    def pages(rows: list[dict[str, Any]]) -> list[Any]:
        ids = [row["bundle_id"] for row in rows]
        return load_mtp_pages(args.taps_dir, ids, expected_compatibility_hash=accepted_hashes)

    eval_pages = pages(resolve_holdout_rows(selection))
    if args.measure_acceptance is not None:
        payload = torch.load(args.measure_acceptance, map_location="cpu", weights_only=False)
        tied = payload.get("tie_output_head", False)
        placeholder = (
            torch.zeros(payload["vocab_size"], payload["d_model"]) if tied else None
        )
        mtp_heads = MtpHeads(
            d_model=payload["d_model"],
            vocab_size=payload["vocab_size"],
            k=payload["k"],
            tie_output_head=tied,
            frozen_output_head_weight=placeholder,
            frozen_output_head_bias=(
                torch.zeros(payload["vocab_size"])
                if tied and "output_head.bias" in payload["state"]
                else None
            ),
        )
        mtp_heads.load_state_dict(payload["state"])
        mtp_heads.to(device=args.device, dtype=torch.bfloat16)
        return {
            "checkpoint": str(args.measure_acceptance),
            "eval_pages": len(eval_pages),
            "label_source": args.label_source,
            **measure_mtp_acceptance(
                mtp_heads, eval_pages, device=args.device,
                max_blocks=args.eval_max_blocks,
                label_source=args.label_source,
                head_weight=tables["head_weight"], head_bias=tables["head_bias"],
            ),
        }
    mtp_heads = MtpHeads(
        d_model=d_model,
        vocab_size=vocab,
        k=args.k,
        tie_output_head=args.tie_output_head,
        frozen_output_head_weight=tables["head_weight"] if args.tie_output_head else None,
        frozen_output_head_bias=tables["head_bias"] if args.tie_output_head else None,
    )
    if not args.tie_output_head:
        warm_start_mtp_heads(mtp_heads, tables["head_weight"], tables["head_bias"])
    # Kept (not deleted with `tables`): label_source="target" needs the
    # target's OWN frozen output table for the whole training run, never
    # `mtp_heads` -- those are the student under training.
    head_weight, head_bias = tables["head_weight"], tables["head_bias"]
    del tables
    train_pages = pages(selection["train"])
    parameters = sum(p.numel() for p in mtp_heads.parameters() if p.requires_grad)
    print(
        f"[mtp] {len(train_pages)} train pages, {len(eval_pages)} held-out pages, k={args.k}, "
        f"{parameters} trainable parameters, tie_output_head={args.tie_output_head}, "
        f"compat {expected_hash}",
        flush=True,
    )
    result = train_mtp_heads_offline(
        mtp_heads=mtp_heads,
        train_pages=train_pages,
        eval_pages=eval_pages,
        steps_per_head=args.steps_per_head,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        ratification_record=args.ratification_record,
        device=args.device,
        seed=args.seed,
        log=lambda line: print(line, flush=True),
        label_source=args.label_source,
        head_weight=head_weight,
        head_bias=head_bias,
    )
    mtp_heads.to(dtype=torch.bfloat16)
    final_path = args.checkpoint_dir / "mtp-heads.pt"
    torch.save(
        {
            "k": args.k,
            "d_model": d_model,
            "vocab_size": vocab,
            "tie_output_head": args.tie_output_head,
            "state": mtp_heads.state_dict(),
            "compatibility_hash": expected_hash,
            "ratification_decision_id": result["ratification_decision_id"],
        },
        final_path,
    )
    mtp_heads.to(device=args.device)
    acceptance = measure_mtp_acceptance(
        mtp_heads, eval_pages, device=args.device, max_blocks=args.eval_max_blocks
    )
    return {
        **result,
        "final_checkpoint": str(final_path),
        "train_pages": len(train_pages),
        "eval_pages": len(eval_pages),
        "parameters": parameters,
        "compatibility_hash": expected_hash,
        "heldout_acceptance_bf16": acceptance,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RFC004 §37.3 dormant MTP-heads trainer")
    parser.add_argument("--out", type=Path, required=True, help="output JSON report path")
    parser.add_argument(
        "--dry-run", action="store_true", help="run one stub step (the only mode implemented)"
    )
    parser.add_argument("--ratification-record", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    real = parser.add_argument_group("real offline mode (ratified, taps export only)")
    real.add_argument("--taps-dir", type=Path, default=None, help="enables real mode")
    real.add_argument("--pages-json", type=Path, default=None)
    real.add_argument("--frozen-tables", type=Path, default=None)
    real.add_argument("--bundle-manifest", type=Path, default=None)
    real.add_argument(
        "--prefix-window", type=int, default=64, help="the window the taps export was bound with"
    )
    real.add_argument("--k", type=int, default=3)
    real.add_argument("--steps-per-head", type=int, default=2000)
    real.add_argument("--batch-size", type=int, default=256)
    real.add_argument("--lr", type=float, default=1e-4)
    real.add_argument("--warmup-steps", type=int, default=50)
    real.add_argument("--eval-every", type=int, default=250)
    real.add_argument(
        "--eval-max-blocks",
        type=int,
        default=8192,
        help="cap on positions scored by the final/--measure-acceptance pass over ALL held-out "
        "pages (2026-09-28 OOM fix: unbounded, this pass alone drove peak RSS to ~11 GB on the "
        "real 150-page held-out set -- same cap s7_dflash_train already applies)",
    )
    real.add_argument("--checkpoint-dir", type=Path, default=None)
    real.add_argument("--device", default="cuda:0")
    real.add_argument("--measure-acceptance", type=Path, default=None)
    real.add_argument(
        "--label-source",
        choices=("target", "gold"),
        default="target",
        help="CE label / acceptance reference (owner 2026-09-28): 'target' (default) is the "
        "target's own on-policy argmax computed from the exported final_hidden; 'gold' is the "
        "teacher-forced export's exact-text token, the paper's known-worse label",
    )
    real.add_argument(
        "--tie-output-head",
        action="store_true",
        help=(
            "share one frozen target output projection across all k heads "
            "instead of k independent ones (see mtp.MtpHeads docstring); "
            "incompatible with a checkpoint trained without this flag"
        ),
    )
    args = parser.parse_args(argv)

    if args.taps_dir is not None:
        result = _run_real_mtp(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0

    torch.manual_seed(args.seed)
    mtp_heads, primary, pairs = _build_stub_scenario()
    optimizer = torch.optim.SGD(mtp_heads.parameters(), lr=1e-3)
    result = train_mtp_step(
        mtp_heads=mtp_heads,
        primary=primary,
        hidden_target_pairs=pairs,
        optimizer=optimizer,
        dry_run=args.dry_run,
        ratification_record=args.ratification_record,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

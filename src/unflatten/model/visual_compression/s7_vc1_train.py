"""RFC005-ERRATA-007 E5 VC-1: train ONLY `adaptive_v0`'s post-encoder
compressor, primary model frozen throughout.

§27.11's `L_visual_compressor` distills the compressor against the frozen
primary as teacher (`SealedDerivedProvenance`); this module adds §21's
budget term from the S7 stage policy draft (never a code literal, DEC-047
condition 4). No primary parameter is ever registered with the optimizer
here, and :func:`assert_primary_frozen` / :func:`assert_state_unchanged`
exist so every caller -- this module's own CLI and its tests -- can prove
that, rather than assume it.

A *real* gradient step additionally requires an explicit ratification
record (:mod:`s7_ratification`); only ``--dry-run`` runs today.

Every report this module writes carries its own ``architecture_contract_passed``
verdict (:func:`unflatten.model.visual_compression.architecture_contract.
check_architecture_contract` over the post-step compressor) -- this is the
"VC-1 report" :mod:`s7_vc4_combined` reads before it will let the combined
PRE+POST arm run at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from unflatten.model.hashing import domain_hash
from unflatten.model.visual_compression.adaptive_v0 import (
    IMPLEMENTATION_HASH,
    AdaptiveVisualMemoryCompressor,
)
from unflatten.model.visual_compression.architecture_contract import check_architecture_contract
from unflatten.model.visual_compression.contracts import VisualMemory
from unflatten.model.visual_compression.losses import SealedDerivedProvenance, l_visual_compressor
from unflatten.model.visual_compression.s7_policy import load_s7_stage_policy, s7_stage_policy_hash
from unflatten.model.visual_compression.s7_ratification import (
    S7RatificationRecord,
    S7RatificationRequired,
    require_ratification,
)

_S7_VC1_MANIFEST_DOMAIN = "rfc005.s7-vc1-run-manifest/v1"


class PrimaryFrozenViolation(RuntimeError):
    """The primary model is not (or is no longer) fully frozen."""


def assert_primary_frozen(primary: nn.Module) -> None:
    """Raise unless every parameter of `primary` has `requires_grad == False`.

    RFC005-ERRATA-007 rule 3: "primary weights FROZEN in S7" -- checked
    directly on the module rather than trusted from the caller's claim.
    """
    trainable = [name for name, p in primary.named_parameters() if p.requires_grad]
    if trainable:
        raise PrimaryFrozenViolation(
            f"primary model has {len(trainable)} trainable parameter(s), must be "
            f"fully frozen for a VC-1 step: {trainable[:5]}"
        )


def snapshot_state(primary: nn.Module) -> dict[str, torch.Tensor]:
    """A detached, cloned copy of `primary`'s parameters, for a before/after diff."""
    return {name: p.detach().clone() for name, p in primary.named_parameters()}


def assert_state_unchanged(before: dict[str, torch.Tensor], primary: nn.Module) -> None:
    """Raise if any parameter in `before` differs from `primary`'s current value."""
    for name, p in primary.named_parameters():
        if not torch.equal(before[name], p.detach()):
            raise PrimaryFrozenViolation(
                f"primary parameter {name!r} changed during a VC-1 step -- the "
                "primary model must never receive a gradient update in S7"
            )


def run_vc1_stub_step(
    *,
    compressor: AdaptiveVisualMemoryCompressor,
    primary: nn.Module,
    memory: VisualMemory,
    policy: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    provenance: SealedDerivedProvenance,
) -> dict[str, float]:
    """One optimizer step over `compressor` only. Distillation target is
    `memory.tokens` itself (the frozen primary's own bypass output) -- the
    student side is re-expanded to the same token count via
    `compressed.retention_map` (`l_visual_compressor`'s own docstring:
    "re-expanded to the teacher's original token count") before the
    comparison, so shapes match whether or not any merge happened this step.

    At step 0 (zero-init `risk_head`) every group is a singleton and the
    re-expansion is the identity, so the student exactly equals the teacher
    and `distill_loss` is exactly `0.0` -- bit-identical to bypass, nothing
    to correct yet. `risk_head`'s hard `torch.argmax` still has no gradient
    of its own; what carries a gradient into it is `adaptive_v0.py`'s
    training-only straight-through merge weight (RFC005-ERRATA-007 E5),
    which only produces a nonzero signal once at least one group actually
    merges (`AdaptiveVisualMemoryCompressor.compress` under `self.training`).
    """
    assert_primary_frozen(primary)
    before = snapshot_state(primary)

    compressed = compressor.compress(memory)
    expanded_student = compressed.coarse_memory.tokens[0].index_select(
        0, compressed.retention_map
    ).unsqueeze(0)
    distill_loss = l_visual_compressor(
        expanded_student, memory.tokens.detach(), provenance=provenance
    )

    vc1_policy = policy["vc1"]
    budget_lambda = vc1_policy["budget_lambda_micro"] / 1_000_000
    target_ratio = vc1_policy["target_compression_ratio_micro"] / 1_000_000
    achieved_ratio = compressed.budget_report.compression_ratio
    # In the graph (VC-1 rehearsal 2026-09-28 finding): on the hard
    # `achieved_ratio` float this term never reached a parameter, and a
    # zero-init head (every token protected, distill_loss == 0) never moved.
    soft_ratio = compressor.soft_compression_ratio(memory)
    budget_term = budget_lambda * torch.relu(target_ratio - soft_ratio)

    optimizer.zero_grad()
    (distill_loss + budget_term).backward()
    optimizer.step()

    assert_primary_frozen(primary)
    assert_state_unchanged(before, primary)

    return {
        "distill_loss": float(distill_loss.detach()),
        "budget_term": float(budget_term.detach()),
        "soft_compression_ratio": float(soft_ratio.detach()),
        "achieved_compression_ratio": achieved_ratio,
    }


def build_vc1_run_manifest(
    *,
    policy_hash: str,
    adaptive_v0_policy_hash: str,
    seed: int,
    ratification_record: S7RatificationRecord | None,
) -> dict[str, Any]:
    """This trainer's own manifest identity -- stage `S7`, child `VC-1` --
    never borrowed from S1-S6's `run_manifest` schemas."""
    manifest: dict[str, Any] = {
        "s7_vc1_run_manifest_schema_version": "0.1.0",
        "stage": "S7",
        "child": "VC-1",
        "policy_hash": policy_hash,
        "adaptive_v0_policy_hash": adaptive_v0_policy_hash,
        "seed": seed,
        "ratification_decision_id": (
            ratification_record.decision_id if ratification_record is not None else None
        ),
    }
    manifest["s7_vc1_run_manifest_hash"] = domain_hash(_S7_VC1_MANIFEST_DOMAIN, manifest)
    return manifest


def train_vc1_step(
    *,
    compressor: AdaptiveVisualMemoryCompressor,
    primary: nn.Module,
    memory: VisualMemory,
    optimizer: torch.optim.Optimizer,
    provenance: SealedDerivedProvenance,
    dry_run: bool,
    ratification_record: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """The gated entry point: refuses a real gradient step without an
    explicit, matching ratification record; `dry_run=True` runs the stub
    step for tests without claiming to be authorized."""
    record = require_ratification(
        child="VC-1", dry_run=dry_run, ratification_record=ratification_record
    )
    policy = load_s7_stage_policy()
    result = run_vc1_stub_step(
        compressor=compressor,
        primary=primary,
        memory=memory,
        policy=policy,
        optimizer=optimizer,
        provenance=provenance,
    )
    manifest = build_vc1_run_manifest(
        policy_hash=s7_stage_policy_hash(policy),
        adaptive_v0_policy_hash=compressor.policy.hash(),
        seed=seed,
        ratification_record=record,
    )
    with torch.no_grad():
        gate_report = check_architecture_contract(
            compressor,
            memory,
            implementation_hash=IMPLEMENTATION_HASH,
            policy_hash=compressor.policy.hash(),
        )
    return {
        **result,
        "architecture_contract_passed": gate_report.passed,
        "manifest": manifest,
        "dry_run": dry_run,
    }


#: DEC-054 is the only decision that authorizes a real VC-1 gradient, and only
#: as a local discovery rehearsal (`docs/decisions/DEC-054-vc1-skuska-na-lokalnej-gpu.md`).
LOCAL_REHEARSAL_DECISION_ID = "DEC-054"
LOCAL_REHEARSAL_RUN_CLASS = "discovery"


def require_local_rehearsal_authorization(
    *, ratification_record: Path | None, run_class: str
) -> S7RatificationRecord:
    """Gate for the real local VC-1 rehearsal (`--local-rehearsal`): a valid
    VC-1 ratification record (the same :func:`require_ratification` gate as
    every S7 child) that is specifically DEC-054, and `run_class ==
    "discovery"` -- DEC-054 authorizes nothing else."""
    if run_class != LOCAL_REHEARSAL_RUN_CLASS:
        raise S7RatificationRequired(
            f"the local VC-1 rehearsal is run_class {LOCAL_REHEARSAL_RUN_CLASS!r} only "
            f"(DEC-054), got {run_class!r}"
        )
    record = require_ratification(
        child="VC-1", dry_run=False, ratification_record=ratification_record
    )
    if record.decision_id != LOCAL_REHEARSAL_DECISION_ID:
        raise S7RatificationRequired(
            f"ratification record {ratification_record} is {record.decision_id!r}; the local "
            f"VC-1 rehearsal is authorized only by {LOCAL_REHEARSAL_DECISION_ID}"
        )
    return record


def _build_stub_scenario(*, d_model: int = 8, n_tokens: int = 4) -> tuple[
    AdaptiveVisualMemoryCompressor, nn.Module, VisualMemory, SealedDerivedProvenance
]:
    compressor = AdaptiveVisualMemoryCompressor(d_model)
    primary = nn.Linear(d_model, d_model)
    primary.requires_grad_(False)
    tokens = torch.randn(1, n_tokens, d_model)
    tokens.requires_grad_(True)
    memory = VisualMemory(
        tokens=tokens,
        key_padding_mask=torch.zeros(1, n_tokens, dtype=torch.bool),
        page_ordinals=torch.zeros(1, dtype=torch.long),
    )
    provenance = SealedDerivedProvenance(
        teacher_checkpoint_hash="sha256:" + "0" * 64,
        component_lock_hash="sha256:" + "0" * 64,
        attachment_points=("visual_compressor.coarse_memory",),
        implementation_hash="sha256:" + "0" * 64,
        output_projection_hash="sha256:" + "0" * 64,
    )
    return compressor, primary, memory, provenance


def _local_rehearsal_parser() -> argparse.ArgumentParser:
    from unflatten.model.visual_compression import vc1_rehearsal as rh

    parser = argparse.ArgumentParser(description=f"{rh.REHEARSAL_TAG} (local RTX 4070)")
    parser.add_argument("--local-rehearsal", choices=("train", "eval"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--ratification-record", type=Path, required=True)
    parser.add_argument("--run-class", default=LOCAL_REHEARSAL_RUN_CLASS)
    parser.add_argument("--bundle", type=Path, default=rh.DEFAULT_BUNDLE)
    parser.add_argument("--selection-dir", type=Path, default=rh.DEFAULT_SELECTION_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument("--stride", type=int, nargs=2, default=(1, 2))
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-targets", type=int, default=1024, help="0 = full page targets (no cut)"
    )
    # train
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--total-steps", type=int, default=1500)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-tf-pages", type=int, default=24)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--ckpt-every", type=int, default=50)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="start from another run's compressor + optimizer (fresh schedule, same page order)",
    )
    parser.add_argument("--w-kl", type=float, default=1.0)
    parser.add_argument("--w-ce", type=float, default=0.0)
    parser.add_argument("--w-hidden", type=float, default=1.0)
    # eval
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument(
        "--add-longest", type=int, default=0, help="also evaluate the N longest remaining pages"
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        help="extra compressor arm NAME=HxW[:CHECKPOINT] (repeatable; no checkpoint = untrained)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import sys

    raw_argv = sys.argv[1:] if argv is None else argv
    if "--local-rehearsal" in raw_argv:
        from unflatten.model.visual_compression import vc1_rehearsal as rh

        args = _local_rehearsal_parser().parse_args(raw_argv)
        record = require_local_rehearsal_authorization(
            ratification_record=args.ratification_record, run_class=args.run_class
        )
        if args.local_rehearsal == "train":
            return rh.run_train(args, record=record)
        if args.checkpoint is None and not args.arm:
            raise SystemExit("--local-rehearsal eval requires --checkpoint or --arm")
        return rh.run_eval(args, record=record)

    parser = argparse.ArgumentParser(description="RFC005-ERRATA-007 VC-1 dormant trainer")
    parser.add_argument("--out", type=Path, required=True, help="output JSON report path")
    parser.add_argument(
        "--dry-run", action="store_true", help="run one stub step (the only mode implemented)"
    )
    parser.add_argument("--ratification-record", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(raw_argv)

    torch.manual_seed(args.seed)
    compressor, primary, memory, provenance = _build_stub_scenario()
    optimizer = torch.optim.SGD(compressor.parameters(), lr=1e-3)
    result = train_vc1_step(
        compressor=compressor,
        primary=primary,
        memory=memory,
        optimizer=optimizer,
        provenance=provenance,
        dry_run=args.dry_run,
        ratification_record=args.ratification_record,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

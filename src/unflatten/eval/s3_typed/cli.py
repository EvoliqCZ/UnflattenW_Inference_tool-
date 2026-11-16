"""T9 -- `python -m unflatten.eval.s3_typed.cli`: a teacher-forced typed
report over the val exact-text cache (design §8, plan row 11).

Two arms, two load paths -- the [governance R1 must_fix] this task exists
to satisfy:

- `--checkpoint` (the S3 checkpoint) loads STRICTLY: its state_dict has
  the dense-head keys.
- `--baseline-checkpoint` (the S2 parent, untrained typed heads) loads
  through `s3_train.load_init_weights` -- the §6 declared-missing rule,
  never through `eval/s2_gate/cli.py:320`'s strict load.

Every pin the baseline arm's `load_init_weights` call needs comes from
the S3 run manifest presented via `--s3-run-manifest`, never from a CLI
flag (design §8):

- `pinned` = the manifest's own `init_from` block, verbatim.
- `expect_cache_identity` = the val exact-text cache's own identity with
  `split` forced to `"train"`. Run 1 warm-starts onto the SAME
  release/policy cache this eval's val cache is a sibling of (D1/D2: "the
  same S2 cache"), differing only by `split` -- exactly the relation
  `eval.s2_gate.decode.refuse_unless_eval_cache_identity_ok` already
  encodes. Deriving it from the checkpoint's OWN stored `cache_identity`
  would make `load_init_weights`'s check tautological (it would always
  pass); deriving it from the val cache's identity keeps it an
  independent expectation, able to catch e.g. the wrong checkpoint file.

`--exact-text-cache-dir` is the VAL cache (`.../val_iid`); reused
provenance/ancestry checks and `eval_cache_identity_from_exact_text_cache`
come from `eval.s2_gate.cli`, imported, never copied.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.eval.s2_gate import cli as s2_cli
from unflatten.eval.s3_typed import report as report_mod
from unflatten.model import checkpoint_ancestry as ca
from unflatten.model import s3_train
from unflatten.model import s3_typed_loss as stl
from unflatten.model.candidate_model import typed_head_children
from unflatten.model.s3_run_manifest import load_s3_run_manifest
from unflatten.model.typed_property_heads import BooleanPropertyHead, EnumPropertyHead


class S3TypedEvalRefused(ValueError):
    """The CLI's own inputs are not usable: no `init_from` block to
    warm-start the baseline arm from, or `--out` already exists. A
    refusal, never a repair."""


# ---------------------------------------------------------------------------
# Baseline-arm pins -- from the S3 run manifest only, never from flags
# ---------------------------------------------------------------------------


def resolve_baseline_pins(
    *, s3_run_manifest: Mapping[str, Any], eval_cache_identity: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(pinned, expect_cache_identity)` for the baseline arm's
    `s3_train.load_init_weights` call -- see the module docstring.
    Ignores everything except its two named arguments: no `argparse
    .Namespace` is threaded through this function, so a caller flag
    cannot influence its result even by accident."""
    init_from = s3_run_manifest.get("init_from")
    if init_from is None:
        raise S3TypedEvalRefused(
            "s3_run_manifest has no init_from block -- no baseline parent to "
            "warm-start the baseline arm from"
        )
    pinned = dict(init_from)
    expect_cache_identity = {**dict(eval_cache_identity), "split": "train"}
    return pinned, expect_cache_identity


# ---------------------------------------------------------------------------
# Model loading -- one strict, one declared-missing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmResult:
    model: Any
    checkpoint_id: str
    cache_identity: dict[str, Any]


def load_trained_arm(
    checkpoint_path: Path, model: Any, *, expect_s1_run_manifest_hash: str, device: Any
) -> ArmResult:
    """The S3 checkpoint: STRICT `load_state_dict` (its state_dict carries
    the dense-head keys already), provenance/ancestry checks reused from
    `eval.s2_gate.cli` verbatim."""
    import torch

    checkpoint_sha256 = ca.file_sha256(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    s2_cli.refuse_unless_checkpoint_provenance_ok(
        state, expect_s1_run_manifest_hash=expect_s1_run_manifest_hash
    )
    checkpoint_id = s2_cli.refuse_unless_checkpoint_ancestry_verified(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        embedded_ancestry_hash=state.get("ancestry_hash"),
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return ArmResult(
        model=model, checkpoint_id=checkpoint_id, cache_identity=dict(state["cache_identity"])
    )


def load_baseline_arm(
    checkpoint_path: Path,
    model: Any,
    *,
    pinned: Mapping[str, Any],
    expect_cache_identity: Mapping[str, Any],
) -> ArmResult:
    """The S2-parent baseline arm: `s3_train.load_init_weights` (§6
    declared-missing rule) -- NEVER the strict load `load_trained_arm`
    uses. The untrained dense-head keys are reported by name in
    `InitReport.fresh_keys`, not silently absorbed."""
    report = s3_train.load_init_weights(
        checkpoint_path, model, pinned=pinned, expect_cache_identity=expect_cache_identity
    )
    model.eval()
    return ArmResult(
        model=model,
        checkpoint_id=report.checkpoint_id,
        cache_identity=dict(expect_cache_identity),
    )


# ---------------------------------------------------------------------------
# Per-page evaluation
# ---------------------------------------------------------------------------


def evaluate_arm(
    model: Any,
    dataset: stl.S3TrainingDataset,
    *,
    allowed_head_ids: Sequence[str],
    lineage_by_bundle: Mapping[str, str] | None,
    ranges: Any,
    image_token_id: int,
    device: Any,
    limit: int | None = None,
) -> list[report_mod.PointerReport]:
    """One teacher-forced pass over `dataset` (`torch.inference_mode`),
    page by page (design §8: "Per page: `encode_hidden` ... anchors ...
    head predictions"). Groups every supervised record by `(head_id,
    pointer)`, decodes each head's own prediction (argmax index for enum,
    `sigmoid(logit) > 0` for boolean, `predict_value` -- always
    `torch.long` -- for integer, RFC004 §25.3), and hands the decoded
    per-example values to `report.pointer_report`.

    `lineage_by_bundle=None` means no `--release-dir` was given -- lineage
    is not tracked at all and every `PointerReport` carries
    `lineage_count=None`/`lineage_unmeasured=True` (never a fabricated
    `0`). When a mapping IS given, a page whose `document_bundle_id` is
    absent from it is REFUSED (`S3TypedEvalRefused`) rather than silently
    falling back to the bundle id as a stand-in lineage -- that silent
    fallback was the should_fix this replaces."""
    import torch

    from unflatten.model import s1_train

    head_id_to_child = {head_id: child for child, head_id in typed_head_children().items()}
    predictions: dict[tuple[str, str], list[Any]] = {}
    targets: dict[tuple[str, str], list[Any]] = {}
    lineage_ids: dict[tuple[str, str], list[str]] = {}
    node_type_of: dict[tuple[str, str], str] = {}
    track_lineage = lineage_by_bundle is not None

    n = len(dataset) if limit is None else min(limit, len(dataset))
    with torch.inference_mode():
        for i in range(n):
            sample = dataset[i]
            bundle_id = sample.base.document_bundle_id
            if track_lineage:
                if bundle_id not in lineage_by_bundle:
                    raise S3TypedEvalRefused(
                        f"document_bundle_id {bundle_id!r} is missing from the "
                        "--release-dir lineage map -- refusing rather than "
                        "falling back to the bundle id as a stand-in lineage"
                    )
                lineage = lineage_by_bundle[bundle_id]
            else:
                lineage = None
            typed_batch = stl.collate_typed([sample], allowed_head_ids=allowed_head_ids)
            if not typed_batch.keys:
                continue
            base_batch = s1_train.collate_training_batch(
                [sample.base], ranges=ranges, image_token_id=image_token_id, device=device
            )
            hidden, prefix_len = model.encode_hidden(base_batch.model_batch)
            for key, key_batch in typed_batch.keys.items():
                # `collate_typed` builds every tensor on CPU regardless of
                # where `model`/`hidden` live (same fix as
                # `s3_typed_loss.typed_losses` -- this loop duplicates that
                # logic rather than calling it, so it needs its own move).
                mask = key_batch.supervise
                if not bool(mask.any()):
                    continue
                head_id, pointer_attr = key
                child_name = head_id_to_child[head_id]
                head = getattr(getattr(model, child_name), pointer_attr)
                anchor = stl.gather_anchor_hidden(
                    hidden,
                    row_index=key_batch.row_index.to(hidden.device),
                    anchor_step=key_batch.anchor_step.to(hidden.device),
                    prefix_len=prefix_len,
                    visual_steps=base_batch.visual_steps,
                )
                supervised_anchor = anchor[mask.to(hidden.device)]
                supervised_target = key_batch.target[mask]
                prediction = head(supervised_anchor)
                if isinstance(head, EnumPropertyHead):
                    decoded = prediction.argmax(dim=-1).tolist()
                    target_values = supervised_target.long().tolist()
                elif isinstance(head, BooleanPropertyHead):
                    decoded = (prediction > 0).tolist()
                    target_values = supervised_target.bool().tolist()
                else:
                    decoded = head.predict_value(prediction).tolist()
                    target_values = supervised_target.long().tolist()

                bucket = (head_id, key_batch.pointer)
                predictions.setdefault(bucket, []).extend(decoded)
                targets.setdefault(bucket, []).extend(target_values)
                if track_lineage:
                    lineage_ids.setdefault(bucket, []).extend([lineage] * len(decoded))
                # Every record under one collated key shares one family
                # (the registry refuses two families for one pointer,
                # `candidate_model._build_typed_head_group`); the first
                # supervised target seen for this key names its node_type.
                if bucket not in node_type_of:
                    node_type_of[bucket] = next(
                        t.node_type
                        for t in sample.typed_targets
                        if t.head_id == head_id
                        and stl.normalize_pointer(t.node_property_pointer) == key_batch.pointer
                    )

    reports = []
    for bucket, preds in predictions.items():
        head_id, pointer = bucket
        node_type = node_type_of[bucket]
        family = stl.FAMILY[(node_type, pointer)]
        reports.append(
            report_mod.pointer_report(
                head_id=head_id,
                node_type=node_type,
                pointer=pointer,
                family=family,
                predictions=preds,
                targets=targets[bucket],
                lineage_ids=lineage_ids.get(bucket) if track_lineage else None,
            )
        )
    return reports


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _lineage_by_bundle(release_dir: Path | None) -> dict[str, str] | None:
    """`{document_bundle_id: lineage_root_id}`, or `None` when no
    `--release-dir` is given. `None` disables lineage tracking entirely
    (`evaluate_arm` then reports `lineage_count=None`/
    `lineage_unmeasured=True`); a real mapping makes `evaluate_arm` REFUSE
    any page whose `document_bundle_id` it does not cover, rather than
    falling back to the bundle id as a stand-in lineage."""
    if release_dir is None:
        return None
    from unflatten.model.s2_plan import load_split_records

    return {
        record["authority_id"]: record["lineage_root_id"]
        for record in load_split_records(release_dir)
    }


@dataclass(frozen=True)
class EvalDependencies:
    """Everything `run_report` needs beyond the files it reads itself --
    GPU-resident and model-shaped, so tests inject a fake one instead of
    constructing this for real (mirrors `eval.s2_gate.cli.DecodeDependencies`:
    a forward pass over real visual features does not fit the local GPU)."""

    trained: ArmResult
    baseline: ArmResult
    dataset: stl.S3TrainingDataset
    allowed_head_ids: Sequence[str]
    lineage_by_bundle: Mapping[str, str] | None
    ranges: Any
    image_token_id: int
    device: Any


def build_eval_dependencies(
    args: argparse.Namespace, *, s3_run_manifest: Mapping[str, Any]
) -> EvalDependencies:
    """The real, GPU-resident construction path -- not exercised by the
    default test suite (mirrors `eval.s2_gate.cli.build_decode_dependencies`'s
    own module note)."""
    import torch

    from unflatten.model import composite_vocab as cv
    from unflatten.model import stage_policy as sp
    from unflatten.model.candidate_loader import load_qwen_candidate
    from unflatten.model.candidate_model import QwenDenseCandidateModel
    from unflatten.model.s1_train import ExactTextTrainingDataset

    device = torch.device(args.device)
    eval_cache_identity = s2_cli.eval_cache_identity_from_exact_text_cache(
        args.exact_text_cache_dir
    )

    base_dataset = ExactTextTrainingDataset(args.exact_text_cache_dir, gradient_bearing=False)
    dataset = stl.S3TrainingDataset(base_dataset)

    stage_policy = sp.load_stage_policy(s3_train._POLICIES_DIR / "s3_stage_policy.json")
    allowed_head_ids = sorted(
        {
            head_id
            for row in stage_policy["task_view_allowlist"]
            if row["loss_family"] == "L_typed"
            for head_id in row["head_ids"]
        }
    )

    sdpa_candidate = load_qwen_candidate(device=str(device))
    ranges = cv.packed_ranges(sdpa_candidate.vocab_size)

    def _build_model() -> QwenDenseCandidateModel:
        return QwenDenseCandidateModel(
            decode=sdpa_candidate.decode,
            inject_visual=sdpa_candidate.inject_visual,
            visual_dim=sdpa_candidate.visual_hidden_size,
            d_model=sdpa_candidate.hidden_size,
            ranges=ranges,
            qwen_embedding_weight=sdpa_candidate.embed_tokens.weight,
            qwen_lm_head_weight=sdpa_candidate.lm_head_weight,
        ).to(device)

    trained = load_trained_arm(
        Path(args.checkpoint),
        _build_model(),
        expect_s1_run_manifest_hash=s3_run_manifest["binding_s1_run_manifest_hash"],
        device=device,
    )
    pinned, expect_cache_identity = resolve_baseline_pins(
        s3_run_manifest=s3_run_manifest, eval_cache_identity=eval_cache_identity
    )
    baseline = load_baseline_arm(
        Path(args.baseline_checkpoint),
        _build_model(),
        pinned=pinned,
        expect_cache_identity=expect_cache_identity,
    )

    return EvalDependencies(
        trained=trained,
        baseline=baseline,
        dataset=dataset,
        allowed_head_ids=allowed_head_ids,
        lineage_by_bundle=_lineage_by_bundle(args.release_dir),
        ranges=ranges,
        image_token_id=sdpa_candidate.image_token_id,
        device=device,
    )


def run_report(
    args: argparse.Namespace,
    *,
    build_dependencies: Callable[..., EvalDependencies] = build_eval_dependencies,
) -> int:
    out_path = Path(args.out)
    if out_path.exists():
        raise S3TypedEvalRefused(f"{out_path} already exists -- refusing to overwrite")

    s3_run_manifest = load_s3_run_manifest(args.s3_run_manifest)
    deps = build_dependencies(args, s3_run_manifest=s3_run_manifest)

    trained_pointers = evaluate_arm(
        deps.trained.model,
        deps.dataset,
        allowed_head_ids=deps.allowed_head_ids,
        lineage_by_bundle=deps.lineage_by_bundle,
        ranges=deps.ranges,
        image_token_id=deps.image_token_id,
        device=deps.device,
        limit=args.limit,
    )
    baseline_pointers = evaluate_arm(
        deps.baseline.model,
        deps.dataset,
        allowed_head_ids=deps.allowed_head_ids,
        lineage_by_bundle=deps.lineage_by_bundle,
        ranges=deps.ranges,
        image_token_id=deps.image_token_id,
        device=deps.device,
        limit=args.limit,
    )

    eval_cache_identity = s2_cli.eval_cache_identity_from_exact_text_cache(
        args.exact_text_cache_dir
    )
    report = report_mod.build_report(
        trained_pointer_reports=trained_pointers,
        baseline_pointer_reports=baseline_pointers,
        eval_cache_identity=eval_cache_identity,
        s3_checkpoint_cache_identity=deps.trained.cache_identity,
        baseline_checkpoint_cache_identity=deps.baseline.cache_identity,
        s3_checkpoint_id=deps.trained.checkpoint_id,
        baseline_checkpoint_id=deps.baseline.checkpoint_id,
    )
    out_path.write_text(json.dumps(report, sort_keys=True, indent=2))
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s3-typed-eval",
        description="Teacher-forced typed-property report (S3 discovery, report-only)",
    )
    parser.add_argument("--s3-run-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--exact-text-cache-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_report(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

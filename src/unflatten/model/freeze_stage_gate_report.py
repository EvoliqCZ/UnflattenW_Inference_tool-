"""Freeze one `stage_gate_report.json` (plan `spec-stage-gate-report-c5`,
task 3).

Reads an already-written `s1_runner.py` report, an `s1_run_manifest`, a
Candidate Track, a Checkpoint Ancestry document, and an exact-text cache
manifest out of `--run-dir` / `--exact-text-cache-dir`; a Component Lock and
an S0 admission report; cross-checks them via
`stage_gate_report.build_stage_gate_report`; and writes
`stage_gate_report.json` under `--out-dir`, refusing to overwrite an
existing one.

`freeze()` (the core) takes an already-assembled `corpus_check_fn` --
`(cache_identity_payload, corpus_descriptor) -> corpus_check dict` -- rather
than a tokenizer, so it stays import-light and testable without a real
Qwen tokenizer. `main()` builds the real closure (pinned tokenizer +
allowlist + runtime-limits `budget_ops`) and is the only place in this
module that imports tokenizer machinery.

RUN_STAGE is pinned to `"S2"`: DEC-009 runs S2 under the frozen S1 loss
policy, and this CLI has no flag for it (plan decision 6's argument list is
closed).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from unflatten.model import checkpoint_ancestry as ca
from unflatten.model import exact_text_cache as etc
from unflatten.model import exact_text_roundtrip_corpus as roundtrip
from unflatten.model import stage_gate_report as sgr
from unflatten.model import stage_policy as sp
from unflatten.model.freeze_candidate_track import CANDIDATE_TRACK_FILENAME
from unflatten.model.freeze_checkpoint_ancestry import CHECKPOINT_ANCESTRY_BASE_FILENAME
from unflatten.model.freeze_s1_run_manifest import (
    EXACT_TEXT_CACHE_EXPECTATION_FILENAME,
    QWEN_TEXT_ALLOWLIST_FILENAME,
    S1_RUN_MANIFEST_FILENAME,
)

#: `src/unflatten/model/freeze_stage_gate_report.py` -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: DEC-009: the S2 run this plan's owner runs today is evaluated under the
#: frozen S1 loss policy -- this CLI is not, itself, S1/S2-generic.
RUN_STAGE = "S2"

#: The frozen `s1_stage_policy.json` (owner decision 2026-09-05) -- the loss
#: policy whose `stage_id` this CLI binds `stage_gate_report.stage` to.
_STAGE_POLICY_PATH = Path(sp.__file__).parent / "policies" / "s1_stage_policy.json"

#: The frozen `s1_runtime_limits_policy.json` (owner decision D3, 2026-09-06)
#: -- `text_encoding_budget_ops` is read from here, hash-verified against the
#: run manifest.
_RUNTIME_LIMITS_POLICY_PATH = (
    Path(sp.__file__).parent / "policies" / "s1_runtime_limits_policy.json"
)

_OUTPUT_NAME = "stage_gate_report.json"


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_required_json(path: Path) -> dict[str, Any]:
    """`_read_json`, but a missing file or malformed JSON is reported as
    `StageGateReportInvalid` naming `path`, rather than letting a bare
    `FileNotFoundError` / `json.JSONDecodeError` propagate."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise sgr.StageGateReportInvalid(f"{path} does not exist") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise sgr.StageGateReportInvalid(f"{path} is not valid JSON: {exc}") from exc


def _evaluated_candidate(
    *,
    ancestry: dict[str, Any],
    run_manifest: dict[str, Any],
    checkpoint: Path | None,
    runner_payload: dict[str, Any],
) -> dict[str, Any]:
    """`checkpoint=None` -> unchanged `base_weights_only`, over `ancestry`'s
    own base model weights.

    A path -> the `checkpoint` branch (plan `spec-runner-inputs-c5b`):
    refuses (`StageGateReportInvalid`) unless the runner report's own
    `evaluated_checkpoint.checkpoint_sha256` matches `checkpoint`'s real
    bytes -- the runner report must have evaluated gates 1-3/6/12/13/15
    against THIS checkpoint's own weights, not merely against some
    checkpoint. `checkpoint_ancestry.verify_sidecar` is then run against
    the checkpoint file on disk, `expect_base_ancestry_hash=ancestry
    ["ancestry_hash"]` and `expect_s1_run_manifest_hash=run_manifest
    ["s1_run_manifest_hash"]`, with `embedded_ancestry_hash` taken from
    the runner report's `evaluated_checkpoint.ancestry_hash` -- the value
    `s1_runner_inputs.load_trained_checkpoint` read out of the checkpoint
    file's own bytes when it loaded it (this module is torch-free by
    construction and never opens the checkpoint itself). `checkpoint_id`
    is DERIVED from that (now-verified) ancestry hash and the checkpoint's
    real sha256 -- never read back out of the sidecar's own stored id.
    """
    if checkpoint is None:
        return {
            "kind": "base_weights_only",
            "weights_set_sha256": ancestry["base_model"]["weights_set_sha256"],
        }
    checkpoint = Path(checkpoint)
    checkpoint_sha256 = ca.file_sha256(checkpoint)
    evaluated_checkpoint = runner_payload.get("evaluated_checkpoint")
    reported_sha256 = (
        evaluated_checkpoint.get("checkpoint_sha256")
        if evaluated_checkpoint is not None
        else None
    )
    if evaluated_checkpoint is None or reported_sha256 != checkpoint_sha256:
        raise sgr.StageGateReportInvalid(
            f"--checkpoint {checkpoint} has sha256 {checkpoint_sha256!r}, but the "
            f"runner report's evaluated_checkpoint.checkpoint_sha256 is "
            f"{reported_sha256!r} -- the runner must have evaluated THIS "
            "checkpoint's own weights, not a different one (or none at all)"
        )
    try:
        ca.verify_sidecar(
            checkpoint,
            expect_base_ancestry_hash=ancestry["ancestry_hash"],
            expect_s1_run_manifest_hash=run_manifest["s1_run_manifest_hash"],
            embedded_ancestry_hash=evaluated_checkpoint["ancestry_hash"],
        )
    except ca.CheckpointAncestryInvalid as exc:
        raise sgr.StageGateReportInvalid(f"--checkpoint sidecar refused: {exc}") from exc
    checkpoint_id = ca.compute_checkpoint_id(
        evaluated_checkpoint["ancestry_hash"], checkpoint_sha256
    )
    return {
        "kind": "checkpoint",
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
    }


def freeze(
    *,
    runner_report: Path,
    run_dir: Path,
    component_lock: Path,
    exact_text_cache_dir: Path,
    s0_report: Path,
    out_dir: Path,
    corpus_check_fn: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    policy: Path | None = None,
    previous_corpus_descriptor: Path | None = None,
    checkpoint: Path | None = None,
    run_stage: str = RUN_STAGE,
) -> dict[str, Any]:
    """Build and write `stage_gate_report.json` under `out_dir`.

    Refuses (`stage_gate_report.StageGateReportInvalid`) on every condition
    `stage_gate_report.build_stage_gate_report` refuses on, plus an already
    existing `out_dir/stage_gate_report.json`, a missing/malformed required
    input file, and a `gradient_bearing=true` run manifest whose `run_dir`
    carries no `qwen_text_allowlist.json` (`RunManifestBindingVerifier`
    requires the allowlist payload for such a manifest; a non-gradient-
    bearing manifest tolerates its absence).
    """
    out_dir = Path(out_dir)
    out_path = out_dir / _OUTPUT_NAME
    if out_path.exists():
        raise sgr.StageGateReportInvalid(
            f"{out_path} already exists -- refusing to overwrite a frozen stage gate report"
        )

    run_dir = Path(run_dir)
    runner_report = Path(runner_report)
    s0_report = Path(s0_report)

    policy_path = Path(policy) if policy is not None else sgr.FROZEN_POLICY_PATH
    policy_doc = sgr.load_stage_gate_policy(policy_path)

    runner_payload = _read_required_json(runner_report)
    run_manifest = _read_required_json(run_dir / S1_RUN_MANIFEST_FILENAME)
    candidate_track_doc = _read_required_json(run_dir / CANDIDATE_TRACK_FILENAME)
    ancestry = _read_required_json(run_dir / CHECKPOINT_ANCESTRY_BASE_FILENAME)
    component_lock_doc = _read_required_json(component_lock)
    s0_report_doc = _read_required_json(s0_report)
    stage_policy_doc = sp.load_stage_policy(_STAGE_POLICY_PATH)

    allowlist_path = run_dir / QWEN_TEXT_ALLOWLIST_FILENAME
    if run_manifest.get("gradient_bearing"):
        if not allowlist_path.exists():
            raise sgr.StageGateReportInvalid(
                f"{allowlist_path} does not exist -- a gradient-bearing "
                "run manifest requires the qwen_text_allowlist payload it "
                "was bound to"
            )
        allowlist = _read_required_json(allowlist_path)
    else:
        allowlist = _read_required_json(allowlist_path) if allowlist_path.exists() else None

    # DEC-MP-4: the cohort a gradient-bearing manifest was frozen with; the
    # binding verifier refuses such a manifest without it (2026-09-14).
    expectation_path = run_dir / EXACT_TEXT_CACHE_EXPECTATION_FILENAME
    expectation = None
    if run_manifest.get("gradient_bearing"):
        if not expectation_path.exists():
            raise sgr.StageGateReportInvalid(
                f"{expectation_path} does not exist -- a gradient-bearing run "
                "manifest is verified only under the cohort expectation it was "
                "frozen with (DEC-MP-4)"
            )
        expectation = etc.CacheExpectation.from_payload(_read_required_json(expectation_path))

    cache_manifest = etc.cache_manifest(exact_text_cache_dir)
    cache_identity_payload = cache_manifest["identity"]

    corpus_descriptor = roundtrip.build_corpus_descriptor(roundtrip.CORPUS_VERSION)
    corpus_check = corpus_check_fn(cache_identity_payload, corpus_descriptor)

    previous_descriptor_doc = (
        _read_json(Path(previous_corpus_descriptor))
        if previous_corpus_descriptor is not None
        else None
    )

    evaluated_candidate = _evaluated_candidate(
        ancestry=ancestry,
        run_manifest=run_manifest,
        checkpoint=checkpoint,
        runner_payload=runner_payload,
    )

    doc = sgr.build_stage_gate_report(
        policy=policy_doc,
        runner_payload=runner_payload,
        runner_report_path=runner_report,
        run_manifest=run_manifest,
        component_lock=component_lock_doc,
        cache_identity_payload=cache_identity_payload,
        candidate_track=candidate_track_doc,
        ancestry=ancestry,
        s0_report=s0_report_doc,
        s0_report_path=s0_report,
        s0_report_relpath=_repo_relative(s0_report),
        stage_policy_doc=stage_policy_doc,
        corpus_descriptor=corpus_descriptor,
        corpus_check=corpus_check,
        evaluated_candidate=evaluated_candidate,
        run_stage=run_stage,
        allowlist=allowlist,
        previous_corpus_descriptor=previous_descriptor_doc,
        expectation=expectation,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")

    return {
        "stage_gate_report_hash": doc["stage_gate_report_hash"],
        "gates_all_pass": doc["gates_all_pass"],
        "promoted": doc["promoted"],
        "promotion_blockers": doc["promotion_blockers"],
        "out_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    from unflatten.model import composite_vocab
    from unflatten.model import qwen_tokenizer_artifacts as qta
    from unflatten.model import runtime_limits as rl

    parser = argparse.ArgumentParser(description="Freeze one stage_gate_report.json.")
    parser.add_argument("--runner-report", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--component-lock", type=Path, required=True)
    parser.add_argument("--exact-text-cache-dir", type=Path, required=True)
    parser.add_argument("--s0-report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--previous-corpus-descriptor", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "A trained checkpoint the S1 runner already evaluated (plan "
            "spec-runner-inputs-c5b): --runner-report must carry an "
            "evaluated_checkpoint whose checkpoint_sha256 matches this file's "
            "real bytes, and its ancestry sidecar must verify against --run-dir's "
            "checkpoint_ancestry_base.json and s1_run_manifest.json. A mismatch "
            "or a runner report with no evaluated_checkpoint raises "
            "StageGateReportInvalid. Omit it -- the default -- to evaluate the "
            "untrained base candidate, which makes promoted=false by "
            "construction (evaluated_candidate_not_run_output)."
        ),
    )
    parser.add_argument(
        "--run-stage",
        choices=("S1", "S2"),
        default=RUN_STAGE,
        help=(
            f"the stage of the run this report evaluates (default: {RUN_STAGE}, "
            "DEC-009's S2 run under the frozen S1 loss policy). Pass S1 for an "
            "S1 canary run, or its report records it as an S2 run."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    run_manifest = _read_required_json(args.run_dir / S1_RUN_MANIFEST_FILENAME)

    runtime_limits_policy = rl.load_runtime_limits_policy(_RUNTIME_LIMITS_POLICY_PATH)
    if rl.runtime_limits_policy_hash(runtime_limits_policy) != run_manifest[
        "runtime_limits_policy_hash"
    ]:
        raise sgr.StageGateReportInvalid(
            f"{_RUNTIME_LIMITS_POLICY_PATH} does not hash to the run manifest's "
            "runtime_limits_policy_hash"
        )
    budget_ops = runtime_limits_policy[rl.BUDGET_POLICY_FIELD]

    tokenizer, snapshot_dir = qta.load_pinned_tokenizer()
    tokenizer_hash = qta.tokenizer_artifact_set_hash(
        snapshot_dir, model_id=qta.QWEN_CANDIDATE_MODEL_ID, revision=qta.QWEN_CANDIDATE_REVISION
    )
    allowlist_payload = _read_required_json(args.run_dir / QWEN_TEXT_ALLOWLIST_FILENAME)
    vocabulary = composite_vocab.make_composite_vocabulary(
        tokenizer_artifact_set_hash=tokenizer_hash,
        allowlist_payload=allowlist_payload,
    )

    def corpus_check_fn(
        cache_identity_payload: dict[str, Any], descriptor: dict[str, Any]
    ) -> dict[str, Any]:
        return roundtrip.check_corpus(
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            budget_ops=budget_ops,
            descriptor=descriptor,
        )

    summary = freeze(
        runner_report=args.runner_report,
        run_dir=args.run_dir,
        component_lock=args.component_lock,
        exact_text_cache_dir=args.exact_text_cache_dir,
        s0_report=args.s0_report,
        out_dir=args.out_dir,
        corpus_check_fn=corpus_check_fn,
        policy=args.policy,
        previous_corpus_descriptor=args.previous_corpus_descriptor,
        checkpoint=args.checkpoint,
        run_stage=args.run_stage,
    )
    print(json.dumps(summary, sort_keys=True))
    # Exit 0 even when promoted=false: a discovery-class stage gate report
    # that correctly says "not promoted, here is why" is this CLI's normal,
    # successful output -- it is a REPORT, not a promotion decision.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

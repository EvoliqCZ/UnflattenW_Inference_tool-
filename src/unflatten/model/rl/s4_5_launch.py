"""S4.5 launcher CLI (DEC-047 addendum A v4 points 9/10, DEC-082): subcommands
``build-policies``, ``freeze-run`` and ``launch``. ``launch`` takes only ``--run-dir`` (and
``--dry-run``): checkpoint, dataset and record come from the sealed RUN.

Exit codes: 0 OK, 2 named refusal (one stderr line ``REFUSED <Type>: <msg>``),
3 stopped by stop file. Heavy imports (``run_manifest``, ``training_policy``) stay
inside the command bodies.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

from unflatten.model.checkpoint_ancestry import CheckpointAncestryInvalid
from unflatten.model.rl import data_prep_support as sup
from unflatten.model.rl.adapters import launch as lp
from unflatten.model.rl.adapters import launch_run as lr
from unflatten.model.rl.adapters.exclusions import DEFAULT_DECISIONS_DIR
from unflatten.model.rl.adapters.launch import BUILD_FILE, BUILD_SCHEMA, POLICY_FILES
from unflatten.model.rl.policy_numbers import build_policies, load_policy_numbers
from unflatten.model.rl.ratification import RlRatificationRequired

#: ``RlPolicyInvalid``, ``DataPrepRefused``, ``LaunchRefused``, ``LoopGuardCapsRefused`` and the
#: other named refusals are ValueErrors; ``RlRatificationRequired`` is a RuntimeError. A KeyError
#: is a defect, never a named refusal.
_REFUSALS: Final = (ValueError, OSError, CheckpointAncestryInvalid, RlRatificationRequired)
_HASH_FIELDS: Final = {
    "reward": "reward_policy_hash",
    "rollout": "rollout_policy_hash",
    "training": "training_policy_hash",
    "stage_gate": "stage_policy_hash",
}


def _composite(manifest_path: str, policies: dict[str, Any], numbers: Any) -> str:
    from unflatten.model.rl.policies import verify_dataset_manifest
    from unflatten.model.rl.run_manifest import composite_policy_hash

    manifest = sup.read_json(manifest_path, "dataset manifest")
    verify_dataset_manifest(manifest)
    # composite_policy_hash reads only these five artefacts; no reference identity needed.
    arts = SimpleNamespace(
        reward_policy=policies["reward"],
        rollout_policy=policies["rollout"],
        training_policy=policies["training"],
        stage_policy=policies["stage_gate"],
        dataset_manifest=manifest,
    )
    return composite_policy_hash(arts, numbers.run_class, numbers.selected_typed_heads)  # type: ignore[arg-type]


def _cmd_build_policies(args: argparse.Namespace) -> dict[str, Any]:
    from unflatten.model.rl.state_map import s45_state_map_hash

    out = sup.out_dir(args.out)
    sup.require_free(out, [*POLICY_FILES.values(), BUILD_FILE])
    numbers = load_policy_numbers(args.numbers)
    policies = build_policies(numbers, status=args.status)
    hashes = {_HASH_FIELDS[k]: policies[k][_HASH_FIELDS[k]] for k in POLICY_FILES}
    composite = reason = None
    if args.dataset_manifest:
        composite = _composite(args.dataset_manifest, policies, numbers)
    else:
        reason = "no --dataset-manifest given"
    build = {
        "schema": BUILD_SCHEMA,
        "status": args.status,
        "decision_id": numbers.decision_id,
        "run_class": numbers.run_class,
        "run_seed": numbers.run_seed,
        "selected_typed_heads": list(numbers.selected_typed_heads),
        "numbers_sha256": numbers.sha256,
        "policy_hashes": hashes,
        "state_map_hash": s45_state_map_hash(numbers.run_class, numbers.selected_typed_heads),
        "composite_policy_hash": composite,
        "composite_policy_hash_reason": reason,
    }
    docs = {POLICY_FILES[k]: policies[k] for k in POLICY_FILES} | {BUILD_FILE: build}
    written = sup.write_docs(out, docs)
    return {"command": "build-policies", "out": str(out), "written": written, **build}


# --- freeze-run -------------------------------------------------------------------------


def _paths(args: argparse.Namespace, build: dict[str, Any]) -> lp.InputPaths:
    pol = Path(args.policies)
    return lp.InputPaths(
        reward_policy=pol / POLICY_FILES["reward"],
        rollout_policy=pol / POLICY_FILES["rollout"],
        training_policy=pol / POLICY_FILES["training"],
        stage_policy=pol / POLICY_FILES["stage_gate"],
        dataset_manifest=Path(args.dataset_manifest),
        selection=Path(args.selection),
        registry=Path(args.registry),
        ancestry_base=Path(args.ancestry_base),
        parent_checkpoint=Path(args.parent_checkpoint),
        parent_run_dir=Path(args.parent_run_dir),
        cache_dir=Path(args.cache_dir),
        release_dir=Path(args.release_dir),
        loop_guard_caps=Path(args.loop_guard_caps),
        visual_input_policy=Path(args.visual_input_policy),
        ratification_record=Path(args.ratification_record),
        decisions_dir=Path(args.decisions_dir),
        artefact_root=None if args.artefact_root is None else Path(args.artefact_root),
        run_class=build["run_class"],
        run_seed=build["run_seed"],
        selected_typed_heads=tuple(build["selected_typed_heads"]),
    )


def _require_free_run(run: Path, checkpoint_dir: Path) -> None:
    if run.exists() or run.is_symlink():
        raise lp.RunDirRefused(f"--out {run} already exists; a frozen run is never reused")
    if checkpoint_dir.exists() and (not checkpoint_dir.is_dir() or any(checkpoint_dir.iterdir())):
        raise lp.RunDirRefused(
            f"--checkpoint-dir {checkpoint_dir} is not empty: a half-written run is not resumed"
        )


def _freeze_document(a: lp.Assembled, record_sha: str) -> dict[str, Any]:
    from unflatten import __file__ as package_file

    commit = Path(package_file).resolve().parents[2] / "COMMIT"
    return {
        "parent_checkpoint_id": a.policy_parent.checkpoint_id,
        "parent_checkpoint_sha256": a.policy_parent.checkpoint_sha256,
        "parent_base_ancestry_hash": a.sidecar["checkpoint"]["base_ancestry_hash"],
        "parent_run_class": a.policy_parent.run_class,
        "typed_head_config_hash": a.policy_parent.typed_head_config_hash,
        "visual_feature_tap": a.artefacts.visual_feature_tap,
        "cache_identity_hash": a.cache_identity_hash,
        "feature_source": a.cache_identity["feature_tap"],
        "loop_guard_caps_sha256": a.caps_sha256,
        "ratification_record_sha256": record_sha,
        "code_commit": commit.read_text(encoding="utf-8").strip() if commit.is_file() else None,
    }


def _launch_values(args: argparse.Namespace, a: lp.Assembled) -> dict[str, Any]:
    def resolved(value: str | None) -> str | None:
        return None if value is None else str(Path(value).resolve())

    return {
        "device": args.device,
        "checkpoint_dir": resolved(args.checkpoint_dir),
        "render_dir": resolved(args.render_dir),
        "stop_file": resolved(args.stop_file),
        "ledger_path": resolved(args.ledger_path),
        "ratification_record": str(a.record_path),
        "decisions_dir": resolved(args.decisions_dir),
        "artefact_root": resolved(args.artefact_root),
        "cache_dir": resolved(args.cache_dir),
        "release_dir": resolved(args.release_dir),
        "store_dir": resolved(args.store_dir),
        "stage_data_matrix": resolved(args.stage_data_matrix),
        "allowlist": resolved(args.allowlist),
        "visual_input_policy": resolved(args.visual_input_policy),
        "parent_checkpoint": resolved(args.parent_checkpoint),
        "parent_run_dir": resolved(args.parent_run_dir),
        "loop_guard_caps": resolved(args.loop_guard_caps),
        "expect_s1_run_manifest_hash": args.expect_s1_run_manifest_hash,
    }


def _write_run(staging: Path, final: Path, args: argparse.Namespace, a: lp.Assembled) -> None:
    p = a.paths
    copies = {  # raw bytes: the owner's files, byte for byte
        POLICY_FILES["reward"]: p.reward_policy, POLICY_FILES["rollout"]: p.rollout_policy,
        POLICY_FILES["training"]: p.training_policy, POLICY_FILES["stage_gate"]: p.stage_policy,
        lp.DATASET_FILE: p.dataset_manifest, lp.SELECTION_FILE: p.selection,
        lp.REGISTRY_FILE: p.registry, lp.ANCESTRY_FILE: p.ancestry_base,
    }  # fmt: skip
    for name, source in copies.items():
        sup.publish(staging, name, source.read_bytes())
    record_sha = "sha256:" + lp.file_sha256(a.record_path)
    docs = {
        lp.EXPECTATION_FILE: {
            "expected_page_keys": lp.read_page_keys(p.cache_dir),
            "approved_exclusions": [],
        },
        lp.LAUNCH_FILE: lp.launch_document(_launch_values(args, a)),
        lp.MANIFEST_FILE: lp.build_preview_manifest(a),
        lp.FREEZE_FILE: _freeze_document(a, record_sha),
    }
    sup.write_docs(staging, docs)
    extras = [
        p.parent_checkpoint.with_name(p.parent_checkpoint.name + ".ancestry.json"),
        p.parent_run_dir / lp.S1_MANIFEST, p.parent_run_dir / lp.S4_MANIFEST,
        p.loop_guard_caps, p.visual_input_policy, Path(args.allowlist),
        Path(args.stage_data_matrix), a.record_path,
    ]  # fmt: skip
    sup.publish(staging, lp.SEAL_FILE, lp.seal_text(staging, final, extras))


def _cmd_freeze_run(args: argparse.Namespace) -> dict[str, Any]:
    from unflatten.model.stage_data_matrix import load_stage_data_matrix

    final = Path(args.out).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    _require_free_run(final, checkpoint_dir)
    build = lp.read_policy_build(args.policies)
    assembled = lp.assemble_inputs(_paths(args, build))
    lp.check_policy_build(build, assembled)
    load_stage_data_matrix(args.stage_data_matrix)
    for flag, value in (("allowlist", args.allowlist), ("store-dir", args.store_dir)):
        if not Path(value).exists():
            raise lp.RunDirRefused(f"--{flag} {value} does not exist")
    p = assembled.paths
    lp.require_outputs_apart(
        final, p.parent_run_dir, p.parent_checkpoint,
        {
            "checkpoint_dir": checkpoint_dir,
            "render_dir": Path(args.render_dir),
            "stop_file": Path(args.stop_file),
            "ledger_path": Path(args.ledger_path),
        },
    )  # fmt: skip
    staging = final.with_name(f".{final.name}.freezing-{os.getpid()}")
    try:
        _write_run(staging, final, args, assembled)
        try:
            os.rename(staging, final)
        except OSError as exc:
            raise lp.RunDirRefused(f"cannot publish {final}: {exc}") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    manifest = json.loads((final / lp.MANIFEST_FILE).read_bytes())
    return {
        "command": "freeze-run",
        "out": str(final),
        "written": sorted(f.name for f in final.iterdir()),
        "run_id": manifest["run_id"],
        "composite_policy_hash": manifest["composite_policy_hash"],
        "run_manifest_hash": manifest["run_manifest_hash"],
    }


# --- launch -----------------------------------------------------------------------------


def _result_summary(result: Any) -> dict[str, Any]:
    stopped = result.stopped_by == "stop_requested"
    return {
        "run_id": result.run_id,
        "start_step": result.start_step,
        "steps_done": result.steps_done,
        "updates_run": result.updates_run,
        "stopped_by": result.stopped_by,
        "checkpoint_paths": [str(p) for p in result.checkpoint_paths],
        "ledger_record_count": result.ledger_record_count,
        "ledger_head_hash": result.ledger_head_hash,
        "data_plan_hash": result.data_plan_hash,
        "selection_hash": result.selection_hash,
        "quality_claim": result.quality_claim,
        "exit_code": 3 if stopped else 0,
    }


def _cmd_launch(args: argparse.Namespace) -> dict[str, Any]:
    from unflatten.model.rl import s4_5_train as train

    ra = lr.assemble(args.run_dir)
    ports = lr.build_ports(ra, args.dry_run)
    admitted = lr.admit(ra, ports)
    a = ra.assembled
    try:
        result = train.run_s4_5(
            ratification_record=a.record_path,
            artefacts=a.artefacts,
            run_class=ra.run_class,
            selected_typed_heads=ra.heads,
            config=train.S4_5Config(enabled=not args.dry_run),
            ports=ports,
            run_seed=ra.run_seed,
            allow_unbound_registry=False,
        )
    except train.S4_5Disabled:
        return {
            "command": "launch", "dry_run": True, "run_id": ra.run_id,
            "composite_policy_hash": a.composite, **admitted,
        }  # fmt: skip
    return {"command": "launch", "dry_run": False, **_result_summary(result)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="s4_5_launch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn: Callable[[argparse.Namespace], dict[str, Any]]) -> Any:
        p = sub.add_parser(name)
        p.set_defaults(run=fn)
        return p

    p = add("build-policies", _cmd_build_policies)
    p.add_argument("--numbers", required=True)
    p.add_argument("--status", required=True, choices=("PROPOSED", "RATIFIED"))
    p.add_argument("--out", required=True)
    p.add_argument("--dataset-manifest", default=None)

    p = add("freeze-run", _cmd_freeze_run)
    for flag in (
        "out", "policies", "dataset-manifest", "selection", "registry", "ancestry-base",
        "parent-checkpoint", "parent-run-dir", "cache-dir", "release-dir", "store-dir",
        "stage-data-matrix", "allowlist", "visual-input-policy", "loop-guard-caps",
        "ratification-record", "checkpoint-dir", "render-dir", "stop-file", "ledger-path",
    ):  # fmt: skip
        p.add_argument(f"--{flag}", required=True)
    p.add_argument("--decisions-dir", default=str(DEFAULT_DECISIONS_DIR))
    p.add_argument("--artefact-root", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--expect-s1-run-manifest-hash", default=None)

    p = add("launch", _cmd_launch)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    try:
        summary = args.run(args)
    except _REFUSALS as exc:
        text = " ".join(str(exc).split()) or repr(exc)
        print(f"REFUSED {type(exc).__name__}: {text}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return int(summary.get("exit_code", 0))


if __name__ == "__main__":
    sys.exit(main())

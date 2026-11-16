"""Freeze generation-0 `checkpoint_ancestry_base.json` (plan
`spec-checkpoint-ancestry-c6` task 2).

Reads a pinned-hashes file (the pre-computed weight/processor hash listing
for the frozen base model), a Model Component Lock, an S0 admission
report, and a release manifest; cross-checks them via
`checkpoint_ancestry.build_base_ancestry`; and writes
`checkpoint_ancestry_base.json` under `--out-dir`, refusing to overwrite an
existing one.

`candidate_loader` (which imports torch) is imported inside :func:`main`
only, never at module scope -- this module, like `checkpoint_ancestry.py`
itself, must stay import-safe without torch.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from unflatten.model import checkpoint_ancestry as ca

#: `src/unflatten/model/freeze_checkpoint_ancestry.py` -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_DECISION_ID_PATTERN = re.compile(r"^(DEC-\d{3})")

#: The filename this CLI writes under `--out-dir` -- the shared contract
#: `freeze_stage_gate_report.py` imports rather than re-hardcoding.
CHECKPOINT_ANCESTRY_BASE_FILENAME = "checkpoint_ancestry_base.json"


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        return str(resolved)


def _decision_ref(path: Path) -> dict[str, Any]:
    """`{decision_id, path, sha256}` from a decision file named `DEC-###...`."""
    match = _DECISION_ID_PATTERN.match(path.name)
    if not match:
        raise ca.CheckpointAncestryInvalid(
            f"decision file {path} does not start with 'DEC-###' -- cannot "
            "derive its decision_id from the filename"
        )
    return {
        "decision_id": match.group(1),
        "path": _repo_relative(path),
        "sha256": ca.file_sha256(path),
    }


def _required(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ca.CheckpointAncestryInvalid(
            f"pinned-hashes file is missing required key {key!r}"
        )
    return payload[key]


def _hashed_file_list(payload: dict[str, Any], key: str) -> list[dict[str, str]]:
    """`{filename: bare_or_prefixed_hex}` (the real pinned-hashes file
    shape) mapped to the schema's `[{path, sha256}]`, sorted by path."""
    mapping = _required(payload, key)
    return [
        {"path": path, "sha256": ca.normalise_sha256(sha)}
        for path, sha in sorted(mapping.items())
    ]


def _base_model(pinned_hashes_path: Path, *, model_id: str) -> dict[str, Any]:
    """The `base_model` object, mapped from the real on-disk pinned-hashes
    shape (`{revision, weights: {filename: hex}, processor: {filename:
    hex}, weights_set_sha256, processor_set_sha256}`, no `model_id`, bare
    hex) into the schema shape -- never spread verbatim, since the two
    shapes disagree on every field but `revision`.

    `model_id` is supplied by the caller (`main`, from
    `candidate_loader.QWEN_CANDIDATE_MODEL_ID`) rather than read from the
    file, exactly like `expected_revision` -- the pinned-hashes file names
    no model.
    """
    payload = json.loads(pinned_hashes_path.read_text())
    return {
        "model_id": model_id,
        "revision": _required(payload, "revision"),
        "weights": _hashed_file_list(payload, "weights"),
        "weights_set_sha256": ca.normalise_sha256(_required(payload, "weights_set_sha256")),
        "processor": _hashed_file_list(payload, "processor"),
        "processor_set_sha256": ca.normalise_sha256(_required(payload, "processor_set_sha256")),
        "pinned_hashes": {
            "path": _repo_relative(pinned_hashes_path),
            "sha256": ca.file_sha256(pinned_hashes_path),
        },
    }


def _load_parent(parent_checkpoint: Path, parent_run_manifest: Path) -> dict[str, Any]:
    """`parent=` for `checkpoint_ancestry.build_base_ancestry` (DEC-028
    D1), loaded via the existing sidecar helpers: the parent's own sidecar
    file is read for the expectations it itself records, then
    `ca.verify_sidecar` re-verifies it against those same values (schema,
    self-hash, derived statuses, checkpoint bytes) rather than trusting the
    file on disk unverified."""
    raw = json.loads(ca.sidecar_path(parent_checkpoint).read_text())
    verified = ca.verify_sidecar(
        parent_checkpoint,
        expect_base_ancestry_hash=raw["checkpoint"]["base_ancestry_hash"],
        expect_s1_run_manifest_hash=raw["checkpoint"]["s1_run_manifest_hash"],
        embedded_ancestry_hash=raw["ancestry_hash"],
    )
    companion = json.loads(parent_run_manifest.read_text())
    return {"sidecar": verified, "companion": companion}


def freeze(
    *,
    pinned_hashes: Path,
    component_lock: Path,
    s0_report: Path,
    release_dir: Path,
    release_name: str,
    store_name: str,
    base_model_decision: Path,
    dataset_decision: Path,
    out_dir: Path,
    expected_revision: str,
    model_id: str,
    distribution_decision: Path | None = None,
    parent_checkpoint: Path | None = None,
    parent_run_manifest: Path | None = None,
    also_exposed_releases: Sequence[tuple[Path, str, str]] = (),
) -> dict[str, Any]:
    """Build and write `checkpoint_ancestry_base.json` under `out_dir`.

    Refuses (`CheckpointAncestryInvalid`) on every condition
    `checkpoint_ancestry.build_base_ancestry` refuses on, plus an already
    existing `out_dir/checkpoint_ancestry_base.json`, plus exactly one of
    `parent_checkpoint` / `parent_run_manifest` given without the other
    (DEC-028 D1: a warm start needs both, a clean-pretrained base needs
    neither).
    """
    if (parent_checkpoint is None) != (parent_run_manifest is None):
        raise ca.CheckpointAncestryInvalid(
            "--parent-checkpoint and --parent-run-manifest must both be given or "
            "both omitted -- refused rather than warm-starting from a checkpoint "
            "with no companion run manifest or vice versa"
        )
    parent = (
        _load_parent(parent_checkpoint, parent_run_manifest)
        if parent_checkpoint is not None
        else None
    )

    release_manifest_path = release_dir / "release-manifest.json"
    doc = ca.build_base_ancestry(
        base_model=_base_model(pinned_hashes, model_id=model_id),
        component_lock=json.loads(component_lock.read_text()),
        s0_report=json.loads(s0_report.read_text()),
        s0_report_path=s0_report,
        release_manifest=json.loads(release_manifest_path.read_text()),
        release_name=release_name,
        store_name=store_name,
        base_model_decision=_decision_ref(base_model_decision),
        dataset_decision=_decision_ref(dataset_decision),
        expected_revision=expected_revision,
        distribution_decision=(
            _decision_ref(distribution_decision) if distribution_decision is not None else None
        ),
        parent=parent,
        also_exposed=[
            {
                "release_name": name,
                "release_root_hash": json.loads(
                    (Path(directory) / "release-manifest.json").read_text()
                )["release_root_hash"],
                "store_name": store,
            }
            for directory, name, store in also_exposed_releases
        ],
    )

    out_dir = Path(out_dir)
    out_path = out_dir / CHECKPOINT_ANCESTRY_BASE_FILENAME
    if out_path.exists():
        raise ca.CheckpointAncestryInvalid(
            f"{out_path} already exists -- refusing to overwrite a frozen base ancestry"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")

    return {
        "ancestry_hash": doc["ancestry_hash"],
        "rights": doc["checkpoint_rights_status"],
        "privacy": doc["checkpoint_privacy_status"],
        "out_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    from unflatten.model.candidate_loader import QWEN_CANDIDATE_MODEL_ID, QWEN_CANDIDATE_REVISION

    parser = argparse.ArgumentParser(
        description="Freeze generation-0 checkpoint_ancestry_base.json."
    )
    parser.add_argument("--pinned-hashes", type=Path, required=True)
    parser.add_argument("--component-lock", type=Path, required=True)
    parser.add_argument("--s0-report", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--store-name", required=True)
    parser.add_argument("--base-model-decision", type=Path, required=True)
    parser.add_argument("--dataset-decision", type=Path, required=True)
    parser.add_argument("--distribution-decision", type=Path, default=None)
    parser.add_argument("--parent-checkpoint", type=Path, default=None)
    parser.add_argument("--parent-run-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--also-exposed-release",
        nargs=3,
        action="append",
        default=[],
        metavar=("RELEASE_DIR", "RELEASE_NAME", "STORE_NAME"),
        help="declare a further dataset exposure (e.g. the warm-start parent's "
        "release, DEC-006); repeatable",
    )
    args = parser.parse_args(argv)

    summary = freeze(
        pinned_hashes=args.pinned_hashes,
        component_lock=args.component_lock,
        s0_report=args.s0_report,
        release_dir=args.release_dir,
        release_name=args.release_name,
        store_name=args.store_name,
        base_model_decision=args.base_model_decision,
        dataset_decision=args.dataset_decision,
        distribution_decision=args.distribution_decision,
        parent_checkpoint=args.parent_checkpoint,
        parent_run_manifest=args.parent_run_manifest,
        also_exposed_releases=[(Path(d), n, s) for d, n, s in args.also_exposed_release],
        out_dir=args.out_dir,
        expected_revision=QWEN_CANDIDATE_REVISION,
        model_id=QWEN_CANDIDATE_MODEL_ID,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

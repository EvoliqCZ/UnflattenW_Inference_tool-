"""Freeze `candidate_track.json` (plan `spec-candidate-track-c4`, task 3).

Reads a Model Component Lock, a `document_kind: "base"` checkpoint ancestry
document and an Architecture Manifest from disk (the manifest is read
explicitly from a file -- extracting it from a provenance document, e.g.
`bindings-*.provenance.json["architecture_manifest"]`, is the caller's job,
not this CLI's); cross-checks them via `candidate_track.build_candidate_track`
against the capability profile the component lock's own
`model_capability_profile` component names (DEC-024 wiring: resolved via
`capability_profile.resolve_capability_profile_from_component_lock`, fail-
closed on an unknown hash, never a hardcoded `PRIMARY_S1_CAPABILITY_PROFILE`
that would refuse a lock legitimately bound to a different ratified
profile); and writes `candidate_track.json` under `--out-dir`, refusing to
overwrite an existing one.

Torch-free by construction, mirroring `freeze_checkpoint_ancestry.py`:
nothing here imports torch (this CLI needs no candidate loader -- the base
model identity check is against the literal `candidate_track.QWEN_BASE_MODEL_ID`,
already cross-checked against `candidate_loader.QWEN_CANDIDATE_MODEL_ID`
elsewhere).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from unflatten.model import candidate_track as ct
from unflatten.model.capability_profile import (
    CapabilityProfileHashUnresolved,
    resolve_capability_profile_from_component_lock,
)

#: The filename this CLI writes under `--out-dir` -- the shared contract
#: `freeze_stage_gate_report.py` imports rather than re-hardcoding.
CANDIDATE_TRACK_FILENAME = "candidate_track.json"


def freeze(
    *,
    component_lock: Path,
    ancestry: Path,
    architecture_manifest: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Build and write `candidate_track.json` under `out_dir`.

    Refuses (`ct.CandidateTrackInvalid`) on every condition
    `candidate_track.build_candidate_track` refuses on, plus an already
    existing `out_dir/candidate_track.json`.
    """
    component_lock_payload = json.loads(component_lock.read_text())
    # Resolved from the lock's own `model_capability_profile` component
    # (DEC-024 wiring), never a hardcoded profile: the track must bind
    # whichever ratified profile this lock actually declares.
    try:
        capability_profile = resolve_capability_profile_from_component_lock(
            component_lock_payload
        )
    except CapabilityProfileHashUnresolved as exc:
        raise ct.CandidateTrackInvalid(
            f"component lock capability profile refused: {exc}"
        ) from exc

    doc = ct.build_candidate_track(
        component_lock=component_lock_payload,
        ancestry=json.loads(ancestry.read_text()),
        capability_profile=capability_profile,
        architecture_manifest=json.loads(architecture_manifest.read_text()),
    )

    out_dir = Path(out_dir)
    out_path = out_dir / CANDIDATE_TRACK_FILENAME
    if out_path.exists():
        raise ct.CandidateTrackInvalid(
            f"{out_path} already exists -- refusing to overwrite a frozen candidate track"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")

    return {
        "candidate_track_id": doc["candidate_track_id"],
        "admission_track_name": doc["admission_track_name"],
        "rights": doc["checkpoint_rights_status"],
        "privacy": doc["checkpoint_privacy_status"],
        "out_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze candidate_track.json.")
    parser.add_argument("--component-lock", type=Path, required=True)
    parser.add_argument("--ancestry", type=Path, required=True)
    parser.add_argument("--architecture-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = freeze(
        component_lock=args.component_lock,
        ancestry=args.ancestry,
        architecture_manifest=args.architecture_manifest,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

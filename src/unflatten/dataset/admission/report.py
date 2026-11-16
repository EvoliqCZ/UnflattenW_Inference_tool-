"""The S0 admission report (RFC005 §9/§13, dataset half).

Two refusals are load-bearing and both are on the real call path, not
only reachable from a test:

**A release that does not verify produces a report, not an exception.**
`build_admission_report` catches the evidence failure and emits a report
whose row 3 is `fail` and whose other eight rows are `unmeasured` — the
report has to be able to say why the release is not an authority.

**`promoted` is refused for an empty admitted set.** §13.3's conditions
("0 ambiguity", "100 % eligible") are all satisfied vacuously by admitting
nothing, and under §12's vocabulary a promoted-but-empty S0 is a worse
state than a blocked one. Whether §13.3 *requires* a non-empty cohort is
not stated in RFC005 and is a governance decision; this code does not
resolve it. It refuses to emit `promoted` over an empty set and emits
`blocked` with `EMPTY_ADMITTED_SET`, and it records the open question
verbatim so the decision is made by whoever may make it.

**The overall `outcome`/`outcome_reasons` are derived, not declared.**
Until schema 0.2.0 (2026-09-06) the top-level verdict was the literal
`"blocked"` regardless of what the dataset half, the model half, ancestry
and trainability smoke actually said — a release could read `promoted`
dataset, `admitted` model, resolved ancestry and a passing smoke and the
report would still say `blocked`. `_overall_verdict` is the pure function
that now derives it from all four §13.3 clauses this module can measure:
dataset outcome, model outcome, legacy-ancestry resolution, and the
trainability smoke document. Each contributes at most one reason and
`promoted` requires the reason list to be empty.

The trainability smoke document is never produced by this module and this
module never imports the suite that runs it (`unflatten.model
.trainability_smoke` imports `torch`, and admission has to stay importable
without it). It is threaded in as a plain JSON document via
`--trainability-smoke`, produced ahead of time by
`python -m unflatten.model.trainability_smoke --out`. **Its absence
blocks**, with `TRAINABILITY_SMOKE_UNMEASURED` — §13.3 lists the smoke as
a pass condition, and an unmeasured condition is not evidence that it
passed.

The model half's `outcome` is derived from the §9.1 artifact probes and
the §9.5 blocker matrix — not asserted — each artifact is probed by a
source search whose pattern and matches are printed in the report. The
model-side half (probes, blocker matrix, verdict) lives in
`model_admission.py`; this module re-exports the frozen data so the S0
suite's `monkeypatch.setattr(report, ...)` pattern keeps working, and
`_model_admission` passes those report-module attributes in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from unflatten.dataset.admission.ancestry import legacy_ancestry
from unflatten.dataset.admission.cohort import build_cohort
from unflatten.dataset.admission.conditions import (
    dataset_conditions,
    unmeasured_conditions,
)
from unflatten.dataset.admission.coverage import coverage_report
from unflatten.dataset.admission.evidence import (
    AdmissionError,
    load_release,
    load_store_index,
)
from unflatten.dataset.admission.migration import migration_report
from unflatten.dataset.admission.model_admission import (
    MODEL_ARTIFACT_PROBES,
    MODEL_BLOCKERS,
    build_model_admission,
)
from unflatten.dataset.admission.supervision import SCAN_MODES, scan_supervision

REPORT_SCHEMA_VERSION = "0.2.0"

#: Dense-V0's one admitted track (must match
#: `unflatten.model.trainability_smoke.DENSE_CONTRACT_CANARY_TRACK`; this
#: module never imports that module, so the value is a literal here).
DENSE_CONTRACT_CANARY_TRACK = "dense_contract_canary"

#: Overall-verdict reason codes for the trainability-smoke clause.
TRAINABILITY_SMOKE_UNMEASURED = "TRAINABILITY_SMOKE_UNMEASURED"
TRAINABILITY_SMOKE_FAILED = "TRAINABILITY_SMOKE_FAILED"

#: Overall-verdict reason codes for the model and ancestry clauses.
MODEL_ADMISSION_BLOCKED = "MODEL_ADMISSION_BLOCKED"
LEGACY_ANCESTRY_UNRESOLVED = "LEGACY_ANCESTRY_UNRESOLVED"

_GOVERNANCE_QUESTION = {
    "question": (
        "does RFC005 §13.3 require a non-empty admitted cohort, or is a "
        "vacuous pass over an empty admitted set a pass?"
    ),
    "status": "OPEN",
    "decided_here": False,
    "behaviour": (
        "this runner never emits `promoted` over an empty admitted set; it "
        "emits `blocked` with EMPTY_ADMITTED_SET and leaves the reading of "
        "§13.3 to governance"
    ),
}


def _model_admission(repo_root: Path) -> dict[str, Any]:
    """The model-side half, implemented in `model_admission.py`.

    The data lives there, but the report-module attributes are read here
    and passed in: the S0 suite patches `report.MODEL_BLOCKERS` /
    `report.MODEL_ARTIFACT_PROBES` and expects to observe the change, so
    the report module stays the single patchable surface.
    """
    return build_model_admission(
        repo_root,
        blockers=MODEL_BLOCKERS,
        probes=MODEL_ARTIFACT_PROBES,
    )


def _dataset_verdict(
    conditions: list[dict[str, Any]], admitted: int
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    failed = [c["row"] for c in conditions if c["status"] == "fail"]
    unmeasured = [c["row"] for c in conditions if c["status"] == "unmeasured"]
    if failed:
        reasons.append(f"BLOCKING_CONDITION_FAILED: §9.2 rows {failed}")
    if unmeasured:
        # §12.2: insufficient evidence is `blocked`, never a post-hoc pass.
        reasons.append(f"INSUFFICIENT_EVIDENCE: §9.2 rows {unmeasured} unmeasured")
    if admitted == 0:
        reasons.append("EMPTY_ADMITTED_SET")
    return ("blocked" if reasons else "promoted"), reasons


def _validate_trainability_smoke(doc: Any) -> dict[str, Any]:
    """Refuse a malformed trainability-smoke document before it becomes
    evidence — fail-closed, and every failure names what was wrong."""

    def _fail(message: str) -> None:
        raise ValueError(f"trainability smoke document: {message}")

    if not isinstance(doc, dict):
        _fail("must be a JSON object")
    version = doc.get("trainability_smoke_schema_version")
    if not isinstance(version, str) or version.split(".")[0] != "0":
        _fail(
            "trainability_smoke_schema_version must be a string with major "
            f"version 0, got {version!r}"
        )
    if type(doc.get("passed")) is not bool:
        _fail(f"passed must be a boolean, got {doc.get('passed')!r}")
    gradient_updates = doc.get("gradient_updates")
    if (
        type(gradient_updates) is not int
        or isinstance(gradient_updates, bool)
        or gradient_updates != 0
    ):
        _fail(f"gradient_updates must be 0, got {gradient_updates!r}")
    if doc.get("not_an_s1_gradient_update") is not True:
        _fail("not_an_s1_gradient_update must be true")
    tracks = doc.get("tracks")
    if not isinstance(tracks, dict):
        _fail(f"tracks must be an object, got {type(tracks).__name__}")
    for name, track in tracks.items():
        if not isinstance(track, dict) or type(track.get("passed")) is not bool:
            _fail(f"tracks[{name!r}].passed must be a boolean")
    return doc


def _load_trainability_smoke(path: Path | str) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    return _validate_trainability_smoke(doc)


def _trainability_smoke_section(
    smoke: dict[str, Any] | None,
    admitted_tracks: list[str],
    source: str | None,
) -> dict[str, Any]:
    admitted_tracks = list(admitted_tracks)
    if smoke is None:
        return {
            "source": source,
            "status": "unmeasured",
            "schema_version": None,
            "passed": None,
            "tracks_required": admitted_tracks,
            "tracks_covered": [],
            "tracks_not_admitted": [],
            "not_an_s1_gradient_update": None,
        }
    tracks = smoke.get("tracks", {})
    covered = [track for track in admitted_tracks if track in tracks]
    not_admitted = [track for track in tracks if track not in admitted_tracks]
    tracks_ok = all(tracks[track].get("passed") is not False for track in covered)
    fully_covered = len(covered) == len(admitted_tracks)
    status = (
        "pass"
        if smoke["passed"] is not False and tracks_ok and fully_covered
        else "fail"
    )
    return {
        "source": source,
        "status": status,
        "schema_version": smoke["trainability_smoke_schema_version"],
        "passed": smoke["passed"],
        "tracks_required": admitted_tracks,
        "tracks_covered": covered,
        "tracks_not_admitted": not_admitted,
        "not_an_s1_gradient_update": smoke["not_an_s1_gradient_update"],
    }


def _overall_verdict(
    dataset_outcome: str,
    dataset_reasons: list[str],
    model: dict[str, Any],
    ancestry: dict[str, Any] | None,
    smoke: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    """Derive the S0 overall verdict from the four §13.3 clauses this
    module can measure. Pure: never mutates its arguments."""
    if dataset_outcome == "blocked" and not dataset_reasons:
        raise ValueError(
            "overall verdict: a blocked dataset outcome must carry at "
            "least one reason"
        )
    reasons: list[str] = list(dataset_reasons)

    if ancestry is None or ancestry.get("value") is not False:
        reasons.append(LEGACY_ANCESTRY_UNRESOLVED)

    model_outcome = model.get("outcome")
    if model_outcome not in ("admitted", "blocked"):
        raise ValueError(f"overall verdict: unknown model outcome {model_outcome!r}")
    if model_outcome == "blocked":
        reasons.append(MODEL_ADMISSION_BLOCKED)

    admitted_tracks = list(model.get("admitted_tracks", []))
    if smoke is None:
        reasons.append(TRAINABILITY_SMOKE_UNMEASURED)
    else:
        if smoke["passed"] is False:
            reasons.append(TRAINABILITY_SMOKE_FAILED)
        tracks = smoke.get("tracks", {})
        for track in admitted_tracks:
            track_doc = tracks.get(track)
            if track_doc is None:
                reasons.append(f"{TRAINABILITY_SMOKE_UNMEASURED}: {track}")
            elif track_doc.get("passed") is False:
                reasons.append(f"{TRAINABILITY_SMOKE_FAILED}: {track}")

    return ("blocked" if reasons else "promoted"), reasons


def _unverified_report(
    release_dir: Path,
    store_dir: Path,
    exc: AdmissionError,
    repo_root: Path,
    trainability_smoke: dict[str, Any] | None,
    trainability_smoke_path: str | None,
) -> dict[str, Any]:
    immutability = {"status": "fail", "checks": exc.checks, "error": str(exc)}
    conditions = unmeasured_conditions(str(exc), immutability)
    outcome, reasons = _dataset_verdict(conditions, 0)
    model = _model_admission(repo_root)
    smoke_section = _trainability_smoke_section(
        trainability_smoke, model["admitted_tracks"], trainability_smoke_path
    )
    overall_outcome, overall_reasons = _overall_verdict(
        outcome, reasons, model, None, trainability_smoke
    )
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_by": "unflatten.dataset.admission",
        "rfc": "RFC005 §9/§13 (§13 RATIFIED 2026-09-02; §9 PROPOSED)",
        "run_class": "discovery",
        "gradient_updates": 0,
        "inputs": {
            "release_dir": str(release_dir),
            "store_dir": str(store_dir),
            "trainability_smoke": trainability_smoke_path,
        },
        "release_immutability": immutability,
        "dataset_admission": {
            "conditions": conditions,
            "admitted": {"documents": 0, "by_task_kind": {}, "excluded": {}},
            "governance_question": _GOVERNANCE_QUESTION,
            "outcome": outcome,
            "outcome_reasons": reasons,
        },
        "coverage": None,
        "migration_report": None,
        "legacy_ancestry": None,
        "model_admission": model,
        "trainability_smoke": smoke_section,
        "outcome": overall_outcome,
        "outcome_reasons": overall_reasons,
    }


def build_admission_report(
    release_dir: Path | str,
    store_dir: Path | str,
    *,
    supervision_scan: str = "full",
    data_roots: list[Path] | tuple[Path, ...] | None = None,
    repo_root: Path | str | None = None,
    trainability_smoke: dict[str, Any] | None = None,
    trainability_smoke_path: str | None = None,
) -> dict[str, Any]:
    """Build the S0 dataset-admission report for one release.

    `supervision_scan` is `"full"` (read every candidate target bundle,
    ~20 GB on the real corpus) or `"none"` (§9.2 row 6 stays unmeasured,
    which blocks — it is not a pass).

    `trainability_smoke` is the already-loaded, already-validated §9.1
    item 11 smoke document (see `_validate_trainability_smoke`), produced
    ahead of time by `python -m unflatten.model.trainability_smoke --out`;
    `None` means unmeasured, which blocks the overall verdict.
    `trainability_smoke_path` is recorded in `inputs` for provenance only.
    """
    if supervision_scan not in SCAN_MODES:
        raise ValueError(
            f"unknown supervision scan mode {supervision_scan!r}; "
            f"expected one of {', '.join(SCAN_MODES)}"
        )
    if trainability_smoke is not None:
        trainability_smoke = _validate_trainability_smoke(trainability_smoke)
    release_dir, store_dir = Path(release_dir), Path(store_dir)
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[4]
    roots = [Path(r) for r in data_roots] if data_roots else [store_dir.parent]

    try:
        ev = load_release(release_dir)
    except AdmissionError as exc:
        return _unverified_report(
            release_dir,
            store_dir,
            exc,
            repo_root,
            trainability_smoke,
            trainability_smoke_path,
        )

    immutability = {"status": "pass", "checks": ev.checks, "error": None}
    store_index = load_store_index(store_dir)
    cohort = build_cohort(ev)
    scan = scan_supervision(
        ev.artifacts, store_dir, list(cohort.full_ast), mode=supervision_scan
    )
    conditions = dataset_conditions(
        ev, store_index, cohort, store_dir, immutability, scan
    )
    outcome, reasons = _dataset_verdict(conditions, len(cohort.admitted))
    coverage = coverage_report(ev, store_index, conditions)
    ancestry = legacy_ancestry(roots)
    migration = migration_report(
        ev, store_index, cohort, conditions, coverage, ancestry, outcome, reasons
    )
    model = _model_admission(repo_root)
    smoke_section = _trainability_smoke_section(
        trainability_smoke, model["admitted_tracks"], trainability_smoke_path
    )
    overall_outcome, overall_reasons = _overall_verdict(
        outcome, reasons, model, ancestry, trainability_smoke
    )
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_by": "unflatten.dataset.admission",
        "rfc": "RFC005 §9/§13 (§13 RATIFIED 2026-09-02; §9 PROPOSED)",
        "run_class": "discovery",
        "gradient_updates": 0,
        "inputs": {
            "release_dir": str(release_dir),
            "store_dir": str(store_dir),
            "release_root_hash": ev.manifest.get("release_root_hash"),
            "component_lock": ev.manifest.get("component_lock"),
            "supervision_scan": supervision_scan,
            "trainability_smoke": trainability_smoke_path,
        },
        "release_immutability": immutability,
        "dataset_admission": {
            "conditions": conditions,
            "admitted": {
                "documents": len(cohort.admitted),
                "train_candidates": len(cohort.candidates),
                "by_task_kind": cohort.by_task_kind,
                "excluded": cohort.excluded,
            },
            "governance_question": _GOVERNANCE_QUESTION,
            "outcome": outcome,
            "outcome_reasons": reasons,
        },
        "coverage": coverage,
        "migration_report": migration,
        "legacy_ancestry": ancestry,
        "model_admission": model,
        "trainability_smoke": smoke_section,
        "outcome": overall_outcome,
        "outcome_reasons": overall_reasons,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.dataset.admission.report",
        description="RFC005 §9/§13 S0 dataset-admission report (no gradient update)",
    )
    parser.add_argument("release_dir", help="immutable RFC003 release directory")
    parser.add_argument("store_dir", help="content-addressed factory store")
    parser.add_argument("--out", help="write the report here instead of stdout")
    parser.add_argument(
        "--supervision-scan",
        choices=SCAN_MODES,
        default="full",
        help="§9.2 row 6: 'full' reads every candidate target bundle; "
        "'none' leaves the row unmeasured, which blocks",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=None,
        help="directory to audit for legacy checkpoints (repeatable; "
        "defaults to the store's parent)",
    )
    parser.add_argument(
        "--trainability-smoke",
        help="output of `python -m unflatten.model.trainability_smoke --out`; "
        "absent means TRAINABILITY_SMOKE_UNMEASURED, which blocks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    smoke = (
        _load_trainability_smoke(args.trainability_smoke)
        if args.trainability_smoke
        else None
    )
    report = build_admission_report(
        args.release_dir,
        args.store_dir,
        supervision_scan=args.supervision_scan,
        data_roots=[Path(p) for p in args.data_root] if args.data_root else None,
        trainability_smoke=smoke,
        trainability_smoke_path=args.trainability_smoke,
    )
    data = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(data + "\n")
    else:
        print(data)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

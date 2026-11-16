"""RFC005-ERRATA-007 E5 VC-4: combined PRE (VC-2) + POST (VC-1) evaluation.

VC-4 trains/evaluates nothing of its own -- it composes evidence VC-1 and
VC-2 already produced. It refuses to run at all unless both children's own
reports (:mod:`s7_vc1_train`, :mod:`s7_vc2_train`) carry
``architecture_contract_passed: true``: a combined arm built on top of an
arm that has not even cleared its own architecture gate is not evidence of
anything (RFC005 §21.2: "Zlyhanie = zamietnutie bez ohladu na agregatnu
metriku" -- the same discipline `vc5_gate_report.py` applies to one arm,
applied here as VC-4's own precondition over two).

The combined arm's own promotion evidence is still evaluated by
:func:`unflatten.model.visual_compression.vc5_gate_report.build_vc5_gate_report`,
never a second, driftable copy of that gate.

A real combined training/evaluation run additionally requires an explicit
ratification record naming child ``"VC-4"`` (:mod:`s7_ratification`); only
``--dry-run`` runs today.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from unflatten.model.hashing import domain_hash
from unflatten.model.visual_compression.adaptive_v0 import (
    IMPLEMENTATION_HASH,
    AdaptiveVisualMemoryCompressor,
)
from unflatten.model.visual_compression.architecture_contract import check_architecture_contract
from unflatten.model.visual_compression.contracts import VisualMemory
from unflatten.model.visual_compression.s7_ratification import (
    S7RatificationRecord,
    require_ratification,
)
from unflatten.model.visual_compression.vc5_gate_report import build_vc5_gate_report

_S7_VC4_MANIFEST_DOMAIN = "rfc005.s7-vc4-run-manifest/v1"

_ARCHITECTURE_GATE_KEY = "architecture_contract_passed"


class VC4ArchitectureGateRefused(RuntimeError):
    """VC-1's or VC-2's own report did not pass its own architecture gate."""


def _report_passed_architecture_gate(report_path: Path) -> bool:
    """Read one child's report file and return its own architecture-gate
    verdict. A missing/malformed field refuses (`.get` default `False`),
    never treated as an implicit pass."""
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return payload.get(_ARCHITECTURE_GATE_KEY) is True


def require_vc1_and_vc2_architecture_gates(vc1_report: Path, vc2_report: Path) -> None:
    """Refuse unless both VC-1's and VC-2's own reports passed their own
    architecture gate. Checked independently for each child so the failure
    message names exactly which one(s) blocked VC-4."""
    failing = [
        child
        for child, report_path in (("VC-1", vc1_report), ("VC-2", vc2_report))
        if not _report_passed_architecture_gate(report_path)
    ]
    if failing:
        raise VC4ArchitectureGateRefused(
            f"VC-4 refuses: {failing} report(s) did not pass their own "
            "architecture gate (or the report file has no "
            f"{_ARCHITECTURE_GATE_KEY!r} field)"
        )


def build_vc4_run_manifest(
    *,
    adaptive_v0_policy_hash: str,
    seed: int,
    ratification_record: S7RatificationRecord | None,
) -> dict[str, Any]:
    """This runner's own manifest identity -- stage `S7`, child `VC-4`,
    never borrowed from VC-1's or VC-2's manifest schemas."""
    manifest: dict[str, Any] = {
        "s7_vc4_run_manifest_schema_version": "0.1.0",
        "stage": "S7",
        "child": "VC-4",
        "adaptive_v0_policy_hash": adaptive_v0_policy_hash,
        "seed": seed,
        "ratification_decision_id": (
            ratification_record.decision_id if ratification_record is not None else None
        ),
    }
    manifest["s7_vc4_run_manifest_hash"] = domain_hash(_S7_VC4_MANIFEST_DOMAIN, manifest)
    return manifest


def run_vc4_combined_dry_run(
    *,
    vc1_report: Path,
    vc2_report: Path,
    compressor: AdaptiveVisualMemoryCompressor,
    memory: VisualMemory,
    vc5_policy: Mapping[str, Any],
    dry_run: bool,
    ratification_record: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """The gated VC-4 entry point.

    Refuses via :func:`require_vc1_and_vc2_architecture_gates` before
    anything else runs. The combined arm's own VC-5 evidence is built from
    the POST (VC-1/`adaptive_v0`) compressor's `architecture_contract`
    report -- the dominant of the erratum's four VC-5 conditions -- with no
    quality/safety/performance metrics supplied yet, so
    `build_vc5_gate_report` reports `insufficient_evidence` for the other
    three rather than a fabricated pass.
    """
    require_vc1_and_vc2_architecture_gates(vc1_report, vc2_report)
    record = require_ratification(
        child="VC-4", dry_run=dry_run, ratification_record=ratification_record
    )
    architecture_report = check_architecture_contract(
        compressor,
        memory,
        implementation_hash=IMPLEMENTATION_HASH,
        policy_hash=compressor.policy.hash(),
    )
    vc5_gate = build_vc5_gate_report(
        architecture_contract_report=architecture_report,
        bypass_quality_metrics=None,
        compressed_quality_metrics=None,
        safety_metrics=None,
        performance_deltas=None,
        vc5_policy=vc5_policy,
    )
    manifest = build_vc4_run_manifest(
        adaptive_v0_policy_hash=compressor.policy.hash(),
        seed=seed,
        ratification_record=record,
    )
    return {"vc5_gate": vc5_gate, "manifest": manifest, "dry_run": dry_run}


def _write_child_report(path: Path, *, architecture_contract_passed: bool) -> None:
    path.write_text(
        json.dumps({_ARCHITECTURE_GATE_KEY: architecture_contract_passed}), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RFC005-ERRATA-007 VC-4 dormant combined runner")
    parser.add_argument("--out", type=Path, required=True, help="output JSON report path")
    parser.add_argument("--vc1-report", type=Path, required=True)
    parser.add_argument("--vc2-report", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the stub composition (the only mode implemented)",
    )
    parser.add_argument("--ratification-record", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    compressor = AdaptiveVisualMemoryCompressor(d_model=8)
    tokens = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8)
    memory = VisualMemory(
        tokens=tokens,
        key_padding_mask=torch.zeros(1, 4, dtype=torch.bool),
        page_ordinals=torch.zeros(1, dtype=torch.long),
    )
    vc5_policy: dict[str, Any] = {}
    result = run_vc4_combined_dry_run(
        vc1_report=args.vc1_report,
        vc2_report=args.vc2_report,
        compressor=compressor,
        memory=memory,
        vc5_policy=vc5_policy,
        dry_run=args.dry_run,
        ratification_record=args.ratification_record,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

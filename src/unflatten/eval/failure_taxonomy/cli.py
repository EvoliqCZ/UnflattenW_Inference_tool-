"""`failure-taxonomy-report` CLI (RFC005-ERRATA-007 E1).

    python -m unflatten.eval.failure_taxonomy --sealed <artefact.json> \
        --cache-split <exact-text split dir> [--chain-report <json>] \
        [--summary <summary.json>] --out <dir>

The role check runs on the artefact alone, before the cache split is
opened: a locked-test or P6 artefact never reaches the gold reader.
Writes `failure_taxonomy_report.json` and `failure_taxonomy_report.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from unflatten.eval import cohort_gold as cg
from unflatten.eval.failure_taxonomy import report as ft
from unflatten.eval.s2_gate import sealed

JSON_NAME = "failure_taxonomy_report.json"
MD_NAME = "failure_taxonomy_report.md"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="failure-taxonomy-report")
    parser.add_argument("--sealed", type=Path, required=True, help="sealed cohort artefact JSON")
    parser.add_argument(
        "--cache-split", type=Path, required=True,
        help="exact-text cache split holding the cohort's gold rows (e.g. .../val_iid)",
    )
    parser.add_argument(
        "--chain-report", type=Path, default=None,
        help="builder_chain_report.json for per-page strict_valid outcomes",
    )
    parser.add_argument(
        "--summary", type=Path, default=None,
        help="summary.json of scripts/render_model_docx_with_defaults.py (per-page failure codes); "
        "takes precedence over --chain-report when both are given",
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    raw = args.sealed.read_bytes()
    artefact = sealed.artefact_from_json(raw.decode("utf-8"))
    ft.check_role(artefact.role)

    bundles = [p.document_bundle_id for p in artefact.pages]
    gold_rows = cg.load_gold_rows(args.cache_split, bundles)
    entries = cg.split_entries(args.cache_split)

    pipeline = None
    pipeline_source = None
    if args.summary is not None:
        pipeline = ft.pipeline_outcomes_from_summary(json.loads(args.summary.read_text()))
        pipeline_source = str(args.summary)
    elif args.chain_report is not None:
        pipeline = ft.pipeline_outcomes_from_chain_report(json.loads(args.chain_report.read_text()))
        pipeline_source = str(args.chain_report)

    report = ft.build_report(
        artefact,
        gold_rows,
        pipeline=pipeline,
        split_entries=entries,
        inputs={
            "sealed_path": str(args.sealed),
            # hash of the FILE; the artefact's file name carries the checkpoint
            # sha256 (`eval_cache_identity.checkpoint_sha256`), not this one
            "sealed_file_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "cache_split": str(args.cache_split),
            "pipeline_source": pipeline_source,
        },
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / JSON_NAME).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    (args.out / MD_NAME).write_text(ft.render_markdown(report))
    headline = {
        r["failure_class"]: f"{r['frequency']['count']}/{r['frequency']['denominator']}"
        for r in report["classes"]
    }
    print(json.dumps({"report_hash": report["report_hash"], "accounting": report["accounting"],
                      "classes": headline}, ensure_ascii=False), file=sys.stdout)
    return 0

"""`train-val-gap` CLI (RFC005-ERRATA-007 E3).

    python -m unflatten.eval.failure_taxonomy.train_val_gap_cli \\
        --train-report <failure_taxonomy_report.json over a train sample> \\
        --val-iid-report <failure_taxonomy_report.json over val_iid> \\
        --out <dir>

Takes two already-built `failure_taxonomy_report.json` files (produced by
`unflatten.eval.failure_taxonomy.cli`), never a sealed artefact directly --
this tool is pure CPU and never opens a cache split. Writes
`train_val_gap.json` and `train_val_gap.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from unflatten.eval.failure_taxonomy import train_val_gap as tvg

JSON_NAME = "train_val_gap.json"
MD_NAME = "train_val_gap.md"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train-val-gap")
    parser.add_argument("--train-report", type=Path, required=True,
                        help="failure_taxonomy_report.json over a train sample")
    parser.add_argument("--val-iid-report", type=Path, required=True,
                        help="failure_taxonomy_report.json over val_iid")
    parser.add_argument("--seed", type=int, default=tvg.DEFAULT_SEED)
    parser.add_argument("--resamples", type=int, default=tvg.DEFAULT_RESAMPLES)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    train_report = json.loads(args.train_report.read_text())
    val_iid_report = json.loads(args.val_iid_report.read_text())
    gap = tvg.gap_report(train_report, val_iid_report, seed=args.seed, resamples=args.resamples)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / JSON_NAME).write_text(json.dumps(gap, indent=1, ensure_ascii=False) + "\n")
    (args.out / MD_NAME).write_text(tvg.render_markdown(gap))
    headline = {r["failure_class"]: r["gap"] for r in gap["classes"]}
    print(json.dumps({"checkpoint_sha256": gap["checkpoint_sha256"], "gap": headline},
                      ensure_ascii=False), file=sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())

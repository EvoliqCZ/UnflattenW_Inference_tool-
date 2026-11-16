"""Preseed a new S4 run's pinned eval pages from its parent run (RFC005 §40.4).

`s4_train.load_or_select_eval_pages` reads `s4_eval_pages.json` verbatim when
it exists, so a child run that must keep the parent's eval pages (so the
selection cohort stays disjoint from them) only has to have the file written
before the first launch. The byte format is the one `load_or_select_eval_pages`
writes: `json.dumps({"pages": ...}, sort_keys=True) + "\\n"`.

CLI: `python -m unflatten.model.s4_eval_pages --source P --run-dir D
--val-expectation E.json --selection-policy S.json --eval-pages N`; prints
the `sha256:` of the written file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from unflatten.model.exact_text_cache import CacheExpectation

_EVAL_PAGES_FILENAME = "s4_eval_pages.json"


class EvalPagesRefused(ValueError):
    """The eval page set cannot be preseeded; nothing was written."""


def preseed_eval_pages(
    source: Path,
    run_dir: Path,
    *,
    val_expectation: CacheExpectation,
    selection_policy: Mapping[str, Any],
    eval_pages: int,
) -> str:
    """Copy `source` into `run_dir/s4_eval_pages.json`; return `sha256:<hex>`
    over the written bytes. Refuses (writing nothing) on an existing
    destination, a page count other than `eval_pages`, a page outside
    `val_expectation.target_keys()`, or overlap with the selection cohort."""
    destination = run_dir / _EVAL_PAGES_FILENAME
    if destination.exists():
        raise EvalPagesRefused(f"{destination} already exists; refusing to overwrite")
    try:
        raw_pages = json.loads(source.read_text())["pages"]
        pages = [(str(bundle), int(page)) for bundle, page in raw_pages]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EvalPagesRefused(f"{source}: unreadable eval pages: {exc!r}") from exc
    if len(pages) != eval_pages:
        raise EvalPagesRefused(
            f"{source} holds {len(pages)} pages but eval_pages is {eval_pages}"
        )
    allowed = set(val_expectation.target_keys())
    outside = sorted(p for p in pages if p not in allowed)
    if outside:
        raise EvalPagesRefused(
            f"eval pages outside the val expectation: {outside}"
        )
    pinned = {(str(b), int(p)) for b, p in selection_policy["cohort"]["pinned_pages"]}
    overlap = sorted(set(pages) & pinned)
    if overlap:
        raise EvalPagesRefused(
            f"eval pages overlap the selection cohort on {overlap} (RFC005 §40.4)"
        )
    payload = json.dumps({"pages": [list(p) for p in pages]}, sort_keys=True) + "\n"
    data = payload.encode()
    destination.write_bytes(data)
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="s4_eval_pages",
        description="Preseed s4_eval_pages.json in a new run dir from a parent run.",
    )
    parser.add_argument("--source", type=Path, required=True,
                        help="parent run's s4_eval_pages.json")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="new run dir (must not hold s4_eval_pages.json)")
    parser.add_argument("--val-expectation", type=Path, required=True,
                        help="CacheExpectation payload JSON of the val cache")
    parser.add_argument("--selection-policy", type=Path, required=True,
                        help="frozen selection policy JSON (cohort.pinned_pages)")
    parser.add_argument("--eval-pages", type=int, required=True,
                        help="expected number of eval pages")
    args = parser.parse_args(argv)
    try:
        expectation = CacheExpectation.from_payload(
            json.loads(args.val_expectation.read_text())
        )
        policy = json.loads(args.selection_policy.read_text())
        digest = preseed_eval_pages(
            args.source, args.run_dir, val_expectation=expectation,
            selection_policy=policy, eval_pages=args.eval_pages,
        )
    except (EvalPagesRefused, OSError, ValueError, KeyError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI of the gold screen (DEC-047 addendum A v4, design P8).

``python -m unflatten.model.rl.gold_screen_cli`` runs the gold chain over the pages of a
frozen selection and writes ``gold-screen.json`` below ``--out`` -- the report
``data_prep build-dataset-manifest --gold-screen-report`` reads (``passed`` / ``refused``)
plus a ``provenance`` block. CPU only, no model, no render (``render_dir=None``), never
trains (no ``torch.optim``). The chain flags are the reward policy's; there is no flag
for them. An existing ``gold-screen.json`` is never overwritten.

Exit codes: 0 screened (refused pages are a result, not a failure), 2 named refusal
(one line on stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.model.rl import data_prep_support as sup

EXIT_OK, EXIT_REFUSED = 0, 2
REPORT_FILE = "gold-screen.json"
SCHEMA = "unflatten.rl.gold-screen/1"


@dataclass
class Seams:
    """Test/CPU seams; ``None`` = the real implementation."""

    #: ``(cache_dir, CacheExpectation) -> reader`` addressing ``expectation.target_keys()``
    open_reader: Callable[[Path, Any], Any] | None = None
    #: ``(cache identity, allowlist payload) -> composite vocabulary``
    make_vocabulary: Callable[[dict[str, Any], dict[str, Any]], Any] | None = None
    load_tokenizer: Callable[[], Any] | None = None
    #: raster fetch seam of ``PageRasterAdapter``
    fetch: Callable[[Path, Any], bytes] | None = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gold_screen_cli",
        description="Screen the gold chain over a frozen selection (CPU, no model, no render).",
    )
    for name in (
        "release-dir",
        "store-dir",
        "cache-dir",
        "expectation",
        "selection",
        "reward-policy",
        "allowlist",
        "out",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    return p


def _real_open_reader(cache_dir: Path, expectation: Any) -> Any:
    from unflatten.model.exact_text_cache import ExactTextCacheReader

    return ExactTextCacheReader(cache_dir, gradient_bearing=True, expectation=expectation)


def _real_vocabulary(identity: dict[str, Any], allowlist: dict[str, Any]) -> Any:
    from unflatten.model import composite_vocab as cv
    from unflatten.model.rl.adapters.gold_screen import GoldScreenRefused

    vocabulary = cv.make_composite_vocabulary(
        tokenizer_artifact_set_hash=identity["tokenizer_artifact_set_hash"],
        allowlist_payload=allowlist,
    )
    if vocabulary.artifact_hash != identity["composite_vocabulary_hash"]:
        raise GoldScreenRefused(
            f"vocabulary.artifact_hash {vocabulary.artifact_hash!r} != cache "
            f"composite_vocabulary_hash {identity['composite_vocabulary_hash']!r}: "
            "not the vocabulary the cache was built with"
        )
    return vocabulary


def _real_tokenizer() -> Any:
    from unflatten.model.qwen_tokenizer_artifacts import load_pinned_tokenizer

    return load_pinned_tokenizer()[0]


def _named_refusals() -> tuple[type[BaseException], ...]:
    from unflatten.model.exact_text_cache import ExactTextCacheIncompatible

    return (ValueError, OSError, ExactTextCacheIncompatible)


def _run(args: argparse.Namespace, seams: Seams) -> dict[str, Any]:
    from unflatten.model.exact_text_cache import CacheExpectation
    from unflatten.model.rl import chain_adapter
    from unflatten.model.rl.adapters import gold_screen as gs
    from unflatten.model.rl.adapters.admission import cache_identity_hash
    from unflatten.model.rl.adapters.release import (
        load_bundle_source,
        read_split_records_verified,
    )
    from unflatten.model.rl.adapters.taxonomy_assembly import bundle_lineage_from_split_records
    from unflatten.model.rl.policies import REWARD_HASH_FIELD, verify_reward_policy
    from unflatten.model.rl.selection import selection_from_document, selection_hash

    out = sup.out_dir(str(args.out))
    sup.require_free(out, [REPORT_FILE])
    reward_policy = sup.read_json(args.reward_policy, "reward policy")
    verify_reward_policy(reward_policy)  # first: a policy that does not verify screens nothing
    selection = selection_from_document(sup.read_json(args.selection, "selection"))
    identity = dict(sup.cache_identity(str(args.cache_dir)))
    expectation = CacheExpectation.from_payload(sup.read_json(args.expectation, "expectation"))
    allowlist = sup.read_json(args.allowlist, "allowlist")
    lineage = bundle_lineage_from_split_records(read_split_records_verified(args.release_dir))
    reader = (seams.open_reader or _real_open_reader)(args.cache_dir, expectation)
    vocabulary = (seams.make_vocabulary or _real_vocabulary)(identity, allowlist)
    tokenizer = (seams.load_tokenizer or _real_tokenizer)()
    screen, counters = gs.screen_with_counters(
        reader=reader,
        expectation=expectation,
        selection=selection,
        reward_policy=reward_policy,
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        store_dir=args.store_dir,
        bundle_lineage=lineage,
        bundle_source=load_bundle_source(args.store_dir),
        fetch=seams.fetch,
    )
    summary = screen.summary()
    document = {
        **sup.gold_screen_document(screen),
        "provenance": {
            "schema": SCHEMA,
            "reward_policy_hash": reward_policy[REWARD_HASH_FIELD],
            "chain_policy_flags": dict(reward_policy["chain_policy_flags"]),
            "table_geometry_column_source": chain_adapter.TABLE_GEOMETRY_COLUMN_SOURCE,
            "selection_hash": selection_hash(selection),
            "cache_identity_hash": cache_identity_hash(identity),
            "counters": dict(sorted(counters.items())),
        },
    }
    sup.write_docs(out, {REPORT_FILE: document})
    return {
        "passed": summary["passed"],
        "refused": summary["refused"],
        "refused_by_counter": dict(sorted(summary["by_kind"].items())),
        "lineages_without_pass": gs.lineages_without_pass(
            screen, lineage, set(selection.lineage_ids())
        ),
        "report": str(out / REPORT_FILE),
    }


def main(argv: Sequence[str] | None = None, *, seams: Seams | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = _run(args, seams or Seams())
    except _named_refusals() as exc:
        line = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        print(f"REFUSED {line}", file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps(summary, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

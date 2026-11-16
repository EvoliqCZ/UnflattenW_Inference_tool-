"""S2 lineage pool and plan freeze (T5a, DEC-027 D3/D6).

Two pure, torch-free functions build the sampler's input over a release's
split records (:func:`load_split_records` -- every split manifest, the same
authority `dataset.samples._split_membership` gives the cache builder):

- :func:`build_lineage_pool` -- one pool record per `lineage_root_id`
  covering the bundles a frozen `CacheExpectation` (DEC-MP-4) targets,
  each carrying DEC-027 D6's neutral weights (no producer of
  `feature_coverage_milli`/`hardness_score_milli`/`features` exists yet)
  and DEC-027 D3's single eligible cell -- the matrix carries all its mass
  on `train/P4/structure_v0/docxcorp`, so a synthetic per-lineage weight
  for the other three (zero-mass) cells would be speculation, not signal.
- :func:`pages_by_lineage` -- the `E \\ X` pages of each lineage, the raw
  material :func:`select_page` draws from at run time.

`select_page` is the view-choice half of §30.1/§30.2: the plan draws
LINEAGES (mass applies before fragment/view/augmentation expansion), and
choosing which page of a drawn lineage to show is a separate, seed-keyed
draw recorded in the exposure ledger's `view_ids` -- independent of the
global RNG, and a pure function of `(run_seed, draw_index)` so a resumed
run reproduces the identical page for the identical draw.

The CLI composes both with `s2_sampler.build_ordered_exposure_plan` and
freezes `ordered_exposure_plan.json` under `--run-dir`, refusing to
overwrite one that already exists -- the same discipline as every other
freeze CLI in this package.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from unflatten.dataset.samples import RELEASE_SPLITS
from unflatten.model.exact_text_cache import CacheExpectation, PageKey
from unflatten.model.s2_sampler import build_ordered_exposure_plan
from unflatten.model.stage_data_matrix import load_stage_data_matrix

#: DEC-027 D3: the matrix carries all its mass on this single cell; every
#: pool record is eligible for exactly it.
ELIGIBLE_CELL: dict[str, str] = {
    "split": "train",
    "pyramid_level": "P4",
    "task_kind": "structure_v0",
    "cohort": "docxcorp",
}

#: DEC-027 D6: neutral weights, since no producer of feature coverage,
#: hardness or rare-feature signal exists yet.
_NEUTRAL_MILLI = 1000

#: The committed policy directory the CLI reads from by default (overridable
#: in tests via `monkeypatch.setattr(s2_plan, "_POLICIES_DIR", ...)`).
_POLICIES_DIR = Path(__file__).parent / "policies"

_ORDERED_EXPOSURE_PLAN_FILENAME = "ordered_exposure_plan.json"


class S2PlanRefused(ValueError):
    """The lineage pool, the page partition, or the page draw cannot be
    built truthfully from the presented inputs: an expectation naming a
    bundle no split record resolves, a bundle with more than one record, a
    record without a split assignment or assigned to a split other than
    `train`, or an empty page sequence. A refusal, never a repair."""


def load_split_records(release_dir: Path | str) -> list[dict[str, Any]]:
    """Every split manifest's records, each tagged with its `split_name`.

    Mirrors the cache builder's authority (`samples._split_membership`):
    `partial.jsonl` carries the real split in `split_name` (a multi-page
    document with some pages ready is a `partial_ready` member of `train`);
    the per-split manifests are named for theirs. Page-level training
    eligibility is NOT decided here -- the builder decided it per task view
    and the pinned expectation records the result. A missing manifest file
    contributes nothing (as in the builder)."""
    records: list[dict[str, Any]] = []
    for name in (*RELEASE_SPLITS, "partial"):
        path = Path(release_dir) / "splits" / f"{name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                record.setdefault("split_name", name)
                records.append(record)
    return records


def _index_split_records(split_records: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    """`{document_bundle_id: record}`, refusing a duplicate bundle -- two
    train records for one bundle make its lineage/eligibility ambiguous."""
    by_bundle: dict[str, dict] = {}
    for record in split_records:
        bundle = record["authority_id"]
        if bundle in by_bundle:
            raise S2PlanRefused(
                f"bundle {bundle!r} appears more than once in the split records "
                "-- a bundle's lineage/eligibility must resolve to exactly one "
                "train record"
            )
        by_bundle[bundle] = dict(record)
    return by_bundle


def _resolve_bundle(
    bundle: str, by_bundle: Mapping[str, dict], split_name: str = "train"
) -> dict:
    record = by_bundle.get(bundle)
    if record is None:
        raise S2PlanRefused(
            f"bundle {bundle!r} (named by the pinned cache expectation) does "
            "not resolve to any record in the presented split manifests -- it "
            f"must resolve to exactly one record assigned to split {split_name!r}"
        )
    split = record.get("split_name")
    if split is None:
        raise S2PlanRefused(
            f"bundle {bundle!r} resolves to a record without split_name -- its "
            "split assignment is unknown and is never assumed "
            "(read records with load_split_records)"
        )
    if split != split_name:
        raise S2PlanRefused(
            f"bundle {bundle!r} is assigned to split {split!r}, not {split_name!r} -- "
            "a pool never draws another split's document"
        )
    return record


def build_lineage_pool(
    *,
    expectation: CacheExpectation,
    split_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One pool record per `lineage_root_id` covering the bundles
    `expectation.target_keys()` names, sorted by `lineage_root_id`."""
    by_bundle = _index_split_records(split_records)
    bundles = {bundle for bundle, _page in expectation.target_keys()}
    lineage_roots: set[str] = set()
    for bundle in bundles:
        record = _resolve_bundle(bundle, by_bundle)
        lineage_roots.add(record["lineage_root_id"])
    return [
        {
            "lineage_root_id": lineage_root_id,
            "eligible_cells": [dict(ELIGIBLE_CELL)],
            "feature_coverage_milli": _NEUTRAL_MILLI,
            "hardness_score_milli": _NEUTRAL_MILLI,
            "features": [],
        }
        for lineage_root_id in sorted(lineage_roots)
    ]


def pages_by_lineage(
    expectation: CacheExpectation,
    split_records: Iterable[Mapping[str, Any]],
    *,
    split_name: str = "train",
) -> dict[str, tuple[PageKey, ...]]:
    """`{lineage_root_id: sorted E \\ X pages}` -- the raw material
    :func:`select_page` draws a page from at run time. `split_name` is the
    one split every page must belong to: `train` for a training pool,
    `val_iid` for the periodic eval subset."""
    by_bundle = _index_split_records(split_records)
    by_lineage: dict[str, list[PageKey]] = {}
    for bundle, page in expectation.target_keys():
        record = _resolve_bundle(bundle, by_bundle, split_name)
        by_lineage.setdefault(record["lineage_root_id"], []).append((bundle, page))
    return {
        lineage_root_id: tuple(sorted(pages))
        for lineage_root_id, pages in by_lineage.items()
    }


def select_page(*, run_seed: int, draw_index: int, pages: Sequence[PageKey]) -> PageKey:
    """The deterministic page a drawn lineage shows at `draw_index`.

    Independent of the global RNG and of `train_steps`'s own page order:
    a pure function of `(run_seed, draw_index)`, so a resumed run
    reproduces the identical page for the identical draw."""
    if not pages:
        raise S2PlanRefused("cannot select a page from an empty pages sequence")
    rng = random.Random(f"s2-page/{run_seed}/{draw_index}")
    return pages[rng.randrange(len(pages))]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.s2_plan",
        description="Build the S2 lineage pool and freeze its ordered exposure plan.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    release_dir: Path = args.release_dir

    expectation = CacheExpectation.from_payload(
        json.loads((run_dir / "exact_text_cache_expectation.json").read_text())
    )
    split_records = load_split_records(release_dir)
    matrix = load_stage_data_matrix(_POLICIES_DIR / "s2_stage_data_matrix.json")
    sampler_policy = json.loads((_POLICIES_DIR / "s2_sampler_policy.json").read_text())
    exposure_policy = json.loads((_POLICIES_DIR / "s2_exposure_policy.json").read_text())

    pool = build_lineage_pool(expectation=expectation, split_records=split_records)
    plan = build_ordered_exposure_plan(
        matrix=matrix,
        lineages=pool,
        sampler_policy=sampler_policy,
        exposure_policy=exposure_policy,
        run_seed=args.seed,
    )

    plan_path = run_dir / _ORDERED_EXPOSURE_PLAN_FILENAME
    if plan_path.exists():
        raise S2PlanRefused(
            f"{plan_path} already exists -- refusing to overwrite a frozen "
            "ordered exposure plan"
        )
    plan_path.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "ordered_exposure_plan_hash": plan["ordered_exposure_plan_hash"],
                "lineage_count": len(pool),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

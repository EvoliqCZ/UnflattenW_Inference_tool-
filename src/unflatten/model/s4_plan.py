"""S4 data matrix and plan freeze (DEC-034, rescope §4.4 variant (a)).

S2/S3 draw every train lineage from the single cell
`train/P4/structure_v0/docxcorp` (DEC-027 D3), synthetic bundles included.
S4 separates them: a synthetic lineage -- every bundle of it carries
`envelope.source_origin == "synthetic_ast"` in the store index -- is
eligible only for `train/P2/structure_v0/synthetic`, a real one only for
the docxcorp cell, and the exposure ledger records which cell each draw
came from.

DEC-034's 20 % is a CAP, not a target. The synthetic mass is computed here
at freeze time from the pool it will be drawn from (a runtime recount is
forbidden, §30.3), and it is the largest mass whose binomial draw count
stays 4 sigma inside the cell's capacity `lineages x max_views`:
`m*E + 4*sqrt(E*m*(1-m)) <= C` with `m <= 0.20`. The sampler draws the cell
per draw, so a mass equal to the capacity would dry the cell out in about
half of all seeds (§30.7 then refuses the plan). The real cell carries the
rest. The frozen matrix is a run artefact (`stage_data_matrix.json` in the
run dir), hashed into the S4 run manifest like the plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from unflatten.model import s2_plan, s4_boost_sets
from unflatten.model.exact_text_cache import CacheExpectation, PageKey
from unflatten.model.hashing import domain_hash, make_id
from unflatten.model.run_binding import compute_exact_text_cache_expectation_hash
from unflatten.model.s2_sampler import (
    ORDERED_EXPOSURE_PLAN_DOMAIN,
    ORDERED_EXPOSURE_PLAN_SCHEMA_VERSION,
    build_ordered_exposure_plan,
    s2_exposure_policy_hash,
    s2_sampler_policy_hash,
    validate_ordered_exposure_plan,
)
from unflatten.model.stage_data_matrix import build_stage_data_matrix, stage_data_matrix_hash

REAL_CELL: dict[str, str] = dict(s2_plan.ELIGIBLE_CELL)
SYNTHETIC_CELL: dict[str, str] = {
    "split": "train",
    "pyramid_level": "P2",
    "task_kind": "structure_v0",
    "cohort": "synthetic",
}

#: DEC-034 point 2: synthetic mass is at most 20 % of lineage-root draws.
SYNTHETIC_MASS_CAP_MICRO = 200_000
_TOTAL_MICRO = 1_000_000
#: Rescope §4.4 (a): the reserve against the binomial draw count, in sigma.
_SIGMA_RESERVE = 4

_SYNTHETIC_ORIGIN = "synthetic_ast"
_POLICIES_DIR = Path(__file__).parent / "policies"
_MATRIX_FILENAME = "stage_data_matrix.json"
_PLAN_FILENAME = "ordered_exposure_plan.json"


class S4PlanRefused(ValueError):
    """The S4 matrix or plan cannot be frozen truthfully from the presented
    store, release and expectation. A refusal, never a repair."""


def load_synthetic_bundles(store_dir: Path | str) -> frozenset[str]:
    """Bundles whose store envelope says `source_origin: synthetic_ast`.

    A row without `source_origin`, or a bundle whose rows disagree, refuses:
    an unclassifiable bundle would silently land in the real cell."""
    origins: dict[str, str] = {}
    with open(Path(store_dir) / "index.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            envelope = json.loads(line)["envelope"]
            bundle = envelope["document_bundle_id"]
            origin = envelope.get("source_origin")
            if origin is None:
                raise S4PlanRefused(
                    f"store row for {bundle!r} carries no source_origin -- a bundle "
                    "is never assumed real"
                )
            if origins.setdefault(bundle, origin) != origin:
                raise S4PlanRefused(
                    f"store rows for {bundle!r} disagree on source_origin "
                    f"({origins[bundle]!r} vs {origin!r})"
                )
    return frozenset(b for b, o in origins.items() if o == _SYNTHETIC_ORIGIN)


def build_s4_lineage_pool(
    *,
    expectation: CacheExpectation,
    split_records: Iterable[Mapping[str, Any]],
    synthetic_bundles: frozenset[str],
) -> list[dict[str, Any]]:
    """`s2_plan.build_lineage_pool` with each lineage moved to the cell its
    bundles belong to; a lineage mixing synthetic and real bundles refuses."""
    split_records = list(split_records)
    pool = s2_plan.build_lineage_pool(expectation=expectation, split_records=split_records)
    pages = s2_plan.pages_by_lineage(expectation, split_records)
    for record in pool:
        lineage = record["lineage_root_id"]
        kinds = {bundle in synthetic_bundles for bundle, _page in pages[lineage]}
        if len(kinds) != 1:
            raise S4PlanRefused(
                f"lineage {lineage!r} mixes synthetic and real bundles -- it cannot "
                "belong to exactly one cell"
            )
        record["eligible_cells"] = [dict(SYNTHETIC_CELL if kinds.pop() else REAL_CELL)]
    return pool


def synthetic_mass_micro(*, synthetic_lineages: int, epoch_size: int, max_views: int) -> int:
    """The largest mass (micro) with `m*E + 4*sigma <= lineages*max_views`,
    capped at 20 %. Exact integer arithmetic: with `m = x/1e6` the rule is
    `C*1e6 - x*E >= 0` and `16*E*x*(1e6 - x) <= (C*1e6 - x*E)**2`."""
    capacity = synthetic_lineages * max_views

    def fits(x: int) -> bool:
        slack = capacity * _TOTAL_MICRO - x * epoch_size
        return slack >= 0 and (
            _SIGMA_RESERVE**2 * epoch_size * x * (_TOTAL_MICRO - x) <= slack * slack
        )

    lo, hi = 0, SYNTHETIC_MASS_CAP_MICRO
    if fits(hi):
        return hi
    while hi - lo > 1:  # fits(lo) and not fits(hi); monotone below m = 0.5
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


def build_s4_matrix(*, synthetic_mass_micro: int) -> dict[str, Any]:
    real_mass = _TOTAL_MICRO - synthetic_mass_micro
    cells = [
        {**REAL_CELL, "mass_micro": real_mass},
        {**SYNTHETIC_CELL, "mass_micro": synthetic_mass_micro},
        {**REAL_CELL, "pyramid_level": "P2", "mass_micro": 0},
        {**SYNTHETIC_CELL, "pyramid_level": "P4", "mass_micro": 0},
    ]
    return build_stage_data_matrix(
        matrix_version="1.0.0",
        stage_id="S4",
        splits=["train"],
        pyramid_levels=["P2", "P4"],
        task_kinds=["structure_v0"],
        cohorts=["docxcorp", "synthetic"],
        cells=cells,
        declared_marginals={
            "cohort": {"docxcorp": real_mass, "synthetic": synthetic_mass_micro},
        },
    )


#: DEC-041 owner decision (page-uniform rework): the weight fields a draw
#: still must carry (closed `ordered_exposure_plan.schema.json`) mean
#: nothing under page-uniform sampling -- there is no weighted draw to
#: record -- so every draw carries the schema's own neutral multiplier.
_NEUTRAL_MILLI = 1000
_PAGE_UNIFORM_META_FILENAME = "s4_page_uniform_plan.json"
#: RFC005-ERRATA-014 E3: boost draws may add at most this many milli of an
#: epoch's page count (per epoch, hence also in total).
BOOST_CAP_MILLI = 120
#: Review MUST FIX 1: page-uniform mode REWRITES `virtual_epoch_size`/
#: `max_views_per_lineage_per_virtual_epoch` and hashes the rewritten
#: values into the plan -- `s4_run_manifest`/`s4_train` must verify
#: against those EFFECTIVE policies, not the committed 15 000-budget
#: files, or a truthfully-frozen page-uniform plan is refused at
#: admission (measured: `sampler_policy_hash` mismatch). These are run
#: artefacts, written into the run dir alongside the plan.
EFFECTIVE_SAMPLER_POLICY_FILENAME = "s4_sampler_policy.effective.json"
EFFECTIVE_EXPOSURE_POLICY_FILENAME = "s4_exposure_policy.effective.json"


def build_page_lineage_map(
    expectation: CacheExpectation,
    split_records: Iterable[Mapping[str, Any]],
    *,
    split_name: str = "train",
) -> dict[PageKey, str]:
    """`{page: lineage_root_id}`, inverted from `s2_plan.pages_by_lineage`
    -- every page in `expectation.target_keys()` belongs to exactly one
    lineage (the pool it comes from is deduplicated by construction), so
    the inversion is total and unambiguous."""
    by_lineage = s2_plan.pages_by_lineage(expectation, split_records, split_name=split_name)
    page_lineage: dict[PageKey, str] = {}
    for lineage, pages in by_lineage.items():
        for page in pages:
            page_lineage[page] = lineage
    return page_lineage


def page_uniform_epoch_order(
    pages: Sequence[PageKey], *, run_seed: int, epoch: int
) -> list[PageKey]:
    """A deterministic full-coverage shuffle of `pages` for one epoch pass
    (owner decision, DEC-041: "po stránkach: každá strana presne N×") --
    every page appears exactly once, order seeded by `(run_seed, epoch)`,
    so different epochs of the SAME run see different orders and the same
    `(run_seed, epoch)` always reproduces the identical order."""
    rng = random.Random(f"s4-page-epoch/{run_seed}/{epoch}")
    order = list(pages)
    rng.shuffle(order)
    return order


def build_page_uniform_order(
    pages: Sequence[PageKey], *, run_seed: int, epochs: int
) -> list[PageKey]:
    """`epochs` deterministic passes over `pages`, concatenated -- every
    page appears exactly `epochs` times, in `epochs` distinct
    (seed, epoch)-derived shuffles."""
    order: list[PageKey] = []
    for epoch in range(epochs):
        order.extend(page_uniform_epoch_order(pages, run_seed=run_seed, epoch=epoch))
    return order


def load_augmentation_ledger(path: Path | str) -> dict[PageKey, str]:
    """`{(document_bundle_id, page_index): view_id}` from the JSONL ledger
    `m2b_augment` writes, one line per augmented page. The view id is read
    from `render_view_id` -- RFC003's name for it, and the key the writer
    actually emits (`identity.render_view_id`). This reader first read
    `view_id`, matching only its own hand-written test rows; the real
    ledger would have raised KeyError here (writer-reader contract test in
    tests/dataset/migration/test_m2b_augment.py). This is a HOOK for T5a's augmentation
    data -- the ledger only names WHICH pages have a variant and what its
    view id is; resolving a `view_id` into actual cache features is not
    this function's job. A page named twice with disagreeing view ids
    refuses -- an ambiguous ledger is never guessed."""
    ledger: dict[PageKey, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["document_bundle_id"], row["page_index"])
            view_id = row["render_view_id"]
            if key in ledger and ledger[key] != view_id:
                raise S4PlanRefused(
                    f"augmentation ledger {path}: {key!r} is named twice with "
                    f"disagreeing view_id ({ledger[key]!r} vs {view_id!r})"
                )
            ledger[key] = view_id
    return ledger


def augmented_epoch_indices(*, augmented_slots: int, epochs: int) -> frozenset[int]:
    """The 0-based epoch indices (the FIRST `augmented_slots` of `epochs`)
    in which a variant-page's slot serves its augmented view rather than
    its clean render -- deterministic and auditable (DEC-041 point 2),
    never randomized. `augmented_slots == epochs` (the default) reproduces
    DEC-041's own arithmetic exactly: with the default, a variant page is
    NEVER seen clean."""
    if not (0 <= augmented_slots <= epochs):
        raise S4PlanRefused(
            f"augmented_slots ({augmented_slots}) must be between 0 and epochs "
            f"({epochs})"
        )
    return frozenset(range(augmented_slots))


def effective_page_uniform_policies(
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
    *,
    epochs: int,
    pages: int,
    draw_limit: int | None = None,
    boost_draws: int = 0,
    boost_factor: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The EFFECTIVE `(sampler_policy, exposure_policy)` a page-uniform
    plan actually hashes: `virtual_epoch_size = pages * epochs` (the
    total draw count, DEC-041 point 1-2) -- or `draw_limit` when the plan
    is a prefix (RFC005-ERRATA-012 LR tail: the run's whole budget is that
    prefix) -- and `max_views_per_lineage_per_virtual_epoch = epochs` (a
    page-uniform "lineage" -- see `build_page_uniform_plan`'s own draws --
    is drawn AT MOST once per epoch, so a lineage owning a page sees at
    most `epochs` views). The ONE place this rewrite happens, so `main()`
    can write these same values to disk for `s4_run_manifest`/`s4_train`
    second time.

    Page-uniform-boost (RFC005-ERRATA-014 E3): `virtual_epoch_size` also
    counts the `boost_draws`, and the per-lineage field becomes
    `epochs * boost_factor` -- a per-PAGE bound (each page is drawn at most
    `boost_factor` times per epoch); a lineage owning several boosted pages
    can exceed it, the sidecar's `max_lineage_views_in_epoch` is the truth."""
    return (
        {
            **sampler_policy,
            "virtual_epoch_size": (
                pages * epochs + boost_draws if draw_limit is None else draw_limit
            ),
        },
        {**exposure_policy, "max_views_per_lineage_per_virtual_epoch": epochs * boost_factor},
    )


def _boost_epoch_order(
    base: list[PageKey], boost: list[PageKey], *, run_seed: int, epoch: int
) -> list[tuple[PageKey, bool]]:
    """`base` with `boost` interleaved at seeded random positions (never a
    block): `(page, is_boost)` in draw order."""
    total = len(base) + len(boost)
    rng = random.Random(f"s4-page-boost/{run_seed}/{epoch}")
    positions = sorted(rng.sample(range(total), len(boost)))
    shuffled = list(boost)
    rng.shuffle(shuffled)
    boost_at = dict(zip(positions, shuffled, strict=True))
    base_iter = iter(base)
    return [
        (boost_at[i], True) if i in boost_at else (next(base_iter), False) for i in range(total)
    ]


def load_exclude_pages(path: Path | str) -> list[PageKey]:
    """`(bundle, page_index)` rows of an exclusion JSONL (`{document_bundle_id, page_index, ...}`);
    extra fields are provenance and ignored. A duplicate row is refused: two rows for one page
    mean the list was built wrong."""
    pages: list[PageKey] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                pages.append((str(row["document_bundle_id"]), int(row["page_index"])))
    if len(set(pages)) != len(pages):
        raise S4PlanRefused(f"{path}: duplicate page rows in the exclusion list")
    return pages


def exclude_pages_sha256(path: Path | str) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_page_uniform_plan(
    *,
    expectation: CacheExpectation,
    split_records: Iterable[Mapping[str, Any]],
    synthetic_bundles: frozenset[str],
    run_seed: int,
    epochs: int,
    sampler_policy: Mapping[str, Any],
    exposure_policy: Mapping[str, Any],
    augmentation_ledger: Mapping[PageKey, str] | None = None,
    augmented_slots: int | None = None,
    draw_limit: int | None = None,
    boost_sets: Mapping[str, Any] | None = None,
    boost_sets_sha256: str | None = None,
    boost_factor: int = 1,
    exclude_pages: Collection[PageKey] | None = None,
    exclude_pages_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The page-uniform ordered exposure plan (owner decision, DEC-041:
    page-uniform, not lineage-uniform -- "každá strana presne N×").

    `draw_limit` (RFC005-ERRATA-012, the LR tail): keep only the FIRST
    `draw_limit` draws of the `epochs` passes -- a prefix of the seeded
    order, so the kept pages are a uniform draw without replacement and
    every page is seen AT MOST `epochs` times, no longer exactly. The
    effective `virtual_epoch_size` becomes `draw_limit` (the run's whole
    budget), the sidecar records `draw_limit`, and `s2_train.require_plan_
    covers_budget` then holds for a `steps == draw_limit` run.

    `boost_sets` (RFC005-ERRATA-014 E3, the `s4_boost_sets` payload, with
    `boost_sets_sha256` the sha of its file): every page in a set is drawn
    `boost_factor` times per epoch (the others once) -- the extra draws are
    interleaved at seeded positions, capped at `BOOST_CAP_MILLI` of the
    epoch and refused together with `draw_limit` (the tail never boosts).
    The sidecar then carries `meta["boost"]`; without boost it is unchanged.

    Returns `(plan, page_uniform_meta)`:

    - `plan` conforms EXACTLY to the shared, closed
      `ordered_exposure_plan.schema.json` -- the same fields
      `build_ordered_exposure_plan` produces, so `s4_train`/
      `s4_run_manifest` accept it unchanged. `lineage_root_id` is still
      the true document lineage (unrelated to how the page inside it was
      chosen); the weight fields are the schema's own neutral value
      (`_NEUTRAL_MILLI`) because there is no weighted draw to record.
    - `page_uniform_meta` is an S4-OWNED sidecar (not part of the shared
      schema, never validated against it) carrying what the schema has no
      room for: the actual page and clean/augmented decision per draw,
      plus the DEC-041 point 4 summary (epochs, totals, clean/augmented
      view counts, the augmentation ledger's identity, the slot
      parameter). `s4_train` reads this sidecar directly for page
      identity instead of `s2_plan.select_page` (which cannot express
      "every page exactly N times" -- it is a per-draw RNG pick over a
      lineage's own page list, unsuitable here since one document can
      hold hundreds of pages).

    `exclude_pages` (owner 29. 10. 2026, whole-page images out of training; with
    `exclude_pages_sha256` the sha of its file): those pages are never drawn, also
    not as boost draws. Every listed page must be in the expectation -- a list
    built for another cache is refused, not half-applied. The sidecar then
    carries `meta["excluded_pages"]`; without the list it is unchanged.

    Refuses (never guesses a mixing rule) if any page belongs to a
    synthetic bundle: DEC-041's production cohort is real pages only, and
    page-uniform mode does not define how a multi-cell matrix with
    synthetic mass would be walked page-by-page.
    """
    if epochs < 1:
        raise S4PlanRefused(f"epochs must be >= 1, got {epochs}")
    split_records = list(split_records)
    pages = list(expectation.target_keys())
    if not pages:
        raise S4PlanRefused("page-uniform mode: the expectation names no pages")
    if exclude_pages is not None:
        if exclude_pages_sha256 is None:
            raise S4PlanRefused("exclude_pages needs exclude_pages_sha256 (the file's identity)")
        excluded = {(str(b), int(p)) for b, p in exclude_pages}
        unknown = sorted(excluded - set(pages))
        if unknown:
            raise S4PlanRefused(
                f"{len(unknown)} excluded page(s) not in the cache expectation "
                f"(first {unknown[0]!r}): the list was built for another cache"
            )
        pages = [page for page in pages if page not in excluded]

    synthetic_pages = [page for page in pages if page[0] in synthetic_bundles]
    if synthetic_pages:
        raise S4PlanRefused(
            f"page-uniform mode does not support synthetic mass -- "
            f"{len(synthetic_pages)} page(s) of the cohort belong to a synthetic "
            f"bundle (first {synthetic_pages[0]!r}); refusing rather than guessing "
            "a mixing rule for a multi-cell matrix"
        )

    page_lineage = build_page_lineage_map(expectation, split_records)
    missing = [page for page in pages if page not in page_lineage]
    if missing:
        raise S4PlanRefused(
            f"page-uniform mode: {len(missing)} page(s) resolve to no lineage under "
            f"the presented split records (first {missing[0]!r})"
        )

    if augmented_slots is None:
        augmented_slots = epochs
    augmented_epochs = augmented_epoch_indices(augmented_slots=augmented_slots, epochs=epochs)
    ledger = dict(augmentation_ledger) if augmentation_ledger else {}
    if draw_limit is not None and not (
        type(draw_limit) is int and 1 <= draw_limit <= len(pages) * epochs
    ):
        raise S4PlanRefused(
            f"draw_limit must be an int in [1, pages x epochs = {len(pages) * epochs}], "
            f"got {draw_limit!r}"
        )

    boost: list[PageKey] = []
    if boost_sets is not None:
        if draw_limit is not None:
            raise S4PlanRefused("page-uniform boost cannot be combined with draw_limit (the tail)")
        if boost_sets_sha256 is None:
            raise S4PlanRefused("page-uniform boost needs boost_sets_sha256 (the file's identity)")
        try:
            boost = s4_boost_sets.boost_draw_pages(boost_sets, pages, factor=boost_factor)
        except ValueError as exc:
            raise S4PlanRefused(f"page-uniform boost: {exc}") from exc
        if len(boost) * 1000 > BOOST_CAP_MILLI * len(pages):
            raise S4PlanRefused(
                f"page-uniform boost: {len(boost)} boost draw(s) per epoch exceed the cap of "
                f"{BOOST_CAP_MILLI} milli of {len(pages)} page(s)"
            )
    elif boost_factor != 1:
        raise S4PlanRefused("boost_factor != 1 needs boost_sets")

    sampler_policy, exposure_policy = effective_page_uniform_policies(
        sampler_policy, exposure_policy, epochs=epochs, pages=len(pages), draw_limit=draw_limit,
        boost_draws=len(boost) * epochs, boost_factor=boost_factor,
    )
    virtual_epoch_size = sampler_policy["virtual_epoch_size"]
    matrix = build_s4_matrix(synthetic_mass_micro=0)
    matrix_hash = stage_data_matrix_hash(matrix)
    sampler_hash = s2_sampler_policy_hash(sampler_policy)
    exposure_hash = s2_exposure_policy_hash(exposure_policy)

    draws: list[dict[str, Any]] = []
    meta_draws: list[dict[str, Any]] = []
    lineage_views: dict[str, int] = {}
    clean_views = 0
    augmented_views = 0
    draw_index = 0
    lineage_epoch_views: dict[tuple[str, int], int] = {}
    for epoch in range(epochs):
        base = page_uniform_epoch_order(pages, run_seed=run_seed, epoch=epoch)
        if boost_sets is None:
            ordered = [(page, False) for page in base]
        else:
            ordered = _boost_epoch_order(base, boost, run_seed=run_seed, epoch=epoch)
        for page, is_boost in ordered:
            lineage = page_lineage[page]
            lineage_views[lineage] = lineage_views.get(lineage, 0) + 1
            lineage_epoch_views[lineage, epoch] = lineage_epoch_views.get((lineage, epoch), 0) + 1
            variant_view_id = ledger.get(page)
            is_augmented = variant_view_id is not None and epoch in augmented_epochs
            if is_augmented:
                augmented_views += 1
            else:
                clean_views += 1
            draws.append(
                {
                    "draw_index": draw_index,
                    "split": REAL_CELL["split"],
                    "pyramid_level": REAL_CELL["pyramid_level"],
                    "task_kind": REAL_CELL["task_kind"],
                    "cohort": REAL_CELL["cohort"],
                    "lineage_root_id": lineage,
                    "feature_coverage_milli": _NEUTRAL_MILLI,
                    "hardness_milli": _NEUTRAL_MILLI,
                    "w_lineage": _NEUTRAL_MILLI,
                    "w_rare_milli": _NEUTRAL_MILLI,
                    "total_weight": _NEUTRAL_MILLI,
                    "lineage_views_in_epoch": lineage_views[lineage],
                }
            )
            meta_draw = {
                "draw_index": draw_index,
                "document_bundle_id": page[0],
                "page_index": page[1],
                "epoch": epoch,
                "kind": "augmented" if is_augmented else "clean",
                "view_id": variant_view_id if is_augmented else None,
            }
            if is_boost:
                meta_draw["boost"] = True
            meta_draws.append(meta_draw)
            draw_index += 1

    if draw_limit is not None:
        # A prefix of the seeded order: draw_index stays its position, the
        # cumulative `lineage_views_in_epoch` of every kept draw is unchanged.
        draws = draws[:draw_limit]
        meta_draws = meta_draws[:draw_limit]
        augmented_views = sum(1 for d in meta_draws if d["kind"] == "augmented")
        clean_views = len(meta_draws) - augmented_views

    plan: dict[str, Any] = {
        "ordered_exposure_plan_schema_version": ORDERED_EXPOSURE_PLAN_SCHEMA_VERSION,
        "run_seed": run_seed,
        "virtual_epoch_size": virtual_epoch_size,
        "matrix_hash": matrix_hash,
        "sampler_policy_hash": sampler_hash,
        "exposure_policy_hash": exposure_hash,
        "feature_lineage_counts": {},
        "draws": draws,
    }
    plan["plan_id"] = make_id("plan", ORDERED_EXPOSURE_PLAN_DOMAIN, plan)
    plan["ordered_exposure_plan_hash"] = domain_hash(ORDERED_EXPOSURE_PLAN_DOMAIN, plan)
    validate_ordered_exposure_plan(plan)

    ledger_sha256 = None
    if augmentation_ledger is not None:
        # CBOR profile map keys must be text strings (`cbor._encode_map`);
        # a `(bundle, page)` tuple key is hashed as a sorted row list instead.
        ledger_rows = [[bundle, page, view_id] for (bundle, page), view_id in ledger.items()]
        ledger_sha256 = domain_hash(
            "rfc005.s4-augmentation-ledger/v1", sorted(ledger_rows)
        )

    meta = {
        "epochs": epochs,
        "pages": len(pages),
        "total_draws": len(draws),
        "clean_views": clean_views,
        "augmented_views": augmented_views,
        "augmented_slots": augmented_slots,
        "augmentation_ledger_sha256": ledger_sha256,
        "draw_limit": draw_limit,
        "draws": meta_draws,
    }
    if exclude_pages is not None:
        meta["excluded_pages"] = {"sha256": exclude_pages_sha256, "count": len(excluded)}
    if boost_sets is not None:
        meta["boost"] = {
            "sets_sha256": boost_sets_sha256,
            "factor": boost_factor,
            "draws": len(boost) * epochs,
            "cap_milli": BOOST_CAP_MILLI,
            "set_sizes": {name: len(members) for name, members in boost_sets["sets"].items()},
            "max_lineage_views_in_epoch": max(lineage_epoch_views.values()),
        }
    return plan, meta


def _load_boost_sets(path: Path, expectation: CacheExpectation) -> dict[str, Any]:
    """The boost-sets payload, refused unless it is a `s4_boost_sets` file built
    for THIS run's cache expectation."""
    payload = json.loads(path.read_text())
    if payload.get("schema") != s4_boost_sets.BOOST_SETS_SCHEMA:
        raise S4PlanRefused(f"{path}: schema is not {s4_boost_sets.BOOST_SETS_SCHEMA!r}")
    expected = compute_exact_text_cache_expectation_hash(expectation.payload())
    if payload.get("expectation_hash") != expected:
        raise S4PlanRefused(
            f"{path} was built for expectation {payload.get('expectation_hash')!r}, "
            f"not this run's {expected!r}"
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.s4_plan",
        description="Freeze the S4 stage data matrix and its ordered exposure plan.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--sampling-mode", choices=("weighted", "page-uniform"), default="weighted",
        help="'weighted' (default): the original §30.3 lineage-weighted sampler, "
        "unchanged. 'page-uniform' (DEC-041, owner decision): every train page "
        "drawn exactly --epochs times.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="page-uniform mode ONLY: passes over the page cohort. Refused under "
        "--sampling-mode weighted if not 1 -- 'weighted' has no epochs concept "
        "(the DEC-041 defect this replaced was exactly that ambiguity).",
    )
    parser.add_argument(
        "--augmentation-ledger", type=Path, default=None,
        help="page-uniform mode only: JSONL {document_bundle_id, page_index, "
        "view_id} naming which pages have an augmented variant (T5a hook).",
    )
    parser.add_argument(
        "--augmented-slots", type=int, default=None,
        help="page-uniform mode only: of a variant page's --epochs slots, how many "
        "serve the augmented view (default: --epochs, DEC-041's own arithmetic).",
    )
    parser.add_argument(
        "--draw-limit", type=int, default=None,
        help="page-uniform mode only (RFC005-ERRATA-012 LR tail): keep only the first N "
        "draws of the seeded order -- the run's whole step budget; every page is then "
        "seen at most --epochs times.",
    )
    parser.add_argument(
        "--boost-sets", type=Path, default=None,
        help="page-uniform mode only (RFC005-ERRATA-014 E3): the frozen `s4_boost_sets` file; "
        "its pages are drawn --boost-factor times per epoch.",
    )
    parser.add_argument(
        "--boost-factor", type=int, default=1,
        help="draws per epoch of every boost-set page (default 1 = no extra draws); "
        ">1 requires --boost-sets.",
    )
    parser.add_argument(
        "--exclude-pages", type=Path, default=None,
        help="page-uniform mode only (owner 29. 10. 2026): JSONL {document_bundle_id, page_index} "
        "of pages never drawn, e.g. whole-page images.",
    )
    args = parser.parse_args(argv)
    if args.sampling_mode == "weighted" and args.exclude_pages is not None:
        raise S4PlanRefused("--exclude-pages is only meaningful under --sampling-mode page-uniform")
    if args.sampling_mode == "weighted" and args.boost_sets is not None:
        raise S4PlanRefused("--boost-sets is only meaningful under --sampling-mode page-uniform")
    if args.boost_factor != 1 and args.boost_sets is None:
        raise S4PlanRefused("--boost-factor > 1 requires --boost-sets")
    if args.boost_sets is not None and args.draw_limit is not None:
        raise S4PlanRefused("--boost-sets cannot be combined with --draw-limit (the tail)")
    if args.sampling_mode == "weighted" and args.epochs != 1:
        raise S4PlanRefused(
            "--epochs is only meaningful under --sampling-mode page-uniform "
            "(the weighted sampler has no epochs concept -- DEC-041 found "
            "exactly this ambiguity a defect)"
        )
    if args.sampling_mode == "weighted" and (
        args.augmentation_ledger is not None
        or args.augmented_slots is not None
        or args.draw_limit is not None
    ):
        raise S4PlanRefused(
            "--augmentation-ledger/--augmented-slots/--draw-limit are only meaningful "
            "under --sampling-mode page-uniform"
        )

    run_dir: Path = args.run_dir
    protected_names = (
        (_MATRIX_FILENAME, _PLAN_FILENAME, _PAGE_UNIFORM_META_FILENAME,
         EFFECTIVE_SAMPLER_POLICY_FILENAME, EFFECTIVE_EXPOSURE_POLICY_FILENAME)
        if args.sampling_mode == "page-uniform"
        else (_MATRIX_FILENAME, _PLAN_FILENAME)
    )
    for name in protected_names:
        if (run_dir / name).exists():
            raise S4PlanRefused(
                f"{run_dir / name} already exists -- refusing to overwrite a frozen artefact"
            )

    expectation = CacheExpectation.from_payload(
        json.loads((run_dir / "exact_text_cache_expectation.json").read_text())
    )
    sampler_policy = json.loads((_POLICIES_DIR / "s4_sampler_policy.json").read_text())
    exposure_policy = json.loads((_POLICIES_DIR / "s4_exposure_policy.json").read_text())
    split_records = s2_plan.load_split_records(args.release_dir)
    synthetic_bundles = load_synthetic_bundles(args.store_dir)

    if args.sampling_mode == "page-uniform":
        ledger = (
            load_augmentation_ledger(args.augmentation_ledger)
            if args.augmentation_ledger is not None
            else None
        )
        boost_payload = (
            _load_boost_sets(args.boost_sets, expectation) if args.boost_sets is not None else None
        )
        plan, meta = build_page_uniform_plan(
            expectation=expectation,
            split_records=split_records,
            synthetic_bundles=synthetic_bundles,
            run_seed=args.seed,
            epochs=args.epochs,
            sampler_policy=sampler_policy,
            exposure_policy=exposure_policy,
            augmentation_ledger=ledger,
            augmented_slots=args.augmented_slots,
            draw_limit=args.draw_limit,
            boost_sets=boost_payload,
            boost_sets_sha256=(
                s4_boost_sets.boost_sets_sha256(args.boost_sets)
                if args.boost_sets is not None else None
            ),
            boost_factor=args.boost_factor,
            exclude_pages=(
                load_exclude_pages(args.exclude_pages) if args.exclude_pages is not None else None
            ),
            exclude_pages_sha256=(
                exclude_pages_sha256(args.exclude_pages) if args.exclude_pages is not None else None
            ),
        )
        matrix = build_s4_matrix(synthetic_mass_micro=0)
        effective_sampler_policy, effective_exposure_policy = effective_page_uniform_policies(
            sampler_policy, exposure_policy, epochs=args.epochs, pages=meta["pages"],
            draw_limit=args.draw_limit,
            boost_draws=meta["boost"]["draws"] if "boost" in meta else 0,
            boost_factor=args.boost_factor,
        )

        (run_dir / _MATRIX_FILENAME).write_text(
            json.dumps(matrix, indent=1, sort_keys=True) + "\n"
        )
        (run_dir / _PLAN_FILENAME).write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
        (run_dir / _PAGE_UNIFORM_META_FILENAME).write_text(
            json.dumps(meta, indent=1, sort_keys=True) + "\n"
        )
        # Review MUST FIX 1: the run artefacts `s4_run_manifest`/`s4_train`
        # must verify the plan against -- the SAME rewritten policies
        # `build_page_uniform_plan` just hashed into it, not the committed
        # 15 000-budget files.
        (run_dir / EFFECTIVE_SAMPLER_POLICY_FILENAME).write_text(
            json.dumps(effective_sampler_policy, indent=1, sort_keys=True) + "\n"
        )
        (run_dir / EFFECTIVE_EXPOSURE_POLICY_FILENAME).write_text(
            json.dumps(effective_exposure_policy, indent=1, sort_keys=True) + "\n"
        )
        print(
            json.dumps(
                {
                    "sampling_mode": "page-uniform",
                    "stage_data_matrix_hash": stage_data_matrix_hash(matrix),
                    "ordered_exposure_plan_hash": plan["ordered_exposure_plan_hash"],
                    "epochs": meta["epochs"],
                    "pages": meta["pages"],
                    "total_draws": meta["total_draws"],
                    "clean_views": meta["clean_views"],
                    "augmented_views": meta["augmented_views"],
                    "augmented_slots": meta["augmented_slots"],
                    "augmentation_ledger_sha256": meta["augmentation_ledger_sha256"],
                    "draw_limit": meta["draw_limit"],
                    **(
                        {"boost_draws": meta["boost"]["draws"]} if "boost" in meta else {}
                    ),
                },
                sort_keys=True,
            )
        )
        return 0

    pool = build_s4_lineage_pool(
        expectation=expectation, split_records=split_records, synthetic_bundles=synthetic_bundles,
    )
    n_synthetic = sum(r["eligible_cells"] == [SYNTHETIC_CELL] for r in pool)
    mass = synthetic_mass_micro(
        synthetic_lineages=n_synthetic,
        epoch_size=sampler_policy["virtual_epoch_size"],
        max_views=exposure_policy["max_views_per_lineage_per_virtual_epoch"],
    )
    matrix = build_s4_matrix(synthetic_mass_micro=mass)
    plan = build_ordered_exposure_plan(
        matrix=matrix,
        lineages=pool,
        sampler_policy=sampler_policy,
        exposure_policy=exposure_policy,
        run_seed=args.seed,
    )

    (run_dir / _MATRIX_FILENAME).write_text(json.dumps(matrix, indent=1, sort_keys=True) + "\n")
    (run_dir / _PLAN_FILENAME).write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "sampling_mode": "weighted",
                "stage_data_matrix_hash": stage_data_matrix_hash(matrix),
                "ordered_exposure_plan_hash": plan["ordered_exposure_plan_hash"],
                "lineage_count": len(pool),
                "synthetic_lineages": n_synthetic,
                "synthetic_mass_micro": mass,
                "synthetic_draws": sum(d["cohort"] == "synthetic" for d in plan["draws"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

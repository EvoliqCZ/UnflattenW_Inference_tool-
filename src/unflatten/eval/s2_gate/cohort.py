"""Produce the S2 exit-gate cohort manifest (`s2_exit_iid_provisional`).

The provisional v11 manifest was written by a script outside the repo; this
is its in-repo producer, used for release-v12 under the S2 profile
(2026-09-15). Every disposition is DERIVED from the two caches, never
declared -- the rule `decode.verify_cohort_accounting` checks:

- a requested page with NO feature-cache row takes the feature builder's own
  skip reason (`build_cache.prepare_page`, plus the capability check its
  build loop applies): `skipped_grammar` -> `grammar_refused`,
  `skipped_too_long` -> `over_length`, `skipped_capability_excluded` ->
  `capability_excluded`, `skipped_unplaced`/`skipped_no_observation`/
  `skipped_fully_unsupervised` -> `unplaced`/`no_observation`/
  `fully_unsupervised`; any other reason is refused (`FEATURE_SKIPS_REFUSED`);
- a page WITH a feature row is `attempt` if the exact-text cache wrote it;
  otherwise its disposition comes from the exact-text build's per-page skip
  row (`page_skip_rows`): `skipped_text_coverage` ->
  `withdrawn_text_coverage_floor`, `skipped_text_encoding` ->
  `text_encoding_failed`, `skipped_projection_invariant` ->
  `projection_invariant`, `skipped_grammar` -> `exact_text_grammar_refused`,
  any other reason or a missing row is refused (a replay-skipped page looks
  identical and must never be relabelled);
- an exact-text cache built before `page_skip_rows` existed falls back to
  counts: below the coverage floor -> `withdrawn_text_coverage_floor`,
  otherwise `text_encoding_failed`, admitted only when the split-wide
  `skipped_text_encoding` count equals the derived one and every other
  exact-text skip bucket is zero -- which a DEC-033 slice of the split can
  never satisfy, so that path refuses and says to rebuild.

The gate's decode scope is `single_page` (`decode._SCOPE_KIND`): the cohort
is single-page documents, one page each.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from unflatten.eval.s2_gate import decode as dm

PageKey = tuple[str, int]

#: DEC-033 §12: the S2 exit gate is measured over two slices. The main
#: number stays comparable with S2 v12b (single-page documents only); the
#: `multi_page_pages` slice reports pages from multi-page documents,
#: separately, not gated (design §9). `SINGLE_PAGE_DOCS_SLICE` is the
#: default -- with it, `selection_rule` carries no `"slice"` key at all, so
#: a manifest built today is byte-identical to one built before this slice
#: existed (`test_no_sample_manifest_unchanged`).
SINGLE_PAGE_DOCS_SLICE = "single_page_docs"
MULTI_PAGE_PAGES_SLICE = "multi_page_pages"
COHORT_SLICES = (SINGLE_PAGE_DOCS_SLICE, MULTI_PAGE_PAGES_SLICE)

#: `role` per slice (design §9): `s2_exit_iid_provisional` stays the
#: default-slice role so nothing downstream that keys on it breaks.
ROLE_BY_SLICE = {
    SINGLE_PAGE_DOCS_SLICE: "s2_exit_iid_provisional",
    MULTI_PAGE_PAGES_SLICE: "s2_exit_iid_multipage_pages",
}

FEATURE_SKIP_TO_DISPOSITION: dict[str, str] = {
    "skipped_grammar": "grammar_refused",
    "skipped_too_long": "over_length",
    "skipped_capability_excluded": "capability_excluded",
    "skipped_unplaced": "unplaced",
    "skipped_no_observation": "no_observation",
    "skipped_fully_unsupervised": "fully_unsupervised",
}

#: Every other page-level reason `build_cache.prepare_page` can give, named
#: so none of them falls through a silent default: each refuses the cohort
#: build (the `_continuation` counters are subsets, never a page's reason).
#: `skipped_no_supervision_scope` is a store defect (a fragment written
#: before §26), `skipped_unencodable` has no gate disposition -- deciding
#: one is a separate call, not a fallback.
FEATURE_SKIPS_REFUSED = frozenset({"skipped_unencodable", "skipped_no_supervision_scope"})

#: The exact-text skip buckets a cohort page may legitimately carry; any
#: other non-zero bucket makes `text_encoding_failed` ambiguous.
_ATTRIBUTABLE_EXACT_SKIPS = frozenset({"skipped_text_coverage", "skipped_text_encoding"})

#: Per page: `build_exact_text_cache.page_skip_rows` names why each
#: unwritten page was skipped, and only the reasons below are a gate
#: disposition. A page skipped for any other reason (replay, typed mask, ...)
#: has a feature row and no written row exactly like an encoding failure
#: does -- naming it is the whole point, so an unnamed one refuses.
EXACT_SKIP_TO_DISPOSITION: dict[str, str] = {
    "skipped_text_encoding": "text_encoding_failed",
    "skipped_text_coverage": dm.sealed.WITHDRAWN_DISPOSITION,
    # `page_projection.PageProjectionInvariantViolation`, recorded as a skip.
    "skipped_projection_invariant": "projection_invariant",
    # The owner-accepted cell-tail loss (RFC004-ERRATA-002 E1 point 2).
    "skipped_grammar": "exact_text_grammar_refused",
}

#: Every other exact-text reason, named: the page is refused, never
#: relabelled (replay/typed-mask/... look exactly like an encoding failure).
EXACT_SKIPS_REFUSED = frozenset(
    {
        "skipped_unencodable",
        "skipped_no_observation",
        "skipped_no_cached_features",
        "skipped_no_sidecar",
        "skipped_schema_mismatch",
        "skipped_typed_mask",
        "skipped_text_authority",
        "skipped_replay",
        "skipped_realisation_mismatch",
    }
)

#: Leakage components are not materialised in the release (RFC003 §40.1 is
#: cited, not measured) -- the same literal the v11 manifest carried.
UNMATERIALISED_LEAKAGE_COMPONENT = "unmaterialised:leakage-word-core-v0"


class CohortBuildRefused(ValueError):
    """The cohort cannot be derived truthfully from the presented caches."""


def assign_dispositions(
    *,
    requested: Sequence[PageKey],
    feature_keys: Iterable[PageKey],
    written_keys: Iterable[PageKey],
    below_floor_keys: Iterable[PageKey],
    feature_skips: Mapping[PageKey, str | None],
    exact_text_skip_counts: Mapping[str, int],
    cache_universe: Iterable[PageKey] | None = None,
    exact_text_skip_rows: Mapping[PageKey, str] | None = None,
) -> dict[PageKey, str]:
    """`{page: disposition}` for every requested page (see module doc).

    `exact_text_skip_rows` is `{page: reason}` from the exact-text build's
    own per-page rows (`build_exact_text_cache`'s `page_skip_rows`). When it
    is present it is AUTHORITATIVE for every requested page that has a
    feature row and no written row: the reason is mapped through
    `EXACT_SKIP_TO_DISPOSITION` and any other reason -- or a missing row --
    refuses by name. When it is absent (an exact-text cache built before
    those rows existed) the old count-based check runs instead: the build's
    split-wide bucket totals must equal the derived per-disposition counts.
    That check cannot hold for a slice of a split (DEC-033 §12), and it
    cannot attribute a single page at all, which is why the per-page rows
    exist; its refusals say so.

    `cache_universe` is the page universe the feature cache may legitimately
    hold; stray rows are judged against it, and it defaults to `requested`.
    A DEC-033 slice (`single_page_docs` / `multi_page_pages`) is a VIEW over
    one split-wide cache: the cache is built once for the whole split, and
    each slice asks for a subset of it, so the slice's own `requested` set
    is not the right yardstick for "this cache was built for a different
    request". The universe is the full split cohort under the same release,
    store and capability profile -- a cache row outside THAT still means a
    wrong split/release/profile and still refuses. Dispositions and the
    accounting derived from them stay over `requested` only.
    """
    requested_set = set(requested)
    feature_keys = set(feature_keys)
    universe = requested_set if cache_universe is None else set(cache_universe)
    stray = sorted(feature_keys - universe)
    if stray:
        raise CohortBuildRefused(
            f"feature cache holds {len(stray)} page(s) outside the requested cohort, "
            f"e.g. {stray[0]} -- the cache was not built for this request"
        )
    written = set(written_keys)
    below_floor = set(below_floor_keys)
    dispositions: dict[PageKey, str] = {}
    for key in requested:
        if key not in feature_keys:
            reason = feature_skips.get(key)
            if reason not in FEATURE_SKIP_TO_DISPOSITION:
                raise CohortBuildRefused(
                    f"page {key} has no feature row and its feature-build reason is "
                    f"{reason!r} -- only {sorted(FEATURE_SKIP_TO_DISPOSITION)} map to a "
                    "gate disposition"
                )
            dispositions[key] = FEATURE_SKIP_TO_DISPOSITION[reason]
        elif key in written:
            dispositions[key] = dm.sealed.ATTEMPT_DISPOSITION
        elif exact_text_skip_rows is not None:
            reason = exact_text_skip_rows.get(key)
            if reason not in EXACT_SKIP_TO_DISPOSITION:
                raise CohortBuildRefused(
                    f"page {key} has a feature row, no written row, and the exact-text "
                    f"build's per-page skip reason is {reason!r} -- only "
                    f"{sorted(EXACT_SKIP_TO_DISPOSITION)} map to a gate disposition, "
                    "refusing to attribute"
                )
            dispositions[key] = EXACT_SKIP_TO_DISPOSITION[reason]
        elif key in below_floor:
            dispositions[key] = dm.sealed.WITHDRAWN_DISPOSITION
        else:
            dispositions[key] = "text_encoding_failed"

    if exact_text_skip_rows is not None:
        return dispositions

    counted = Counter(dispositions.values())
    for bucket, count in sorted(exact_text_skip_counts.items()):
        if bucket in _ATTRIBUTABLE_EXACT_SKIPS or bucket.endswith("_continuation"):
            continue
        if count:
            raise CohortBuildRefused(
                f"exact-text build reports {bucket}={count}: such a page has a feature row "
                "and no written row, indistinguishable from an encoding failure -- refusing "
                "to attribute. This cache carries no per-page skip rows, so only split-wide "
                "totals can be compared; rebuild the exact-text cache to attribute per page"
            )
    checks = (
        ("skipped_text_encoding", "text_encoding_failed"),
        ("skipped_text_coverage", dm.sealed.WITHDRAWN_DISPOSITION),
    )
    for bucket, disposition in checks:
        if exact_text_skip_counts.get(bucket, 0) != counted[disposition]:
            raise CohortBuildRefused(
                f"exact-text build reports {bucket}={exact_text_skip_counts.get(bucket, 0)} "
                f"but the caches derive {counted[disposition]} {disposition} page(s). This "
                "cache carries no per-page skip rows, so a split-wide total is all there is "
                "to compare against -- it cannot match a slice of the split (DEC-033 §12); "
                "rebuild the exact-text cache to attribute per page"
            )
    return dispositions


def _population_pages_sha256(dispositions: Mapping[PageKey, str]) -> str:
    """sha256 hex digest over the sorted `"bundle\\tpage"` lines of every
    page in `dispositions` -- pins the population a sample was drawn from
    (DEC-029)."""
    lines = sorted(f"{bundle}\t{page}" for bundle, page in dispositions)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def sample_dispositions(
    dispositions: Mapping[PageKey, str], *, size: int, seed: int
) -> dict[PageKey, str]:
    """Rank every `(bundle, page)` key by `sha256(f"{seed}:{bundle}:{page}")`
    and keep the first `size`, deterministic for a given `(size, seed)`
    (DEC-029). Refuses a size below 1 or above the population."""
    population = len(dispositions)
    if size < 1:
        raise CohortBuildRefused(f"sample size {size} < 1")
    if size > population:
        raise CohortBuildRefused(
            f"sample size {size} exceeds population {population}"
        )
    ranked = sorted(
        dispositions,
        key=lambda key: hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).hexdigest(),
    )
    chosen = ranked[:size]
    return {key: dispositions[key] for key in sorted(chosen)}


def apply_slice_to_selection_rule(
    selection_rule: dict[str, Any], slice_name: str
) -> dict[str, Any]:
    """Mutates and returns `selection_rule`, adding `"slice": slice_name`
    only when `slice_name` is not `SINGLE_PAGE_DOCS_SLICE` -- a
    default-slice manifest built today must stay byte-identical to one
    built before this slice existed (`test_no_sample_manifest_unchanged`).
    Refuses an unknown slice name rather than silently recording it."""
    if slice_name not in COHORT_SLICES:
        raise CohortBuildRefused(
            f"unknown cohort slice {slice_name!r} -- must be one of {COHORT_SLICES}"
        )
    if slice_name != SINGLE_PAGE_DOCS_SLICE:
        selection_rule["slice"] = slice_name
    return selection_rule


def _resolve_sample_args(
    sample_pages: int | None, sample_seed: int | None
) -> tuple[int, int] | None:
    """`(size, seed)` when both `--sample-pages`/`--sample-seed` are given,
    `None` when neither is -- refuses a partial pair (DEC-029)."""
    if sample_pages is None and sample_seed is None:
        return None
    if sample_pages is None:
        raise CohortBuildRefused("--sample-seed given without --sample-pages")
    if sample_seed is None:
        raise CohortBuildRefused("--sample-pages given without --sample-seed")
    return sample_pages, sample_seed


def build_cohort_manifest(
    *,
    role: str,
    dispositions: Mapping[PageKey, str],
    lineages: Mapping[str, str],
    selection_rule: Mapping[str, Any],
) -> dict[str, Any]:
    """The sealed manifest payload, verified through the gate's own loader
    (`load_cohort_manifest` + `verify_cohort_accounting`) before return."""
    pages = [
        {
            "document_bundle_id": bundle,
            "lineage_root_id": lineages[bundle],
            "leakage_component_id": UNMATERIALISED_LEAKAGE_COMPONENT,
            "disposition": disposition,
        }
        for (bundle, _page), disposition in sorted(dispositions.items())
    ]
    rule = {
        **dict(selection_rule),
        "requested_bundles_sha256": dm.requested_bundles_sha256(
            page["document_bundle_id"] for page in pages
        ),
    }
    accounting = dm.compute_cohort_accounting(
        [dm.CohortPage(**page) for page in pages]
    ).as_dict()
    raw: dict[str, Any] = {
        "role": role,
        "policy_hash": dm.selection_rule_policy_hash(rule),
        "selection_rule": rule,
        "expected_accounting": accounting,
        "pages": pages,
    }
    raw["sha256"] = dm.cohort_manifest_sha256(raw)
    dm.verify_cohort_accounting(dm.load_cohort_manifest(raw))
    return raw


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - real-data wiring
    from unflatten.model import build_cache as bc
    from unflatten.model.build_exact_text_cache import verify_feature_grammar_binding
    from unflatten.model.capability_profile import resolve_capability_profile_by_id
    from unflatten.model.exact_text_cache import ExactTextCacheReader, cache_manifest
    from unflatten.model.feature_cache import FeatureCacheReader

    parser = argparse.ArgumentParser(
        prog="python -m unflatten.eval.s2_gate.cohort", description=__doc__
    )
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("store_dir", type=Path)
    parser.add_argument("feature_cache_dir", type=Path)
    parser.add_argument("exact_text_cache_dir", type=Path)
    parser.add_argument("--exact-text-summary", type=Path, required=True)
    parser.add_argument("--capability-profile", required=True)
    parser.add_argument("--split", default="val_iid")
    parser.add_argument(
        "--slice",
        choices=COHORT_SLICES,
        default=SINGLE_PAGE_DOCS_SLICE,
        help="DEC-033 S2 gate cohort slice: single-page documents (the main "
        "number, comparable with S2 v12b) or pages from multi-page "
        "documents (reported, not gated)",
    )
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--render-dpi", type=int, default=300)
    parser.add_argument("--sample-pages", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise CohortBuildRefused(f"{args.out} already exists -- refusing to overwrite")
    sample_args = _resolve_sample_args(args.sample_pages, args.sample_seed)

    profile = resolve_capability_profile_by_id(args.capability_profile)
    # The split-wide cohort: what the one feature cache behind both slices
    # may legitimately hold (see `assign_dispositions`). Enumerated through
    # the builder's own planner, so the universe cannot drift from the
    # slice filter it is a superset of.
    split_plan = bc._plan_build(
        args.release_dir, args.store_dir, args.feature_cache_dir,
        split=args.split, task_kind="structure_v0", allow_dev_only=False,
        single_page_only=False, multi_page_only=False,
        only_bundles=None, only_bundles_path=None,
    )
    cache_universe = {(s.document_bundle_id, s.page_index) for s in split_plan.samples}
    plan = bc._plan_build(
        args.release_dir, args.store_dir, args.feature_cache_dir,
        split=args.split, task_kind="structure_v0", allow_dev_only=False,
        single_page_only=args.slice == SINGLE_PAGE_DOCS_SLICE,
        multi_page_only=args.slice == MULTI_PAGE_PAGES_SLICE,
        only_bundles=None, only_bundles_path=None,
    )
    samples = {(s.document_bundle_id, s.page_index): s for s in plan.samples}
    requested = sorted(samples)
    reader = FeatureCacheReader(args.feature_cache_dir)
    verify_feature_grammar_binding(
        reader.manifest,
        source_dir=args.feature_cache_dir,
        capability_profile=profile,
        legacy={},
    )
    feature_keys = {
        (reader[i].document_bundle_id, int(reader[i].page_index)) for i in range(len(reader))
    }

    feature_skips: dict[PageKey, str | None] = {}
    for key in requested:
        if key in feature_keys:
            continue
        page = bc.prepare_page(
            args.store_dir, samples[key], max_steps=args.max_steps, capability_profile=profile
        )
        if page.skip is None and bc._uses_excluded_capability(
            page.target_ids, page.input_ids, capability_profile=profile
        ):
            feature_skips[key] = "skipped_capability_excluded"
        else:
            feature_skips[key] = page.skip

    # `ExactTextCacheWriter.record_summary` merges its fields into the
    # manifest's TOP level (`extra_manifest`) -- there is no `summary` key.
    summary = cache_manifest(args.exact_text_cache_dir)
    floor = float(summary["min_text_coverage"])
    below_floor = {
        (row["document_bundle_id"], int(row["page_index"]))
        for row in summary.get("text_coverage_rows", [])
        if not row["written"]
        and row["store_chars"]
        and row["granted_chars"] / row["store_chars"] < floor
    }
    # Absent (not empty) in an exact-text cache built before per-page skip
    # rows existed -- `assign_dispositions` then falls back to the
    # split-wide count check and says so if it refuses.
    raw_skip_rows = summary.get("page_skip_rows")
    exact_text_skip_rows = (
        None
        if raw_skip_rows is None
        else {
            (row["document_bundle_id"], int(row["page_index"])): row["reason"]
            for row in raw_skip_rows
        }
    )
    written = set(ExactTextCacheReader(args.exact_text_cache_dir, gradient_bearing=False).page_keys)
    build_summary = json.loads(args.exact_text_summary.read_text())
    skip_counts = {k: int(v) for k, v in build_summary.items() if k.startswith("skipped_")}

    dispositions = assign_dispositions(
        requested=requested,
        feature_keys=feature_keys,
        written_keys=written,
        below_floor_keys=below_floor,
        feature_skips=feature_skips,
        exact_text_skip_counts=skip_counts,
        cache_universe=cache_universe,
        exact_text_skip_rows=exact_text_skip_rows,
    )
    if sample_args is not None:
        size, seed = sample_args
        population_count = len(dispositions)
        population_pages_sha256 = _population_pages_sha256(dispositions)
        dispositions = sample_dispositions(dispositions, size=size, seed=seed)

    release = json.loads((args.release_dir / "release-manifest.json").read_text())
    selection_rule = {
        "split": args.split,
        "task_kind": "structure_v0",
        "single_page_only": args.slice == SINGLE_PAGE_DOCS_SLICE,
        "capability_profile": profile.profile_id,
        "exclude_capability_profile": True,
        "min_text_coverage": floor,
        "render_dpi": args.render_dpi,
        "release_root_hash": release["release_root_hash"],
        "leakage_fingerprint_policy_hash": release["component_lock"][
            "leakage_fingerprint_policy_hash"
        ],
        "component_disjointness": (
            "derived from allocator.py:6-17 (RFC003 §40.1): the split atom is the connected "
            "component of the leakage graph; whole components go to one split. Component ids "
            "are NOT materialised in the release. This is a citation, not a measurement."
        ),
    }
    apply_slice_to_selection_rule(selection_rule, args.slice)
    if sample_args is not None:
        size, seed = sample_args
        selection_rule["sample"] = {
            "method": "sha256_rank",
            "size": size,
            "seed": seed,
            "population_count": population_count,
            "population_pages_sha256": population_pages_sha256,
        }
    raw = build_cohort_manifest(
        role=ROLE_BY_SLICE[args.slice],
        dispositions=dispositions,
        lineages={s.document_bundle_id: s.lineage_root_id for s in plan.samples},
        selection_rule=selection_rule,
    )
    dm.verify_cohort_accounting(
        dm.load_cohort_manifest(raw), feature_presence=lambda b: (b, 0) in feature_keys
    )
    args.out.write_text(json.dumps(raw, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), **raw["expected_accounting"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

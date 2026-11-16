"""S1 canary cohort lineage helpers over `release-v10` (`dense-v0-m2-m3-plan.md`
task 3, DEC-007 D1.1, RFC005 §14.4 gate 9).

**Cohort selection itself lives in `build_cache.py`, not here.** An
earlier version of this module reimplemented single-page filtering and
capability exclusion independently of `build_cache.build`'s
`single_page_only`/`exclude_capability_profile` flags, landed concurrently
by another agent -- two selection implementations risking silent drift
(theirs reads decoded `target_ids`/`input_ids`, this one read raw
target-bundle structural events) is exactly the kind of fork this project's
git-audit note warns about, so the duplicate was retired. The real cohort
is whatever `build_cache.build(..., single_page_only=True,
exclude_capability_profile=True)` wrote into a `FeatureCacheReader`-openable
cache; read it back through that reader, not through a second loader here.

What remains here is genuinely not duplicated elsewhere: turning a set of
cached `document_bundle_id`s into the `lineage_root_id`s gate 9 needs.
`feature_cache.CacheEntry`/`CacheIdentity` carry no per-page lineage field
(identity is cache-level, entries carry only `document_bundle_id`), so this
is the resolution step between "which documents did the cache write" and
"is that set lineage-disjoint from eval/test" (gate 9's own question).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CohortLineage:
    """What `resolve_cohort_lineage` found, INCLUDING what it did not.

    `lineage_root_ids` is the set gate 9 compares against eval/test.
    `requested_bundle_ids`/`resolved_bundle_ids` are DOCUMENTS, and they
    are here because the two counts are not interchangeable: several
    documents can share one lineage root, so a root count says nothing
    about how much of the cohort was actually resolved.
    """

    lineage_root_ids: frozenset[str]
    requested_bundle_ids: frozenset[str]
    resolved_bundle_ids: frozenset[str]

    @property
    def unresolved_bundle_ids(self) -> frozenset[str]:
        """Requested documents that matched no row in any split manifest."""
        return self.requested_bundle_ids - self.resolved_bundle_ids


def resolve_cohort_lineage(
    release_dir: Path | str, bundle_ids: Iterable[str]
) -> CohortLineage:
    """`lineage_root_id` for each of `bundle_ids`, resolved against
    `release_dir`'s own split manifests (`splits/*.jsonl`) -- the same
    authority `dataset.samples.iter_page_samples` reads split/lineage
    membership from. A `bundle_id` absent from every split manifest is
    never invented: a cache entry for a document the release no longer
    lists is a store/release mismatch to investigate, not a lineage root
    to guess.

    **It is also REPORTED, not merely skipped** (2026-09-04). The previous
    version of this function returned the root set alone, so unmatched
    documents vanished with no count and no signal -- and gate 9, whose
    only input was that set, could report `pass` on a cohort where 149 of
    150 documents had resolved to nothing: a statement about one document,
    read by every caller as a statement about the cohort. The empty case
    was already handled; the PARTIAL case was invisible. Callers must
    carry `unresolved_bundle_ids` into their evidence, which is why this
    returns a `CohortLineage` rather than a `frozenset` (renamed from
    `lineage_root_ids_for_bundle_ids` so a stale caller fails loudly
    instead of silently iterating the wrong object).
    """
    release_dir = Path(release_dir)
    wanted = frozenset(bundle_ids)
    roots: set[str] = set()
    resolved: set[str] = set()
    for split in ("train", "val_iid", "val_template_ood", "test", "partial"):
        path = release_dir / "splits" / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["authority_id"] in wanted:
                roots.add(row["lineage_root_id"])
                resolved.add(row["authority_id"])
    return CohortLineage(
        lineage_root_ids=frozenset(roots),
        requested_bundle_ids=wanted,
        resolved_bundle_ids=frozenset(resolved),
    )


class EvalTestSplitManifestsMissing(FileNotFoundError):
    """None of `val_iid`/`val_template_ood`/`test` split manifests exist
    under `release_dir/splits/` -- a release-directory mishap (wrong path,
    unmounted store, a release that never finished writing its splits),
    not evidence that eval/test lineage is empty. Raised rather than
    returning an empty set, so gate 9 (RFC005 §14.4) cannot read this
    silence as "disjoint by default" (DEC-007 D8)."""


def eval_test_lineage_root_ids(release_dir: Path | str) -> frozenset[str]:
    """Every `lineage_root_id` in `val_iid`/`val_template_ood`/`test` --
    gate 9's `eval_test_lineage_root_ids` argument. Reads `splits/*.jsonl`
    directly (the same authority `iter_page_samples`'s `_split_membership`
    resolves split assignment from) rather than the `train` cohort's own
    negative space.

    **Raises if NONE of the three split files exist** -- that is a
    release-directory mishap, and a caller must not be able to read the
    resulting empty set as "everything is disjoint by default" (the exact
    failure mode 2026-09-05's review caught in `gate_9_loader_lineage_disjoint`,
    which additionally refuses an empty result regardless of how it got
    that way -- this raise is the first line of defense, not the only one).
    A split file that EXISTS but is empty is a real, legitimate result
    (`release-v10`'s own `val_template_ood` split has 0 documents today)
    and is not raised on -- only the total absence of all three is
    treated as a mishap.
    """
    release_dir = Path(release_dir)
    splits = ("val_iid", "val_template_ood", "test")
    paths = [release_dir / "splits" / f"{name}.jsonl" for name in splits]
    if not any(p.exists() for p in paths):
        raise EvalTestSplitManifestsMissing(
            f"none of {[str(p) for p in paths]} exist -- refusing to report an "
            "empty eval/test lineage set, which would read as 'nothing to check "
            "against' rather than 'this release directory could not be read'"
        )
    roots: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            roots.add(json.loads(line)["lineage_root_id"])
    return frozenset(roots)

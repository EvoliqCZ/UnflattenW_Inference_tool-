"""S3 typed-supervision census producer (T6, design §7.3 step 5, §11 D7).

Counts, over one exact-text cache's pinned cohort `E \\ X`, how much typed
supervision (RFC004 §24/§26) each head actually has to train on:
`supervised` / `known_null` / `unsupervised` / `deferred` per
`(head_id, node_property_pointer)` and per `head_id`, plus the excluded
`cardinality` / `union_arm` totals (§4.5) and, per `head_id`, the count of
DISTINCT `lineage_root_id` its supervised targets touch. The floor gate
(`s3_run_manifest`, plan row 5) reads this artefact, not a live re-count.

`per_pointer` is keyed by the RAW, indexed `node_property_pointer` exactly
as the cache stores it (e.g. `/layout/tabs/0/position_mpt`, `/layout/tabs
/1/position_mpt` as two distinct pointer keys) -- classification alone
normalizes (below); the count breakdown deliberately keeps the index so a
consumer can see per-slot coverage rather than one pooled `/layout/tabs/*/
position_mpt` bucket.

**Masking rules (§4.3), the load-bearing ordering:**

0. Classification looks up `(node_type, normalize_pointer(pointer))` --
   `FAMILY`/`DEFERRED` are keyed by the NORMALIZED pointer, not the
   indexed one the cache stores, matching the cache's own lookup
   (`exact_text_cache.py:871`, `typed_property_targets.py:393`). A key in
   neither set is refused (`TypedCensusError`), never treated as
   family-bearing by default -- the two sets partition all 183 assigned
   pointers today (155 + 28, no overlap), but that is a fact about the
   registry, not something this module may assume silently.
1. `(node_type, pointer) in DEFERRED` -> always `deferred`, regardless of
   `supervision_mask` -- the coordinator's raw prod census (§11 D7) counted
   these as `supervised`, and splitting them out is this module's whole
   reason to exist.
2. `known_mask and value is None` -> `known_null` -- no family has a null
   class (RFC005 §27.5), so this is excluded from `supervised` even when
   `supervision_mask` happens to be true.
3. `supervision_mask` -> `supervised`.
4. else -> `unsupervised`.

**Performance (§7.3 step 5).** The prod cache is 3.1 TB; reading it through
`ExactTextCacheReader.__getitem__` hashes every shard's bytes, which the
coordinator measured would take hours. This producer instead reads each
shard directly with `torch.load(path, weights_only=True, mmap=True)` and
keeps only the two cheap checks that make the read trustworthy: the
manifest's declared `(document_bundle_id, page_index)` key list must match
the shard's rows positionally (the same check `ExactTextCacheReader.
_shard_rows` applies, minus the sha256 hash), and the number of rows
counted must equal `len(expectation.target_keys())`. Both are refusals,
never repairs.

Lineages come from `s2_plan.load_split_records`, keyed by `authority_id`
(the bundle id) -> `lineage_root_id`; the CLI builds `{document_bundle_id:
lineage_root_id}` from it and hands the pure counting function a plain
mapping, not a dependency on the release loader.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from unflatten.dataset.targets.properties import normalize_pointer
from unflatten.model.exact_text_cache import (
    CacheExpectation,
    ExactTextCacheReader,
    PageKey,
)
from unflatten.model.hashing import domain_hash
from unflatten.model.run_binding import (
    compute_exact_text_cache_expectation_hash,
    compute_exact_text_cache_identity_hash,
)
from unflatten.model.s2_plan import load_split_records
from unflatten.model.typed_property_pointer_map import DEFERRED, FAMILY

#: `census_hash = domain_hash(_CENSUS_DOMAIN, payload_without_census_hash)`.
#: Owned here: no other module computes a `census_hash`.
_CENSUS_DOMAIN = "rfc005.s3-typed-census/v1"

#: The four buckets a `TypedPropertyTarget` classifies into (§4.3).
#: `cardinality` and `union_arm` are separate, un-attributed totals
#: (§4.5): neither record type carries a `head_id`.
_BUCKETS: tuple[str, ...] = ("supervised", "known_null", "unsupervised", "deferred")


class TypedCensusError(ValueError):
    """The census cannot be built truthfully from the presented inputs: a
    row's bundle has no lineage mapping, a shard's rows do not match the
    manifest's declared keys, the counted rows do not equal the pinned
    expectation, or the output path already exists."""


def _empty_counts() -> dict[str, int]:
    return {bucket: 0 for bucket in _BUCKETS}


def _classify(
    *,
    node_type: str,
    pointer: str,
    known_mask: bool,
    supervision_mask: bool,
    value: Any,
) -> str:
    """One `TypedPropertyTarget` payload's bucket (§4.3 ordering).

    `FAMILY`/`DEFERRED` are keyed by the NORMALIZED pointer (`*` segments,
    not a concrete index); the cache stores the indexed form
    (`/layout/tabs/0/position_mpt`). Membership is checked on
    `normalize_pointer(pointer)`, the same call the cache itself makes
    before this lookup (`exact_text_cache.py:871`,
    `typed_property_targets.py:393`) -- checking the raw indexed pointer
    would silently miss every DEFERRED array leaf and count it supervised,
    the exact error this module exists to prevent.

    The two sets partition all 183 assigned pointers (155 FAMILY + 28
    DEFERRED, no overlap, coordinator-verified); a pointer in neither is
    refused rather than defaulted to family-bearing, so a future registry
    change cannot silently count an unknown pointer as supervised.
    """
    key = (node_type, normalize_pointer(pointer))
    if key in DEFERRED:
        return "deferred"
    if key not in FAMILY:
        raise TypedCensusError(
            f"{key!r} is neither an assigned FAMILY pointer nor a recorded "
            "DEFERRED one -- a registry-completeness bug, not a pointer to "
            "count as though it were family-bearing"
        )
    if known_mask and value is None:
        return "known_null"
    if supervision_mask:
        return "supervised"
    return "unsupervised"


def _partial_census(
    rows: Iterable[Mapping[str, Any]],
    lineage_by_bundle: Mapping[str, str],
) -> dict[str, Any]:
    """One shard's (or the whole cohort's) contribution: counts plus the
    per-head lineage SETS, not yet collapsed to a count -- collapsing here
    would undercount a lineage split across shards."""
    per_head: dict[str, dict[str, int]] = {}
    per_pointer: dict[str, dict[str, dict[str, int]]] = {}
    lineage_sets: dict[str, set[str]] = {}
    cardinality = 0
    union_arm = 0
    rows_seen = 0

    for row in rows:
        rows_seen += 1
        bundle_id = row["document_bundle_id"]
        if bundle_id not in lineage_by_bundle:
            raise TypedCensusError(
                f"{bundle_id!r} (page {row.get('page_index')!r}) has no "
                "lineage_root_id in the presented mapping -- every counted "
                "row must resolve to exactly one lineage"
            )
        lineage_root_id = lineage_by_bundle[bundle_id]

        for target in row["typed_targets"]:
            head_id = target["head_id"]
            pointer = target["node_property_pointer"]
            bucket = _classify(
                node_type=target["node_type"],
                pointer=pointer,
                known_mask=target["known_mask"],
                supervision_mask=target["supervision_mask"],
                value=target["value"],
            )
            head_counts = per_head.setdefault(head_id, _empty_counts())
            head_counts[bucket] += 1
            pointer_counts = per_pointer.setdefault(head_id, {}).setdefault(
                pointer, _empty_counts()
            )
            pointer_counts[bucket] += 1
            if bucket == "supervised":
                lineage_sets.setdefault(head_id, set()).add(lineage_root_id)

        cardinality += len(row["typed_cardinalities"])
        union_arm += len(row["typed_union_arms"])

    return {
        "per_head": per_head,
        "per_pointer": per_pointer,
        "cardinality": cardinality,
        "union_arm": union_arm,
        "lineage_sets": lineage_sets,
        "rows_seen": rows_seen,
    }


def merge_partials(partials: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold several `_partial_census` results (e.g. one per shard) into
    one, summing counts and UNIONING lineage sets (never summing their
    sizes, which would double-count a lineage seen in more than one
    shard)."""
    merged_per_head: dict[str, dict[str, int]] = {}
    merged_per_pointer: dict[str, dict[str, dict[str, int]]] = {}
    merged_lineages: dict[str, set[str]] = {}
    cardinality = 0
    union_arm = 0
    rows_seen = 0

    for partial in partials:
        for head_id, counts in partial["per_head"].items():
            total = merged_per_head.setdefault(head_id, _empty_counts())
            for bucket in _BUCKETS:
                total[bucket] += counts[bucket]
        for head_id, pointers in partial["per_pointer"].items():
            merged_pointers = merged_per_pointer.setdefault(head_id, {})
            for pointer, counts in pointers.items():
                total = merged_pointers.setdefault(pointer, _empty_counts())
                for bucket in _BUCKETS:
                    total[bucket] += counts[bucket]
        for head_id, lineages in partial["lineage_sets"].items():
            merged_lineages.setdefault(head_id, set()).update(lineages)
        cardinality += partial["cardinality"]
        union_arm += partial["union_arm"]
        rows_seen += partial["rows_seen"]

    return {
        "per_head": merged_per_head,
        "per_pointer": merged_per_pointer,
        "cardinality": cardinality,
        "union_arm": union_arm,
        "lineage_sets": merged_lineages,
        "rows_seen": rows_seen,
    }


def _finalize(partial: Mapping[str, Any]) -> dict[str, Any]:
    """Collapse a (merged) partial into the public counts shape: each
    head's `lineage_count` is `len` of its lineage set, and the set itself
    (not JSON-serializable) is dropped."""
    per_head = {
        head_id: {**counts, "lineage_count": len(partial["lineage_sets"].get(head_id, ()))}
        for head_id, counts in partial["per_head"].items()
    }
    return {
        "per_head": per_head,
        "per_pointer": partial["per_pointer"],
        "cardinality": partial["cardinality"],
        "union_arm": partial["union_arm"],
    }


def compute_typed_census(
    rows: Iterable[Mapping[str, Any]],
    lineage_by_bundle: Mapping[str, str],
) -> dict[str, Any]:
    """Pure function: raw row payloads (the shard-storage dict shape --
    `document_bundle_id`, `typed_targets`, `typed_cardinalities`,
    `typed_union_arms`, each typed record itself a plain dict) plus
    `{document_bundle_id: lineage_root_id}` -> the counts payload (§7.3
    step 5). No I/O, no torch, no hashing -- CLI-only concerns live below."""
    return _finalize(_partial_census(rows, lineage_by_bundle))


def build_census_payload(
    counts: Mapping[str, Any],
    *,
    cache_identity_hash: str,
    expectation_hash: str,
) -> dict[str, Any]:
    """The frozen artefact payload: `counts` plus the two bindings this
    census must carry (design §7.3 step 5) and its own `census_hash` over
    all three -- dropping `expectation_hash` from what gets hashed would
    let a census silently answer for an expectation it never bound."""
    payload = {
        "cache_identity_hash": cache_identity_hash,
        "expectation_hash": expectation_hash,
        "counts": counts,
    }
    return {**payload, "census_hash": domain_hash(_CENSUS_DOMAIN, payload)}


# ---------------------------------------------------------------------------
# Shard I/O (mmap, no sha256 -- §7.3 step 5's performance note)
# ---------------------------------------------------------------------------

#: Set once per worker process by `_init_worker`; a Pool-friendly way to
#: avoid re-pickling `lineage_by_bundle` / `target_keys` on every task.
_worker_state: dict[str, Any] = {}


def _init_worker(
    cache_dir: str,
    lineage_by_bundle: Mapping[str, str],
    target_keys: frozenset[PageKey],
) -> None:
    _worker_state["cache_dir"] = Path(cache_dir)
    _worker_state["lineage_by_bundle"] = lineage_by_bundle
    _worker_state["target_keys"] = target_keys


def _census_one_shard(shard_entry: Mapping[str, Any]) -> dict[str, Any]:
    """Read one shard via mmap and return its partial census over the rows
    in `target_keys`, after checking the manifest's declared keys match
    the shard's rows positionally (the identity binding this producer
    keeps even though it skips `ExactTextCacheReader`'s per-shard sha256,
    design §7.3 step 5)."""
    cache_dir: Path = _worker_state["cache_dir"]
    lineage_by_bundle: Mapping[str, str] = _worker_state["lineage_by_bundle"]
    target_keys: frozenset[PageKey] = _worker_state["target_keys"]

    name = shard_entry["name"]
    declared: Sequence[Sequence[Any]] = shard_entry["keys"]
    path = cache_dir / name
    rows = torch.load(path, weights_only=True, mmap=True)
    if len(rows) != len(declared):
        raise TypedCensusError(
            f"{path}: {len(rows)} row(s), manifest declares {len(declared)}"
        )
    selected: list[Mapping[str, Any]] = []
    for position, row in enumerate(rows):
        key = (row["document_bundle_id"], row["page_index"])
        if key != tuple(declared[position]):
            raise TypedCensusError(
                f"{path} row {position} is page {key!r}, but the manifest's "
                f"shard key list declares {tuple(declared[position])!r} -- the "
                "scope declaration does not describe the rows it claims to"
            )
        if key in target_keys:
            selected.append(row)
    return _partial_census(selected, lineage_by_bundle)


def build_census(
    cache_dir: Path | str,
    *,
    expectation: CacheExpectation,
    lineage_by_bundle: Mapping[str, str],
    workers: int = 1,
) -> dict[str, Any]:
    """Open the cache (manifest-only, cheap), read every shard raw, and
    return the frozen census payload for `expectation.target_keys()`.

    Refuses (`TypedCensusError`) if a shard's rows do not match its
    declared keys, or if the number of rows counted does not equal
    `len(expectation.target_keys())` -- the binding checks this producer
    keeps even on the fast mmap path."""
    cache_dir = Path(cache_dir)
    reader = ExactTextCacheReader(
        cache_dir, gradient_bearing=False, expectation=expectation
    )
    cache_identity_hash = compute_exact_text_cache_identity_hash(asdict(reader.identity))
    expectation_hash = compute_exact_text_cache_expectation_hash(expectation.payload())
    target_keys = frozenset(expectation.target_keys())
    shard_entries = list(reader.manifest["shard_index"])

    if workers <= 1:
        _init_worker(str(cache_dir), lineage_by_bundle, target_keys)
        partials = [_census_one_shard(entry) for entry in shard_entries]
    else:
        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(str(cache_dir), lineage_by_bundle, target_keys),
        ) as pool:
            partials = pool.map(_census_one_shard, shard_entries)

    merged = merge_partials(partials)
    if merged["rows_seen"] != len(target_keys):
        raise TypedCensusError(
            f"counted {merged['rows_seen']} row(s) but the expectation names "
            f"{len(target_keys)} page(s) in E \\ X -- a census over a "
            "different cohort than the one it is pinned to is refused"
        )
    counts = _finalize(merged)
    return build_census_payload(
        counts,
        cache_identity_hash=cache_identity_hash,
        expectation_hash=expectation_hash,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.s3_typed_census",
        description=(
            "Count S3 typed-property supervision per head/pointer over one "
            "exact-text cache's pinned E \\ X cohort."
        ),
    )
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--expectation", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    if args.out.exists():
        raise TypedCensusError(
            f"{args.out} already exists -- refusing to overwrite a frozen "
            "typed supervision census"
        )

    expectation = CacheExpectation.from_payload(
        json.loads(args.expectation.read_text())
    )
    split_records = load_split_records(args.release_dir)
    lineage_by_bundle = {
        record["authority_id"]: record["lineage_root_id"] for record in split_records
    }

    payload = build_census(
        args.cache_dir,
        expectation=expectation,
        lineage_by_bundle=lineage_by_bundle,
        workers=args.workers,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"census_hash": payload["census_hash"], "out": str(args.out)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

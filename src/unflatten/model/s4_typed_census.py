"""S4 typed-supervision census producer (T6, design §7.3 step 5, §11 D7).

Counts, over one exact-text cache's pinned cohort `E \\ X`, how much typed
supervision (RFC004 §24/§26) each head actually has to train on:
`supervised` / `known_null` / `unsupervised` / `deferred` per
`(head_id, node_property_pointer)` and per `head_id`, plus the excluded
`cardinality` / `union_arm` totals (§4.5) and, per `head_id`, the count of
DISTINCT `lineage_root_id` its supervised targets touch. The floor gate
(`s4_run_manifest`, plan row 5) reads this artefact, not a live re-count.

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
1. `(node_type, pointer) in DEFERRED` -> `deferred`, regardless of
   `supervision_mask` -- the coordinator's raw prod census (§11 D7) counted
   these as `supervised`, and splitting them out is this module's whole
   reason to exist. **Narrowed by T4** (design §5.1/§10.1 T4): the 21
   colour pointers (`typed_head_config.family_override` returns
   `"palette"`) follow rules 2-4 below like any other family instead --
   only the 4 border-OBJECT pointers (`"borders_null"`) and the 3
   remaining DEFERRED strings (`/language`, `/style/font_family`,
   `/marker/suffix`) still classify as `deferred` unconditionally.
2. `known_mask and value is None` -> `known_null` -- no family has a null
   class (RFC005 §27.5), so this is excluded from `supervised` even when
   `supervision_mask` happens to be true. **Narrowed by T4**: this applies
   to the 17 non-nullable colour pointers (which, like every other
   family, have no `null` class); the 4 nullable colour pointers DO have
   one (`typed_head_config.NULLABLE_COLOR_POINTERS`), so `null` there
   follows rules 3-4 instead.
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

#: `census_hash = domain_hash(_CENSUS_DOMAIN, payload_without_census_hash)`.
#: Owned by `s3_typed_census` and imported, not restated: S4 reads the same
#: census artefact, and a second copy would stop matching on a version bump.
from unflatten.model.s3_typed_census import _CENSUS_DOMAIN
from unflatten.model.typed_head_config import (
    NULLABLE_COLOR_POINTERS,
    family_override,
    typed_head_config_hash,
)
from unflatten.model.typed_property_pointer_map import (
    DEFERRED,
    FAMILY,
    repeated_containers,
)
from unflatten.renderer.api import FontPolicy
from unflatten.renderer.fonts import resolve_font

#: The four buckets a `TypedPropertyTarget` classifies into (§4.3).
#: `cardinality` and `union_arm` are separate, un-attributed totals
#: (§4.5): neither record type carries a `head_id`.
_BUCKETS: tuple[str, ...] = ("supervised", "known_null", "unsupervised", "deferred")

# ---------------------------------------------------------------------------
# T1 (design `spec-typed-heads-missing-families` §10.1): training-cache
# census inputs for the typed head config literals -- cardinality caps,
# colour/suffix value distributions, `run:/style/font_family` `resolve_font`
# classification.
# ---------------------------------------------------------------------------

#: `container_pointer -> owning node_type`, derived (not listed) from the
#: registry the same way `default_policy._EXTRA_CONTAINER_POINTERS` and
#: `typed_property_targets` already do -- a container's node type is a fact
#: about the registry, never restated by hand.
_CARDINALITY_CONTAINERS: dict[str, str] = {
    pointer: node_type for node_type, pointer in repeated_containers()
}

#: The 21 colour pointers (§5.1 rozpis): every DEFERRED `string` pointer
#: except the three that get their own family (`/language`,
#: `/style/font_family`, `/marker/suffix`) -- derived so a registry change
#: cannot silently drop one from the histogram.
_NON_COLOR_STRING_POINTERS = ("/language", "/style/font_family", "/marker/suffix")
_COLOR_POINTERS: frozenset[tuple[str, str]] = frozenset(
    key
    for key, value_type in DEFERRED.items()
    if value_type == "string" and key[1] not in _NON_COLOR_STRING_POINTERS
)

_SUFFIX_POINTER: tuple[str, str] = ("list", "/marker/suffix")
_FONT_POINTER: tuple[str, str] = ("run", "/style/font_family")

#: Pointers this producer builds a raw value histogram for (§10.1 T1 (d)):
#: the 21 colour pointers plus `list:/marker/suffix`. `font_family` is
#: handled separately (`_font_hist`) because it needs `resolve_font`
#: classification, not a raw value tally.
_VALUE_HISTOGRAM_POINTERS: frozenset[tuple[str, str]] = _COLOR_POINTERS | {_SUFFIX_POINTER}

#: W0-8 / RFC005-ERRATA-008 §4 open item 4 ("Census hodnôt: s4_typed_census
#: nemá histogram hodnôt regresných cieľov; chyba E1 by ním bola
#: viditeľná."): report-only, no gate reads it. Fixed, deterministic bins
#: by decimal magnitude -- not adaptive quantiles, so two runs over
#: different data are directly comparable and a defect that skews every
#: value to one bin (e.g. the union-arm zero label defect) is visible at a
#: glance rather than smoothed away by a data-dependent bin edge.
def _magnitude_bin(value: int) -> str:
    if value == 0:
        return "0"
    if value < 0:
        return "neg"
    return f"1e{len(str(value)) - 1}"

#: The pinned default font policy (`renderer.api.FontPolicy()`): the same
#: policy `masks.py`'s font-substitution producer resolves against, so this
#: census's classification matches what the cache's own masks already say.
_FONT_POLICY = FontPolicy()


def _empty_font_hist() -> dict[str, int]:
    return {
        "in_pack": 0,
        "fallback_map": 0,
        "fallback_default": 0,
        "unresolved": 0,
        "unknown_no_value": 0,
    }


def _record_value(hist: dict[str, dict[str, int]], key: str, value: Any) -> None:
    bucket = hist.setdefault(key, {})
    label = "null" if value is None else str(value)
    bucket[label] = bucket.get(label, 0) + 1


def _classify_font(font_hist: dict[str, int], value: Any) -> None:
    if value is None:
        font_hist["unknown_no_value"] += 1
        return
    _used, issue = resolve_font(
        value,
        None,
        "",
        _FONT_POLICY.mode,
        _FONT_POLICY.fallback_map,
        _FONT_POLICY.fallback_default,
    )
    if issue is None:
        font_hist["in_pack"] += 1
    elif issue.details.get("via") == "map":
        font_hist["fallback_map"] += 1
    elif issue.details.get("via") == "default":
        font_hist["fallback_default"] += 1
    else:
        font_hist["unresolved"] += 1


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
    override = family_override(key[0], key[1])
    if key in DEFERRED and override is None:
        return "deferred"
    if override == "palette":
        # T4 (design §5.1/§10.1 T4, T8's Stav note): a colour pointer no
        # longer counts as `deferred` -- `null` is a real supervised class
        # on the 4 nullable pointers (rule 2 narrows to pointers whose
        # family has NO `null` class at all); the 17 non-nullable
        # pointers keep the pre-existing `known_null` bucket for a
        # schema-anomalous `value=None`.
        if known_mask and value is None and key not in NULLABLE_COLOR_POINTERS:
            return "known_null"
        if not known_mask:
            return "unsupervised"
        return "supervised" if supervision_mask else "unsupervised"
    if override == "borders_null":
        # The object pointer only ever reaches this module when its value
        # is null (`typed_property_targets.py`'s own docstring) -- the
        # `False` ("has border leaves") branch needs cross-record
        # synthesis this per-record classifier does not attempt, so BOTH
        # branches stay `deferred` here (T8's Stav note names this
        # explicitly as out of scope, not a silent omission).
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
    would undercount a lineage split across shards.

    Also computes the T1 (§10.1) extras: the cardinality histogram, the
    `/layout/tabs`-style "supervised node with no container record" /
    "container record with no supervised leaf" (`tabs_subset_of_sup`, R1b)
    counts, and the colour/suffix/font value histograms. All of it is
    computed PER ROW and only then summed -- a `node_id` only ever appears
    within the one row (page) that owns it, so the set difference behind
    `nodes_without_container_record` never needs a cross-shard set.
    """
    per_head: dict[str, dict[str, int]] = {}
    per_pointer: dict[str, dict[str, dict[str, int]]] = {}
    lineage_sets: dict[str, set[str]] = {}
    cardinality = 0
    union_arm = 0
    rows_seen = 0

    cardinality_hist: dict[str, dict[str, int]] = {}
    nodes_without_container_record: dict[str, int] = {
        container_pointer: 0 for container_pointer in _CARDINALITY_CONTAINERS
    }
    tabs_subset_of_sup = 0
    value_hist: dict[str, dict[str, int]] = {}
    font_hist = _empty_font_hist()
    regression_hist: dict[str, dict[str, Any]] = {}

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

        node_type_by_id: dict[str, str] = {}
        supervised_node_ids: set[str] = set()

        for target in row["typed_targets"]:
            head_id = target["head_id"]
            node_type = target["node_type"]
            pointer = target["node_property_pointer"]
            node_type_by_id[target["node_id"]] = node_type
            if target["supervision_mask"]:
                supervised_node_ids.add(target["node_id"])

            bucket = _classify(
                node_type=node_type,
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

            key = (node_type, pointer)
            if key in _VALUE_HISTOGRAM_POINTERS:
                _record_value(value_hist, f"{node_type}|{pointer}", target["value"])
            elif key == _FONT_POINTER:
                _classify_font(font_hist, target["value"])

            if (
                bucket == "supervised"
                and FAMILY.get((node_type, normalize_pointer(pointer))) == "integer"
                and isinstance(target["value"], int)
            ):
                entry = regression_hist.setdefault(
                    f"{head_id}|{pointer}",
                    {"count": 0, "zero_count": 0, "min": None, "max": None, "hist": {}},
                )
                value = target["value"]
                entry["count"] += 1
                if value == 0:
                    entry["zero_count"] += 1
                entry["min"] = value if entry["min"] is None else min(entry["min"], value)
                entry["max"] = value if entry["max"] is None else max(entry["max"], value)
                label = _magnitude_bin(value)
                entry["hist"][label] = entry["hist"].get(label, 0) + 1

        cardinality_node_ids_by_container: dict[str, set[str]] = {}
        for record in row["typed_cardinalities"]:
            container_pointer = record["container_pointer"]
            bucket = cardinality_hist.setdefault(container_pointer, {})
            count_label = str(record["count"])
            bucket[count_label] = bucket.get(count_label, 0) + 1
            cardinality_node_ids_by_container.setdefault(container_pointer, set()).add(
                record["node_id"]
            )
            if container_pointer == "/layout/tabs" and record["node_id"] not in supervised_node_ids:
                tabs_subset_of_sup += 1

        for container_pointer, owning_node_type in _CARDINALITY_CONTAINERS.items():
            supervised_of_type = {
                node_id
                for node_id, node_type in node_type_by_id.items()
                if node_type == owning_node_type and node_id in supervised_node_ids
            }
            missing = supervised_of_type - cardinality_node_ids_by_container.get(
                container_pointer, set()
            )
            nodes_without_container_record[container_pointer] += len(missing)

        cardinality += len(row["typed_cardinalities"])
        union_arm += len(row["typed_union_arms"])

    return {
        "per_head": per_head,
        "per_pointer": per_pointer,
        "cardinality": cardinality,
        "union_arm": union_arm,
        "lineage_sets": lineage_sets,
        "rows_seen": rows_seen,
        "cardinality_hist": cardinality_hist,
        "nodes_without_container_record": nodes_without_container_record,
        "tabs_subset_of_sup": tabs_subset_of_sup,
        "value_hist": value_hist,
        "font_hist": font_hist,
        "regression_hist": regression_hist,
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

    cardinality_hist: dict[str, dict[str, int]] = {}
    nodes_without_container_record: dict[str, int] = {}
    tabs_subset_of_sup = 0
    value_hist: dict[str, dict[str, int]] = {}
    font_hist = _empty_font_hist()
    regression_hist: dict[str, dict[str, Any]] = {}

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

        for container_pointer, histogram in partial["cardinality_hist"].items():
            merged_bucket = cardinality_hist.setdefault(container_pointer, {})
            for count_label, n in histogram.items():
                merged_bucket[count_label] = merged_bucket.get(count_label, 0) + n
        for container_pointer, n in partial["nodes_without_container_record"].items():
            nodes_without_container_record[container_pointer] = (
                nodes_without_container_record.get(container_pointer, 0) + n
            )
        tabs_subset_of_sup += partial["tabs_subset_of_sup"]
        for pointer_key, histogram in partial["value_hist"].items():
            merged_bucket = value_hist.setdefault(pointer_key, {})
            for label, n in histogram.items():
                merged_bucket[label] = merged_bucket.get(label, 0) + n
        for label, n in partial["font_hist"].items():
            font_hist[label] += n
        for reg_key, entry in partial["regression_hist"].items():
            merged_entry = regression_hist.setdefault(
                reg_key, {"count": 0, "zero_count": 0, "min": None, "max": None, "hist": {}}
            )
            merged_entry["count"] += entry["count"]
            merged_entry["zero_count"] += entry["zero_count"]
            merged_entry["min"] = (
                entry["min"]
                if merged_entry["min"] is None
                else min(merged_entry["min"], entry["min"])
            )
            merged_entry["max"] = (
                entry["max"]
                if merged_entry["max"] is None
                else max(merged_entry["max"], entry["max"])
            )
            for label, n in entry["hist"].items():
                merged_entry["hist"][label] = merged_entry["hist"].get(label, 0) + n

    return {
        "per_head": merged_per_head,
        "per_pointer": merged_per_pointer,
        "cardinality": cardinality,
        "union_arm": union_arm,
        "lineage_sets": merged_lineages,
        "rows_seen": rows_seen,
        "cardinality_hist": cardinality_hist,
        "nodes_without_container_record": nodes_without_container_record,
        "tabs_subset_of_sup": tabs_subset_of_sup,
        "value_hist": value_hist,
        "font_hist": font_hist,
        "regression_hist": regression_hist,
    }


def _finalize(partial: Mapping[str, Any]) -> dict[str, Any]:
    """Collapse a (merged) partial into the public counts shape: each
    head's `lineage_count` is `len` of its lineage set, and the set itself
    (not JSON-serializable) is dropped.

    `cardinality_max` is derived from `cardinality_hist` here, not carried
    separately through `_partial_census`/`merge_partials`: a max is *only*
    correct once every shard's histogram has been folded in.
    """
    per_head = {
        head_id: {**counts, "lineage_count": len(partial["lineage_sets"].get(head_id, ()))}
        for head_id, counts in partial["per_head"].items()
    }
    cardinality_max = {
        container_pointer: max(int(label) for label, n in histogram.items() if n > 0)
        for container_pointer, histogram in partial["cardinality_hist"].items()
        if any(n > 0 for n in histogram.values())
    }
    return {
        "per_head": per_head,
        "per_pointer": partial["per_pointer"],
        "cardinality": partial["cardinality"],
        "union_arm": partial["union_arm"],
        "cardinality_hist": partial["cardinality_hist"],
        "cardinality_max": cardinality_max,
        "nodes_without_container_record": partial["nodes_without_container_record"],
        "tabs_subset_of_sup": partial["tabs_subset_of_sup"],
        "value_hist": partial["value_hist"],
        "font_hist": partial["font_hist"],
        "regression_hist": partial["regression_hist"],
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
    typed_head_config_hash: str,
) -> dict[str, Any]:
    """The frozen artefact payload: `counts` plus the three bindings this
    census must carry (design §7.3 step 5; `typed_head_config_hash` added
    by design §7/§9 R13, T8) and its own `census_hash` over all four --
    dropping any one of them from what gets hashed would let a census
    silently answer for a cache/expectation/palette-and-cap configuration
    it never bound (a palette or cap change would leave a stale census
    valid, design §9 R13)."""
    payload = {
        "cache_identity_hash": cache_identity_hash,
        "expectation_hash": expectation_hash,
        "typed_head_config_hash": typed_head_config_hash,
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
        typed_head_config_hash=typed_head_config_hash(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.s4_typed_census",
        description=(
            "Count S4 typed-property supervision per head/pointer over one "
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

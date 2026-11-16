"""S4.5 train-only page loader and ``RlDataPlan`` (C1, D8/D10; dormant).

A library over an RFC003 release for rollouts. It keeps only the pages whose
bundle's split record says ``train`` AND whose lineage root is one of the
dataset manifest's ``lineage_ids``; everything else the release holds is
counted and refused, never loaded.

Records are read through the hash-verified reader
(``adapters.release.read_split_records_verified``). A real RFC003 record has no
``cohort`` / ``source``; the source of a bundle comes from the mandatory
``bundle_source`` map (store index) and a bundle missing from it is refused as
``source_unresolved`` -- counted, never guessed.

**The guard must be able to fail.** A loader that never met a forbidden
record cannot claim it filtered one, so a release that yields an empty
included set OR an empty refused set (val / test / forbidden source /
excluded lineage) raises :class:`RlDataGuardFailed`. The counts are returned
in :meth:`RlTrainPages.summary` for the ledger.

No pixels, caches or stores are read here: cache rows, gold rows and
``DocumentReference`` builders are injected callables (:class:`RlPageAccessors`)
invoked lazily; a page without cached features carries an on-the-fly stub
(its image path is only recorded). ``load_train_pages`` has no reward input.

``RlDataPlan`` references the frozen S4 ``StageDataMatrix`` by hash and the
manifest's ``selection_hash``; it adds no stage id.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from unflatten.model.hashing import domain_hash
from unflatten.model.rl.policies import FORBIDDEN_SOURCES, verify_dataset_manifest

DATA_PLAN_DOMAIN: Final = "unflatten.rl.data-plan/v0"
TRAIN_SPLIT: Final = "train"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

PageKey = tuple[str, int]
Accessor = Callable[[PageKey], Any]


class RlDataRefused(ValueError):
    """The release / manifest / pages cannot be loaded truthfully."""


class RlDataGuardFailed(RlDataRefused):
    """The train-only guard did not observe both sides (included, refused)."""


@dataclass(frozen=True)
class RlPageAccessors:
    """Injected readers; none is called at load time."""

    cache_row: Accessor | None = None
    gold_row: Accessor | None = None
    document_reference: Accessor | None = None


@dataclass(frozen=True)
class RlVisualSource:
    """``cache``: features come from the page's cache row. ``on_the_fly``:
    pixels are needed; ``image_path`` is recorded, never opened here."""

    kind: str
    image_path: Path | None = None


@dataclass(frozen=True)
class RlPage:
    page_key: PageKey
    lineage_root_id: str
    split_name: str
    visual: RlVisualSource
    accessors: RlPageAccessors = field(repr=False, compare=False, default=RlPageAccessors())

    def _call(self, name: str) -> Any:
        fn = getattr(self.accessors, name)
        if fn is None:
            raise RlDataRefused(f"no {name} accessor injected for page {self.page_key}")
        return fn(self.page_key)

    def cache_row(self) -> Any:
        return self._call("cache_row")

    def gold_row(self) -> Any:
        return self._call("gold_row")

    def document_reference(self) -> Any:
        return self._call("document_reference")


@dataclass(frozen=True)
class RlTrainPages:
    pages: tuple[RlPage, ...]
    included_records: int
    refused_by_reason: Mapping[str, int]
    unselected_records: int
    dropped_pages: int
    dataset_manifest_hash: str

    def summary(self) -> dict[str, Any]:
        kinds = Counter(p.visual.kind for p in self.pages)
        return {
            "included_records": self.included_records,
            "refused_records": sum(self.refused_by_reason.values()),
            "refused_by_reason": dict(sorted(self.refused_by_reason.items())),
            "unselected_records": self.unselected_records,
            "included_pages": len(self.pages),
            "dropped_pages": self.dropped_pages,
            "cached_pages": kinds["cache"],
            "on_the_fly_pages": kinds["on_the_fly"],
            "dataset_manifest_hash": self.dataset_manifest_hash,
        }


def _refusal_reason(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    bundle_source: Mapping[str, str],
    registry_excluded: Collection[str],
) -> str | None:
    split = record.get("split_name")
    if split != TRAIN_SPLIT:
        return f"split:{split}"
    source = bundle_source.get(record["authority_id"])
    if source is None:
        return "source_unresolved"
    # NOT a protection (design inv. 13): ``source`` is "docxcorp" or "synthetic"
    # (store index), neither is in FORBIDDEN_SOURCES, so this branch cannot fire
    # for a map from ``load_bundle_source``. The forbidden-source protection is
    # the lineage exclusion below / the exclusion registry.
    if source in FORBIDDEN_SOURCES:
        return f"source:{source}"
    if record["lineage_root_id"] in manifest["exclude_lineage_ids"]:
        return "excluded_lineage"
    if record["lineage_root_id"] in registry_excluded:
        return "registry_excluded_lineage"
    if source not in manifest["sources"]:
        return f"source_not_in_manifest:{source}"
    return None


def _classify(
    records: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    bundle_source: Mapping[str, str],
    registry_excluded: Collection[str],
) -> tuple[dict[str, Mapping[str, Any]], Counter[str], int, dict[str, str], set[str]]:
    selected = set(manifest["lineage_ids"])
    by_bundle: dict[str, Mapping[str, Any]] = {}
    included: dict[str, Mapping[str, Any]] = {}
    refused: Counter[str] = Counter()
    unselected = 0
    lineage_refusals: dict[str, str] = {}
    for rec in records:
        bundle = rec["authority_id"]
        if bundle in by_bundle:
            raise RlDataRefused(f"bundle {bundle!r} appears in more than one split record")
        by_bundle[bundle] = rec
        reason = _refusal_reason(rec, manifest, bundle_source, registry_excluded)
        if reason is not None:
            refused[reason] += 1
            lineage_refusals.setdefault(rec["lineage_root_id"], reason)
        elif rec["lineage_root_id"] in selected:
            included[bundle] = rec
        else:
            unselected += 1
    return included, refused, unselected, lineage_refusals, set(by_bundle)


def _check_lineages(
    manifest: Mapping[str, Any],
    included: Mapping[str, Mapping[str, Any]],
    lineage_refusals: Mapping[str, str],
) -> None:
    included_lineages = {r["lineage_root_id"] for r in included.values()}
    for lineage in manifest["lineage_ids"]:
        if lineage in lineage_refusals:
            raise RlDataRefused(
                f"manifest lineage {lineage!r} is refused by the release "
                f"({lineage_refusals[lineage]})"
            )
    unresolved = sorted(set(manifest["lineage_ids"]) - included_lineages)
    if unresolved:
        raise RlDataRefused(f"manifest lineage(s) {unresolved} resolve to no train record")


def load_train_pages(
    release_dir: Path | str,
    dataset_manifest: Mapping[str, Any],
    *,
    bundle_source: Mapping[str, str],
    page_keys: Sequence[PageKey],
    cached_page_keys: Collection[PageKey] = frozenset(),
    accessors: RlPageAccessors | None = None,
    page_image_path: Callable[[PageKey], Path] | None = None,
    registry_exclude_lineage_ids: Collection[str] = frozenset(),
) -> RlTrainPages:
    """Train-only pages of ``release_dir`` selected by ``dataset_manifest``.

    ``page_keys`` is the pinned ``(bundle, page_index)`` list (cache
    expectation ``E \\ X``); pages of a non-included bundle are dropped and
    counted. ``bundle_source`` is ``{bundle: cohort}`` from the store index
    (``adapters.release.load_bundle_source``); ``registry_exclude_lineage_ids``
    is where the exclusion registry's lineages are fed (empty until it exists).
    Raises :class:`RlDataGuardFailed` unless at least one record and page was
    included and at least one record was refused."""
    from unflatten.model.rl.adapters.release import read_split_records_verified

    verify_dataset_manifest(dataset_manifest)
    included, refused, unselected, lineage_refusals, known = _classify(
        read_split_records_verified(release_dir),
        dataset_manifest,
        bundle_source,
        frozenset(registry_exclude_lineage_ids),
    )
    if not included:
        raise RlDataGuardFailed("inclusive set is empty (no train record selected by the manifest)")
    _check_lineages(dataset_manifest, included, lineage_refusals)
    acc = accessors or RlPageAccessors()
    pages: list[RlPage] = []
    dropped = 0
    for key in page_keys:
        if key[0] not in known:
            raise RlDataRefused(f"page {key} names bundle {key[0]!r} with no split record")
        rec = included.get(key[0])
        if rec is None:
            dropped += 1
            continue
        if key in cached_page_keys:
            visual = RlVisualSource("cache")
        else:
            path = page_image_path(key) if page_image_path is not None else None
            visual = RlVisualSource("on_the_fly", path)
        pages.append(RlPage(key, rec["lineage_root_id"], rec["split_name"], visual, acc))
    out = RlTrainPages(
        tuple(pages),
        len(included),
        dict(refused),
        unselected,
        dropped,
        dataset_manifest["dataset_manifest_hash"],
    )
    _guard(out)
    return out


def _guard(out: RlTrainPages) -> None:
    if not out.pages:
        raise RlDataGuardFailed("inclusive set has no page: no train page survived the filter")
    if not out.refused_by_reason:
        raise RlDataGuardFailed(
            "refused set is empty: the release held no val/test/forbidden record, so the "
            "train-only filter was never exercised and cannot be claimed"
        )


@dataclass(frozen=True)
class RlDataPlan:
    stage_data_matrix_hash: str
    selection_hash: str
    dataset_manifest_hash: str
    plan_hash: str
    #: the exclusion registry the pages were filtered by; ``None`` = not bound
    #: (and then absent from the hashed body, so the plan hash is the legacy one).
    exclusion_registry_hash: str | None = None


def build_rl_data_plan(
    *,
    stage_data_matrix_hash: str,
    dataset_manifest: Mapping[str, Any],
    exclusion_registry_hash: str | None = None,
) -> RlDataPlan:
    """Reference the frozen S4 StageDataMatrix and the manifest's selection.

    ``exclusion_registry_hash`` (optional) binds the exclusion registry into the
    plan body; the manifest schema is frozen, so the registry cannot live there."""
    if not isinstance(stage_data_matrix_hash, str) or not _HASH_RE.match(stage_data_matrix_hash):
        raise RlDataRefused(
            f"stage_data_matrix_hash must be sha256:<64 hex>, got {stage_data_matrix_hash!r}"
        )
    verify_dataset_manifest(dataset_manifest)
    body: dict[str, Any] = {
        "stage_data_matrix_hash": stage_data_matrix_hash,
        "selection_hash": dataset_manifest["selection_hash"],
        "dataset_manifest_hash": dataset_manifest["dataset_manifest_hash"],
    }
    if exclusion_registry_hash is not None:
        if not isinstance(exclusion_registry_hash, str) or not _HASH_RE.match(
            exclusion_registry_hash
        ):
            raise RlDataRefused(
                f"exclusion_registry_hash must be sha256:<64 hex>, got {exclusion_registry_hash!r}"
            )
        body["exclusion_registry_hash"] = exclusion_registry_hash
    return RlDataPlan(plan_hash=domain_hash(DATA_PLAN_DOMAIN, body), **body)

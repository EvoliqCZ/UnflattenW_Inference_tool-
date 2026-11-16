"""Strict admission + rights/privacy provenance (design inv. 9; torch-free).

``require_strict_admission`` is the RL-side half of RFC003-ERRATA-001 E5: a cache
whose identity says ``admission != "strict"`` (``dev_only``, absent, anything
else) is refused. ``ExactTextCacheReader`` already refuses ``dev_only`` for a
``gradient_bearing`` reader at open, but a reader opened ``gradient_bearing=False``
passes that check; this function does not take it on trust, and for a reader it
also demands ``gradient_bearing`` and a pinned ``expectation`` (without one the
reader exposes every physical row, ``C`` and not ``E \\ X``).

``rights_privacy_refs`` derives the ``rights_privacy_refs`` of a dataset manifest
from the release instead of typing them: it refuses when ANY bundle of an
included lineage has a pending, failed or absent privacy gate
(``dataset.samples._privacy_pending``; a bundle with no gate row is pending, a
missing grant is never a pass).

``exact_text_cache`` imports ``torch`` and is never imported here: the cache is
duck-typed (``identity`` / ``admission`` attributes or mapping keys).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

from unflatten.model.hashing import domain_hash

CACHE_IDENTITY_DOMAIN: Final = "unflatten.rlprep.cache-identity/v0"
STRICT: Final = "strict"
_SHOWN: Final = 5


class AdmissionRefused(ValueError):
    """The cache or the pages are not admitted for gradient-bearing RL use."""


class PrivacyReviewPending(AdmissionRefused):
    """Included lineage(s) carry a pending / failed / absent privacy review."""


def _identity_payload(identity: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(identity) and not isinstance(identity, type):
        return dataclasses.asdict(identity)
    if isinstance(identity, Mapping):
        return dict(identity)
    raise AdmissionRefused(f"cache identity of type {type(identity).__name__} is not readable")


def _identity_of(source: Any) -> Any:
    if hasattr(source, "identity") and hasattr(source, "gradient_bearing"):
        if source.gradient_bearing is not True:
            raise AdmissionRefused(
                "cache reader is not gradient_bearing: its admission was checked "
                "at the weaker diagnostic boundary"
            )
        if getattr(source, "expectation", None) is None:
            raise AdmissionRefused(
                "cache reader has no pinned expectation: it exposes every physical row"
            )
        return source.identity
    return source


def require_strict_admission(source: Any) -> str:
    """Return ``"strict"`` or raise :class:`AdmissionRefused`. ``source`` is an
    ``ExactTextCacheReader``, an ``ExactTextCacheIdentity`` or its mapping."""
    payload = _identity_payload(_identity_of(source))
    admission = payload.get("admission")
    if admission != STRICT:
        raise AdmissionRefused(
            f"cache admission is {admission!r}; gradient-bearing RL requires {STRICT!r} "
            "(RFC003-ERRATA-001 E5)"
        )
    return STRICT


def cache_identity_hash(source: Any) -> str:
    return domain_hash(CACHE_IDENTITY_DOMAIN, _identity_payload(_identity_of(source)))


def _pending_bundles(release: Path, bundles: Iterable[str]) -> list[str]:
    from unflatten.dataset.samples import _pinned_rows, _privacy_pending, _release_manifest

    rows = _pinned_rows(release, _release_manifest(release), "gate_report")
    by_bundle = {row["document_bundle_id"]: row for row in rows}
    return sorted(b for b in bundles if b not in by_bundle or _privacy_pending(by_bundle[b]))


def rights_privacy_refs(
    release_dir: Path | str, cache_identity: Any, lineage_ids: Iterable[str]
) -> list[str]:
    """Sorted provenance strings for a dataset manifest, or a named refusal.

    Refuses: non-strict admission; a cache identity bound to another release;
    no / unknown lineage; any included bundle with a pending privacy review."""
    from unflatten.dataset.samples import _release_manifest, release_root_hash
    from unflatten.model.rl.adapters.release import read_split_records_verified

    require_strict_admission(cache_identity)
    release = Path(release_dir)
    root = release_root_hash(release)
    bound = _identity_payload(_identity_of(cache_identity)).get("dataset_release_root_hash")
    if bound != root:
        raise AdmissionRefused(f"cache is bound to release {bound!r}, not {root!r}")
    wanted = set(lineage_ids)
    if not wanted:
        raise AdmissionRefused("no lineage_ids: nothing to attest privacy for")
    records = read_split_records_verified(release)
    unknown = sorted(wanted - {r["lineage_root_id"] for r in records})
    if unknown:
        raise AdmissionRefused(f"{len(unknown)} lineage(s) not in the release: {unknown[:_SHOWN]}")
    bundles = sorted({r["authority_id"] for r in records if r["lineage_root_id"] in wanted})
    pending = _pending_bundles(release, bundles)
    if pending:
        raise PrivacyReviewPending(
            f"{len(pending)} of {len(bundles)} bundle(s) of the included lineages have a "
            f"pending/failed/absent privacy review, e.g. {pending[:_SHOWN]}"
        )
    gate = _release_manifest(release)["gate_report"]
    return sorted(
        [
            f"release_root_hash:{root}",
            f"cache_identity:{cache_identity_hash(cache_identity)}",
            f"gate_report:{gate['sha256']}",
            f"privacy_review_complete:{len(bundles)}_bundles_{len(wanted)}_lineages",
        ]
    )

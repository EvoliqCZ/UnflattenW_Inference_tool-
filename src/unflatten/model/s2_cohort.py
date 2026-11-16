"""S2 P1/P2 development cohort loader (DEC-013, `p1p2-cohort-build-report.md`).

The cohort document (`~/unflatten-data/cohort-p1p2-development.json`) is
self-contained: a closed list of bundle records plus one `cohort_hash`,
computed by the cohort build script over the whole document with the hash
key removed. Cohort SELECTION (which bundles belong, at which pyramid
level, lineage-disjoint from the m2a-v5 canary/eval) was performed by that
script and is recorded in the document and its build report -- this module
does not re-select, it only loads and VERIFIES:

- the document is a closed envelope (unknown fields are a schema change,
  not noise);
- `counts` reconciles exactly with the bundle list (document count,
  per-level page counts against per-level document counts);
- `cohort_hash` recomputes to the stored value.

The hash is the integrity anchor: a document whose seal does not verify is
a different generation or a tampered one. It is refused, never repaired,
and never recomputed-then-accepted -- recomputing it here is the check,
not a fallback. The hash formula is pinned to the generator's: SHA-256
over the UTF-8 of the canonical compact JSON (sorted keys, separators
`, `/ `:`) of the document without the `cohort_hash` key.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

_BUNDLE_FIELDS = frozenset({"document_bundle_id", "lineage_root_id", "pyramid_level"})
_TOP_FIELDS = frozenset(
    {
        "bundles",
        "cohort_hash",
        "cohort_role",
        "counts",
        "decision_ref",
        "generator_version",
        "render_dpi",
        "samples_per_recipe",
        "source_dir",
    }
)
_LEVEL_RE = re.compile(r"^P[1-5]$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class S2CohortRefused(ValueError):
    """The document at the given path is not a well-formed S2 cohort
    document: malformed JSON, an open envelope, a field of the wrong
    type, or `counts` that do not reconcile with the bundle list. A
    refusal, not a repair -- the loader never normalises a document into
    acceptance."""


class CohortHashMismatch(S2CohortRefused):
    """The document is structurally valid but its stored `cohort_hash`
    does not equal the recomputed one: a different generation, a partial
    write, or tampering. Never re-pinned by the loader."""


class CohortLineageOverlap(S2CohortRefused):
    """`assert_lineage_disjoint` found cohort lineage roots shared with
    the compared set -- the S2 analogue of the S1 gate 9 overlap, with
    every shared root named in the message."""


@dataclass(frozen=True)
class CohortBundle:
    document_bundle_id: str
    lineage_root_id: str
    pyramid_level: str


@dataclass(frozen=True)
class S2Cohort:
    """A verified S2 cohort document.

    `bundles` is ordered as on disk; `pages_by_level` is the document's
    own `counts.pages`, kept read-only. Document counts and lineage-root
    counts are kept separate on purpose: several documents can share one
    lineage root, so a root count says nothing about how many documents
    the cohort actually holds (the same conflation S1's gate 9 history
    caught and this project's audit note warns about).
    """

    cohort_hash: str
    cohort_role: str
    bundles: tuple[CohortBundle, ...]
    pages_by_level: MappingProxyType[str, int]
    decision_ref: str
    generator_version: str
    render_dpi: int
    samples_per_recipe: int
    source_dir: str

    @property
    def bundle_ids(self) -> frozenset[str]:
        return frozenset(b.document_bundle_id for b in self.bundles)

    @property
    def lineage_root_ids(self) -> frozenset[str]:
        return frozenset(b.lineage_root_id for b in self.bundles)

    @property
    def levels(self) -> frozenset[str]:
        return frozenset(b.pyramid_level for b in self.bundles)

    @property
    def document_count(self) -> int:
        """DOCUMENTS, not lineage roots -- never the two collapsed."""
        return len(self.bundles)


def _refuse(message: str) -> S2CohortRefused:
    return S2CohortRefused(f"s2 cohort refused: {message}")


def _require_positive_int(value: object, name: str) -> int:
    # `type is int` on purpose: `bool` is an `int` subclass and a
    # `true` count would otherwise pass an `isinstance` check.
    if type(value) is not int or value <= 0:
        raise _refuse(f"{name} must be a positive integer, got {value!r}")
    return value


def _canon_hash(doc: dict) -> str:
    minus = {k: v for k, v in doc.items() if k != "cohort_hash"}
    canon = json.dumps(minus, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(canon).hexdigest()


def load_s2_cohort(path: Path | str) -> S2Cohort:
    """Load and verify the S2 cohort document at `path`.

    Raises `S2CohortRefused` (or a subclass) on any structural,
    consistency, or hash violation; `FileNotFoundError` on a missing
    path -- that is a wrong location, not a document to repair.
    """
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _refuse(f"{path} is not valid JSON ({exc})") from exc
    if not isinstance(doc, dict):
        raise _refuse(f"top level must be an object, got {type(doc).__name__}")

    extra = set(doc) - _TOP_FIELDS
    missing = _TOP_FIELDS - set(doc)
    if extra or missing:
        raise _refuse(f"top-level keys must be exactly {sorted(_TOP_FIELDS)}; "
                      f"extra={sorted(extra)} missing={sorted(missing)}")

    stored_hash = doc["cohort_hash"]
    if not isinstance(stored_hash, str) or not _HASH_RE.match(stored_hash):
        raise _refuse(f"cohort_hash must match 'sha256:<64 hex>', got {stored_hash!r}")

    raw_bundles = doc["bundles"]
    if type(raw_bundles) is not list or not raw_bundles:
        raise _refuse("bundles must be a non-empty list")
    bundles: list[CohortBundle] = []
    seen: set[str] = set()
    for i, row in enumerate(raw_bundles):
        if not isinstance(row, dict) or set(row) != _BUNDLE_FIELDS:
            raise _refuse(f"bundles[{i}] must have exactly the fields {sorted(_BUNDLE_FIELDS)}")
        for field in sorted(_BUNDLE_FIELDS):
            value = row[field]
            if not isinstance(value, str) or not value:
                raise _refuse(f"bundles[{i}].{field} must be a non-empty string, got {value!r}")
        if not _LEVEL_RE.match(row["pyramid_level"]):
            raise _refuse(
                f"bundles[{i}].pyramid_level must be one of P1..P5, got {row['pyramid_level']!r}"
            )
        if row["document_bundle_id"] in seen:
            raise _refuse(
                f"duplicate document_bundle_id {row['document_bundle_id']!r} (bundles[{i}])"
            )
        seen.add(row["document_bundle_id"])
        bundles.append(
            CohortBundle(
                document_bundle_id=row["document_bundle_id"],
                lineage_root_id=row["lineage_root_id"],
                pyramid_level=row["pyramid_level"],
            )
        )

    for name in ("cohort_role", "decision_ref", "generator_version", "source_dir"):
        value = doc[name]
        if not isinstance(value, str) or not value:
            raise _refuse(f"{name} must be a non-empty string, got {value!r}")

    render_dpi = _require_positive_int(doc["render_dpi"], "render_dpi")
    samples_per_recipe = _require_positive_int(doc["samples_per_recipe"], "samples_per_recipe")

    counts = doc["counts"]
    if not isinstance(counts, dict) or set(counts) != {"documents", "pages"}:
        raise _refuse("counts must be an object with exactly 'documents' and 'pages'")
    documents = _require_positive_int(counts["documents"], "counts.documents")
    if documents != len(bundles):
        raise _refuse(
            f"counts.documents is {documents} but the document lists {len(bundles)} bundles"
        )
    pages = counts["pages"]
    if not isinstance(pages, dict):
        raise _refuse("counts.pages must be an object")
    levels = frozenset(b.pyramid_level for b in bundles)
    if set(pages) != levels:
        raise _refuse(
            f"counts.pages keys {sorted(pages)} do not match the bundle levels {sorted(levels)}"
        )
    per_level: dict[str, int] = {}
    for b in bundles:
        per_level[b.pyramid_level] = per_level.get(b.pyramid_level, 0) + 1
    for level in sorted(levels):
        page_count = _require_positive_int(pages[level], f"counts.pages[{level!r}]")
        if page_count < per_level[level]:
            raise _refuse(
                f"counts.pages[{level!r}] is {page_count} but {per_level[level]} documents sit "
                "at that level -- every document has at least one page"
            )

    if _canon_hash(doc) != stored_hash:
        raise CohortHashMismatch(
            f"cohort_hash mismatch at {path}: stored {stored_hash}, "
            f"recomputed {_canon_hash(doc)} -- refusing a document whose seal does not verify"
        )

    return S2Cohort(
        cohort_hash=stored_hash,
        cohort_role=doc["cohort_role"],
        bundles=tuple(bundles),
        pages_by_level=MappingProxyType({k: pages[k] for k in sorted(pages)}),
        decision_ref=doc["decision_ref"],
        generator_version=doc["generator_version"],
        render_dpi=render_dpi,
        samples_per_recipe=samples_per_recipe,
        source_dir=doc["source_dir"],
    )


def lineage_overlap(cohort: S2Cohort, other_lineage_root_ids: Iterable[str]) -> frozenset[str]:
    """The cohort's lineage roots that also appear in
    `other_lineage_root_ids` -- the S2 analogue of the set S1's gate 9
    compares against. An empty result is disjoint only if
    `other_lineage_root_ids` is a legitimate non-empty read; this function
    does not decide that for the caller, and an empty `other` set must not
    be read as proof of disjointness (the same discipline as
    `s1_cohort.EvalTestSplitManifestsMissing`)."""
    return cohort.lineage_root_ids & frozenset(other_lineage_root_ids)


def assert_lineage_disjoint(cohort: S2Cohort, other_lineage_root_ids: Iterable[str]) -> None:
    """Raise `CohortLineageOverlap` naming every shared root, else return
    `None`."""
    overlap = lineage_overlap(cohort, other_lineage_root_ids)
    if overlap:
        raise CohortLineageOverlap(
            f"cohort shares {len(overlap)} lineage root(s) with the compared set: "
            f"{sorted(overlap)}"
        )

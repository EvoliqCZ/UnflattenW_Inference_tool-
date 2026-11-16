"""S4.5 frozen hard-example selection (C2, design D10, ERRATA-010 E3.2 / E7.4 / E7.5; dormant).

A selection is derived ONLY from a failure-taxonomy report over a **train**
cohort whose ``checkpoint_id`` is the policy parent's checkpoint id, and is
frozen (hashed into the dataset manifest's ``selection_hash``) before the
first gradient. No online mining: nothing here accepts a reward, and
:func:`assert_not_reweighted` proves the frozen selection did not move.

What the report offers: per failure class it keeps up to ``MAX_EXAMPLES``
example bundles, not a per-page score. Hardness is therefore the number of
classes a bundle appears in as an example, times ``HARDNESS_PER_CLASS_MICRO``,
summed per lineage. It is a count over a *sample* of failures, not a measured
page difficulty; the selection records this as its ``policy``.

``bundle_lineage`` (bundle id -> lineage root id) is injected; a bundle it
cannot resolve is counted as ``unmapped_bundles``, never guessed.

The domain constant lives here (``ids.py`` is frozen by ``test_ids``).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from unflatten.eval.failure_taxonomy import report as ft
from unflatten.model.hashing import domain_hash

SELECTION_DOMAIN: Final = "unflatten.rl.hard-example-selection/v0"
SELECTION_POLICY: Final = "failure-taxonomy-example-bundles.v0"
#: opt-in: every bundle of a class's ``affected_slice`` instead of its <= MAX_EXAMPLES examples
SELECTION_POLICY_AFFECTED: Final = "failure-taxonomy-affected-bundles.v0"
#: opt-in: only the (bundle, page_index) pages hit by the chosen failure classes
SELECTION_POLICY_AFFECTED_PAGES: Final = "failure-taxonomy-affected-pages.v0"
BUNDLES_FROM_EXAMPLES: Final = "examples"
BUNDLES_FROM_AFFECTED: Final = "affected"
BUNDLES_FROM_AFFECTED_PAGES: Final = "affected-pages"
_BUNDLES_FROM_POLICY: Final = {
    BUNDLES_FROM_EXAMPLES: SELECTION_POLICY,
    BUNDLES_FROM_AFFECTED: SELECTION_POLICY_AFFECTED,
    BUNDLES_FROM_AFFECTED_PAGES: SELECTION_POLICY_AFFECTED_PAGES,
}
REPORTS_DOMAIN: Final = "unflatten.rl.hard-example-selection-reports/v0"
HARDNESS_PER_CLASS_MICRO: Final = 1_000_000
STATUS_READY: Final = "ready"
STATUS_INSUFFICIENT: Final = "insufficient_evidence"
#: role tokens that name a cohort which is not the train split
_NON_TRAIN_TOKENS: Final = frozenset(
    {"val", "validation", "dev", "calibration", "control", "p5", "s5", "holdout"}
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class SelectionRefused(ValueError):
    """The report / manifest cannot back a frozen train-only selection."""


class SelectionReweighted(SelectionRefused):
    """The frozen selection differs from the one hashed before training."""


@dataclass(frozen=True)
class SelectedLineage:
    lineage_root_id: str
    hardness_micro: int
    bundle_ids: tuple[str, ...]
    #: ``(bundle_id, page_index)`` pages to train on; empty = every page of the lineage
    pages: tuple[tuple[str, int], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "lineage_root_id": self.lineage_root_id,
            "hardness_micro": self.hardness_micro,
            "bundle_ids": list(self.bundle_ids),
        }
        if self.pages:  # existing selections stay byte-identical
            doc["pages"] = [[b, k] for b, k in self.pages]
        return doc


@dataclass(frozen=True)
class FrozenSelection:
    status: str
    entries: tuple[SelectedLineage, ...]
    policy: str
    report_hash: str
    checkpoint_id: str
    role: str
    budget: int
    dropped_outside_allowed: int
    unmapped_bundles: int
    truncated_by_budget: int
    #: the reports of a multi-report selection (provenance, outside the hashed
    #: ``payload()``); ``report_hash`` is then their combined hash. Empty for a
    #: legacy / single-report selection -- use :meth:`all_report_hashes`.
    report_hashes: tuple[str, ...] = ()
    #: failure classes of an ``affected-pages`` selection (sorted); empty for the others
    classes: tuple[str, ...] = ()
    #: lineages removed from the selection on request (sorted); empty = none
    #: requested, and then the key stays out of the hashed payload.
    dropped_lineages: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status == STATUS_READY

    def all_report_hashes(self) -> tuple[str, ...]:
        return self.report_hashes or (self.report_hash,)

    def lineage_ids(self) -> tuple[str, ...]:
        return tuple(e.lineage_root_id for e in self.entries)

    def selected_pages(self) -> frozenset[tuple[str, int]]:
        """Union of the entries' pages; empty when the selection is lineage-level."""
        return frozenset(p for e in self.entries for p in e.pages)

    def payload(self) -> dict[str, Any]:
        """Hashed body. ``report_hash`` is provenance only and stays out: the same
        selection derived from a differently ordered but equal-content report
        is the same frozen selection."""
        body = {
            "status": self.status,
            "policy": self.policy,
            "checkpoint_id": self.checkpoint_id,
            "role": self.role,
            "budget": self.budget,
            "dropped_outside_allowed": self.dropped_outside_allowed,
            "unmapped_bundles": self.unmapped_bundles,
            "truncated_by_budget": self.truncated_by_budget,
            "entries": [e.as_dict() for e in self.entries],
        }
        if self.classes:
            body["classes"] = list(self.classes)
        if self.dropped_lineages:  # existing selections stay byte-identical
            body["dropped_lineages"] = list(self.dropped_lineages)
        return body


def _assert_train_cohort(role: object) -> str:
    if not isinstance(role, str) or not role:
        raise SelectionRefused(f"report role must be a non-empty string, got {role!r}")
    tokens = set(_TOKEN_RE.findall(role.lower()))
    if "train" not in tokens:
        raise SelectionRefused(f"report role {role!r} is not a train cohort (no 'train' token)")
    other = sorted(tokens & _NON_TRAIN_TOKENS)
    if other:
        raise SelectionRefused(f"report role {role!r} names a non-train cohort token(s) {other}")
    return role


def _bundle_hits(report: Mapping[str, Any], bundles_from: str) -> Counter[str]:
    hits: Counter[str] = Counter()
    for record in report["classes"]:
        if bundles_from == BUNDLES_FROM_AFFECTED:
            seen = set(record["affected_slice"]["bundle_ids"])
        else:
            seen = {ex["bundle"] for ex in record["cause_evidence"]["examples"]}
        hits.update(seen)
    return hits


def _page_hits(
    report: Mapping[str, Any], classes: frozenset[str]
) -> tuple[Counter[str], dict[str, set[int]], set[str]]:
    """Hits of the chosen classes only: per bundle, plus the page index of this report."""
    page_index = (report.get("inputs") or {}).get("page_index")
    if isinstance(page_index, bool) or not isinstance(page_index, int) or page_index < 0:
        raise SelectionRefused(
            f"affected-pages needs report inputs.page_index (int >= 0), got {page_index!r}"
        )
    hits: Counter[str] = Counter()
    pages: dict[str, set[int]] = {}
    present: set[str] = set()
    for record in report["classes"]:
        if record["failure_class"] not in classes:
            continue
        present.add(record["failure_class"])
        bundles = set(record["affected_slice"]["bundle_ids"])
        hits.update(bundles)
        for b in bundles:
            pages.setdefault(b, set()).add(page_index)
    return hits, pages, present


def _page_mode_hits(
    batch: Sequence[Mapping[str, Any]], wanted: frozenset[str]
) -> tuple[Counter[str], dict[str, set[tuple[str, int]]]]:
    """Per-bundle hits and ``(bundle, page_index)`` pages over all reports; every
    wanted class must occur in at least one report."""
    hits: Counter[str] = Counter()
    bundle_pages: dict[str, set[tuple[str, int]]] = {}
    present: set[str] = set()
    for report in batch:
        r_hits, r_pages, r_present = _page_hits(report, wanted)
        hits.update(r_hits)
        present |= r_present
        for b, ks in r_pages.items():
            bundle_pages.setdefault(b, set()).update((b, k) for k in ks)
    absent = sorted(wanted - present)
    if absent:
        raise SelectionRefused(f"classes {absent} are in none of the given reports")
    return hits, bundle_pages


def _check_report(report: Mapping[str, Any], policy_parent_checkpoint_id: str) -> tuple[str, str]:
    """Validate one report; return ``(role, report_hash)``."""
    role = report.get("role")
    if isinstance(role, str):
        try:
            ft.check_role(role)
        except ft.ArtefactRoleRefused as exc:
            raise SelectionRefused(str(exc)) from exc
    _assert_train_cohort(role)
    if report.get("checkpoint_id") != policy_parent_checkpoint_id:
        raise SelectionRefused(
            f"report checkpoint_id {report.get('checkpoint_id')!r} != policy parent "
            f"{policy_parent_checkpoint_id!r}: a taxonomy of another checkpoint is not this "
            "policy's failure set"
        )
    claimed = report.get("report_hash")
    if not isinstance(claimed, str) or ft.hashed_report_hash(report) != claimed:
        raise SelectionRefused("report_hash does not match the report body (edited report)")
    return role, claimed  # type: ignore[return-value]


def _drop_listed(
    kept: Sequence[SelectedLineage], drop_lineages: Collection[str] | None
) -> tuple[str, ...]:
    if drop_lineages is None:
        return ()
    if isinstance(drop_lineages, str) or not all(isinstance(i, str) and i for i in drop_lineages):
        raise SelectionRefused("drop_lineages must be a collection of non-empty strings")
    wanted = set(drop_lineages)
    missing = sorted(wanted - {e.lineage_root_id for e in kept})
    if missing:
        raise SelectionRefused(f"drop_lineages not in the selection: {missing}")
    return tuple(sorted(wanted))


def build_selection(
    reports: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    policy_parent_checkpoint_id: str,
    allowed_lineage_ids: Collection[str],
    bundle_lineage: Mapping[str, str],
    budget: int,
    bundles_from: str = BUNDLES_FROM_EXAMPLES,
    classes: Collection[str] | None = None,
    drop_lineages: Collection[str] | None = None,
) -> FrozenSelection:
    """One report, or several (one per page index ``k``) of the same checkpoint and role.

    Hardness of a bundle is the number of failure classes it is an example of,
    summed over the reports it appears in. With one report the result (and its
    ``selection_hash``) is exactly what the single-report builder produced.

    ``bundles_from="affected"`` (opt-in) counts every bundle of a class's
    ``affected_slice`` instead of its <= ``MAX_EXAMPLES`` examples; the choice is
    recorded in the hashed ``policy``. Budget and lineage logic are unchanged.

    ``bundles_from="affected-pages"`` (opt-in) requires ``classes`` (failure-class
    names): only those classes count, and each bundle keeps the ``(bundle, page_index)``
    pages of the reports it was hit in, so RL trains on those pages only.

    ``drop_lineages`` (opt-in, lineage root ids) removes those lineages from the
    selection **after** the budget cut: we only remove, never refill from the
    lineages the budget cut off. Every id must be in the kept selection (a typo or a
    budget-cut id is refused, never ignored). The sorted ids are recorded in the
    hashed selection as ``dropped_lineages`` only when given."""
    if bundles_from not in _BUNDLES_FROM_POLICY:
        raise SelectionRefused(
            f"bundles_from must be one of {sorted(_BUNDLES_FROM_POLICY)}, got {bundles_from!r}"
        )
    page_mode = bundles_from == BUNDLES_FROM_AFFECTED_PAGES
    wanted = frozenset(classes) if classes is not None else frozenset()
    if page_mode and (isinstance(classes, str) or not wanted):
        raise SelectionRefused("affected-pages requires a non-empty collection of classes")
    if not page_mode and classes is not None:
        raise SelectionRefused(f"classes is only valid with affected-pages, not {bundles_from!r}")
    batch = [reports] if isinstance(reports, Mapping) else list(reports)
    if not batch:
        raise SelectionRefused("no report given")
    if not isinstance(policy_parent_checkpoint_id, str) or not policy_parent_checkpoint_id:
        raise SelectionRefused("policy_parent_checkpoint_id must be a non-empty string")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise SelectionRefused(f"budget must be an int >= 1, got {budget!r}")
    checked = [_check_report(r, policy_parent_checkpoint_id) for r in batch]
    roles = {role for role, _ in checked}
    if len(roles) != 1:
        raise SelectionRefused(f"reports carry different roles: {sorted(roles)}")
    role = checked[0][0]
    hashes = tuple(h for _, h in checked)
    if len(set(hashes)) != len(hashes):
        raise SelectionRefused("the same report is given twice (its examples would count twice)")
    claimed = hashes[0] if len(hashes) == 1 else domain_hash(REPORTS_DOMAIN, sorted(hashes))
    hits: Counter[str] = Counter()
    bundle_pages: dict[str, set[tuple[str, int]]] = {}
    if page_mode:
        hits, bundle_pages = _page_mode_hits(batch, wanted)
    else:
        for report in batch:
            hits.update(_bundle_hits(report, bundles_from))

    allowed = set(allowed_lineage_ids)
    per_lineage: dict[str, dict[str, int]] = {}
    unmapped = dropped = 0
    for bundle, n_classes in hits.items():
        lineage = bundle_lineage.get(bundle)
        if lineage is None:
            unmapped += 1
        elif lineage not in allowed:
            dropped += 1
        else:
            per_lineage.setdefault(lineage, {})[bundle] = n_classes * HARDNESS_PER_CLASS_MICRO
    entries = sorted(
        (
            SelectedLineage(
                lin,
                sum(b.values()),
                tuple(sorted(b)),
                tuple(sorted(p for bid in b for p in bundle_pages.get(bid, ()))),
            )
            for lin, b in per_lineage.items()
        ),
        key=lambda e: (-e.hardness_micro, e.lineage_root_id),
    )
    kept = tuple(entries[:budget])
    truncated = len(entries) - len(kept)
    dropped_lineages = _drop_listed(kept, drop_lineages)
    if dropped_lineages:
        kept = tuple(e for e in kept if e.lineage_root_id not in dropped_lineages)
    return FrozenSelection(
        status=STATUS_READY if kept else STATUS_INSUFFICIENT,
        entries=kept,
        policy=_BUNDLES_FROM_POLICY[bundles_from],
        report_hash=claimed,
        checkpoint_id=policy_parent_checkpoint_id,
        role=role,
        budget=budget,
        dropped_outside_allowed=dropped,
        unmapped_bundles=unmapped,
        truncated_by_budget=truncated,
        report_hashes=hashes if len(hashes) > 1 else (),
        classes=tuple(sorted(wanted)),
        dropped_lineages=dropped_lineages,
    )


def selection_hash(selection: FrozenSelection) -> str:
    if not selection.ready:
        raise SelectionRefused(
            "selection is insufficient_evidence (no lineage survived): it has no hash to bind"
        )
    return domain_hash(SELECTION_DOMAIN, selection.payload())


def require_selection_matches_manifest(
    selection: FrozenSelection, dataset_manifest: Mapping[str, Any]
) -> str:
    """Return the bound hash, or refuse when the manifest binds another one."""
    bound = dataset_manifest.get("selection_hash")
    if not isinstance(bound, str) or not _HASH_RE.match(bound):
        raise SelectionRefused(f"manifest selection_hash is not sha256:<64 hex>: {bound!r}")
    actual = selection_hash(selection)
    if bound != actual:
        raise SelectionRefused(
            f"dataset manifest selection_hash {bound} != frozen selection hash {actual}"
        )
    outside = sorted(set(selection.lineage_ids()) - set(dataset_manifest["lineage_ids"]))
    if outside:
        raise SelectionRefused(f"selected lineages not in the manifest's lineage_ids: {outside}")
    return actual


def assert_not_reweighted(selection_before_hash: str, selection_now: FrozenSelection) -> None:
    """Refuse when the selection differs from the one hashed before training."""
    now = selection_hash(selection_now)
    if now != selection_before_hash:
        raise SelectionReweighted(
            f"frozen selection changed after freeze: {selection_before_hash} -> {now}"
        )


SELECTION_DOCUMENT_HASH_FIELD: Final = "selection_hash"


def selection_document(selection: FrozenSelection) -> dict[str, Any]:
    """The on-disk JSON form: the hashed ``payload()`` plus ``report_hash``
    (provenance, outside the hash) and the ``selection_hash`` it must recompute to."""
    doc: dict[str, Any] = {
        **selection.payload(),
        "report_hash": selection.report_hash,
        SELECTION_DOCUMENT_HASH_FIELD: selection_hash(selection),
    }
    if selection.report_hashes:  # single-report documents stay byte-identical
        doc["report_hashes"] = list(selection.report_hashes)
    return doc


def _document_report_hashes(doc: Mapping[str, Any]) -> tuple[str, ...]:
    declared = doc.get("report_hashes")
    if declared is None:  # legacy / single-report document
        return ()
    hashes = tuple(str(h) for h in declared)
    if len(hashes) < 2 or len(set(hashes)) != len(hashes):
        raise ValueError(f"report_hashes must hold >= 2 distinct hashes: {declared!r}")
    if doc["report_hash"] != domain_hash(REPORTS_DOMAIN, sorted(hashes)):
        raise ValueError("report_hash is not the combined hash of report_hashes")
    return hashes


def _require_policy_matches_page_fields(selection: FrozenSelection) -> None:
    paged = selection.policy == SELECTION_POLICY_AFFECTED_PAGES
    if paged and (not selection.classes or not all(e.pages for e in selection.entries)):
        raise SelectionRefused(
            f"policy {selection.policy!r} needs non-empty classes and pages on every entry"
        )
    if not paged and (selection.classes or selection.selected_pages()):
        raise SelectionRefused(
            f"policy {selection.policy!r} must not carry classes or pages (page-level fields "
            f"belong to {SELECTION_POLICY_AFFECTED_PAGES!r})"
        )


def _document_dropped_lineages(doc: Mapping[str, Any]) -> tuple[str, ...]:
    if "dropped_lineages" not in doc:
        return ()
    declared = doc["dropped_lineages"]
    if (
        not isinstance(declared, list)
        or not declared
        or not all(isinstance(i, str) and i for i in declared)
        or declared != sorted(set(declared))
    ):
        raise SelectionRefused(
            "dropped_lineages must be a non-empty sorted list of distinct non-empty "
            f"strings, got {declared!r}"
        )
    return tuple(declared)


def selection_from_document(doc: Mapping[str, Any]) -> FrozenSelection:
    """Inverse of :func:`selection_document`; refuses a document whose declared
    ``selection_hash`` does not recompute (an edited file never loads silently)."""
    if not isinstance(doc, Mapping):
        raise SelectionRefused(f"selection document must be an object, got {type(doc).__name__}")
    try:
        entries = tuple(
            SelectedLineage(
                e["lineage_root_id"],
                int(e["hardness_micro"]),
                tuple(e["bundle_ids"]),
                tuple((str(b), int(k)) for b, k in e.get("pages", ())),
            )
            for e in doc["entries"]
        )
        selection = FrozenSelection(
            status=doc["status"],
            entries=entries,
            policy=doc["policy"],
            report_hash=doc["report_hash"],
            checkpoint_id=doc["checkpoint_id"],
            role=doc["role"],
            budget=doc["budget"],
            dropped_outside_allowed=doc["dropped_outside_allowed"],
            unmapped_bundles=doc["unmapped_bundles"],
            truncated_by_budget=doc["truncated_by_budget"],
            report_hashes=_document_report_hashes(doc),
            classes=tuple(str(c) for c in doc.get("classes", ())),
            dropped_lineages=_document_dropped_lineages(doc),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionRefused(f"selection document is malformed: {exc!r}") from exc
    _require_policy_matches_page_fields(selection)
    still_in = sorted(set(selection.dropped_lineages) & set(selection.lineage_ids()))
    if still_in:
        raise SelectionRefused(f"dropped_lineages are still among the entries: {still_in}")
    claimed = doc.get(SELECTION_DOCUMENT_HASH_FIELD)
    actual = selection_hash(selection)
    if claimed != actual:
        raise SelectionRefused(
            f"selection document declares {claimed!r}, its body hashes to {actual!r}"
        )
    return selection

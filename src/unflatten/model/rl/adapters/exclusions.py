"""Exclusion registry: which lineages S4.5 may never train on (DP8, inv. 3/15; torch-free).

``FORBIDDEN_SOURCES`` is a list of names, and on a real release none of them is a
field of a record: the only protection that can fire is "this lineage is in a
list". This module is where that list is built and verified, with one entry per
label, so that a forgotten label is an error and never an empty list.

* ``artefact`` -- a file whose bytes are sha256/size pinned and which names pages
  (an S5 control sample, a selection policy, the release's own ``test`` split).
  Every page it names must resolve to a lineage through the hash-verified split
  records; an unresolved page is listed in the error, never dropped.
* ``declared_absent`` -- the artefact does not exist; ``decision_ref`` must
  resolve to an existing file under ``docs/decisions/`` that names the label and
  declares a ratified status (``DECIDED`` / ``RATIFIED``; ``PROPOSED`` is refused).

The registry also carries ``cross_release_non_train``: bundles that were
non-train in another local release (splits are not stable across releases, the
rule of ``scripts/dspark_rehearsal.py:excluded_bundle_ids``), mapped to their
lineage in THIS release. A bundle the release does not hold cannot be loaded
from it and contributes nothing.

``exclude_lineage_ids`` is what ``data.load_train_pages`` takes as
``registry_exclude_lineage_ids``. The artefact ``kind`` is a closed enum, not a
free path reader.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from unflatten.model.hashing import domain_hash
from unflatten.model.rl.adapters.exclusions_io import RATIFIED_STATUSES, decision_status
from unflatten.model.rl.policies import FORBIDDEN_SOURCES, RlPolicyInvalid, _schema_check

REGISTRY_DOMAIN: Final = "unflatten.rlprep.exclusion-registry/v0"
SCHEMA_NAME: Final = "rl_exclusion_registry"
SCHEMA_VERSION: Final = "1.0.0"
HASH_FIELD: Final = "registry_hash"
KIND_S5_CONTROL_SAMPLE: Final = "s5_control_sample"
KIND_SELECTION_POLICY: Final = "selection_policy"
KIND_RELEASE_SPLIT_TEST: Final = "release_split_test"
ARTEFACT_KINDS: Final = (KIND_S5_CONTROL_SAMPLE, KIND_SELECTION_POLICY, KIND_RELEASE_SPLIT_TEST)
DECISIONS_PREFIX: Final = "docs/decisions/"
DEFAULT_DECISIONS_DIR: Final = Path(__file__).resolve().parents[5] / "docs" / "decisions"
TEST_SPLIT: Final = "test"
TRAIN_SPLIT: Final = "train"


class ExclusionRegistryRefused(ValueError):
    """The exclusion registry cannot be built / trusted."""


class ExclusionRegistryMalformed(ExclusionRegistryRefused):
    """Schema violation (closed at every level)."""


class ExclusionLabelMissing(ExclusionRegistryRefused):
    """A ``FORBIDDEN_SOURCES`` label has no entry (or an unknown/duplicate one is present)."""


class ExclusionArtefactMismatch(ExclusionRegistryRefused):
    """The artefact file is absent, or its sha256/size/lineages differ from the entry."""


class ExclusionArtefactUnresolved(ExclusionRegistryRefused):
    """An artefact names pages that resolve to no lineage of the release."""


class ExclusionArtefactEmpty(ExclusionRegistryRefused):
    """An artefact excludes nothing: no page, or pages but no lineage."""


class ExclusionDecisionRefInvalid(ExclusionRegistryRefused):
    """``declared_absent`` without a ``decision_ref`` that resolves and names the label."""


class ExclusionReleaseMismatch(ExclusionRegistryRefused):
    """The registry was built over another release than the one it is applied to."""


class ExclusionRegistryHashMismatch(ExclusionRegistryRefused):
    """``registry_hash`` does not recompute."""


@dataclass(frozen=True)
class ArtefactSpec:
    """Builder input for an ``artefact`` entry. ``ref`` is absolute or relative to
    ``artefact_root``; for ``release_split_test`` it is ignored (the manifest pin
    of the ``test`` split is used)."""

    kind: str
    ref: str = ""


@dataclass(frozen=True)
class DeclaredAbsentSpec:
    decision_ref: str | None


EntrySpec = ArtefactSpec | DeclaredAbsentSpec


def _sha(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def registry_hash_of(doc: Mapping[str, Any]) -> str:
    return domain_hash(REGISTRY_DOMAIN, {k: v for k, v in doc.items() if k != HASH_FIELD})


# --- release access ----------------------------------------------------------------


def _release_records(release_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    from unflatten.dataset.samples import release_root_hash
    from unflatten.model.rl.adapters.release import read_split_records_verified

    records = read_split_records_verified(release_dir)
    return release_root_hash(release_dir), records


def _bundle_lineages(records: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        out[rec["authority_id"]].add(rec["lineage_root_id"])
    return out


# --- artefact readers (closed kinds) ---------------------------------------------


def _resolve_path(ref: str, artefact_root: Path | None, what: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    if artefact_root is None:
        raise ExclusionArtefactMismatch(f"{what}: relative ref {ref!r} needs an artefact_root")
    return artefact_root / path


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise ExclusionArtefactMismatch(f"{what}: artefact file {path} is missing")
    try:
        return json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise ExclusionArtefactMismatch(f"{what}: {path} is not JSON ({exc})") from exc


def _bundles_s5_control_sample(doc: Any, what: str) -> list[str]:
    pages = doc.get("pages") if isinstance(doc, dict) else None
    if not isinstance(pages, list):
        raise ExclusionArtefactMismatch(f"{what}: control sample has no 'pages' list")
    out: list[str] = []
    for i, page in enumerate(pages):
        bundle = page.get("document_bundle_id") if isinstance(page, dict) else None
        if not isinstance(bundle, str) or not bundle:
            raise ExclusionArtefactMismatch(f"{what}: pages[{i}] has no document_bundle_id")
        out.append(bundle)
    return out


def _bundles_selection_policy(doc: Any, what: str) -> list[str]:
    cohort = doc.get("cohort") if isinstance(doc, dict) else None
    pinned = cohort.get("pinned_pages") if isinstance(cohort, dict) else None
    if not isinstance(pinned, list):
        raise ExclusionArtefactMismatch(f"{what}: selection policy has no cohort.pinned_pages")
    out: list[str] = []
    for i, pair in enumerate(pinned):
        if not (isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str)):
            raise ExclusionArtefactMismatch(
                f"{what}: cohort.pinned_pages[{i}] is not [bundle, page]"
            )
        out.append(pair[0])
    return out


def _test_split_ref(release_dir: Path, what: str) -> str:
    from unflatten.dataset.samples import SampleSelectionError, _release_manifest

    manifest = _release_manifest(release_dir)
    for entry in manifest.get("split_manifests") or ():
        if isinstance(entry, dict) and entry.get("split_name") == TEST_SPLIT:
            return str(entry["ref"])
    raise SampleSelectionError(f"{what}: the release manifest pins no {TEST_SPLIT} split")


def _read_artefact(
    kind: str,
    ref: str,
    *,
    release_dir: Path,
    records: Sequence[Mapping[str, Any]],
    artefact_root: Path | None,
    what: str,
) -> tuple[str, bytes, list[str]]:
    """``(ref, file bytes, bundles named)`` for one closed artefact kind."""
    if kind == KIND_RELEASE_SPLIT_TEST:
        ref = _test_split_ref(release_dir, what)
        data = (release_dir / ref).read_bytes()
        bundles = [r["authority_id"] for r in records if r.get("split_name") == TEST_SPLIT]
        return ref, data, bundles
    if kind not in ARTEFACT_KINDS:
        raise ExclusionArtefactMismatch(f"{what}: artefact kind {kind!r} not in {ARTEFACT_KINDS}")
    path = _resolve_path(ref, artefact_root, what)
    doc = _read_json(path, what)
    reader = (
        _bundles_s5_control_sample if kind == KIND_S5_CONTROL_SAMPLE else _bundles_selection_policy
    )
    return ref, path.read_bytes(), reader(doc, what)


def _s5_release_root(data: bytes) -> str | None:
    """The release a control sample was drawn from: its own ``release_root_hash``."""
    try:
        doc = json.loads(data)
    except json.JSONDecodeError:
        return None
    root = doc.get("release_root_hash") if isinstance(doc, dict) else None
    return root if isinstance(root, str) else None


def _artefact_lineages(
    kind: str,
    data: bytes,
    bundles: Sequence[str],
    *,
    lineages: Mapping[str, set[str]],
    foreign: Mapping[str, Mapping[str, set[str]]] | None,
    recorded_roots: Iterable[str] = (),
    what: str,
) -> list[str] | None:
    """Lineages of an artefact's pages. An S5 control sample drawn from an older
    release (its bundle ids are not in this one; lineage ids are stable across
    releases) resolves through that release's hash-verified records, which must be
    passed as a cross release (DEC-079 addendum 2026-10-17). ``foreign=None`` (a
    verify without the other release) trusts the sealed lineages only when that
    release is one the registry recorded at build time -- returns ``None``."""
    if kind != KIND_S5_CONTROL_SAMPLE or all(b in lineages for b in bundles):
        return _resolve_lineages(bundles, lineages, what)
    own_root = _s5_release_root(data)
    if foreign is None and own_root is not None and own_root in set(recorded_roots):
        return None
    if foreign is None or own_root is None or own_root not in foreign:
        unresolved = sorted({b for b in bundles if b not in lineages})
        raise ExclusionArtefactUnresolved(
            f"{what}: {len(unresolved)} bundle(s) resolve to no lineage of the release: "
            f"{unresolved}; the sample is bound to release {own_root!r} -- if that is an older "
            "release, pass it via --cross-release"
        )
    return _resolve_lineages(bundles, foreign[own_root], what)


def _resolve_lineages(
    bundles: Iterable[str], lineages: Mapping[str, set[str]], what: str
) -> list[str]:
    unresolved = sorted({b for b in bundles if b not in lineages})
    if unresolved:
        raise ExclusionArtefactUnresolved(
            f"{what}: {len(unresolved)} bundle(s) resolve to no lineage of the "
            f"release: {unresolved}"
        )
    return sorted({lin for b in bundles for lin in lineages[b]})


# --- decision refs -------------------------------------------------------------------


def _check_decision_ref(label: str, ref: Any, decisions_dir: Path) -> None:
    what = f"declared_absent {label!r}"
    if not isinstance(ref, str) or not ref:
        raise ExclusionDecisionRefInvalid(f"{what}: no decision_ref (an unattested absence)")
    if not ref.startswith(DECISIONS_PREFIX) or ".." in Path(ref).parts:
        raise ExclusionDecisionRefInvalid(
            f"{what}: decision_ref {ref!r} is not under {DECISIONS_PREFIX}"
        )
    path = decisions_dir / ref[len(DECISIONS_PREFIX) :]
    if not path.is_file():
        raise ExclusionDecisionRefInvalid(f"{what}: decision_ref {ref!r} resolves to no file")
    if label not in path.read_text(encoding="utf-8"):
        raise ExclusionDecisionRefInvalid(f"{what}: decision {ref!r} does not name the label")
    status = decision_status(path.read_text(encoding="utf-8"))
    if status not in RATIFIED_STATUSES:
        raise ExclusionDecisionRefInvalid(
            f"{what}: decision {ref!r} has status {status!r}, not one of "
            f"{sorted(RATIFIED_STATUSES)} (an absence attested by an unratified decision)"
        )


# --- build ----------------------------------------------------------------------------


def _foreign_lineages(cross_release_dirs: Iterable[Path | str]) -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for other in cross_release_dirs:
        root, records = _release_records(Path(other))
        out[root] = _bundle_lineages(records)
    return out


def _cross_release_entries(
    cross_release_dirs: Iterable[Path | str], lineages: Mapping[str, set[str]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for other in cross_release_dirs:
        root, records = _release_records(Path(other))
        non_train = {r["authority_id"] for r in records if r.get("split_name") != TRAIN_SPLIT}
        here = sorted({lin for b in non_train if b in lineages for lin in lineages[b]})
        out.append({"release_root_hash": root, "lineage_ids": here})
    return sorted(out, key=lambda e: e["release_root_hash"])


def _entry_from_spec(
    label: str,
    spec: EntrySpec,
    *,
    release_dir: Path,
    records: Sequence[Mapping[str, Any]],
    lineages: Mapping[str, set[str]],
    foreign: Mapping[str, Mapping[str, set[str]]],
    artefact_root: Path | None,
    decisions_dir: Path,
) -> dict[str, Any]:
    if isinstance(spec, DeclaredAbsentSpec):
        _check_decision_ref(label, spec.decision_ref, decisions_dir)
        return {
            "label": label, "status": "declared_absent", "kind": None, "ref": None,
            "sha256": None, "size_bytes": None, "lineage_ids": [],
            "decision_ref": spec.decision_ref,
        }  # fmt: skip
    what = f"artefact {label!r}"
    ref, data, bundles = _read_artefact(
        spec.kind, spec.ref, release_dir=release_dir, records=records,
        artefact_root=artefact_root, what=what,
    )  # fmt: skip
    if not bundles:
        raise ExclusionArtefactEmpty(f"{what}: names no page, so it would exclude nothing")
    resolved = _artefact_lineages(
        spec.kind, data, bundles, lineages=lineages, foreign=foreign, what=what
    )
    return {
        "label": label, "status": "artefact", "kind": spec.kind, "ref": ref,
        "sha256": _sha(data), "size_bytes": len(data),
        "lineage_ids": resolved, "decision_ref": None,
    }  # fmt: skip


def build_exclusion_registry(
    release_dir: Path | str,
    entries: Mapping[str, EntrySpec],
    *,
    cross_release_dirs: Iterable[Path | str] = (),
    artefact_root: Path | str | None = None,
    decisions_dir: Path | str = DEFAULT_DECISIONS_DIR,
) -> dict[str, Any]:
    """Seal a registry over ``release_dir``; every label of ``FORBIDDEN_SOURCES``
    needs a spec. The result is re-verified before it is returned."""
    release = Path(release_dir)
    missing = [lab for lab in FORBIDDEN_SOURCES if lab not in entries]
    unknown = sorted(set(entries) - set(FORBIDDEN_SOURCES))
    if missing or unknown:
        raise ExclusionLabelMissing(f"missing label(s) {missing}, unknown label(s) {unknown}")
    root, records = _release_records(release)
    lineages = _bundle_lineages(records)
    cross_dirs = list(cross_release_dirs)
    foreign = _foreign_lineages(cross_dirs)
    root_dir = None if artefact_root is None else Path(artefact_root)
    built = [
        _entry_from_spec(
            label, entries[label], release_dir=release, records=records, lineages=lineages,
            foreign=foreign, artefact_root=root_dir, decisions_dir=Path(decisions_dir),
        )
        for label in FORBIDDEN_SOURCES
    ]  # fmt: skip
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release_root_hash": root,
        "entries": built,
        "cross_release_non_train": _cross_release_entries(cross_dirs, lineages),
    }
    doc = {**body, HASH_FIELD: registry_hash_of(body)}
    verify_exclusion_registry(
        doc, release_root_hash=root, release_dir=release, artefact_root=root_dir,
        decisions_dir=decisions_dir,
    )  # fmt: skip
    return doc


# --- verify ---------------------------------------------------------------------------


def _check_structure(doc: Mapping[str, Any]) -> None:
    try:
        _schema_check(SCHEMA_NAME, doc)
    except RlPolicyInvalid as exc:
        raise ExclusionRegistryMalformed(str(exc)) from exc
    expected = registry_hash_of(doc)
    if doc[HASH_FIELD] != expected:
        raise ExclusionRegistryHashMismatch(
            f"registry_hash {doc[HASH_FIELD]!r} != recomputed {expected!r}"
        )
    labels = [e["label"] for e in doc["entries"]]
    missing = [lab for lab in FORBIDDEN_SOURCES if lab not in labels]
    extra = sorted({lab for lab in labels if labels.count(lab) > 1})
    if missing or extra:
        raise ExclusionLabelMissing(f"missing label(s) {missing}, duplicated label(s) {extra}")


def _verify_artefact(
    entry: Mapping[str, Any],
    *,
    release_dir: Path,
    records: Sequence[Mapping[str, Any]],
    lineages: Mapping[str, set[str]],
    foreign: Mapping[str, Mapping[str, set[str]]] | None,
    recorded_roots: Iterable[str],
    artefact_root: Path | None,
) -> None:
    what = f"artefact {entry['label']!r}"
    ref, data, bundles = _read_artefact(
        entry["kind"], entry["ref"], release_dir=release_dir, records=records,
        artefact_root=artefact_root, what=what,
    )  # fmt: skip
    if _sha(data) != entry["sha256"] or len(data) != entry["size_bytes"]:
        raise ExclusionArtefactMismatch(
            f"{what}: {ref} is {_sha(data)} / {len(data)} bytes, entry pins "
            f"{entry['sha256']} / {entry['size_bytes']}"
        )
    if not bundles:
        raise ExclusionArtefactEmpty(f"{what}: names no page, so it would exclude nothing")
    derived = _artefact_lineages(
        entry["kind"], data, bundles, lineages=lineages, foreign=foreign,
        recorded_roots=recorded_roots, what=what,
    )  # fmt: skip
    recorded = list(entry["lineage_ids"])
    if not recorded:
        raise ExclusionArtefactEmpty(f"{what}: empty lineage_ids but the artefact names pages")
    if derived is not None and sorted(recorded) != derived:
        raise ExclusionArtefactMismatch(
            f"{what}: lineage_ids differ from the artefact's pages "
            f"(recorded {len(recorded)}, derived {len(derived)})"
        )


def verify_exclusion_registry(
    doc: Mapping[str, Any],
    *,
    release_root_hash: str,
    release_dir: Path | str,
    artefact_root: Path | str | None = None,
    decisions_dir: Path | str = DEFAULT_DECISIONS_DIR,
    cross_release_dirs: Iterable[Path | str] | None = None,
) -> None:
    """Refuse (named errors) unless ``doc`` is a sealed, closed, truthful registry
    for ``release_dir``. ``cross_release_dirs`` (optional) re-derives the
    cross-release lists; without it they are trusted under the self-hash."""
    _check_structure(doc)
    if doc["release_root_hash"] != release_root_hash:
        raise ExclusionReleaseMismatch(
            f"registry built over {doc['release_root_hash']!r}, applied to {release_root_hash!r}"
        )
    root, records = _release_records(Path(release_dir))
    if root != release_root_hash:
        raise ExclusionReleaseMismatch(
            f"release_dir has root hash {root!r}, not {release_root_hash!r}"
        )
    lineages = _bundle_lineages(records)
    cross_dirs = None if cross_release_dirs is None else list(cross_release_dirs)
    foreign = None if cross_dirs is None else _foreign_lineages(cross_dirs)
    recorded_roots = [e["release_root_hash"] for e in doc["cross_release_non_train"]]
    root_dir = None if artefact_root is None else Path(artefact_root)
    for entry in doc["entries"]:
        if entry["status"] == "declared_absent":
            _check_decision_ref(entry["label"], entry["decision_ref"], Path(decisions_dir))
        else:
            _verify_artefact(
                entry, release_dir=Path(release_dir), records=records, lineages=lineages,
                foreign=foreign, recorded_roots=recorded_roots, artefact_root=root_dir,
            )  # fmt: skip
    if cross_dirs is not None:
        derived = _cross_release_entries(cross_dirs, lineages)
        if derived != list(doc["cross_release_non_train"]):
            raise ExclusionArtefactMismatch(
                "cross_release_non_train differs from the other release(s) re-read"
            )


def exclude_lineage_ids(doc: Mapping[str, Any]) -> frozenset[str]:
    """Union of every entry's and every cross-release list's lineages. The
    document is structure- and self-hash-checked first, so a tampered registry
    never yields a (smaller) list."""
    _check_structure(doc)
    out: set[str] = set()
    for entry in doc["entries"]:
        out.update(entry["lineage_ids"])
    for cross in doc["cross_release_non_train"]:
        out.update(cross["lineage_ids"])
    return frozenset(out)

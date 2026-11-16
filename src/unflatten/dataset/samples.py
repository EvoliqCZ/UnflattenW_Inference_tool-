"""Release → training sample enumeration.

The bridge between an immutable release and a training loop. It answers
one question: *which page images may this task learn from, in this
split?*

Two indexes have to be joined to answer it, and joining them wrong is
how leakage or biased training sets happen:

- **the release** decides *which split* a document belongs to
  (`splits/*.jsonl`, lineage-first), *which task views* it qualifies for
  (`quality/gate-report.jsonl`, RFC003 §33.4) and *which bytes it meant*
  (`artifacts/index.jsonl`);
- **the store** holds those bytes, content-addressed under `objects/`.

Artifacts are resolved from the release's pinned index, never from the
store's append-only `renders.jsonl`. That ledger is mutable — one more
M2b row, or an `m2b --fresh` that rewrites it, would otherwise make an
UNCHANGED release start delivering different page images, sidecars,
fragments and targets. A release that cannot prove which bytes it meant
is not an authority.

Selecting on the document-level `eligibility` instead of the per-task one
would silently discard every `partial_ready` document, which on the real
corpus is ~two thirds of the population and — because long documents fail
full geometry coverage far more often than short ones — almost all of the
long ones. That is why the split manifests are NOT used as the task
membership list: `partial.jsonl` records its members' `split_name`, so a
document can be in the train split and still be listed there.

Nothing here is trusted because a ledger says so. The gate report and
the artifact index are resolved through the release manifest and must
match their manifest refs (the report additionally its
`quality/gate-report.sha256` sidecar), and the manifest's own
`release_root_hash` must recompute — both decide what trains, so both
are checked against the release identity that covers them. Every
resolved artifact ref must agree with its content address (the store is
content-addressed: `objects/<sha256>`), exist, and have the declared
size.

No object bytes are read here; a sample carries the `ArtifactRef`s and
the caller reads them through `gates.store.read_object`, which hashes
the bytes on read. That keeps this layer torch-free and cheap to test
while the byte-level check still happens exactly once, at read time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.dataset.gates.task_eligibility import TASK_KINDS
from unflatten.dataset.identity import domain_hash
from unflatten.dataset.split.release import RELEASE_ROOT_DOMAIN

#: Split names a release can assign (RFC003 §40).
RELEASE_SPLITS: tuple[str, ...] = ("train", "val_iid", "val_template_ood", "test")

#: Split manifest files a release writes. `partial.jsonl` is not a split —
#: it holds members of every split whose document-level eligibility is
#: `partial_ready`, each tagged with its own `split_name`.
_MANIFEST_FILES: tuple[str, ...] = (*RELEASE_SPLITS, "partial")


class SampleSelectionError(ValueError):
    """The requested split/task selection cannot be satisfied."""


@dataclass(frozen=True)
class PageSample:
    """One rendered page, with everything needed to build a training item."""

    document_bundle_id: str
    lineage_root_id: str
    split: str
    task_kind: str
    page_index: int
    page_count: int
    image: dict[str, Any]
    target_bundle: dict[str, Any] | None
    observation_sidecar: dict[str, Any] | None
    mask_projection: dict[str, Any] | None
    #: Index of per-page fragments (RFC003 §26). Document-level targets
    #: describe the whole document, so labelling a page with them would
    #: mark page 1 as containing a table that is on page 3. The fragment
    #: for this page is what carries page-local slices; it is resolved
    #: lazily from this index, not read during enumeration.
    fragment_index: dict[str, Any] | None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _split_membership(release_dir: Path) -> dict[str, tuple[str, str]]:
    """`bundle_id -> (split_name, lineage_root_id)` across all manifests."""
    membership: dict[str, tuple[str, str]] = {}
    for name in _MANIFEST_FILES:
        for row in _read_jsonl(release_dir / "splits" / f"{name}.jsonl"):
            # `partial.jsonl` carries the real split in `split_name`; the
            # per-split manifests are named for theirs.
            split = row.get("split_name", name)
            membership[row["authority_id"]] = (split, row["lineage_root_id"])
    return membership


def _release_manifest(release_dir: Path) -> dict[str, Any]:
    """The release manifest, with its own identity re-verified."""
    manifest_path = release_dir / "release-manifest.json"
    if not manifest_path.is_file():
        raise SampleSelectionError(
            f"release {release_dir} has no release-manifest.json; nothing it "
            "contains can be verified against a release identity"
        )
    manifest = json.loads(manifest_path.read_bytes())
    projection = {k: v for k, v in manifest.items() if k != "release_root_hash"}
    if domain_hash(RELEASE_ROOT_DOMAIN, projection) != manifest.get(
        "release_root_hash"
    ):
        raise SampleSelectionError(
            f"release {release_dir}: release_root_hash does not recompute "
            "from the manifest projection"
        )
    return manifest


def release_root_hash(release_dir: Path) -> str:
    """The release's own re-verified identity hash (see `_release_manifest`)."""
    return str(_release_manifest(release_dir)["release_root_hash"])


def _pinned_rows_entry(
    release_dir: Path, entry: Any, *, key_label: str
) -> list[dict[str, Any]]:
    """Rows of one JSONL artifact ref, hash-verified against `entry`.

    `key_label` names the entry for error messages only — the caller may
    pass a single manifest key (``"gate_report"``) or a split-manifest
    entry's own ``split_name`` (§4.3: `_pinned_rows` calls this once per
    manifest key; `m2c.load_prior_assignment` calls it once per entry of
    the `split_manifests` LIST plus `partial_manifest`)."""
    if not isinstance(entry, dict):
        raise SampleSelectionError(
            f"release {release_dir}: the manifest records no {key_label}, so "
            "what may train is outside the release identity"
        )
    path = release_dir / entry["ref"]
    if not path.is_file():
        raise SampleSelectionError(f"release {key_label} {entry['ref']} is missing")
    data = path.read_bytes()
    digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
    if digest != entry.get("sha256") or len(data) != entry.get("size_bytes"):
        raise SampleSelectionError(
            f"release {key_label} {entry['ref']} does not match the manifest ref "
            f"({digest} / {len(data)} bytes)"
        )
    if key_label == "gate_report":
        sidecar = release_dir / "quality" / "gate-report.sha256"
        if not sidecar.is_file() or sidecar.read_text().strip() != digest:
            raise SampleSelectionError(
                "release gate-report.sha256 does not pin the gate report bytes"
            )
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def _pinned_rows(
    release_dir: Path, manifest: dict[str, Any], key: str
) -> list[dict[str, Any]]:
    """Rows of a JSONL artifact the manifest pins, hash-verified."""
    return _pinned_rows_entry(release_dir, manifest.get(key), key_label=key)


def _privacy_pending(row: dict[str, Any]) -> bool:
    """Does this gate result record an unfinished privacy decision?"""
    for check in row.get("gate_report", []):
        if check.get("gate_id") != "privacy":
            continue
        return check.get("status") != "pass" or any(
            issue.get("code") == "PRIVACY_REVIEW_PENDING"
            for issue in check.get("issues", [])
        )
    return True  # no privacy gate recorded at all is not a completed review


def _verify_ref(store_dir: Path, ref: Any, what: str) -> dict[str, Any]:
    """Check one ArtifactRef against the content-addressed store."""
    if not isinstance(ref, dict):
        raise SampleSelectionError(f"{what}: artifact ref is not an object")
    declared, path = ref.get("sha256"), ref.get("ref")
    if not isinstance(declared, str) or not declared.startswith("sha256:"):
        raise SampleSelectionError(f"{what}: artifact ref declares no sha256")
    if path != f"objects/{declared[len('sha256:'):]}":
        raise SampleSelectionError(
            f"{what}: ref {path!r} is not the content address of {declared}"
        )
    resolved = store_dir / path
    if not resolved.is_file():
        raise SampleSelectionError(f"{what}: object {path!r} is missing")
    size = resolved.stat().st_size
    if size != ref.get("size_bytes"):
        raise SampleSelectionError(
            f"{what}: object {path!r} has size {size}, declared "
            f"{ref.get('size_bytes')!r}"
        )
    return ref


def iter_page_samples(
    release_dir: Path | str,
    store_dir: Path | str,
    split: str,
    task_kind: str,
    *,
    eligibilities: Sequence[str] = ("train_ready",),
    require_privacy_review: bool = False,
) -> Iterator[PageSample]:
    """Yield one `PageSample` per rendered page eligible for `task_kind`.

    Order is deterministic: by document bundle id, then page index.

    `require_privacy_review` is the training/admission boundary
    (RFC005 §9.2): an incomplete privacy decision is BLOCKED for
    training, while the same release stays usable as dev/research
    inventory (§9.3 run classes `research | smoke | coverage-building`).
    With it set, a selected document whose recorded privacy decision is
    not complete raises instead of quietly training on it.
    """
    if task_kind not in TASK_KINDS:
        raise SampleSelectionError(
            f"unknown task kind {task_kind!r}; expected one of {', '.join(TASK_KINDS)}"
        )
    if split not in RELEASE_SPLITS:
        raise SampleSelectionError(
            f"unknown split {split!r}; expected one of {', '.join(RELEASE_SPLITS)}"
        )
    release_dir, store_dir = Path(release_dir), Path(store_dir)
    manifest = _release_manifest(release_dir)
    membership = _split_membership(release_dir)
    rows = _pinned_rows(release_dir, manifest, "gate_report")
    views = {
        row["document_bundle_id"]: row.get("task_eligibility") or {} for row in rows
    }
    pending_privacy = {
        row["document_bundle_id"]
        for row in rows
        if _privacy_pending(row)
    }
    artifacts = {
        row["document_bundle_id"]: row.get("artifacts") or {}
        for row in _pinned_rows(release_dir, manifest, "artifact_index")
    }
    wanted = frozenset(eligibilities)

    for bundle_id in sorted(membership):
        assigned_split, lineage_root_id = membership[bundle_id]
        if assigned_split != split:
            continue
        view = views.get(bundle_id, {}).get(task_kind)
        if view is None or view.get("eligibility") not in wanted:
            continue
        if require_privacy_review and bundle_id in pending_privacy:
            raise SampleSelectionError(
                f"{bundle_id} carries an incomplete privacy review "
                "(PRIVACY_REVIEW_PENDING); RFC005 §9.2 blocks it from "
                "training. Use a release whose privacy decisions are "
                "closed, or run as dev/research inventory without "
                "require_privacy_review."
            )
        # A document with no pinned page image has no model input,
        # whatever its labels say — it cannot be a sample.
        bundle_artifacts = artifacts.get(bundle_id) or {}
        pages = bundle_artifacts.get("pages") or []
        refs = {
            name: _verify_ref(store_dir, ref, f"{bundle_id}/{name}")
            for name in (
                "target_bundle",
                "observation_sidecar",
                "mask_projection",
                "fragment_index",
            )
            if (ref := bundle_artifacts.get(name)) is not None
        }
        for page_index, image in enumerate(pages):
            yield PageSample(
                document_bundle_id=bundle_id,
                lineage_root_id=lineage_root_id,
                split=split,
                task_kind=task_kind,
                page_index=page_index,
                page_count=len(pages),
                image=_verify_ref(
                    store_dir, image, f"{bundle_id}/pages/{page_index}"
                ),
                target_bundle=refs.get("target_bundle"),
                observation_sidecar=refs.get("observation_sidecar"),
                mask_projection=refs.get("mask_projection"),
                fragment_index=refs.get("fragment_index"),
            )


def count_page_samples(
    release_dir: Path | str,
    store_dir: Path | str,
    split: str,
    task_kind: str,
    *,
    eligibilities: Sequence[str] = ("train_ready",),
    require_privacy_review: bool = False,
) -> int:
    """Number of page samples `iter_page_samples` would yield."""
    return sum(
        1
        for _ in iter_page_samples(
            release_dir,
            store_dir,
            split,
            task_kind,
            eligibilities=eligibilities,
            require_privacy_review=require_privacy_review,
        )
    )

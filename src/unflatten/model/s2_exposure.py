"""S2 exposure ledger, run-attempt manifests and release exposure manifests
(RFC005 §44.2–§44.3).

The exposure ledger is the append-only authority for what an S2 run has
shown the model: one record per exposure batch, each carrying the ordered
task-example view ids, the lineage root, fragment and augmentation ids,
the sample count and the optimizer step (§44.2). Records chain by hash
(``prev_record_hash`` → ``record_hash``), so a rewritten, truncated or
reordered ledger is detectable by re-walking the chain
(:meth:`ExposureLedger.verify`).

Everything else in this module is a DERIVED, hash-pinned view over the
ledger:

- A **RunAttemptManifest** (§44.3) is written when an execution attempt
  starts and stays ``terminal_state = "running"`` forever. Finalising the
  attempt creates a NEW manifest document in one of ``completed |
  failed | cancelled`` — the running document is never edited, because a
  crash between an edit and its write would leave a manifest claiming
  ``running`` with no way to know whether it ever ended. ``attempt_id``
  is derived from ``(run_id, attempt_index)``, so re-execution takes the
  next index and can never collide with a burned one.
- A **ReleaseExposureManifest** is a checkpoint derived entirely from the
  ledger (lineage exposure counts, fragment/augmentation id sets, record
  count, ledger root hash). Merging a further attempt into it
  (:func:`merge_release`) is a MONOTONIC union: the ledger only grows, so
  every count can only go up — a decrease means the ledger no longer
  contains the release's snapshot and is refused, never repaired.

Design decisions this module makes explicit (the RFC leaves them open):

- an attempt that crashed BEFORE recording any exposure contributes zero
  ledger records; it still appears in a release's ``attempt_ids`` because
  its RunAttemptManifest exists, but adds nothing to the exposure sets;
- a release built from a ledger with no records is refused: a release is
  a statement about recorded exposure, and there is nothing to checkpoint
  without at least one record (the ledger itself remains the durable
  authority and outlives any release);
- finalising an attempt re-reads the ledger snapshot, so the terminal
  manifest says what the ledger looks like at the END of the attempt,
  while the running one pinned it at the start;
- all quantities are integers: the deterministic CBOR profile these
  hashes are computed under has no float.

This module is torch-free on purpose: exposure bookkeeping is pure
deterministic state that must stay importable and testable without a
model stack.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema

from unflatten.model.hashing import domain_hash, make_id

_SCHEMAS_DIR = Path(__file__).parent / "schemas"

RUN_ATTEMPT_MANIFEST_SCHEMA_VERSION = "1.0.0"
RELEASE_EXPOSURE_MANIFEST_SCHEMA_VERSION = "1.0.0"

#: Domain constants — each artefact kind hashes under its own domain so a
#: valid-but-wrong artefact can never collide across kinds.
EXPOSURE_RECORD_DOMAIN = "rfc005.s2-exposure-record/v1"
EXPOSURE_LEDGER_ROOT_DOMAIN = "rfc005.s2-exposure-ledger-root/v1"
EXECUTION_PROFILE_DOMAIN = "rfc005.s2-execution-profile/v1"
RUN_ATTEMPT_DOMAIN = "rfc005.s2-run-attempt-manifest/v1"
RUN_ATTEMPT_ID_DOMAIN = "rfc005.s2-run-attempt-id/v1"
RELEASE_EXPOSURE_DOMAIN = "rfc005.s2-release-exposure-manifest/v1"
RELEASE_EXPOSURE_ID_DOMAIN = "rfc005.s2-release-exposure-id/v1"

_RUN_ATTEMPT_STATES = ("running", "completed", "failed", "cancelled")
_TERMINAL_STATES = ("completed", "failed", "cancelled")

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ATTEMPT_ID_RE = re.compile(r"^attempt_[0-9a-f]{64}$")


class ExposureLedgerInvalid(ValueError):
    """The ledger or an append to it is malformed: a field of the wrong
    type, a duplicate id where the ledger's append-only shape allows none,
    or a run identity mismatch. A refusal, not a repair."""


class ExposureLedgerBroken(ValueError):
    """Re-walking the ledger's hash chain failed: a record's stored hash
    does not match a recomputation, a ``prev_record_hash`` link is broken,
    or an index is out of sequence. The ledger must be treated as
    untrusted and the run's exposure re-derived from source artefacts."""


class RunAttemptManifestInvalid(ValueError):
    """The RunAttemptManifest document is not well-formed: schema
    violation, a self-hash that does not verify, or a state transition the
    §44.3 crash-safe protocol forbids (finalising something already
    terminal)."""


class ReleaseExposureInvalid(ValueError):
    """A release exposure manifest cannot be built or verified: a ledger
    that does not contain the release's snapshot, a monotonicity
    violation on merge, an attempt id the ledger references that the
    release omits, or an empty ledger (nothing to release)."""


@dataclass(frozen=True, slots=True)
class ExposureRecord:
    """One append-only exposure record (§44.2). ``cell`` is the
    (split, pyramid_level, task_kind, cohort) coordinate the drawn example
    came from; ``plan_draw_index`` links the record back to its draw in
    the frozen ordered exposure plan (None for exposures not planned by a
    virtual epoch, e.g. a retry)."""

    record_index: int
    run_id: str
    attempt_id: str
    lineage_root_id: str
    cell: tuple[str, str, str, str]
    view_ids: tuple[str, ...]
    fragment_ids: tuple[str, ...]
    augmentation_ids: tuple[str, ...]
    sample_count: int
    optimizer_step: int
    plan_draw_index: int | None
    prev_record_hash: str | None
    record_hash: str


@cache
def _run_attempt_validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMAS_DIR / "run_attempt_manifest.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


@cache
def _release_validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMAS_DIR / "release_exposure_manifest.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _require_nonempty_str(value: object, name: str, *, exc: type[ValueError]) -> str:
    if not isinstance(value, str) or not value:
        raise exc(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_int(value: object, name: str, *, minimum: int, exc: type[ValueError]) -> int:
    # `type is int` on purpose: `bool` is an `int` subclass and a `true`
    # quantity would otherwise pass an `isinstance` check.
    if type(value) is not int or value < minimum:
        raise exc(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_id_sequence(value: object, name: str, *, exc: type[ValueError]) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise exc(f"{name} must be a list of unique non-empty ids, got {value!r}")
    items = tuple(value)
    if len(set(items)) != len(items):
        dupes = sorted({item for item in items if items.count(item) > 1})
        raise exc(f"{name} contains duplicate ids: {dupes}")
    for item in items:
        if not isinstance(item, str) or not item:
            raise exc(f"{name} may only hold non-empty strings, got {item!r}")
    return items


def _require_cell(value: object, name: str, *, exc: type[ValueError]) -> tuple[str, str, str, str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise exc(
            f"{name} must be a (split, pyramid_level, task_kind, cohort) 4-tuple, "
            f"got {value!r}"
        )
    for item in value:
        if not isinstance(item, str) or not item:
            raise exc(f"{name} may only hold non-empty strings, got {item!r}")
    return tuple(value)  # type: ignore[return-value]


def _require_integrity_record(value: object, name: str, *, exc: type[ValueError]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise exc(f"{name} must be a non-empty {{name: sha256:...}} record, got {value!r}")
    record: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str) or not key:
            raise exc(f"{name} keys must be non-empty strings, got {key!r}")
        if not isinstance(val, str) or not _SHA256_RE.match(val):
            raise exc(f"{name}[{key!r}] must be a 'sha256:<64 hex>' digest, got {val!r}")
        record[key] = val
    return record


def _require_profile(value: object, name: str, *, exc: type[ValueError]) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise exc(f"{name} must be a non-empty {{name: string}} profile, got {value!r}")
    profile: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str) or not key:
            raise exc(f"{name} keys must be non-empty strings, got {key!r}")
        if not isinstance(val, str) or not val:
            raise exc(f"{name}[{key!r}] must be a non-empty string, got {val!r}")
        profile[key] = val
    return profile


def execution_profile_hash(profile: Mapping[str, str]) -> str:
    """The §44.3 'execution_profile semantic hash': one domain hash over
    the closed {name: string} profile an attempt ran under."""
    clean = _require_profile(profile, "execution_profile", exc=RunAttemptManifestInvalid)
    return domain_hash(EXECUTION_PROFILE_DOMAIN, clean)


def _record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """The record's hashable payload: the deterministic CBOR profile
    encodes sequences as ``list`` only, so the tuple-typed fields are
    list-ified here (the record itself keeps tuples for immutability)."""
    payload = dict(record)
    for key in ("cell", "view_ids", "fragment_ids", "augmentation_ids"):
        payload[key] = list(payload[key])
    return payload


def _record_hash(record: Mapping[str, Any]) -> str:
    return domain_hash(EXPOSURE_RECORD_DOMAIN, _record_payload(record))


def _root_hash(run_id: str, record_count: int, last_record_hash: str | None) -> str:
    return domain_hash(
        EXPOSURE_LEDGER_ROOT_DOMAIN,
        {"run_id": run_id, "record_count": record_count, "last_record_hash": last_record_hash},
    )


class ExposureLedger:
    """The append-only exposure ledger of one S2 run (§44.2).

    The ledger is the durable authority: a completed checkpoint may only
    ever reference a snapshot of it, it can never delete earlier
    exposure. ``verify()`` re-walks the hash chain and is the explicit
    integrity pass — the root hash pins only the last record, so a
    mid-ledger rewrite is caught by the walk, not the root.
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = _require_nonempty_str(run_id, "run_id", exc=ExposureLedgerInvalid)
        self._records: list[ExposureRecord] = []

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def records(self) -> tuple[ExposureRecord, ...]:
        return tuple(self._records)

    @property
    def record_count(self) -> int:
        return len(self._records)

    def append(
        self,
        *,
        attempt_id: str,
        lineage_root_id: str,
        cell: Sequence[str],
        view_ids: Sequence[str],
        fragment_ids: Sequence[str] = (),
        augmentation_ids: Sequence[str] = (),
        sample_count: int,
        optimizer_step: int,
        plan_draw_index: int | None = None,
    ) -> ExposureRecord:
        """Append one exposure record, chaining it to the current last
        record's hash. Refuses (never repairs) any malformed field or a
        record that would leave the ledger's append-only shape."""
        _require_nonempty_str(attempt_id, "attempt_id", exc=ExposureLedgerInvalid)
        if not _ATTEMPT_ID_RE.match(attempt_id):
            raise ExposureLedgerInvalid(
                f"attempt_id must be an 'attempt_<64 hex>' id, got {attempt_id!r}"
            )
        lineage_root_id = _require_nonempty_str(
            lineage_root_id, "lineage_root_id", exc=ExposureLedgerInvalid
        )
        cell_key = _require_cell(cell, "cell", exc=ExposureLedgerInvalid)
        view_ids = _require_id_sequence(view_ids, "view_ids", exc=ExposureLedgerInvalid)
        fragment_ids = _require_id_sequence(
            fragment_ids, "fragment_ids", exc=ExposureLedgerInvalid
        )
        augmentation_ids = _require_id_sequence(
            augmentation_ids, "augmentation_ids", exc=ExposureLedgerInvalid
        )
        sample_count = _require_int(
            sample_count, "sample_count", minimum=1, exc=ExposureLedgerInvalid
        )
        optimizer_step = _require_int(
            optimizer_step, "optimizer_step", minimum=0, exc=ExposureLedgerInvalid
        )
        if plan_draw_index is not None:
            _require_int(plan_draw_index, "plan_draw_index", minimum=0, exc=ExposureLedgerInvalid)

        record_index = len(self._records)
        prev = self._records[-1].record_hash if self._records else None
        record = ExposureRecord(
            record_index=record_index,
            run_id=self._run_id,
            attempt_id=attempt_id,
            lineage_root_id=lineage_root_id,
            cell=cell_key,
            view_ids=view_ids,
            fragment_ids=fragment_ids,
            augmentation_ids=augmentation_ids,
            sample_count=sample_count,
            optimizer_step=optimizer_step,
            plan_draw_index=plan_draw_index,
            prev_record_hash=prev,
            record_hash=_record_hash(
                {
                    "record_index": record_index,
                    "run_id": self._run_id,
                    "attempt_id": attempt_id,
                    "lineage_root_id": lineage_root_id,
                    "cell": cell_key,
                    "view_ids": view_ids,
                    "fragment_ids": fragment_ids,
                    "augmentation_ids": augmentation_ids,
                    "sample_count": sample_count,
                    "optimizer_step": optimizer_step,
                    "plan_draw_index": plan_draw_index,
                    "prev_record_hash": prev,
                }
            ),
        )
        self._records.append(record)
        return record

    def root_hash(self) -> str:
        """The ledger's root: a domain hash over (run_id, record_count,
        last record hash). Pins the state a manifest or release was
        built against."""
        last = self._records[-1].record_hash if self._records else None
        return _root_hash(self._run_id, len(self._records), last)

    def verify(self) -> None:
        """Re-walk the chain and recompute every record hash. Raises
        :class:`ExposureLedgerBroken` on the first inconsistency."""
        prev: str | None = None
        for expected_index, record in enumerate(self._records):
            if record.record_index != expected_index:
                raise ExposureLedgerBroken(
                    f"record at position {expected_index} claims index {record.record_index}"
                )
            if record.run_id != self._run_id:
                raise ExposureLedgerBroken(
                    f"record {expected_index} names run {record.run_id!r}, "
                    f"ledger belongs to {self._run_id!r}"
                )
            if record.prev_record_hash != prev:
                raise ExposureLedgerBroken(
                    f"record {expected_index} breaks the prev_record_hash chain"
                )
            recomputed = _record_hash(
                {
                    "record_index": record.record_index,
                    "run_id": record.run_id,
                    "attempt_id": record.attempt_id,
                    "lineage_root_id": record.lineage_root_id,
                    "cell": record.cell,
                    "view_ids": record.view_ids,
                    "fragment_ids": record.fragment_ids,
                    "augmentation_ids": record.augmentation_ids,
                    "sample_count": record.sample_count,
                    "optimizer_step": record.optimizer_step,
                    "plan_draw_index": record.plan_draw_index,
                    "prev_record_hash": record.prev_record_hash,
                }
            )
            if recomputed != record.record_hash:
                raise ExposureLedgerBroken(
                    f"record {expected_index} record_hash does not verify"
                )
            prev = record.record_hash

    def lineage_exposure(self) -> dict[str, int]:
        """{lineage_root_id: number of records naming it as the drawn
        lineage root}."""
        counts: dict[str, int] = {}
        for record in self._records:
            counts[record.lineage_root_id] = counts.get(record.lineage_root_id, 0) + 1
        return dict(sorted(counts.items()))

    def fragment_ids(self) -> tuple[str, ...]:
        return tuple(sorted({fid for r in self._records for fid in r.fragment_ids}))

    def augmentation_ids(self) -> tuple[str, ...]:
        return tuple(sorted({aid for r in self._records for aid in r.augmentation_ids}))

    def attempt_ids(self) -> frozenset[str]:
        return frozenset(r.attempt_id for r in self._records)

    def save_jsonl(self, path: Path | str) -> None:
        """Write one JSON record per line, in append order, to `path`.

        Same crash-safe discipline as `s1_train.save_checkpoint`: written
        to a `.tmp` sibling first, then `os.replace`d into place, so a
        crash mid-write cannot truncate the previous good ledger file.
        Each line carries the record's own `record_hash`/`prev_record_hash`
        -- :meth:`load_jsonl` treats those as a checksum on the rest of the
        row, not as trusted state, so an edited or reordered file is
        refused rather than silently re-accepted under new hashes."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        lines = "".join(
            json.dumps(asdict(record), sort_keys=True) + "\n" for record in self._records
        )
        temporary.write_text(lines, encoding="utf-8")
        os.replace(temporary, path)

    @classmethod
    def load_jsonl(cls, path: Path | str, run_id: str) -> ExposureLedger:
        """Read a ledger back from `path`, rebuilding it record by record
        through :meth:`append` and then re-verifying the chain
        (:meth:`verify`).

        Every stored `record_hash` is checked against the value
        :meth:`append` recomputes from the row's own fields (index and
        chain position included): an edited field or a reordered line
        changes what gets recomputed and is refused, never repaired. A row
        naming a different `run_id` than the caller's is refused rather
        than silently relabeled."""
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ExposureLedgerInvalid(f"cannot read {path}: {exc}") from exc
        ledger = cls(run_id)
        for line_no, line in enumerate(text.splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExposureLedgerInvalid(
                    f"{path} line {line_no + 1} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ExposureLedgerInvalid(f"{path} line {line_no + 1} must be a JSON object")
            if row.get("run_id") != run_id:
                raise ExposureLedgerInvalid(
                    f"{path} line {line_no + 1} names run {row.get('run_id')!r}, "
                    f"load_jsonl was given run_id {run_id!r}"
                )
            record = ledger.append(
                attempt_id=row["attempt_id"],
                lineage_root_id=row["lineage_root_id"],
                cell=row["cell"],
                view_ids=row["view_ids"],
                fragment_ids=row.get("fragment_ids", ()),
                augmentation_ids=row.get("augmentation_ids", ()),
                sample_count=row["sample_count"],
                optimizer_step=row["optimizer_step"],
                plan_draw_index=row.get("plan_draw_index"),
            )
            # `append` derives `record_index`/`prev_record_hash` from
            # POSITION, never from the row -- a tampered display value in
            # either field would otherwise be silently ignored rather than
            # refused, because it plays no part in what gets recomputed.
            if row.get("record_index") != record.record_index:
                raise ExposureLedgerBroken(
                    f"{path} line {line_no + 1}: stored record_index "
                    f"{row.get('record_index')!r} does not match its position "
                    f"{record.record_index} -- the file was edited or reordered"
                )
            if row.get("prev_record_hash") != record.prev_record_hash:
                raise ExposureLedgerBroken(
                    f"{path} line {line_no + 1}: stored prev_record_hash does not "
                    "match the previous record's actual hash -- the file was "
                    "edited or reordered"
                )
            if row.get("record_hash") != record.record_hash:
                raise ExposureLedgerBroken(
                    f"{path} line {line_no + 1}: stored record_hash does not match "
                    "the value recomputed from its own fields and chain position "
                    "-- the file was edited, reordered, or is not the ledger it "
                    "claims to be"
                )
        ledger.verify()
        return ledger


def validate_run_attempt_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate against the closed §44.3 schema (no self-hash check)."""
    errors = list(_run_attempt_validator().iter_errors(manifest))
    if errors:
        raise RunAttemptManifestInvalid(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )


def verify_run_attempt_manifest(manifest: Mapping[str, Any]) -> None:
    """Schema validation PLUS the self-hash check: a manifest whose
    ``run_attempt_manifest_hash`` does not match its payload was tampered
    with (or never produced by this module) and is refused."""
    validate_run_attempt_manifest(manifest)
    payload = {k: v for k, v in manifest.items() if k != "run_attempt_manifest_hash"}
    if domain_hash(RUN_ATTEMPT_DOMAIN, payload) != manifest["run_attempt_manifest_hash"]:
        raise RunAttemptManifestInvalid(
            "run_attempt_manifest_hash does not verify against the payload"
        )


def validate_release_exposure_manifest(release: Mapping[str, Any]) -> None:
    """Validate against the closed release schema (no self-hash check)."""
    errors = list(_release_validator().iter_errors(release))
    if errors:
        raise ReleaseExposureInvalid(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )


def verify_release_exposure_manifest(release: Mapping[str, Any]) -> None:
    """Schema validation PLUS the self-hash check."""
    validate_release_exposure_manifest(release)
    payload = {k: v for k, v in release.items() if k != "release_exposure_manifest_hash"}
    if domain_hash(RELEASE_EXPOSURE_DOMAIN, payload) != release["release_exposure_manifest_hash"]:
        raise ReleaseExposureInvalid(
            "release_exposure_manifest_hash does not verify against the payload"
        )


def _check_ledger_run(ledger: ExposureLedger, run_id: str, *, exc: type[ValueError]) -> None:
    if not isinstance(ledger, ExposureLedger):
        raise exc(f"ledger must be an ExposureLedger, got {type(ledger).__name__}")
    if ledger.run_id != run_id:
        raise exc(f"ledger belongs to run {ledger.run_id!r}, expected {run_id!r}")


def start_attempt(
    *,
    run_id: str,
    attempt_index: int,
    execution_profile: Mapping[str, str],
    start_integrity: Mapping[str, str],
    ledger: ExposureLedger,
) -> dict[str, Any]:
    """Create the immutable RunAttemptManifest for an execution attempt
    start (§44.3). The document stays ``terminal_state = "running"`` for
    its whole life; the ledger snapshot fields pin the ledger state at
    the moment of start."""
    run_id = _require_nonempty_str(run_id, "run_id", exc=RunAttemptManifestInvalid)
    attempt_index = _require_int(
        attempt_index, "attempt_index", minimum=0, exc=RunAttemptManifestInvalid
    )
    profile = _require_profile(
        execution_profile, "execution_profile", exc=RunAttemptManifestInvalid
    )
    start_integrity = _require_integrity_record(
        start_integrity, "start_integrity_record", exc=RunAttemptManifestInvalid
    )
    _check_ledger_run(ledger, run_id, exc=RunAttemptManifestInvalid)

    attempt_id = make_id(
        "attempt", RUN_ATTEMPT_ID_DOMAIN, {"run_id": run_id, "attempt_index": attempt_index}
    )
    payload: dict[str, Any] = {
        "run_attempt_manifest_schema_version": RUN_ATTEMPT_MANIFEST_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "attempt_index": attempt_index,
        "run_id": run_id,
        "execution_profile_hash": domain_hash(EXECUTION_PROFILE_DOMAIN, profile),
        "terminal_state": "running",
        "exposure_ledger_root_hash": ledger.root_hash(),
        "exposure_record_count": ledger.record_count,
        "start_integrity_record": start_integrity,
        "end_integrity_record": None,
    }
    payload["run_attempt_manifest_hash"] = domain_hash(RUN_ATTEMPT_DOMAIN, payload)
    verify_run_attempt_manifest(payload)
    return payload


def finalize_attempt(
    *,
    manifest: Mapping[str, Any],
    terminal_state: str,
    ledger: ExposureLedger,
    end_integrity: Mapping[str, str],
) -> dict[str, Any]:
    """Create the NEW manifest that ends an attempt in one of
    ``completed | failed | cancelled``. The running document is never
    edited: this returns a new document inheriting the attempt identity
    and start record, re-reading the ledger snapshot so the terminal
    manifest says what the ledger looks like at the END of the attempt.

    Refuses a manifest that is already terminal, whose self-hash does not
    verify, or whose run the ledger does not belong to.
    """
    if terminal_state not in _TERMINAL_STATES:
        raise RunAttemptManifestInvalid(
            f"terminal_state must be one of {_TERMINAL_STATES}, got {terminal_state!r}"
        )
    verify_run_attempt_manifest(manifest)
    if manifest["terminal_state"] != "running":
        raise RunAttemptManifestInvalid(
            f"manifest {manifest['attempt_id']} is already "
            f"{manifest['terminal_state']} — an attempt finalises exactly once"
        )
    end_integrity = _require_integrity_record(
        end_integrity, "end_integrity_record", exc=RunAttemptManifestInvalid
    )
    _check_ledger_run(ledger, manifest["run_id"], exc=RunAttemptManifestInvalid)

    finalised = dict(manifest)
    finalised["terminal_state"] = terminal_state
    finalised["end_integrity_record"] = end_integrity
    finalised["exposure_ledger_root_hash"] = ledger.root_hash()
    finalised["exposure_record_count"] = ledger.record_count
    finalised["run_attempt_manifest_hash"] = domain_hash(
        RUN_ATTEMPT_DOMAIN,
        {k: v for k, v in finalised.items() if k != "run_attempt_manifest_hash"},
    )
    verify_run_attempt_manifest(finalised)
    return finalised


def _ledger_contains_snapshot(ledger: ExposureLedger, release: Mapping[str, Any]) -> None:
    """The release's (record_count, root_hash) must be a PREFIX state of
    the current ledger: the ledger grew past it, never shrunk or
    diverged. A violation means the ledger is not the one the release was
    built from."""
    count = release["exposure_record_count"]
    if count > ledger.record_count:
        raise ReleaseExposureInvalid(
            f"ledger has {ledger.record_count} records but the release snapshots "
            f"{count} — the ledger does not contain the release's snapshot"
        )
    last = ledger.records[count - 1].record_hash if count else None
    expected_root = _root_hash(ledger.run_id, count, last)
    if expected_root != release["exposure_ledger_root_hash"]:
        raise ReleaseExposureInvalid(
            "the ledger's first records do not hash to the release's "
            "exposure_ledger_root_hash — the ledger was rewritten or is a different run"
        )


def create_release(
    *,
    run_id: str,
    attempts: Sequence[Mapping[str, Any]],
    ledger: ExposureLedger,
) -> dict[str, Any]:
    """Derive the ReleaseExposureManifest for a set of attempts entirely
    from the ledger. Every ledger record's attempt id must be covered by
    ``attempts`` (a release may not silently omit exposure); an attempt
    with no ledger records (crashed before any exposure) is still a
    legitimate member. Refuses an empty ledger — a release is a
    statement about recorded exposure and there is nothing to release
    without at least one record."""
    run_id = _require_nonempty_str(run_id, "run_id", exc=ReleaseExposureInvalid)
    _check_ledger_run(ledger, run_id, exc=ReleaseExposureInvalid)
    if not attempts:
        raise ReleaseExposureInvalid("attempts must name at least one RunAttemptManifest")
    if ledger.record_count == 0:
        raise ReleaseExposureInvalid(
            "the ledger has no records — a release is a statement about recorded "
            "exposure and there is nothing to release yet"
        )

    seen: set[str] = set()
    for attempt in attempts:
        verify_run_attempt_manifest(attempt)
        if attempt["run_id"] != run_id:
            raise ReleaseExposureInvalid(
                f"attempt {attempt['attempt_id']} belongs to run {attempt['run_id']!r}, "
                f"expected {run_id!r}"
            )
        if attempt["attempt_id"] in seen:
            raise ReleaseExposureInvalid(
                f"attempt {attempt['attempt_id']} is named twice in attempts"
            )
        seen.add(attempt["attempt_id"])

    uncovered = ledger.attempt_ids() - seen
    if uncovered:
        raise ReleaseExposureInvalid(
            "the ledger recorded exposure for attempt(s) the release omits: "
            f"{sorted(uncovered)} — a release may not drop recorded exposure"
        )

    base: dict[str, Any] = {
        "release_exposure_manifest_schema_version": RELEASE_EXPOSURE_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "attempt_ids": sorted(seen),
        "exposure_ledger_root_hash": ledger.root_hash(),
        "exposure_record_count": ledger.record_count,
        "lineage_exposure": ledger.lineage_exposure(),
        "fragment_ids": list(ledger.fragment_ids()),
        "augmentation_ids": list(ledger.augmentation_ids()),
    }
    release_id = make_id("release", RELEASE_EXPOSURE_ID_DOMAIN, base)
    payload = {**base, "release_id": release_id}
    payload["release_exposure_manifest_hash"] = domain_hash(RELEASE_EXPOSURE_DOMAIN, payload)
    verify_release_exposure_manifest(payload)
    return payload


def merge_release(
    *,
    release: Mapping[str, Any],
    attempt: Mapping[str, Any],
    ledger: ExposureLedger,
) -> dict[str, Any]:
    """Merge one further attempt into an existing release: a MONOTONIC
    union. The ledger must still contain the release's snapshot (prefix
    check), every lineage's exposure count and the record count may only
    grow, the new attempt may not already be in the release, and the run
    must be the same. The new release is re-derived from the current
    ledger, which is the monotone union by construction."""
    verify_release_exposure_manifest(release)
    verify_run_attempt_manifest(attempt)
    if attempt["run_id"] != release["run_id"]:
        raise ReleaseExposureInvalid(
            f"attempt belongs to run {attempt['run_id']!r}, "
            f"release belongs to run {release['run_id']!r}"
        )
    _check_ledger_run(ledger, release["run_id"], exc=ReleaseExposureInvalid)
    if attempt["attempt_id"] in release["attempt_ids"]:
        raise ReleaseExposureInvalid(
            f"attempt {attempt['attempt_id']} is already in the release — "
            "merging it again would double-count its exposure"
        )
    _ledger_contains_snapshot(ledger, release)

    current_counts = ledger.lineage_exposure()
    for lineage, views in release["lineage_exposure"].items():
        if current_counts.get(lineage, 0) < views:
            raise ReleaseExposureInvalid(
                f"lineage {lineage} exposure decreased from {views} to "
                f"{current_counts.get(lineage, 0)} — the ledger no longer contains "
                "the release's exposure"
            )
    if set(release["fragment_ids"]) - set(ledger.fragment_ids()):
        raise ReleaseExposureInvalid(
            "the ledger no longer contains a fragment id the release recorded"
        )
    if set(release["augmentation_ids"]) - set(ledger.augmentation_ids()):
        raise ReleaseExposureInvalid(
            "the ledger no longer contains an augmentation id the release recorded"
        )
    uncovered = ledger.attempt_ids() - (set(release["attempt_ids"]) | {attempt["attempt_id"]})
    if uncovered:
        raise ReleaseExposureInvalid(
            "the ledger recorded exposure for attempt(s) neither the release nor "
            f"the merged attempt name: {sorted(uncovered)}"
        )

    base = {
        "release_exposure_manifest_schema_version": RELEASE_EXPOSURE_MANIFEST_SCHEMA_VERSION,
        "run_id": release["run_id"],
        "attempt_ids": sorted(set(release["attempt_ids"]) | {attempt["attempt_id"]}),
        "exposure_ledger_root_hash": ledger.root_hash(),
        "exposure_record_count": ledger.record_count,
        "lineage_exposure": ledger.lineage_exposure(),
        "fragment_ids": list(ledger.fragment_ids()),
        "augmentation_ids": list(ledger.augmentation_ids()),
    }
    release_id = make_id("release", RELEASE_EXPOSURE_ID_DOMAIN, base)
    payload = {**base, "release_id": release_id}
    payload["release_exposure_manifest_hash"] = domain_hash(RELEASE_EXPOSURE_DOMAIN, payload)
    verify_release_exposure_manifest(payload)
    return payload


def load_run_attempt_manifest(path: Path | str) -> dict[str, Any]:
    """Read a frozen RunAttemptManifest from disk, verified on the way in
    (schema + self-hash)."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise RunAttemptManifestInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RunAttemptManifestInvalid(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RunAttemptManifestInvalid(f"{path} must hold a JSON object")
    verify_run_attempt_manifest(payload)
    return payload


def load_release_exposure_manifest(path: Path | str) -> dict[str, Any]:
    """Read a frozen ReleaseExposureManifest from disk, verified on the
    way in (schema + self-hash)."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise ReleaseExposureInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseExposureInvalid(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ReleaseExposureInvalid(f"{path} must hold a JSON object")
    verify_release_exposure_manifest(payload)
    return payload

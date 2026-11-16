"""S4.5 exposure ledger (RFC005-ERRATA-010 E3.1/E7.1/E7.5; dormant).

An own hash-chained ledger rather than a subclass or a widened
``s2_exposure.ExposureRecord`` (which has no rollout fields and whose payload
is pinned): it composes the S2 ``_require_*`` field validators and nothing
else. One record = one lineage-root input with its views, the number of
rollouts drawn for it and a digest of their seeds.

Admission is re-applied on every ``append``, ``verify`` and ``from_document``:
a record is refused if its split is not ``train``, its source or cohort is one
of the dataset manifest's ``forbidden_sources``, its source is not a manifest
source, or its lineage is one of the manifest's ``exclude_lineage_ids``
(calibration / dev / S5-control) or not in ``lineage_ids``. A forged record
re-hashed into the chain therefore still fails ``verify()``.

``guard_intervention_counts`` holds, per rollout, how many sampled tokens the
loop guard substituted (``None`` = not measured, hashed as null, never 0): the
update path does not cover those positions (D2), so the dependence is visible.

*Double exposure* means the same view id drawn twice within one epoch; a later
epoch may legitimately re-draw it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Final

import jsonschema.validators

from unflatten.model.hashing import domain_hash
from unflatten.model.rl.ids import EXPOSURE_LEDGER_DOMAIN, ROLLOUT_SEED_DOMAIN
from unflatten.model.rl.policies import MANIFEST_HASH_FIELD, verify_dataset_manifest
from unflatten.model.rl.rollout import derive_rollout_seed
from unflatten.model.s2_exposure import (
    _require_cell,
    _require_id_sequence,
    _require_int,
    _require_nonempty_str,
)

SCHEMA_VERSION: Final = "1.0.0"
TRAIN_SPLIT: Final = "train"
_SCHEMA_PATH: Final = Path(__file__).parent / "schemas" / "rl_exposure_ledger.schema.json"
_DIGEST_PREFIX: Final = "sha256:"


class RlExposureInvalid(ValueError):
    """A malformed field or an append that would leave the ledger's shape."""


class RlExposureForbidden(RlExposureInvalid):
    """A forbidden source/split/lineage tried to enter (or is in) the exposure."""


class RlDoubleExposure(RlExposureInvalid):
    """The same view was drawn twice within one epoch."""


class RlExposureBroken(ValueError):
    """Re-walking the chain or the document hash failed."""


@dataclass(frozen=True, slots=True)
class RlExposureRecord:
    record_index: int
    run_id: str
    lineage_root_id: str
    source: str
    cell: tuple[str, str, str, str]
    epoch: int
    view_ids: tuple[str, ...]
    rollout_count: int
    guard_intervention_counts: tuple[int | None, ...]
    rollout_seed_digest: str
    reward_policy_hash: str
    optimizer_step: int
    prev_record_hash: str | None
    record_hash: str


_HASHED = (
    "record_index run_id lineage_root_id source cell epoch view_ids rollout_count "
    "guard_intervention_counts rollout_seed_digest reward_policy_hash optimizer_step "
    "prev_record_hash"
).split()


def _payload(record: Any) -> dict[str, Any]:
    out = {k: (record[k] if isinstance(record, Mapping) else getattr(record, k)) for k in _HASHED}
    out["cell"] = list(out["cell"])
    out["view_ids"] = list(out["view_ids"])
    out["guard_intervention_counts"] = list(out["guard_intervention_counts"])
    return out


def _require_guard_counts(value: Any, rollout_count: int) -> tuple[int | None, ...]:
    """One entry per rollout: a count (>= 0) or ``None`` (not measured).

    ``None`` is stored as ``None`` and hashed as such; it is never coerced to 0
    (an unmeasured substitution count is not "no interventions").
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RlExposureInvalid(f"guard_intervention_counts must be a sequence, got {value!r}")
    if len(value) != rollout_count:
        raise RlExposureInvalid(
            f"guard_intervention_counts has {len(value)} entries for rollout_count {rollout_count}"
        )
    for i, v in enumerate(value):
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
            raise RlExposureInvalid(
                f"guard_intervention_counts[{i}] must be an int >= 0 or None, got {v!r}"
            )
    return tuple(value)


def _record_hash(record: Any) -> str:
    return domain_hash(EXPOSURE_LEDGER_DOMAIN, {"kind": "record", **_payload(record)})


def rollout_seed_digest(run_seed: int, doc_id: str, rollout_count: int) -> str:
    """Digest of the ``rollout_count`` seeds ``H(run_seed, doc_id, i)`` the sampler uses."""
    seeds = [derive_rollout_seed(run_seed, doc_id, i) for i in range(rollout_count)]
    return domain_hash(
        ROLLOUT_SEED_DOMAIN,
        {"kind": "seed_digest", "run_seed": run_seed, "doc_id": doc_id, "seeds": seeds},
    )


@cache
def _validator() -> Any:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


class RlExposureLedger:
    def __init__(
        self, run_id: str, *, dataset_manifest: Mapping[str, Any], reward_policy_hash: str
    ) -> None:
        self._run_id = _require_nonempty_str(run_id, "run_id", exc=RlExposureInvalid)
        verify_dataset_manifest(dataset_manifest)
        self._manifest = dataset_manifest
        self._reward_policy_hash = _require_nonempty_str(
            reward_policy_hash, "reward_policy_hash", exc=RlExposureInvalid
        )
        self._records: list[RlExposureRecord] = []

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def records(self) -> tuple[RlExposureRecord, ...]:
        return tuple(self._records)

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def dataset_manifest_hash(self) -> str:
        return str(self._manifest[MANIFEST_HASH_FIELD])

    @property
    def reward_policy_hash(self) -> str:
        return self._reward_policy_hash

    def _admit(self, *, lineage: str, source: str, cell: Sequence[str]) -> None:
        m = self._manifest
        forbidden = set(m["forbidden_sources"])
        if cell[0] != TRAIN_SPLIT:
            raise RlExposureForbidden(f"split {cell[0]!r} refused: S4.5 draws only 'train'")
        for label in (source, cell[3]):
            if label in forbidden:
                raise RlExposureForbidden(f"forbidden source {label!r} may not be exposed")
        if source not in m["sources"]:
            raise RlExposureForbidden(f"source {source!r} is not in the dataset manifest sources")
        if lineage in set(m["exclude_lineage_ids"]):
            raise RlExposureForbidden(
                f"lineage {lineage!r} is a calibration/dev/control lineage of the manifest"
            )
        if lineage not in set(m["lineage_ids"]):
            raise RlExposureForbidden(f"lineage {lineage!r} is not in the dataset manifest")

    def _seen_views(self) -> set[tuple[int, str]]:
        return {(r.epoch, v) for r in self._records for v in r.view_ids}

    def append(
        self,
        *,
        lineage_root_id: str,
        source: str,
        cell: Sequence[str],
        epoch: int,
        view_ids: Sequence[str],
        rollout_count: int,
        guard_intervention_counts: Sequence[int | None],
        rollout_seed_digest: str,
        optimizer_step: int,
    ) -> RlExposureRecord:
        exc = RlExposureInvalid
        lineage = _require_nonempty_str(lineage_root_id, "lineage_root_id", exc=exc)
        src = _require_nonempty_str(source, "source", exc=exc)
        cell_key = _require_cell(cell, "cell", exc=exc)
        epoch = _require_int(epoch, "epoch", minimum=0, exc=exc)
        views = _require_id_sequence(view_ids, "view_ids", exc=exc)
        if not views:
            raise exc("view_ids must not be empty")
        count = _require_int(rollout_count, "rollout_count", minimum=1, exc=exc)
        guard = _require_guard_counts(guard_intervention_counts, count)
        step = _require_int(optimizer_step, "optimizer_step", minimum=0, exc=exc)
        digest = _require_nonempty_str(rollout_seed_digest, "rollout_seed_digest", exc=exc)
        if not digest.startswith(_DIGEST_PREFIX) or len(digest) != len(_DIGEST_PREFIX) + 64:
            raise exc(f"rollout_seed_digest must be 'sha256:<64 hex>', got {digest!r}")
        self._admit(lineage=lineage, source=src, cell=cell_key)
        repeated = sorted(v for v in views if (epoch, v) in self._seen_views())
        if repeated:
            raise RlDoubleExposure(f"view(s) {repeated} already exposed in epoch {epoch}")
        body = {
            "record_index": len(self._records),
            "run_id": self._run_id,
            "lineage_root_id": lineage,
            "source": src,
            "cell": cell_key,
            "epoch": epoch,
            "view_ids": views,
            "rollout_count": count,
            "guard_intervention_counts": guard,
            "rollout_seed_digest": digest,
            "reward_policy_hash": self._reward_policy_hash,
            "optimizer_step": step,
            "prev_record_hash": self._records[-1].record_hash if self._records else None,
        }
        record = RlExposureRecord(**body, record_hash=_record_hash(body))
        self._records.append(record)
        return record

    def verify(self) -> None:
        prev: str | None = None
        seen: set[tuple[int, str]] = set()
        for index, r in enumerate(self._records):
            if r.record_index != index or r.run_id != self._run_id:
                raise RlExposureBroken(f"record {index}: index/run id out of sequence")
            if r.prev_record_hash != prev:
                raise RlExposureBroken(f"record {index} breaks the prev_record_hash chain")
            if r.reward_policy_hash != self._reward_policy_hash:
                raise RlExposureBroken(f"record {index}: reward_policy_hash differs from the run's")
            if _record_hash(r) != r.record_hash:
                raise RlExposureBroken(f"record {index} record_hash does not verify")
            self._admit(lineage=r.lineage_root_id, source=r.source, cell=r.cell)
            for view in r.view_ids:
                if (r.epoch, view) in seen:
                    raise RlDoubleExposure(f"view {view!r} exposed twice in epoch {r.epoch}")
                seen.add((r.epoch, view))
            prev = r.record_hash

    def lineage_exposure(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self._records:
            counts[r.lineage_root_id] = counts.get(r.lineage_root_id, 0) + 1
        return dict(sorted(counts.items()))

    def lineage_ids(self) -> frozenset[str]:
        return frozenset(r.lineage_root_id for r in self._records)

    def to_document(self) -> dict[str, Any]:
        rows = [{**_payload(r), "record_hash": r.record_hash} for r in self._records]
        body = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._run_id,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "reward_policy_hash": self._reward_policy_hash,
            "record_count": len(rows),
            "last_record_hash": rows[-1]["record_hash"] if rows else None,
            "records": rows,
        }
        return {
            **body,
            "exposure_ledger_hash": domain_hash(EXPOSURE_LEDGER_DOMAIN, {"kind": "ledger", **body}),
        }

    @classmethod
    def from_document(
        cls, doc: Mapping[str, Any], *, dataset_manifest: Mapping[str, Any]
    ) -> RlExposureLedger:
        errors = sorted(_validator().iter_errors(doc), key=lambda e: list(e.absolute_path))
        if errors:
            where = "/" + "/".join(str(p) for p in errors[0].absolute_path)
            raise RlExposureBroken(f"exposure ledger invalid at {where}: {errors[0].message}")
        verify_dataset_manifest(dataset_manifest)
        if doc["dataset_manifest_hash"] != dataset_manifest[MANIFEST_HASH_FIELD]:
            raise RlExposureInvalid(
                f"dataset_manifest_hash {doc['dataset_manifest_hash']!r} does not match the "
                f"presented dataset manifest {dataset_manifest[MANIFEST_HASH_FIELD]!r}"
            )
        ledger = cls(
            doc["run_id"],
            dataset_manifest=dataset_manifest,
            reward_policy_hash=doc["reward_policy_hash"],
        )
        for row in doc["records"]:
            rebuilt = ledger.append(
                lineage_root_id=row["lineage_root_id"],
                source=row["source"],
                cell=row["cell"],
                epoch=row["epoch"],
                view_ids=row["view_ids"],
                rollout_count=row["rollout_count"],
                guard_intervention_counts=row["guard_intervention_counts"],
                rollout_seed_digest=row["rollout_seed_digest"],
                optimizer_step=row["optimizer_step"],
            )
            if {**_payload(rebuilt), "record_hash": rebuilt.record_hash} != dict(row):
                raise RlExposureBroken(
                    f"record {rebuilt.record_index}: stored fields/record_hash do not match "
                    "the chain re-derived from its own content"
                )
        ledger.verify()
        if ledger.to_document() != dict(doc):
            raise RlExposureBroken(
                "record_count / last_record_hash / exposure_ledger_hash do not match the "
                "re-derived ledger"
            )
        return ledger

"""S4.5 checkpoint I/O and child ancestry (RFC005-ERRATA-010 E7.1, design D4; dormant).

Writes and reads an RL checkpoint through the public ``s1_train`` surface and
makes it acceptable as a *parent* of the next stage (S5 / G1P / S6) through
``checkpoint_ancestry.build_base_ancestry(parent=...)``.

Slot semantics (D4). The ancestry sidecar slot ``checkpoint.s1_run_manifest_hash``
carries the RL run manifest's ``s1_run_manifest_hash`` -- by construction equal
to its own ``run_manifest_hash``. The manifest is the "companion" that
``_parent_entry`` asks a parent to present (top-level ``run_class`` plus a
``s1_run_manifest_hash`` equal to the slot). It is not an S1-schema document, so
nothing here uses S1 companion verification.

Atomicity. ``s1_train.save_checkpoint`` writes the shared payload (model,
optimizer, step, RNG) to a *staging* path with no ancestry. This module adds the
RL keys (``typed_head_config_hash``, ``visual_feature_tap``,
``rl_run_manifest_hash``) and the embedded ``ancestry_hash`` to a ``.tmp``
sibling and lands it with a single ``os.replace``; the sidecar is written only
afterwards. S4's stamp-after-write second write is deliberately not reproduced:
the final path never holds an unstamped checkpoint. A crash after the replace
and before the sidecar leaves a checkpoint with no (or a stale) sidecar, which
``verify_sidecar`` refuses -- fail closed.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import torch

from unflatten.model import checkpoint_ancestry as ca
from unflatten.model import s1_train
from unflatten.model.hashing import domain_hash
from unflatten.model.rl.exposure import RlExposureBroken, RlExposureLedger
from unflatten.model.rl.ids import RUN_MANIFEST_DOMAIN

_SLOT: Final = "s1_run_manifest_hash"
_HASH: Final = "run_manifest_hash"
_SELF_FIELDS: Final = (_SLOT, _HASH)
_RL_MANIFEST_KEY: Final = "rl_run_manifest_hash"
_CACHE_KEY: Final = "identity_hash"


class RlCheckpointRefused(ValueError):
    """A checkpoint, its sidecar, manifest or ledger cannot be used for this run."""


def _require_manifest(manifest: Mapping[str, Any]) -> str:
    """Return the manifest hash after the cheap self-consistency checks.

    The caller has already run ``verify_rl_run_manifest`` against the real
    artefacts; this only refuses a manifest whose slot / self hash / class are
    missing or disagree, so a stale or hand-edited dict cannot reach a sidecar.
    """
    for key in (
        *_SELF_FIELDS,
        "run_class",
        "run_id",
        "cache_identity",
        "typed_head_config_hash",
        "visual_feature_tap",
    ):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise RlCheckpointRefused(f"run manifest has no usable {key!r}")
    body = {k: v for k, v in manifest.items() if k not in _SELF_FIELDS}
    derived = domain_hash(RUN_MANIFEST_DOMAIN, body)
    if not (manifest[_SLOT] == manifest[_HASH] == derived):
        raise RlCheckpointRefused(
            f"run manifest slot {manifest[_SLOT]!r} / hash {manifest[_HASH]!r} / "
            f"re-derived {derived!r} disagree"
        )
    return str(manifest[_SLOT])


def cache_identity_mapping(manifest: Mapping[str, Any]) -> dict[str, str]:
    """The ``cache_identity`` payload mapping for this manifest (s1 stores a mapping)."""
    return {_CACHE_KEY: str(manifest["cache_identity"])}


def save_rl_checkpoint(
    path: Path | str,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    progress: s1_train.TrainingProgress,
    run_manifest: Mapping[str, Any],
    feature_source: str,
    ancestry_base: Mapping[str, Any] | None = None,
) -> None:
    """Write an RL checkpoint at ``path`` atomically (see module docstring).

    ``run_manifest`` is a verified RL run manifest 1.1.0. ``ancestry_base`` is a
    verified ``document_kind: "base"`` ancestry; ``None`` writes no sidecar and
    embeds ``ancestry_hash: None`` (diagnostic, as in s1).
    """
    manifest_hash = _require_manifest(run_manifest)
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(final.name + ".stage")
    temporary = final.with_name(final.name + ".tmp")
    doc = None
    if ancestry_base is not None:
        doc = ca.extend_for_checkpoint(
            dict(ancestry_base),
            step=progress.step,
            run_id=str(run_manifest["run_id"]),
            s1_run_manifest_hash=manifest_hash,
        )
    try:
        s1_train.save_checkpoint(
            staging,
            model=model,
            optimizer=optimizer,
            progress=progress,
            cache_identity=cache_identity_mapping(run_manifest),
            s1_run_manifest_hash=manifest_hash,
            feature_source=feature_source,
            ancestry_base=None,
            run_id=None,
        )
        state = torch.load(staging, map_location="cpu", weights_only=False)
        state["ancestry_hash"] = doc["ancestry_hash"] if doc is not None else None
        state["typed_head_config_hash"] = run_manifest["typed_head_config_hash"]
        state["visual_feature_tap"] = run_manifest["visual_feature_tap"]
        state[_RL_MANIFEST_KEY] = manifest_hash
        torch.save(state, temporary)
        os.replace(temporary, final)
    finally:
        for leftover in (staging, staging.with_name(staging.name + ".tmp"), temporary):
            leftover.unlink(missing_ok=True)
    if doc is not None:
        doc = ca.finalize_checkpoint_document(doc, final)
        ca.write_sidecar(doc, final)


def _peek(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def load_rl_checkpoint(
    path: Path | str,
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    run_manifest: Mapping[str, Any],
    expect_base_ancestry_hash: str | None,
    expect_feature_source: str,
    restore_rng: bool = True,
) -> tuple[s1_train.TrainingProgress, dict[str, Any]]:
    """Resume from an RL checkpoint, refusing anything that is not this run.

    Every expectation (manifest hash == slot, cache identity, feature source,
    typed-head config, visual tap, sidecar + base ancestry) is checked on the
    state before ``model``/``optimizer``/RNG are touched. ``None`` for
    ``expect_base_ancestry_hash`` is refused: an RL resume is always
    gradient-bearing.
    """
    manifest_hash = _require_manifest(run_manifest)
    if expect_base_ancestry_hash is None:
        raise RlCheckpointRefused("an RL resume needs expect_base_ancestry_hash")
    path = Path(path)
    state = _peek(path)
    for key, want in (
        (_RL_MANIFEST_KEY, manifest_hash),
        ("typed_head_config_hash", run_manifest["typed_head_config_hash"]),
        ("visual_feature_tap", run_manifest["visual_feature_tap"]),
    ):
        if state.get(key) != want:
            raise RlCheckpointRefused(f"checkpoint {key} {state.get(key)!r} != expected {want!r}")
    try:
        return s1_train.load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expect_s1_run_manifest_hash=manifest_hash,
            expect_base_ancestry_hash=expect_base_ancestry_hash,
            expect_cache_identity=cache_identity_mapping(run_manifest),
            expect_feature_source=expect_feature_source,
            restore_rng=restore_rng,
        )
    except s1_train.S1TrainingRefused as exc:
        raise RlCheckpointRefused(f"RL checkpoint resume refused: {exc}") from exc


def require_resume_ledger_consistent(
    ledger: RlExposureLedger, step: int, *, records_per_update: int
) -> None:
    """Resume rule: a checkpoint at update step ``k`` pairs with a ledger that
    holds exactly ``k * records_per_update`` records.

    One ledger record is one lineage-root input (``RlExposureLedger.append``);
    ``records_per_update`` = ``accumulation_steps`` x inputs drawn per
    micro-batch. Update ``j`` (0-based, ``record.optimizer_step == j``) must
    hold exactly ``records_per_update`` records for every ``j < k`` and none for
    ``j >= k``; so a gap (a missing update), an extra uncheckpointed update and
    a count that merely adds up are all refused. The chain is re-walked first.
    """
    if records_per_update < 1 or step < 0:
        raise RlCheckpointRefused(
            f"step {step} / records_per_update {records_per_update} must be >= 0 / >= 1"
        )
    try:
        ledger.verify()
    except RlExposureBroken as exc:
        raise RlCheckpointRefused(f"resume ledger chain is broken: {exc}") from exc
    expected = step * records_per_update
    if ledger.record_count != expected:
        raise RlCheckpointRefused(
            f"checkpoint at step {step} expects {expected} ledger records "
            f"({records_per_update} per update), ledger holds {ledger.record_count}"
        )
    per_update = Counter(r.optimizer_step for r in ledger.records)
    wanted = dict.fromkeys(range(step), records_per_update)
    if dict(per_update) != wanted:
        raise RlCheckpointRefused(
            "ledger optimizer_step histogram "
            f"{dict(sorted(per_update.items()))} != one block of {records_per_update} "
            f"per update 0..{step - 1}: an update is missing or duplicated"
        )


def build_child_ancestry(
    checkpoint_path: Path | str,
    *,
    run_manifest: Mapping[str, Any],
    expect_base_ancestry_hash: str,
    **base_kwargs: Any,
) -> dict[str, Any]:
    """The ``document_kind: "base"`` ancestry of a downstream stage warm-started
    from this RL checkpoint (the S5 / G1P / S6 shape).

    The sidecar is verified against the file bytes, the embedded hash, the
    manifest's slot and ``expect_base_ancestry_hash`` first; then
    ``build_base_ancestry(parent={"sidecar", "companion": run_manifest})`` does
    the rest (companion slot == sidecar slot, DEC-006 exposure subset).
    ``base_kwargs`` are ``build_base_ancestry``'s other arguments, unchanged.
    """
    manifest_hash = _require_manifest(run_manifest)
    path = Path(checkpoint_path)
    state = _peek(path)
    try:
        sidecar = ca.verify_sidecar(
            path,
            expect_base_ancestry_hash=expect_base_ancestry_hash,
            expect_s1_run_manifest_hash=manifest_hash,
            embedded_ancestry_hash=state.get("ancestry_hash"),
        )
        return ca.build_base_ancestry(
            parent={"sidecar": sidecar, "companion": dict(run_manifest)}, **base_kwargs
        )
    except ca.CheckpointAncestryInvalid as exc:
        raise RlCheckpointRefused(f"RL checkpoint refused as a parent: {exc}") from exc

"""DEC-083 G: full-state continuation across the named augmentation repair.

The parent's checkpoint and ledger remain byte-identical. A new child starts
its own ledger at the parent's global step, restoring the original checkpoint
through the existing strict loader, including optimizer and global RNG state.
"""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.model import checkpoint_ancestry as ca
from unflatten.model import s2_exposure as ex
from unflatten.model import s2_run_manifest as s2m
from unflatten.model import s4_run_manifest as sm
from unflatten.model.hashing import domain_hash, make_id

PARENT_DIR = "continuation-parent"
KIND = "augmentation-policy-repair.v1"
KIND_EXTERNAL_VIEWS = "external-views.v1"
KINDS = (KIND, KIND_EXTERNAL_VIEWS)
RATIFICATION_FILE = "ratification.json"
RATIFICATION_DECISION_ID = "DEC-085"
# DEC-086: owner decision for swapping the external-views table of a running external-views child.
RATIFICATION_DECISION_IDS = (RATIFICATION_DECISION_ID, "DEC-086")
OLD_POLICY = "sha256:a3bbc5571c5b13c206f837c9e48dd1cf7eea4633840f7139493a094a0a69a00f"
NEW_POLICY = "sha256:8537b3a69da7cebfca4f423ca886d54f0430a4dff693efd7e0258c42572502f8"
UNCHANGED_FILES = (
    "s1_run_manifest.json",
    "run_config.json",
    "architecture-manifest.json",
    "ordered_exposure_plan.json",
    "s4_page_uniform_plan.json",
    "stage_data_matrix.json",
    "s4_sampler_policy.effective.json",
    "s4_exposure_policy.effective.json",
    "typed_supervision_census.json",
    "exact_text_cache_expectation.json",
    "qwen_text_allowlist.json",
    "s4_eval_pages.json",
    "val-expectation-input.json",
)


class ContinuationRefused(ValueError):
    """The explicit continuation is incomplete or changes another training input."""


_GENERIC_POPPED = (
    "run_id",
    "s4_run_manifest_hash",
    "s4_run_manifest_schema_version",
    "init_from",
    "state_continuation",
)


def is_external_views_child(manifest: Mapping[str, Any]) -> bool:
    """True for a manifest that is itself an external-views.v1 continuation child."""
    return (
        (manifest.get("state_continuation") or {}).get("kind") == KIND_EXTERNAL_VIEWS
        and manifest.get("init_from") is not None
        and "external_views" in manifest["visual_input"]
    )


def parent_ledger_start(parent: Mapping[str, Any]) -> int:
    """Global step at which the parent's own ledger begins (0 for the original clean run)."""
    return parent["init_from"]["step"] if is_external_views_child(parent) else 0


def validate_transition(
    parent: Mapping[str, Any], child: Mapping[str, Any], kind: str | None = None
) -> None:
    """`kind`: the child's bound state_continuation.kind, else the argument, else KIND."""
    bound = (child.get("state_continuation") or {}).get("kind")
    if kind is not None and bound is not None and kind != bound:
        raise ContinuationRefused(f"requested kind {kind!r} contradicts the bound kind {bound!r}")
    kind = bound or kind or KIND
    if kind not in KINDS:
        raise ContinuationRefused(f"unsupported continuation kind {kind!r}")
    if is_external_views_child(parent):
        if kind != KIND_EXTERNAL_VIEWS:
            raise ContinuationRefused("only external-views.v1 may continue an external-views child")
    elif "state_continuation" in parent or parent.get("init_from") is not None:
        raise ContinuationRefused(
            "the parent must be the original clean S4 run or an external-views.v1 child"
        )
    pin = child.get("init_from") or {}
    if (
        parent.get("run_class") != "promotion"
        or parent.get("stage_id") != "S4"
        or pin.get("parent_run_id") != parent["run_id"]
        or pin.get("parent_run_class") != "promotion"
        or pin.get("parent_s1_run_manifest_hash") != parent["binding_s1_run_manifest_hash"]
        or type(pin.get("step")) is not int
        or pin["step"] < 1
    ):
        raise ContinuationRefused("continuation needs the verified original S4 promotion parent")
    if kind == KIND_EXTERNAL_VIEWS:
        _validate_external_views(parent, child)
    else:
        _validate_augmentation_repair(parent, child)


def _validate_augmentation_repair(parent: Mapping[str, Any], child: Mapping[str, Any]) -> None:
    if "external_views" in parent["visual_input"] or "external_views" in child["visual_input"]:
        raise ContinuationRefused("external_views belongs to the external-views.v1 continuation")
    old_aug = parent["visual_input"]["augmentation"]
    new_aug = child["visual_input"]["augmentation"]
    expected = {**old_aug, "policy_id": "augmentation-v2-readable-retry", "policy_hash": NEW_POLICY}
    if (
        old_aug.get("policy_id") != "augmentation-v2"
        or old_aug.get("policy_hash") != OLD_POLICY
        or new_aug != expected
        or old_aug.get("selection") != [2, 5]
    ):
        raise ContinuationRefused("only the named augmentation profile repair is admitted")
    old, new = copy.deepcopy(dict(parent)), copy.deepcopy(dict(child))
    for doc in (old, new):
        for key in _GENERIC_POPPED:
            doc.pop(key, None)
        doc["visual_input"].pop("augmentation")
    if old != new:
        raise ContinuationRefused(
            "only augmentation may change; data, LR, losses and selection stay frozen"
        )


def _validate_external_views(parent: Mapping[str, Any], child: Mapping[str, Any]) -> None:
    old_vi, new_vi = parent["visual_input"], child["visual_input"]
    swap = is_external_views_child(parent)
    if "external_views" in old_vi and not swap:
        raise ContinuationRefused("the parent already carries external_views")
    if "external_views" not in new_vi:
        raise ContinuationRefused(
            "an external-views.v1 child must carry visual_input.external_views"
        )
    if (
        swap
        and new_vi["external_views"]["table_sha256"] == old_vi["external_views"]["table_sha256"]
    ):
        raise ContinuationRefused("an external-views swap must name a different table_sha256")
    if new_vi["augmentation"] != old_vi["augmentation"]:
        raise ContinuationRefused("external-views.v1 leaves augmentation byte-equal")
    old, new = copy.deepcopy(dict(parent)), copy.deepcopy(dict(child))
    for doc in (old, new):
        for key in _GENERIC_POPPED:
            doc.pop(key, None)
    new["visual_input"].pop("external_views")
    old["visual_input"].pop("external_views", None)
    if old != new:
        raise ContinuationRefused(
            "only visual_input.external_views may be added; data, LR, losses and selection "
            "stay frozen"
        )


def require_mode(
    manifest: Mapping[str, Any],
    *,
    resume: Path | None,
    init_from: Path | None,
    eval_only: bool = False,
) -> None:
    if (
        "state_continuation" in manifest
        and not eval_only
        and (resume is None or init_from is not None)
    ):
        raise ContinuationRefused(
            "state_continuation requires --resume, never a weights-only --init-from"
        )


def require_ledger(
    *, ledger: ex.ExposureLedger, step: int, batch_size: int, start_step: int
) -> None:
    expected = (step - start_step) * batch_size
    if step < start_step or ledger.record_count != expected:
        raise ContinuationRefused(
            f"checkpoint step {step}, parent offset {start_step}: expecting {expected} "
            f"new ledger records, ledger holds {ledger.record_count}"
        )
    if start_step:
        if batch_size != 1:
            raise ContinuationRefused("augmentation continuation requires batch_size=1")
        for index, record in enumerate(ledger.records, start=start_step):
            if record.plan_draw_index != index or record.optimizer_step != index + 1:
                raise ContinuationRefused("child ledger reset, skipped or duplicated a global draw")
    elif expected and ledger.records[-1].plan_draw_index != expected - 1:
        raise ContinuationRefused("resumed ledger's last global plan_draw_index disagrees")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def bind_manifest(
    *,
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    parent_ledger: Path,
    decision: Path,
    kind: str = KIND,
) -> dict[str, Any]:
    """Finish a freshly built child with a hash-bound, explicit history boundary."""
    sm.verify_s4_run_manifest(parent)
    sm.verify_s4_run_manifest(child)
    validate_transition(parent, child, kind)
    ledger = ex.ExposureLedger.load_jsonl(parent_ledger, parent["run_id"])
    require_ledger(
        ledger=ledger,
        step=child["init_from"]["step"],
        batch_size=1,
        start_step=parent_ledger_start(parent),
    )
    result = copy.deepcopy(dict(child))
    result["s4_run_manifest_schema_version"] = "1.6.0"
    result["state_continuation"] = {
        "kind": kind,
        "parent_s4_run_manifest_hash": parent["s4_run_manifest_hash"],
        "parent_ledger_sha256": ca.file_sha256(parent_ledger),
        "parent_ledger_root_hash": ledger.root_hash(),
        "decision_sha256": ca.file_sha256(decision),
    }
    base = {k: v for k, v in result.items() if k not in {"run_id", "s4_run_manifest_hash"}}
    result["run_id"] = make_id(sm.RUN_ID_PREFIX, sm.RUN_ID_DOMAIN, base)
    result["s4_run_manifest_hash"] = domain_hash(
        sm.S4_RUN_MANIFEST_DOMAIN, {**base, "run_id": result["run_id"]}
    )
    sm.verify_s4_run_manifest(result)
    return result


def _require_ratification(path: Path, manifest: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise ContinuationRefused(f"external-views.v1 needs {RATIFICATION_FILE} in {PARENT_DIR}/")
    doc = _json(path)
    if not isinstance(doc, dict):
        raise ContinuationRefused(f"{RATIFICATION_FILE} must be an object")
    if (
        doc.get("decision_id") not in RATIFICATION_DECISION_IDS
        or doc.get("kind") != KIND_EXTERNAL_VIEWS
        or doc.get("ratified") is not True
        or not doc.get("ratified_by")
        or not doc.get("ratified_at")
    ):
        raise ContinuationRefused(
            f"{RATIFICATION_FILE} is not a ratified {'/'.join(RATIFICATION_DECISION_IDS)} "
            f"{KIND_EXTERNAL_VIEWS} record"
        )
    if (
        doc.get("external_views_table_sha256")
        != manifest["visual_input"]["external_views"]["table_sha256"]
    ):
        raise ContinuationRefused("ratification names a different external-views table")


@dataclass(frozen=True)
class Context:
    step: int
    parent_checkpoint_sha256: str
    parent_s1_hash: str
    parent_base_hash: str
    parent_ancestry_hash: str


def load_context(run_dir: Path, manifest: Mapping[str, Any]) -> Context | None:
    """Verify pinned original artefacts before any model or GPU is constructed."""
    block = manifest.get("state_continuation")
    if block is None:
        return None
    root = run_dir / PARENT_DIR
    parent = sm.load_s4_run_manifest(root / "s4_run_manifest.json")
    if manifest["s4_run_manifest_schema_version"] != "1.6.0" or block["kind"] not in KINDS:
        raise ContinuationRefused("unsupported continuation kind/schema")
    validate_transition(parent, manifest, block["kind"])
    if block["kind"] == KIND_EXTERNAL_VIEWS:
        _require_ratification(root / RATIFICATION_FILE, manifest)
    for key, actual in (
        ("parent_s4_run_manifest_hash", parent["s4_run_manifest_hash"]),
        ("parent_ledger_sha256", ca.file_sha256(root / "exposure_ledger.jsonl")),
        ("decision_sha256", ca.file_sha256(root / "decision.md")),
    ):
        if block[key] != actual:
            raise ContinuationRefused(f"continuation parent {key} does not verify")
    for name in UNCHANGED_FILES:
        if ca.file_sha256(run_dir / name) != ca.file_sha256(root / name):
            raise ContinuationRefused(f"continuation changed frozen {name}")
    companion = _json(run_dir / "s1_run_manifest.json")
    if (
        s2m._verify_companion(companion) != manifest["binding_s1_run_manifest_hash"]
        or companion["run_class"] != "promotion"
    ):
        raise ContinuationRefused("continuation companion must bind the same promotion run")
    if companion["run_config"]["sampler"]["batch_size"] != 1:
        raise ContinuationRefused("continuation requires batch_size=1")
    pin = manifest["init_from"]
    if pin["step"] >= companion["run_config"]["steps"]:
        raise ContinuationRefused("parent already exhausted the complete logical plan")
    ledger = ex.ExposureLedger.load_jsonl(root / "exposure_ledger.jsonl", parent["run_id"])
    require_ledger(
        ledger=ledger, step=pin["step"], batch_size=1, start_step=parent_ledger_start(parent)
    )
    if ledger.root_hash() != block["parent_ledger_root_hash"]:
        raise ContinuationRefused("parent ledger root does not verify")
    path = root / "checkpoint.pt"
    raw = _json(ca.sidecar_path(path))
    doc = ca.verify_sidecar(
        path,
        expect_base_ancestry_hash=pin["parent_base_ancestry_hash"],
        expect_s1_run_manifest_hash=pin["parent_s1_run_manifest_hash"],
        embedded_ancestry_hash=raw["ancestry_hash"],
    )
    if any(
        doc["checkpoint"][k] != v
        for k, v in {
            "checkpoint_sha256": pin["checkpoint_sha256"],
            "checkpoint_id": pin["checkpoint_id"],
            "run_id": pin["parent_run_id"],
            "step": pin["step"],
        }.items()
    ):
        raise ContinuationRefused("pinned parent checkpoint differs")
    return Context(
        pin["step"],
        pin["checkpoint_sha256"],
        pin["parent_s1_run_manifest_hash"],
        pin["parent_base_ancestry_hash"],
        doc["ancestry_hash"],
    )


def checkpoint_expectations(
    *,
    context: Context | None,
    path: Path,
    state: Mapping[str, Any],
    companion_hash: str,
    base_ancestry_hash: str,
) -> dict[str, str]:
    """Select strict parent admission only for its exact checkpoint; otherwise child admission."""
    if context is not None:
        if (
            state["step"] == context.step
            and state.get("ancestry_hash") == context.parent_ancestry_hash
        ):
            if ca.file_sha256(path) != context.parent_checkpoint_sha256:
                raise ContinuationRefused("bootstrap checkpoint bytes differ from pinned parent")
            return {
                "expect_s1_run_manifest_hash": context.parent_s1_hash,
                "expect_base_ancestry_hash": context.parent_base_hash,
            }
        if state["step"] <= context.step:
            raise ContinuationRefused("child checkpoint must be after the inherited parent step")
    return {
        "expect_s1_run_manifest_hash": companion_hash,
        "expect_base_ancestry_hash": base_ancestry_hash,
    }


def prepare_external_views_child(
    parent_run_dir: Path,
    child_run_dir: Path,
    *,
    external_views_fields: Mapping[str, Any],
    ratification: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the external-views.v1 child from a finished clean S4 run or external-views child dir.

    The child dir must already hold the freshly frozen UNCHANGED_FILES (same bytes as the
    parent's). Writes continuation-parent/ and the child's s4_run_manifest.json.
    """
    parent_dir, child_dir = Path(parent_run_dir), Path(child_run_dir)
    parent = sm.load_s4_run_manifest(parent_dir / "s4_run_manifest.json")
    checkpoint = parent_dir / "checkpoint.pt"
    companion = _json(parent_dir / "s1_run_manifest.json")
    child = copy.deepcopy(dict(parent))
    child["init_from"] = sm.init_from_block_from_checkpoint(checkpoint, companion)
    child["visual_input"]["external_views"] = copy.deepcopy(dict(external_views_fields))
    base = {k: v for k, v in child.items() if k not in {"run_id", "s4_run_manifest_hash"}}
    child["run_id"] = make_id(sm.RUN_ID_PREFIX, sm.RUN_ID_DOMAIN, base)
    child["s4_run_manifest_hash"] = domain_hash(
        sm.S4_RUN_MANIFEST_DOMAIN, {**base, "run_id": child["run_id"]}
    )
    archive = child_dir / PARENT_DIR
    archive.mkdir(parents=True, exist_ok=True)
    for name in (*UNCHANGED_FILES, "s4_run_manifest.json", "exposure_ledger.jsonl"):
        shutil.copyfile(parent_dir / name, archive / name)
    for name in ("checkpoint.pt", ca.sidecar_path(checkpoint).name):
        shutil.copyfile(parent_dir / name, archive / name)
    (archive / RATIFICATION_FILE).write_text(json.dumps(dict(ratification), sort_keys=True))
    decision = archive / "decision.md"
    decision.write_text(
        f"{ratification.get('decision_id', RATIFICATION_DECISION_ID)}: "
        "external-views.v1 continuation, "
        f"table {external_views_fields['table_sha256']}\n"
    )
    bound = bind_manifest(
        parent=parent,
        child=child,
        parent_ledger=archive / "exposure_ledger.jsonl",
        decision=decision,
        kind=KIND_EXTERNAL_VIEWS,
    )
    (child_dir / "s4_run_manifest.json").write_text(json.dumps(bound))
    return bound

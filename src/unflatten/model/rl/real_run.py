"""S4.5 real-run composer (RFC005-ERRATA-010, DEC-047 addendum A v2; PROPOSED, dormant).

``run_real`` is the update loop behind the ratification gate of
:func:`unflatten.model.rl.s4_5_train.run_s4_5`. It is a library function: no CLI
flag reaches it and it opens nothing itself -- every real resource arrives
through :class:`~unflatten.model.rl.real_run_ports.RealRunPorts`.

Order (each step before the next; nothing before the gate touches an optimizer,
a dataset, a checkpoint or a renderer):

1. ports complete (``RealRunPortMissing``), run manifest re-verified against the
   artefacts;
2. data: ``open_pages`` -> train-only page check (every page ``train``, lineage
   in the manifest and not excluded, manifest hash equal) -> selection checkpoint
   equals the policy parent -> ``require_selection_matches_manifest`` ->
   ``build_rl_data_plan``;
3. models: ``open_candidate`` -> ``open_reference`` -> reference frozen and
   sharing the backbone (re-checked here, independent of the port);
4. ``freeze_to_state_map`` then ``build_rl_optimizer(..., reference=reference)``
   with the same ``run_class`` / ``selected_typed_heads`` as ``update_step``;
5. ledger + resume: the latest ``rl-step-*.pt`` is loaded and
   ``require_resume_ledger_consistent`` must hold (also for step 0: a stale
   ledger is refused);
6. per update step: ``accumulation_steps`` micro-batches (one page each) ->
   ``update_step`` -> one ledger record per page -> every ``checkpoint_every``
   steps ``write_ledger`` THEN ``save_rl_checkpoint``; ``assert_not_reweighted``
   after the first update and at the end; stop at ``max_update_steps`` or
   ``stop_requested()`` (a final checkpoint is written for steps not yet saved).

Known limits, stated rather than worked around:

* ``state_map`` admits only ``typed_heads.*`` groups, and ``walk_typed`` draws
  the ``expert_adapter.dense`` cardinality heads only when that group id is in
  ``selected_typed_heads``; therefore those heads stay argmax in this run.
* a crash between ``write_ledger`` and ``save_rl_checkpoint`` leaves a ledger
  ahead of the newest checkpoint; the resume rule refuses it (fail closed).
* the result never claims quality (``quality_claim`` is the constant ``False``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

from torch import nn

from unflatten.model import s1_train
from unflatten.model.candidate_model import typed_head_children
from unflatten.model.rl.checkpoint_io import (
    load_rl_checkpoint,
    require_resume_ledger_consistent,
    save_rl_checkpoint,
)
from unflatten.model.rl.data import TRAIN_SPLIT, build_rl_data_plan
from unflatten.model.rl.exposure import RlExposureLedger
from unflatten.model.rl.loss import RlLossProvenance
from unflatten.model.rl.policies import MANIFEST_HASH_FIELD, REWARD_HASH_FIELD, ROLLOUT_HASH_FIELD
from unflatten.model.rl.ratification import RlRatificationRequired
from unflatten.model.rl.real_run_loop import build_microbatch
from unflatten.model.rl.real_run_ports import (
    AllBuildersInvalid,
    LoopContext,
    OpenedPages,
    RealRunPortMissing,
    RealRunPorts,
    RealRunRefused,
    RegistryUnbound,
)
from unflatten.model.rl.real_run_result import (
    RealRunResult,
    StepTelemetry,
    append_ledger,
    step_telemetry,
)
from unflatten.model.rl.real_run_score import BUILDER_CRASH_OUTCOME
from unflatten.model.rl.reference import (
    assert_reference_frozen,
    assert_reference_shares_frozen_backbone,
)
from unflatten.model.rl.run_manifest import RunArtefacts, policy_status, verify_rl_run_manifest
from unflatten.model.rl.selection import (
    assert_not_reweighted,
    require_selection_matches_manifest,
)
from unflatten.model.rl.state_map import SUPERVISED_AUXILIARY_ONLY
from unflatten.model.rl.update import (
    build_rl_optimizer,
    freeze_to_state_map,
    update_step,
)

__all__ = [
    "CHECKPOINT_GLOB",
    "INPUTS_PER_MICROBATCH",
    "RealRunPortMissing",
    "RealRunRefused",
    "RealRunResult",
    "RegistryUnbound",
    "StepTelemetry",
    "checkpoint_name",
    "run_real",
]

#: one micro-batch is one page's rollout group, i.e. one ledger record per micro-batch.
INPUTS_PER_MICROBATCH: Final = 1
CHECKPOINT_GLOB: Final = "rl-step-*.pt"


def checkpoint_name(step: int) -> str:
    """Zero-padded, so the lexical order of the names is the numeric order."""
    return f"rl-step-{step:08d}.pt"


def _check_pages(
    opened: OpenedPages, manifest: Mapping[str, Any], parent_checkpoint_id: str
) -> str:
    """Train-only re-check of what the loader returned; returns the frozen selection hash."""
    train = opened.train_pages
    if train.dataset_manifest_hash != manifest[MANIFEST_HASH_FIELD]:
        raise RealRunRefused(
            f"pages were loaded under dataset manifest {train.dataset_manifest_hash}, "
            f"the run binds {manifest[MANIFEST_HASH_FIELD]}"
        )
    if not train.pages:
        raise RealRunRefused("the loader returned no train page")
    allowed, excluded = set(manifest["lineage_ids"]), set(manifest["exclude_lineage_ids"])
    for page in train.pages:
        if page.split_name != TRAIN_SPLIT:
            raise RealRunRefused(
                f"page {page.page_key} has split {page.split_name!r}: S4.5 draws only 'train'"
            )
        if page.lineage_root_id in excluded or page.lineage_root_id not in allowed:
            raise RealRunRefused(
                f"page {page.page_key} lineage {page.lineage_root_id!r} is not an allowed "
                "train lineage of the dataset manifest"
            )
    if opened.selection.checkpoint_id != parent_checkpoint_id:
        raise RealRunRefused(
            f"frozen selection was derived for checkpoint {opened.selection.checkpoint_id!r}, "
            f"the policy parent is {parent_checkpoint_id!r}"
        )
    return require_selection_matches_manifest(opened.selection, manifest)


def _enable_trainable_extras(model: nn.Module, heads: tuple[str, ...], weight: int) -> None:
    """The model builds every typed-head group and the node-visual aligner frozen
    (S4 unfreezes them itself) and ``freeze_to_state_map`` only ever freezes. The
    selected head groups -- and the aligner when the supervised weight is positive --
    are therefore made trainable here, before the state map is enforced."""
    children = typed_head_children()
    unknown = sorted(set(heads) - set(children.values()))
    if unknown:
        raise RealRunRefused(f"selected typed head group(s) {unknown} are not model head groups")
    for child, head_id in children.items():
        if head_id in heads:
            getattr(model, child).requires_grad_(True)
    if weight > 0:
        for group in SUPERVISED_AUXILIARY_ONLY:
            getattr(model, group).requires_grad_(True)


def _require_gate_evidence(manifest: Mapping[str, Any], artefacts: RunArtefacts) -> None:
    """Defence in depth: ``run_real`` called around ``run_s4_5`` still needs the
    ratification the gate would have required (a bound record, RATIFIED policies)."""
    if artefacts.ratification is None or manifest.get("ratification_record_hash") is None:
        raise RlRatificationRequired(
            "run_real needs a manifest and artefacts bound to a ratification record; "
            "none is bound (use run_s4_5, which gates first)"
        )
    if policy_status(artefacts) != "RATIFIED":
        raise RlRatificationRequired(
            f"policy status is {policy_status(artefacts)!r}; only RATIFIED policies may train"
        )


def _refuse_all_builders_invalid(records: list[Any]) -> None:
    """Tripwire (before ``update_step``): a step whose every rollout, over every
    micro-batch, was a ``BuilderReportInvalid`` failed candidate is refused. Mixed
    outcomes still train; the named outcome stays in the telemetry."""
    outcomes = [o for rec, _ in records for o in rec.outcomes]
    if outcomes and all(o == BUILDER_CRASH_OUTCOME for o in outcomes):
        raise AllBuildersInvalid(
            f"all {len(outcomes)} rollouts of the update step ended as "
            f"{BUILDER_CRASH_OUTCOME}: no learning signal, update refused"
        )


def _latest_checkpoint(directory: Path) -> Path | None:
    found = sorted(Path(directory).glob(CHECKPOINT_GLOB))
    return found[-1] if found else None


def _ledger_for_run(
    ports: RealRunPorts, run_id: str, arts: RunArtefacts, step: int, per_update: int
) -> RlExposureLedger:
    assert ports.open_ledger is not None
    ledger = ports.open_ledger(run_id, arts.dataset_manifest, arts.reward_policy[REWARD_HASH_FIELD])
    require_resume_ledger_consistent(ledger, step, records_per_update=per_update)
    return ledger


def run_real(
    *,
    manifest: Mapping[str, Any],
    artefacts: RunArtefacts,
    run_class: str,
    selected_typed_heads: Iterable[str],
    ports: RealRunPorts,
    allow_unbound_registry: bool = False,
) -> RealRunResult:
    """The S4.5 update loop (see the module docstring for the order).

    ``allow_unbound_registry`` is a FIXTURE-ONLY escape: a real run's pages must carry
    the exclusion-registry hash (``RegistryUnbound`` otherwise, before any model work)."""
    _require_gate_evidence(manifest, artefacts)
    heads = tuple(sorted(set(selected_typed_heads)))
    tp, rp = artefacts.training_policy, artefacts.rollout_policy
    weight = tp["supervised_weight_micro"]
    if type(weight) is not int:
        raise RealRunRefused(f"supervised_weight_micro is not frozen: {weight!r}")
    ports.require_complete(supervised_weight_micro=weight, render_probe_mode=rp["render_probe"])
    verify_rl_run_manifest(manifest, artefacts)
    assert ports.open_pages is not None and ports.read_selection is not None
    assert ports.open_candidate is not None and ports.open_reference is not None
    assert ports.checkpoint_dir is not None and ports.ancestry_base is not None
    assert ports.stop_requested is not None and ports.write_ledger is not None
    assert ports.feature_source is not None

    pages = ports.open_pages(artefacts.dataset_manifest)
    frozen_hash = _check_pages(
        pages, artefacts.dataset_manifest, artefacts.policy_parent.checkpoint_id
    )
    if pages.exclusion_registry_hash is None and not allow_unbound_registry:
        raise RegistryUnbound(
            "the opened pages carry no exclusion_registry_hash: a real run's data plan "
            "must bind the exclusion registry"
        )
    plan = build_rl_data_plan(
        stage_data_matrix_hash=pages.stage_data_matrix_hash,
        dataset_manifest=artefacts.dataset_manifest,
        exclusion_registry_hash=pages.exclusion_registry_hash,
    )

    opened = ports.open_candidate()
    if not isinstance(opened.policy, nn.Module):
        raise RealRunRefused(f"open_candidate returned a {type(opened.policy).__name__} policy")
    reference = ports.open_reference(opened)
    assert_reference_frozen(reference)
    assert_reference_shares_frozen_backbone(
        opened.policy, reference, candidate_parameters=opened.candidate_parameters
    )
    _enable_trainable_extras(opened.policy, heads, weight)
    freeze_to_state_map(opened.policy, run_class, heads, auxiliary_groups_trainable=weight > 0)
    optimizer = build_rl_optimizer(opened.policy, tp, run_class, heads, reference=reference)

    accumulation, every = tp["accumulation_steps"], tp["checkpoint_every"]
    per_update = accumulation * INPUTS_PER_MICROBATCH
    run_id = str(manifest["run_id"])
    ancestry = ports.ancestry_base()
    start = 0
    latest = _latest_checkpoint(ports.checkpoint_dir)
    if latest is not None:
        progress, _ = load_rl_checkpoint(
            latest,
            model=opened.policy,
            optimizer=optimizer,
            run_manifest=manifest,
            expect_base_ancestry_hash=ancestry["ancestry_hash"],
            expect_feature_source=ports.feature_source,
        )
        start = progress.step
    ledger = _ledger_for_run(ports, run_id, artefacts, start, per_update)
    assert ports.render_dir is not None
    Path(ports.render_dir).mkdir(parents=True, exist_ok=True)  # the RFC002 compile target

    ctx = LoopContext(
        opened=opened,
        reference=reference,
        rollout_policy=rp,
        reward_policy=artefacts.reward_policy,
        heads=heads,
        run_seed=int(manifest["run_seed"]),
        provenance=RlLossProvenance(
            reward_policy_hash=artefacts.reward_policy[REWARD_HASH_FIELD],
            rollout_policy_hash=rp[ROLLOUT_HASH_FIELD],
            reference_checkpoint_id=artefacts.reference.checkpoint_id,
            reference_ancestry_hash=artefacts.reference.ancestry_hash,
            typed_head_config_hash=artefacts.policy_parent.typed_head_config_hash,
        ),
        supervised_weight_micro=weight,
        ports=ports,
    )
    return _loop(
        ctx,
        optimizer,
        ledger,
        pages,
        manifest=manifest,
        artefacts=artefacts,
        run_class=run_class,
        ancestry=ancestry,
        start=start,
        frozen_hash=frozen_hash,
        plan_hash=plan.plan_hash,
        accumulation=accumulation,
        every=every,
    )


def _loop(
    ctx: LoopContext,
    optimizer: Any,
    ledger: RlExposureLedger,
    pages: OpenedPages,
    *,
    manifest: Mapping[str, Any],
    artefacts: RunArtefacts,
    run_class: str,
    ancestry: Mapping[str, Any],
    start: int,
    frozen_hash: str,
    plan_hash: str,
    accumulation: int,
    every: int,
) -> RealRunResult:
    ports, tp = ctx.ports, artefacts.training_policy
    assert ports.stop_requested and ports.write_ledger and ports.read_selection
    assert ports.checkpoint_dir is not None and ports.feature_source is not None
    train_pages = pages.train_pages.pages
    checkpoints: list[Path] = []
    telemetry: list[StepTelemetry] = []
    step, saved, stopped_by, checked = start, start, "max_update_steps", False

    def persist(at: int) -> None:
        assert_not_reweighted(frozen_hash, ports.read_selection())  # type: ignore[misc]
        ports.write_ledger(ledger)  # type: ignore[misc]  # ledger BEFORE the checkpoint
        path = Path(ports.checkpoint_dir) / checkpoint_name(at)  # type: ignore[arg-type]
        save_rl_checkpoint(
            path,
            model=ctx.opened.policy,
            optimizer=optimizer,
            progress=s1_train.TrainingProgress(step=at),
            run_manifest=manifest,
            feature_source=ports.feature_source,  # type: ignore[arg-type]
            ancestry_base=ancestry,
        )
        checkpoints.append(path)

    while step < tp["max_update_steps"]:
        if ports.stop_requested():
            stopped_by = "stop_requested"
            break
        micro, records = [], []
        for k in range(accumulation):
            index = step * accumulation + k
            mb, rec = build_microbatch(
                ctx, train_pages[index % len(train_pages)], epoch=index // len(train_pages)
            )
            micro.append(mb)
            records.append((rec, index // len(train_pages)))
        _refuse_all_builders_invalid(records)
        result = update_step(
            ctx.opened.policy,
            optimizer,
            micro,
            training_policy=tp,
            rollout_policy=ctx.rollout_policy,
            step=step,
            run_class=run_class,
            selected_typed_heads=ctx.heads,
        )
        append_ledger(ledger, records, step)
        telemetry.append(step_telemetry(result, [r for r, _ in records]))
        step += 1
        if not checked:
            assert_not_reweighted(frozen_hash, ports.read_selection())
            checked = True
        if step % every == 0:
            persist(step)
            saved = step
    if step > saved:
        persist(step)
    return RealRunResult(
        run_id=str(manifest["run_id"]),
        start_step=start,
        steps_done=step,
        updates_run=step - start,
        stopped_by=stopped_by,
        checkpoint_paths=tuple(checkpoints),
        ledger_record_count=ledger.record_count,
        ledger_head_hash=ledger.records[-1].record_hash if ledger.records else None,
        data_plan_hash=plan_hash,
        selection_hash=frozen_hash,
        telemetry=tuple(telemetry),
    )

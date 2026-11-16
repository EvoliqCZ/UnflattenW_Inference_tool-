"""S3 discovery optimizer construction, warm-start weight loading, and
admission (spec-s3-code-readiness design §5-§7). `main` runs admission
steps 1-6 (design §7.3), builds the model and optimizer, then runs the
training loop (`run_s3_training`, design §7.4) -- including `--resume` and
the early typed refusal -- and the CLI (design §9).

Steps 1-4 reuse `s2_train`'s admission functions VERBATIM (same field
names on the S3 manifest as on the S2 one): `require_step2_bound_to_companion`,
`require_step3_policies_bound`, `require_plan_covers_budget`,
`resolve_plan_pages`, `plan_dataset_order`. Their `S2TrainingRefused` is
re-raised here as `S3TrainingRefused` so a caller only ever sees this
module's own refusal type. Step 5 (the typed-supervision floor gate) and
step 6 (typed-head unfreeze/re-freeze + stage-state) are this module's own,
because they reason about fields no S2 run manifest carries.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from unflatten.model import checkpoint_ancestry as ca
from unflatten.model import s1_train
from unflatten.model import s2_exposure as ex
from unflatten.model import s2_train as s2t
from unflatten.model import s3_typed_loss as stl
from unflatten.model import stage_policy as spolicy
from unflatten.model import stage_state_policy as ssp
from unflatten.model.candidate_model import (
    QwenDenseCandidateModel,
    parameter_group_ids,
    typed_head_children,
)
from unflatten.model.exact_text_cache import CacheExpectation, PageKey
from unflatten.model.s2_sampler import load_s2_exposure_policy, load_s2_sampler_policy
from unflatten.model.s3_run_manifest import (
    S3RunManifestInvalid,
    head_meets_floor,
    load_s3_run_manifest,
)
from unflatten.model.s3_run_manifest import (
    _l_typed_head_ids as l_typed_head_ids,
)
from unflatten.model.s3_run_manifest import (
    _verify_census_against_companion as verify_census_against_companion,
)
from unflatten.model.stage_data_matrix import load_stage_data_matrix

#: The exposure-ledger cell every S3 discovery record is written under
#: (RFC005-ERRATA-004 E4: run 1 reuses the S2 matrix literally) -- the same
#: tuple `s2_train.CELL` names, re-exported rather than restated so the two
#: modules cannot silently drift apart.
CELL = s2t.CELL

#: `s2_train`'s ledger filename convention, reused verbatim (design §7.4).
_LEDGER_FILENAME = s2t._LEDGER_FILENAME

#: How many optimizer steps of THIS run_s3_training call the early typed
#: refusal waits for before checking (design §7.4): "after min(steps, 50)
#: steps" -- never more than the run's own total budget.
_EARLY_REFUSAL_STEP_CAP = 50

#: The committed policy directory `main()` reads from by default
#: (overridable in tests via `monkeypatch.setattr(s3_train, "_POLICIES_DIR", ...)`,
#: the same pattern `s2_train._POLICIES_DIR` uses -- the two directories are
#: the SAME path in production, kept as separate module attributes so each
#: module's tests can monkeypatch independently).
_POLICIES_DIR = Path(__file__).parent / "policies"

#: The state_dict key prefix of the dense property-head child an S2
#: checkpoint always lacks (`candidate_model._DENSE_HEAD_GROUP_ATTR`,
#: design §4.5/§6, D6 = A). Not imported from `candidate_model` because
#: that name is private to it; this module only needs the prefix, which
#: is exactly the child's attribute name plus a dot.
_DENSE_HEAD_STATE_DICT_PREFIX = "expert_adapter_dense_heads."


class S3TrainingRefused(RuntimeError):
    """`build_s3_optimizer` refused to build the optimizer: a
    `requires_grad` parameter with no group mapping, or a mapped group
    whose S3 policy state for `run_class` does not allow `trainable`
    (design §5). Also raised by `load_init_weights` for any of its six
    ordered refusals (design §6)."""


@dataclasses.dataclass(frozen=True)
class InitReport:
    """What `load_init_weights` actually did, for the caller's manifest /
    attempt record: the verified `checkpoint_id`, the `step` the parent
    checkpoint pinned, and the exact declared-fresh key names that were
    left at the target model's own initialization (design §6 step 5).

    `fresh_source` names which parent shape was seen: `"absent_in_parent"`
    (an old-code S2 parent that predates the dense-head child) or
    `"reinitialized_from_frozen_parent"` (a new-code S2 parent that carries
    the heads, frozen in S2, whose values were discarded for the target's
    own initialization). Either way the heads start at the same values."""

    checkpoint_id: str
    step: int
    fresh_keys: tuple[str, ...]
    fresh_source: str
    cross_release_fields: tuple[str, ...] = ()


#: Cache identity fields a warm start may cross (DEC-035 B rule, applied to
#: S3 on release-v14-r4, owner 2026-09-21): the release and the materializer
#: change, the vocabulary, grammar, head registry and profile must not.
CROSS_RELEASE_FIELDS: frozenset[str] = frozenset(
    {"dataset_release_root_hash", "training_target_materializer_hash"}
)


def declared_fresh_keys(model: QwenDenseCandidateModel) -> frozenset[str]:
    """The exact `model.state_dict()` key names a warm start from an S2
    checkpoint is DECLARED to miss: the dense property-head child's
    parameters (design §6 step 5, §12 breaking-change note) -- computed
    from `model` itself, not hardcoded, so a change to the head registry
    changes this set rather than silently going undetected."""
    return frozenset(
        key for key in model.state_dict() if key.startswith(_DENSE_HEAD_STATE_DICT_PREFIX)
    )


def load_init_weights(
    path: Path | str,
    model: QwenDenseCandidateModel,
    *,
    pinned: Mapping[str, Any],
    expect_cache_identity: Mapping[str, Any],
) -> InitReport:
    """Warm-start `model`'s weights from the S2 checkpoint at `path`
    (design §6) -- weights only, never `--resume`: no optimizer state, no
    RNG restore, and this function takes no optimizer at all.

    `pinned` is the S3 manifest's `init_from` block: `checkpoint_sha256`,
    `checkpoint_id`, `parent_s1_run_manifest_hash`,
    `parent_base_ancestry_hash`, `step`. Refusals run in this order, each
    raising `S3TrainingRefused` naming the failed check:

    1. `path`'s actual file bytes hash to `pinned["checkpoint_sha256"]`.
    2. The checkpoint's sidecar verifies (`checkpoint_ancestry.verify_sidecar`)
       against `parent_base_ancestry_hash` / `parent_s1_run_manifest_hash` /
       the embedded `state["ancestry_hash"]`, and the checkpoint_id
       recomputed from the verified ancestry hash and the file's sha256
       equals `pinned["checkpoint_id"]`.
    3. `state["feature_source"] == "cache"` -- never a null-model checkpoint.
    4. `state["cache_identity"]` equals `expect_cache_identity` except in
       `CROSS_RELEASE_FIELDS` (release, materializer); the crossed fields are
       returned in `cross_release_fields`.
    5. `model.load_state_dict(state["model_state_dict"], strict=False)`
       reports no unexpected keys and a missing-key set that is either
       exactly `declared_fresh_keys(model)` (old-code parent: the heads keep
       the target's own initialization) or empty (new-code parent, heads
       frozen in S2: the parent's head values are overwritten with the
       target's own pre-load initialization, so S3's heads start fresh
       regardless of which code trained the parent). Any other missing set
       is refused -- a partial subset of the heads, or any non-head key.
    """
    path = Path(path)
    actual_sha256 = ca.file_sha256(path)
    if actual_sha256 != pinned["checkpoint_sha256"]:
        raise S3TrainingRefused(
            f"load_init_weights: checkpoint sha256 {actual_sha256!r} != pinned "
            f"{pinned['checkpoint_sha256']!r} -- refusing to warm-start from a "
            "checkpoint whose bytes do not match the S3 manifest's init_from pin"
        )

    state = torch.load(path, map_location="cpu", weights_only=False)

    try:
        doc = ca.verify_sidecar(
            path,
            expect_base_ancestry_hash=pinned["parent_base_ancestry_hash"],
            expect_s1_run_manifest_hash=pinned["parent_s1_run_manifest_hash"],
            embedded_ancestry_hash=state.get("ancestry_hash"),
        )
    except ca.CheckpointAncestryInvalid as exc:
        raise S3TrainingRefused(
            f"load_init_weights: checkpoint ancestry sidecar refused: {exc}"
        ) from exc

    checkpoint_id = ca.compute_checkpoint_id(doc["ancestry_hash"], actual_sha256)
    if checkpoint_id != pinned["checkpoint_id"]:
        raise S3TrainingRefused(
            f"load_init_weights: recomputed checkpoint_id {checkpoint_id!r} != "
            f"pinned {pinned['checkpoint_id']!r}"
        )

    if state.get("feature_source") != "cache":
        raise S3TrainingRefused(
            f"load_init_weights: checkpoint feature_source={state.get('feature_source')!r} "
            "!= 'cache' -- a null-model checkpoint cannot warm-start S3"
        )

    parent_identity = state.get("cache_identity") or {}
    differing = sorted(
        key
        for key in set(parent_identity) | set(expect_cache_identity)
        if parent_identity.get(key) != expect_cache_identity.get(key)
    )
    not_crossable = [key for key in differing if key not in CROSS_RELEASE_FIELDS]
    if not_crossable:
        raise S3TrainingRefused(
            "load_init_weights: checkpoint cache_identity != expect_cache_identity "
            f"in {not_crossable} -- only {sorted(CROSS_RELEASE_FIELDS)} may differ "
            "across a warm start"
        )

    fresh = declared_fresh_keys(model)
    fresh_init = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key in fresh
    }
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if unexpected:
        raise S3TrainingRefused(
            f"load_init_weights: model.load_state_dict reports unexpected keys "
            f"{list(unexpected)!r} -- a warm start must not silently ignore keys "
            "the checkpoint carries"
        )
    if set(missing) == fresh:
        fresh_source = "absent_in_parent"
    elif not missing:
        fresh_source = "reinitialized_from_frozen_parent"
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if key in fresh:
                    value.copy_(fresh_init[key])
    else:
        raise S3TrainingRefused(
            "load_init_weights: model.load_state_dict is missing "
            f"{sorted(set(missing) - fresh)!r} beyond the declared-fresh dense-head "
            f"keys and {len(set(missing) & fresh)} of {len(fresh)} declared-fresh "
            "keys -- neither an old-code parent (all declared-fresh keys missing) "
            "nor a new-code parent (none missing)"
        )

    return InitReport(
        checkpoint_id=pinned["checkpoint_id"],
        step=pinned["step"],
        fresh_keys=tuple(sorted(fresh)),
        fresh_source=fresh_source,
        cross_release_fields=tuple(differing),
    )


def build_s3_optimizer(
    model: QwenDenseCandidateModel,
    *,
    lr_standard: float,
    lr_reduced: float,
    algorithm: str,
    run_class: str | None,
) -> torch.optim.Optimizer:
    """Two param groups over `model`'s `requires_grad` parameters --
    `param_groups[0]` at `lr_standard`, `param_groups[1]` at `lr_reduced` --
    split by `stage_state_policy.stage_lr_class("s3", group_id,
    run_class=run_class)` over `candidate_model.parameter_group_ids(model)`
    (design §5). Mirrors `s1_runner.build_s1_optimizer`'s construction
    (supported algorithms, the empty-parameter-set refusal); unlike it,
    this function must also refuse per-parameter, because S3 trains a
    STRICT SUBSET of `model.parameters()` chosen by policy rather than
    everything `requires_grad` reports:

    - a `requires_grad` parameter `parameter_group_ids` does not map to any
      group (an unaccounted trainable tensor an optimizer built from
      `model.parameters()` would otherwise silently train);
    - a mapped group whose S3 policy state for `run_class` is not
      `trainable` (`stage_lr_class` raises `KeyError`/`ValueError` for
      that; both are re-raised as `S3TrainingRefused` with the parameter
      name attached).
    """
    group_id_of = parameter_group_ids(model)
    standard: list[torch.nn.Parameter] = []
    reduced: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_id = group_id_of.get(name)
        if group_id is None:
            raise S3TrainingRefused(
                f"parameter {name!r} requires grad but "
                "candidate_model.parameter_group_ids maps it to no "
                "trainable group -- an optimizer built from it would "
                "silently train an unaccounted parameter"
            )
        try:
            lr_class = ssp.stage_lr_class("s3", group_id, run_class=run_class)
        except (KeyError, ValueError) as exc:
            raise S3TrainingRefused(
                f"parameter {name!r} maps to group {group_id!r}, which the "
                f"S3 policy (run_class={run_class!r}) does not allow "
                f"trainable: {exc}"
            ) from exc
        (standard if lr_class == ssp.STANDARD else reduced).append(param)

    if not standard and not reduced:
        raise ValueError(
            "model reports no trainable parameters -- an optimizer over an "
            "empty parameter set would make the S3 loop vacuous"
        )

    param_groups = [
        {"params": standard, "lr": lr_standard},
        {"params": reduced, "lr": lr_reduced},
    ]
    if algorithm == "sgd":
        return torch.optim.SGD(param_groups)
    if algorithm == "adamw":
        return torch.optim.AdamW(param_groups)
    raise ValueError(
        f"unknown optimizer algorithm {algorithm!r} -- expected 'sgd' or 'adamw'"
    )


# ---------------------------------------------------------------------------
# Step 5 -- typed supervision census / floor gate (design §7.3 step 5)
# ---------------------------------------------------------------------------


def require_step5_typed_supervision(
    *,
    s3_manifest: Mapping[str, Any],
    companion: Mapping[str, Any],
    stage_policy: Mapping[str, Any],
    census: Mapping[str, Any],
) -> None:
    """The typed-supervision census bound to THIS run -- never trusted from
    the manifest alone: re-verified against the presented companion
    (`s3_run_manifest._verify_census_against_companion`) and against the
    manifest's own pinned `typed_supervision_census_hash` -- and the floor
    gate over every `L_typed` head the committed stage policy declares:

    - if every `L_typed` head is listed in `heads_frozen_for_no_supervision`,
      refused outright (a typed run that trains no typed head);
    - a head NOT listed there must MEET both `head_meets_floor` dimensions
      (`s3_run_manifest.head_meets_floor`, the one floor predicate -- never
      re-implemented here);
    - a head listed there must NOT meet them (an unused freeze is refused
      too, the DEC-027 D2 rule, re-checked here even though the manifest
      builder already enforces it -- the manifest and the census presented
      to a run are not necessarily the ones the builder saw).
    """
    try:
        census_hash = verify_census_against_companion(census, companion)
    except S3RunManifestInvalid as exc:
        raise S3TrainingRefused(f"typed supervision census refused: {exc}") from exc
    if census_hash != s3_manifest["typed_supervision_census_hash"]:
        raise S3TrainingRefused(
            f"typed supervision census_hash {census_hash!r} does not match the "
            "pinned s3_run_manifest typed_supervision_census_hash "
            f"{s3_manifest['typed_supervision_census_hash']!r}"
        )

    floor = s3_manifest["typed_supervision_floor"]
    frozen = set(s3_manifest["heads_frozen_for_no_supervision"])
    head_ids = l_typed_head_ids(stage_policy)
    if head_ids and frozen >= head_ids:
        raise S3TrainingRefused(
            "every L_typed head is listed in heads_frozen_for_no_supervision -- "
            "a typed run that trains no typed head"
        )

    per_head = census["counts"]["per_head"]
    for head_id in sorted(head_ids):
        entry = per_head.get(head_id, {})
        meets_floor = head_meets_floor(entry, floor)
        if head_id in frozen:
            if meets_floor:
                raise S3TrainingRefused(
                    f"{head_id!r} is listed in heads_frozen_for_no_supervision but "
                    "its census entry meets the supervision floor -- an unused "
                    "freeze is a lie too (DEC-027 D2 rule)"
                )
        elif not meets_floor:
            raise S3TrainingRefused(
                f"{head_id!r} is not listed in heads_frozen_for_no_supervision but "
                f"its census entry (supervised={entry.get('supervised', 0)}, "
                f"lineage_count={entry.get('lineage_count', 0)}) does not meet the "
                f"supervision floor {dict(floor)!r}"
            )


# ---------------------------------------------------------------------------
# Step 6 -- typed-head unfreeze/re-freeze, stage-state, trainable typed half
# (design §5, §7.3 step 6)
# ---------------------------------------------------------------------------


def unfreeze_and_refreeze_typed_heads(
    model: QwenDenseCandidateModel, heads_frozen_for_no_supervision: Sequence[str]
) -> None:
    """Unfreeze every typed-head child, then re-freeze exactly the ones the
    S3 manifest lists as `heads_frozen_for_no_supervision` -- the order
    matters: a group NOT listed must end up trainable even though S1 froze
    it at construction."""
    frozen = set(heads_frozen_for_no_supervision)
    for child_name, head_id in typed_head_children().items():
        getattr(model, child_name).requires_grad_(head_id not in frozen)


def require_stage_states(
    states: Mapping[str, str],
    *,
    run_class: str,
    heads_frozen_for_no_supervision: Sequence[str],
) -> None:
    """Pure over a `{group_id: state}` map (design §7.3 step 6): every
    group must be a state `stage_state_policy` allows for `"s3"` under
    `run_class` (`validate_stage_state`'s `run_class` argument already
    encodes the RFC005-ERRATA-004 E2 discovery-vs-promotion split -- this
    function never restates it), PLUS the one thing the policy cannot
    express on its own: a `typed_heads.*` group resting `frozen` that is
    NOT in `heads_frozen_for_no_supervision` is refused even though
    `frozen` is itself a policy-allowed discovery state for it -- the
    policy says a typed head MAY be frozen, the manifest says WHICH."""
    frozen = set(heads_frozen_for_no_supervision)
    for group_id, state in states.items():
        try:
            ssp.validate_stage_state("s3", group_id, state, run_class=run_class)
        except ValueError as exc:
            raise S3TrainingRefused(f"stage-state check refused: {exc}") from exc
        if group_id.startswith("typed_heads.") and state == "frozen" and group_id not in frozen:
            raise S3TrainingRefused(
                f"{group_id!r} is frozen but is not listed in "
                "heads_frozen_for_no_supervision -- the S3 discovery policy admits "
                "a frozen typed head only by name (RFC005-ERRATA-004 E2)"
            )


def require_step6_stage_state(
    *,
    model: QwenDenseCandidateModel,
    run_class: str,
    heads_frozen_for_no_supervision: Sequence[str],
) -> None:
    require_stage_states(
        s2t._model_group_states(model),
        run_class=run_class,
        heads_frozen_for_no_supervision=heads_frozen_for_no_supervision,
    )


def require_trainable_typed_half(states: Mapping[str, str]) -> None:
    """At least one `typed_heads.*` group must be trainable -- the
    deferred half of DEC-017 reading B (`run_binding.py:342-352`), placed
    here so `run_binding` stays frozen for S1/S2 (design §7.3 step 6)."""
    if not any(
        group_id.startswith("typed_heads.") and state == "trainable"
        for group_id, state in states.items()
    ):
        raise S3TrainingRefused(
            "no typed_heads.* group is trainable -- an S3 typed run must train "
            "at least one typed head (design §7.3 step 6)"
        )


# ---------------------------------------------------------------------------
# Grad clip and init_from CLI/manifest agreement (design §5, §6)
# ---------------------------------------------------------------------------


def require_grad_clip_bound(*, companion: Mapping[str, Any], cli_grad_clip: float) -> None:
    """`run_config.json`'s (the companion's) `optimizer.grad_clip_micro` is
    the ONE authoritative field for the grad clip (design §5, [governance
    R1]); S3 refuses a null value -- the S2 gap this design's §0 finding
    3 records -- and a CLI value that disagrees with it."""
    grad_clip_micro = companion["run_config"]["optimizer"]["grad_clip_micro"]
    if grad_clip_micro is None:
        raise S3TrainingRefused(
            "companion run_config.optimizer.grad_clip_micro is null -- S3 refuses "
            "an unpinned grad clip (design §5, governance R1)"
        )
    cli_micro = int(round(cli_grad_clip * 1e6))
    if cli_micro != grad_clip_micro:
        raise S3TrainingRefused(
            f"--grad-clip {cli_grad_clip!r} (grad_clip_micro {cli_micro}) does not "
            f"match the frozen run_config.optimizer.grad_clip_micro {grad_clip_micro}"
        )


def require_init_from_cli_manifest_agreement(
    *, manifest_init_from: Mapping[str, Any] | None, cli_init_from: Path | None
) -> None:
    """`--init-from` and the S3 manifest's `init_from` must agree on
    PRESENCE in both directions (design §6) -- neither may silently win."""
    if manifest_init_from is None and cli_init_from is not None:
        raise S3TrainingRefused(
            "--init-from was supplied but the S3 manifest's init_from is null"
        )
    if manifest_init_from is not None and cli_init_from is None:
        raise S3TrainingRefused(
            "the S3 manifest pins init_from but --init-from was not supplied"
        )


def require_init_from_ancestry_lineage(
    *, ancestry: Mapping[str, Any], init_from: Mapping[str, Any] | None
) -> None:
    """ERRATA-004 E6 lineage cross-check between `checkpoint_ancestry_base.json`
    and the S3 manifest's `init_from` (design §6): the two must agree on
    whether this run is a warm start, and if it is, on WHICH checkpoint.

    - `init_from` present: the base's `initialization_kind` must be
      `"warm_start"` and its `parents[0].checkpoint_id` /
      `parents[0].checkpoint_sha256` must equal `init_from`'s own.
    - `init_from` null: the base's `initialization_kind` must be
      `"clean_pretrained"`.
    """
    initialization_kind = ancestry["initialization_kind"]
    if init_from is None:
        if initialization_kind != "clean_pretrained":
            raise S3TrainingRefused(
                "s3_run_manifest init_from is null but checkpoint_ancestry_base "
                f"initialization_kind is {initialization_kind!r}, not "
                "'clean_pretrained'"
            )
        return

    if initialization_kind != "warm_start":
        raise S3TrainingRefused(
            "s3_run_manifest pins init_from but checkpoint_ancestry_base "
            f"initialization_kind is {initialization_kind!r}, not 'warm_start'"
        )
    parents = ancestry.get("parents") or []
    if not parents:
        raise S3TrainingRefused(
            "checkpoint_ancestry_base initialization_kind is warm_start but "
            "carries no parents[0] entry"
        )
    parent = parents[0]
    if parent["checkpoint_id"] != init_from["checkpoint_id"]:
        raise S3TrainingRefused(
            f"checkpoint_ancestry_base parents[0].checkpoint_id "
            f"({parent['checkpoint_id']!r}) != s3_run_manifest init_from.checkpoint_id "
            f"({init_from['checkpoint_id']!r})"
        )
    if parent["checkpoint_sha256"] != init_from["checkpoint_sha256"]:
        raise S3TrainingRefused(
            f"checkpoint_ancestry_base parents[0].checkpoint_sha256 "
            f"({parent['checkpoint_sha256']!r}) != s3_run_manifest "
            f"init_from.checkpoint_sha256 ({init_from['checkpoint_sha256']!r})"
        )


def require_resume_init_from_agreement(
    *,
    cli_resume: Path | None,
    manifest_init_from: Mapping[str, Any] | None,
    cli_init_from: Path | None,
) -> None:
    """Resolves the `--resume` / `--init-from` conflict the launch runbook
    (design §9) surfaced: resuming reproduces "the same command ... WITHOUT
    `--init-from`" even when the manifest pins one, because a resumed
    checkpoint already carries the warm start in its own weights and
    ancestry.

    - `--resume` together with `--init-from` is refused: a resume restores
      weights from its own checkpoint (`load_checkpoint`), so a presented
      `--init-from` could never be applied and silently doing nothing with
      it would hide the conflict rather than name it.
    - `--resume` without `--init-from` is accepted UNCONDITIONALLY, even
      when the manifest pins `init_from` -- `load_checkpoint` (called
      instead of `load_init_weights` on this path) verifies the resumed
      checkpoint's own ancestry against the S3 base (warm-start) ancestry,
      which is where that lineage is actually re-checked.
    - With no `--resume`, this delegates to
      :func:`require_init_from_cli_manifest_agreement` unchanged -- the
      original four-way PRESENCE agreement still applies to a fresh run.
    """
    if cli_resume is not None and cli_init_from is not None:
        raise S3TrainingRefused(
            "--resume and --init-from were both supplied -- a resume restores "
            "weights from its own checkpoint (load_checkpoint); --init-from "
            "only applies to a fresh warm start and the runbook's resume "
            "command never carries it (design §9)"
        )
    if cli_resume is None:
        require_init_from_cli_manifest_agreement(
            manifest_init_from=manifest_init_from, cli_init_from=cli_init_from
        )


# ---------------------------------------------------------------------------
# The loop (design §7.4)
# ---------------------------------------------------------------------------


def _grad_norm_groups(model: QwenDenseCandidateModel) -> dict[str, float]:
    """Grouped by the first dotted component of the parameter's own name --
    the same convention `s1_train.train_step` uses, now including the
    typed-head children (design §4.6)."""
    grouped: dict[str, list[torch.Tensor]] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grouped.setdefault(name.split(".")[0], []).append(param.grad)
    return {group: s1_train._grad_norm(grads) for group, grads in grouped.items()}


def run_s3_training(
    *,
    run_dir: Path,
    model: QwenDenseCandidateModel,
    dataset: stl.S3TrainingDataset,
    optimizer: torch.optim.Optimizer,
    image_token_id: int,
    plan_pages: Sequence[PageKey],
    plan_lineages: Sequence[str],
    plan_order: Sequence[int],
    companion_hash: str,
    s3_run_id: str,
    steps: int,
    batch_size: int,
    stage_policy: Mapping[str, Any],
    allowed_head_ids: Collection[str],
    grad_clip: float,
    device: torch.device | str = "cpu",
    checkpoint_every: int = 0,
    out: Path | None = None,
    start_step: int = 0,
    ledger: ex.ExposureLedger | None = None,
    execution_profile: Mapping[str, str] | None = None,
    ancestry_base: Mapping[str, Any] | None = None,
    integrity: Mapping[str, str],
    num_workers: int = 0,
) -> ex.ExposureLedger:
    """Run the S3 optimizer steps from `start_step` up to `steps` -- the
    run's TOTAL budget -- over the frozen plan order (design §7.4). Mirrors
    `s2_train.run_training`'s structure (attempt bracketing, one ledger
    record per SAMPLE, checkpoint/ledger at the same cadence) but ALSO:
    collates the typed batch, gathers typed losses from the SAME `hidden`
    `s1_train.scored_logits` would slice for structure/text, clips with the
    caller's own `grad_clip` (no default -- design §5's "passed explicitly,
    never a default"), and refuses the run early if it trains no typed
    supervision at all within its first `min(steps, 50)` steps OF THIS
    ATTEMPT -- a resume that starts past the global step 50 must not be
    refused on the strength of one batch it has not had 50 steps to prove
    itself over (coordinator review 2026-09-15: the check used to compare
    the GLOBAL step against 50, so a resume from e.g. step 250 could be
    refused on its very first (251st) batch alone).

    `grad_clip` has no default value on purpose (unlike
    `s2_train.run_training`'s `grad_clip: float = 1.0`): a caller that
    forgets to pass it gets a `TypeError`, never a silent 1.0.
    """
    if start_step >= steps:
        raise S3TrainingRefused(
            f"start_step {start_step} has already reached the run's total budget "
            f"of {steps} step(s) -- nothing left to run"
        )
    ledger = ledger if ledger is not None else ex.ExposureLedger(s3_run_id)
    execution_profile = (
        dict(execution_profile)
        if execution_profile
        else s2t.execution_profile_for(device, num_workers)
    )
    attempts_dir = s2t._attempts_dir(run_dir)
    attempt_index = s2t._next_attempt_index(run_dir)
    attempt = ex.start_attempt(
        run_id=s3_run_id,
        attempt_index=attempt_index,
        execution_profile=execution_profile,
        start_integrity=integrity,
        ledger=ledger,
    )
    (attempts_dir / f"attempt-{attempt_index:04d}-start.json").write_text(
        json.dumps(attempt, sort_keys=True)
    )
    if num_workers > 0:
        s2t.write_execution_profile_sibling(attempts_dir, attempt_index, execution_profile)

    typed_supervised_seen: dict[str, int] = {}

    def _finalize(terminal_state: str) -> None:
        finalized = ex.finalize_attempt(
            manifest=attempt,
            terminal_state=terminal_state,
            ledger=ledger,
            end_integrity=integrity,
        )
        (attempts_dir / f"attempt-{attempt_index:04d}-final.json").write_text(
            json.dumps(finalized, sort_keys=True)
        )
        # The cumulative typed_supervised_seen[head_id] "goes into the
        # attempt-final record" (design §4.6/§7.4) -- as a SIBLING file,
        # not a key merged into the manifest itself:
        # `run_attempt_manifest.schema.json` is `additionalProperties:
        # false` (§44.3's closed envelope), so an extra key would make this
        # attempt's own manifest fail its own schema the moment anything
        # re-verifies it (`ex.verify_run_attempt_manifest`).
        (attempts_dir / f"attempt-{attempt_index:04d}-typed-supervision.json").write_text(
            json.dumps(dict(typed_supervised_seen), sort_keys=True)
        )

    # Steps OF THIS ATTEMPT, never the global step -- a resume starting
    # past step 50 must get its own `min(steps, 50)`-step grace window,
    # not zero.
    early_refusal_steps_this_attempt = min(steps - start_step, _EARLY_REFUSAL_STEP_CAP)
    # Hashed once, not once per step: `stage_policy` is loop-invariant.
    stage_policy_hash_value = spolicy.stage_policy_hash(stage_policy)
    step = start_step
    try:
        batches = s2t.plan_batches(
            dataset, plan_order, start_step=start_step, steps=steps,
            batch_size=batch_size, num_workers=num_workers,
        )
        # Same timing fields as `s2_train.run_training`.
        waited_from = time.perf_counter()
        for samples in batches:
            data_wait_s = time.perf_counter() - waited_from
            step += 1
            lo, hi = (step - 1) * batch_size, step * batch_size
            indices = plan_order[lo:hi]
            batch = s1_train.collate_training_batch(
                [s.base for s in samples],
                ranges=model.ranges,
                image_token_id=image_token_id,
                device=device,
            )
            typed_batch = stl.collate_typed(samples, allowed_head_ids=allowed_head_ids)

            optimizer.zero_grad(set_to_none=True)
            hidden, prefix_len = model.encode_hidden(batch.model_batch)
            composite_hidden = hidden[:, prefix_len + batch.visual_steps :, :]
            logits = model.composite_head(composite_hidden)
            total, report = stl.s3_total_loss(
                logits=logits,
                target_ids=batch.target_ids,
                loss_mask=batch.loss_mask,
                loss_family_ids=batch.loss_family_ids,
                model=model,
                hidden=hidden,
                prefix_len=prefix_len,
                visual_steps=batch.visual_steps,
                typed_batch=typed_batch,
                stage_policy=stage_policy,
            )
            total.backward()
            grad_norm_groups = _grad_norm_groups(model)
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
            grad_norm_total = float(norm)
            if not bool(torch.isfinite(norm)):
                optimizer.zero_grad(set_to_none=True)
                raise s1_train.NonFiniteStepError(f"gradient norm is {grad_norm_total}")
            optimizer.step()

            for offset, idx in enumerate(indices):
                draw_index = lo + offset
                bundle, page = plan_pages[draw_index]
                ledger.append(
                    attempt_id=attempt["attempt_id"],
                    lineage_root_id=plan_lineages[draw_index],
                    cell=CELL,
                    view_ids=[f"{bundle}:{page}"],
                    sample_count=1,
                    optimizer_step=step,
                    plan_draw_index=draw_index,
                )

            for head_id, entry in report["typed_heads"].items():
                typed_supervised_seen[head_id] = (
                    typed_supervised_seen.get(head_id, 0) + entry["supervised_count"]
                )

            print(
                json.dumps(
                    {
                        "step": step,
                        "plan_draw_index": list(range(lo, hi)),
                        "families": report["families"],
                        "typed_heads": report["typed_heads"],
                        "typed_excluded": report["typed_excluded"],
                        "lambda": report["lambda"],
                        "stage_policy_hash": stage_policy_hash_value,
                        "grad_norm_total": grad_norm_total,
                        "grad_norm_groups": grad_norm_groups,
                        "data_wait_s": round(data_wait_s, 4),
                        "step_s": round(time.perf_counter() - waited_from, 4),
                    }
                ),
                flush=True,
            )

            steps_this_attempt = step - start_step
            if (
                steps_this_attempt >= early_refusal_steps_this_attempt
                and sum(typed_supervised_seen.values()) == 0
            ):
                # Finalized by the `except` clause below, exactly like any
                # other mid-loop failure -- this raise is not special-cased.
                raise S3TrainingRefused(
                    f"trained nothing typed after {steps_this_attempt} step(s) of "
                    f"THIS ATTEMPT (>= min(steps - start_step, "
                    f"{_EARLY_REFUSAL_STEP_CAP})={early_refusal_steps_this_attempt}) -- "
                    "an S3 typed run that has seen zero typed supervision is refused"
                )

            if out is not None and checkpoint_every and step % checkpoint_every == 0:
                _checkpoint(
                    out, model=model, optimizer=optimizer, step=step,
                    dataset=dataset, companion_hash=companion_hash, s3_run_id=s3_run_id,
                    ancestry_base=ancestry_base,
                )
                ledger.save_jsonl(run_dir / _LEDGER_FILENAME)
            waited_from = time.perf_counter()
        if out is not None:
            _checkpoint(
                out, model=model, optimizer=optimizer, step=step,
                dataset=dataset, companion_hash=companion_hash, s3_run_id=s3_run_id,
                ancestry_base=ancestry_base,
            )
            ledger.save_jsonl(run_dir / _LEDGER_FILENAME)
    except Exception:
        _finalize("failed")
        raise
    _finalize("completed")
    return ledger


def _checkpoint(
    out: Path,
    *,
    model: QwenDenseCandidateModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    dataset: stl.S3TrainingDataset,
    companion_hash: str,
    s3_run_id: str,
    ancestry_base: Mapping[str, Any] | None = None,
) -> None:
    s1_train.save_checkpoint(
        out,
        model=model,
        optimizer=optimizer,
        progress=s1_train.TrainingProgress(step=step),
        cache_identity=dataclasses.asdict(dataset.base.identity),
        s1_run_manifest_hash=companion_hash,
        feature_source=dataset.base.feature_source,
        ancestry_base=ancestry_base,
        run_id=s3_run_id,
    )
    print(f"[s3_train] checkpoint step {step} -> {out}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run S3 discovery admission steps 1-6 (design §7.3), build the model
    and optimizer, then run the training loop (`run_s3_training`, design
    §7.4) -- including `--resume` and the early typed refusal -- and the
    CLI (design §9).
    """
    import argparse

    from unflatten.model import composite_vocab as cv
    from unflatten.model import s2_plan as s2p
    from unflatten.model.candidate_loader import load_qwen_candidate

    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.s3_train", description=__doc__
    )
    parser.add_argument("cache", type=Path, help="exact-text cache directory")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lr-reduced", type=float, required=True)
    parser.add_argument("--grad-clip", type=float, required=True)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), required=True)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--preflight-rows", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="data prefetch worker processes; 0 = serial fetch in this process",
    )
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error(f"--num-workers must be >= 0, got {args.num_workers}")

    torch.manual_seed(args.seed)
    run_dir: Path = args.run_dir

    ancestry = json.loads((run_dir / "checkpoint_ancestry_base.json").read_text())
    ca.verify_base(ancestry)
    # The same guard s1_train/s2_train carry: a checkpoint under an S0
    # audited root would be classified as unaudited legacy ancestry by the
    # next S0 audit. Checked before any other file is read.
    if args.out:
        s1_train.refuse_out_under_audited_roots(args.out, ancestry)

    companion = json.loads((run_dir / "s1_run_manifest.json").read_text())
    allowlist = json.loads((run_dir / "qwen_text_allowlist.json").read_text())
    expectation = CacheExpectation.from_payload(
        json.loads((run_dir / "exact_text_cache_expectation.json").read_text())
    )

    # --- step 1: the S1 admission chain, over the companion manifest -----
    dataset = s1_train.ExactTextTrainingDataset(
        args.cache, gradient_bearing=True, expectation=expectation
    )
    identity = dataclasses.asdict(dataset.identity)
    s1_train.require_gradient_admissible(
        cache_identity=identity,
        run_manifest=companion,
        allowlist=allowlist,
        ancestry=ancestry,
        expectation=expectation,
        cli_lr=args.lr,
        cli_algorithm=args.optimizer,
        cli_batch_size=args.batch_size,
        cli_seed=args.seed,
        cli_steps=args.steps,
        control_arm=False,
    )
    require_grad_clip_bound(companion=companion, cli_grad_clip=args.grad_clip)

    # --- step 2: the S3 manifest is bound to this companion --------------
    s3_manifest = load_s3_run_manifest(run_dir / "s3_run_manifest.json")
    try:
        s2t.require_step2_bound_to_companion(s2_manifest=s3_manifest, companion=companion)
    except s2t.S2TrainingRefused as exc:
        raise S3TrainingRefused(str(exc)) from exc
    lr_reduced_micro = int(round(args.lr_reduced * 1e6))
    if lr_reduced_micro != s3_manifest["lr_reduced_micro"]:
        raise S3TrainingRefused(
            f"--lr-reduced {args.lr_reduced!r} (lr_reduced_micro {lr_reduced_micro}) "
            "does not match the frozen s3_run_manifest lr_reduced_micro "
            f"{s3_manifest['lr_reduced_micro']}"
        )

    # --- step 3: committed policies + s3 stage policy verify against it --
    matrix = load_stage_data_matrix(_POLICIES_DIR / "s2_stage_data_matrix.json")
    sampler_policy = load_s2_sampler_policy(_POLICIES_DIR / "s2_sampler_policy.json")
    exposure_policy = load_s2_exposure_policy(_POLICIES_DIR / "s2_exposure_policy.json")
    stage_policy = spolicy.load_stage_policy(_POLICIES_DIR / "s3_stage_policy.json")
    plan = json.loads((run_dir / "ordered_exposure_plan.json").read_text())
    try:
        s2t.require_step3_policies_bound(
            s2_manifest=s3_manifest,
            matrix=matrix,
            sampler_policy=sampler_policy,
            exposure_policy=exposure_policy,
            plan=plan,
        )
    except s2t.S2TrainingRefused as exc:
        raise S3TrainingRefused(str(exc)) from exc
    stage_policy_hash = spolicy.stage_policy_hash(stage_policy)
    if stage_policy_hash != s3_manifest["s3_stage_policy_hash"]:
        raise S3TrainingRefused(
            f"committed s3_stage_policy_hash ({stage_policy_hash}) does not match "
            f"the pinned s3_run_manifest value ({s3_manifest['s3_stage_policy_hash']})"
        )

    # --- step 4: every draw resolves inside the training dataset ---------
    try:
        s2t.require_plan_covers_budget(plan=plan, steps=args.steps, batch_size=args.batch_size)
        split_records = s2p.load_split_records(args.release_dir)
        plan_pages = s2t.resolve_plan_pages(
            plan=plan, expectation=expectation, split_records=split_records
        )
        plan_order = s2t.plan_dataset_order(plan_pages, expectation)
    except s2t.S2TrainingRefused as exc:
        raise S3TrainingRefused(str(exc)) from exc
    plan_lineages = [draw["lineage_root_id"] for draw in plan["draws"]]

    # --- step 5: typed supervision census / floor gate --------------------
    census = json.loads((run_dir / "typed_supervision_census.json").read_text())
    require_step5_typed_supervision(
        s3_manifest=s3_manifest,
        companion=companion,
        stage_policy=stage_policy,
        census=census,
    )

    # --- model build; step 6 ----------------------------------------------
    require_resume_init_from_agreement(
        cli_resume=args.resume,
        manifest_init_from=s3_manifest["init_from"],
        cli_init_from=args.init_from,
    )
    require_init_from_ancestry_lineage(
        ancestry=ancestry, init_from=s3_manifest["init_from"]
    )
    candidate = load_qwen_candidate(device=args.device)
    model = QwenDenseCandidateModel(
        decode=candidate.decode,
        inject_visual=candidate.inject_visual,
        visual_dim=candidate.visual_hidden_size,
        d_model=candidate.hidden_size,
        ranges=cv.packed_ranges(candidate.vocab_size),
        qwen_embedding_weight=candidate.embed_tokens.weight,
        qwen_lm_head_weight=candidate.lm_head_weight,
    )
    model.to(args.device)

    unfreeze_and_refreeze_typed_heads(model, s3_manifest["heads_frozen_for_no_supervision"])
    if args.init_from is not None:
        init_report = load_init_weights(
            args.init_from,
            model,
            pinned=s3_manifest["init_from"],
            expect_cache_identity=identity,
        )
        print(
            json.dumps(
                {
                    "init_from": {
                        "checkpoint_id": init_report.checkpoint_id,
                        "step": init_report.step,
                        "fresh_keys": list(init_report.fresh_keys),
                        "fresh_source": init_report.fresh_source,
                        "cross_release_fields": list(init_report.cross_release_fields),
                    }
                }
            ),
            flush=True,
        )
    require_step6_stage_state(
        model=model,
        run_class=s3_manifest["run_class"],
        heads_frozen_for_no_supervision=s3_manifest["heads_frozen_for_no_supervision"],
    )
    require_trainable_typed_half(s2t._model_group_states(model))

    # --- step 7: optimizer; --resume or fresh; preflight (design §7.3) ---
    optimizer = build_s3_optimizer(
        model,
        lr_standard=args.lr,
        lr_reduced=args.lr_reduced,
        algorithm=args.optimizer,
        run_class=s3_manifest["run_class"],
    )
    typed_dataset = stl.S3TrainingDataset(dataset)

    start_step = 0
    ledger = ex.ExposureLedger(s3_manifest["run_id"])
    if args.resume:
        # `--init-from` is never presented here (require_resume_init_from_
        # agreement already refused that combination): `load_checkpoint`
        # restores weights, optimizer and RNG in one step, so
        # `load_init_weights` is skipped entirely on the resume path.
        progress, _state = s1_train.load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            expect_cache_identity=identity,
            expect_feature_source=dataset.feature_source,
            expect_s1_run_manifest_hash=companion["s1_run_manifest_hash"],
            expect_base_ancestry_hash=ancestry["ancestry_hash"],
        )
        start_step = progress.step
        ledger = ex.ExposureLedger.load_jsonl(
            run_dir / _LEDGER_FILENAME, s3_manifest["run_id"]
        )
        try:
            s2t.require_resume_ledger_consistent(
                ledger=ledger, step=start_step, batch_size=args.batch_size
            )
        except s2t.S2TrainingRefused as exc:
            raise S3TrainingRefused(str(exc)) from exc
        # Design §5: the reduced LR is pinned in the S3 manifest and
        # re-asserted at resume by reading the restored optimizer's own
        # group 1 -- a checkpoint whose optimizer state disagrees (e.g. an
        # LR schedule that moved it) must not silently keep whatever it
        # carried.
        resumed_reduced_lr = optimizer.param_groups[1]["lr"]
        if resumed_reduced_lr != args.lr_reduced:
            raise S3TrainingRefused(
                f"resumed optimizer's reduced-LR group is at "
                f"{resumed_reduced_lr!r}, not --lr-reduced {args.lr_reduced!r} "
                "-- resume must reassert the reduced LR"
            )

    preflight_indices = s1_train.preflight_row_indices(len(dataset), args.preflight_rows)
    if preflight_indices:
        report = s1_train.preflight_native_signal(
            model,
            [typed_dataset[i].base for i in preflight_indices],
            image_token_id=candidate.image_token_id,
            device=args.device,
        )
        print(json.dumps({"preflight": report}), flush=True)
        s1_train.require_preflight_passes(report, gradient_bearing=True)

    run_s3_training(
        run_dir=run_dir,
        model=model,
        dataset=typed_dataset,
        optimizer=optimizer,
        image_token_id=candidate.image_token_id,
        plan_pages=plan_pages,
        plan_lineages=plan_lineages,
        plan_order=plan_order,
        companion_hash=companion["s1_run_manifest_hash"],
        s3_run_id=s3_manifest["run_id"],
        steps=args.steps,
        batch_size=args.batch_size,
        stage_policy=stage_policy,
        allowed_head_ids=l_typed_head_ids(stage_policy),
        grad_clip=args.grad_clip,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        out=args.out,
        start_step=start_step,
        ledger=ledger,
        ancestry_base=ancestry,
        integrity={
            "s3_run_manifest": s3_manifest["s3_run_manifest_hash"],
            "ordered_exposure_plan": plan["ordered_exposure_plan_hash"],
        },
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())

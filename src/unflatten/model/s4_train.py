"""S4 discovery optimizer construction, warm-start weight loading, and
admission (spec-s3-code-readiness design §5-§7). `main` runs admission
steps 1-6 (design §7.3), builds the model and optimizer, then runs the
training loop (`run_s4_training`, design §7.4) -- including `--resume` and
the early typed refusal -- and the CLI (design §9).

Steps 1-4 reuse `s2_train`'s admission functions VERBATIM (same field
names on the S4 manifest as on the S2 one): `require_step2_bound_to_companion`,
`require_step3_policies_bound`, `require_plan_covers_budget`,
`resolve_plan_pages`, `plan_dataset_order`. Their `S2TrainingRefused` is
re-raised here as `S4TrainingRefused` so a caller only ever sees this
module's own refusal type. Step 5 (the typed-supervision floor gate) and
step 6 (typed-head unfreeze/re-freeze + stage-state) are this module's own,
because they reason about fields no S2 run manifest carries.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import shutil
import signal
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from unflatten.dataset.augment import policy as aug_policy
from unflatten.model import aug_feature_cache as afc
from unflatten.model import checkpoint_ancestry as ca
from unflatten.model import checkpoint_eval_compare as cec
from unflatten.model import lr_schedule as lrs
from unflatten.model import onfly_visual as ov
from unflatten.model import s1_train
from unflatten.model import s2_exposure as ex
from unflatten.model import s2_train as s2t
from unflatten.model import s4_continuation as sc
from unflatten.model import s4_plan as s4p
from unflatten.model import s4_typed_loss as stl
from unflatten.model import stage_policy as spolicy
from unflatten.model import stage_state_policy as ssp
from unflatten.model import typed_head_config as thc
from unflatten.model.candidate_model import (
    QwenDenseCandidateModel,
    parameter_group_ids,
    typed_head_children,
)
from unflatten.model.capability_profile import (
    STANDALONE_PAGE_PROFILE_IDS,
    CapabilityProfileHashUnresolved,
    resolve_capability_profile_by_hash,
)
from unflatten.model.exact_text_cache import (
    CacheExpectation,
    PageKey,
    cache_manifest,
    page_key_set_sha256,
)
from unflatten.model.external_views import MAX_SHARE_MILLI, ExternalViews, ExternalViewsRefused
from unflatten.model.level_router.policy import (
    LevelRouterPolicy,
    LevelRouterPolicyError,
    load_level_router_policy,
)
from unflatten.model.level_router.ratification import (
    LevelRouterRatificationRequired,
    require_level_router_ratification,
)
from unflatten.model.level_router.variant_switch import (
    CHECKPOINT_POLICY_KEY as LEVEL_ROUTER_CHECKPOINT_KEY,
)
from unflatten.model.level_router.variant_switch import (
    level_router_checkpoint_payload,
    level_router_variant,
)
from unflatten.model.run_binding import compute_exact_text_cache_identity_hash
from unflatten.model.s2_sampler import load_s2_exposure_policy, load_s2_sampler_policy
from unflatten.model.s4_run_manifest import (
    S4RunManifestInvalid,
    data_quality_gate_hash_for,
    head_meets_floor,
    load_s4_run_manifest,
    new_pointer_supervised_counts,
    pointer_meets_floor,
)
from unflatten.model.s4_run_manifest import (
    _l_typed_head_ids as l_typed_head_ids,
)
from unflatten.model.s4_run_manifest import (
    _verify_census_against_companion as verify_census_against_companion,
)
from unflatten.model.stage_data_matrix import load_stage_data_matrix

#: The stage data matrix `s4_plan` freezes into the run dir (DEC-034: real
#: and synthetic lineages in separate cells, synthetic mass capped at 20 %).
_MATRIX_FILENAME = "stage_data_matrix.json"
_CELL_AXES = ("split", "pyramid_level", "task_kind", "cohort")

#: `s2_train`'s ledger filename convention, reused verbatim (design §7.4).
_LEDGER_FILENAME = s2t._LEDGER_FILENAME

#: How many optimizer steps of THIS run_s4_training call the early typed
#: refusal waits for before checking (design §7.4): "after min(steps, 50)
#: steps" -- never more than the run's own total budget.
_EARLY_REFUSAL_STEP_CAP = 50

#: The committed policy directory `main()` reads from by default
#: (overridable in tests via `monkeypatch.setattr(s4_train, "_POLICIES_DIR", ...)`,
#: the same pattern `s2_train._POLICIES_DIR` uses -- the two directories are
#: the SAME path in production, kept as separate module attributes so each
#: module's tests can monkeypatch independently).
_POLICIES_DIR = Path(__file__).parent / "policies"

#: The state_dict key prefix of the dense property-head child an S2
#: checkpoint always lacks (`candidate_model._DENSE_HEAD_GROUP_ATTR`,
#: design §4.5/§6, D6 = A). Not imported from `candidate_model` because
#: that name is private to it; this module only needs the prefix, which
#: is exactly the child's attribute name plus a dot.


class S4TrainingRefused(RuntimeError):
    """`build_s4_optimizer` refused to build the optimizer: a
    `requires_grad` parameter with no group mapping, or a mapped group
    whose S4 policy state for `run_class` does not allow `trainable`
    (design §5). Also raised by `load_init_weights` for any of its six
    ordered refusals (design §6)."""


class S4OverfitStopped(RuntimeError):
    """`--stop-on-overfit` ended the run early: `OVERFIT_SUSPECTED` fired
    `overfit_patience` diagnostic evals in a row (DEC-041 point 5). This
    is an explicit, owner-chosen DEVIATION from RFC005 §12.2 -- which
    reserves early stopping to `free_running` metrics -- never the
    default: `run_s4_training` only ever raises this when the caller
    passed `stop_on_overfit=True`."""


@dataclasses.dataclass(frozen=True)
class InitReport:
    """What `load_init_weights` actually did, for the caller's manifest /
    attempt record: the verified `checkpoint_id` and the `step` the parent
    checkpoint pinned.

    Three legal `fresh_source` values (design §7 rev 3, §10.1 T8):

    - `"carried_from_s3_parent"`: every model key was found in the parent
      (`missing == unexpected == ()`) -- S4's parent is S3, which trained
      the dense heads, so this is the common case; `fresh_keys` is empty.
    - `"absent_in_parent"`: the ONLY missing keys are
      `declared_new_keys(model)` (an entirely new head module the parent
      never carried at all, e.g. a future T2/T4 head) -- `fresh_keys`
      names them.
    - `"carried_from_s3_parent_transplanted_to_slots"`: the parent still
      carries the OLD, pre-`SlotHead` single-instance tensor for one or
      more of `typed_head_config.STAR_POINTERS`; each is broadcast
      (`copy_`) into every slot of the new `SlotHead` (T3) -- bitwise
      identical in every slot, never re-initialized. `replaced_keys` then
      carries one ledger row per replaced parent key
      (`{parent_key, slot_keys, rule: "broadcast", parent_shape}`,
      design §7's `init_from.replaced_keys`); `fresh_keys` still only
      names `declared_new_keys(model)`, if any.

    Any other `missing`/`unexpected` shape is refused by
    `load_init_weights` before this dataclass is ever constructed.
    """

    checkpoint_id: str
    step: int
    fresh_keys: tuple[str, ...]
    fresh_source: str
    replaced_keys: tuple[dict[str, Any], ...] = ()


def declared_new_keys(model: QwenDenseCandidateModel) -> frozenset[str]:
    """`model.state_dict()` keys under a `typed_head_config.
    NEW_HEAD_KEY_PREFIXES` prefix -- an entirely new head module (a
    future T2/T4 head) the S3 parent never carried AT ALL, distinct from
    a `replaced_keys` slot key (which the parent DOES carry, under its
    pre-`SlotHead` name). Empty until `NEW_HEAD_KEY_PREFIXES` grows past
    its current empty tuple (design §7 rev 3, §10.1 T8 point 1)."""
    prefixes = thc.NEW_HEAD_KEY_PREFIXES
    # RFC007 (DEC-047 dormant): a router-bearing v16 model warm-started
    # from the v15b winner -- which never carried the router -- starts the
    # router fresh (`absent_in_parent`). Empty for every model without one.
    if getattr(model, "level_router", None) is not None:
        prefixes = (*prefixes, "level_router.")
    if not prefixes:
        return frozenset()
    return frozenset(
        key for key in model.state_dict() if any(key.startswith(p) for p in prefixes)
    )


def replaced_keys(model: QwenDenseCandidateModel) -> dict[str, list[str]]:
    """`{parent_key: [slot_key, ...]}` for every `typed_head_config.
    STAR_POINTERS` entry whose child attribute on `model` ALREADY exposes
    the T3 `SlotHead` naming convention (an `nn.ModuleList` named
    `typed_head_config.SLOT_HEAD_LIST_ATTRIBUTE`, design's own choice,
    §10.1 T3/T8) -- derived purely from `model.state_dict()` key strings,
    never from importing a `SlotHead` class (T3 has not landed).

    For a star pointer whose child attribute is still the pre-T3
    single-instance head (today's model), no `"...slots.<i>..."` key
    exists to find, so that entry contributes nothing here -- this
    function returns an empty mapping against today's `QwenDenseCandidateModel`
    and only starts reporting entries once T3 builds `SlotHead` to match.
    """
    state_keys = set(model.state_dict())
    result: dict[str, list[str]] = {}
    for entry in thc.STAR_POINTERS:
        cap = getattr(thc, entry["cap_attribute"])
        prefix = (
            f"{entry['child_attribute']}.{entry['pointer_attribute']}."
            f"{thc.SLOT_HEAD_LIST_ATTRIBUTE}."
        )
        by_suffix: dict[str, dict[int, str]] = {}
        for key in state_keys:
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            index_str, sep, suffix = rest.partition(".")
            if not sep or not index_str.isdigit():
                continue
            by_suffix.setdefault(suffix, {})[int(index_str)] = key
        for suffix, by_index in by_suffix.items():
            if set(by_index) != set(range(cap)):
                # An incomplete slot run (not every 0..cap-1 index present)
                # is not the T3 convention this function recognises --
                # left for `load_init_weights`'s generic refusal to catch
                # via a `missing`/`unexpected` shape that matches none of
                # the three legal states, never silently patched over here.
                continue
            parent_key = f"{entry['child_attribute']}.{entry['pointer_attribute']}.{suffix}"
            result[parent_key] = [by_index[i] for i in range(cap)]
    return result


def load_init_weights(
    path: Path | str,
    model: QwenDenseCandidateModel,
    *,
    pinned: Mapping[str, Any],
    expect_cache_identity: Mapping[str, Any],
) -> InitReport:
    """Warm-start `model`'s weights from the S3 checkpoint at `path`
    (design §6) -- weights only, never `--resume`: no optimizer state, no
    RNG restore, and this function takes no optimizer at all.

    `pinned` is the S4 manifest's `init_from` block: `checkpoint_sha256`,
    `checkpoint_id`, `parent_s1_run_manifest_hash`,
    `parent_base_ancestry_hash`, `step`. Refusals run in this order, each
    raising `S4TrainingRefused` naming the failed check:

    1. `path`'s actual file bytes hash to `pinned["checkpoint_sha256"]`.
    2. The checkpoint's sidecar verifies (`checkpoint_ancestry.verify_sidecar`)
       against `parent_base_ancestry_hash` / `parent_s1_run_manifest_hash` /
       the embedded `state["ancestry_hash"]`, and the checkpoint_id
       recomputed from the verified ancestry hash and the file's sha256
       equals `pinned["checkpoint_id"]`.
    3. `state["feature_source"] == "cache"` -- never a null-model checkpoint.
    4. `state["cache_identity"] == expect_cache_identity`.
    5. `state.get("typed_head_config_hash")`, if present, equals the
       CURRENT `typed_head_config.typed_head_config_hash()` (design §9
       R12) -- absent (a legacy checkpoint, e.g. the real
       `s4-discovery-v14-r4/checkpoint.pt`) makes no claim and is not
       refused on this ground alone.
    6. `model.load_state_dict(state["model_state_dict"], strict=False)`'s
       `missing`/`unexpected` keys match exactly ONE of three legal states
       (design §7 rev 3, `InitReport`'s own docstring): every key carried
       (`"carried_from_s3_parent"`); a nonempty subset of
       `declared_new_keys(model)` missing (`"absent_in_parent"`); or
       `unexpected` is exactly
       `replaced_keys(model)`'s parent-key set and `missing` is exactly
       `declared_new_keys(model) | {every slot key}`
       (`"carried_from_s3_parent_transplanted_to_slots"` -- the old,
       pre-`SlotHead` single-instance tensor is broadcast into every slot,
       R14a/b/c/d). Any other shape is refused -- an S3 parent that
       neither carries every key nor exactly one of these two named
       shortfalls is not a warm start this function recognises. (The S3
       trainer reinitializes heads a frozen S2 parent carried; copying
       that rule here discarded S3's typed-head training in the first
       v14-r4 launch.)
    """
    path = Path(path)
    actual_sha256 = ca.file_sha256(path)
    if actual_sha256 != pinned["checkpoint_sha256"]:
        raise S4TrainingRefused(
            f"load_init_weights: checkpoint sha256 {actual_sha256!r} != pinned "
            f"{pinned['checkpoint_sha256']!r} -- refusing to warm-start from a "
            "checkpoint whose bytes do not match the S4 manifest's init_from pin"
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
        raise S4TrainingRefused(
            f"load_init_weights: checkpoint ancestry sidecar refused: {exc}"
        ) from exc

    checkpoint_id = ca.compute_checkpoint_id(doc["ancestry_hash"], actual_sha256)
    if checkpoint_id != pinned["checkpoint_id"]:
        raise S4TrainingRefused(
            f"load_init_weights: recomputed checkpoint_id {checkpoint_id!r} != "
            f"pinned {pinned['checkpoint_id']!r}"
        )

    if state.get("feature_source") != "cache":
        raise S4TrainingRefused(
            f"load_init_weights: checkpoint feature_source={state.get('feature_source')!r} "
            "!= 'cache' -- a null-model checkpoint cannot warm-start S4"
        )

    if dict(expect_cache_identity) != state.get("cache_identity"):
        raise S4TrainingRefused(
            "load_init_weights: checkpoint cache_identity != expect_cache_identity "
            "-- warm-starting onto a different cache is a different run"
        )

    # 5. `typed_head_config_hash` (design §7 rev 3, §9 R12): a LEGACY
    #    checkpoint predating this field (e.g. the real
    #    `s4-discovery-v14-r4/checkpoint.pt` this task's warm start targets)
    #    carries no key at all and makes no claim to compare -- refusing it
    #    on that ground would make the very warm start this design exists
    #    for impossible. A checkpoint that DOES carry the key is held to it:
    #    a present but DIFFERENT hash is refused, never silently
    #    reinterpreted under today's palette/caps.
    checkpoint_thc_hash = state.get("typed_head_config_hash")
    current_thc_hash = thc.typed_head_config_hash()
    if checkpoint_thc_hash is not None and checkpoint_thc_hash != current_thc_hash:
        raise S4TrainingRefused(
            f"load_init_weights: checkpoint typed_head_config_hash "
            f"{checkpoint_thc_hash!r} != current {current_thc_hash!r} -- warm-"
            "starting typed heads under a different palette/cap configuration "
            "is refused, not silently reinterpreted"
        )

    declared_new = declared_new_keys(model)
    replaced = replaced_keys(model)
    replaced_parent_keys = frozenset(replaced)
    replaced_slot_keys = frozenset(
        slot_key for slot_keys in replaced.values() for slot_key in slot_keys
    )

    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    missing_set, unexpected_set = frozenset(missing), frozenset(unexpected)

    if not unexpected_set and not missing_set:
        return InitReport(
            checkpoint_id=pinned["checkpoint_id"],
            step=pinned["step"],
            fresh_keys=(),
            fresh_source="carried_from_s3_parent",
        )
    if not unexpected_set and missing_set and missing_set <= declared_new:
        return InitReport(
            checkpoint_id=pinned["checkpoint_id"],
            step=pinned["step"],
            fresh_keys=tuple(sorted(missing_set)),
            fresh_source="absent_in_parent",
        )
    if (
        replaced_parent_keys
        and unexpected_set == replaced_parent_keys
        and missing_set == declared_new | replaced_slot_keys
    ):
        return InitReport(
            checkpoint_id=pinned["checkpoint_id"],
            step=pinned["step"],
            fresh_keys=tuple(sorted(declared_new)),
            fresh_source="carried_from_s3_parent_transplanted_to_slots",
            replaced_keys=_transplant_replaced_keys(
                model, parent_state=state["model_state_dict"], replaced=replaced
            ),
        )

    raise S4TrainingRefused(
        f"load_init_weights: model.load_state_dict reports missing="
        f"{sorted(missing_set)!r} unexpected={sorted(unexpected_set)!r}, which "
        "matches none of the three legal warm-start states (carried_from_"
        "s3_parent, absent_in_parent, carried_from_s3_parent_transplanted_"
        "to_slots) -- an S3 parent that neither carries every key nor has "
        "only the recognised new/replaced keys absent is refused, never a "
        "silently partial load"
    )


def _transplant_replaced_keys(
    model: QwenDenseCandidateModel,
    *,
    parent_state: Mapping[str, torch.Tensor],
    replaced: Mapping[str, list[str]],
) -> tuple[dict[str, Any], ...]:
    """Broadcast (`copy_`) each `parent_key`'s single-instance tensor into
    every one of its `slot_keys` (design §7 rev 3: "the old head trained a
    mix of all indices ... the marginal is the correct start of every
    slot"), and return the `init_from.replaced_keys` ledger rows.

    Every shape is checked BEFORE any tensor is mutated (R14d): a shape
    mismatch names `parent_shape` in the refusal, never surfaces as a bare
    `copy_` `RuntimeError` partway through.
    """
    target_state = model.state_dict()
    for parent_key, slot_keys in replaced.items():
        parent_shape = list(parent_state[parent_key].shape)
        for slot_key in slot_keys:
            slot_shape = list(target_state[slot_key].shape)
            if slot_shape != parent_shape:
                raise S4TrainingRefused(
                    f"load_init_weights: replaced_keys transplant refused -- "
                    f"parent {parent_key!r} parent_shape={parent_shape} but slot "
                    f"{slot_key!r} has shape {slot_shape}"
                )

    ledger: list[dict[str, Any]] = []
    for parent_key, slot_keys in replaced.items():
        parent_tensor = parent_state[parent_key]
        parent_shape = list(parent_tensor.shape)
        with torch.no_grad():
            for slot_key in slot_keys:
                target_state[slot_key].copy_(parent_tensor)
        ledger.append(
            {
                "parent_key": parent_key,
                "slot_keys": list(slot_keys),
                "rule": "broadcast",
                "parent_shape": parent_shape,
            }
        )
    return tuple(ledger)


def build_s4_optimizer(
    model: QwenDenseCandidateModel,
    *,
    lr_standard: float,
    lr_reduced: float,
    algorithm: str,
    run_class: str | None,
) -> torch.optim.Optimizer:
    """Two param groups over `model`'s `requires_grad` parameters --
    `param_groups[0]` at `lr_standard`, `param_groups[1]` at `lr_reduced` --
    split by `stage_state_policy.stage_lr_class("s4", group_id,
    run_class=run_class)` over `candidate_model.parameter_group_ids(model)`
    (design §5). Mirrors `s1_runner.build_s1_optimizer`'s construction
    (supported algorithms, the empty-parameter-set refusal); unlike it,
    this function must also refuse per-parameter, because S4 trains a
    STRICT SUBSET of `model.parameters()` chosen by policy rather than
    everything `requires_grad` reports:

    - a `requires_grad` parameter `parameter_group_ids` does not map to any
      group (an unaccounted trainable tensor an optimizer built from
      `model.parameters()` would otherwise silently train);
    - a mapped group whose S4 policy state for `run_class` is not
      `trainable` (`stage_lr_class` raises `KeyError`/`ValueError` for
      that; both are re-raised as `S4TrainingRefused` with the parameter
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
            raise S4TrainingRefused(
                f"parameter {name!r} requires grad but "
                "candidate_model.parameter_group_ids maps it to no "
                "trainable group -- an optimizer built from it would "
                "silently train an unaccounted parameter"
            )
        try:
            lr_class = ssp.stage_lr_class("s4", group_id, run_class=run_class)
        except (KeyError, ValueError) as exc:
            raise S4TrainingRefused(
                f"parameter {name!r} maps to group {group_id!r}, which the "
                f"S4 policy (run_class={run_class!r}) does not allow "
                f"trainable: {exc}"
            ) from exc
        (standard if lr_class == ssp.STANDARD else reduced).append(param)

    if not standard and not reduced:
        raise ValueError(
            "model reports no trainable parameters -- an optimizer over an "
            "empty parameter set would make the S4 loop vacuous"
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
    s4_manifest: Mapping[str, Any],
    companion: Mapping[str, Any],
    stage_policy: Mapping[str, Any],
    census: Mapping[str, Any],
) -> None:
    """The typed-supervision census bound to THIS run -- never trusted from
    the manifest alone: re-verified against the presented companion
    (`s4_run_manifest._verify_census_against_companion`) and against the
    manifest's own pinned `typed_supervision_census_hash` -- and the floor
    gate over every `L_typed` head the committed stage policy declares:

    - if every `L_typed` head is listed in `heads_frozen_for_no_supervision`,
      refused outright (a typed run that trains no typed head);
    - a head NOT listed there must MEET both `head_meets_floor` dimensions
      (`s4_run_manifest.head_meets_floor`, the one floor predicate -- never
      re-implemented here);
    - a head listed there must NOT meet them (an unused freeze is refused
      too, the DEC-027 D2 rule, re-checked here even though the manifest
      builder already enforces it -- the manifest and the census presented
      to a run are not necessarily the ones the builder saw).
    """
    try:
        census_hash = verify_census_against_companion(census, companion)
    except S4RunManifestInvalid as exc:
        raise S4TrainingRefused(f"typed supervision census refused: {exc}") from exc
    if census_hash != s4_manifest["typed_supervision_census_hash"]:
        raise S4TrainingRefused(
            f"typed supervision census_hash {census_hash!r} does not match the "
            "pinned s4_run_manifest typed_supervision_census_hash "
            f"{s4_manifest['typed_supervision_census_hash']!r}"
        )

    floor = s4_manifest["typed_supervision_floor"]
    frozen = set(s4_manifest["heads_frozen_for_no_supervision"])
    head_ids = l_typed_head_ids(stage_policy)
    if head_ids and frozen >= head_ids:
        raise S4TrainingRefused(
            "every L_typed head is listed in heads_frozen_for_no_supervision -- "
            "a typed run that trains no typed head"
        )

    per_head = census["counts"]["per_head"]
    for head_id in sorted(head_ids):
        entry = per_head.get(head_id, {})
        meets_floor = head_meets_floor(entry, floor)
        if head_id in frozen:
            if meets_floor:
                raise S4TrainingRefused(
                    f"{head_id!r} is listed in heads_frozen_for_no_supervision but "
                    "its census entry meets the supervision floor -- an unused "
                    "freeze is a lie too (DEC-027 D2 rule)"
                )
        elif not meets_floor:
            raise S4TrainingRefused(
                f"{head_id!r} is not listed in heads_frozen_for_no_supervision but "
                f"its census entry (supervised={entry.get('supervised', 0)}, "
                f"lineage_count={entry.get('lineage_count', 0)}) does not meet the "
                f"supervision floor {dict(floor)!r}"
            )

    # Design §9 R8, §10.1 T8 point 4: the per-POINTER supervision
    # disclosure, re-checked here for the same reason as the head-level
    # floor above -- the manifest and the census presented to a live run
    # are not necessarily the ones the builder saw. Unlike the head-level
    # check, this one gates no parameter (see s4_run_manifest's docstring
    # for why "disclosed", not "frozen").
    disclosed_pointers = set(s4_manifest["low_supervision_pointers_disclosed"])
    pointer_counts = new_pointer_supervised_counts(census)
    for key in sorted(pointer_counts):
        supervised = pointer_counts[key]
        pointer_meets = pointer_meets_floor(supervised, floor)
        if key in disclosed_pointers:
            if pointer_meets:
                raise S4TrainingRefused(
                    f"{key!r} is listed in low_supervision_pointers_disclosed but "
                    "its census supervised count meets min_supervised_per_pointer "
                    "-- a stale disclosure is a lie too (design §9 R8)"
                )
        elif not pointer_meets:
            raise S4TrainingRefused(
                f"{key!r} is not listed in low_supervision_pointers_disclosed but "
                f"its census supervised count ({supervised}) does not meet "
                f"min_supervised_per_pointer ({floor['min_supervised_per_pointer']}) "
                "(design §9 R8)"
            )


# ---------------------------------------------------------------------------
# Step 6 -- typed-head unfreeze/re-freeze, stage-state, trainable typed half
# (design §5, §7.3 step 6)
# ---------------------------------------------------------------------------


def unfreeze_and_refreeze_typed_heads(
    model: QwenDenseCandidateModel, heads_frozen_for_no_supervision: Sequence[str]
) -> None:
    """Unfreeze every typed-head child, then re-freeze exactly the ones the
    S4 manifest lists as `heads_frozen_for_no_supervision` -- the order
    matters: a group NOT listed must end up trainable even though S1 froze
    it at construction. The `node_visual_aligner` child is S4's own
    alignment head (not a typed head): it is always unfrozen here, since
    `L_alignment` is an unconditional S4 family (design §5)."""
    frozen = set(heads_frozen_for_no_supervision)
    for child_name, head_id in typed_head_children().items():
        getattr(model, child_name).requires_grad_(head_id not in frozen)
    model.node_visual_aligner.requires_grad_(True)


def require_stage_states(
    states: Mapping[str, str],
    *,
    run_class: str,
    heads_frozen_for_no_supervision: Sequence[str],
) -> None:
    """Pure over a `{group_id: state}` map (design §7.3 step 6): every
    group must be a state `stage_state_policy` allows for `"s4"` under
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
            ssp.validate_stage_state("s4", group_id, state, run_class=run_class)
        except ValueError as exc:
            raise S4TrainingRefused(f"stage-state check refused: {exc}") from exc
        if group_id.startswith("typed_heads.") and state == "frozen" and group_id not in frozen:
            raise S4TrainingRefused(
                f"{group_id!r} is frozen but is not listed in "
                "heads_frozen_for_no_supervision -- the S4 discovery policy admits "
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
        raise S4TrainingRefused(
            "no typed_heads.* group is trainable -- an S4 typed run must train "
            "at least one typed head (design §7.3 step 6)"
        )


def require_level_router_admission(
    *,
    s4_manifest: Mapping[str, Any],
    policy_path: Path | None,
    ratification_record: Path | None,
) -> LevelRouterPolicy | None:
    """RFC007 variant B (PROPOSED, DEC-047 dormant): `None` -- today's dense
    model -- unless BOTH the frozen manifest and the CLI name the same
    level-router policy, AND the owner's ratification record for exactly that
    policy and stage S4 exists. Checked on every invocation, `--resume` and
    `--admission-only` included, the same way `mask_inactive_union_arms` is.
    """
    policy = None
    if policy_path is not None:
        try:
            policy = load_level_router_policy(policy_path)
        except LevelRouterPolicyError as exc:
            raise S4TrainingRefused(f"--level-router-policy: {exc}") from exc
    cli_hash = policy.policy_hash() if policy is not None else None
    manifest_hash = s4_manifest.get("level_router_policy_hash")
    if cli_hash != manifest_hash:
        raise S4TrainingRefused(
            f"--level-router-policy (policy_hash {cli_hash!r}) does not match the frozen "
            f"s4_run_manifest level_router_policy_hash {manifest_hash!r} -- this run "
            "(including a --resume) must be launched with the router it was frozen with"
        )
    if policy is None:
        return None
    try:
        require_level_router_ratification(
            stage="S4", policy=policy, ratification_record=ratification_record
        )
    except LevelRouterRatificationRequired as exc:
        raise S4TrainingRefused(str(exc)) from exc
    return policy


# ---------------------------------------------------------------------------
# Grad clip and init_from CLI/manifest agreement (design §5, §6)
# ---------------------------------------------------------------------------


def require_grad_clip_bound(*, companion: Mapping[str, Any], cli_grad_clip: float) -> None:
    """`run_config.json`'s (the companion's) `optimizer.grad_clip_micro` is
    the ONE authoritative field for the grad clip (design §5, [governance
    R1]); S4 refuses a null value -- the S2 gap this design's §0 finding
    3 records -- and a CLI value that disagrees with it."""
    grad_clip_micro = companion["run_config"]["optimizer"]["grad_clip_micro"]
    if grad_clip_micro is None:
        raise S4TrainingRefused(
            "companion run_config.optimizer.grad_clip_micro is null -- S4 refuses "
            "an unpinned grad clip (design §5, governance R1)"
        )
    cli_micro = int(round(cli_grad_clip * 1e6))
    if cli_micro != grad_clip_micro:
        raise S4TrainingRefused(
            f"--grad-clip {cli_grad_clip!r} (grad_clip_micro {cli_micro}) does not "
            f"match the frozen run_config.optimizer.grad_clip_micro {grad_clip_micro}"
        )


def require_init_from_cli_manifest_agreement(
    *, manifest_init_from: Mapping[str, Any] | None, cli_init_from: Path | None
) -> None:
    """`--init-from` and the S4 manifest's `init_from` must agree on
    PRESENCE in both directions (design §6) -- neither may silently win."""
    if manifest_init_from is None and cli_init_from is not None:
        raise S4TrainingRefused(
            "--init-from was supplied but the S4 manifest's init_from is null"
        )
    if manifest_init_from is not None and cli_init_from is None:
        raise S4TrainingRefused(
            "the S4 manifest pins init_from but --init-from was not supplied"
        )


def require_init_from_ancestry_lineage(
    *, ancestry: Mapping[str, Any], init_from: Mapping[str, Any] | None
) -> None:
    """ERRATA-004 E6 lineage cross-check between `checkpoint_ancestry_base.json`
    and the S4 manifest's `init_from` (design §6): the two must agree on
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
            raise S4TrainingRefused(
                "s4_run_manifest init_from is null but checkpoint_ancestry_base "
                f"initialization_kind is {initialization_kind!r}, not "
                "'clean_pretrained'"
            )
        return

    if initialization_kind != "warm_start":
        raise S4TrainingRefused(
            "s4_run_manifest pins init_from but checkpoint_ancestry_base "
            f"initialization_kind is {initialization_kind!r}, not 'warm_start'"
        )
    parents = ancestry.get("parents") or []
    if not parents:
        raise S4TrainingRefused(
            "checkpoint_ancestry_base initialization_kind is warm_start but "
            "carries no parents[0] entry"
        )
    parent = parents[0]
    if parent["checkpoint_id"] != init_from["checkpoint_id"]:
        raise S4TrainingRefused(
            f"checkpoint_ancestry_base parents[0].checkpoint_id "
            f"({parent['checkpoint_id']!r}) != s4_run_manifest init_from.checkpoint_id "
            f"({init_from['checkpoint_id']!r})"
        )
    if parent["checkpoint_sha256"] != init_from["checkpoint_sha256"]:
        raise S4TrainingRefused(
            f"checkpoint_ancestry_base parents[0].checkpoint_sha256 "
            f"({parent['checkpoint_sha256']!r}) != s4_run_manifest "
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
      checkpoint's own ancestry against the S4 base (warm-start) ancestry,
      which is where that lineage is actually re-checked.
    - With no `--resume`, this delegates to
      :func:`require_init_from_cli_manifest_agreement` unchanged -- the
      original four-way PRESENCE agreement still applies to a fresh run.
    """
    if cli_resume is not None and cli_init_from is not None:
        raise S4TrainingRefused(
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


def plan_cells(plan: Mapping[str, Any]) -> list[tuple[str, str, str, str]]:
    """The matrix cell of every plan draw, in draw order -- what the
    exposure ledger records, so synthetic draws stay distinguishable."""
    return [tuple(draw[axis] for axis in _CELL_AXES) for draw in plan["draws"]]  # type: ignore[misc]


def load_run_matrix(run_dir: Path) -> dict[str, Any]:
    """The stage data matrix frozen into `run_dir` by `s4_plan`."""
    path = run_dir / _MATRIX_FILENAME
    if not path.exists():
        raise S4TrainingRefused(
            f"{path} is missing -- freeze it with `python -m unflatten.model.s4_plan`"
        )
    return load_stage_data_matrix(path)


def require_story_weight_standalone_profile(
    *, story_decision_weight_milli: int, cache_identity: Mapping[str, Any]
) -> None:
    """RFC004-ERRATA-012 E3: the story-decision mask finds its positions from the
    standalone-page framing. The legacy windowed framing
    (`FRAGMENT_BEGIN, RESUME_NODE(document), SECTION_OPEN`) yields zero positions,
    so `--story-decision-weight != 1` would silently do nothing -- refused."""
    if story_decision_weight_milli == 1000:
        return
    try:
        profile = resolve_capability_profile_by_hash(
            cache_identity.get("capability_profile_hash")
        )
    except CapabilityProfileHashUnresolved as exc:
        raise S4TrainingRefused(
            f"--story-decision-weight: cache capability profile unresolved: {exc}"
        ) from exc
    if profile.profile_id not in STANDALONE_PAGE_PROFILE_IDS:
        raise S4TrainingRefused(
            f"--story-decision-weight (story_decision_weight_milli "
            f"{story_decision_weight_milli}) needs a standalone-page cache profile "
            f"(one of {sorted(STANDALONE_PAGE_PROFILE_IDS)}); the cache is bound to "
            f"{profile.profile_id!r}, whose framing yields no story decision positions "
            "-- the lever would be silently inert"
        )


_BOOST_BINDING_KEYS = ("sets_sha256", "factor", "draws")


def require_boost_block_bound(
    *, page_uniform_meta: Mapping[str, Any] | None, s4_manifest: Mapping[str, Any]
) -> None:
    """RFC005-ERRATA-014 E3 / design §4.3: the page-uniform sidecar sits outside
    the hashed plan, so its `boost` block (sets file sha, factor, draw count) is
    compared with the manifest's `page_uniform_boost`. Absent on both sides is
    the no-boost run; present on one side only, or unequal, is refused."""
    sidecar_boost = None if page_uniform_meta is None else page_uniform_meta.get("boost")
    bound = (
        None
        if sidecar_boost is None
        else {k: sidecar_boost.get(k) for k in _BOOST_BINDING_KEYS}
    )
    frozen = s4_manifest.get("page_uniform_boost")
    if bound != frozen:
        raise S4TrainingRefused(
            f"the page-uniform sidecar boost block {bound!r} does not match the frozen "
            f"s4_run_manifest page_uniform_boost {frozen!r} -- the sidecar and the "
            "manifest were not frozen together"
        )


def resolve_plan_pages_for_run(
    *,
    page_uniform_meta: Mapping[str, Any] | None,
    plan: Mapping[str, Any],
    expectation: CacheExpectation,
    split_records: Sequence[Mapping[str, Any]],
) -> tuple[list[PageKey], list[str] | None, list[str | None] | None]:
    """The page (and, in page-uniform mode, the clean/augmented decision)
    for every draw, in draw order (T1b, DEC-041).

    With a page-uniform sidecar (`s4_plan.build_page_uniform_plan`'s
    `s4_page_uniform_plan.json`), every draw's page comes from it
    VERBATIM, along with its `kind`/`view_id` -- `s2_plan.select_page`'s
    RNG pick over a lineage's own page list is never consulted, because
    it is a per-draw hash choice that cannot express "every page exactly
    N times" for a multi-page document (a document holding hundreds of
    pages would otherwise see the same handful repeatedly and the rest
    never). Without a sidecar (legacy/weighted mode), resolution is
    exactly `s2_train.resolve_plan_pages`, unchanged, and the returned
    `plan_view_kind`/`plan_view_id` are both `None`.
    """
    if page_uniform_meta is None:
        plan_pages = s2t.resolve_plan_pages(
            plan=plan, expectation=expectation, split_records=split_records
        )
        return plan_pages, None, None
    meta_draws = page_uniform_meta["draws"]
    plan_draws = plan["draws"]
    # SHOULD FIX 3 (review): the sidecar sits outside the shared, hashed
    # plan schema -- these asserts are what stands in for a binding hash.
    # A sidecar swapped for an unrelated run's, or one whose draws drifted
    # out of order relative to the ledger-bound `ordered_exposure_plan.json`
    # it was frozen alongside, is refused here rather than silently served.
    if len(meta_draws) != len(plan_draws):
        raise S4TrainingRefused(
            f"page-uniform sidecar has {len(meta_draws)} draw(s) but the plan has "
            f"{len(plan_draws)} -- they were not frozen together"
        )
    page_lineage = s4p.build_page_lineage_map(expectation, split_records)
    for i, (meta_draw, plan_draw) in enumerate(zip(meta_draws, plan_draws, strict=True)):
        if meta_draw["draw_index"] != i or plan_draw["draw_index"] != i:
            raise S4TrainingRefused(
                f"draw {i}: sidecar draw_index {meta_draw['draw_index']!r} / plan "
                f"draw_index {plan_draw['draw_index']!r} does not match its position"
            )
        page: PageKey = (meta_draw["document_bundle_id"], meta_draw["page_index"])
        lineage = page_lineage.get(page)
        if lineage != plan_draw["lineage_root_id"]:
            raise S4TrainingRefused(
                f"draw {i}: sidecar page {page!r} belongs to lineage {lineage!r}, "
                f"not the plan's lineage_root_id {plan_draw['lineage_root_id']!r}"
            )
    plan_pages = [(d["document_bundle_id"], d["page_index"]) for d in meta_draws]
    plan_view_kind = [d["kind"] for d in meta_draws]
    plan_view_id = [d["view_id"] for d in meta_draws]
    return plan_pages, plan_view_kind, plan_view_id


#: The pinned eval subset, frozen into the run dir the first time an eval
#: fires (review MUST FIX 2) -- present on every resume, never reselected.
_EVAL_PAGES_FILENAME = "s4_eval_pages.json"


def select_eval_pages(
    *,
    pages: Sequence[PageKey],
    page_lineage: Mapping[PageKey, str],
    eval_pages: int,
    run_seed: int,
) -> list[PageKey]:
    """A deterministic, seeded subset of `pages` of size `min(eval_pages,
    len(pages))` (review MUST FIX 2): scoring the WHOLE val_iid cache every
    `--eval-every` steps (10 000+ pages x ~40 MB) would make one eval take
    days, over ~237 evals in a production run. Pages are grouped by
    `page_lineage` and drawn ROUND-ROBIN across lineages (one page per
    lineage per round, each lineage's own page order seeded and shuffled)
    so the subset spreads across documents rather than exhausting a single
    long one -- "a handful of pages per lineage over many pages of one
    document" (review's own wording). `val_iid` is lineage-disjoint from
    train by construction (the release split), so this selection is a
    lineage-disjoint cohort with no further filtering needed here.
    """
    by_lineage: dict[str, list[PageKey]] = {}
    for page in pages:
        by_lineage.setdefault(page_lineage[page], []).append(page)
    rng = random.Random(f"s4-eval-pages/{run_seed}")
    for lineage in by_lineage:
        rng.shuffle(by_lineage[lineage])
    lineage_order = sorted(by_lineage)
    rng.shuffle(lineage_order)
    queues = [by_lineage[lineage] for lineage in lineage_order]
    selected: list[PageKey] = []
    round_index = 0
    while len(selected) < min(eval_pages, len(pages)):
        progressed = False
        for queue in queues:
            if round_index < len(queue):
                selected.append(queue[round_index])
                progressed = True
                if len(selected) == eval_pages:
                    break
        if not progressed:
            break
        round_index += 1
    return selected


def load_or_select_eval_pages(
    *,
    run_dir: Path,
    all_pages: Sequence[PageKey],
    page_lineage: Mapping[PageKey, str],
    eval_pages: int,
    run_seed: int,
) -> tuple[list[PageKey], str]:
    """The pinned eval page set for this run: read back verbatim if
    `s4_eval_pages.json` already exists (a resume, or a second attempt --
    "the curve is provably over one fixed set across resume", review MUST
    FIX 2), otherwise selected now and written once. Returns `(pages,
    page_key_set_sha256)`."""
    path = run_dir / _EVAL_PAGES_FILENAME
    if path.exists():
        payload = json.loads(path.read_text())
        pages = [(bundle, page) for bundle, page in payload["pages"]]
    else:
        pages = select_eval_pages(
            pages=all_pages, page_lineage=page_lineage, eval_pages=eval_pages,
            run_seed=run_seed,
        )
        path.write_text(
            json.dumps({"pages": [list(p) for p in pages]}, sort_keys=True) + "\n"
        )
    return pages, f"sha256:{page_key_set_sha256(pages)}"


# ---------------------------------------------------------------------------
# Periodic diagnostic eval + overfit signal (DEC-041 points 3/5)
# ---------------------------------------------------------------------------

#: Diagnostic eval records, one JSON line per eval, in `run_dir` itself
#: (not an attempt sibling): the history must survive across attempts so a
#: resumed run's overfit streak is exactly what an unbroken run would have
#: seen (design mirrors `_LEDGER_FILENAME`'s own run-dir placement).
_EVAL_LOG_FILENAME = "s4_eval.jsonl"

#: The persisted pinned eval rows (`_EvalRowCache`, 28.9.) plus a
#: `.meta.json` sidecar -- in `run_dir`, reused across attempts.
_EVAL_ROWS_FILENAME = "s4_eval_rows.pt"

#: The named code DEC-041 point 5 asks for: val loss rising while train
#: loss falls, `overfit_patience` evals in a row.
OVERFIT_SIGNAL = "OVERFIT_SUSPECTED"


class _EvalRowCache:
    """The pinned eval rows, read once and then served from memory.

    The val cache reader keeps a few shards; the 300 pinned pages are
    scattered over ~200 of 320 shards (1.4 GB each, loaded and
    integrity-checked in full), so every eval re-read ~270 GB -- ~14 min
    per eval every ~21 min of S4 v15b training (24.9.). On first access all
    pinned rows are read in index order (= shard order), then kept: 300 rows
    of ~43 MB each own their storage. Lazy, so `--admission-only` never
    pays for it; an index outside the pinned set is a `KeyError`.

    `persist_path` (28.9.): that first read still cost ~14-18 min at GPU
    0 % after EVERY (re)start -- long enough to be read as a hang and killed.
    With a path, the rows are saved once (atomic write, sha256 in a sidecar
    meta file) and later starts reload them in seconds. A file whose key
    (`persist_key`: eval pages + val cache + expectation), pinned indices or
    sha256 do not match is never trusted: the rows are rebuilt from the val
    cache and the file replaced."""

    def __init__(
        self,
        dataset: Any,
        indices: Sequence[int],
        *,
        persist_path: Path | None = None,
        persist_key: str | None = None,
    ) -> None:
        self._dataset = dataset
        self._indices = sorted(set(indices))
        self._rows: dict[int, Any] | None = None
        self._persist_path = persist_path
        self._persist_key = persist_key

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        if self._rows is None:
            self.materialize()
        assert self._rows is not None
        return self._rows[index]

    def materialize(self) -> str:
        """Load the rows now: `"loaded"` from the persisted file, or
        `"built"` from the val cache (and persisted, when a path is set)."""
        started = time.perf_counter()
        rows = self._load_persisted() if self._persist_path is not None else None
        source = "loaded"
        if rows is None:
            rows = {i: self._dataset[i] for i in self._indices}
            source = "built"
            if self._persist_path is not None:
                self._save(rows)
        self._rows = rows
        print(
            json.dumps(
                {
                    "eval_rows": {
                        "source": source,
                        "rows": len(rows),
                        "seconds": round(time.perf_counter() - started, 1),
                    }
                }
            ),
            flush=True,
        )
        return source

    def _meta_path(self) -> Path:
        assert self._persist_path is not None
        return self._persist_path.with_name(self._persist_path.name + ".meta.json")

    def _load_persisted(self) -> dict[int, Any] | None:
        path, meta_path = self._persist_path, self._meta_path()
        assert path is not None
        if not (path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text())
        if meta.get("key") != self._persist_key or meta.get("indices") != self._indices:
            print(json.dumps({"eval_rows": {"stale": str(path)}}), flush=True)
            return None
        if _sha256_file(path) != meta.get("sha256"):
            print(json.dumps({"eval_rows": {"corrupt": str(path)}}), flush=True)
            return None
        rows = torch.load(path, map_location="cpu", weights_only=False)
        if sorted(rows) != self._indices:
            return None
        return rows

    def _save(self, rows: dict[int, Any]) -> None:
        path, meta_path = self._persist_path, self._meta_path()
        assert path is not None
        tmp = path.with_name(path.name + ".tmp")
        torch.save(rows, tmp)
        digest = _sha256_file(tmp)
        tmp.replace(path)
        meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
        meta_tmp.write_text(
            json.dumps({"key": self._persist_key, "indices": self._indices, "sha256": digest})
        )
        meta_tmp.replace(meta_path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _onfly_eval_rows(eval_dataset: Any) -> Any:
    """The rows the on-the-fly eval attaches pixels to: the eval-row cache itself.

    On a page-image (7.0.0) cache the persisted rows carry `page_image` refs, not
    features, so they are exactly what `OnTheFlyDrawDataset` needs. Reading the
    cache's `_dataset` instead (as before 2026-10-06) re-read the 300 pinned pages
    from ~650 val shards at EVERY eval (~33 min per 1 000 steps of fresh S4).
    Indices are the base dataset's, so the eval order is unchanged."""
    return eval_dataset


def evaluate_s4(
    *,
    model: QwenDenseCandidateModel,
    dataset: stl.S4TrainingDataset | _EvalRowCache,
    indices: Sequence[int],
    image_token_id: int,
    batch_size: int,
    stage_policy: Mapping[str, Any],
    allowed_head_ids: Collection[str],
    device: torch.device | str = "cpu",
    mask_inactive_union_arms: bool = False,
    consistency_lambda: float = 0.0,
    structural_table_region: str = "abstain_not_visible",
    collapsed_region_rule: str = "abstain_container",
    story_decision_weight: float = 1.0,
    alignment_type_weights: Mapping[str, float] | None = None,
    visual_encoder: ov.OnTheFlyVisualEncoder | None = None,
    per_page: bool = False,
) -> dict[str, Any]:
    """Teacher-forced diagnostic loss over `indices` (design DEC-041 point
    3, RFC005 §12.2): `model.eval()` and `torch.no_grad()` for the
    duration, model mode restored on the way out regardless of outcome.
    Free_running decode is the only signal `RFC005` §12.2 allows to select
    or stop a run -- this teacher-forced loss is a monitoring diagnostic
    only, never a gate, which is why the caller labels every record it
    writes from this report `"diagnostic": true`.

    `visual_encoder` (DEC-056 on-the-fly path): `dataset` is then an
    `onfly_visual.OnTheFlyDrawDataset` over the eval rows (no augmentation)
    and each batch's features are computed by the frozen ViT before the
    forward; `None` (v15b) evaluates cached features exactly as before.

    `per_page=True` (needs `batch_size` 1, so a batch IS a page) adds `per_page`: one
    `{"total", "families"}` row per index, in order, for paired comparison of
    checkpoints (`checkpoint_eval_compare`). Off, the report keeps the keys the
    `s4_eval.jsonl` records have always had.
    """
    if not indices:
        raise S4TrainingRefused("evaluate_s4: indices is empty -- nothing to evaluate")
    if per_page and batch_size != 1:
        raise S4TrainingRefused(
            f"evaluate_s4: per_page needs batch_size 1 (a batch is a page), got {batch_size}"
        )
    page_rows: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    family_totals: dict[str, float] = {}
    family_counts: dict[str, int] = {}
    total_loss_sum = 0.0
    n_batches = 0
    try:
        with torch.no_grad():
            for lo in range(0, len(indices), batch_size):
                batch_indices = indices[lo : lo + batch_size]
                samples = [dataset[i] for i in batch_indices]
                if visual_encoder is not None:
                    samples = [draw.sample for draw in samples]
                batch = s1_train.collate_training_batch(
                    [s.base for s in samples],
                    ranges=model.ranges,
                    image_token_id=image_token_id,
                    device=device,
                )
                if visual_encoder is not None:
                    batch = visual_encoder.materialize(batch)
                typed_batch = stl.collate_typed(
                    samples,
                    allowed_head_ids=allowed_head_ids,
                    mask_inactive_union_arms=mask_inactive_union_arms,
                )
                region_batch = stl.collate_regions(
                    samples, **stl.region_kwargs(structural_table_region, collapsed_region_rule)
                )
                hidden, prefix_len = model.encode_hidden(batch.model_batch)
                composite_hidden = hidden[:, prefix_len + batch.visual_steps :, :]
                logits = model.composite_head(composite_hidden)
                total, report = stl.s4_total_loss(
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
                    region_batch=region_batch,
                    image_grid_thw=batch.model_batch.image_grid_thw,
                    consistency_lambda=consistency_lambda,
                    story_decision_weight=story_decision_weight,
                    ranges=model.ranges,
                    alignment_type_weights=alignment_type_weights,
                )
                total_loss_sum += float(total.detach())
                n_batches += 1
                if per_page:
                    page_rows.append(
                        {
                            "total": float(total.detach()),
                            "families": {
                                name: entry["mean"] for name, entry in report["families"].items()
                            },
                        }
                    )
                for name, entry in report["families"].items():
                    if entry["mean"] is None:
                        continue
                    family_totals[name] = family_totals.get(name, 0.0) + entry["mean"]
                    family_counts[name] = family_counts.get(name, 0) + 1
    finally:
        model.train(was_training)
    if n_batches == 0:
        raise S4TrainingRefused("evaluate_s4: no batch scored anything")
    result: dict[str, Any] = {
        "total_loss": total_loss_sum / n_batches,
        "families": {name: family_totals[name] / family_counts[name] for name in family_totals},
        "n_batches": n_batches,
    }
    if per_page:
        result["per_page"] = page_rows
    return result


def evaluate_checkpoints(
    *,
    model: QwenDenseCandidateModel,
    checkpoints: Sequence[Path],
    dataset: Any,
    indices: Sequence[int],
    eval_pages: Sequence[PageKey],
    verify: Callable[[Path, Mapping[str, Any]], None],
    evaluate_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Teacher-force every checkpoint on the SAME pages, one page per batch, and pair
    them (`checkpoint_eval_compare.build_report`; the first checkpoint is the baseline).

    `verify(path, state)` is the caller's admission for a loaded checkpoint (identity,
    manifest, ancestry sidecar); it raises to refuse. The weights are loaded STRICT into
    `model`, in order, so each member is scored by its own weights and nothing else.
    Monitoring only (RFC005 §12.2): the report selects and promotes nothing."""
    if len(checkpoints) < 2:
        raise S4TrainingRefused("evaluate_checkpoints: need at least two checkpoints to compare")
    names = [p.name for p in checkpoints]
    if len(set(names)) != len(names):
        names = [str(p) for p in checkpoints]
    members: list[dict[str, Any]] = []
    for name, path in zip(names, checkpoints, strict=True):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in state:
            raise S4TrainingRefused(f"{path}: not a training checkpoint (no model_state_dict)")
        verify(path, state)
        try:
            model.load_state_dict(state["model_state_dict"])
        except RuntimeError as exc:
            raise S4TrainingRefused(f"{path}: model.load_state_dict refused: {exc}") from exc
        report = evaluate_s4(
            model=model, dataset=dataset, indices=indices, batch_size=1,
            per_page=True, **evaluate_kwargs,
        )
        members.append(
            {
                "name": name,
                "path": str(path),
                "step": int(state["step"]),
                "sha256": ca.file_sha256(path),
                "averaged_from": state.get("averaged_from"),
                "pages": [f"{bundle}:{page}" for bundle, page in eval_pages],
                "per_page": report["per_page"],
            }
        )
        del state
    built = cec.build_report(members)
    for member in members:
        detail = built["members"][member["name"]]
        detail.update(
            {k: member[k] for k in ("path", "sha256", "averaged_from")},
        )
    built["pages"] = members[0]["pages"]
    built["per_page"] = {m["name"]: m["per_page"] for m in members}
    return built


def _load_eval_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def overfit_streak(records: Sequence[Mapping[str, Any]]) -> int:
    """The number of TRAILING eval records (from the end) where val total
    loss rose and train loss fell relative to the immediately preceding
    record. Pure over the eval log alone -- a resume needs no separate
    counter to persist, because replaying the log (which already survives
    a resume, `_load_eval_log` reads it fresh every time) reproduces this
    exactly. A record whose own or the previous record's
    `train_loss_recent` is `None` (no step of this attempt contributed to
    the window before that eval fired) breaks the streak rather than
    guessing."""
    streak = 0
    for i in range(len(records) - 1, 0, -1):
        cur, prev = records[i], records[i - 1]
        cur_train = cur.get("train_loss_recent")
        prev_train = prev.get("train_loss_recent")
        if cur_train is None or prev_train is None:
            break
        val_rose = cur["val"]["total_loss"] > prev["val"]["total_loss"]
        train_fell = cur_train < prev_train
        if not (val_rose and train_fell):
            break
        streak += 1
    return streak


# ---------------------------------------------------------------------------
# Checkpoint history (DEC-041 point 4)
# ---------------------------------------------------------------------------


def _checkpoint_history_path(out: Path, step: int) -> Path:
    """`out` with `-step-<step>` spliced before its suffix, e.g.
    `checkpoint.pt` -> `checkpoint-step-000001234.pt` -- a history entry no
    later save can overwrite, distinct from `out` itself, which stays the
    single "latest" pointer `--resume` reads."""
    return out.with_name(f"{out.stem}-step-{step:09d}{out.suffix}")


def _best_eval_path(out: Path) -> Path:
    """`checkpoint.pt` -> `checkpoint-best-eval.pt`: the checkpoint saved at
    the lowest diagnostic val loss so far (DEC-041 point 4), overwritten
    only by a better one. A candidate for the post-hoc free-running
    selection (DEC-041 A), never a selection by itself."""
    return out.with_name(f"{out.stem}-best-eval{out.suffix}")


#: DEC-041 point 4 / review SHOULD FIX 5: ~237 step copies at ~5.2 GB each
#: is ~1.2 TB for one production run. "Keep last + a periodic sample" --
#: `out` is always the last (never gated), a step-numbered copy is kept
#: only every `keep_every` steps.
DEFAULT_CHECKPOINT_KEEP_EVERY = 10_000


def _save_checkpoint_with_history(
    out: Path,
    *,
    model: QwenDenseCandidateModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    dataset: stl.S4TrainingDataset,
    companion_hash: str,
    s4_run_id: str,
    ancestry_base: Mapping[str, Any] | None = None,
    keep_every: int = DEFAULT_CHECKPOINT_KEEP_EVERY,
    keep_as_best: bool = False,
    visual_feature_tap: str | None = None,
) -> None:
    """Write the "latest" checkpoint at `out` (`_checkpoint`, unchanged --
    what `--resume` reads) and, only every `keep_every` steps, ALSO a
    step-numbered copy that is never overwritten (review SHOULD FIX 5: an
    intermediate copy at every `checkpoint_every` cadence was ~1.2 TB over
    one production run). The copy is a byte-for-byte duplicate of what
    `_checkpoint` just wrote, sidecar included, so the ancestry sidecar's
    `checkpoint_sha256` (computed over the file's own bytes) still matches
    the copy -- no ancestry document is recomputed for the history path."""
    _checkpoint(
        out,
        model=model,
        optimizer=optimizer,
        step=step,
        dataset=dataset,
        companion_hash=companion_hash,
        s4_run_id=s4_run_id,
        ancestry_base=ancestry_base,
        visual_feature_tap=visual_feature_tap,
    )
    copies = []
    if keep_as_best:
        copies.append(_best_eval_path(out))
    if keep_every > 0 and step % keep_every == 0:
        copies.append(_checkpoint_history_path(out, step))
    sidecar = ca.sidecar_path(out)
    for copy_path in copies:
        shutil.copy2(out, copy_path)
        if sidecar.exists():
            shutil.copy2(sidecar, ca.sidecar_path(copy_path))
        print(f"[s4_train] checkpoint copy step {step} -> {copy_path}", flush=True)


class AugmentationLedger:
    """Aggregates `OnTheFlyDraw.aug_report` into one attempt's
    `attempt-NNNN-augmentation.json` (design §10, RFC003-ERRATA-004).

    A schema-1 (or no) augmentation reports nothing per draw, so its payload
    is `views_total` alone; schema 2 adds the applied/skipped/rejected
    counters and the three shares. `applied_views` is `selected_views -
    legibility_exhausted` by construction: an exhausted page is served clean.
    """

    def __init__(
        self, augmentation: ov.OnTheFlyAugmentation | None, *, external: bool = False
    ) -> None:
        self._external = external
        self.external_views: dict[str, int] = {}
        self._v2 = augmentation is not None and augmentation.schema_major >= 2
        self._declared_milli: int | None = None
        if self._v2:
            policy = augmentation.policy
            self._declared_milli = (
                1000 * policy["selection_numerator"] // policy["selection_denominator"]
            )
        self.views_total = 0
        self.selected_views = 0
        self.legibility_exhausted = 0
        self.applied: dict[str, int] = {}
        self.skipped: dict[str, dict[str, int]] = {}
        self.legibility_rejected: dict[str, int] = {}
        self._readable_retry = augmentation is not None and augmentation.readable_retry
        self.ink_free_retry_attempts = 0
        self.ink_free_retry_applied = 0
        self.envelope_retry_attempts = 0
        self.envelope_retry_applied = 0

    def add(self, aug_report: Mapping[str, Any] | None) -> None:
        self.views_total += 1
        if not self._v2 or aug_report is None:
            return
        self.selected_views += int(aug_report["selected"])
        self.legibility_exhausted += int(aug_report["legibility_exhausted"])
        self.ink_free_retry_attempts += int(aug_report.get("ink_free_retry_attempts", 0))
        self.ink_free_retry_applied += int(aug_report.get("ink_free_retry_applied", 0))
        self.envelope_retry_attempts += int(aug_report.get("envelope_retry_attempts", 0))
        self.envelope_retry_applied += int(aug_report.get("envelope_retry_applied", 0))
        for op in aug_report["applied"]:
            self.applied[op] = self.applied.get(op, 0) + 1
        for op, reason in aug_report["skipped"].items():
            reasons = self.skipped.setdefault(op, {})
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in aug_report["legibility_rejected"].items():
            self.legibility_rejected[reason] = self.legibility_rejected.get(reason, 0) + count

    def add_external(self, kind: str) -> None:
        """An external (scan / photo / augraphy) draw: not a `views_total` view (DEC-085)."""
        self.external_views[kind] = self.external_views.get(kind, 0) + 1

    def payload(self) -> dict[str, Any]:
        result = self._base_payload()
        if self._external:
            result["external_views"] = dict(sorted(self.external_views.items()))
        return result

    def _base_payload(self) -> dict[str, Any]:
        if not self._v2:
            return {"views_total": self.views_total}
        applied_views = self.selected_views - self.legibility_exhausted

        def share(count: int, whole: int) -> int | None:
            # No views yet: the share is undefined, never a numeric zero.
            return None if whole == 0 else int(round(1000 * count / whole))

        result = {
            "views_total": self.views_total,
            "selected_views": self.selected_views,
            "applied_views": applied_views,
            "applied": dict(sorted(self.applied.items())),
            "skipped": {op: dict(sorted(r.items())) for op, r in sorted(self.skipped.items())},
            "legibility_rejected": dict(sorted(self.legibility_rejected.items())),
            "legibility_exhausted": self.legibility_exhausted,
            "declared_share_milli": self._declared_milli,
            "selected_share_milli": share(self.selected_views, self.views_total),
            "augmented_share_milli": share(applied_views, self.views_total),
            "applied_share_milli_by_op": {
                op: share(count, applied_views) for op, count in sorted(self.applied.items())
            },
        }
        if self._readable_retry:
            result["ink_free_retry_attempts"] = self.ink_free_retry_attempts
            result["ink_free_retry_applied"] = self.ink_free_retry_applied
            result["envelope_retry_attempts"] = self.envelope_retry_attempts
            result["envelope_retry_applied"] = self.envelope_retry_applied
        return result


def run_s4_training(
    *,
    run_dir: Path,
    model: QwenDenseCandidateModel,
    dataset: stl.S4TrainingDataset,
    optimizer: torch.optim.Optimizer,
    image_token_id: int,
    plan_pages: Sequence[PageKey],
    plan_lineages: Sequence[str],
    plan_cells: Sequence[tuple[str, str, str, str]],
    plan_order: Sequence[int],
    companion_hash: str,
    s4_run_id: str,
    steps: int,
    batch_size: int,
    stage_policy: Mapping[str, Any],
    allowed_head_ids: Collection[str],
    grad_clip: float,
    device: torch.device | str = "cpu",
    checkpoint_every: int = 0,
    checkpoint_keep_every: int = DEFAULT_CHECKPOINT_KEEP_EVERY,
    out: Path | None = None,
    start_step: int = 0,
    ledger: ex.ExposureLedger | None = None,
    execution_profile: Mapping[str, str] | None = None,
    ancestry_base: Mapping[str, Any] | None = None,
    integrity: Mapping[str, str],
    num_workers: int = 0,
    eval_every: int = 0,
    eval_dataset: stl.S4TrainingDataset | _EvalRowCache | None = None,
    eval_order: Sequence[int] | None = None,
    eval_batch_size: int = 1,
    eval_pages_hash: str | None = None,
    overfit_patience: int = 3,
    stop_on_overfit: bool = False,
    plan_view_kind: Sequence[str] | None = None,
    plan_view_id: Sequence[str | None] | None = None,
    augmented_features: afc.AugFeatureCacheReader | None = None,
    stop_requested: Callable[[], bool] | None = None,
    mask_inactive_union_arms: bool = False,
    consistency_lambda: float = 0.0,
    structural_table_region: str = "abstain_not_visible",
    collapsed_region_rule: str = "abstain_container",
    story_decision_weight: float = 1.0,
    alignment_type_weights: Mapping[str, float] | None = None,
    visual_encoder: ov.OnTheFlyVisualEncoder | None = None,
    visual_prefetch: bool = False,
    lr_schedule: lrs.LrSchedule | None = None,
    lr_bases: Sequence[float] | None = None,
    mask_color_unseen: bool = False,
) -> ex.ExposureLedger:
    """Run the S4 optimizer steps from `start_step` up to `steps` -- the
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

    `eval_every` (DEC-041 point 3): every `eval_every`-th step runs
    :func:`evaluate_s4` over `eval_dataset`/`eval_order` and appends a
    `"diagnostic": true` record to `run_dir/s4_eval.jsonl` -- teacher-
    forced, monitoring only (RFC005 §12.2), never a gate by itself. Each
    record also carries the overfit signal (`overfit_streak` over the
    log, DEC-041 point 5): `OVERFIT_SIGNAL` once val loss has risen while
    train loss fell for `overfit_patience` evals running. The signal never
    stops the run unless `stop_on_overfit=True`, an explicit deviation
    from §12.2 the caller opts into.

    `plan_view_kind`/`plan_view_id` (T1b, page-uniform mode only): one
    `"clean"`/`"augmented"` + view id per DRAW (not per lineage/page --
    the SAME page can be clean in one epoch's slot and augmented in
    another's, per `s4_plan`'s `s4_page_uniform_plan.json` sidecar). A
    `"clean"` draw trains the fetched sample unchanged; an `"augmented"`
    one has its `features`/`grid_thw`/`region_targets` REPLACED from
    `augmented_features` (`aug_feature_cache.AugFeatureCacheReader`) --
    every other field (typed targets, text tokens) stays the clean row's
    own, since augmentation only perturbs the image. `None` (legacy/
    weighted mode) trains every draw as fetched, exactly as before.

    `visual_encoder` (DEC-056 addendum A, on-the-fly ViT): `dataset` must be
    an `onfly_visual.OnTheFlyDrawDataset` (addressed by plan DRAW index; the
    worker attaches the page's pixels and that draw's augmentation view),
    each collated batch is `visual_encoder.materialize`d (frozen ViT, no
    grad, DeepStack levels when the policy has them) before the forward, the
    ledger records each draw's own view id, and a DeepStack checkpoint is
    stamped with its `visual_feature_tap`. Fixed DEC-040 views
    (`plan_view_kind`/`augmented_features`) are refused with it -- the
    on-the-fly views replace them. `eval_dataset` must then be an
    `OnTheFlyDrawDataset` too. `None` (default): the cached path, unchanged.

    `visual_prefetch` (fresh S4 speed knob, default off, on-the-fly only):
    step k+1's collate + ViT is issued (side CUDA stream on a GPU) before
    step k's forward/backward -- `onfly_visual.prefetch_materialized`. The
    trained weights, ledger and losses are bit-identical to the plain path;
    only `data_wait_s` and memory change. With a feature cache on the
    encoder, clean draws are served from / written to it (`cache_refs`).
    """
    if visual_prefetch and visual_encoder is None:
        raise S4TrainingRefused("visual_prefetch needs the on-the-fly visual path (visual_encoder)")
    if (
        visual_encoder is not None
        and plan_view_kind is not None
        and all(kind == "clean" for kind in plan_view_kind)
    ):
        # A page-uniform plan without fixed DEC-040 views (the fresh S4 run):
        # every draw is clean and the on-the-fly augmentation picks the views,
        # so there is ONE exposure plan, not two.
        plan_view_kind = plan_view_id = None
    if visual_encoder is not None:
        if not isinstance(dataset, ov.OnTheFlyDrawDataset):
            raise S4TrainingRefused(
                "visual_encoder needs an onfly_visual.OnTheFlyDrawDataset (draw-indexed)"
            )
        if plan_view_kind is not None or augmented_features is not None:
            raise S4TrainingRefused(
                "fixed DEC-040 views (plan_view_kind / augmented_features) and the "
                "on-the-fly visual path are two exposure plans; a run is one of them"
            )
        if list(dataset.plan_order) != list(plan_order):
            raise S4TrainingRefused("the draw dataset's plan_order is not this run's")
        if bool(getattr(model, "deepstack", False)) != visual_encoder.policy.deepstack:
            raise S4TrainingRefused(
                f"model deepstack={getattr(model, 'deepstack', False)} but the visual "
                f"policy's DeepStack taps are {visual_encoder.policy.deepstack_visual_indexes}"
            )
        if eval_every and eval_dataset is not None and not isinstance(
            eval_dataset, ov.OnTheFlyDrawDataset
        ):
            raise S4TrainingRefused("on-the-fly eval needs an OnTheFlyDrawDataset eval_dataset")
    elif getattr(model, "deepstack", False):
        raise S4TrainingRefused(
            "a DeepStack model trains only on the on-the-fly visual path (visual_encoder)"
        )
    checkpoint_dataset = dataset.base if visual_encoder is not None else dataset
    checkpoint_tap = (
        visual_encoder.policy.visual_feature_tap()
        if visual_encoder is not None and visual_encoder.policy.deepstack
        else None
    )
    if start_step >= steps:
        raise S4TrainingRefused(
            f"start_step {start_step} has already reached the run's total budget "
            f"of {steps} step(s) -- nothing left to run"
        )
    ledger = ledger if ledger is not None else ex.ExposureLedger(s4_run_id)
    execution_profile = (
        dict(execution_profile)
        if execution_profile
        else s2t.execution_profile_for(device, num_workers)
    )
    attempts_dir = s2t._attempts_dir(run_dir)
    attempt_index = s2t._next_attempt_index(run_dir)
    attempt = ex.start_attempt(
        run_id=s4_run_id,
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

    external_on = visual_encoder is not None and getattr(dataset, "external", None) is not None
    aug_ledger = AugmentationLedger(
        getattr(dataset, "augmentation", None) if visual_encoder is not None else None,
        external=external_on,
    )
    typed_supervised_seen: dict[str, int] = {}
    region_supervision_seen = {"status": 0, "box": 0}
    region_buckets = stl.region_excluded_buckets(structural_table_region, collapsed_region_rule)
    region_excluded_seen = dict.fromkeys(region_buckets, 0)
    region_evidence_by_node_type: dict[str, dict[str, Any]] = {}

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
        (attempts_dir / f"attempt-{attempt_index:04d}-augmentation.json").write_text(
            json.dumps(aug_ledger.payload(), sort_keys=True)
        )
        # Region targets are not typed-property targets. Keep their evidence
        # in an attempt sibling rather than widening the closed run-attempt
        # manifest or fabricating a typed-head identifier for the aligner.
        (attempts_dir / f"attempt-{attempt_index:04d}-region-supervision.json").write_text(
            json.dumps(
                {
                    "box_supervised": region_supervision_seen["box"],
                    "excluded": region_excluded_seen,
                    "by_node_type": region_evidence_by_node_type,
                    "status_supervised": region_supervision_seen["status"],
                },
                sort_keys=True,
            )
        )

    # Steps OF THIS ATTEMPT, never the global step -- a resume starting
    # past step 50 must get its own `min(steps, 50)`-step grace window,
    # not zero.
    early_refusal_steps_this_attempt = min(steps - start_step, _EARLY_REFUSAL_STEP_CAP)
    # Hashed once, not once per step: `stage_policy` is loop-invariant.
    stage_policy_hash_value = spolicy.stage_policy_hash(stage_policy)
    step = start_step
    # Train loss since the last eval (or since this attempt started, on a
    # resume that lands mid-interval) -- reset every time an eval fires.
    train_loss_window: list[float] = []
    best_eval_step: int | None = None  # the step whose eval set a new val minimum
    # DEC-041/T1b: cumulative page-uniform view counts THIS ATTEMPT served,
    # printed on every step line -- the run's own record of clean vs
    # augmented draws (`plan_view_kind is None` in legacy/weighted mode,
    # where the split is not tracked at all).
    clean_draws_served = 0
    augmented_draws_served = 0
    external_draws_served = 0  # DEC-085; reported only when external views are on
    # SHOULD FIX 6 (review): a resume must not leave a step <= start_step
    # eval record that a FRESH step at the exact same step number would
    # duplicate -- `overfit_streak` compares consecutive records and a
    # duplicate pair (identical val/train) would read as "flat", never
    # "rising", silently hiding a real streak. Truncate once, up front.
    eval_log_path_for_resume = run_dir / _EVAL_LOG_FILENAME
    if eval_log_path_for_resume.exists():
        kept = [r for r in _load_eval_log(eval_log_path_for_resume) if r["step"] <= start_step]
        with open(eval_log_path_for_resume, "w") as f:
            for record in kept:
                f.write(json.dumps(record, sort_keys=True) + "\n")
    try:
        batches = s2t.plan_batches(
            dataset,
            plan_order if visual_encoder is None else range(len(plan_order)),
            start_step=start_step, steps=steps,
            batch_size=batch_size, num_workers=num_workers,
        )
        if visual_prefetch:
            assert visual_encoder is not None
            batches = ov.prefetch_materialized(
                batches,
                encoder=visual_encoder,
                collate=lambda draws: s1_train.collate_training_batch(
                    [draw.sample.base for draw in draws],
                    ranges=model.ranges,
                    image_token_id=image_token_id,
                    device=device,
                ),
            )
        # Same timing fields as `s2_train.run_training`.
        waited_from = time.perf_counter()
        for samples in batches:
            data_wait_s = time.perf_counter() - waited_from
            step += 1
            lo, hi = (step - 1) * batch_size, step * batch_size
            indices = plan_order[lo:hi]

            draw_view_ids: list[str] | None = None
            prefetched_batch = None
            cache_refs = None
            if visual_encoder is not None:
                if visual_prefetch:
                    samples, prefetched_batch = samples
                else:
                    cache_refs = visual_encoder.cache_refs(samples)
                draw_view_ids = [draw.view_id for draw in samples]
                augmented_now = sum(1 for draw in samples if draw.augmented)
                external_now = sum(1 for draw in samples if draw.external_kind)
                augmented_draws_served += augmented_now
                external_draws_served += external_now
                clean_draws_served += len(samples) - augmented_now - external_now
                for draw in samples:
                    if draw.external_kind:
                        aug_ledger.add_external(draw.external_kind)
                    else:
                        aug_ledger.add(draw.aug_report)
                samples = [draw.sample for draw in samples]
            else:
                for _ in samples:
                    aug_ledger.add(None)

            if plan_view_kind is not None:
                effective_samples = []
                for offset, sample in enumerate(samples):
                    draw_index = lo + offset
                    kind = plan_view_kind[draw_index]
                    if kind == "clean":
                        clean_draws_served += 1
                        effective_samples.append(sample)
                        continue
                    augmented_draws_served += 1
                    view_id = plan_view_id[draw_index] if plan_view_id else None
                    if augmented_features is None:
                        raise S4TrainingRefused(
                            f"draw {draw_index} is augmented (view_id={view_id!r}) but no "
                            "--augmented-features store was given"
                        )
                    try:
                        aug_row = augmented_features.get(view_id)
                    except afc.AugFeatureCacheError as exc:
                        raise S4TrainingRefused(
                            f"draw {draw_index}: augmented view {view_id!r} could not be "
                            f"read from the aug feature store: {exc}"
                        ) from exc
                    expected_page = plan_pages[draw_index]
                    actual_page = (aug_row["document_bundle_id"], aug_row["page_index"])
                    if actual_page != expected_page:
                        raise S4TrainingRefused(
                            f"draw {draw_index}: aug row for view {view_id!r} is page "
                            f"{actual_page!r}, not the plan's {expected_page!r}"
                        )
                    new_base = dataclasses.replace(
                        sample.base,
                        features=aug_row["features"],
                        grid_thw=aug_row["grid_thw"],
                    )
                    effective_samples.append(
                        dataclasses.replace(
                            sample, base=new_base, region_targets=aug_row["region_targets"]
                        )
                    )
                samples = effective_samples

            batch = prefetched_batch
            if batch is None:
                batch = s1_train.collate_training_batch(
                    [s.base for s in samples],
                    ranges=model.ranges,
                    image_token_id=image_token_id,
                    device=device,
                )
            typed_batch = stl.collate_typed(
                samples,
                allowed_head_ids=allowed_head_ids,
                mask_inactive_union_arms=mask_inactive_union_arms,
                **({"mask_color_unseen": True} if mask_color_unseen else {}),
            )
            region_batch = stl.collate_regions(
                samples, **stl.region_kwargs(structural_table_region, collapsed_region_rule)
            )
            if visual_encoder is not None and prefetched_batch is None:
                batch = visual_encoder.materialize(batch, cache_refs=cache_refs)

            optimizer.zero_grad(set_to_none=True)
            hidden, prefix_len = model.encode_hidden(batch.model_batch)
            composite_hidden = hidden[:, prefix_len + batch.visual_steps :, :]
            logits = model.composite_head(composite_hidden)
            total, report = stl.s4_total_loss(
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
                region_batch=region_batch,
                image_grid_thw=batch.model_batch.image_grid_thw,
                consistency_lambda=consistency_lambda,
                story_decision_weight=story_decision_weight,
                ranges=model.ranges,
                alignment_type_weights=alignment_type_weights,
            )
            train_loss_window.append(float(total.detach()))
            total.backward()
            grad_norm_groups = _grad_norm_groups(model)
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
            grad_norm_total = float(norm)
            if not bool(torch.isfinite(norm)):
                optimizer.zero_grad(set_to_none=True)
                raise s1_train.NonFiniteStepError(f"gradient norm is {grad_norm_total}")
            if lr_schedule is not None:
                assert lr_bases is not None
                lrs.apply(optimizer, lr_schedule, step, total_steps=steps, bases=lr_bases)
            optimizer.step()

            for offset, idx in enumerate(indices):
                draw_index = lo + offset
                bundle, page = plan_pages[draw_index]
                # Review SHOULD FIX 4: an augmented draw's exposure record
                # names the render_view_id it actually served -- "bundle:page"
                # for both kinds would make a clean and an augmented view of
                # the SAME page indistinguishable in the ledger.
                if draw_view_ids is not None:
                    view_id_str = draw_view_ids[offset]
                elif plan_view_kind is not None and plan_view_kind[draw_index] == "augmented":
                    view_id_str = plan_view_id[draw_index] if plan_view_id else None
                else:
                    view_id_str = f"{bundle}:{page}"
                ledger.append(
                    attempt_id=attempt["attempt_id"],
                    lineage_root_id=plan_lineages[draw_index],
                    cell=plan_cells[draw_index],
                    view_ids=[view_id_str],
                    sample_count=1,
                    optimizer_step=step,
                    plan_draw_index=draw_index,
                )

            for head_id, entry in report["typed_heads"].items():
                typed_supervised_seen[head_id] = (
                    typed_supervised_seen.get(head_id, 0) + entry["supervised_count"]
                )
            region_supervision_seen["status"] += int(region_batch.supervise_status.sum())
            region_supervision_seen["box"] += int(region_batch.supervise_box.sum())
            for bucket, count in region_batch.excluded.items():
                region_excluded_seen[bucket] += count
            for node_type, counts in region_batch.excluded_by_node_type.items():
                entry = region_evidence_by_node_type.setdefault(
                    node_type,
                    {
                        "box_supervised": 0,
                        "excluded": dict.fromkeys(region_buckets, 0),
                        "status_supervised": 0,
                    },
                )
                for bucket, count in counts.items():
                    entry["excluded"][bucket] += count
            for node_type, supervise_box in zip(
                region_batch.node_types, region_batch.supervise_box.tolist()
            ):
                entry = region_evidence_by_node_type.setdefault(
                    node_type,
                    {
                        "box_supervised": 0,
                        "excluded": dict.fromkeys(region_buckets, 0),
                        "status_supervised": 0,
                    },
                )
                entry["status_supervised"] += 1
                entry["box_supervised"] += int(supervise_box)

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
                        "view_counts": (
                            None
                            if plan_view_kind is None and visual_encoder is None
                            else {
                                "clean": clean_draws_served,
                                "augmented": augmented_draws_served,
                                **({"external": external_draws_served} if external_on else {}),
                            }
                        ),
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
                raise S4TrainingRefused(
                    f"trained nothing typed after {steps_this_attempt} step(s) of "
                    f"THIS ATTEMPT (>= min(steps - start_step, "
                    f"{_EARLY_REFUSAL_STEP_CAP})={early_refusal_steps_this_attempt}) -- "
                    "an S4 typed run that has seen zero typed supervision is refused"
                )

            if eval_every and eval_dataset is not None and step % eval_every == 0:
                train_loss_recent = (
                    sum(train_loss_window) / len(train_loss_window)
                    if train_loss_window
                    else None
                )
                val_report = evaluate_s4(
                    model=model,
                    dataset=eval_dataset,
                    indices=eval_order or (),
                    image_token_id=image_token_id,
                    batch_size=eval_batch_size,
                    stage_policy=stage_policy,
                    allowed_head_ids=allowed_head_ids,
                    device=device,
                    mask_inactive_union_arms=mask_inactive_union_arms,
                    consistency_lambda=consistency_lambda,
                    structural_table_region=structural_table_region,
                    collapsed_region_rule=collapsed_region_rule,
                    story_decision_weight=story_decision_weight,
                    alignment_type_weights=alignment_type_weights,
                    **({} if visual_encoder is None else {"visual_encoder": visual_encoder}),
                )
                eval_log_path = run_dir / _EVAL_LOG_FILENAME
                eval_history = _load_eval_log(eval_log_path)
                eval_record: dict[str, Any] = {
                    "step": step,
                    "diagnostic": True,
                    "note": (
                        "RFC005 §12.2: teacher-forced, monitoring only -- "
                        "never selects or stops a run by itself"
                    ),
                    "val": val_report,
                    "train_loss_recent": train_loss_recent,
                    "eval_pages_hash": eval_pages_hash,
                }
                eval_record["best_so_far"] = all(
                    val_report["total_loss"] < r["val"]["total_loss"] for r in eval_history
                )
                if eval_record["best_so_far"]:
                    best_eval_step = step
                eval_history.append(eval_record)
                streak = overfit_streak(eval_history)
                eval_record["consecutive_worsening"] = streak
                eval_record["overfit_signal"] = (
                    OVERFIT_SIGNAL if streak >= overfit_patience else None
                )
                with open(eval_log_path, "a") as f:
                    f.write(json.dumps(eval_record, sort_keys=True) + "\n")
                print(json.dumps({"eval": eval_record}), flush=True)
                train_loss_window = []
                if eval_record["overfit_signal"] and stop_on_overfit:
                    raise S4OverfitStopped(
                        f"val loss rose while train loss fell for {streak} consecutive "
                        f"eval(s) (>= --eval-patience {overfit_patience}) at step {step}"
                    )

            if out is not None and checkpoint_every and step % checkpoint_every == 0:
                _save_checkpoint_with_history(
                    out, model=model, optimizer=optimizer, step=step,
                    dataset=checkpoint_dataset, companion_hash=companion_hash,
                    s4_run_id=s4_run_id, ancestry_base=ancestry_base,
                    keep_every=checkpoint_keep_every, keep_as_best=step == best_eval_step,
                    visual_feature_tap=checkpoint_tap,
                )
                ledger.save_jsonl(run_dir / _LEDGER_FILENAME)
            if stop_requested is not None and stop_requested():
                # Owner 24.9.: night-only training; the launcher asks for a
                # stop at 05:30. Checkpoint THIS step so RESUME=1 loses nothing.
                if out is not None and not (checkpoint_every and step % checkpoint_every == 0):
                    _save_checkpoint_with_history(
                        out, model=model, optimizer=optimizer, step=step,
                        dataset=checkpoint_dataset, companion_hash=companion_hash,
                        s4_run_id=s4_run_id, ancestry_base=ancestry_base,
                        keep_every=checkpoint_keep_every,
                        keep_as_best=step == best_eval_step,
                        visual_feature_tap=checkpoint_tap,
                    )
                    ledger.save_jsonl(run_dir / _LEDGER_FILENAME)
                _finalize("cancelled")
                print(json.dumps({"stopped_on_request": {"step": step}}), flush=True)
                return ledger
            waited_from = time.perf_counter()
        if out is not None:
            _save_checkpoint_with_history(
                out, model=model, optimizer=optimizer, step=step,
                dataset=checkpoint_dataset, companion_hash=companion_hash,
                s4_run_id=s4_run_id, ancestry_base=ancestry_base,
                keep_every=checkpoint_keep_every, keep_as_best=step == best_eval_step,
                visual_feature_tap=checkpoint_tap,
            )
            ledger.save_jsonl(run_dir / _LEDGER_FILENAME)
    except S4OverfitStopped as exc:
        _finalize("cancelled")
        print(json.dumps({"stopped_overfit": str(exc)}), flush=True)
        return ledger
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
    dataset: stl.S4TrainingDataset,
    companion_hash: str,
    s4_run_id: str,
    ancestry_base: Mapping[str, Any] | None = None,
    visual_feature_tap: str | None = None,
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
        run_id=s4_run_id,
    )
    _stamp_typed_head_config_hash(
        out, ancestry_base=ancestry_base, step=step, run_id=s4_run_id,
        s1_run_manifest_hash=companion_hash,
        level_router_policy=level_router_checkpoint_payload(model),
        visual_feature_tap=visual_feature_tap,
    )
    print(f"[s4_train] checkpoint step {step} -> {out}", flush=True)


def _stamp_typed_head_config_hash(
    path: Path,
    *,
    ancestry_base: Mapping[str, Any] | None,
    step: int,
    run_id: str,
    s1_run_manifest_hash: str | None,
    level_router_policy: Mapping[str, Any] | None = None,
    visual_feature_tap: str | None = None,
) -> None:
    """Add `typed_head_config_hash` to an already-written checkpoint file
    (design §7 rev 3, T8 binding (ii)). `s1_train.save_checkpoint` is the
    shared, stage-agnostic checkpoint writer -- frozen while other stages'
    runs are live (design §5 [governance R1]) and carries no
    `typed_head_config` concept -- so S4 stamps its own identity field onto
    the file it just wrote as a second, small, non-atomic write, rather
    than growing that shared writer's contract for one stage.

    Changing the file's bytes changes its sha256, so the sidecar
    `save_checkpoint` already wrote (for the PRE-stamp bytes) would go
    stale the moment this function stopped there -- `checkpoint_ancestry.
    verify_sidecar`'s `checkpoint_sha256` check would then refuse every
    future read of this very checkpoint. This function therefore also
    recomputes the ancestry document (`extend_for_checkpoint`, the SAME
    pure inputs `save_checkpoint` used, so the embedded `ancestry_hash`
    this function leaves untouched still matches) against the NEW bytes
    and re-writes the sidecar (`finalize_checkpoint_document` +
    `write_sidecar`, `checkpoint_ancestry.write_sidecar` overwrites
    unconditionally).

    A crash between the state-dict write and the sidecar re-write leaves a
    checkpoint with NO `typed_head_config_hash` key and a sidecar for the
    OLD bytes -- `load_init_weights` treats a missing key as "no claim to
    compare" (never a silent pass under a wrong value, the same
    fail-closed shape `ancestry_hash: None` already uses for a diagnostic
    checkpoint), and the stale sidecar fails closed on `verify_sidecar`'s
    own `checkpoint_sha256` check rather than being silently trusted.
    `ancestry_base=None` (a diagnostic run, no sidecar at all) stamps the
    state dict only, matching `save_checkpoint`'s own no-sidecar path.

    `visual_feature_tap` (DEC-056): stamped only for a DeepStack run
    (`"pooler_output+deepstack:..."`); absent means `"pooler_output"`, so a
    v15b-shaped run writes exactly the keys it always did, and every
    loader refuses a checkpoint whose tap differs from its own
    (`onfly_visual.require_checkpoint_visual_feature_tap`).
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    state["typed_head_config_hash"] = thc.typed_head_config_hash()
    # RFC007: stamped only for a router-bearing model, so the runtime can
    # rebuild it; absent == no router (every v15b checkpoint).
    if level_router_policy is not None:
        state[LEVEL_ROUTER_CHECKPOINT_KEY] = dict(level_router_policy)
    if visual_feature_tap is not None:
        state["visual_feature_tap"] = visual_feature_tap
    torch.save(state, path)
    if ancestry_base is not None:
        doc = ca.extend_for_checkpoint(
            dict(ancestry_base), step=step, run_id=run_id,
            s1_run_manifest_hash=s1_run_manifest_hash,
        )
        doc = ca.finalize_checkpoint_document(doc, path)
        ca.write_sidecar(doc, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _lr_schedule_from_cli(
    args: Any, s4_manifest: Mapping[str, Any]
) -> lrs.LrSchedule | None:
    """The CLI's LR tail, refused unless it is exactly the frozen manifest's own
    `lr_schedule` block (absent == constant) -- same discipline as
    `--consistency-lambda`, on every invocation including `--resume`."""
    if args.lr_schedule == "constant":
        if args.lr_schedule_start_step is not None or args.lr_schedule_floor_milli is not None:
            raise S4TrainingRefused(
                "--lr-schedule-start-step/--lr-schedule-floor-milli were given without "
                "--lr-schedule linear_tail"
            )
        cli_block = None
    else:
        if args.lr_schedule_start_step is None or args.lr_schedule_floor_milli is None:
            raise S4TrainingRefused(
                f"--lr-schedule {args.lr_schedule} needs --lr-schedule-start-step and "
                "--lr-schedule-floor-milli"
            )
        cli_block = {
            "kind": args.lr_schedule,
            "start_step": args.lr_schedule_start_step,
            "floor_milli": args.lr_schedule_floor_milli,
        }
    manifest_block = s4_manifest.get("lr_schedule")
    if cli_block != manifest_block:
        raise S4TrainingRefused(
            f"--lr-schedule* {cli_block!r} does not match the frozen s4_run_manifest "
            f"lr_schedule {manifest_block!r} (absent == constant) -- this run (including a "
            "--resume) must be launched with the schedule it was frozen under"
        )
    try:
        schedule = lrs.from_manifest_block(cli_block)
        if schedule is not None:
            lrs.require_within_budget(schedule, total_steps=args.steps)
    except ValueError as exc:
        raise S4TrainingRefused(f"--lr-schedule start_step/floor: {exc}") from exc
    return schedule


def main(argv: list[str] | None = None) -> int:
    """Run S4 discovery admission steps 1-6 (design §7.3), build the model
    and optimizer, then run the training loop (`run_s4_training`, design
    §7.4) -- including `--resume` and the early typed refusal -- and the
    CLI (design §9).
    """
    import argparse

    from unflatten.model import composite_vocab as cv
    from unflatten.model import s2_plan as s2p
    from unflatten.model.candidate_loader import load_qwen_candidate

    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.s4_train", description=__doc__
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
    parser.add_argument(
        "--keep-every", type=int, default=DEFAULT_CHECKPOINT_KEEP_EVERY,
        help="DEC-041 point 4: keep a never-overwritten step-numbered checkpoint "
        "copy only every N steps (the 'latest' pointer at --out is always kept, "
        "unaffected). Default 10000.",
    )
    parser.add_argument("--preflight-rows", type=int, default=4)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="data prefetch worker processes; 0 = serial fetch in this process",
    )
    parser.add_argument(
        "--eval-every", type=int, default=0,
        help="DEC-041 point 3: run a diagnostic teacher-forced val eval every N steps; "
        "0 = never",
    )
    parser.add_argument(
        "--val-cache", type=Path, default=None,
        help="exact-text cache directory for the pinned val_iid cohort (required if "
        "--eval-every > 0)",
    )
    parser.add_argument(
        "--val-expectation", type=Path, default=None,
        help="frozen CacheExpectation payload for --val-cache (required if "
        "--eval-every > 0)",
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--eval-pages", type=int, default=300,
        help="DEC-041 point 3 (review MUST FIX 2): a PINNED, seeded subset of "
        "val_iid pages an eval scores -- never the whole val cache (a 10 000+ "
        "page cache would make one eval take days). Selected once, written to "
        "the run dir, and reused verbatim on every resume.",
    )
    parser.add_argument(
        "--eval-patience", type=int, default=3,
        help="DEC-041 point 5: consecutive worsening evals before OVERFIT_SUSPECTED",
    )
    parser.add_argument(
        "--stop-on-overfit", action="store_true",
        help="stop the run when OVERFIT_SUSPECTED fires -- an explicit deviation from "
        "RFC005 §12.2, off by default",
    )
    parser.add_argument(
        "--augmented-features", type=Path, default=None,
        help="T1b/DEC-041: an aug_feature_cache store dir. Required iff the "
        "page-uniform plan has augmented draws; refused otherwise.",
    )
    parser.add_argument(
        "--mask-inactive-union-arms", action="store_true",
        help="RFC005-ERRATA-008 E1 (PROPOSED, v16 only): drop a typed leaf whose "
        "pointer is not REQUIRED by its node's recorded union arm at collate "
        "(s4_typed_loss.collate_typed). Default off -- must match the frozen "
        "s4_run_manifest's own mask_inactive_union_arms (absent == off) on every "
        "invocation, including --resume, so the running v15b promotion cannot "
        "pick this up from a bare git pull.",
    )
    parser.add_argument(
        "--consistency-lambda", type=float, default=0.0,
        help="RFC004-ERRATA-004 item R3 (draft RFC004-ERRATA-006, v16 only): the "
        "weight of s4_typed_loss.section_margin_consistency_loss's additive term "
        "in the S4 total. Default 0.0 (off) -- at 0.0 the term is never computed, "
        "so the loss and its gradients are bit-identical to a run without this "
        "flag. Must match the frozen s4_run_manifest's own "
        "consistency_lambda_milli (absent == 0) on every invocation, including "
        "--resume.",
    )
    parser.add_argument(
        "--lr-schedule", choices=("constant", *lrs.LR_SCHEDULE_KINDS), default="constant",
        help="Dormant LR tail (model.lr_schedule). Default 'constant' -- the LR every frozen "
        "run used. 'linear_tail' needs --lr-schedule-start-step and "
        "--lr-schedule-floor-milli, and must match the frozen s4_run_manifest's own "
        "lr_schedule block (absent == constant) on every invocation, including --resume.",
    )
    parser.add_argument(
        "--eval-checkpoints", type=Path, nargs="+", default=None, metavar="CKPT",
        help="Eval-only mode: teacher-force each checkpoint (first = baseline) on this run's "
        "pinned eval pages, one page per batch, write a paired comparison to --eval-report "
        "and exit. Takes the run's own launch argv (minus --resume); no gradient step, no "
        "write into the run dir. Monitoring only (RFC005 §12.2).",
    )
    parser.add_argument(
        "--eval-report", type=Path, default=None,
        help="Where --eval-checkpoints writes its JSON report (refused if it exists).",
    )
    parser.add_argument("--lr-schedule-start-step", type=int, default=None)
    parser.add_argument("--lr-schedule-floor-milli", type=int, default=None)
    parser.add_argument(
        "--level-router-policy", type=Path, default=None,
        help="RFC007 variant B (PROPOSED, v16 only): a LevelRouterPolicy JSON file. "
        "Its policy_hash must match the frozen s4_run_manifest's own "
        "level_router_policy_hash (absent == no router) on every invocation, "
        "including --resume. Refused without --level-router-ratification.",
    )
    parser.add_argument(
        "--level-router-ratification", type=Path, default=None,
        help="RFC007: the owner's ratification record for this exact policy and "
        "stage S4 (docs/decisions/ratifications/). None exists today.",
    )
    parser.add_argument(
        "--structural-table-region",
        choices=list(stl.STRUCTURAL_TABLE_REGION_POLICIES),
        default="abstain_not_visible",
        help="DEC-055 V6b (PROPOSED, v16 dense fallback): 'exclude' stops training "
        "boxless structural table/row/cell records as 'not visible' in the region "
        "channel. Must match the frozen s4_run_manifest's structural_table_region "
        "(absent == abstain_not_visible) on every invocation, including --resume.",
    )
    parser.add_argument(
        "--collapsed-region-rule",
        choices=list(stl.COLLAPSED_REGION_RULES),
        default="abstain_container",
        help="RFC004-ERRATA-012 E1 (v17): 'exclude' stops training a node whose box the "
        "augmentation collapsed as a region record (bucket collapsed_by_view). Must match "
        "the frozen s4_run_manifest's collapsed_region_rule (absent == abstain_container) "
        "on every invocation, including --resume.",
    )
    parser.add_argument(
        "--external-views", type=Path, default=None, metavar="PATH",
        help="DEC-085 (external-views.v1 continuation): the sha-pinned table of external "
        "scan / photo / augraphy renderings of train pages that REPLACE a deterministic "
        "share of train draws (never eval). Needs a frozen s4_run_manifest whose "
        "state_continuation.kind is external-views.v1 and whose visual_input.external_views "
        "block matches this table; on every invocation, including --resume.",
    )
    parser.add_argument(
        "--story-decision-weight", type=float, default=1.0,
        help="RFC004-ERRATA-012 E3 (v17): weight of the L_structure tokens that decide the "
        "page's story pattern (default 1.0 = off). Must match the frozen s4_run_manifest's "
        "story_decision_weight_milli (absent == 1000) on every invocation, including --resume.",
    )
    parser.add_argument(
        "--alignment-type-weight", action="append", default=[], metavar="TYPE=FLOAT",
        help="RFC004-ERRATA-012 E2 (v17): L_alignment weight of one node type; repeatable. "
        "Must match the frozen s4_run_manifest's alignment_type_weights_milli (absent == {}) "
        "on every invocation, including --resume.",
    )
    parser.add_argument(
        "--explicit-zero-bin-r1", action="store_true",
        help="RFC004-ERRATA-006 E1 (R1, PROPOSED, v16): REFUSED in s4_train -- "
        "the explicit zero bin exists only on bins_plus_residual integer heads, "
        "and an S4 run pins typed_integer_codec direct_regression (D5); R1 is an "
        "S6 flag until an integer-codec decision for the dense fallback exists.",
    )
    parser.add_argument(
        "--visual-input-policy", type=Path, default=None,
        help="DEC-056 addendum A: train on the on-the-fly ViT path -- PATH is the "
        "VisualInputPolicy JSON the page-image (7.0.0) cache was built under "
        "(DeepStack on iff it names taps). Requires --page-image-store. Must match "
        "the frozen s4_run_manifest's visual_input block.",
    )
    parser.add_argument(
        "--page-image-store", type=Path, default=None,
        help="the object store the page-image cache's PNG references resolve in",
    )
    parser.add_argument(
        "--onfly-augmentation-seed", type=int, default=None,
        help="DEC-040 augmentation re-drawn per (seed, step, page) on the on-the-fly "
        "path; absent = no augmentation. Recorded in the run manifest's visual_input.",
    )
    parser.add_argument(
        "--onfly-augmentation-policy", type=Path, default=None,
        help="augmentation policy file (schema 1.0.0 or 2.0.0) for --onfly-augmentation-seed; "
        "its hash and id must equal the frozen manifest's visual_input.augmentation. "
        "Absent = the shipped v1 default.",
    )
    parser.add_argument(
        "--vit-prefetch-overlap", action="store_true",
        help="fresh S4 speed knob (default off, the pilot decides): issue the next step's "
        "ViT on a side CUDA stream before the current step's forward/backward. "
        "Bit-identical to the plain path, so not bound in the run manifest.",
    )
    parser.add_argument(
        "--vit-batch-pages", action="store_true",
        help="fresh S4 knob: one tower call per multi-page batch (a no-op at batch 1; "
        "used by eval with --eval-batch-size > 1). Bound in the manifest's visual_input.",
    )
    parser.add_argument(
        "--vit-feature-cache", type=Path, default=None,
        help="fresh S4 fallback (default off, the pilot decides): epoch-1 write-through "
        "fp8 cache of clean views' ViT outputs under DIR; epochs 2+ read it, augmented "
        "views are always recomputed. Bound in the manifest's visual_input.",
    )
    parser.add_argument(
        "--data-quality-gate", type=Path, default=None,
        help="DEC-057: the passing data-quality-gate.json binding the train (and val) "
        "cache. Required for caches built under a v16 target policy; its hash must "
        "equal the frozen s4_run_manifest's data_quality_gate_hash (absent == pre-v16).",
    )
    parser.add_argument(
        "--materialize-eval-rows", action="store_true",
        help="with --admission-only: also build (or verify) the persisted eval rows "
        "file in the run dir, so the first eval after the next start loads them in "
        "seconds instead of re-reading ~270 GB of val shards (28.9.)",
    )
    parser.add_argument(
        "--admission-only", action="store_true",
        help="run every admission check (steps 1-5, plan/aug-store binding, val "
        "cache) then print a one-line summary and exit 0 -- CPU-only, never "
        "loads the model. For the production launcher to call before docker stop.",
    )
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error(f"--num-workers must be >= 0, got {args.num_workers}")
    if args.eval_every < 0:
        parser.error(f"--eval-every must be >= 0, got {args.eval_every}")
    if args.consistency_lambda < 0:
        parser.error(f"--consistency-lambda must be >= 0, got {args.consistency_lambda}")
    if (args.visual_input_policy is None) != (args.page_image_store is None):
        parser.error("--visual-input-policy and --page-image-store go together")
    if args.onfly_augmentation_seed is not None and args.visual_input_policy is None:
        parser.error("--onfly-augmentation-seed needs --visual-input-policy")
    if args.onfly_augmentation_policy is not None and args.onfly_augmentation_seed is None:
        parser.error("--onfly-augmentation-policy needs --onfly-augmentation-seed")
    if args.visual_input_policy is None and (
        args.vit_prefetch_overlap or args.vit_batch_pages or args.vit_feature_cache is not None
    ):
        parser.error("the --vit-* knobs need --visual-input-policy (the on-the-fly path)")
    story_decision_weight_milli = _weight_milli_or_error(
        parser, "--story-decision-weight", args.story_decision_weight
    )
    alignment_type_weights_milli = _alignment_weights_milli_or_error(
        parser, args.alignment_type_weight
    )
    if args.explicit_zero_bin_r1:
        from unflatten.model.s4_run_manifest import R1_REFUSAL

        parser.error(R1_REFUSAL)
    if args.eval_every and (args.val_cache is None or args.val_expectation is None):
        parser.error("--eval-every > 0 requires both --val-cache and --val-expectation")
    if args.eval_every and args.out and args.checkpoint_every and (
        args.eval_every % args.checkpoint_every
    ):
        # checkpoint-best-eval.pt is copied at the checkpoint save on the best
        # eval's own step -- an eval step with no save would never be kept.
        parser.error(
            f"--eval-every {args.eval_every} must be a multiple of --checkpoint-every "
            f"{args.checkpoint_every} when --out is set"
        )

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

    # --- step 2: the S4 manifest is bound to this companion --------------
    s4_manifest = load_s4_run_manifest(run_dir / "s4_run_manifest.json")
    try:
        s2t.require_step2_bound_to_companion(s2_manifest=s4_manifest, companion=companion)
    except s2t.S2TrainingRefused as exc:
        raise S4TrainingRefused(str(exc)) from exc
    lr_reduced_micro = int(round(args.lr_reduced * 1e6))
    if lr_reduced_micro != s4_manifest["lr_reduced_micro"]:
        raise S4TrainingRefused(
            f"--lr-reduced {args.lr_reduced!r} (lr_reduced_micro {lr_reduced_micro}) "
            "does not match the frozen s4_run_manifest lr_reduced_micro "
            f"{s4_manifest['lr_reduced_micro']}"
        )
    # RFC005-ERRATA-008 E1 (PROPOSED): a manifest frozen before this field
    # existed has no key at all -- absent == off, never a silent True.
    manifest_mask_inactive_union_arms = s4_manifest.get("mask_inactive_union_arms", False)
    if args.mask_inactive_union_arms != manifest_mask_inactive_union_arms:
        raise S4TrainingRefused(
            f"--mask-inactive-union-arms {args.mask_inactive_union_arms!r} does not "
            "match the frozen s4_run_manifest mask_inactive_union_arms "
            f"{manifest_mask_inactive_union_arms!r} -- this run (including a "
            "--resume) must be launched with the same flag it was started under"
        )
    # R3 (RFC004-ERRATA-004 item R3, draft RFC004-ERRATA-006): same pattern
    # as mask_inactive_union_arms -- a manifest frozen before this field
    # existed has no key at all; absent == 0 (off), never a silent nonzero.
    cli_consistency_lambda_milli = int(round(args.consistency_lambda * 1000))
    manifest_consistency_lambda_milli = s4_manifest.get("consistency_lambda_milli", 0)
    if cli_consistency_lambda_milli != manifest_consistency_lambda_milli:
        raise S4TrainingRefused(
            f"--consistency-lambda {args.consistency_lambda!r} "
            f"(consistency_lambda_milli {cli_consistency_lambda_milli}) does not "
            "match the frozen s4_run_manifest consistency_lambda_milli "
            f"{manifest_consistency_lambda_milli!r} -- this run (including a "
            "--resume) must be launched with the same flag it was started under"
        )
    manifest_story_milli = s4_manifest.get("story_decision_weight_milli", 1000)
    if story_decision_weight_milli != manifest_story_milli:
        raise S4TrainingRefused(
            f"--story-decision-weight {args.story_decision_weight!r} "
            f"(story_decision_weight_milli {story_decision_weight_milli}) does not match the "
            f"frozen s4_run_manifest story_decision_weight_milli {manifest_story_milli!r} -- "
            "this run (including a --resume) must be launched with the same flag it was "
            "started under"
        )
    require_story_weight_standalone_profile(
        story_decision_weight_milli=story_decision_weight_milli, cache_identity=identity
    )
    manifest_type_milli = s4_manifest.get("alignment_type_weights_milli", {})
    if alignment_type_weights_milli != manifest_type_milli:
        raise S4TrainingRefused(
            f"--alignment-type-weight {alignment_type_weights_milli!r} (milli) does not match "
            f"the frozen s4_run_manifest alignment_type_weights_milli {manifest_type_milli!r} "
            "-- this run (including a --resume) must be launched with the same flags it was "
            "started under"
        )
    lr_schedule = _lr_schedule_from_cli(args, s4_manifest)
    level_router_policy = require_level_router_admission(
        s4_manifest=s4_manifest,
        policy_path=args.level_router_policy,
        ratification_record=args.level_router_ratification,
    )
    # v16 (s4_run_manifest 1.4.0): V6b, R1 and the on-the-fly visual input,
    # each absent == the v15b behaviour, each cross-checked like the two above.
    visual_policy, onfly_augmentation = _v16_cli_bindings(args, s4_manifest)
    if args.external_views is not None:
        _require_external_views_admissible(s4_manifest, visual_policy)
    if visual_policy is not None and not etc_is_page_image_identity(identity):
        raise S4TrainingRefused(
            "--visual-input-policy needs a page-image (7.0.0) cache; this one holds features"
        )
    if visual_policy is not None and identity["feature_tap"] != visual_policy.feature_tap():
        raise S4TrainingRefused(
            f"the cache was built under visual input {identity['feature_tap']!r}, not "
            f"this policy's {visual_policy.feature_tap()!r}"
        )
    if visual_policy is None and etc_is_page_image_identity(identity):
        raise S4TrainingRefused(
            "a page-image cache holds no features: pass --visual-input-policy"
        )
    expected_tap = (
        visual_policy.visual_feature_tap() if visual_policy is not None else ov.POOLER_OUTPUT_TAP
    )

    # DEC-057 point 6: the caches this run reads, through the gate; `None`
    # (pre-v16 caches) must meet an absent manifest key, a v16 hash its own.
    gate_manifests = {"train": dataset.reader.manifest}
    if args.eval_every:
        gate_manifests["val_iid"] = cache_manifest(args.val_cache)
    try:
        gate_hash = data_quality_gate_hash_for(gate_manifests, args.data_quality_gate)
    except S4RunManifestInvalid as exc:
        raise S4TrainingRefused(str(exc)) from exc
    if gate_hash != s4_manifest.get("data_quality_gate_hash"):
        raise S4TrainingRefused(
            f"data-quality gate hash {gate_hash!r} does not match the frozen "
            f"s4_run_manifest data_quality_gate_hash "
            f"{s4_manifest.get('data_quality_gate_hash')!r}"
        )

    # --- step 3: committed (or, page-uniform, EFFECTIVE) policies ---------
    # A page-uniform run's actual sampler/exposure policies are NOT the
    # committed files -- `s4_plan.build_page_uniform_plan` rewrites
    # `virtual_epoch_size`/`max_views_per_lineage_per_virtual_epoch` to the
    # page-cohort budget and hashes THOSE into the plan. Verifying against
    # the committed files here would recompute a DIFFERENT hash and refuse
    # a truthfully-frozen plan at admission (the review's MUST FIX 1) --
    # so this step must read whichever files `s4_plan` actually hashed,
    # which it writes to the run dir as `s4p.EFFECTIVE_*_POLICY_FILENAME`
    # in page-uniform mode. Weighted mode is unchanged: committed files.
    page_uniform_meta_path = run_dir / s4p._PAGE_UNIFORM_META_FILENAME
    page_uniform_meta = (
        json.loads(page_uniform_meta_path.read_text())
        if page_uniform_meta_path.exists()
        else None
    )
    matrix = load_run_matrix(run_dir)
    if page_uniform_meta is not None:
        sampler_policy = load_s2_sampler_policy(
            run_dir / s4p.EFFECTIVE_SAMPLER_POLICY_FILENAME
        )
        exposure_policy = load_s2_exposure_policy(
            run_dir / s4p.EFFECTIVE_EXPOSURE_POLICY_FILENAME
        )
    else:
        sampler_policy = load_s2_sampler_policy(_POLICIES_DIR / "s4_sampler_policy.json")
        exposure_policy = load_s2_exposure_policy(_POLICIES_DIR / "s4_exposure_policy.json")
    stage_policy = spolicy.load_stage_policy(_POLICIES_DIR / "s4_stage_policy.json")
    plan = json.loads((run_dir / "ordered_exposure_plan.json").read_text())
    try:
        s2t.require_step3_policies_bound(
            s2_manifest=s4_manifest,
            matrix=matrix,
            sampler_policy=sampler_policy,
            exposure_policy=exposure_policy,
            plan=plan,
        )
    except s2t.S2TrainingRefused as exc:
        raise S4TrainingRefused(str(exc)) from exc
    stage_policy_hash = spolicy.stage_policy_hash(stage_policy)
    if stage_policy_hash != s4_manifest["s4_stage_policy_hash"]:
        raise S4TrainingRefused(
            f"committed s4_stage_policy_hash ({stage_policy_hash}) does not match "
            f"the pinned s4_run_manifest value ({s4_manifest['s4_stage_policy_hash']})"
        )

    # --- step 4: every draw resolves inside the training dataset ---------
    require_boost_block_bound(page_uniform_meta=page_uniform_meta, s4_manifest=s4_manifest)
    try:
        s2t.require_plan_covers_budget(plan=plan, steps=args.steps, batch_size=args.batch_size)
        split_records = s2p.load_split_records(args.release_dir)
        plan_pages, plan_view_kind, plan_view_id = resolve_plan_pages_for_run(
            page_uniform_meta=page_uniform_meta,
            plan=plan,
            expectation=expectation,
            split_records=split_records,
        )
        plan_order = s2t.plan_dataset_order(plan_pages, expectation)
    except s2t.S2TrainingRefused as exc:
        raise S4TrainingRefused(str(exc)) from exc
    plan_lineages = [draw["lineage_root_id"] for draw in plan["draws"]]
    cells = plan_cells(plan)

    # --- augmented feature store (T1b, page-uniform mode only) -----------
    has_augmented_draws = plan_view_kind is not None and "augmented" in plan_view_kind
    augmented_features: afc.AugFeatureCacheReader | None = None
    if has_augmented_draws and args.augmented_features is None:
        raise S4TrainingRefused(
            "the page-uniform plan has augmented draws but --augmented-features "
            "was not given"
        )
    if args.augmented_features is not None and not has_augmented_draws:
        raise S4TrainingRefused(
            "--augmented-features was given but the plan has no augmented draws"
        )
    if args.augmented_features is not None:
        augmented_features = afc.AugFeatureCacheReader(args.augmented_features)
        augmented_features.require_bound_to(compute_exact_text_cache_identity_hash(identity))

    # --- step 5: typed supervision census / floor gate --------------------
    census = json.loads((run_dir / "typed_supervision_census.json").read_text())
    require_step5_typed_supervision(
        s4_manifest=s4_manifest,
        companion=companion,
        stage_policy=stage_policy,
        census=census,
    )

    # --- val cache + expectation (moved ahead of model load: pure CPU file
    # I/O, and `--admission-only` must be able to verify it without ever
    # touching the GPU) --------------------------------------------------
    eval_dataset: stl.S4TrainingDataset | _EvalRowCache | None = None
    eval_order: list[int] | None = None
    eval_pages_hash: str | None = None
    eval_pages: list[PageKey] = []
    if args.eval_every:
        val_expectation = CacheExpectation.from_payload(
            json.loads(args.val_expectation.read_text())
        )
        val_base = s1_train.ExactTextTrainingDataset(
            args.val_cache, gradient_bearing=False, expectation=val_expectation
        )
        eval_dataset = stl.S4TrainingDataset(val_base)
        try:
            val_page_lineage = s4p.build_page_lineage_map(
                val_expectation, split_records, split_name="val_iid"
            )
        except s2p.S2PlanRefused as exc:
            raise S4TrainingRefused(f"val expectation: {exc}") from exc
        eval_pages, eval_pages_hash = load_or_select_eval_pages(
            run_dir=run_dir,
            all_pages=val_expectation.target_keys(),
            page_lineage=val_page_lineage,
            eval_pages=args.eval_pages,
            run_seed=args.seed,
        )
        eval_order = s2t.plan_dataset_order(eval_pages, val_expectation)
        eval_rows_key = hashlib.sha256(
            json.dumps(
                {
                    "schema": "s4-eval-rows/1",
                    "eval_pages_hash": eval_pages_hash,
                    "val_cache_manifest": _sha256_file(args.val_cache / "manifest.json"),
                    "val_expectation": _sha256_file(args.val_expectation),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        eval_dataset = _EvalRowCache(
            eval_dataset,
            eval_order,
            persist_path=run_dir / _EVAL_ROWS_FILENAME,
            persist_key=eval_rows_key,
        )

    # DEC-085: the table is read, verified and counted here -- once, in the parent, before
    # any DataLoader worker forks; workers inherit the loaded object, they never load it.
    external: ExternalViews | None = None
    if args.external_views is not None:
        try:
            external = ExternalViews.load(
                args.external_views,
                train_pages=set(expectation.target_keys()),
                plan_pages=set(plan_pages),
                eval_pages=set(eval_pages),
                collapsed_region_rule=args.collapsed_region_rule,
            )
        except ExternalViewsRefused as exc:
            raise S4TrainingRefused(str(exc)) from exc
        # The full block (table sha, seed, fractions, entry counts) against the manifest.
        _v16_cli_bindings(args, s4_manifest, external_views=external.manifest_fields())
    sc.require_mode(s4_manifest, resume=args.resume, init_from=args.init_from,
                    eval_only=args.eval_checkpoints is not None)
    continuation = sc.load_context(run_dir, s4_manifest)
    require_resume_init_from_agreement(
        cli_resume=args.resume,
        manifest_init_from=s4_manifest["init_from"],
        cli_init_from=args.init_from,
    )
    require_init_from_ancestry_lineage(
        ancestry=ancestry, init_from=s4_manifest["init_from"]
    )
    resume_step: int | None = None
    if args.resume and args.admission_only:
        # The model-free half of the resume path (review 2026-09-24): the
        # checkpoint's own refusals plus ledger/step and reduced-LR agreement,
        # on a CPU load of the checkpoint -- so RESUME=1 is refused before
        # `docker stop`, not after. The real resume below repeats them.
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        s1_train.verify_checkpoint_state(
            args.resume,
            state,
            expect_cache_identity=identity,
            expect_feature_source=dataset.feature_source,
            **sc.checkpoint_expectations(
                context=continuation, path=args.resume, state=state,
                companion_hash=companion["s1_run_manifest_hash"],
                base_ancestry_hash=ancestry["ancestry_hash"],
            ),
        )
        resume_step = int(state["step"])
        _require_tap(state, expected_tap, what=f"--resume {args.resume}")
        try:
            resume_ledger = ex.ExposureLedger.load_jsonl(
                run_dir / _LEDGER_FILENAME, s4_manifest["run_id"]
            )
            if continuation is None:
                s2t.require_resume_ledger_consistent(
                    ledger=resume_ledger, step=resume_step, batch_size=args.batch_size,
                )
            else:
                sc.require_ledger(ledger=resume_ledger, step=resume_step,
                                  batch_size=args.batch_size, start_step=continuation.step)
        except (s2t.S2TrainingRefused, sc.ContinuationRefused) as exc:
            raise S4TrainingRefused(str(exc)) from exc
        saved_reduced_lr = state["optimizer_state_dict"]["param_groups"][1]["lr"]
        expected_reduced_lr = lrs.expected_group_lrs(
            lr_schedule, resume_step, total_steps=args.steps,
            bases=(args.lr, args.lr_reduced),
        )[1]
        if saved_reduced_lr != expected_reduced_lr:
            raise S4TrainingRefused(
                f"checkpoint's reduced-LR group is at {saved_reduced_lr!r}, not "
                f"{expected_reduced_lr!r} (--lr-reduced {args.lr_reduced!r}"
                f"{'' if lr_schedule is None else f' at the schedule at step {resume_step}'})"
                " -- resume must reassert the reduced LR"
            )

    if args.eval_checkpoints is not None:
        if args.eval_report is None:
            raise S4TrainingRefused("--eval-checkpoints requires --eval-report")
        if args.eval_report.exists():
            raise S4TrainingRefused(f"--eval-report {args.eval_report} already exists")
        if not args.eval_every:
            raise S4TrainingRefused(
                "--eval-checkpoints requires --eval-every (it scores the pinned eval pages)"
            )
        if args.resume or args.admission_only:
            raise S4TrainingRefused(
                "--eval-checkpoints cannot be combined with --resume or --admission-only"
            )
        if args.eval_batch_size != 1:
            raise S4TrainingRefused(
                "--eval-checkpoints needs --eval-batch-size 1 (a batch is a page)"
            )
        if len(args.eval_checkpoints) < 2:
            raise S4TrainingRefused(
                "--eval-checkpoints needs at least two checkpoints to compare"
            )
    elif args.eval_report is not None:
        raise S4TrainingRefused("--eval-report is only used with --eval-checkpoints")

    if args.materialize_eval_rows and not args.admission_only:
        raise S4TrainingRefused("--materialize-eval-rows requires --admission-only")
    eval_rows_source: str | None = None
    if args.admission_only and args.materialize_eval_rows:
        if not isinstance(eval_dataset, _EvalRowCache):
            raise S4TrainingRefused("--materialize-eval-rows requires --eval-every")
        eval_rows_source = eval_dataset.materialize()

    external_counts: dict[str, Any] | None = None
    if external is not None:
        # Exact, from the plan in memory: what this run will draw, not an estimate.
        stop = min(len(plan_pages), args.steps * args.batch_size)
        if resume_step is None and args.resume:
            # A real resume (not --admission-only) learns its step only after the
            # model build; read just that integer now so the window is the same.
            resume_step = _checkpoint_step(args.resume)
        # Draw indices, not steps: a resume at step N has already consumed N * batch_size.
        first_draw = min((resume_step or 0) * args.batch_size, stop)
        external_counts = external.count(plan_pages, first_draw, stop)
        if external_counts["share_milli"] > MAX_SHARE_MILLI:
            raise S4TrainingRefused(
                f"external views would replace {external_counts['share_milli']} milli of the "
                f"draws, above the {MAX_SHARE_MILLI} milli cap (DEC-085)"
            )

    if args.admission_only:
        # Every admission check up to here has already run (steps 1-5,
        # plan/aug-store binding, val cache); the model is never loaded
        # onto the GPU and no step is taken. The production launcher calls
        # this BEFORE `docker stop`, so a bad argument or an unadmittable
        # plan never costs a service outage.
        print(
            json.dumps(
                {
                    "admission_only": True,
                    "steps": args.steps,
                    "plan_draws": len(plan["draws"]),
                    "clean_draws": (
                        sum(1 for k in plan_view_kind if k == "clean")
                        if plan_view_kind is not None
                        else None
                    ),
                    "augmented_draws": (
                        sum(1 for k in plan_view_kind if k == "augmented")
                        if plan_view_kind is not None
                        else None
                    ),
                    "eval_pages": len(eval_order) if eval_order is not None else None,
                    "eval_rows": eval_rows_source,
                    "resume_step": resume_step,
                    **(
                        {}
                        if external_counts is None
                        else {
                            "external_views": {
                                **external_counts,
                                "table_sha256": external.table_sha256,
                            }
                        }
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    # --- model build; step 6 ----------------------------------------------
    candidate = load_qwen_candidate(device=args.device)
    model = QwenDenseCandidateModel(
        decode=candidate.decode,
        inject_visual=candidate.inject_visual,
        visual_dim=candidate.visual_hidden_size,
        d_model=candidate.hidden_size,
        ranges=cv.packed_ranges(candidate.vocab_size),
        qwen_embedding_weight=candidate.embed_tokens.weight,
        qwen_lm_head_weight=candidate.lm_head_weight,
        variant=level_router_variant(level_router_policy),
        # Only when selected: the v15b model is built with exactly the
        # keyword set it always was.
        **(
            {"deepstack": True, "image_token_id": candidate.image_token_id}
            if visual_policy is not None and visual_policy.deepstack
            else {}
        ),
    )
    model.to(args.device)
    visual_encoder: ov.OnTheFlyVisualEncoder | None = None
    processor: Any = None
    if visual_policy is not None:
        from transformers import AutoImageProcessor

        from unflatten.model.candidate_loader import (
            QWEN_CANDIDATE_MODEL_ID,
            QWEN_CANDIDATE_REVISION,
        )

        processor = AutoImageProcessor.from_pretrained(
            QWEN_CANDIDATE_MODEL_ID, revision=QWEN_CANDIDATE_REVISION
        )
        visual_encoder = ov.OnTheFlyVisualEncoder(
            candidate, policy=visual_policy, device=args.device,
            batch_pages=args.vit_batch_pages,
            feature_cache=(
                None
                if args.vit_feature_cache is None
                else ov.ViTFeatureCache(args.vit_feature_cache, policy=visual_policy)
            ),
        )

    unfreeze_and_refreeze_typed_heads(model, s4_manifest["heads_frozen_for_no_supervision"])
    if args.init_from is not None:
        init_report = load_init_weights(
            args.init_from,
            model,
            pinned=s4_manifest["init_from"],
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
                        "replaced_keys": list(init_report.replaced_keys),
                    }
                }
            ),
            flush=True,
        )
    require_step6_stage_state(
        model=model,
        run_class=s4_manifest["run_class"],
        heads_frozen_for_no_supervision=s4_manifest["heads_frozen_for_no_supervision"],
    )
    require_trainable_typed_half(s2t._model_group_states(model))

    # --- step 7: optimizer; --resume or fresh; preflight (design §7.3) ---
    optimizer = build_s4_optimizer(
        model,
        lr_standard=args.lr,
        lr_reduced=args.lr_reduced,
        algorithm=args.optimizer,
        run_class=s4_manifest["run_class"],
    )
    typed_dataset = stl.S4TrainingDataset(dataset)

    start_step = 0
    ledger = ex.ExposureLedger(s4_manifest["run_id"])
    if args.resume:
        # `--init-from` is never presented here (require_resume_init_from_
        # agreement already refused that combination): `load_checkpoint`
        # restores weights, optimizer and RNG in one step, so
        # `load_init_weights` is skipped entirely on the resume path.
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        _require_tap(resume_state, expected_tap, what=f"--resume {args.resume}")
        resume_expectations = sc.checkpoint_expectations(
            context=continuation, path=args.resume, state=resume_state,
            companion_hash=companion["s1_run_manifest_hash"],
            base_ancestry_hash=ancestry["ancestry_hash"],
        )
        del resume_state
        progress, _state = s1_train.load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            expect_cache_identity=identity,
            expect_feature_source=dataset.feature_source,
            **resume_expectations,
        )
        start_step = progress.step
        ledger = ex.ExposureLedger.load_jsonl(
            run_dir / _LEDGER_FILENAME, s4_manifest["run_id"]
        )
        try:
            if continuation is None:
                s2t.require_resume_ledger_consistent(
                    ledger=ledger, step=start_step, batch_size=args.batch_size
                )
            else:
                sc.require_ledger(ledger=ledger, step=start_step,
                                  batch_size=args.batch_size, start_step=continuation.step)
        except (s2t.S2TrainingRefused, sc.ContinuationRefused) as exc:
            raise S4TrainingRefused(str(exc)) from exc
        # Design §5: the reduced LR is pinned in the S4 manifest and
        # re-asserted at resume by reading the restored optimizer's own
        # group 1 -- a checkpoint whose optimizer state disagrees (e.g. an
        # LR schedule that moved it) must not silently keep whatever it
        # carried.
        resumed_reduced_lr = optimizer.param_groups[1]["lr"]
        expected_reduced_lr = lrs.expected_group_lrs(
            lr_schedule, start_step, total_steps=args.steps,
            bases=(args.lr, args.lr_reduced),
        )[1]
        if resumed_reduced_lr != expected_reduced_lr:
            raise S4TrainingRefused(
                f"resumed optimizer's reduced-LR group is at "
                f"{resumed_reduced_lr!r}, not {expected_reduced_lr!r} "
                f"(--lr-reduced {args.lr_reduced!r}"
                f"{'' if lr_schedule is None else f' at the schedule at step {start_step}'}) "
                "-- resume must reassert the reduced LR"
            )

    preflight_indices = (
        []
        if args.eval_checkpoints is not None  # eval-only: no gradient step, nothing to preflight
        else s1_train.preflight_row_indices(len(dataset), args.preflight_rows)
    )
    if preflight_indices:
        preflight_samples = [typed_dataset[i] for i in preflight_indices]
        if visual_encoder is not None:
            preflight_samples = [
                ov.attach_pixels(
                    sample, store_dir=args.page_image_store, processor=processor,
                    policy=visual_policy,
                )
                for sample in preflight_samples
            ]
        report = s1_train.preflight_native_signal(
            model,
            [sample.base for sample in preflight_samples],
            image_token_id=candidate.image_token_id,
            device=args.device,
            **({} if visual_encoder is None else {"visual_encoder": visual_encoder}),
        )
        print(json.dumps({"preflight": report}), flush=True)
        s1_train.require_preflight_passes(report, gradient_bearing=True)

    # SIGTERM = "stop after this step, checkpoint it" (the launcher's 05:30
    # stop); the loop polls the flag once per step.
    stop_signal = {"received": False}

    def _on_sigterm(signum: int, frame: Any) -> None:
        stop_signal["received"] = True
        print(json.dumps({"stop_requested": {"signal": signum}}), flush=True)

    # Restored on the way out: an in-process caller (the test suite) must not
    # keep a handler that swallows SIGTERM, nor hand it to processes it forks later.
    train_dataset: Any = typed_dataset
    onfly_kwargs: dict[str, Any] = {}
    if visual_encoder is not None:
        train_dataset = ov.OnTheFlyDrawDataset(
            typed_dataset, plan_order=plan_order, batch_size=args.batch_size,
            store_dir=args.page_image_store, processor=processor,
            policy=visual_policy, augmentation=onfly_augmentation,
            **({} if external is None else {"external": external}),
        )
        onfly_kwargs["visual_encoder"] = visual_encoder
        if args.vit_prefetch_overlap:
            onfly_kwargs["visual_prefetch"] = True
        if eval_dataset is not None:
            val_rows = _onfly_eval_rows(eval_dataset)
            eval_dataset = ov.OnTheFlyDrawDataset(
                val_rows, plan_order=range(len(val_rows)), batch_size=1,
                store_dir=args.page_image_store, processor=processor, policy=visual_policy,
            )
    if external is not None:
        onfly_kwargs["mask_color_unseen"] = True
    if args.structural_table_region != "abstain_not_visible":
        onfly_kwargs["structural_table_region"] = args.structural_table_region
    if args.collapsed_region_rule != "abstain_container":
        onfly_kwargs["collapsed_region_rule"] = args.collapsed_region_rule
    # The manifest-bound milli values (cross-checked above) are the weights trained with.
    loss_weight_kwargs: dict[str, Any] = {}
    if story_decision_weight_milli != 1000:
        loss_weight_kwargs["story_decision_weight"] = story_decision_weight_milli / 1000
    if alignment_type_weights_milli:
        loss_weight_kwargs["alignment_type_weights"] = {
            t: m / 1000 for t, m in alignment_type_weights_milli.items()
        }
    onfly_kwargs.update(loss_weight_kwargs)
    if args.eval_checkpoints is not None:
        def _verify(path: Path, state: Mapping[str, Any]) -> None:
            s1_train.verify_checkpoint_state(
                path,
                state,
                expect_cache_identity=identity,
                expect_feature_source=dataset.feature_source,
                expect_s1_run_manifest_hash=companion["s1_run_manifest_hash"],
                expect_base_ancestry_hash=ancestry["ancestry_hash"],
            )
            _require_tap(state, expected_tap, what=f"--eval-checkpoints {path}")

        report = evaluate_checkpoints(
            model=model,
            checkpoints=args.eval_checkpoints,
            dataset=eval_dataset,
            indices=eval_order or (),
            eval_pages=eval_pages,
            verify=_verify,
            evaluate_kwargs={
                "image_token_id": candidate.image_token_id,
                "stage_policy": stage_policy,
                "allowed_head_ids": l_typed_head_ids(stage_policy),
                "device": args.device,
                "mask_inactive_union_arms": args.mask_inactive_union_arms,
                "consistency_lambda": args.consistency_lambda,
                "structural_table_region": args.structural_table_region,
                **(
                    {}
                    if args.collapsed_region_rule == "abstain_container"
                    else {"collapsed_region_rule": args.collapsed_region_rule}
                ),
                **loss_weight_kwargs,
                **({} if visual_encoder is None else {"visual_encoder": visual_encoder}),
            },
        )
        report["eval_pages_hash"] = eval_pages_hash
        report["note"] = (
            "RFC005 §12.2: teacher-forced, monitoring only -- never selects or promotes a "
            "checkpoint; free-running decode is the selection signal"
        )
        tmp = args.eval_report.with_name(args.eval_report.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        tmp.replace(args.eval_report)
        print(json.dumps({"eval_checkpoints_report": str(args.eval_report)}), flush=True)
        return 0
    previous_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        run_s4_training(
            run_dir=run_dir,
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            image_token_id=candidate.image_token_id,
            plan_pages=plan_pages,
            plan_lineages=plan_lineages,
            plan_cells=cells,
            plan_order=plan_order,
            companion_hash=companion["s1_run_manifest_hash"],
            s4_run_id=s4_manifest["run_id"],
            steps=args.steps,
            batch_size=args.batch_size,
            stage_policy=stage_policy,
            allowed_head_ids=l_typed_head_ids(stage_policy),
            grad_clip=args.grad_clip,
            device=args.device,
            checkpoint_every=args.checkpoint_every,
            checkpoint_keep_every=args.keep_every,
            out=args.out,
            start_step=start_step,
            ledger=ledger,
            ancestry_base=ancestry,
            integrity={
                "s4_run_manifest": s4_manifest["s4_run_manifest_hash"],
                "ordered_exposure_plan": plan["ordered_exposure_plan_hash"],
            },
            num_workers=args.num_workers,
            eval_every=args.eval_every,
            eval_dataset=eval_dataset,
            eval_order=eval_order,
            eval_batch_size=args.eval_batch_size,
            eval_pages_hash=eval_pages_hash,
            overfit_patience=args.eval_patience,
            stop_on_overfit=args.stop_on_overfit,
            plan_view_kind=plan_view_kind,
            plan_view_id=plan_view_id,
            augmented_features=augmented_features,
            stop_requested=lambda: stop_signal["received"],
            mask_inactive_union_arms=args.mask_inactive_union_arms,
            consistency_lambda=args.consistency_lambda,
            lr_schedule=lr_schedule,
            lr_bases=(args.lr, args.lr_reduced),
            **onfly_kwargs,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


def etc_is_page_image_identity(identity: Mapping[str, Any]) -> bool:
    from unflatten.model.exact_text_cache import is_page_image_identity

    return is_page_image_identity(identity)


def _checkpoint_step(path: Path) -> int:
    """A checkpoint's step without loading its tensors: the ancestry sidecar's
    `checkpoint.step` when present, else the `step` of an mmap-ed load."""
    sidecar = ca.sidecar_path(Path(path))
    if sidecar.is_file():
        return int(json.loads(sidecar.read_text(encoding="utf-8"))["checkpoint"]["step"])
    return int(torch.load(path, map_location="cpu", weights_only=False, mmap=True)["step"])


def _require_tap(state: Mapping[str, Any], expected: str, *, what: str) -> None:
    try:
        ov.require_checkpoint_visual_feature_tap(state, expected=expected, what=what)
    except ov.OnTheFlyVisualError as exc:
        raise S4TrainingRefused(str(exc)) from exc


def _weight_milli_or_error(parser: Any, flag: str, weight: float) -> int:
    """A loss weight as the integer milli the manifest binds; `parser.error`
    unless it is positive and still >= 1 milli after rounding (a weight that
    rounds to 0 would mean 'no label', not 'small')."""
    if not (weight > 0 and math.isfinite(weight)):
        parser.error(f"{flag} must be > 0 and finite, got {weight!r}")
    milli = int(round(weight * 1000))
    if milli < 1:
        parser.error(f"{flag} must be at least 0.001 (it rounds to 0 milli), got {weight!r}")
    return milli


def _alignment_weights_milli_or_error(parser: Any, specs: Sequence[str]) -> dict[str, int]:
    weights: dict[str, int] = {}
    flag = "--alignment-type-weight"
    for spec in specs:
        node_type, sep, raw = spec.partition("=")
        try:
            value = float(raw)
        except ValueError:
            value = None
        if not sep or not node_type or value is None:
            parser.error(f"{flag} expects TYPE=FLOAT, got {spec!r}")
        if node_type in weights:
            parser.error(f"{flag} given twice for {node_type!r}")
        weights[node_type] = _weight_milli_or_error(parser, f"{flag} {node_type}", value)
    return weights


def _build_onfly_augmentation(
    args: Any, s4_manifest: Mapping[str, Any]
) -> ov.OnTheFlyAugmentation:
    """The augmentation `--onfly-augmentation-seed` names: the shipped v1 default, or
    the policy file of `--onfly-augmentation-policy` (RFC003-ERRATA-004), whose hash and
    id must be the ones the manifest was frozen with."""
    policy_path = getattr(args, "onfly_augmentation_policy", None)
    if policy_path is None:
        return ov.OnTheFlyAugmentation(seed=args.onfly_augmentation_seed)
    try:
        policy = aug_policy.load_policy(policy_path)
    except (OSError, ValueError) as exc:
        raise S4TrainingRefused(
            f"--onfly-augmentation-policy {str(policy_path)!r} cannot be loaded: {exc}"
        ) from exc
    augmentation = ov.OnTheFlyAugmentation(policy=policy, seed=args.onfly_augmentation_seed)
    frozen = (s4_manifest.get("visual_input") or {}).get("augmentation") or {}
    for key in ("policy_hash", "policy_id"):
        if frozen.get(key) != augmentation.manifest_fields()[key]:
            raise S4TrainingRefused(
                f"the augmentation policy file {str(policy_path)!r} has {key} "
                f"{augmentation.manifest_fields()[key]!r}, the frozen s4_run_manifest "
                f"visual_input.augmentation has {frozen.get(key)!r}"
            )
    return augmentation


def _require_external_views_admissible(
    s4_manifest: Mapping[str, Any], visual_policy: ov.VisualInputPolicy | None
) -> None:
    """`--external-views` only on the on-the-fly path of a frozen external-views.v1 child."""
    kind = (s4_manifest.get("state_continuation") or {}).get("kind")
    if kind != sc.KIND_EXTERNAL_VIEWS:
        raise S4TrainingRefused(
            f"--external-views needs a frozen s4_run_manifest whose state_continuation.kind is "
            f"{sc.KIND_EXTERNAL_VIEWS!r}, this one has {kind!r}"
        )
    if visual_policy is None:
        raise S4TrainingRefused(
            "--external-views needs the on-the-fly visual path (--visual-input-policy)"
        )


def _v16_cli_bindings(
    args: Any,
    s4_manifest: Mapping[str, Any],
    external_views: Mapping[str, Any] | None = None,
) -> tuple[ov.VisualInputPolicy | None, ov.OnTheFlyAugmentation | None]:
    """Cross-check the v16 flags against the frozen manifest (1.4.0 keys,
    absent == v15b) and build the visual policy + augmentation they name.

    `external_views` is the loaded table's `manifest_fields()`. `--external-views` is
    loaded only once the plan and the eval pages are known, so a first call without
    the fields (flag given) compares everything but that sub-block; `main` repeats the call
    with them for the full equality."""
    manifest_region = s4_manifest.get("structural_table_region", "abstain_not_visible")
    if args.structural_table_region != manifest_region:
        raise S4TrainingRefused(
            f"--structural-table-region {args.structural_table_region!r} does not match "
            f"the frozen s4_run_manifest structural_table_region {manifest_region!r}"
        )
    cli_collapsed = getattr(args, "collapsed_region_rule", "abstain_container")
    manifest_collapsed = s4_manifest.get("collapsed_region_rule", "abstain_container")
    if cli_collapsed != manifest_collapsed:
        raise S4TrainingRefused(
            f"--collapsed-region-rule {cli_collapsed!r} does not match "
            f"the frozen s4_run_manifest collapsed_region_rule {manifest_collapsed!r}"
        )
    policy = None
    augmentation = None
    if args.visual_input_policy is not None:
        policy = ov.VisualInputPolicy.from_payload(
            json.loads(Path(args.visual_input_policy).read_text())
        )
        if args.onfly_augmentation_seed is not None:
            augmentation = _build_onfly_augmentation(args, s4_manifest)
    cli_block = (
        None
        if policy is None
        else ov.visual_input_manifest_block(
            policy,
            augmentation,
            feature_cache=getattr(args, "vit_feature_cache", None) is not None,
            batch_pages=bool(getattr(args, "vit_batch_pages", False)),
            external_views=external_views,
        )
    )
    manifest_block = s4_manifest.get("visual_input")
    if (
        getattr(args, "external_views", None) is not None
        and external_views is None
        and isinstance(manifest_block, dict)
    ):
        manifest_block = {k: v for k, v in manifest_block.items() if k != "external_views"}
    if cli_block != manifest_block:
        raise S4TrainingRefused(
            f"the visual input flags give {cli_block!r} but the frozen s4_run_manifest "
            f"visual_input is {manifest_block!r} -- a run (and its --resume) keeps the "
            "visual path, DeepStack tap and augmentation seed it was frozen with"
        )
    return policy, augmentation


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())

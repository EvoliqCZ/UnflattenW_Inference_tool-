"""T3 — the S4 typed-property loss channel (design §4): sample -> collate
-> anchor gather -> masked per-head loss -> λ total -> report.

New module, per DEC-027 D1: `s1_train.py` is not edited. `S4TrainingDataset`
composes `ExactTextTrainingDataset` (does not subclass it) and reads
`dataset.reader[index]` -- the real attribute, confirmed by reading
`ExactTextTrainingDataset.__init__` (`s1_train.py`), which stores
`self.reader = ExactTextCacheReader(...)`. `randomize`/`shuffle` are
refused on the base dataset: S4 run 1 has no control arm.

**Enum domain source, a deliberate deviation from the task brief.**
`candidate_model._build_typed_head_group` sizes every `EnumPropertyHead`
from `typed_property_pointer_map.enum_domain` (already landed, T2). That
module and the older diagnostic `diagnostics.typed_property_head
.enum_domain` are NOT interchangeable: measured over every registered
enum pointer, 52 of 57 return the values in a DIFFERENT order (same set,
same length, different position -- e.g. `paragraph:/layout/alignment` is
`('left', 'center', ...)` in one and `('center', 'left', ...)` in the
other). Indexing against the diagnostic module's order while the head was
built and classifies against the production module's order would not
crash -- both return the same length -- it would silently swap which
class index means which enum value, for the great majority of pointers.
This module therefore indexes against `typed_property_pointer_map
.enum_domain`, the same source the head was sized from, not the
diagnostic module a since-superseded design note named. Flagged for the
task report; not something to route around silently
(`measurement-that-can-contradict`).

**Anchor rule (RFC005-ERRATA-004 E7).** The teacher-forced hidden state at
composite step `anchor_token_index` is the one that PREDICTS the node's
opening token (`packed_input_ids = [bos, *packed_ids[:-1]]`,
`training_materializer.py:450`) -- no `- 1`. `gather_anchor_hidden` is the
one place this arithmetic is written.

**Masking (§4.3).** A `TypedPropertyTarget` enters a per-key batch with
`supervise = supervision_mask`, `target` filled from its stored value, and
stays in the tensor even when `supervise=False` -- `masked_mean`
(`typed_property_heads.py`) needs that row present to prove a masked
record contributes exactly zero gradient. A record is dropped entirely,
never merely masked, when:

* `(node_type, pointer)` is `DEFERRED` (`typed_excluded.deferred`);
* `value is None` -- `known_mask=True` is the documented "explicit null"
  case (RFC003 §29.1, `typed_property_targets.py:151-153`) and is counted
  as `known_null`; `known_mask=False` forces `value=None` too
  (`typed_property_targets.effective_typed_masks`, :163-166) and is
  counted as `unsupervised` -- neither has a value any family's `loss()`
  can encode, so both are excluded from the per-key tensors, but the
  bucket split matches `s4_typed_census`'s (`known ∧ value is None` vs.
  `¬known`), not a broader shared use of `known_null`.

`typed_union_arms` carry no `head_id` (§4.5) and are excluded and counted
wholesale, never entering a per-key batch. They do, however, GATE the
leaves under their container when `collate_typed(mask_inactive_union_arms=
True)` (RFC005-ERRATA-008 E1, PROPOSED, opt-in -- see `collate_typed`'s own
docstring; default `False` is the pre-ERRATA-008 behaviour): a leaf
whose pointer is not among the REQUIRED keys of the node's recorded arm
(`ast_required.union_arm_leaves`) is dropped and counted as
`inactive_union_arm` -- `table:/layout/width` `auto` tolerates
`value_mpt: 0` without carrying a width, and MEASURED 2026-09-26 that
tolerated zero was 50.8 % of the `value_mpt` head's supervised targets
on val_iid (5 201 gold tables). A leaf under a union container whose
node recorded NO arm is dropped the same way: a union never gets a
default arm. `typed_cardinalities` for
`table:/columns` and `section:/columns/widths_mpt` (T3's scope) are the
same: wholesale-excluded, counted in `excluded["cardinality"]`.

**`paragraph:/layout/tabs` cardinality (T2, design §4.2/§4.4) is
different**: it has a real head (`typed_head_config.CARDINALITY_HEADS`,
`candidate_model._build_typed_head_group`'s dense-group `layout_tabs`
child) and therefore a real per-key batch, `("expert_adapter.dense",
"layout_tabs")`. Two rules build that batch's rows, both from RAW
`sample.typed_targets`/`typed_cardinalities` -- *before* the DEFERRED/
known_null filters above drop anything from a leaf's own per-key tensor:

* a `TypedRepeatedCardinality` record already present for `/layout/tabs`
  on a node is used AS-IS, whatever its `count`/`supervision_mask` --
  including a withdrawn one (`supervision_mask=False`, its own true
  count, never rewritten to `0`);
* `zero_count.v0`: a node with NO such record but with `>= 1` raw
  `supervision_mask=True` target ANYWHERE on it gets a synthesized
  `count=0, supervise=True` row (design §4.4's source-level separability
  argument: `_cardinalities`, `typed_property_targets.py:498-542`, emits
  a record for every container the node emitted at all, even withdrawn,
  so the ONLY way to reach "supervised node, no `/layout/tabs` record" is
  `tabs == []`).

A count `> K_TABS` is EXCLUDED and counted (`cardinality_over_cap`),
never clamped -- clamping would teach the head that a truncated list is
correct (the same argument `TypedRepeatedCardinality`'s own docstring
makes against capping at write time).

Ships no optimizer and authorizes no gradient update; `s4_total_loss`
computes a scalar and a report, nothing more.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from unflatten.dataset.targets.properties import normalize_pointer
from unflatten.model import events_vocab, s1_train
from unflatten.model import typed_head_config as thc
from unflatten.model.ast_required import union_arm_leaves
from unflatten.model.candidate_model import _DENSE_HEAD_ID, _pointer_attribute, typed_head_children
from unflatten.model.composite_vocab import PackedRanges
from unflatten.model.exact_text_cache import ExactTextCacheRow
from unflatten.model.node_visual_aligner import STATUS_VOCABULARY
from unflatten.model.property_head_registry import head_for_pointer
from unflatten.model.region_codec import PPM_MAX
from unflatten.model.region_targets import RegionTarget
from unflatten.model.training_materializer import LOSS_FAMILY_IDS
from unflatten.model.typed_head_config import (
    BORDER_OBJECT_POINTERS,
    CARDINALITY_HEADS,
    K_TABS,
    NULLABLE_COLOR_POINTERS,
    family_override,
    palette_domain,
    star_pointer_cap_attributes,
)
from unflatten.model.typed_property_pointer_map import (
    DEFERRED,
    FAMILY,
    enum_domain,
    union_containers,
)
from unflatten.model.typed_property_targets import (
    # `_split_container_index` is imported private on purpose (same
    # rationale as `exact_text_cache.py`'s own import of it): the
    # container/index split is ONE mechanism, owned by the module that
    # writes these records.
    TypedPropertyTarget,
    TypedRepeatedCardinality,
    TypedUnionArm,
    _split_container_index,
)

#: `container_pointer -> cap` for every `CARDINALITY_HEADS` entry
#: (currently just `/layout/tabs`), resolved once at import time -- these
#: are the ONLY container pointers whose `TypedRepeatedCardinality` records
#: enter a per-key batch instead of the wholesale `cardinality` bucket.
_CARDINALITY_CAPS: dict[str, int] = {"/layout/tabs": K_TABS}
assert set(_CARDINALITY_CAPS) == {container for _nt, container, _cap in CARDINALITY_HEADS}

#: The six exclusion buckets this module reports (§4.5/§4.6). `unsupervised`
#: counts records that STAY in the per-key tensors with `supervise=False`;
#: the others are records that never enter a tensor at all.
#: `cardinality_over_cap` (T2, §4.2 R3): a `CARDINALITY_HEADS` container's
#: count exceeded its cap -- excluded and counted, never clamped.
#: `color_out_of_palette` (T4, §9 R4): a colour value below the pinned
#: palette threshold -- excluded and counted, never mapped to the nearest
#: kept hex value.
#: `inactive_union_arm` (RFC005-ERRATA-008 E1, PROPOSED): a leaf under a
#: union container that the node's recorded arm does not REQUIRE (or
#: under a container with no recorded arm) -- excluded and counted, never
#: entered as the numeric zero the schema merely tolerates.
_EXCLUDED_BUCKETS: tuple[str, ...] = (
    "cardinality",
    "cardinality_over_cap",
    "slot_over_cap",
    "union_arm",
    "inactive_union_arm",
    "deferred",
    "known_null",
    "unsupervised",
    "color_out_of_palette",
)

#: `pointer_attribute -> cap_attribute` for the 5 `*`-pointer star groups
#: (T3, design §4.3/§10.1): what `collate_typed` uses to recognise a slot
#: leaf and route it into a per-NODE `[N, K]` batch instead of the flat
#: per-record `raw` batch every other pointer uses.
_STAR_POINTER_CAP_ATTRIBUTES: dict[str, str] = star_pointer_cap_attributes()


def _star_pointer_cap(pointer_attr: str) -> int | None:
    """The current cap for a star pointer attribute, or `None` if
    `pointer_attr` is not one -- read dynamically (never cached as an
    `int`) so a test's `monkeypatch.setattr(typed_head_config, "K_TABS",
    ...)` is honoured."""
    cap_attribute = _STAR_POINTER_CAP_ATTRIBUTES.get(pointer_attr)
    if cap_attribute is None:
        return None
    return getattr(thc, cap_attribute)


#: Opt-in exclusion bucket for `collate_typed(mask_color_unseen=True)`: a
#: palette (colour) record of a page whose view did not show colour. The
#: key exists in `excluded` only when the mask is on.
COLOR_UNSEEN_BUCKET = "color_unseen_by_view"


class S4TypedLossRefused(ValueError):
    """A typed record cannot be collated truthfully: its `head_id` is
    outside the stage policy's `L_typed` allowlist, or its enum value is
    outside the pointer's schema domain. Always a refusal, never a
    best-effort repair."""


# ---------------------------------------------------------------------------
# Sample / dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class S4TrainingSample:
    """One page: the S1-format base sample plus its typed channel and its
    region channel (the raw `RegionTarget` records, design §4.1)."""

    base: s1_train.ExactTextTrainingSample
    typed_targets: tuple[TypedPropertyTarget, ...]
    typed_cardinalities: tuple[TypedRepeatedCardinality, ...]
    typed_union_arms: tuple[TypedUnionArm, ...]
    region_targets: tuple[RegionTarget, ...]
    #: Nodes with at least one region box the view's augmentation collapsed
    #: off the page (RFC004-ERRATA-012 E1). Empty = every pre-v17 sample.
    collapsed_region_node_ids: frozenset[str] = frozenset()
    #: The view's colour was not visible (an external view dropped the
    #: colour): every palette-family record of this page is excluded, never
    #: encoded as a class. Only collatable with `mask_color_unseen=True`.
    color_unseen: bool = False

    def __setstate__(self, state: Any) -> None:
        """Restore from the generated `__getstate__` (a list of values in
        field order). A state written before trailing fields were added (a
        cached `s4_eval_rows.pt`) is shorter: the missing trailing fields
        get their dataclass defaults instead of staying unset slots. A
        state longer than the field list (written by newer code) refuses."""
        fields = dataclasses.fields(self)
        if len(state) > len(fields):
            raise ValueError(
                f"S4TrainingSample state has {len(state)} values but the "
                f"class has {len(fields)} fields: written by newer code"
            )
        for i, f in enumerate(fields):
            if i < len(state):
                value = state[i]
            elif f.default is not dataclasses.MISSING:
                value = f.default
            elif f.default_factory is not dataclasses.MISSING:
                value = f.default_factory()
            else:
                raise ValueError(
                    f"S4TrainingSample state lacks required field {f.name!r}"
                )
            object.__setattr__(self, f.name, value)


def _base_sample(row: ExactTextCacheRow) -> s1_train.ExactTextTrainingSample:
    return s1_train.ExactTextTrainingSample(
        document_bundle_id=row.document_bundle_id,
        page_index=row.page_index,
        features=row.features,
        grid_thw=row.grid_thw,
        packed_input_ids=row.packed_input_ids,
        packed_token_ids=row.packed_token_ids,
        effective_loss_mask=row.effective_loss_mask,
        loss_family_ids=row.loss_family_ids,
        page_image=getattr(row, "page_image", None),
    )


class S4TrainingDataset(Dataset):
    """Composes an `ExactTextTrainingDataset` (does not subclass it):
    `__getitem__` reads `base.reader[index]` once and fills both the S1
    base sample and the typed channel from the SAME row object.

    `randomize`/`shuffle` are refused on `base` -- S4 run 1 has no control
    arm (design §4.1); a page whose features were replaced by noise or a
    partner page's features would still carry ITS OWN typed targets,
    silently pairing a real label with the wrong (or no) visual evidence.
    """

    def __init__(self, base: s1_train.ExactTextTrainingDataset) -> None:
        if base.randomize or base.shuffle:
            raise S4TypedLossRefused(
                "S4TrainingDataset refuses a base ExactTextTrainingDataset "
                f"with randomize={base.randomize!r} shuffle={base.shuffle!r} -- "
                "S4 run 1 has no control arm and a perturbed-features row "
                "would still carry its own page's typed targets"
            )
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> S4TrainingSample:
        row = self.base.reader[index]
        return S4TrainingSample(
            base=_base_sample(row),
            typed_targets=row.typed_targets,
            typed_cardinalities=row.typed_cardinalities,
            typed_union_arms=row.typed_union_arms,
            region_targets=row.region_targets,
        )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypedKeyBatch:
    """One `(head_id, pointer_attribute)` group's flat tensors, `[N]` each."""

    head_id: str
    pointer: str
    row_index: torch.Tensor  # int64 [N]
    anchor_step: torch.Tensor  # int64 [N] -- composite step, no `- 1` (E7)
    target: torch.Tensor  # dtype per family: bool / long / long
    supervise: torch.Tensor  # bool [N]


@dataclass(frozen=True, slots=True)
class TypedBatch:
    """`collate_typed`'s output: per-key tensors plus the §4.5 exclusion
    counts (`typed_excluded`, design §4.6)."""

    keys: dict[tuple[str, str], TypedKeyBatch] = field(default_factory=dict)
    excluded: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_EXCLUDED_BUCKETS, 0))


def _encode_target(node_type: str, pointer: str, family: str, value: Any) -> Any:
    """The raw target value for one supervised/unsupervised (but non-null)
    record, per §4.2: enum -> schema-order domain index (refused if the
    stored value is not in the domain); boolean -> bool; integer -> the
    raw int, unchanged."""
    if family == "enum":
        domain = enum_domain(node_type, pointer)
        if value not in domain:
            raise S4TypedLossRefused(
                f"{(node_type, pointer)!r} stores enum value {value!r}, which is "
                f"not in its schema domain {domain!r}"
            )
        return domain.index(value)
    if family == "boolean":
        return bool(value)
    return int(value)


def _target_dtype(family: str) -> torch.dtype:
    if family == "boolean":
        return torch.bool
    return torch.long


def _zero_count_records(sample: S4TrainingSample) -> list[TypedRepeatedCardinality]:
    """`zero_count.v0` (design §4.4): one synthetic `count=0, supervise=True`
    record per `CARDINALITY_HEADS` container/node_type pair, for every node
    that has `>= 1` supervised RAW typed target (`sample.typed_targets`,
    counted BEFORE this module's own DEFERRED/known_null filters -- a node
    whose only supervised target is a deferred string pointer must still
    get its zero) but carries NO `TypedRepeatedCardinality` record for that
    container at all -- a record already present (even a withdrawn one,
    `supervision_mask=False`) always rules instead, whatever it says."""
    existing_node_ids: dict[str, set[str]] = {}
    for cardinality in sample.typed_cardinalities:
        existing_node_ids.setdefault(cardinality.container_pointer, set()).add(
            cardinality.node_id
        )

    supervised_node_ids: dict[str, set[str]] = {}
    anchor_by_node: dict[str, int] = {}
    for target in sample.typed_targets:
        anchor_by_node.setdefault(target.node_id, target.anchor_token_index)
        if target.supervision_mask:
            supervised_node_ids.setdefault(target.node_type, set()).add(target.node_id)

    zero_records: list[TypedRepeatedCardinality] = []
    for node_type, container_pointer, _cap_attribute in CARDINALITY_HEADS:
        already_recorded = existing_node_ids.get(container_pointer, set())
        missing = supervised_node_ids.get(node_type, set()) - already_recorded
        # This order reaches batched loss reductions and gradient accumulation.
        # A set's process-specific hash order would make exact resume diverge.
        for node_id in sorted(missing):
            zero_records.append(
                TypedRepeatedCardinality(
                    node_id=node_id,
                    container_pointer=container_pointer,
                    count=0,
                    known_mask=True,
                    observable_mask=True,
                    supervision_mask=True,
                    anchor_token_index=anchor_by_node[node_id],
                )
            )
    return zero_records


def _borders_null_records(sample: S4TrainingSample) -> list[dict[str, Any]]:
    """`borders_null.v0` (design §5.1/§10.1 T4): one boolean record per
    node that emits EITHER an explicit null object target (a `TypedProperty
    Target` for the container pointer ITSELF, `value=None` -- the object
    "reaches this module only when its value is null", `typed_property_
    targets.py`'s own module docstring) OR at least one leaf target under
    the container (`<container>/...`) -- built from RAW `sample.
    typed_targets`, the same "read raw, derive once" shape T2's
    `_zero_count_records` already uses. A node carrying neither gets no
    record at all (design §2.1: the partition is exact on the measured
    sample). An explicit null record, when present, always rules (matches
    `zero_count.v0`'s "record already present always rules" precedent) --
    a node cannot legitimately carry both, but if it did, the leaves would
    be ignored rather than averaged against the direct answer.
    """
    records: list[dict[str, Any]] = []
    for node_type, container in BORDER_OBJECT_POINTERS:
        head_id = head_for_pointer(node_type, container)
        pointer_attr = f"{_pointer_attribute(container)}_is_null"
        explicit_null: dict[str, TypedPropertyTarget] = {}
        leaves: dict[str, list[TypedPropertyTarget]] = {}
        for target in sample.typed_targets:
            if target.node_type != node_type:
                continue
            pointer = normalize_pointer(target.node_property_pointer)
            if pointer == container:
                explicit_null.setdefault(target.node_id, target)
            elif pointer.startswith(container + "/"):
                leaves.setdefault(target.node_id, []).append(target)
        for node_id, target in explicit_null.items():
            records.append(
                {
                    "head_id": head_id,
                    "pointer": container,
                    "pointer_attr": pointer_attr,
                    "value": True,
                    "known": target.known_mask,
                    "supervise": target.supervision_mask,
                    "anchor": target.anchor_token_index,
                }
            )
        for node_id, node_leaves in leaves.items():
            if node_id in explicit_null:
                continue
            records.append(
                {
                    "head_id": head_id,
                    "pointer": container,
                    "pointer_attr": pointer_attr,
                    "value": False,
                    "known": all(leaf.known_mask for leaf in node_leaves),
                    "supervise": all(leaf.supervision_mask for leaf in node_leaves),
                    "anchor": node_leaves[0].anchor_token_index,
                }
            )
    return records


@cache
def _union_arm_table() -> dict[str, tuple[tuple[str, dict[str, frozenset[str]]], ...]]:
    """`node_type -> ((container, {arm: REQUIRED leaf pointers}), ...)` for
    every typed-property union (`typed_property_pointer_map.
    union_containers`), resolved once from the schema."""
    table: dict[str, list[tuple[str, dict[str, frozenset[str]]]]] = {}
    for (node_type, container), _arms in union_containers().items():
        table.setdefault(node_type, []).append(
            (container, union_arm_leaves(node_type, container))
        )
    return {node_type: tuple(entries) for node_type, entries in table.items()}


def _inactive_union_leaf(
    node_type: str, pointer: str, recorded_arms: Mapping[str, str]
) -> bool:
    """RFC005-ERRATA-008 E1: `pointer` (normalized) sits under one of
    `node_type`'s union containers but is not a REQUIRED leaf of the arm
    the node recorded for that container (`recorded_arms`: `container ->
    arm`, from the sample's `typed_union_arms`). A container the node
    recorded no arm for keeps only the leaves EVERY arm requires (the
    `mode` discriminant); every arm-specific leaf under it is inactive --
    a union never gets a default arm."""
    for container, arms in _union_arm_table().get(node_type, ()):
        if not pointer.startswith(container + "/"):
            continue
        recorded = recorded_arms.get(container)
        if recorded is None:
            active = frozenset.intersection(*arms.values())
        else:
            active = arms.get(recorded, frozenset())
        return pointer not in active
    return False


def collate_typed(
    samples: Sequence[S4TrainingSample],
    *,
    allowed_head_ids: Collection[str],
    mask_inactive_union_arms: bool = False,
    mask_color_unseen: bool = False,
) -> TypedBatch:
    """Group every sample's typed records by `(head_id, pointer_attribute)`
    (`candidate_model._pointer_attribute`), per §4.2/§4.3.

    `mask_inactive_union_arms` (RFC005-ERRATA-008 E1, PROPOSED, v16 only)
    gates the E1 exclusion below: default `False` reproduces this
    function's pre-ERRATA-008 (commit afbf291) behaviour byte-for-byte --
    every leaf enters its per-key batch regardless of the node's recorded
    union arm, and `excluded["inactive_union_arm"]` stays 0. Only
    `mask_inactive_union_arms=True` drops a leaf whose pointer is not
    REQUIRED by the node's recorded arm. The running v15b promotion must
    never pick up E1 merely from a `git pull`, so this stays an explicit
    opt-in threaded from the caller (`s4_train`'s
    `--mask-inactive-union-arms`), never a default.

    `mask_color_unseen` (default `False`, `excluded` byte-identical): adds
    the `color_unseen_by_view` bucket and drops EVERY palette-family record
    of a sample with `color_unseen=True` (border-presence stays supervised).
    A `color_unseen` sample without the flag is refused."""
    raw: dict[tuple[str, str], list[tuple[int, int, Any, bool, str, str]]] = {}
    #: `(head_id, pointer_attribute) -> {node_id: entry}`, T3's per-NODE
    #: slot aggregation (design §4.3/§10.1): a star pointer's leaves fill
    #: ONE row per node, indexed by their own array position, never by
    #: list-append order.
    slot_nodes: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    excluded = dict.fromkeys(_EXCLUDED_BUCKETS, 0)
    if mask_color_unseen:
        excluded[COLOR_UNSEEN_BUCKET] = 0

    for row_index, sample in enumerate(samples):
        if sample.color_unseen and not mask_color_unseen:
            raise S4TypedLossRefused(
                f"sample {sample.base.document_bundle_id!r} page "
                f"{sample.base.page_index} has color_unseen=True but "
                "collate_typed(mask_color_unseen=False): its colour labels "
                "would be supervised against a view that did not show colour"
            )
        #: `node_id -> {container_pointer: arm}` -- the arm every union
        #: leaf below is gated on (E1). Built from the sample's own
        #: `typed_union_arms`, which every cache row carries.
        arms_by_node: dict[str, dict[str, str]] = {}
        for arm in sample.typed_union_arms:
            arms_by_node.setdefault(arm.node_id, {})[arm.container_pointer] = arm.arm
        for target in sample.typed_targets:
            pointer = normalize_pointer(target.node_property_pointer)
            if mask_inactive_union_arms and _inactive_union_leaf(
                target.node_type, pointer, arms_by_node.get(target.node_id, {})
            ):
                excluded["inactive_union_arm"] += 1
                continue
            override = family_override(target.node_type, pointer)
            if override == "borders_null":
                # Handled wholesale below (`_borders_null_records`), which
                # reads RAW `sample.typed_targets` itself -- an explicit
                # null object target processed here too would double-count
                # it.
                continue
            if override == "palette":
                if sample.color_unseen:
                    excluded[COLOR_UNSEEN_BUCKET] += 1
                    continue
                nullable = (target.node_type, pointer) in NULLABLE_COLOR_POINTERS
                domain = palette_domain(nullable=nullable)
                if target.value is None:
                    if not target.known_mask:
                        excluded["unsupervised"] += 1
                        continue
                    if not nullable:
                        # R9's mirror: this pointer's domain has NO `null`
                        # class at all -- a schema-anomalous None falls
                        # back to the pre-existing known_null bucket
                        # rather than being silently encoded.
                        excluded["known_null"] += 1
                        continue
                    value = domain.index("null")
                elif target.value not in domain:
                    excluded["color_out_of_palette"] += 1
                    continue
                else:
                    value = domain.index(target.value)
                if target.head_id not in allowed_head_ids:
                    raise S4TypedLossRefused(
                        f"typed target on {target.node_id!r} "
                        f"({target.node_property_pointer!r}) stores head_id "
                        f"{target.head_id!r}, which is outside the stage "
                        f"policy's L_typed head_ids {sorted(allowed_head_ids)!r}"
                    )
                if not target.supervision_mask:
                    excluded["unsupervised"] += 1
                key = (target.head_id, _pointer_attribute(pointer))
                raw.setdefault(key, []).append(
                    (
                        row_index,
                        target.anchor_token_index,
                        value,
                        target.supervision_mask,
                        pointer,
                        "palette",
                    )
                )
                continue
            if (target.node_type, pointer) in DEFERRED:
                excluded["deferred"] += 1
                continue
            if target.value is None:
                # Matches `s4_typed_census`'s bucket split (design §7
                # should_fix): `known_mask ∧ value is None` is the explicit-
                # null case (`known_null`); `¬known_mask` (which also forces
                # `value=None`, `typed_property_targets.effective_typed_masks`)
                # has never been observed at all and belongs in
                # `unsupervised`, not `known_null` -- neither carries a value
                # a family's `loss()` can encode, so both are still dropped
                # here rather than entered into a per-key tensor.
                excluded["known_null" if target.known_mask else "unsupervised"] += 1
                continue
            if target.head_id not in allowed_head_ids:
                raise S4TypedLossRefused(
                    f"typed target on {target.node_id!r} ({target.node_property_pointer!r}) "
                    f"stores head_id {target.head_id!r}, which is outside the stage "
                    f"policy's L_typed head_ids {sorted(allowed_head_ids)!r}"
                )
            family = FAMILY.get((target.node_type, pointer))
            if family is None:
                raise S4TypedLossRefused(
                    f"{(target.node_type, pointer)!r} is in neither FAMILY nor "
                    "DEFERRED -- registry drift, not data to route around"
                )
            pointer_attr = _pointer_attribute(pointer)
            cap = _star_pointer_cap(pointer_attr)
            if cap is not None:
                # T3 (design §4.3/§10.1): a `*`-pointer leaf. `index` is
                # the leaf's OWN array position (`_split_container_index`
                # on the RAW, unnormalized pointer) -- never list order --
                # and beyond `cap` it is excluded and counted, never
                # clamped (the same rule R3 already applies to counts).
                split = _split_container_index(target.node_property_pointer)
                if split is None:
                    raise S4TypedLossRefused(
                        f"{target.node_property_pointer!r} matched star pointer "
                        f"attribute {pointer_attr!r} but carries no array index"
                    )
                _container, index = split
                if index >= cap:
                    excluded["slot_over_cap"] += 1
                    continue
                value = _encode_target(target.node_type, pointer, family, target.value)
                if not target.supervision_mask:
                    excluded["unsupervised"] += 1
                key = (target.head_id, pointer_attr)
                node_entries = slot_nodes.setdefault(key, {})
                entry = node_entries.setdefault(
                    target.node_id,
                    {
                        "row_index": row_index,
                        "anchor": target.anchor_token_index,
                        "target": [0] * cap,
                        "supervise": [False] * cap,
                        "family": family,
                        "pointer": pointer,
                    },
                )
                entry["target"][index] = value
                entry["supervise"][index] = target.supervision_mask
                continue
            value = _encode_target(target.node_type, pointer, family, target.value)
            if not target.supervision_mask:
                excluded["unsupervised"] += 1
            key = (target.head_id, pointer_attr)
            raw.setdefault(key, []).append(
                (
                    row_index,
                    target.anchor_token_index,
                    value,
                    target.supervision_mask,
                    pointer,
                    family,
                )
            )
        for cardinality in (*sample.typed_cardinalities, *_zero_count_records(sample)):
            cap = _CARDINALITY_CAPS.get(cardinality.container_pointer)
            if cap is None:
                # T3's scope (`table:/columns`, `section:/columns/widths_
                # mpt`): no head yet, wholesale-excluded as before.
                excluded["cardinality"] += 1
                continue
            if cardinality.count > cap:
                excluded["cardinality_over_cap"] += 1
                continue
            key = (_DENSE_HEAD_ID, _pointer_attribute(cardinality.container_pointer))
            raw.setdefault(key, []).append(
                (
                    row_index,
                    cardinality.anchor_token_index,
                    cardinality.count,
                    cardinality.supervision_mask,
                    cardinality.container_pointer,
                    "cardinality",
                )
            )
        for record in _borders_null_records(sample):
            if not record["known"]:
                excluded["unsupervised"] += 1
                continue
            if not record["supervise"]:
                excluded["unsupervised"] += 1
            key = (record["head_id"], record["pointer_attr"])
            raw.setdefault(key, []).append(
                (
                    row_index,
                    record["anchor"],
                    record["value"],
                    record["supervise"],
                    record["pointer"],
                    "boolean",
                )
            )
        excluded["union_arm"] += len(sample.typed_union_arms)

    keys: dict[tuple[str, str], TypedKeyBatch] = {}
    for (head_id, pointer_attr), records in raw.items():
        family = records[0][5]
        keys[(head_id, pointer_attr)] = TypedKeyBatch(
            head_id=head_id,
            pointer=records[0][4],
            row_index=torch.tensor([r[0] for r in records], dtype=torch.long),
            anchor_step=torch.tensor([r[1] for r in records], dtype=torch.long),
            target=torch.tensor([r[2] for r in records], dtype=_target_dtype(family)),
            supervise=torch.tensor([r[3] for r in records], dtype=torch.bool),
        )
    for (head_id, pointer_attr), node_entries in slot_nodes.items():
        entries = list(node_entries.values())
        family = entries[0]["family"]
        keys[(head_id, pointer_attr)] = TypedKeyBatch(
            head_id=head_id,
            pointer=entries[0]["pointer"],
            row_index=torch.tensor([e["row_index"] for e in entries], dtype=torch.long),
            anchor_step=torch.tensor([e["anchor"] for e in entries], dtype=torch.long),
            target=torch.tensor([e["target"] for e in entries], dtype=_target_dtype(family)),
            supervise=torch.tensor([e["supervise"] for e in entries], dtype=torch.bool),
        )
    return TypedBatch(keys=keys, excluded=excluded)


# ---------------------------------------------------------------------------
# Anchor gather
# ---------------------------------------------------------------------------


def gather_anchor_hidden(
    hidden: torch.Tensor,
    *,
    row_index: torch.Tensor,
    anchor_step: torch.Tensor,
    prefix_len: int,
    visual_steps: int,
) -> torch.Tensor:
    """`hidden[row_index, pos]`, `pos = prefix_len + visual_steps +
    anchor_step` (RFC005-ERRATA-004 E7, no `- 1`). `hidden` is
    `encode_hidden`'s full `(batch, prefix_len + steps, d_model)` output --
    the same tensor `scored_logits` slices -- so structure/text and typed
    share one forward pass."""
    pos = prefix_len + visual_steps + anchor_step
    return hidden[row_index, pos]


def typed_losses(
    model: Any,
    hidden: torch.Tensor,
    *,
    prefix_len: int,
    visual_steps: int,
    typed_batch: TypedBatch,
) -> dict[tuple[str, str], tuple[torch.Tensor, int] | None]:
    """`{(head_id, pointer_attribute): (loss, supervised_count) | None}`.
    `None` for a key with zero supervised examples in this batch -- kept
    out of any mean rather than reported as a `0.0` that would read as a
    measurement (§4.4)."""
    head_id_to_child = {head_id: child for child, head_id in typed_head_children().items()}
    results: dict[tuple[str, str], tuple[torch.Tensor, int] | None] = {}
    for key, batch in typed_batch.keys.items():
        supervised_count = int(batch.supervise.sum())
        if supervised_count == 0:
            results[key] = None
            continue
        head_id, pointer_attr = key
        child_name = head_id_to_child[head_id]
        head = getattr(getattr(model, child_name), pointer_attr)
        # `collate_typed` builds every tensor on CPU regardless of where
        # `model`/`hidden` live -- move the whole key batch onto
        # `hidden`'s device here, the one place every caller (train and
        # eval) routes through `typed_losses`.
        anchor = gather_anchor_hidden(
            hidden,
            row_index=batch.row_index.to(hidden.device),
            anchor_step=batch.anchor_step.to(hidden.device),
            prefix_len=prefix_len,
            visual_steps=visual_steps,
        )
        prediction = head(anchor)
        loss = head.loss(
            prediction, batch.target.to(hidden.device), batch.supervise.to(hidden.device)
        )
        results[key] = (loss, supervised_count)
    return results


# ---------------------------------------------------------------------------
# Node-visual-aligner region channel (design spec-node-visual-aligner
# §4.3/§4.4): load-time derivation, `collate_regions`, the masked
# alignment loss, and the `L_alignment` term in the S4 total.
# ---------------------------------------------------------------------------

#: The §4.3 exclusion buckets -- a record that is never scored, with a
#: named reason, never dropped silently.
REGION_EXCLUDED_BUCKETS = ("multi_region", "unresolved", "withdrawn")

#: DEC-055 addendum B (V6b): what a `structural` table/row/cell record with
#: no box trains as. `"abstain_not_visible"` is the v15b behaviour (and the
#: default); `"exclude"` scores nothing and counts it under
#: `REGION_EXCLUDED_STRUCTURAL_TABLE` -- an empty row or cell IS visible
#: (grid, rules), so "not visible" is a wrong label, and until its box can
#: be derived from table geometry (V6a) no label is better than a wrong one.
STRUCTURAL_TABLE_REGION_POLICIES = ("abstain_not_visible", "exclude")
REGION_EXCLUDED_STRUCTURAL_TABLE = "structural_table"
_STRUCTURAL_TABLE_TYPES = frozenset({"table", "row", "cell"})


#: RFC004-ERRATA-012 E1: what a node trains as when augmentation collapsed
#: one of its boxes. `"abstain_container"` is the v16 behaviour (default);
#: `"exclude"` scores nothing and counts `REGION_EXCLUDED_COLLAPSED`.
COLLAPSED_REGION_RULES = ("abstain_container", "exclude")
REGION_EXCLUDED_COLLAPSED = "collapsed_by_view"


def region_excluded_buckets(
    structural_table_region: str, collapsed_region_rule: str = "abstain_container"
) -> tuple[str, ...]:
    """The exclusion buckets `collate_regions` fills under this policy -- the
    one source for every accumulator that sums `RegionBatch.excluded`."""
    buckets = REGION_EXCLUDED_BUCKETS
    if structural_table_region == "exclude":
        buckets = buckets + (REGION_EXCLUDED_STRUCTURAL_TABLE,)
    if collapsed_region_rule == "exclude":
        buckets = buckets + (REGION_EXCLUDED_COLLAPSED,)
    return buckets


def region_kwargs(
    structural_table_region: str, collapsed_region_rule: str = "abstain_container"
) -> dict[str, Any]:
    """Extra `collate_regions` keywords for a trainer: empty for the default,
    so a v15b-shaped run calls `collate_regions` with its literal old shape."""
    kwargs: dict[str, Any] = {}
    if structural_table_region != "abstain_not_visible":
        kwargs["structural_table_region"] = structural_table_region
    if collapsed_region_rule != "abstain_container":
        kwargs["collapsed_region_rule"] = collapsed_region_rule
    return kwargs

#: `STATUS_VOCABULARY` indices (region / abstain_not_visible /
#: abstain_container) -- indexed, not re-spelled, so the vocabulary stays
#: the single source of truth.
STATUS_REGION = STATUS_VOCABULARY.index("region")
STATUS_ABSTAIN_NOT_VISIBLE = STATUS_VOCABULARY.index("abstain_not_visible")
STATUS_ABSTAIN_CONTAINER = STATUS_VOCABULARY.index("abstain_container")


class _RegionExcluded(Exception):
    """A raw `RegionTarget` that must not be scored. `bucket` selects the
    exclusion counter on the `RegionBatch`."""

    def __init__(self, bucket: str) -> None:
        super().__init__(bucket)
        self.bucket = bucket


def _derive_region_row(
    target: RegionTarget,
    *,
    structural_table_region: str = "abstain_not_visible",
    collapsed: bool = False,
    collapsed_region_rule: str = "abstain_container",
) -> tuple[int, tuple[int, int, int, int], bool]:
    """One raw record -> `(status_id, box, supervise_box)` per the locked
    §4.3 priority (design `spec-node-visual-aligner` §4.3):

    1. `supervision_mask` False -> `withdrawn` (checked FIRST);
    2. `k >= 1` (or `collapsed` under rule `exclude`, proving a box existed
       before the view) with `page_evidence != "observed"` -> builder-invariant
       drift, REFUSED (never reclassified, so never masked by 2b);
    2b. `collapsed` and rule `exclude` -> `collapsed_by_view` (before 3: a node
       with two regions, one collapsed, arrives with k == 1);
    3. `k == 1` + observed -> `region`, box = `regions[0]`, box supervised;
    4. `k >= 2` + observed -> `multi_region` excluded (never merged);
    5. `k == 0`: `structural` -> `abstain_not_visible`, `observed` ->
       `abstain_container` (status supervised, box is the zero fill and is
       never a label), `unresolved` -> excluded (never guessed).
    """
    k = len(target.regions)
    if not target.supervision_mask:
        raise _RegionExcluded("withdrawn")
    if (
        k >= 1 or (collapsed and collapsed_region_rule == "exclude")
    ) and target.page_evidence != "observed":
        raise S4TypedLossRefused(
            f"node {target.node_id!r}: {k} region(s) with "
            f"page_evidence={target.page_evidence!r} violates the builder "
            "invariant (page_evidence == 'observed' iff k >= 1) -- the "
            "stored payload is builder-invariant drift, refused"
        )
    if collapsed and collapsed_region_rule == "exclude":
        raise _RegionExcluded(REGION_EXCLUDED_COLLAPSED)
    if k == 1:
        return STATUS_REGION, target.regions[0], True
    if k >= 2:
        raise _RegionExcluded("multi_region")
    if target.page_evidence == "structural":
        if (
            structural_table_region == "exclude"
            and target.node_type in _STRUCTURAL_TABLE_TYPES
        ):
            raise _RegionExcluded(REGION_EXCLUDED_STRUCTURAL_TABLE)
        return STATUS_ABSTAIN_NOT_VISIBLE, (0, 0, 0, 0), False
    if target.page_evidence == "observed":
        return STATUS_ABSTAIN_CONTAINER, (0, 0, 0, 0), False
    raise _RegionExcluded("unresolved")


@dataclass(frozen=True, slots=True)
class RegionBatch:
    """The collated region channel for one S4 step (design §4.3/§4.4).

    `bbox_target` is a zero fill for abstain rows -- rectangular, never a
    label. `excluded` counts the §4.3 exclusion buckets."""

    row_index: torch.Tensor
    anchor_step: torch.Tensor
    status_target: torch.Tensor
    bbox_target: torch.Tensor
    supervise_status: torch.Tensor
    supervise_box: torch.Tensor
    node_types: tuple[str, ...]
    excluded: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(REGION_EXCLUDED_BUCKETS, 0)
    )
    excluded_by_node_type: dict[str, dict[str, int]] = field(default_factory=dict)


def collate_regions(
    samples: Sequence[S4TrainingSample],
    *,
    structural_table_region: str = "abstain_not_visible",
    collapsed_region_rule: str = "abstain_container",
) -> RegionBatch:
    """Derive the region channel from raw `RegionTarget` records (design
    §4.3). One scored record -> one row; exclusions are counted by bucket,
    builder-invariant drift is refused (never reclassified). Every tensor
    is CPU int64/bool, as in `collate_typed` -- `alignment_losses` moves
    onto `hidden`'s device.

    `structural_table_region` (DEC-055 V6b, see
    `STRUCTURAL_TABLE_REGION_POLICIES`): `"exclude"` adds the
    `structural_table` bucket; the default returns exactly the v15b batch.
    `collapsed_region_rule` (`COLLAPSED_REGION_RULES`): `"exclude"` adds the
    `collapsed_by_view` bucket for `sample.collapsed_region_node_ids`."""
    if structural_table_region not in STRUCTURAL_TABLE_REGION_POLICIES:
        raise S4TypedLossRefused(
            f"structural_table_region must be one of {STRUCTURAL_TABLE_REGION_POLICIES}, "
            f"got {structural_table_region!r}"
        )
    if collapsed_region_rule not in COLLAPSED_REGION_RULES:
        raise S4TypedLossRefused(
            f"collapsed_region_rule must be one of {COLLAPSED_REGION_RULES}, "
            f"got {collapsed_region_rule!r}"
        )
    buckets = region_excluded_buckets(structural_table_region, collapsed_region_rule)
    row_index: list[int] = []
    anchor_step: list[int] = []
    status: list[int] = []
    boxes: list[tuple[int, int, int, int]] = []
    supervise_box: list[bool] = []
    node_types: list[str] = []
    excluded = dict.fromkeys(buckets, 0)
    excluded_by_node_type: dict[str, dict[str, int]] = {}
    for row, sample in enumerate(samples):
        # Read only behind the opt-in rule: the default path never touches it.
        collapsed_ids = (
            sample.collapsed_region_node_ids
            if collapsed_region_rule == "exclude"
            else frozenset()
        )
        for target in sample.region_targets:
            try:
                status_id, box, supervise = _derive_region_row(
                    target,
                    structural_table_region=structural_table_region,
                    collapsed=target.node_id in collapsed_ids,
                    collapsed_region_rule=collapsed_region_rule,
                )
            except _RegionExcluded as exc:
                excluded[exc.bucket] += 1
                type_excluded = excluded_by_node_type.setdefault(
                    target.node_type, dict.fromkeys(buckets, 0)
                )
                type_excluded[exc.bucket] += 1
                continue
            row_index.append(row)
            anchor_step.append(target.anchor_token_index)
            status.append(status_id)
            boxes.append(box)
            supervise_box.append(supervise)
            node_types.append(target.node_type)
    return RegionBatch(
        row_index=torch.tensor(row_index, dtype=torch.long),
        anchor_step=torch.tensor(anchor_step, dtype=torch.long),
        status_target=torch.tensor(status, dtype=torch.long),
        bbox_target=torch.tensor(boxes, dtype=torch.long).reshape(-1, 4),
        supervise_status=torch.ones(len(row_index), dtype=torch.bool),
        supervise_box=torch.tensor(supervise_box, dtype=torch.bool),
        node_types=tuple(node_types),
        excluded=excluded,
        excluded_by_node_type=excluded_by_node_type,
    )


def _giou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Half-open generalized IoU per row, differentiable in `pred` only
    (`target` is a leaf). `pred`/`target` are `(N, 4)` `(x1, y1, x2, y2)`
    in ppm. The enclosing area clamps its factors at `1e-6` so a
    degenerate prediction divides by a finite number, and the result is
    clamped to `[-1, 1]` (the mathematical bound)."""
    px1, py1, px2, py2 = (pred[:, i] for i in range(4))
    tx1, ty1, tx2, ty2 = (target[:, i].to(pred.dtype) for i in range(4))
    inter_w = (
        torch.minimum(px2, tx2) - torch.maximum(px1, tx1)
    ).clamp(min=0.0)
    inter_h = (
        torch.minimum(py2, ty2) - torch.maximum(py1, ty1)
    ).clamp(min=0.0)
    inter = inter_w * inter_h
    pred_area = (px2 - px1).clamp(min=0.0) * (py2 - py1).clamp(min=0.0)
    target_area = (tx2 - tx1) * (ty2 - ty1)
    union = pred_area + target_area - inter
    enclos = (
        (torch.maximum(px2, tx2) - torch.minimum(px1, tx1)).clamp(min=1e-6)
        * (torch.maximum(py2, ty2) - torch.minimum(py1, ty1)).clamp(min=1e-6)
    )
    # GIoU is IoU minus the fraction of the smallest enclosing rectangle
    # which neither box occupies.  Keeping the two terms explicit avoids
    # the tempting, but wrong, ``(intersection - (C - union)) / C`` form.
    iou = inter / union.clamp(min=1e-6)
    giou = iou - (enclos - union).clamp(min=0.0) / enclos
    return giou.clamp(min=-1.0, max=1.0)


def _region_visual_rows(
    hidden: torch.Tensor,
    image_grid_thw: torch.Tensor,
    row_index: torch.Tensor,
    *,
    prefix_len: int,
    visual_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select and de-left-pad visual states for scored region rows.

    ``collate_training_batch`` left-pads each row's placeholder span to the
    batch maximum. The aligner, however, expects a row's valid tokens at the
    start of its input so its grid mask can remove only right padding.
    """
    batch_size = hidden.shape[0]
    if image_grid_thw.shape[0] == 1:
        grids = image_grid_thw.expand(batch_size, 3)
    elif image_grid_thw.shape[0] == batch_size:
        grids = image_grid_thw
    else:
        raise ValueError(
            f"image_grid_thw has {image_grid_thw.shape[0]} rows for hidden batch "
            f"size {batch_size}; expected 1 or {batch_size}"
        )
    selected_grids = grids.index_select(0, row_index)
    steps = (selected_grids[:, 1] // 2) * (selected_grids[:, 2] // 2)
    if bool((steps > visual_steps).any()):
        raise ValueError(
            "image_grid_thw requires more packed visual steps than the hidden "
            f"visual span ({visual_steps})"
        )
    visual = hidden[:, prefix_len : prefix_len + visual_steps, :].index_select(0, row_index)
    positions = torch.arange(visual_steps, device=hidden.device)
    starts = visual_steps - steps.to(hidden.device)
    source_positions = starts[:, None] + positions[None, :]
    source_positions = torch.where(
        positions[None, :] < steps[:, None].to(hidden.device), source_positions, 0
    )
    gather_index = source_positions[..., None].expand(-1, -1, visual.shape[-1])
    compacted = visual.gather(1, gather_index)
    valid = positions[None, :] < steps[:, None].to(hidden.device)
    return compacted * valid[..., None].to(compacted.dtype), selected_grids


def _require_positive_weights(weights: Mapping[str, float]) -> None:
    bad = {key: value for key, value in weights.items() if not value > 0}
    if bad:
        raise S4TypedLossRefused(
            f"node_type_weights must be positive numbers, got {bad!r}"
        )


def _node_type_row_weight(
    region_batch: RegionBatch,
    node_type_weights: Mapping[str, float] | None,
    device: torch.device,
) -> torch.Tensor | None:
    if node_type_weights is None:
        return None
    return torch.tensor(
        [float(node_type_weights.get(t, 1.0)) for t in region_batch.node_types],
        dtype=torch.float32,
        device=device,
    )


def _weighted_mean(
    values: torch.Tensor, weight: torch.Tensor, supervise: torch.Tensor
) -> torch.Tensor:
    """`Σ v·w·sup / Σ w·sup`; an empty or zero weight sum is refused, not divided."""
    mass = weight * supervise
    denominator = mass.sum()
    if not bool(denominator > 0):
        raise S4TypedLossRefused("weighted mean has a non-positive weight sum")
    return (values * mass).sum() / denominator


def alignment_losses(
    model: Any,
    hidden: torch.Tensor,
    region_batch: RegionBatch,
    image_grid_thw: torch.Tensor,
    *,
    prefix_len: int,
    visual_steps: int,
    node_type_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor | None, int]:
    """The masked alignment loss (design §4.4):

    `L_alignment = masked_mean_status(CE) + masked_mean_box(L1 + 1 - GIoU)`

    -- the two terms pool over their OWN masks (a box never votes on the
    status mean and vice versa), then add. ONE aligner call for the whole
    batch: anchors are gathered at `prefix_len + visual_steps +
    anchor_step` (the same span `typed_losses` reads), the visual span is
    `hidden[:, prefix_len : prefix_len + visual_steps, :]`, and the grid
    is indexed per row. The predicted box is the continuous
    `sigmoid(bbox_logits) * PPM_MAX` (cx, cy, w, h) folded to
    `(x1, y1, x2, y2)` with both edges clamped to the ppm frame.

    `node_type_weights` (RFC004-ERRATA-012 E2, default `None` = the body
    above, unchanged): a per-row weight `weights.get(node_type, 1.0)` on
    BOTH means -- `Σ ce·w·sup / Σ w·sup`, the box mean likewise. A weight
    that is not a positive number is refused (a zero would turn a present
    label into "no label" and could divide 0/0).

    An empty channel returns `(None, 0)` WITHOUT touching the head -- the
    S4 total then reports `mean: None` and adds no `L_alignment` term."""
    if node_type_weights is not None:
        _require_positive_weights(node_type_weights)
    n = int(region_batch.status_target.numel())
    if n == 0:
        return None, 0
    device = hidden.device
    row_index = region_batch.row_index.to(device)
    anchor = gather_anchor_hidden(
        hidden,
        row_index=row_index,
        anchor_step=region_batch.anchor_step.to(device),
        prefix_len=prefix_len,
        visual_steps=visual_steps,
    )
    # One visual span per batch row, shared by all of its regions: a copy
    # per region was ~600 x 8 580 steps x d_model = 38.8 GiB (S4 v15b 24.9.).
    rows, visual_row = torch.unique(row_index, sorted=True, return_inverse=True)
    visual, grid = _region_visual_rows(
        hidden,
        image_grid_thw.to(device),
        rows,
        prefix_len=prefix_len,
        visual_steps=visual_steps,
    )
    out = model.node_visual_aligner(anchor, visual, grid, visual_row=visual_row)

    status_ce = F.cross_entropy(
        out.status_logits, region_batch.status_target.to(device), reduction="none"
    )
    supervise_status = region_batch.supervise_status.to(device)
    supervise_box = region_batch.supervise_box.to(device)
    row_weight = _node_type_row_weight(region_batch, node_type_weights, device)
    if row_weight is None:
        status_mean = (status_ce * supervise_status).sum() / supervise_status.sum()
    else:
        status_mean = _weighted_mean(status_ce, row_weight, supervise_status)

    if not bool(supervise_box.any()):
        return status_mean, n
    cx = torch.sigmoid(out.bbox_logits[:, 0].float()) * PPM_MAX
    cy = torch.sigmoid(out.bbox_logits[:, 1].float()) * PPM_MAX
    w = torch.sigmoid(out.bbox_logits[:, 2].float()) * PPM_MAX
    h = torch.sigmoid(out.bbox_logits[:, 3].float()) * PPM_MAX
    x1 = torch.minimum((cx - w / 2).clamp(min=0.0), PPM_MAX - w)
    y1 = torch.minimum((cy - h / 2).clamp(min=0.0), PPM_MAX - h)
    pred_box = torch.stack((x1, y1, x1 + w, y1 + h), dim=1)
    target_box = region_batch.bbox_target.to(device).float()
    l1 = F.l1_loss(pred_box, target_box, reduction="none").sum(dim=1) / PPM_MAX
    box_term = l1 + (1.0 - _giou(pred_box, target_box))
    if row_weight is None:
        box_mean = (box_term * supervise_box).sum() / supervise_box.sum()
    else:
        box_mean = _weighted_mean(box_term, row_weight, supervise_box)
    return status_mean + box_mean, n


# ---------------------------------------------------------------------------
# R3 consistency term (RFC004-ERRATA-004 item R3, draft RFC004-ERRATA-006):
# a document-invariant penalty added inside L_typed, opt-in via λ.
# ---------------------------------------------------------------------------

#: The three section-node pointers this term matches on the SAME
#: `(row_index, anchor_step)` identity every pointer of one node already
#: shares -- no new cache field, no cross-node lookup, and (unlike a
#: table's `/columns/widths_mpt/*`, owned by `table_plan`, not a typed
#: head, per `property_head_registry`) all three already live in
#: `TypedBatch` under the SAME head. Scoped to margins-vs-page-width only;
#: folding `/columns/widths_mpt/*` in too needs the capped SlotHead
#: reconstruction and is an OPEN follow-up (see the erratum), not silently
#: included here.
_CONSISTENCY_PAGE_WIDTH_KEY = (_DENSE_HEAD_ID, "page_width_mpt")
_CONSISTENCY_MARGIN_LEFT_KEY = (_DENSE_HEAD_ID, "margins_left_mpt")
_CONSISTENCY_MARGIN_RIGHT_KEY = (_DENSE_HEAD_ID, "margins_right_mpt")


def _consistency_matched_positions(
    *batches: TypedKeyBatch,
) -> list[tuple[int, ...]]:
    """One `(pos_in_batch_0, pos_in_batch_1, ...)` tuple per `(row_index,
    anchor_step)` that is SUPERVISED in every given batch -- the shared
    node identity every pointer of one node already carries. A row missing
    from, or unsupervised in, any one batch contributes nothing (never
    guessed)."""
    indices = [
        {
            (int(r), int(a)): i
            for i, (r, a, s) in enumerate(
                zip(
                    batch.row_index.tolist(),
                    batch.anchor_step.tolist(),
                    batch.supervise.tolist(),
                )
            )
            if s
        }
        for batch in batches
    ]
    shared = set(indices[0])
    for other in indices[1:]:
        shared &= set(other)
    return [tuple(index[key] for index in indices) for key in sorted(shared)]


def section_margin_consistency_loss(
    model: Any,
    hidden: torch.Tensor,
    *,
    prefix_len: int,
    visual_steps: int,
    typed_batch: TypedBatch,
) -> tuple[torch.Tensor | None, int]:
    """R3: `page_width_mpt` must exceed `margins_left_mpt +
    margins_right_mpt` on the SAME section node -- a document invariant
    already named in CLAUDE.md ("column widths sum sanely"), scoped here
    to the margins-vs-page-width pair (see the module-level comment for
    why `/columns/widths_mpt/*` is not folded in).

    A pure structural regularizer over the heads' own CONTINUOUS
    predictions (`IntegerPropertyHead.predict_continuous`, differentiable,
    never the rounded `predict_value`) -- it needs no separate target
    tensor. Returns `(None, 0)` when the three pointers share no
    supervised anchor this batch, exactly like every other S4 family
    channel with nothing to score."""
    width_batch = typed_batch.keys.get(_CONSISTENCY_PAGE_WIDTH_KEY)
    left_batch = typed_batch.keys.get(_CONSISTENCY_MARGIN_LEFT_KEY)
    right_batch = typed_batch.keys.get(_CONSISTENCY_MARGIN_RIGHT_KEY)
    if width_batch is None or left_batch is None or right_batch is None:
        return None, 0
    matched = _consistency_matched_positions(width_batch, left_batch, right_batch)
    if not matched:
        return None, 0

    head_id_to_child = {head_id: child for child, head_id in typed_head_children().items()}
    child_name = head_id_to_child[_DENSE_HEAD_ID]
    width_head = getattr(getattr(model, child_name), "page_width_mpt")
    left_head = getattr(getattr(model, child_name), "margins_left_mpt")
    right_head = getattr(getattr(model, child_name), "margins_right_mpt")

    def _continuous(
        head: Any, batch: TypedKeyBatch, positions: list[int]
    ) -> torch.Tensor:
        idx = torch.tensor(positions, dtype=torch.long)
        anchor = gather_anchor_hidden(
            hidden,
            row_index=batch.row_index[idx].to(hidden.device),
            anchor_step=batch.anchor_step[idx].to(hidden.device),
            prefix_len=prefix_len,
            visual_steps=visual_steps,
        )
        return head.predict_continuous(head(anchor))

    width_positions = [m[0] for m in matched]
    left_positions = [m[1] for m in matched]
    right_positions = [m[2] for m in matched]
    width_value = _continuous(width_head, width_batch, width_positions)
    left_value = _continuous(left_head, left_batch, left_positions)
    right_value = _continuous(right_head, right_batch, right_positions)

    violation = F.relu(left_value + right_value - width_value)
    return violation.mean(), len(matched)


# ---------------------------------------------------------------------------
# λ total
# ---------------------------------------------------------------------------


_STORY_CLOSE_BY_OPEN = {
    events_vocab.TOKEN_IDS["HEADER_OPEN"]: events_vocab.TOKEN_IDS["HEADER_CLOSE"],
    events_vocab.TOKEN_IDS["FOOTER_OPEN"]: events_vocab.TOKEN_IDS["FOOTER_CLOSE"],
}
_PRE_SECTION_FRAMING = frozenset(
    events_vocab.TOKEN_IDS[name] for name in ("FRAGMENT_BEGIN", "DOCUMENT_OPEN")
)
_SECTION_OPEN_ID = events_vocab.TOKEN_IDS["SECTION_OPEN"]


def _story_decision_positions(row: Sequence[int], structural_count: int) -> list[int]:
    """Decision positions of one packed target row (RFC004-ERRATA-012 E3);
    same semantics as `page_start_policy.story_pattern_from_structural_names`."""
    i = 0
    while i < len(row):  # only framing may precede the first SECTION_OPEN
        token = row[i]
        i += 1
        if token == _SECTION_OPEN_ID:
            break
        if token not in _PRE_SECTION_FRAMING:  # incl. non-structural ids
            return []
    else:
        return []
    positions: list[int] = []
    while i < len(row):
        positions.append(i)
        close = _STORY_CLOSE_BY_OPEN.get(row[i]) if row[i] < structural_count else None
        if close is None:
            break
        i += 1
        while i < len(row) and row[i] != close:  # skip the whole story
            i += 1
        i += 1  # past the close
    return positions


def story_decision_mask(target_ids: torch.Tensor, ranges: PackedRanges) -> torch.Tensor:
    """Bool `[B, T]`: the positions where the grammar chooses between header,
    footer and body -- the first after the first `SECTION_OPEN` and the first
    after each closed `HEADER_*`/`FOOTER_*` story there (RFC004-ERRATA-012 E3).
    A token scan, no grammar replay; the scan ends at the first other token."""
    mask = torch.zeros(target_ids.shape, dtype=torch.bool)
    for b, row in enumerate(target_ids.detach().cpu().tolist()):
        for position in _story_decision_positions(row, ranges.structural_count):
            mask[b, position] = True
    return mask.to(target_ids.device)


def _family_mean(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_family_ids: torch.Tensor,
    family_id: int,
    position_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, int]:
    """Non-detached masked-CE mean over one family's scored tokens, so the
    S4 total can weight it by its own λ before backward -- `s1_train
    .family_loss_report` detaches (it is a report, not a loss term) and
    `s1_train.batch_loss` pools both families into one unweighted mean, so
    neither can be reused verbatim here (design §4.4).

    `position_weights` (`[B, T]`, default `None` = the plain mean): the mean
    becomes `Σ ce·w / Σ w` over the selected positions."""
    per_position = s1_train.masked_cross_entropy(logits, target_ids, loss_mask)
    flat_mask = loss_mask.reshape(-1)
    selected = flat_mask & (loss_family_ids.reshape(-1) == family_id)
    count = int(selected.sum())
    if count == 0:
        return None, 0
    if position_weights is not None:
        weights = position_weights.reshape(-1)[selected].to(per_position.dtype)
        return _weighted_mean(
            per_position.reshape(-1)[selected], weights, torch.ones_like(weights)
        ), count
    return per_position.reshape(-1)[selected].sum() / count, count


def s4_total_loss(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_family_ids: torch.Tensor,
    model: Any,
    hidden: torch.Tensor,
    prefix_len: int,
    visual_steps: int,
    typed_batch: TypedBatch,
    stage_policy: Mapping[str, Any],
    region_batch: RegionBatch,
    image_grid_thw: torch.Tensor,
    consistency_lambda: float = 0.0,
    story_decision_weight: float = 1.0,
    ranges: PackedRanges | None = None,
    alignment_type_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The S4 backward scalar plus its step report (§4.4/§4.6).

    `L = Σ λ_f · mean_f` over `{L_structure, L_text, L_typed,
    L_alignment}`; a family with zero scored tokens (or zero supervised
    typed keys / region rows) contributes nothing and reports `mean:
    None` -- an empty region channel touches neither the aligner head nor
    `lambdas["L_alignment"]`. `L_typed` is the MACRO mean over
    contributing `(head_id, pointer)` keys -- one key, one vote, so 12
    anchor pointers on a floating image cannot swamp one paragraph
    alignment pointer (§4.4's rejected micro-pooled variant).

    `consistency_lambda` (R3, RFC004-ERRATA-004 item R3, draft RFC004-
    ERRATA-006): opt-in, default `0.0`. At `0.0` this function calls
    :func:`section_margin_consistency_loss` NOT AT ALL -- no extra head
    forward, no extra report key -- so `total` and its gradients are
    bit-identical to a caller that never passes the argument. A caller-
    supplied `λ` comes from the `s4_train` CLI flag, never from
    `stage_policy["loss_weights_milli"]`, so the consistency term's weight
    can be tuned without touching the frozen stage policy.

    `story_decision_weight` (RFC004-ERRATA-012 E3): at `1.0` the story mask is
    NOT computed and `L_structure` is the plain mean (bit-identical total and
    gradients); otherwise `L_structure` weights the story-decision positions
    (`ranges` required) and reports `story_decision_positions`.
    `alignment_type_weights` (E2): per-`node_type` weights, see
    :func:`alignment_losses`; `None` is the unweighted body.
    """
    if not story_decision_weight > 0:
        raise S4TypedLossRefused(
            f"story_decision_weight must be a positive number, got {story_decision_weight!r}"
        )
    story_weights = None
    story_positions = None
    if story_decision_weight != 1.0:
        if ranges is None:
            raise S4TypedLossRefused("story_decision_weight != 1.0 requires `ranges`")
        story_mask = story_decision_mask(target_ids, ranges).to(logits.device)
        story_positions = int(story_mask.sum())
        story_weights = 1.0 + (story_decision_weight - 1.0) * story_mask.float()
    lambdas = {
        name: milli / 1000 for name, milli in stage_policy["loss_weights_milli"].items()
    }

    structure_mean, structure_count = _family_mean(
        logits,
        target_ids,
        loss_mask,
        loss_family_ids,
        LOSS_FAMILY_IDS["L_structure"],
        story_weights,
    )
    text_mean, text_count = _family_mean(
        logits, target_ids, loss_mask, loss_family_ids, LOSS_FAMILY_IDS["L_text"]
    )

    per_key = typed_losses(
        model, hidden, prefix_len=prefix_len, visual_steps=visual_steps, typed_batch=typed_batch
    )
    contributing = {key: value for key, value in per_key.items() if value is not None}
    l_typed_mean = (
        torch.stack([loss for loss, _ in contributing.values()]).mean()
        if contributing
        else None
    )

    terms: list[torch.Tensor] = []
    families: dict[str, Any] = {}
    for name, mean, count in (
        ("L_structure", structure_mean, structure_count),
        ("L_text", text_mean, text_count),
    ):
        if mean is None:
            families[name] = {"mean": None, "token_count": 0}
            continue
        terms.append(lambdas[name] * mean)
        families[name] = {"mean": float(mean.detach()), "token_count": count}
    if story_positions is not None:
        families["L_structure"]["story_decision_positions"] = story_positions

    typed_supervised_total = sum(count for _, count in contributing.values())
    if l_typed_mean is None:
        families["L_typed"] = {
            "mean": None,
            "key_count": len(typed_batch.keys),
            "supervised_count": typed_supervised_total,
        }
    else:
        terms.append(lambdas["L_typed"] * l_typed_mean)
        families["L_typed"] = {
            "mean": float(l_typed_mean.detach()),
            "key_count": len(contributing),
            "supervised_count": typed_supervised_total,
        }

    if consistency_lambda:
        consistency_mean, consistency_count = section_margin_consistency_loss(
            model,
            hidden,
            prefix_len=prefix_len,
            visual_steps=visual_steps,
            typed_batch=typed_batch,
        )
        if consistency_mean is None:
            families["L_consistency"] = {"mean": None, "supervised_count": 0}
        else:
            terms.append(consistency_lambda * consistency_mean)
            families["L_consistency"] = {
                "mean": float(consistency_mean.detach()),
                "supervised_count": consistency_count,
            }

    alignment_mean, alignment_count = alignment_losses(
        model,
        hidden,
        region_batch,
        image_grid_thw,
        prefix_len=prefix_len,
        visual_steps=visual_steps,
        node_type_weights=alignment_type_weights,
    )
    if alignment_mean is None:
        # An empty region channel touches neither the head nor its λ.
        families["L_alignment"] = {"mean": None, "supervised_count": 0}
    else:
        terms.append(lambdas["L_alignment"] * alignment_mean)
        families["L_alignment"] = {
            "mean": float(alignment_mean.detach()),
            "supervised_count": alignment_count,
        }

    if not terms:
        raise ValueError(
            "no loss family scored anything in this batch -- nothing to optimize"
        )
    total = torch.stack(terms).sum()
    if not bool(torch.isfinite(total)):
        raise s1_train.NonFiniteStepError(f"S4 total loss is {float(total.detach())}")

    typed_heads: dict[str, Any] = {}
    for (head_id, pointer_attr), value in contributing.items():
        loss, count = value
        entry = typed_heads.setdefault(head_id, {"means": [], "supervised_count": 0, "keys": []})
        entry["means"].append(float(loss.detach()))
        entry["supervised_count"] += count
        entry["keys"].append(pointer_attr)
    typed_heads_report = {
        head_id: {
            "mean": sum(entry["means"]) / len(entry["means"]),
            "supervised_count": entry["supervised_count"],
            "keys": entry["keys"],
        }
        for head_id, entry in typed_heads.items()
    }

    report_lambdas = dict(lambdas)
    if consistency_lambda:
        report_lambdas["L_consistency"] = consistency_lambda
    report = {
        "families": families,
        "typed_heads": typed_heads_report,
        "typed_excluded": dict(typed_batch.excluded),
        "lambda": report_lambdas,
    }
    return total, report

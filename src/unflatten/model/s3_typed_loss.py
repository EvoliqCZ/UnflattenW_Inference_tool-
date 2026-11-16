"""T3 — the S3 typed-property loss channel (design §4): sample -> collate
-> anchor gather -> masked per-head loss -> λ total -> report.

New module, per DEC-027 D1: `s1_train.py` is not edited. `S3TrainingDataset`
composes `ExactTextTrainingDataset` (does not subclass it) and reads
`dataset.reader[index]` -- the real attribute, confirmed by reading
`ExactTextTrainingDataset.__init__` (`s1_train.py`), which stores
`self.reader = ExactTextCacheReader(...)`. `randomize`/`shuffle` are
refused on the base dataset: S3 run 1 has no control arm.

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
  bucket split matches `s3_typed_census`'s (`known ∧ value is None` vs.
  `¬known`), not a broader shared use of `known_null`.

`typed_cardinalities` and `typed_union_arms` carry no `head_id` (§4.5) and
are excluded and counted wholesale, never entering a per-key batch.

Ships no optimizer and authorizes no gradient update; `s3_total_loss`
computes a scalar and a report, nothing more.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset

from unflatten.dataset.targets.properties import normalize_pointer
from unflatten.model import s1_train
from unflatten.model.candidate_model import _pointer_attribute, typed_head_children
from unflatten.model.exact_text_cache import ExactTextCacheRow
from unflatten.model.training_materializer import LOSS_FAMILY_IDS
from unflatten.model.typed_property_pointer_map import DEFERRED, FAMILY, enum_domain
from unflatten.model.typed_property_targets import (
    TypedPropertyTarget,
    TypedRepeatedCardinality,
    TypedUnionArm,
)

#: The four exclusion buckets this module reports (§4.5/§4.6). `unsupervised`
#: counts records that STAY in the per-key tensors with `supervise=False`;
#: the other three are records that never enter a tensor at all.
_EXCLUDED_BUCKETS: tuple[str, ...] = (
    "cardinality",
    "union_arm",
    "deferred",
    "known_null",
    "unsupervised",
)


class S3TypedLossRefused(ValueError):
    """A typed record cannot be collated truthfully: its `head_id` is
    outside the stage policy's `L_typed` allowlist, or its enum value is
    outside the pointer's schema domain. Always a refusal, never a
    best-effort repair."""


# ---------------------------------------------------------------------------
# Sample / dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class S3TrainingSample:
    """One page: the S1-format base sample plus its typed channel."""

    base: s1_train.ExactTextTrainingSample
    typed_targets: tuple[TypedPropertyTarget, ...]
    typed_cardinalities: tuple[TypedRepeatedCardinality, ...]
    typed_union_arms: tuple[TypedUnionArm, ...]


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
    )


class S3TrainingDataset(Dataset):
    """Composes an `ExactTextTrainingDataset` (does not subclass it):
    `__getitem__` reads `base.reader[index]` once and fills both the S1
    base sample and the typed channel from the SAME row object.

    `randomize`/`shuffle` are refused on `base` -- S3 run 1 has no control
    arm (design §4.1); a page whose features were replaced by noise or a
    partner page's features would still carry ITS OWN typed targets,
    silently pairing a real label with the wrong (or no) visual evidence.
    """

    def __init__(self, base: s1_train.ExactTextTrainingDataset) -> None:
        if base.randomize or base.shuffle:
            raise S3TypedLossRefused(
                "S3TrainingDataset refuses a base ExactTextTrainingDataset "
                f"with randomize={base.randomize!r} shuffle={base.shuffle!r} -- "
                "S3 run 1 has no control arm and a perturbed-features row "
                "would still carry its own page's typed targets"
            )
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> S3TrainingSample:
        row = self.base.reader[index]
        return S3TrainingSample(
            base=_base_sample(row),
            typed_targets=row.typed_targets,
            typed_cardinalities=row.typed_cardinalities,
            typed_union_arms=row.typed_union_arms,
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
            raise S3TypedLossRefused(
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


def collate_typed(
    samples: Sequence[S3TrainingSample], *, allowed_head_ids: Collection[str]
) -> TypedBatch:
    """Group every sample's typed records by `(head_id, pointer_attribute)`
    (`candidate_model._pointer_attribute`), per §4.2/§4.3."""
    raw: dict[tuple[str, str], list[tuple[int, int, Any, bool, str, str]]] = {}
    excluded = dict.fromkeys(_EXCLUDED_BUCKETS, 0)

    for row_index, sample in enumerate(samples):
        for target in sample.typed_targets:
            pointer = normalize_pointer(target.node_property_pointer)
            if (target.node_type, pointer) in DEFERRED:
                excluded["deferred"] += 1
                continue
            if target.value is None:
                # Matches `s3_typed_census`'s bucket split (design §7
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
                raise S3TypedLossRefused(
                    f"typed target on {target.node_id!r} ({target.node_property_pointer!r}) "
                    f"stores head_id {target.head_id!r}, which is outside the stage "
                    f"policy's L_typed head_ids {sorted(allowed_head_ids)!r}"
                )
            family = FAMILY.get((target.node_type, pointer))
            if family is None:
                raise S3TypedLossRefused(
                    f"{(target.node_type, pointer)!r} is in neither FAMILY nor "
                    "DEFERRED -- registry drift, not data to route around"
                )
            value = _encode_target(target.node_type, pointer, family, target.value)
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
                    family,
                )
            )
        excluded["cardinality"] += len(sample.typed_cardinalities)
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
# λ total
# ---------------------------------------------------------------------------


def _family_mean(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_family_ids: torch.Tensor,
    family_id: int,
) -> tuple[torch.Tensor | None, int]:
    """Non-detached masked-CE mean over one family's scored tokens, so the
    S3 total can weight it by its own λ before backward -- `s1_train
    .family_loss_report` detaches (it is a report, not a loss term) and
    `s1_train.batch_loss` pools both families into one unweighted mean, so
    neither can be reused verbatim here (design §4.4)."""
    per_position = s1_train.masked_cross_entropy(logits, target_ids, loss_mask)
    flat_mask = loss_mask.reshape(-1)
    selected = flat_mask & (loss_family_ids.reshape(-1) == family_id)
    count = int(selected.sum())
    if count == 0:
        return None, 0
    return per_position.reshape(-1)[selected].sum() / count, count


def s3_total_loss(
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
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The S3 backward scalar plus its step report (§4.4/§4.6).

    `L = Σ λ_f · mean_f` over `{L_structure, L_text, L_typed}`; a family
    with zero scored tokens (or zero supervised typed keys) contributes
    nothing and reports `mean: None`. `L_typed` is the MACRO mean over
    contributing `(head_id, pointer)` keys -- one key, one vote, so 12
    anchor pointers on a floating image cannot swamp one paragraph
    alignment pointer (§4.4's rejected micro-pooled variant).
    """
    lambdas = {
        name: milli / 1000 for name, milli in stage_policy["loss_weights_milli"].items()
    }

    structure_mean, structure_count = _family_mean(
        logits, target_ids, loss_mask, loss_family_ids, LOSS_FAMILY_IDS["L_structure"]
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

    if not terms:
        raise ValueError(
            "no loss family scored anything in this batch -- nothing to optimize"
        )
    total = torch.stack(terms).sum()
    if not bool(torch.isfinite(total)):
        raise s1_train.NonFiniteStepError(f"S3 total loss is {float(total.detach())}")

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

    report = {
        "families": families,
        "typed_heads": typed_heads_report,
        "typed_excluded": dict(typed_batch.excluded),
        "lambda": lambdas,
    }
    return total, report

"""Grad-capable typed-head walker for the S4.5 RL paths (design D7 rollout side, D2).

``typed_decode.decode_typed`` is ``@torch.inference_mode()`` and runs over decode
anchors, so it cannot carry a gradient. ``walk_typed`` is its twin over a
GENERATED token stream: the node lifecycle is replayed over the token ids, each
node's anchor is the TEACHER-FORCED hidden state at
``prefix_len + visual_steps + event_index`` (RFC005-ERRATA-004 E7, no ``- 1``,
``s4_typed_loss.gather_anchor_hidden``), and the emitted pointer set is exactly
``decode_typed``'s (registry order, FAMILY filter, star-slot expansion, palette,
border all-or-nothing, union-arm drop, optional objects, first-line offset,
integer clamp). In ``argmax`` mode the output is byte-equal to ``decode_typed``.

Modes: ``argmax`` (no gradient, no log-probs), ``sample`` (a head is drawn only
if its group is in ``selected_typed_heads`` AND its kind is in the policy's
``typed_head_sampling``; log-prob carries a gradient) and ``replay`` (the same
walk with the class indices fixed by ``forced_indices`` -- the update side).
Integer/regression heads are never drawn and never get a log-prob. Values that
the projection drops (inactive union arm, suppressed border leaves, optional
objects) are drawn but carry no log-prob record (``discarded_samples``): they
do not reach the AST, so they do not belong to the policy gradient.

``top_p_micro`` must be 1_000_000 here: the log-prob is the tempered softmax,
so a truncated sampler would be a different distribution (D2, same rule as
``legal_logprobs``).

Uses private ``typed_decode._inactive_arm_leaves`` (pure, no state) so the
union-arm rule cannot drift from decode.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import torch

from unflatten.model import events_vocab
from unflatten.model import typed_head_config as thc
from unflatten.model import typed_projection as tp
from unflatten.model.candidate_model import (
    _DENSE_HEAD_ID,
    _pointer_attribute,
    typed_head_children,
)
from unflatten.model.composite_vocab import CompositeVocabulary, StructuralRef
from unflatten.model.node_lifecycle import NodeLifecycleTracker, NodeState
from unflatten.model.property_head_registry import head_for_pointer, property_head_registry
from unflatten.model.rl.loss import typed_categorical_logprobs
from unflatten.model.rl.rollout import select_head_value
from unflatten.model.rl.scorers import MICRO
from unflatten.model.s4_typed_loss import gather_anchor_hidden
from unflatten.model.typed_decode import TypedPrediction, _inactive_arm_leaves
from unflatten.model.typed_property_heads import BooleanPropertyHead, EnumPropertyHead
from unflatten.model.typed_property_pointer_map import FAMILY, enum_domain

MODES: Final = ("argmax", "sample", "replay")
_TOKEN_NAME_BY_ID: Final = {local_id: name for name, local_id in events_vocab.TOKEN_IDS.items()}


class TypedSamplingRefused(ValueError):
    """The typed walk's inputs break its contract (fail closed)."""


@dataclass(frozen=True, slots=True)
class HeadLogProb:
    """One sampled typed decision: ``logprob`` is a scalar tensor (grad in
    ``sample``/``replay``), ``sampled_index`` the class index it was taken at."""

    local_node_id: str
    head_id: str
    pointer: str
    kind: str
    sampled_index: int
    logprob: torch.Tensor


@dataclass(frozen=True, slots=True)
class TypedWalk:
    predictions: tuple[TypedPrediction, ...]
    logprobs: tuple[HeadLogProb, ...]
    discarded_samples: int
    #: The teacher-forced node states (anchors) the walk ran over; the region
    #: decode of an RL candidate reads the same anchors.
    node_states: tuple[NodeState, ...] = ()


class _Walker:
    def __init__(
        self,
        heads: Any,
        *,
        mode: str,
        policy: Mapping[str, Any] | None,
        generator: torch.Generator | None,
        selected: Collection[str],
        forced: Mapping[tuple[str, str], int],
    ) -> None:
        self.heads = heads
        self.mode = mode
        self.policy = policy
        self.generator = generator
        self.selected = frozenset(selected)
        self.forced = forced
        self.child_of = {head_id: child for child, head_id in typed_head_children().items()}
        self.star_caps = thc.star_pointer_cap_attributes()
        self.records: dict[str, HeadLogProb] = {}
        self.unforced: set[str] = set()
        self.discarded = 0
        self.anchor = torch.zeros(0)

    # -- one head ---------------------------------------------------------
    def _is_drawn(self, head_id: str, kind: str) -> bool:
        if self.mode == "argmax" or head_id not in self.selected:
            return False
        assert self.policy is not None
        return kind in self.policy["typed_head_sampling"]

    def _index(
        self, kind: str, logits: torch.Tensor, head_id: str, state: NodeState, pointer: str
    ) -> int:
        """Class index; records the log-prob when the head is drawn."""
        if not self._is_drawn(head_id, kind):
            if kind == "boolean":
                return int(bool(logits.reshape(-1)[0] > 0))
            return int(logits.reshape(-1).argmax())
        assert self.policy is not None
        if self.mode == "sample":
            if self.generator is None:
                raise TypedSamplingRefused("sample mode needs a torch.Generator")
            index = select_head_value(kind, logits.detach(), self.policy, self.generator)
        else:
            forced = self.forced.get((state.local_node_id, pointer))
            if forced is None:
                self.unforced.add(pointer)
                return (
                    int(bool(logits.reshape(-1)[0] > 0))
                    if kind == "boolean"
                    else int(logits.reshape(-1).argmax())
                )
            index = forced
        logprob = typed_categorical_logprobs(
            logits.reshape(-1),
            torch.tensor(index, device=logits.device),
            kind=kind,
            temperature_micro=self.policy["temperature_micro"],
        )
        self.unforced.discard(pointer)
        self.records[pointer] = HeadLogProb(
            state.local_node_id, head_id, pointer, kind, index, logprob
        )
        return index

    def _generic(self, head: Any, state: NodeState, head_id: str, pointer: str, key: str) -> Any:
        prediction = head(self.anchor.unsqueeze(0))
        if isinstance(head, EnumPropertyHead):
            index = self._index("enum", prediction[0], head_id, state, key)
            return enum_domain(state.node_type, pointer)[index]
        if isinstance(head, BooleanPropertyHead):
            return bool(self._index("boolean", prediction.reshape(-1), head_id, state, key))
        value = int(head.predict_value(prediction).reshape(-1)[0])
        return tp.clamp_integer(state.node_type, pointer, value)

    def _palette(self, head: Any, state: NodeState, head_id: str, pointer: str, nullable: bool):
        prediction = head(self.anchor.unsqueeze(0))
        domain = thc.palette_domain(nullable=nullable)
        label = domain[self._index("palette", prediction[0], head_id, state, pointer)]
        return None if nullable and label == tp.NULL_COLOUR_LABEL else label

    def _head(self, head_id: str, attribute: str) -> Any:
        return getattr(getattr(self.heads, self.child_of[head_id]), attribute)

    # -- one node ---------------------------------------------------------
    def _plain_values(self, state: NodeState, values: dict[str, Any], heads: dict[str, str]):
        for (node_type, pointer), head_id in sorted(property_head_registry().items()):
            if node_type != state.node_type or (node_type, pointer) not in FAMILY:
                continue
            attribute = _pointer_attribute(pointer)
            if attribute in self.star_caps:
                continue
            head = self._head(head_id, attribute)
            values[pointer] = self._generic(head, state, head_id, pointer, pointer)
            heads[pointer] = head_id
        for node_type, pointer in thc.COLOR_POINTERS:
            if node_type != state.node_type:
                continue
            head_id = head_for_pointer(node_type, pointer)
            head = self._head(head_id, _pointer_attribute(pointer))
            nullable = (node_type, pointer) in thc.NULLABLE_COLOR_POINTERS
            values[pointer] = self._palette(head, state, head_id, pointer, nullable)
            heads[pointer] = head_id

    def _borders(self, state: NodeState, values: dict[str, Any], heads: dict[str, str]) -> set:
        suppressed: set[str] = set()
        for node_type, container in thc.BORDER_OBJECT_POINTERS:
            if node_type != state.node_type:
                continue
            head_id = head_for_pointer(node_type, container)
            head = self._head(head_id, _pointer_attribute(container) + "_is_null")
            logit = head(self.anchor.unsqueeze(0))
            if bool(self._index("boolean", logit.reshape(-1), head_id, state, container)):
                values[container] = None
                heads[container] = head_id
                suppressed.update(thc.border_leaf_pointers(container))
        return suppressed

    def _slots(self, state: NodeState, values: dict[str, Any], tabs_count: int | None):
        out: list[TypedPrediction] = []
        for (node_type, pointer), head_id in sorted(property_head_registry().items()):
            if node_type != state.node_type or (node_type, pointer) not in FAMILY:
                continue
            attribute = _pointer_attribute(pointer)
            cap = self.star_caps.get(attribute)
            if cap is None:
                continue
            if node_type == "section" and values.get("/columns/equal_width") is True:
                continue
            count = tabs_count if node_type == "paragraph" else getattr(thc, cap)
            if count is None:
                raise TypedSamplingRefused("paragraph tab slots without a cardinality head")
            slots = self._head(head_id, attribute).slots
            for i in range(count):
                indexed = pointer.replace("*", str(i), 1)
                value = self._generic(slots[i], state, head_id, pointer, indexed)
                out.append(TypedPrediction(state.local_node_id, node_type, indexed, head_id, value))
        return out

    def _cardinality(self, state: NodeState) -> tuple[list[TypedPrediction], int | None]:
        out: list[TypedPrediction] = []
        tabs: int | None = None
        for node_type, container, _cap in thc.CARDINALITY_HEADS:
            if state.node_type != node_type:
                continue
            head = self._head(_DENSE_HEAD_ID, _pointer_attribute(container))
            logits = head(self.anchor.unsqueeze(0))[0]
            count = self._index("cardinality", logits, _DENSE_HEAD_ID, state, container)
            out.append(
                TypedPrediction(state.local_node_id, node_type, container, _DENSE_HEAD_ID, count)
            )
            if container == "/layout/tabs":
                tabs = count
        return out, tabs

    def node(self, state: NodeState) -> list[TypedPrediction]:
        if state.anchor is None:
            raise TypedSamplingRefused(f"node {state.local_node_id} has no anchor")
        self.anchor = state.anchor
        values: dict[str, Any] = {}
        heads: dict[str, str] = {}
        self._plain_values(state, values, heads)
        suppressed = self._borders(state, values, heads)
        tp.one_first_line_offset(values)
        drop = (
            _inactive_arm_leaves(state.node_type, values)
            | suppressed
            | tp.optional_objects_without_presence_head(state.node_type, values)
        )
        for pointer in drop:
            if self.records.pop(pointer, None) is not None:
                self.discarded += 1
        self.unforced -= drop
        out = [
            TypedPrediction(state.local_node_id, state.node_type, p, heads[p], v)
            for p, v in values.items()
            if p not in drop
        ]
        cardinality, tabs = self._cardinality(state)
        slots = self._slots(state, values, tabs)
        return out + cardinality + slots


def _node_states(
    token_ids: Sequence[int],
    vocabulary: CompositeVocabulary,
    hidden: torch.Tensor,
    *,
    row: int,
    prefix_len: int,
    visual_steps: int,
) -> tuple[NodeState, ...]:
    tracker = NodeLifecycleTracker()
    for index, packed in enumerate(token_ids):
        token = vocabulary.unpack(packed)
        if not isinstance(token, StructuralRef):
            continue
        position = prefix_len + visual_steps + index
        if position >= hidden.shape[1]:
            raise TypedSamplingRefused(
                f"anchor position {position} (event {index}) is outside hidden "
                f"length {hidden.shape[1]}: hidden must cover the teacher-forced stream"
            )
        anchor = gather_anchor_hidden(
            hidden,
            row_index=torch.tensor(row, device=hidden.device),
            anchor_step=torch.tensor(index, device=hidden.device),
            prefix_len=prefix_len,
            visual_steps=visual_steps,
        )
        tracker.apply_structural_token(
            _TOKEN_NAME_BY_ID[token.local_id], event_index=index, anchor=anchor
        )
    return tracker.states


def _check_policy(mode: str, policy: Mapping[str, Any] | None) -> None:
    if mode not in MODES:
        raise TypedSamplingRefused(f"mode {mode!r} not in {MODES}")
    if mode == "argmax":
        return
    if policy is None:
        raise TypedSamplingRefused(f"{mode} mode needs the rollout policy")
    if policy["top_p_micro"] != MICRO:
        raise TypedSamplingRefused(
            f"typed log-probs are the tempered softmax; top_p_micro must be {MICRO}, "
            f"got {policy['top_p_micro']!r}"
        )


def walk_typed(
    model_heads: Any,
    hidden: torch.Tensor,
    token_ids: Sequence[int],
    vocabulary: CompositeVocabulary,
    *,
    prefix_len: int,
    visual_steps: int,
    policy: Mapping[str, Any] | None = None,
    generator: torch.Generator | None = None,
    mode: str = "argmax",
    selected_typed_heads: Collection[str] = (),
    forced_indices: Mapping[tuple[str, str], int] | None = None,
    row: int = 0,
) -> TypedWalk:
    """Typed predictions (+ log-probs of the drawn heads) over ``token_ids``.

    ``hidden`` is ``encode_hidden``'s ``(batch, prefix_len + visual + steps, d)``
    output of the teacher-forced pass over ``[placeholders][BOS, t0 .. t_{n-2}]``.
    ``model_heads`` is any object carrying the typed head groups as children
    (a ``QwenDenseCandidateModel`` or a bare head holder). ``selected_typed_heads``
    are group ids (``typed_heads.row_cell`` ...); only those heads are drawn.
    ``argmax`` runs under ``no_grad`` and equals ``decode_typed``.
    """
    _check_policy(mode, policy)
    forced = forced_indices or {}
    if forced and mode != "replay":
        raise TypedSamplingRefused("forced_indices are only valid in replay mode")
    states = _node_states(
        token_ids, vocabulary, hidden, row=row, prefix_len=prefix_len, visual_steps=visual_steps
    )
    walker = _Walker(
        model_heads,
        mode=mode,
        policy=policy,
        generator=generator,
        selected=selected_typed_heads,
        forced=forced,
    )
    predictions: list[TypedPrediction] = []
    logprobs: list[HeadLogProb] = []
    discarded = 0
    with torch.set_grad_enabled(mode != "argmax" and torch.is_grad_enabled()):
        for state in states:
            if state.anchor is None:
                continue
            walker.records, walker.unforced, walker.discarded = {}, set(), 0
            predictions.extend(walker.node(state))
            if walker.unforced:
                raise TypedSamplingRefused(
                    f"replay: node {state.local_node_id} has a drawn head with no "
                    f"forced index: {sorted(walker.unforced)}"
                )
            logprobs.extend(walker.records.values())
            discarded += walker.discarded
    return TypedWalk(tuple(predictions), tuple(logprobs), discarded, states)

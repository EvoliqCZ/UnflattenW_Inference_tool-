"""Typed decoding (AST Builder design §2, T1): the typed property heads
applied to the anchors `decode_free_running` keeps on each `NodeState`.

The anchor is the hidden state that predicted the node's opening token --
the same position teacher-forced training gathers (`s3_typed_loss
.gather_anchor_hidden`; alignment proven by
`test_decode_anchors_equal_the_training_hidden_state`). Values are decoded
exactly as the teacher-forced typed eval does: enum = argmax into
`enum_domain`, boolean = logit > 0, integer = `predict_value` (DEC-032).

A union container (`/layout/spacing/line`, `/anchor/horizontal`, ...) keeps
only the leaves of the arm its predicted `mode` selects; the others are not
emitted (the schema forbids them). Deferred (string/object) pointers get no
head and no prediction.

**`/layout/tabs`'s element COUNT (T2, design §4.3/§10.1 T2) is decoded
separately from the loop above.** It is not in `property_head_registry`/
`FAMILY` at all -- a count is a tensor identity, not an RFC001 pointer
(§4.5) -- so it is read from `typed_head_config.CARDINALITY_HEADS`
instead, one `TypedPrediction` per cardinality-headed node type, value =
`CardinalityHead.predict_count` (argmax over `0..k_max`). A later task
(T3) is what actually places `k` indexed slot values under this count;
this module only emits the count itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from unflatten.model import ast_required
from unflatten.model import typed_head_config as thc
from unflatten.model import typed_projection as tp
from unflatten.model.candidate_model import (
    _DENSE_HEAD_ID,
    _pointer_attribute,
    typed_head_children,
)
from unflatten.model.node_lifecycle import NodeState
from unflatten.model.property_head_registry import head_for_pointer, property_head_registry
from unflatten.model.typed_head_config import (
    BORDER_OBJECT_POINTERS,
    CARDINALITY_HEADS,
    COLOR_POINTERS,
    NULLABLE_COLOR_POINTERS,
    border_leaf_pointers,
    palette_domain,
    star_pointer_cap_attributes,
)
from unflatten.model.typed_property_heads import BooleanPropertyHead, EnumPropertyHead
from unflatten.model.typed_property_pointer_map import FAMILY, enum_domain, union_containers


@dataclass(frozen=True, slots=True)
class TypedPrediction:
    """One decoded `(node, pointer)` value. `value` is `None` only for a
    `borders_is_null=True` container pointer (design §5.1/§10.1 T4) --
    every other family always emits a concrete value."""

    local_node_id: str
    node_type: str
    pointer: str
    head_id: str
    value: int | str | bool | None


def _value(head: Any, anchor: torch.Tensor, node_type: str, pointer: str) -> int | str | bool:
    prediction = head(anchor.unsqueeze(0))
    if isinstance(head, EnumPropertyHead):
        return enum_domain(node_type, pointer)[int(prediction.argmax(dim=-1)[0])]
    if isinstance(head, BooleanPropertyHead):
        return bool(prediction.reshape(-1)[0] > 0)
    value = int(head.predict_value(prediction).reshape(-1)[0])
    return tp.clamp_integer(node_type, pointer, value)


def _palette_value(head: Any, anchor: torch.Tensor, *, nullable: bool) -> str | None:
    """R9: the domain comes from `typed_head_config.palette_domain`, never
    `enum_domain` (the schema carries a `pattern` for a colour, not an
    `enum`) -- a non-nullable pointer's domain structurally has no `null`
    entry, so `argmax` can never select it."""
    prediction = head(anchor.unsqueeze(0))
    domain = palette_domain(nullable=nullable)
    label = domain[int(prediction.argmax(dim=-1)[0])]
    # The trailing "null" CLASS means "no colour": JSON null, never the
    # string "null", which the schema refuses (25.9.).
    return None if nullable and label == tp.NULL_COLOUR_LABEL else label


def _inactive_arm_leaves(node_type: str, values: dict[str, Any]) -> set[str]:
    inactive: set[str] = set()
    for (union_type, container), _arms in union_containers().items():
        if union_type != node_type:
            continue
        arms = ast_required.union_arm_leaves(node_type, container)
        active = arms.get(values.get(f"{container}/mode"), frozenset())
        for leaves in arms.values():
            inactive |= leaves - active
    return inactive


@torch.inference_mode()
def decode_typed(model: Any, node_states: Iterable[NodeState]) -> tuple[TypedPrediction, ...]:
    """Every family pointer registered for each anchored node's type, in
    node order then registry order.

    A `*`-pointer (`typed_head_config.STAR_POINTERS`, T3, design §4.3/
    §10.1) is skipped by the generic loop above -- its child is a
    `SlotHead`, not a bare family instance -- and decoded separately,
    below, into `k` INDEXED `TypedPrediction`s (`/layout/tabs/0/
    position_mpt`, ...): `k = N` (the `/layout/tabs` element count T2's
    `CardinalityHead` predicts) for the 3 tabs pointers, or the full `K`
    cap for `table:/columns/*/width_mpt` and `section:/columns/widths_mpt/
    *` -- neither has a count head (design §4.1: the count is DERIVED at
    build time from what the model already emitted), so decode emits every
    slot and the builder's derivation post-pass keeps only the real count,
    reporting the rest as `discarded_inputs` (U6, `ast_builder.py`'s
    `report_stray_as_discarded`)."""
    child_of = {head_id: child for child, head_id in typed_head_children().items()}
    registry = sorted(property_head_registry().items())
    star_caps = star_pointer_cap_attributes()
    out: list[TypedPrediction] = []
    for state in node_states:
        if state.anchor is None:
            continue
        values: dict[str, Any] = {}
        heads: dict[str, str] = {}
        for (node_type, pointer), head_id in registry:
            if node_type != state.node_type or (node_type, pointer) not in FAMILY:
                continue
            if _pointer_attribute(pointer) in star_caps:
                continue
            head = getattr(getattr(model, child_of[head_id]), _pointer_attribute(pointer))
            values[pointer] = _value(head, state.anchor, node_type, pointer)
            heads[pointer] = head_id
        # T4 (design §5.1/§10.1): the 21 colour pointers are DEFERRED
        # strings (not in `FAMILY`), so the generic loop above never
        # reaches them -- decoded here through `PaletteHead`'s own domain
        # (`typed_head_config.palette_domain`, R9: `enum_domain` would
        # raise, the schema carries a `pattern` here, not an `enum`).
        for node_type, pointer in COLOR_POINTERS:
            if node_type != state.node_type:
                continue
            head_id = head_for_pointer(node_type, pointer)
            head = getattr(getattr(model, child_of[head_id]), _pointer_attribute(pointer))
            values[pointer] = _palette_value(
                head, state.anchor, nullable=(node_type, pointer) in NULLABLE_COLOR_POINTERS
            )
            heads[pointer] = head_id
        # T4: the 4 border-object pointers are DEFERRED objects. `True` ->
        # write `null` and suppress all 12 leaves of that side (style/
        # width_mpt from the generic loop above, colour from the palette
        # loop above) -- R11: the leaves are all-or-nothing, never a
        # partial object. `False` -> emit nothing extra; the 12 leaves
        # already decoded above stand as-is.
        suppressed_borders: set[str] = set()
        for node_type, container in BORDER_OBJECT_POINTERS:
            if node_type != state.node_type:
                continue
            head_id = head_for_pointer(node_type, container)
            attr = _pointer_attribute(container) + "_is_null"
            head = getattr(getattr(model, child_of[head_id]), attr)
            is_null = bool(head(state.anchor.unsqueeze(0)).reshape(-1)[0] > 0)
            if is_null:
                values[container] = None
                heads[container] = head_id
                suppressed_borders.update(border_leaf_pointers(container))
        tp.one_first_line_offset(values)
        drop = (
            _inactive_arm_leaves(state.node_type, values)
            | suppressed_borders
            | tp.optional_objects_without_presence_head(state.node_type, values)
        )
        out.extend(
            TypedPrediction(state.local_node_id, state.node_type, p, heads[p], v)
            for p, v in values.items()
            if p not in drop
        )
        tabs_count: int | None = None
        for node_type, container_pointer, _cap_attribute in CARDINALITY_HEADS:
            if state.node_type != node_type:
                continue
            head = getattr(
                getattr(model, child_of[_DENSE_HEAD_ID]), _pointer_attribute(container_pointer)
            )
            count = int(head.predict_count(head(state.anchor.unsqueeze(0)))[0])
            out.append(
                TypedPrediction(
                    state.local_node_id, state.node_type, container_pointer, _DENSE_HEAD_ID, count
                )
            )
            if container_pointer == "/layout/tabs":
                tabs_count = count
        for (node_type, pointer), head_id in registry:
            if node_type != state.node_type or (node_type, pointer) not in FAMILY:
                continue
            pointer_attr = _pointer_attribute(pointer)
            cap_attribute = star_caps.get(pointer_attr)
            if cap_attribute is None:
                continue
            if node_type == "section" and values.get("/columns/equal_width") is True:
                # Design §10.1 T3: a section whose predicted `equal_width`
                # is `true` emits NOTHING for `/columns/widths_mpt/*` --
                # matching the derivation (`section.columns.widths_mpt.
                # length.v0`, `equal_width=true` => `[]`) instead of handing
                # the builder `K_SECTION_WIDTHS` slots it would report as
                # `discarded_inputs` for nothing.
                continue
            slot_count = tabs_count if node_type == "paragraph" else getattr(thc, cap_attribute)
            head = getattr(getattr(model, child_of[head_id]), pointer_attr)
            for i in range(slot_count):
                value = _value(head.slots[i], state.anchor, node_type, pointer)
                indexed_pointer = pointer.replace("*", str(i), 1)
                out.append(
                    TypedPrediction(
                        state.local_node_id, state.node_type, indexed_pointer, head_id, value
                    )
                )
    return tuple(out)

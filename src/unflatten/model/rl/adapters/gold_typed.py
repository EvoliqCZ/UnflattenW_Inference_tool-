"""Typed authority of the gold adapter: placement and row-mask reading (torch-free).

Typed values come from the cache row's ``typed_targets`` / ``typed_cardinalities``
under the RFC003 known / observable / supervision masks. The chain AST only provides
the id space: decoder body states are matched to its body preorder, and a mismatch is
a named refusal, never a guess.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from unflatten.model.rl.adapters.gold_errors import GoldTypedMappingFailed

#: Node types whose subtree is not part of the body preorder.
STORY_TYPES: Final = frozenset({"header", "footer", "footnote", "endnote"})
TABS: Final = "/layout/tabs"


@dataclass(frozen=True)
class Placement:
    """Where decoder state positions land: body AST ids, or outside the body."""

    body: Mapping[int, tuple[str, str]]  # position -> (ast id, node type)
    outside: Mapping[int, str]  # position -> node type (stories, frame subtrees)
    state_by_position: Mapping[int, Any]


def position(state: Any) -> int | None:
    return state.open_event_index if state.slot_kind == "opened_node" else state.leaf_event_index


def placement(page: Any, ast: Mapping[str, Any]) -> Placement:
    from unflatten.model.rl.alignment import node_ids_from_ast
    from unflatten.model.rl.reward_inputs import _CHILD_KEYS, preorder_from_ast

    by_local = {s.local_node_id: s for s in page.node_states}

    def chain(state: Any) -> list[Any]:
        out: list[Any] = []
        parent = state.parent_local_node_id
        while parent is not None:
            out.append(by_local[parent])
            parent = by_local[parent].parent_local_node_id
        return out

    body: list[tuple[Any, int]] = []
    outside: dict[int, str] = {}
    positions: dict[int, Any] = {}
    for state in page.node_states:
        pos = position(state)
        ancestors = chain(state)
        if pos is not None:
            positions[pos] = state
        hidden = (
            any(a.node_type in STORY_TYPES or a.node_type not in _CHILD_KEYS for a in ancestors)
            or state.node_type in STORY_TYPES
        )
        if hidden:
            if pos is not None:
                outside[pos] = state.node_type
        else:
            body.append((state, len(ancestors)))
    decoder = [(depth, s.node_type) for s, depth in body]
    ast_preorder = list(preorder_from_ast(ast))
    if decoder != ast_preorder:
        raise GoldTypedMappingFailed(
            f"decoder body preorder ({len(decoder)} nodes) is not the gold AST body preorder "
            f"({len(ast_preorder)} nodes): typed targets cannot be placed on gold node ids"
        )
    ids = node_ids_from_ast(ast)
    mapped = {
        position(s): (nid, s.node_type)
        for (s, _d), nid in zip(body, ids, strict=True)
        if position(s) is not None
    }
    return Placement(mapped, outside, positions)  # type: ignore[arg-type]


def supervised(target: Any) -> bool:
    return bool(target.known_mask and target.observable_mask and target.supervision_mask)


def typed_authority(
    row: Any, placement: Placement, counters: Counter[str]
) -> dict[str, dict[str, Any]]:
    """``{gold node id: {pointer: value}}`` from the row's supervised targets."""
    out: dict[str, dict[str, Any]] = {}
    items: list[tuple[int, str, str, Any]] = [
        (t.anchor_token_index, t.node_type, t.node_property_pointer, t.value)
        for t in row.typed_targets
        if supervised(t)
    ] + [
        (c.anchor_token_index, "", c.container_pointer, c.count)
        for c in row.typed_cardinalities
        if supervised(c)
    ]
    for anchor, node_type, pointer, value in items:
        if anchor in placement.outside:
            counters["typed_outside_body"] += 1
            continue
        placed = placement.body.get(anchor)
        if placed is None:
            raise GoldTypedMappingFailed(
                f"typed target {pointer!r} anchors at token {anchor}, which opens no node"
            )
        nid, state_type = placed
        if node_type and node_type != state_type:
            raise GoldTypedMappingFailed(
                f"typed target {pointer!r} is a {node_type!r} target but token {anchor} opens "
                f"a {state_type!r}"
            )
        out.setdefault(nid, {})[pointer] = value
    return out


def known(target: Any) -> bool:
    return bool(target.known_mask)


def predictions(row: Any, state_by_position: Mapping[int, Any]) -> list[Any]:
    """Every KNOWN target as the ``TypedPrediction`` list the chain consumes.

    The chain input is wider than the reward's typed truth (``typed_authority`` stays
    supervised-only): a DEC-049 withdrawn piece (known, not supervised) is still a
    value the model decode would have fed the builder, so omitting it leaves the gold
    AST unbuildable. A paragraph with no ``/layout/tabs`` record gets the explicit zero
    count the model always emits. Two different values for one ``(node, pointer)`` are
    refused here, before the chain could turn them into an opaque stream failure.

    The zero-tabs rule covers EVERY paragraph, unlike ``s4_typed_loss._zero_count_records``
    (supervised nodes only): it mirrors what ``decode_typed`` emits for the model side and
    never reaches the reward (``typed_authority`` is supervised-only)."""
    from unflatten.model.candidate_model import _DENSE_HEAD_ID
    from unflatten.model.typed_decode import TypedPrediction

    seen: dict[tuple[int, str], Any] = {}
    out: list[Any] = []
    labelled = [
        (t.anchor_token_index, t.node_property_pointer, t.head_id, t.value)
        for t in row.typed_targets
        if known(t)
    ]
    labelled += [
        (c.anchor_token_index, c.container_pointer, "cardinality", c.count)
        for c in row.typed_cardinalities
        if known(c)
    ]
    for anchor, pointer, head_id, value in labelled:
        state = state_by_position.get(anchor)
        if state is None:
            continue  # placed (or refused) by _typed_authority
        if seen.setdefault((anchor, pointer), value) != value:
            raise GoldTypedMappingFailed(
                f"{state.local_node_id!r} {pointer!r} has two different known values"
            )
        out.append(TypedPrediction(state.local_node_id, state.node_type, pointer, head_id, value))
    for anchor, state in state_by_position.items():
        if state.node_type == "paragraph" and (anchor, TABS) not in seen:
            out.append(TypedPrediction(state.local_node_id, "paragraph", TABS, _DENSE_HEAD_ID, 0))
    return out


_SPAN_POINTERS: Final = frozenset({"/row_span", "/col_span"})


def unsupervised_spans(row: Any, state_by_position: Mapping[int, Any]) -> int:
    """Cell span targets fed to the chain although known-not-supervised (DEC-049 withdrawn)."""
    return sum(
        1
        for t in row.typed_targets
        if t.node_property_pointer in _SPAN_POINTERS
        and known(t)
        and not t.supervision_mask
        and t.anchor_token_index in state_by_position
    )


def region_predictions(row: Any, state_by_position: Mapping[int, Any]) -> tuple[Any, ...]:
    """The row's region targets as sealed region predictions on the replayed node states.

    Mirrors ``quality_gate.gc._gold_regions``: the existing load-time policy decides
    what is excluded; a box is never fabricated or merged."""
    from unflatten.eval.s2_gate import sealed
    from unflatten.model.node_visual_aligner import STATUS_VOCABULARY
    from unflatten.model.s4_typed_loss import _derive_region_row, _RegionExcluded

    out: list[dict[str, Any]] = []
    for target in row.region_targets:
        state = state_by_position.get(target.anchor_token_index)
        if state is None or state.node_type != target.node_type:
            raise GoldTypedMappingFailed(
                f"region target {target.node_id!r} ({target.node_type!r}) anchors at token "
                f"{target.anchor_token_index}, which opens no matching node"
            )
        try:
            status, bbox, supervise_box = _derive_region_row(target)
        except _RegionExcluded:
            continue
        regions = (
            [
                {
                    "page_ordinal": 0,
                    "bbox_ppm": list(bbox),
                    "continuation": "complete",
                    "confidence_ppm": 1_000_000,
                }
            ]
            if supervise_box
            else []
        )
        out.append(
            {
                "local_node_id": state.local_node_id,
                "node_type": state.node_type,
                "alignment_status": STATUS_VOCABULARY[status],
                "confidence_ppm": 1_000_000,
                "regions": regions,
            }
        )
    return sealed._sealed_region_predictions(out)

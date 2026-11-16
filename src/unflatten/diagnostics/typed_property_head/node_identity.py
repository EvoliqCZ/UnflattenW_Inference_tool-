"""Design §2.3 — decoder-local node id -> canonical AST node id replay-and-zip.

This mapping does not exist anywhere else in the codebase: `NodeLifecycleTracker`
assigns tracker-local ids (`"n0", "n1", ...`, 0-based, acceptance order); the
canonical AST assigns ids (`"n1", "n2", ...`, 1-based, preorder) later, by the
not-yet-built AST Builder (`node_lifecycle.py` module docstring). Both halves
of the gold data are built from the SAME preorder walk
(`dataset.targets.events.preorder_nodes`), so replaying the gold structural
event stream through a fresh tracker and zipping the result against that walk
is how the alignment is verified directly, per page, rather than trusted from
the general argument alone (design §2.3).

The tracker's very first created state is always its synthesized `document`
root (the token stream is always wrapped in `DOCUMENT_OPEN`/`DOCUMENT_CLOSE`
by `encode_structural`, mirroring `derive_event_streams`'s synthesized `doc`
node), which owns no canonical id — `preorder_nodes` explicitly excludes it.
Skipping that one state is the "off-by-one index shift" design §2.3 names:
once skipped, the tracker's own `f"n{k}"` local id naming and the canonical
`f"n{k}"` preorder id naming are the SAME string at the SAME position, for a
token stream that never diverges from the gold stream.
"""

from __future__ import annotations

from typing import Any

import torch

from unflatten.dataset.targets.events import preorder_nodes
from unflatten.diagnostics.typed_property_head.anchor_extraction import extract_anchor
from unflatten.model.encode import encode_structural
from unflatten.model.node_lifecycle import NodeLifecycleTracker, NodeState


class NodeIdentityMismatch(ValueError):
    """The tracker's replay of the gold event stream did not line up,
    node-for-node, with the canonical AST's preorder node ids.

    Raised on ANY divergence -- differing node counts, a node-type
    mismatch, or an id that is not exactly the expected `f"n{i+1}"` at the
    same position -- never tolerated silently, per design §2.3.
    """


# Sentinel distinguishing "caller did not pass hidden at all" from
# "caller explicitly passed hidden=None" (finding 4): the former keeps
# `replay_and_align`'s original two-value-free return (a plain mapping
# dict), so every pre-existing caller/test is unaffected; the latter is
# `dataset.assemble_examples`'s single-replay call, which always needs
# the replayed `NodeState`s (anchor-filled or not) alongside the mapping,
# because `hidden=None` is itself a legitimate call shape there.
_HIDDEN_NOT_PASSED = object()


def replay_and_align(
    ast: dict[str, Any],
    gold_event_stream: dict[str, Any],
    *,
    hidden: torch.Tensor | None = _HIDDEN_NOT_PASSED,  # type: ignore[assignment]
    prefix_len: int = 0,
    visual_tokens: int = 0,
) -> dict[str, str] | tuple[dict[str, str], list[NodeState]]:
    """Map tracker-local node id -> canonical AST node id for one page.

    `gold_event_stream` is the RFC003 structural stream shape (e.g.
    `derive_event_streams(ast)["structural"]`): it is replayed through a
    fresh `NodeLifecycleTracker` via the same `encode_structural` bridge
    the decoder token alphabet uses, then zipped against
    `preorder_nodes(ast)` in order.

    Raises `NodeIdentityMismatch` if the tracker's first state is not the
    synthesized `document` root, if the replayed node count does not match
    the canonical preorder node count, or if any position's tracker id,
    canonical id or node type disagrees with the expected `f"n{i+1}"`
    alignment.

    Finding 4 (single replay): when `hidden` is passed as a keyword
    argument (even `hidden=None`), each created state's `anchor` is
    filled from `hidden` (via `anchor_extraction.extract_anchor`, or left
    `None` if `hidden is None`) during this SAME replay, and the return
    becomes `(mapping, states)` -- `states` are the exact replayed
    `NodeState` objects (identity, not a copy), so a caller like
    `dataset.assemble_examples` never needs a second, independent replay
    of the same stream to also collect anchors. When `hidden` is not
    passed at all, the return is the plain mapping dict, unchanged from
    before finding 4.
    """
    fill_anchors = hidden is not _HIDDEN_NOT_PASSED
    tokens = encode_structural(gold_event_stream)
    tracker = NodeLifecycleTracker()
    for index, token_name in enumerate(tokens):
        anchor = (
            extract_anchor(
                hidden, prefix_len=prefix_len, visual_tokens=visual_tokens, event_index=index
            )
            if fill_anchors and hidden is not None
            else None
        )
        tracker.apply_structural_token(token_name, event_index=index, anchor=anchor)

    states = tracker.states
    if not states or states[0].node_type != "document":
        raise NodeIdentityMismatch(
            "expected the tracker's first created state to be the synthesized "
            f"document root; got {states[0].node_type if states else None!r}"
        )
    replayed = states[1:]

    canonical_nodes = list(preorder_nodes(ast))
    if len(replayed) != len(canonical_nodes):
        raise NodeIdentityMismatch(
            f"tracker produced {len(replayed)} non-document node state(s) but "
            f"the canonical AST has {len(canonical_nodes)} preorder node(s)"
        )

    mapping: dict[str, str] = {}
    for i, (tracker_state, canonical_node) in enumerate(zip(replayed, canonical_nodes)):
        expected_id = f"n{i + 1}"
        canonical_id = canonical_node.get("id")
        if (
            tracker_state.local_node_id != expected_id
            or canonical_id != expected_id
            or tracker_state.node_type != canonical_node.get("type")
        ):
            raise NodeIdentityMismatch(
                f"node {i} misaligned: tracker id={tracker_state.local_node_id!r} "
                f"type={tracker_state.node_type!r}, canonical id={canonical_id!r} "
                f"type={canonical_node.get('type')!r}, expected id {expected_id!r}"
            )
        mapping[tracker_state.local_node_id] = canonical_id

    if not fill_anchors:
        return mapping
    return mapping, replayed

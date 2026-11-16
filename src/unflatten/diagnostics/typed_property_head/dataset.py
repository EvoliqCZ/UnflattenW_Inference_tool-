"""Design §5 — supervision assembly from stream C, for one page at a time.

Joins, per page:

- `dataset.targets.properties.emit_property_labels` (stream C: property
  VALUES with `known_mask`/`origin`/`verification`, RFC003 §27/§29.4);
- `dataset.targets.events.derive_event_streams` (the structural stream);
- `node_identity.replay_and_align` (tracker id -> canonical id, task 1);
- `anchor_extraction` (the per-node hidden state, task 2, with the
  missing-anchor guard, task 3);
- `pointer_map.DIAGNOSTIC_POINTER_TO_FAMILY` (which of the 29 slice-1
  pointers to keep).

**Hard boundary (design §5, boundary doc Q4)**: this module reads stream C
directly from the same canonical AST the rest of the dataset layer
already builds from. It never imports M1/S1's exact-text-cache module,
constructs an `s1_run_manifest`, or touches `LossUnit`/`Locator`/
`LossFamily`. See
`tests/diagnostics/typed_property_head/test_dataset_boundary_scan.py`.

`hidden=None` is a legitimate call shape (no per-node hidden state
available for this page yet) -- it is NOT silently skipped: every
in-scope node ends up with `anchor=None`, and `check_anchors_present`
raises `MissingAnchorError` for the whole page, per design §2.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from unflatten.dataset.targets.events import derive_event_streams
from unflatten.dataset.targets.properties import emit_property_labels
from unflatten.diagnostics.typed_property_head.anchor_extraction import check_anchors_present
from unflatten.diagnostics.typed_property_head.node_identity import replay_and_align
from unflatten.diagnostics.typed_property_head.pointer_map import (
    DIAGNOSTIC_POINTER_TO_FAMILY,
)


@dataclass(frozen=True)
class PropertyExample:
    """One `(anchor, value, known_mask, value_type, origin, verification)`
    training example (design §5 step 4). `origin`/`verification` ride
    along unused by the loss so a future review can stratify results by
    label provenance without re-deriving it (design §5)."""

    node_id: str
    node_type: str
    pointer: str
    family: str
    anchor: torch.Tensor
    value: Any
    known_mask: bool
    value_type: str
    origin: str
    verification: str


def assemble_examples(
    ast: dict[str, Any],
    hidden: torch.Tensor | None,
    *,
    prefix_len: int,
    visual_tokens: int,
    label_origin: str = "synthetic_generator",
    verification: str = "unverified",
) -> list[PropertyExample]:
    """Assemble slice-1 typed property training examples for one page.

    Raises `MissingAnchorError` (via `check_anchors_present`) if any node
    in the page has no anchor -- including every node, if `hidden` is
    `None`. Raises `NodeIdentityMismatch` (via `replay_and_align`) if the
    gold event stream does not line up with the canonical AST.
    """
    labels = emit_property_labels(ast, label_origin=label_origin, verification=verification)
    gold_event_stream = derive_event_streams(ast)["structural"]

    # Finding 4: a single replay of the gold stream both aligns tracker
    # ids to canonical ids AND fills anchors from `hidden` -- passing
    # `hidden=hidden` as a keyword (even when `hidden is None`) selects
    # `replay_and_align`'s `(mapping, states)` return shape.
    tracker_to_canonical, states = replay_and_align(
        ast, gold_event_stream, hidden=hidden, prefix_len=prefix_len, visual_tokens=visual_tokens
    )
    canonical_to_tracker = {v: k for k, v in tracker_to_canonical.items()}

    check_anchors_present(states)
    state_by_tracker_id = {state.local_node_id: state for state in states}

    examples: list[PropertyExample] = []
    for node_id, entry in labels.items():
        node_type = entry["node_type"]
        for field in entry["fields"]:
            pointer = field["node_property_pointer"]
            family = DIAGNOSTIC_POINTER_TO_FAMILY.get((node_type, pointer))
            if family is None:
                continue
            tracker_id = canonical_to_tracker[node_id]
            state = state_by_tracker_id[tracker_id]
            examples.append(
                PropertyExample(
                    node_id=node_id,
                    node_type=node_type,
                    pointer=pointer,
                    family=family,
                    anchor=state.anchor,
                    value=field["value"],
                    known_mask=field["known_mask"],
                    value_type=field["value_type"],
                    origin=field["origin"],
                    verification=field["verification"],
                )
            )
    return examples

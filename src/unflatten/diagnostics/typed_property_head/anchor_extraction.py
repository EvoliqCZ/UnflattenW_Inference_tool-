"""Design §2.2/§2.4 — teacher-forced anchor index arithmetic and extraction.

**Index arithmetic, derived from the code (not the design's prose) in
`candidate_model.py:235-331` and `free_running_decode.py`'s own use of
`encode_hidden`, then checked against the design's formula.**

`QwenDenseCandidateModel.encode_hidden` returns `hidden` of shape
`(batch, prefix_len + steps, d_model)`: `prefix_len` positions from
`document_context_encoder`, then `steps = composite_input_ids.shape[1]`
positions from the merged composite sequence -- and `composite_input_ids`
already carries the visual placeholder block FIRST (one packed
`QwenRef` per visual token, in order), then the grammar-generated stream,
exactly the `placeholder_block + accepted` layout `free_running_decode.py`
builds by hand every step.

`free_running_decode.py` computes `hidden` from
`input_ids = [placeholder_block (visual_tokens long) ; accepted (k long)]`
BEFORE appending the chosen token, reads `hidden[:, -1, :]` (position
`prefix_len + visual_tokens + k - 1`) as `next_position_hidden`, appends
the chosen token (now `len(accepted) == k + 1`), and THEN calls
`tracker.apply_structural_token(..., event_index=len(accepted) - 1, ...)`
-- so `event_index == k`. Substituting: the anchor position is
`prefix_len + visual_tokens + event_index - 1`. This is exactly the
formula design §2.2 states; deriving it independently from the two call
sites above confirms the design's arithmetic rather than contradicting it
-- no discrepancy found here.

For the teacher-forced case, `composite_input_ids` for a whole page already
IS `placeholder_block + <gold accepted stream>` (`TeacherForcedBatch`'s own
docstring: "every position standing for a visual token MUST be a packed
`QwenRef`... in order... ahead of whatever the grammar generates" is the
free-running framing; the teacher-forced fixture is built the same way, one
forward pass over the whole gold stream instead of one step at a time), so
the same formula applies with `event_index` read from the gold token
stream's 0-based position within the non-visual tail (`derive_event_streams`
+ `NodeLifecycleTracker` replay, §2.2/§2.3), not recomputed per step.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from unflatten.model.node_lifecycle import NodeState


class MissingAnchorError(ValueError):
    """A `NodeState` in scope has `anchor is None` (design §2.4). `anchor`
    is an optional kwarg on `open()`/`leaf()`/`apply_structural_token()`
    with no API-level guarantee of non-`None` -- this check is the only
    thing that enforces it, so it must raise, never skip silently."""


class AnchorIndexOutOfRange(IndexError):
    """`anchor_index` computed a position outside `hidden`'s sequence
    length -- raised, never clamped, per design §2.2's exactness
    requirement: a wrong index silently clamped to a valid one would read a
    real but meaningless hidden state instead of surfacing the bug."""


def anchor_index(*, prefix_len: int, visual_tokens: int, event_index: int, seq_len: int) -> int:
    """The 0-based position in an `encode_hidden` `hidden` tensor (sequence
    length `seq_len`) that is the teacher-forced anchor for a gold
    structural token at 0-based `event_index` within the accepted-token
    stream (see module docstring for the derivation).

    Raises `AnchorIndexOutOfRange` if the computed position falls outside
    `[0, seq_len)` -- this is a caller bug (wrong `prefix_len`/
    `visual_tokens`/`event_index`), not a case to clamp around.
    """
    idx = prefix_len + visual_tokens + event_index - 1
    if idx < 0 or idx >= seq_len:
        raise AnchorIndexOutOfRange(
            f"anchor index {idx} (prefix_len={prefix_len}, visual_tokens={visual_tokens}, "
            f"event_index={event_index}) is out of range for hidden sequence length {seq_len}"
        )
    return idx


def extract_anchor(
    hidden: torch.Tensor, *, prefix_len: int, visual_tokens: int, event_index: int
) -> torch.Tensor:
    """The `(d_model,)` anchor slice of `hidden` (shape `(seq_len, d_model)`,
    i.e. one page's already-batch-indexed `encode_hidden` output) for a gold
    structural token at `event_index` -- see `anchor_index` for the position
    formula.
    """
    if hidden.dim() != 2:
        raise ValueError(
            "extract_anchor expects hidden shaped (seq_len, d_model) for one "
            f"page; got shape {tuple(hidden.shape)}"
        )
    idx = anchor_index(
        prefix_len=prefix_len,
        visual_tokens=visual_tokens,
        event_index=event_index,
        seq_len=hidden.shape[0],
    )
    return hidden[idx]


def check_anchors_present(states: Iterable[NodeState]) -> None:
    """Design §2.4: raise `MissingAnchorError` if any in-scope `NodeState`
    has `anchor is None`. `anchor is None` is a bug signal under V0's
    `document`-mode-only tracker, never a valid runtime case to route
    around -- silently dropping such a node would fabricate an implicit
    `no_supervision` indistinguishable from a real `known_mask=false`
    (design §2.4, §6)."""
    for state in states:
        if state.anchor is None:
            raise MissingAnchorError(
                f"node {state.local_node_id!r} ({state.node_type!r}, "
                f"slot_kind={state.slot_kind!r}) has anchor=None"
            )

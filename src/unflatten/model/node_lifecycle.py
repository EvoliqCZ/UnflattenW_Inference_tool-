"""Local node slots/anchors -- design §5.4 D4, §8 M3 item 2.

A `NodeState` is created only for an ACCEPTED structural open/leaf token,
never invented ahead of decode: `NodeLifecycleTracker.apply_structural_token`
is fed one already-grammar-legal token name at a time, in exactly the order
`composite_grammar.CompositeGrammarController.step` accepted it, and derives
`parent_local_node_id` only from ITS OWN local open-node stack -- never from
a canonical id or an arbitrary parent id the model would have to produce
(design §5.4: "Model negeneruje canonical ID ani arbitrary parent ID").
Canonical preorder ids are assigned later, by the (not-yet-built) AST
Builder, from the *closed* node list this tracker hands off.

The real training cache is fragment-shaped, not document-shaped (measured
2026-09-07 on `cache-exact-text-300dpi-ab-1176-c5`: every page's target is
`FRAGMENT_BEGIN, RESUME_NODE(document), ..., SUSPEND_NODE(document),
END_FRAGMENT`, never `END_DOCUMENT`), and `grammar.py`'s fragment protocol
(§18.8) legally offers `RESUME_NODE`/`SUSPEND_NODE` from the very first
structural token onward. This tracker therefore handles both: a
`RESUME_NODE(t)` re-establishes an already-existing node as a local
`continuation_proxy` slot (`resume_event_index` set, `open_event_index`
left `None` -- this tracker never saw and never invents that node's
original open), and a matching `SUSPEND_NODE(t)` pops that slot off the
local open stack WITHOUT sealing it (`close_event_index` stays `None` --
the node itself lives on past this window's boundary; only this window's
local view of it ends). `slot_kind="continuation_proxy"` and
`resume_event_index` on `NodeState` exist for exactly this.

This module authorizes no gradient update: it declares dataclasses and a
plain state-machine class; `anchor`/`content_summary` hold tensors handed
in by the caller, but nothing here computes a gradient or builds a module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch

from unflatten.model import events_vocab

SlotKind = Literal["opened_node", "leaf", "continuation_proxy"]
SLOT_KINDS: frozenset[str] = frozenset({"opened_node", "leaf", "continuation_proxy"})

_OPEN_NODE_TYPE: dict[str, str] = {
    token: node_type for node_type, token in events_vocab.OPEN_TOKEN.items()
}
_CLOSE_TOKEN_SET: frozenset[str] = frozenset(events_vocab.CLOSE_TOKEN.values())
_LEAF_NODE_TYPE: dict[str, str] = {
    token: node_type for node_type, token in events_vocab.LEAF_TOKEN.items()
}
_RESUME_NODE_TYPE: dict[str, str] = {
    token: node_type for node_type, token in events_vocab.RESUME_TOKEN.items()
}
_RESUME_TOKEN_SET: frozenset[str] = frozenset(events_vocab.RESUME_TOKEN.values())
_SUSPEND_TOKEN_SET: frozenset[str] = frozenset(events_vocab.SUSPEND_TOKEN.values())

#: Every structural token name this tracker refuses via `UnsupportedResumeInV0`.
#: Public so `composite_grammar` can exclude the same names from the V0
#: legal-token domain instead of restating "V0 never resumes/suspends" as a
#: second, independently-maintained list -- the two must never diverge on
#: what that sentence means.
RESUME_OR_SUSPEND_TOKEN_NAMES: frozenset[str] = _RESUME_TOKEN_SET | _SUSPEND_TOKEN_SET


class NodeLifecycleError(ValueError):
    """The tracker's OWN local-stack invariant was violated -- e.g. `close()`
    with nothing open. Distinct from `grammar.GrammarViolation`: the grammar
    controller already rejected any token that would cause this before the
    tracker ever saw it, so reaching this error means the caller fed tokens
    out of the order the controller actually accepted them in."""


class UnsupportedResumeInV0(NotImplementedError):
    """Retained type for callers that still reference it (e.g.
    `eval/s2_gate/decode.py`'s `unsupported_resume` outcome category); as of
    the fragment-protocol fix above, `NodeLifecycleTracker` no longer raises
    it -- `RESUME_NODE`/`SUSPEND_NODE` are handled, not refused."""


@dataclass(frozen=True, slots=True)
class NodeState:
    """One local node slot (design §5.4). `local_node_id` is tracker-assigned
    (`"n<k>"`, acceptance order) -- never a canonical preorder id, which the
    AST Builder assigns later from the closed node list."""

    local_node_id: str
    slot_kind: SlotKind
    node_type: str
    parent_local_node_id: str | None
    open_event_index: int | None = None
    leaf_event_index: int | None = None
    resume_event_index: int | None = None
    close_event_index: int | None = None
    anchor: torch.Tensor | None = None
    content_summary: torch.Tensor | None = None


class NodeLifecycleTracker:
    """Builds `NodeState` entries from an accepted structural token stream."""

    def __init__(self) -> None:
        self._states: list[NodeState] = []
        self._open_stack: list[int] = []  # indices into self._states
        self._counter = 0

    @property
    def states(self) -> tuple[NodeState, ...]:
        """All node states created so far, acceptance order (open/leaf order,
        not closure order) -- closed states are updated in place, not moved."""
        return tuple(self._states)

    def _next_id(self) -> str:
        node_id = f"n{self._counter}"
        self._counter += 1
        return node_id

    def _parent(self) -> str | None:
        if not self._open_stack:
            return None
        return self._states[self._open_stack[-1]].local_node_id

    def open(
        self, node_type: str, *, event_index: int, anchor: torch.Tensor | None = None
    ) -> NodeState:
        """A container OPEN token was accepted: push a new `opened_node` slot."""
        state = NodeState(
            local_node_id=self._next_id(),
            slot_kind="opened_node",
            node_type=node_type,
            parent_local_node_id=self._parent(),
            open_event_index=event_index,
            anchor=anchor,
        )
        self._states.append(state)
        self._open_stack.append(len(self._states) - 1)
        return state

    def leaf(
        self, node_type: str, *, event_index: int, anchor: torch.Tensor | None = None
    ) -> NodeState:
        """A single-token leaf control was accepted: one closed `leaf` slot,
        never pushed onto the open stack (§18.6: it opens and closes at once)."""
        state = NodeState(
            local_node_id=self._next_id(),
            slot_kind="leaf",
            node_type=node_type,
            parent_local_node_id=self._parent(),
            leaf_event_index=event_index,
            anchor=anchor,
        )
        self._states.append(state)
        return state

    def close(self, *, event_index: int) -> NodeState:
        """A container CLOSE token was accepted: pop and seal the top slot."""
        if not self._open_stack:
            raise NodeLifecycleError("close() called with no open node on the local stack")
        index = self._open_stack.pop()
        closed = replace(self._states[index], close_event_index=event_index)
        self._states[index] = closed
        return closed

    def resume(
        self, node_type: str, *, event_index: int, anchor: torch.Tensor | None = None
    ) -> NodeState:
        """A `RESUME_NODE(t)` token was accepted: push a new `continuation_proxy`
        slot for an already-existing node -- never a second `opened_node` for
        it (this tracker never saw, and never invents, that node's original
        open event)."""
        state = NodeState(
            local_node_id=self._next_id(),
            slot_kind="continuation_proxy",
            node_type=node_type,
            parent_local_node_id=self._parent(),
            resume_event_index=event_index,
            anchor=anchor,
        )
        self._states.append(state)
        self._open_stack.append(len(self._states) - 1)
        return state

    def suspend(self, *, event_index: int) -> NodeState:
        """A `SUSPEND_NODE(t)` token was accepted: pop the resumed frame's
        local view WITHOUT sealing it (`close_event_index` stays `None` --
        the node lives on past this window's boundary; only this window's
        proxy for it ends here)."""
        del event_index  # no field records it; suspend never seals close_event_index
        if not self._open_stack:
            raise NodeLifecycleError("suspend() called with no open node on the local stack")
        index = self._open_stack.pop()
        return self._states[index]

    def apply_structural_token(
        self, token_name: str, *, event_index: int, anchor: torch.Tensor | None = None
    ) -> NodeState | None:
        """Route one accepted structural token name to open/leaf/close.

        Returns the `NodeState` it created or closed, or `None` for a token
        that owns no node slot of its own (`TEXT_BEGIN`/`TEXT_END` belong to
        the current `run` node; `END_DOCUMENT`/`FRAGMENT_BEGIN`/
        `END_FRAGMENT` are stream-level controls, not nodes).
        """
        if token_name in _RESUME_TOKEN_SET:
            return self.resume(
                _RESUME_NODE_TYPE[token_name], event_index=event_index, anchor=anchor
            )
        if token_name in _SUSPEND_TOKEN_SET:
            return self.suspend(event_index=event_index)
        if token_name in _OPEN_NODE_TYPE:
            return self.open(_OPEN_NODE_TYPE[token_name], event_index=event_index, anchor=anchor)
        if token_name in _LEAF_NODE_TYPE:
            return self.leaf(_LEAF_NODE_TYPE[token_name], event_index=event_index, anchor=anchor)
        if token_name in _CLOSE_TOKEN_SET:
            return self.close(event_index=event_index)
        return None

"""RFC004 §18.4–§18.5 Grammar Controller.

A deterministic pushdown state machine over the event alphabet
(:mod:`unflatten.model.events_vocab`), derived from the RFC001
parent-child rules and the ACTIVE capability profile (§18.4) —
``word-core-v0``, ``word-visual-v1`` or ``word-visual-v1.1``. States are immutable;
:func:`step` with a token outside :func:`allowed_events`
raises :class:`GrammarViolation` — masking invalid transitions is
mandatory for both training-compatible decode and production inference.

The parent-child matrix is IMPORTED from
:mod:`unflatten.ast.semantic_validator` (``_PARENT_CHILD`` /
``_BLOCK_TYPES`` / ``_INLINE_TYPES``) so the two can never drift; a sync
test additionally pins the imported tables. On top of the matrix the
grammar enforces (§18.5):

- ``document`` first, sections inside, ``DOCUMENT_CLOSE`` then the
  ``END_DOCUMENT`` termination claim at empty stack (terminal),
- ``cell`` closes only with >= 1 paragraph child AND a paragraph as its
  last child (the AST side enforces the same rule via
  ``CELL_MISSING_PARAGRAPH`` / ``CELL_LAST_CHILD_NOT_PARAGRAPH`` in
  :mod:`unflatten.ast.semantic_validator` — no dataset-grammar drift),
- nested tables masked in the ``word-core-v0`` model path (no
  ``TABLE_OPEN`` — or ``RESUME_NODE(table)`` — while a table frame is on
  the stack); ``word-visual-v1`` lifts that mask (DEC-003 §1),
- ``TEXT_BEGIN`` only inside a run, ``TEXT_END`` only after
  ``TEXT_BEGIN``, at most one text span per run (per window),
- ``hyperlink`` closes only with >= 1 child (the schema puts
  ``minItems: 1`` on ``hyperlink.children``),
- ``note_ref`` masked anywhere inside a ``footnote``/``endnote`` body
  (``NOTE_REF_IN_NOTE`` on the AST side; the parent-child matrix cannot
  express it, since a note body holds blocks which hold inlines).
- ``text_frame`` is eligible only under ``word-visual-v1.1`` and is masked
  inside another text frame (``NESTED_TEXT_FRAME_UNSUPPORTED``).

The alphabet spans both capability profiles, so profile eligibility is a
MASK here (§18.5 "V1/V2 nodes su maskovane"), not an absent token:
:func:`child_types` intersects the RFC001 matrix with
``events_vocab.node_types(profile)``. Under ``word-core-v0`` the
``word-visual-v1`` types appear in no mask at all; under
``word-visual-v1`` they do.

Stories (``header``/``footer`` on a section, ``footnote``/``endnote`` on
the document) are NOT children in the §15 matrix — they hang off their own
AST keys. The grammar attaches them as ORDERED child groups taken from
:mod:`unflatten.ast.canonicalizer` (``_SECTION_CHILD_KEYS``,
``_NOTE_KEYS``), so the event stream's order is the canonical preorder by
construction: section → headers → footers → body, and document →
sections → footnotes → endnotes. A frame's ``group`` index only ever
moves forward, which is what forbids a header after body content. That
order is load-bearing: RFC003 fragment ownership and every ``ast_hash``
inherit it.

Fragment protocol (§18.8): ``FRAGMENT_BEGIN``, then outer→inner
``RESUME_NODE`` claims (ordering enforced by the same parent-child
matrix; each pushes a continuation-proxy frame; the chain is ANCHORED —
at empty stack only ``RESUME_NODE(document)`` is a legal resume), then
content, then ``SUSPEND_NODE`` unwinding only resumed frames inner→outer
(a resumed node may instead close normally), and ``END_FRAGMENT``
terminal at balanced (empty) stack with no pending text span.
``END_DOCUMENT`` is NEVER legal in fragment mode: when the resumed
document frame closes or suspends, the only remaining token is
``END_FRAGMENT``.

Documented A2 decisions where the RFC is underspecified:

- Empty containers: the AST schema puts no ``minItems`` on ``sections`` /
  ``blocks`` / ``items`` / ``rows`` / ``cells``, so the grammar allows
  empty document/section/list/table/row (only the cell rule is explicit
  in §18.5). Semantic/grid validity stays the validator's job.
- Fragment root context: with no left overlap a fragment may open a
  ``section`` or any block directly (never ``DOCUMENT_OPEN`` — a full
  document uses document mode); ``RESUME_NODE(document)`` covers windows
  resuming at document level between sections.
- A resumed ``cell`` may close with zero local children (its mandatory
  paragraph may live left of the boundary); with local children the
  last-child-is-paragraph rule still applies.
- Only resumed frames can be suspended: a node opened inside this window
  that continues right is out of scope for A2 (assembler concern).
- A resumed frame starts at child group 0: the window carries no evidence
  of how many stories already closed left of the boundary, so ordering
  within a resumed frame is not enforced (same reasoning as the resumed
  ``cell`` relaxation). Full-document decode is where the order is pinned.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from unflatten.ast.canonicalizer import _NOTE_KEYS, _SECTION_CHILD_KEYS
from unflatten.ast.semantic_validator import (
    _BLOCK_TYPES,
    _HYPERLINK_CHILD_TYPES,
    _INLINE_TYPES,
    _PARENT_CHILD,
    _STORY_BLOCK_TYPES,
)
from unflatten.model.events_vocab import (
    CLOSE_TOKEN,
    CONTAINER_TYPES,
    END_DOCUMENT,
    END_FRAGMENT,
    FRAGMENT_BEGIN,
    LEAF_TOKEN,
    OPEN_TOKEN,
    PROFILE_NODE_TYPES,
    RESUME_TOKEN,
    SUSPEND_TOKEN,
    TEXT_BEGIN,
    TEXT_END,
    TOKENS,
    node_types,
)
from unflatten.model.hashing import domain_hash

PROFILES: frozenset[str] = frozenset(PROFILE_NODE_TYPES)

#: Modules whose source decides what this controller permits: the phase
#: logic here, the token alphabet, and RFC001's parent-child matrix.
CONTROLLER_MODULES: tuple[str, ...] = (
    "unflatten.ast.canonicalizer",
    "unflatten.ast.semantic_validator",
    "unflatten.model.events_vocab",
    "unflatten.model.grammar",
)

_CONTROLLER_HASH_DOMAIN = "rfc004.grammar-controller/v1"


def _module_digest(module: Any) -> str:
    """sha256 of a module's source file."""
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def controller_hash() -> str:
    """Identity of the code that decides what this grammar permits.

    Computed from the modules as they are AT RUN TIME, the way
    `m2b.derivation_hash` is, so a phase-logic change or a parent-child
    rule change moves it with nobody having to remember anything. A
    hand-maintained constant would be the `MIGRATION_VERSION` defect
    again: a field that claims to identify a version and identifies
    nothing once someone forgets to bump it.

    The cost is that a comment edit moves it too. That is affordable
    here because this is a LINEAGE field, not a freshness one: nothing
    re-derives a cache from it, and the only thing it decides is whether
    a train and a val cache were built by the same controller. Were it
    ever used to invalidate an existing cache, source bytes would be far
    too eager and this should become a digest of the controller's
    behaviour instead.
    """
    payload = {
        name: _module_digest(importlib.import_module(name))
        for name in CONTROLLER_MODULES
    }
    return domain_hash(_CONTROLLER_HASH_DOMAIN, payload)

# Profile view of the RFC001 matrix: a node type the active capability
# profile does not admit disappears from both the parent set and the child
# taxonomies. Keyed on `events_vocab.node_types(profile)` rather than on a
# hardcoded name (`hyperlink` used to be spelled out here), so growing a
# profile is a one-line change in the alphabet.
_RULE_SETS: dict[str, frozenset[str]] = {
    "block": frozenset(_BLOCK_TYPES),
    "story_block": frozenset(_STORY_BLOCK_TYPES),
    "inline": frozenset(_INLINE_TYPES),
    "hyperlink": frozenset(_HYPERLINK_CHILD_TYPES),
}

# Story keys -> the node type they hold. The ORDER of the tuples is the
# canonicalizer's preorder (`_SECTION_CHILD_KEYS`, `_NOTE_KEYS`); the
# grammar reads it from there so the two cannot drift.
_SECTION_KEY_TYPE: dict[str, str | None] = {
    "headers": "header",
    "footers": "footer",
    "children": None,  # the §15 matrix rule for `section`
}
_NOTE_KEY_TYPE: dict[str, str | None] = {"footnotes": "footnote", "endnotes": "endnote"}


def _rule_children(rule: set[str] | str) -> frozenset[str]:
    if not isinstance(rule, str):
        return frozenset(rule)
    try:
        return _RULE_SETS[rule]
    except KeyError:
        raise ValueError(
            f"semantic_validator._PARENT_CHILD uses taxonomy sentinel {rule!r}, "
            f"which this grammar does not know; supported: {sorted(_RULE_SETS)}"
        ) from None


def _story_group(key: str, mapping: dict[str, str | None], eligible: frozenset[str]):
    """The child group held by AST key `key`, or None for the matrix rule."""
    if key not in mapping:
        raise ValueError(
            f"canonicalizer grew child key {key!r}; the grammar has no node type "
            f"for it and would silently drop that part of the preorder"
        )
    node_type = mapping[key]
    return None if node_type is None else frozenset({node_type}) & eligible


def _build_child_groups(profile: str) -> dict[str, tuple[frozenset[str], ...]]:
    """Per parent type, its child groups in canonical preorder."""
    eligible = node_types(profile)
    groups: dict[str, tuple[frozenset[str], ...]] = {}
    for parent, rule in _PARENT_CHILD.items():
        if parent not in eligible:
            continue
        groups[parent] = (_rule_children(rule) & eligible,)

    if "section" in groups:
        matrix_children = groups["section"][0]
        ordered = []
        for key in _SECTION_CHILD_KEYS:
            group = _story_group(key, _SECTION_KEY_TYPE, eligible)
            group = matrix_children if group is None else group
            if group:
                ordered.append(group)
        groups["section"] = tuple(ordered)

    if "document" in groups:
        ordered = [groups["document"][0]]
        for key in _NOTE_KEYS:
            group = _story_group(key, _NOTE_KEY_TYPE, eligible)
            if group:
                ordered.append(group)
        groups["document"] = tuple(ordered)
    return groups


@cache
def _child_groups(profile: str) -> Mapping[str, tuple[frozenset[str], ...]]:
    return MappingProxyType(_build_child_groups(profile))


@cache
def child_types(profile: str) -> Mapping[str, frozenset[str]]:
    """Flattened parent -> admissible child types under `profile`."""
    return MappingProxyType(
        {
            parent: frozenset().union(*groups)
            for parent, groups in _child_groups(profile).items()
        }
    )


@cache
def _fragment_root_children(profile: str) -> frozenset[str]:
    """A window may start at a section boundary, inside a section body, or
    inside a story that a page cut through."""
    table = child_types(profile)
    return table["document"] | table["section"]


_CONTAINER_SET = frozenset(CONTAINER_TYPES)
_LEAF_SET = frozenset(LEAF_TOKEN)
_NOTE_TYPES = frozenset({"footnote", "endnote"})

# Reverse token lookups.
_OPEN_BY_TOKEN = {v: k for k, v in OPEN_TOKEN.items()}
_CLOSE_BY_TOKEN = {v: k for k, v in CLOSE_TOKEN.items()}
_LEAF_BY_TOKEN = {v: k for k, v in LEAF_TOKEN.items()}
_RESUME_BY_TOKEN = {v: k for k, v in RESUME_TOKEN.items()}
_SUSPEND_BY_TOKEN = {v: k for k, v in SUSPEND_TOKEN.items()}

_PHASE_START = "start"
_PHASE_DOCUMENT = "document"
_PHASE_DOCUMENT_CLOSED = "document_closed"
_PHASE_TERMINATED = "terminated"
_PHASE_FRAGMENT_RESUME = "fragment_resume"
_PHASE_FRAGMENT_CONTENT = "fragment_content"
_PHASE_FRAGMENT_SUSPEND = "fragment_suspend"
_PHASE_FRAGMENT_DONE = "fragment_done"

_TERMINAL_PHASES = frozenset({_PHASE_TERMINATED, _PHASE_FRAGMENT_DONE})
_FRAGMENT_PHASES = frozenset(
    {_PHASE_FRAGMENT_RESUME, _PHASE_FRAGMENT_CONTENT, _PHASE_FRAGMENT_SUSPEND}
)

# Run text-span states.
_TEXT_NONE = 0
_TEXT_OPEN = 1
_TEXT_DONE = 2


class GrammarViolation(ValueError):
    """A decode token was not in the allowed_events mask of the state."""


@dataclass(frozen=True, slots=True)
class _Frame:
    node_type: str
    resumed: bool = False
    #: Index into this frame's child groups; only ever moves forward, so
    #: `headers -> footers -> body` and `sections -> footnotes -> endnotes`
    #: cannot be emitted out of canonical preorder.
    group: int = 0
    child_count: int = 0
    has_paragraph_child: bool = False
    last_child_is_paragraph: bool = False
    text_state: int = _TEXT_NONE


@dataclass(frozen=True, slots=True)
class GrammarState:
    profile: str
    phase: str
    stack: tuple[_Frame, ...]


def fresh(profile: str) -> GrammarState:
    """Initial state; the first token selects document or fragment mode."""
    if profile not in PROFILES:
        raise ValueError(f"unknown grammar profile {profile!r}; supported: {sorted(PROFILES)}")
    return GrammarState(profile=profile, phase=_PHASE_START, stack=())


def allowed_events(state: GrammarState) -> frozenset[str]:
    """The exact set of tokens `step` accepts from `state` (§18.4 mask)."""
    phase = state.phase
    if phase in _TERMINAL_PHASES:
        return frozenset()
    if phase == _PHASE_START:
        return frozenset({OPEN_TOKEN["document"], FRAGMENT_BEGIN})
    if phase == _PHASE_DOCUMENT_CLOSED:
        return frozenset({END_DOCUMENT})

    stack = state.stack
    top = stack[-1] if stack else None

    if phase == _PHASE_FRAGMENT_SUSPEND:
        if top is None:
            return frozenset({END_FRAGMENT})
        # Suspends only unwind resumed frames; they form the whole residual
        # stack once suspension starts.
        return frozenset({SUSPEND_TOKEN[top.node_type]})

    allowed: set[str] = set()

    if top is None:
        # Fragment root (in document mode the document frame is on the
        # stack until DOCUMENT_CLOSE flips the phase).
        allowed |= _open_tokens(_fragment_root_children(state.profile), state)
        allowed.add(END_FRAGMENT)
    elif top.node_type == "run":
        if top.text_state == _TEXT_OPEN:
            return frozenset({TEXT_END})
        if top.text_state == _TEXT_NONE:
            allowed.add(TEXT_BEGIN)
        allowed.add(CLOSE_TOKEN["run"])
    else:
        allowed |= _open_tokens(_eligible_children(state, top), state)
        if _close_allowed(top):
            allowed.add(CLOSE_TOKEN[top.node_type])

    if phase == _PHASE_FRAGMENT_RESUME:
        allowed |= _resume_tokens(state, top)
    if phase in _FRAGMENT_PHASES and top is not None and top.resumed:
        if not (top.node_type == "run" and top.text_state == _TEXT_OPEN):
            allowed.add(SUSPEND_TOKEN[top.node_type])
    return frozenset(allowed)


def step(state: GrammarState, token: str) -> GrammarState:
    """Apply one token; raises GrammarViolation unless it is allowed."""
    if token not in allowed_events(state):
        raise GrammarViolation(
            f"token {token!r} not allowed in phase {state.phase!r} "
            f"(stack: {[f.node_type for f in state.stack]}; "
            f"allowed: {sorted(allowed_events(state))})"
        )
    return _apply(state, token)


def _eligible_children(state: GrammarState, top: _Frame) -> frozenset[str]:
    """The child types `top` may still open: its current group onward."""
    groups = _child_groups(state.profile)[top.node_type]
    return frozenset().union(*groups[top.group :]) if top.group < len(groups) else frozenset()


def _masked_children(state: GrammarState) -> frozenset[str]:
    """Child types no frame may open here, whatever the matrix says.

    Both rules are context-sensitive over the whole stack, which is
    exactly what a parent-child matrix cannot express.
    """
    masked: set[str] = set()
    if state.profile == "word-core-v0" and any(f.node_type == "table" for f in state.stack):
        masked.add("table")  # nested tables masked in word-core-v0 (§18.5)
    if any(f.node_type in _NOTE_TYPES for f in state.stack):
        masked.add("note_ref")  # NOTE_REF_IN_NOTE on the AST side
    if any(f.node_type == "text_frame" for f in state.stack):
        masked.add("text_frame")  # NESTED_TEXT_FRAME_UNSUPPORTED on AST side
    return frozenset(masked)


def _open_tokens(child_types: frozenset[str], state: GrammarState) -> set[str]:
    tokens: set[str] = set()
    for child in child_types - _masked_children(state):
        if child in _LEAF_SET:
            tokens.add(LEAF_TOKEN[child])
        else:
            tokens.add(OPEN_TOKEN[child])
    return tokens


def _resume_tokens(state: GrammarState, top: _Frame | None) -> set[str]:
    if top is None:
        # Resume chains are anchored: a boundary claim at empty stack can
        # only be the document frame — deeper resumes must spell out their
        # ancestor chain (outer→inner) through the parent-child matrix.
        resumable = frozenset({"document"})
    elif top.node_type == "run":
        resumable = frozenset()
    else:
        resumable = _eligible_children(state, top) & _CONTAINER_SET
    return {
        RESUME_TOKEN[t]
        for t in resumable - _masked_children(state)
        if t in RESUME_TOKEN
    }


def _close_allowed(frame: _Frame) -> bool:
    if frame.node_type == "hyperlink":
        # schema: `hyperlink.children` has minItems 1. A resumed hyperlink
        # may have its children left of the boundary.
        return frame.resumed or frame.child_count >= 1
    if frame.node_type != "cell":
        return True
    if frame.resumed and frame.child_count == 0:
        # mandatory paragraph may live left of the fragment boundary
        return True
    return frame.has_paragraph_child and frame.last_child_is_paragraph


def _apply(state: GrammarState, token: str) -> GrammarState:
    phase = state.phase
    stack = state.stack
    if phase == _PHASE_FRAGMENT_RESUME and token not in _RESUME_BY_TOKEN:
        phase = _PHASE_FRAGMENT_CONTENT

    if token == FRAGMENT_BEGIN:
        return replace(state, phase=_PHASE_FRAGMENT_RESUME)
    if token == END_FRAGMENT:
        return replace(state, phase=_PHASE_FRAGMENT_DONE)
    if token == END_DOCUMENT:
        return replace(state, phase=_PHASE_TERMINATED)
    if token in _OPEN_BY_TOKEN:
        node_type = _OPEN_BY_TOKEN[token]
        if node_type == "document":
            phase = _PHASE_DOCUMENT
        stack = _advance_group(state, stack, node_type)
        return replace(state, phase=phase, stack=stack + (_Frame(node_type),))
    if token in _RESUME_BY_TOKEN:
        frame = _Frame(_RESUME_BY_TOKEN[token], resumed=True)
        return replace(state, phase=phase, stack=stack + (frame,))
    if token in _LEAF_BY_TOKEN:
        leaf_type = _LEAF_BY_TOKEN[token]
        new_stack = _record_child(_advance_group(state, stack, leaf_type), leaf_type)
        return replace(state, phase=phase, stack=new_stack)
    if token == TEXT_BEGIN:
        return replace(state, phase=phase, stack=_replace_top(stack, text_state=_TEXT_OPEN))
    if token == TEXT_END:
        return replace(state, phase=phase, stack=_replace_top(stack, text_state=_TEXT_DONE))
    if token in _CLOSE_BY_TOKEN:
        closed_type = _CLOSE_BY_TOKEN[token]
        new_stack = stack[:-1]
        if closed_type == "document":
            if phase in _FRAGMENT_PHASES:
                # END_DOCUMENT is never legal in fragment mode: once the
                # resumed document frame genuinely ends in this window the
                # only terminal is END_FRAGMENT (empty-stack suspend phase).
                return replace(state, phase=_PHASE_FRAGMENT_SUSPEND, stack=new_stack)
            return replace(state, phase=_PHASE_DOCUMENT_CLOSED, stack=new_stack)
        new_stack = _record_child(new_stack, closed_type)
        return replace(state, phase=phase, stack=new_stack)
    if token in _SUSPEND_BY_TOKEN:
        return replace(state, phase=_PHASE_FRAGMENT_SUSPEND, stack=stack[:-1])
    raise AssertionError(f"unhandled allowed token {token!r}")  # pragma: no cover


def _advance_group(
    state: GrammarState, stack: tuple[_Frame, ...], child_type: str
) -> tuple[_Frame, ...]:
    """Move the top frame to the group `child_type` belongs to.

    Applied when the child OPENS, so a header can never follow body
    content even though the body child is still on the stack.
    """
    if not stack:
        return stack  # fragment root: nothing to order against
    top = stack[-1]
    groups = _child_groups(state.profile).get(top.node_type)
    if not groups or len(groups) == 1:
        return stack
    for index in range(top.group, len(groups)):
        if child_type in groups[index]:
            return stack if index == top.group else _replace_top(stack, group=index)
    return stack


def _record_child(stack: tuple[_Frame, ...], child_type: str) -> tuple[_Frame, ...]:
    if not stack:
        return stack  # fragment-root child: nothing to record on
    is_paragraph = child_type == "paragraph"
    return _replace_top(
        stack,
        child_count=stack[-1].child_count + 1,
        has_paragraph_child=stack[-1].has_paragraph_child or is_paragraph,
        last_child_is_paragraph=is_paragraph,
    )


def _replace_top(stack: tuple[_Frame, ...], **changes) -> tuple[_Frame, ...]:
    return stack[:-1] + (replace(stack[-1], **changes),)


__all__ = [
    "GrammarState",
    "GrammarViolation",
    "PROFILES",
    "allowed_events",
    "child_types",
    "fresh",
    "step",
]


# Alphabet-closure sanity: every token the grammar can ever emit into a
# mask is drawn from the maps above plus the fixed control tokens, all of
# which are members of TOKENS.
assert (
    set(_OPEN_BY_TOKEN)
    | set(_CLOSE_BY_TOKEN)
    | set(_LEAF_BY_TOKEN)
    | set(_RESUME_BY_TOKEN)
    | set(_SUSPEND_BY_TOKEN)
    | {TEXT_BEGIN, TEXT_END, END_DOCUMENT, FRAGMENT_BEGIN, END_FRAGMENT}
) <= TOKENS

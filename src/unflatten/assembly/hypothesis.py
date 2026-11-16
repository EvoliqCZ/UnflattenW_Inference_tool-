"""One page as the assembler sees it (assembler design §4).

`from_token_ids` is the ONE adapter shared by gold and model: it parses a
standalone page's packed composite token ids into a page-local tree, so
anything gold-only has to be actively added to leak, never forgotten to
be removed. Node ids, origins, loss units and every other dataset locator
are absent by construction -- the only id a `PageNode` carries is the
tracker's page-local `n<k>`.

Rules never see a `PageNode`. They see a `RuleView` (`rule_view`), which
drops the typed pointers in `HIDDEN_FROM_RULES` and exposes text only as
its first/last non-whitespace scalar (design §4.3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from unflatten.model import composite_vocab as cv
from unflatten.model import events_vocab, piece_withdrawal
from unflatten.model import exact_text as et
from unflatten.model.node_lifecycle import (
    RESUME_OR_SUSPEND_TOKEN_NAMES,
    NodeLifecycleError,
    NodeLifecycleTracker,
    NodeState,
)
from unflatten.model.property_head_registry import property_head_registry

if TYPE_CHECKING:
    from unflatten.model.prediction_ir import RawDocumentHypothesis

type TypedValue = int | str | bool | None

_TOKEN_NAME_BY_ID: dict[int, str] = {i: name for name, i in events_vocab.TOKEN_IDS.items()}
_CLOSE_TOKEN_NAMES = frozenset(events_vocab.CLOSE_TOKEN.values())
_NODE_TOKEN_NAMES = frozenset(events_vocab.OPEN_TOKEN.values()) | frozenset(
    events_vocab.LEAF_TOKEN.values()
)
_NOTE_NODE_TYPES = frozenset({"footnote", "endnote"})
_PIECE_KINDS = ("whole", "first", "middle", "last")

#: Pointers that carry information from ANOTHER page (design §0.4/§0.5):
#: a story's gold kind encodes the page sequence; a section's break type
#: and a cell's row span describe what a continuation piece cannot show.
_CROSS_PAGE_POINTERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("header", "/kind"),
        ("footer", "/kind"),
        ("section", "/break_type"),
        ("cell", "/row_span"),
    }
)


class PageStreamInvalid(ValueError):
    """A page token stream is not one well-formed standalone page: broken
    envelope, unbalanced or mismatched nodes, a resume/suspend claim, text
    outside a run, or a typed anchor that names no node token."""


@dataclass(frozen=True, slots=True)
class PageInput:
    """The only input signals a rule may read besides the page's own
    hypothesis (design §4.2): page order and rendered pixel size."""

    page_ordinal: int
    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        for name, minimum in (("page_ordinal", 0), ("width_px", 1), ("height_px", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"PageInput.{name} must be an int >= {minimum}, got {value!r}")


@dataclass(frozen=True, slots=True)
class PageNode:
    local_id: str  # tracker "n<k>", page-local
    node_type: str
    children: tuple[PageNode, ...]
    text: str | None  # run only: its decoded TEXT_BEGIN..TEXT_END spans
    typed: Mapping[str, TypedValue]  # pointer -> value; absent = unknown


@dataclass(frozen=True, slots=True)
class PageHypothesis:
    input: PageInput
    sections: tuple[PageNode, ...]
    notes: tuple[PageNode, ...]


# ---------------------------------------------------------------------------
# Token ids -> page tree
# ---------------------------------------------------------------------------


def from_token_ids(
    token_ids: Sequence[int],
    *,
    vocabulary: cv.CompositeVocabulary,
    tokenizer: et.QwenTokenizerLike,
    typed_by_anchor: Mapping[int, Mapping[str, TypedValue]],
    page_input: PageInput,
) -> PageHypothesis:
    """Parse one standalone page (`FRAGMENT_BEGIN ... END_FRAGMENT`) into a
    page-local tree, replaying its structural tokens through
    `NodeLifecycleTracker` with `event_index = token position`.
    `typed_by_anchor[p]` attaches to the node whose open/leaf token sits at
    position `p`; an anchor on any other position is refused, never
    dropped."""
    refs = [vocabulary.unpack(int(token_id)) for token_id in token_ids]
    names = [
        _TOKEN_NAME_BY_ID[ref.local_id] if isinstance(ref, cv.StructuralRef) else None
        for ref in refs
    ]
    _check_envelope(names)
    tracker = NodeLifecycleTracker()
    open_nodes: list[NodeState] = []
    texts: dict[str, list[str]] = {}
    i = 1
    while i < len(refs) - 1:
        name = names[i]
        if name is None:
            raise PageStreamInvalid(f"text token outside TEXT_BEGIN/TEXT_END at position {i}")
        if name == events_vocab.TEXT_BEGIN:
            if not open_nodes or open_nodes[-1].node_type != "run":
                raise PageStreamInvalid(f"TEXT_BEGIN at position {i} is outside a run")
            end = _text_end(names, i)
            span = et.decode_tokens(
                tuple(refs[i + 1 : end]), vocabulary=vocabulary, tokenizer=tokenizer
            )
            texts.setdefault(open_nodes[-1].local_node_id, []).append(span)
            i = end + 1
            continue
        _apply_structural(tracker, open_nodes, name, i)
        i += 1
    if open_nodes:
        unclosed = [(state.local_node_id, state.node_type) for state in open_nodes]
        raise PageStreamInvalid(f"unclosed nodes at the end of the page: {unclosed}")
    return _build_page(tracker.states, texts, typed_by_anchor, page_input)


def from_model_hypothesis(
    raw: RawDocumentHypothesis,
    *,
    vocabulary: cv.CompositeVocabulary,
    tokenizer: et.QwenTokenizerLike,
    page_input: PageInput,
) -> PageHypothesis:
    """The model side of `from_token_ids` (AST Builder design §3): a
    terminated free-running decode, its typed predictions keyed by the
    token position of the node they were read from. The decode tracker's
    node ids are not the parser's (it also sees the envelope and text
    delimiters), so the join is by position, never by id."""
    if not raw.terminated:
        raise PageStreamInvalid("only a terminated decode is a page hypothesis")
    node_by_id = {state.local_node_id: state for state in raw.node_states}
    typed_by_anchor: dict[int, dict[str, TypedValue]] = {}
    for prediction in raw.typed_predictions:
        state = node_by_id.get(prediction.local_node_id)
        if state is None:
            raise PageStreamInvalid(
                f"typed prediction names unknown decode node {prediction.local_node_id!r}"
            )
        if state.node_type != prediction.node_type:
            raise PageStreamInvalid(
                f"typed prediction for {prediction.node_type!r} names "
                f"{state.local_node_id} of type {state.node_type!r}"
            )
        position = (
            state.open_event_index if state.slot_kind == "opened_node" else state.leaf_event_index
        )
        slot = typed_by_anchor.setdefault(position, {})
        if prediction.pointer in slot:
            raise PageStreamInvalid(
                f"{state.local_node_id}: two typed predictions for {prediction.pointer}"
            )
        slot[prediction.pointer] = prediction.value
    return from_token_ids(
        raw.accepted_tokens,
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        typed_by_anchor=typed_by_anchor,
        page_input=page_input,
    )


def _check_envelope(names: Sequence[str | None]) -> None:
    if not names or names[0] != events_vocab.FRAGMENT_BEGIN:
        raise PageStreamInvalid("a page must start with FRAGMENT_BEGIN")
    if len(names) < 2 or names[-1] != events_vocab.END_FRAGMENT:
        raise PageStreamInvalid("a page must end with END_FRAGMENT")
    for position, name in enumerate(names[1:-1], start=1):
        if name in (events_vocab.FRAGMENT_BEGIN, events_vocab.END_FRAGMENT):
            raise PageStreamInvalid(f"{name} at position {position} inside the page")


def _text_end(names: Sequence[str | None], begin: int) -> int:
    for position in range(begin + 1, len(names)):
        if names[position] == events_vocab.TEXT_END:
            return position
        if names[position] is not None:
            break
    raise PageStreamInvalid(f"TEXT_BEGIN at position {begin} has no matching TEXT_END")


def _apply_structural(
    tracker: NodeLifecycleTracker, open_nodes: list[NodeState], name: str, position: int
) -> None:
    if name in _CLOSE_TOKEN_NAMES:
        if not open_nodes:
            raise PageStreamInvalid(f"{name} at position {position} with nothing open")
        if events_vocab.CLOSE_TOKEN[open_nodes[-1].node_type] != name:
            raise PageStreamInvalid(
                f"{name} at position {position} does not close open "
                f"{open_nodes[-1].node_type!r}"
            )
        open_nodes.pop()
    elif name in RESUME_OR_SUSPEND_TOKEN_NAMES:
        raise PageStreamInvalid(f"{name} at position {position}: a standalone page never resumes")
    elif name not in _NODE_TOKEN_NAMES:
        raise PageStreamInvalid(f"{name} at position {position} is not allowed inside a page")
    try:
        state = tracker.apply_structural_token(name, event_index=position)
    except NodeLifecycleError as exc:
        raise PageStreamInvalid(f"{name} at position {position}: {exc}") from exc
    if state is not None and state.slot_kind == "opened_node" and state.close_event_index is None:
        open_nodes.append(state)


def _build_page(
    states: Sequence[NodeState],
    texts: Mapping[str, Sequence[str]],
    typed_by_anchor: Mapping[int, Mapping[str, TypedValue]],
    page_input: PageInput,
) -> PageHypothesis:
    id_by_position = {
        state.open_event_index if state.slot_kind == "opened_node" else state.leaf_event_index:
        state.local_node_id
        for state in states
    }
    stray = sorted(set(typed_by_anchor) - set(id_by_position))
    if stray:
        raise PageStreamInvalid(f"typed anchor positions {stray} name no node open/leaf token")
    typed_by_id = {id_by_position[p]: typed for p, typed in typed_by_anchor.items()}
    children_of: dict[str | None, list[NodeState]] = {}
    for state in states:
        children_of.setdefault(state.parent_local_node_id, []).append(state)

    def build(state: NodeState) -> PageNode:
        return PageNode(
            local_id=state.local_node_id,
            node_type=state.node_type,
            children=tuple(build(child) for child in children_of.get(state.local_node_id, ())),
            text="".join(texts.get(state.local_node_id, ())) if state.node_type == "run" else None,
            typed=MappingProxyType(dict(typed_by_id.get(state.local_node_id, {}))),
        )

    sections: list[PageNode] = []
    notes: list[PageNode] = []
    for state in children_of.get(None, ()):
        if state.node_type == "section":
            sections.append(build(state))
        elif state.node_type in _NOTE_NODE_TYPES:
            notes.append(build(state))
        else:
            raise PageStreamInvalid(
                f"top-level {state.node_type!r} node {state.local_node_id}: a page holds "
                "only sections and notes"
            )
    return PageHypothesis(input=page_input, sections=tuple(sections), notes=tuple(notes))


# ---------------------------------------------------------------------------
# What rules may see (design §4.3)
# ---------------------------------------------------------------------------


def derive_hidden_from_rules() -> frozenset[tuple[str, str]]:
    """`(node_type, pointer)` pairs no rule may read: every DEC-033 §10
    withdrawal, from its one source, over every registry node type and
    piece kind, plus the cross-page pointers. Reads `piece_withdrawal` at
    call time, so a changed list is seen without a re-import."""
    node_types = {node_type for node_type, _ in property_head_registry()}
    withdrawn = {
        (node_type, pointer)
        for node_type in node_types
        for kind in _PIECE_KINDS
        for pointer in piece_withdrawal.withdrawn_pointers_for_piece(node_type, kind)
    }
    return frozenset(withdrawn) | _CROSS_PAGE_POINTERS


HIDDEN_FROM_RULES: frozenset[tuple[str, str]] = derive_hidden_from_rules()


def _run_text(node: PageNode) -> str:
    return (node.text or "") + "".join(_run_text(child) for child in node.children)


def first_visible(node: PageNode) -> str | None:
    """First scalar of the node's concatenated run text that is not Unicode
    whitespace; `None` when there is none."""
    return next((scalar for scalar in _run_text(node) if not scalar.isspace()), None)


def last_visible(node: PageNode) -> str | None:
    """Last scalar of the node's concatenated run text that is not Unicode
    whitespace; `None` when there is none."""
    return next((scalar for scalar in reversed(_run_text(node)) if not scalar.isspace()), None)


@dataclass(frozen=True, slots=True)
class RuleView:
    """A node as a join rule sees it: no id, no raw text, no edge
    whitespace, no hidden pointer."""

    node_type: str
    children: tuple[RuleView, ...]
    typed: Mapping[str, TypedValue]
    first_visible: str | None
    last_visible: str | None


def rule_view(node: PageNode) -> RuleView:
    return RuleView(
        node_type=node.node_type,
        children=tuple(rule_view(child) for child in node.children),
        typed=MappingProxyType(
            {
                pointer: value
                for pointer, value in node.typed.items()
                if (node.node_type, pointer) not in HIDDEN_FROM_RULES
            }
        ),
        first_visible=first_visible(node),
        last_visible=last_visible(node),
    )

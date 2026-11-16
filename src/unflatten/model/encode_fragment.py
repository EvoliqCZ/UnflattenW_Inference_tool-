"""Page-local token sequences from a RFC003 §26 Page Fragment.

`encode.encode_structural` encodes a whole document. A training sample is
one *page*, and using the document stream as its target asks page 1 to
predict what is on page 7 — unlearnable for anything multi-page, which at
5.84 pages per document is most of the corpus. This module encodes what
the page actually contains.

The RFC004 §18.2 alphabet already anticipates this: `FRAGMENT_BEGIN` /
`END_FRAGMENT` wrap the page, and `RESUME_NODE(type)` / `SUSPEND_NODE(type)`
claim containers that cross the page boundary. A paragraph split across
two pages is resumed on the second and suspended on the first, so neither
page has to invent a structure it cannot see.

The fragment tells us which is which: a slice's `continuation` is
`complete` when the node lives entirely on this page, and `middle`,
`start` or `end` when it crosses a boundary. `structural_event_ranges`
gives the exact span of the document event stream the page owns.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from unflatten.model.events_vocab import (
    CLOSE_TOKEN,
    END_FRAGMENT,
    FRAGMENT_BEGIN,
    LEAF_TOKEN,
    OPEN_TOKEN,
    RESUME_TOKEN,
    SUSPEND_TOKEN,
    TEXT_BEGIN,
    TEXT_END,
)

#: A node with one of these continuations began before this page.
_STARTED_EARLIER = frozenset({"middle", "end"})

#: A node with one of these continuations goes on after this page.
_CONTINUES_AFTER = frozenset({"middle", "start"})


def _owned_event_indices(fragment: dict[str, Any]) -> set[int]:
    indices: set[int] = set()
    for entry in fragment.get("slices", ()):
        for span in entry.get("structural_event_ranges", ()):
            indices.update(range(int(span["start"]), int(span["end"])))
    return indices


def _presented_text_indices(
    fragment: dict[str, Any], events: list[dict[str, Any]], owned: set[int]
) -> set[int]:
    """Text events this page presents by grant, without owning their structure.

    design §4.1: a page can hold a `char_ranges` grant for a node whose
    `structural_event_ranges` belong to another page (`structure_owner` is
    the *first* intersecting window, `char_ranges` are per page-owner). That
    disagreement is what leaves a granted continuation with no text slot.
    This is additive only: `_owned_event_indices` above is untouched, and
    every event it already counts stays counted the same way.
    """
    granted_nodes = {
        entry["node_id"] for entry in fragment.get("slices", ()) if entry.get("char_ranges")
    }
    return {
        event["index"]
        for event in events
        if event["op"] == "text"
        and event.get("node_id") in granted_nodes
        and event["index"] not in owned
    }


def encode_fragment(
    fragment: dict[str, Any], structural: dict[str, Any]
) -> list[str]:
    """Token sequence for the page `fragment` covers."""
    return encode_fragment_with_origins(fragment, structural)[0]


def encode_fragment_with_origins(
    fragment: dict[str, Any], structural: dict[str, Any]
) -> tuple[list[str], list[int | None]]:
    """The token sequence, plus the source event index of each token.

    A token's origin is the document event that caused it to be emitted,
    or `None` for the envelope's own protocol tokens (`FRAGMENT_BEGIN`,
    the resume chain, the trailing suspends, `END_FRAGMENT`) — those
    describe the window, not any node in it.

    Two attribution rules, and both directions matter:

    - a container OPENED by `reconcile` is credited to the owned event
      that needed it, because the frame exists only as a consequence of
      placing that event. A consumer withdrawing supervision for a node
      therefore withdraws the structure it implied — the safe direction;
    - a frame CLOSED by `reconcile` is credited to the frame's own
      origin, not to the later event that triggered the close. §18.4
      requires a node opened inside the window to close inside it, so
      that close is emitted while placing some *other* node; crediting
      it to the trigger would leave the close of an unplaceable
      paragraph scored while its open was withdrawn.

    Follows the §18.4 fragment protocol exactly, because the grammar —
    not this encoder — defines it:

    - a fragment is a window *into* a document, so the document frame is
      always RESUMED and never opened (`DOCUMENT_OPEN` is not a legal
      fragment-root child);
    - resumes form one contiguous chain, outer to inner, before any
      content;
    - a node opened inside the window must CLOSE inside it, even when the
      real node continues onto the next page — that page resumes it;
    - only resumed frames may be SUSPENDED, inner to outer, and the
      residual stack must be empty before `END_FRAGMENT`.

    A third case, added by design §4.1 (DEC-MP-1/DEC-MP-5): a page can
    hold a `char_ranges` text grant for a node whose `op == "text"` event
    this page does not own. That event is *presented* here even though it
    is not *owned* — `presentation = owned | presented_text` — so its
    `TEXT_BEGIN`/`TEXT_END` are emitted with the event's own true origin
    (the delimiter provenance DEC-MP-1 decides), and the bootstrap prewalk
    starts from `min(presentation)`, not `min(owned)`, so a continuation
    preceding everything this page owns still resumes its ancestor chain.
    Both boundary geometries fall out of the existing machinery unchanged:
    if nothing owned follows, the resumed chain is SUSPENDED at the end;
    if owned content follows under a different local structure, the next
    `reconcile()` closes it locally with `origin=None` — the same
    frame-closed-by-reconcile rule above, now reachable from a presented,
    not owned, event too.
    """
    owned = _owned_event_indices(fragment)
    events = list(structural.get("events", ()))
    presentation = owned | _presented_text_indices(fragment, events, owned)

    node_type_by_open: dict[int, str] = {}
    leaf_opens: set[int] = set()
    for event in events:
        if event["op"] == "open":
            node_type_by_open[event["index"]] = event["node_type"]
            if event["node_type"] in LEAF_TOKEN:
                leaf_opens.add(event["index"])

    # Containers open when the page begins: their open event is outside
    # this window but they enclose something inside it.
    depth: list[tuple[int, str]] = []
    # NAVRH §4 / design §4.1: the bootstrap prewalk must start from the
    # first LOCAL PRESENTATION, not the first owned event, or a page that
    # presents (but does not own) a continuation preceding everything it
    # owns would resume nothing for it.
    first_local = min(presentation) if presentation else len(events) + 1
    for event in events:
        if event["index"] >= first_local:
            break
        if event["op"] == "open" and event["index"] not in leaf_opens:
            depth.append((event["index"], event["node_type"]))
        elif event["op"] == "close" and depth and depth[-1][0] == event["open_index"]:
            depth.pop()
    open_before_pairs: list[tuple[int | None, str]] = list(depth)
    if not open_before_pairs or open_before_pairs[0][1] != "document":
        # a window always sits inside a document, even when its first
        # owned event is the document open itself
        open_before_pairs = [
            (None, "document"),
            *[(i, t) for i, t in open_before_pairs if t != "document"],
        ]

    tokens: list[str] = []
    origins: list[int | None] = []
    #: Event whose emission we are in the middle of; None outside one.
    current: list[int | None] = [None]

    def emit(*names: str) -> None:
        tokens.extend(names)
        origins.extend([current[0]] * len(names))

    emit(FRAGMENT_BEGIN)
    resumed: list[str] = []
    resumed_identity: list[int | None] = []
    for identity, node_type in open_before_pairs:
        if node_type in RESUME_TOKEN:
            emit(RESUME_TOKEN[node_type])
            resumed.append(node_type)
            resumed_identity.append(identity)

    # Content. Owned events are not always contiguous — a page can own a
    # run deep inside a list and a later top-level paragraph — and across
    # such a gap the true stack changes. Walking every event keeps the real
    # stack; each owned event is emitted only after the emitted stack has
    # been reconciled to match it.
    #
    # The emitted stack carries a resumed flag per frame: a resumed node
    # MAY close normally (§18.4), and one that does must not also be
    # suspended at the end.
    resumed_open_indices = {index for index, _ in depth}
    #: (node_type, was_resumed, origin event of the frame, node identity --
    #: the frame's own true open-event index, or `None` for the document)
    stack: list[tuple[str, bool, int | None, int | None]] = [
        (t, True, None, identity) for t, identity in zip(resumed, resumed_identity)
    ]
    true_stack: list[tuple[int, str]] = []

    def close_top() -> None:
        node_type, _resumed, origin, _identity = stack.pop()
        previous = current[0]
        current[0] = origin
        emit(CLOSE_TOKEN[node_type])
        current[0] = previous

    def reconcile() -> None:
        # The document frame is the window's permanent base: it is resumed
        # at FRAGMENT_BEGIN and suspended at the end, never opened or closed
        # here, so reconciliation works strictly above it.
        #
        # Compared by (identity, type), not type alone: two structurally
        # distinct branches can share an ancestor type shape (a split table
        # row's two continuing cells are both `cell > paragraph > run`), and
        # a type-only comparison cannot tell them apart -- it silently
        # leaves the first branch's frame open across the boundary instead
        # of closing it and opening the second (NALEZ-reconcile-zlieva-podla-typu).
        want = [(i, t) for i, t in true_stack if t in OPEN_TOKEN and t != "document"]
        base = 1
        common = 0
        while (
            common < len(want)
            and base + common < len(stack)
            and want[common] == (stack[base + common][3], stack[base + common][0])
        ):
            common += 1
        while len(stack) > base + common:
            close_top()
        for identity, node_type in want[common:]:
            emit(OPEN_TOKEN[node_type])
            stack.append((node_type, False, current[0], identity))

    for event in events:
        index = event["index"]
        op = event["op"]
        owned_here = index in owned
        presented_here = index in presentation
        current[0] = index if presented_here else None

        if presented_here and op in ("open", "text"):
            reconcile()

        # A presented-not-owned event is always `op == "text"` (it comes
        # only from `_presented_text_indices`, which filters on that), so
        # `owned_here` still drives `open`/`close` unchanged, and this
        # branch is the only new one.
        if presented_here and op == "text":
            emit(TEXT_BEGIN, TEXT_END)
        elif owned_here:
            if op == "open":
                node_type = event["node_type"]
                if index in resumed_open_indices or node_type == "document":
                    pass
                elif node_type in LEAF_TOKEN:
                    emit(LEAF_TOKEN[node_type])
                elif node_type in OPEN_TOKEN:
                    emit(OPEN_TOKEN[node_type])
                    stack.append((node_type, False, index, index))
                else:
                    raise ValueError(
                        f"node type {node_type!r} has no token in the V0 event "
                        f"alphabet (event index {index})"
                    )
            elif op == "close":
                open_index = event["open_index"]
                node_type = node_type_by_open.get(open_index)
                if (
                    open_index not in leaf_opens
                    and node_type is not None
                    and node_type != "document"
                    and len(stack) > 1
                    and stack[-1][0] == node_type
                ):
                    close_top()
            else:
                raise ValueError(f"unknown structural op {op!r}")

        if op == "open" and index not in leaf_opens:
            true_stack.append((index, event["node_type"]))
        elif op == "close" and true_stack and true_stack[-1][0] == event["open_index"]:
            true_stack.pop()

    # Non-resumed frames close; resumed ones suspend, inner to outer.
    # Past the last owned event these are envelope protocol, not content.
    current[0] = None
    while len(stack) > 1 and not stack[-1][1]:
        close_top()
    while stack:
        node_type, was_resumed, origin, _identity = stack.pop()
        current[0] = origin
        emit(SUSPEND_TOKEN[node_type] if was_resumed else CLOSE_TOKEN[node_type])
    current[0] = None
    emit(END_FRAGMENT)
    return tokens, origins


def encode_standalone_fragment_with_origins(
    fragment: dict[str, Any],
    structural: dict[str, Any],
    *,
    dropped_node_ids: frozenset[str] = frozenset(),
    rehome: Mapping[str, str] | None = None,
) -> tuple[list[str], list[int | None]]:
    """The DEC-033 filtered-preorder-walk encoding for `s2-standalone-page-v1`.

    RFC003-ERRATA-002 / RFC004-ERRATA-002 (design §3): under this profile
    a page is a closed document of its own, never a window into a larger
    stream, so `RESUME_NODE`/`SUSPEND_NODE` are never emitted — unlike
    :func:`encode_fragment_with_origins`, which this function does not
    call or modify (that path stays exactly as it is for every other
    profile, `s1-*`/`s2-typed-widened-v1` included).

    `present = {slice.node_id for slice in fragment.slices}` is
    ancestor-closed (the descendant-union page rule, `envelope.py:27-29`,
    always leaves `ancestor_context` empty), so filtering
    `structural.events` to `event.node_id in present` is enough on its
    own: a well-formed nested bracket sequence filtered by an
    ancestor-closed node subset is still well-formed, in the same
    canonical preorder (header -> footer -> body, `grammar.py:44-52`) the
    unfiltered stream already carries. No stack, no reconciliation.

    The `document` node is the one exception: its open/close map to the
    envelope's own `FRAGMENT_BEGIN`/`END_FRAGMENT` (origin `None`), not to
    a `DOCUMENT_OPEN`/`DOCUMENT_CLOSE` token — legal today at the fragment
    root (`grammar.py:352-355`, `top is None` admits `SECTION_OPEN`
    directly). Every other present node's open emits `OPEN_TOKEN`/
    `LEAF_TOKEN`, its text emits `TEXT_BEGIN TEXT_END`, its close emits
    `CLOSE_TOKEN` — each with the event's own true index as origin, so a
    node present on N pages is scored on N pages, not once
    (RFC003-ERRATA-002 E2: not a duplicate, each page is a different
    input).

    `dropped_node_ids` is the DEC-033 §5/§6 page-normalization projection
    (`page_projection.project`), computed by the caller BEFORE this
    function runs and removed from `present` here -- so a dropped node's
    tokens are simply never emitted (no `TEXT_BEGIN`, no loss unit),
    never masked out after the fact.

    ``rehome`` (`v16-targets.v1` V1d, `v16_target_policy.rehome_inherited_stories`):
    header/footer story id -> the section it is emitted in on this page --
    after that section's own headers (a header) or own footers (a footer),
    before its body, each event keeping its own origin. The caller drops
    the story-only section it came from (``dropped_node_ids``). Empty or
    ``None``: the stream is exactly the unrehomed one.
    """
    present = {
        entry["node_id"] for entry in fragment.get("slices", ())
    } - dropped_node_ids
    events = list(structural.get("events", ()))
    if rehome:
        events = _rehomed_events(events, rehome)

    node_type_by_open: dict[int, str] = {}
    leaf_opens: set[int] = set()
    for event in events:
        if event["op"] == "open":
            node_type_by_open[event["index"]] = event["node_type"]
            if event["node_type"] in LEAF_TOKEN:
                leaf_opens.add(event["index"])

    tokens: list[str] = [FRAGMENT_BEGIN]
    origins: list[int | None] = [None]

    for event in events:
        if event.get("node_id") not in present:
            continue
        op = event["op"]
        index = event["index"]
        if op == "open":
            node_type = event["node_type"]
            if node_type == "document":
                continue
            if index in leaf_opens:
                tokens.append(LEAF_TOKEN[node_type])
                origins.append(index)
            elif node_type in OPEN_TOKEN:
                tokens.append(OPEN_TOKEN[node_type])
                origins.append(index)
            else:
                raise ValueError(
                    f"node type {node_type!r} has no token in the V0 event "
                    f"alphabet (event index {index})"
                )
        elif op == "text":
            tokens.extend((TEXT_BEGIN, TEXT_END))
            origins.extend((index, index))
        elif op == "close":
            open_index = event["open_index"]
            if open_index in leaf_opens:
                continue
            node_type = node_type_by_open.get(open_index)
            if node_type == "document":
                continue
            tokens.append(CLOSE_TOKEN[node_type])
            origins.append(index)
        else:
            raise ValueError(f"unknown structural op {op!r}")

    tokens.append(END_FRAGMENT)
    origins.append(None)
    return tokens, origins


def _rehomed_events(
    events: list[dict[str, Any]], rehome: Mapping[str, str]
) -> list[dict[str, Any]]:
    """`events` with every story in ``rehome`` cut out of its own section
    and spliced into its target section: headers after the target's own
    headers, footers after its own footers (header -> footer -> body, the
    canonical preorder `grammar.py` admits)."""
    position = {event["index"]: p for p, event in enumerate(events)}
    close_of: dict[int, int] = {}
    open_of: dict[str, int] = {}
    for p, event in enumerate(events):
        if event["op"] == "close":
            close_of[position[event["open_index"]]] = p
        elif event["op"] == "open":
            open_of[event["node_id"]] = p
    missing = sorted((set(rehome) | set(rehome.values())) - set(open_of))
    if missing:
        raise ValueError(f"rehome names node(s) {missing} the structural events do not open")
    moved: dict[str, list[list[dict[str, Any]]]] = {}
    for story, section in rehome.items():
        start = open_of[story]
        moved.setdefault(section, []).append(events[start : close_of[start] + 1])
    for spans in moved.values():
        spans.sort(key=lambda span: span[0]["index"])

    out: list[dict[str, Any]] = []
    p = 0
    while p < len(events):
        event = events[p]
        if event["op"] == "open" and event["node_id"] in rehome:
            p = close_of[p] + 1
            continue
        if event["op"] == "open" and event["node_id"] in moved:
            end = close_of[p]
            headers: list[list[dict[str, Any]]] = []
            footers: list[list[dict[str, Any]]] = []
            rest: list[list[dict[str, Any]]] = []
            q = p + 1
            while q < end:
                child_end = close_of.get(q, q)
                child = events[q : child_end + 1]
                kind = child[0].get("node_type")
                bucket = headers if kind == "header" else footers if kind == "footer" else rest
                bucket.append(child)
                q = child_end + 1
            incoming = moved[event["node_id"]]
            headers += [span for span in incoming if span[0]["node_type"] == "header"]
            footers += [span for span in incoming if span[0]["node_type"] == "footer"]
            out.append(event)
            for span in (*headers, *footers, *rest):
                out.extend(span)
            out.append(events[end])
            p = end + 1
            continue
        out.append(event)
        p += 1
    return out

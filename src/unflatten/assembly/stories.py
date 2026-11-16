"""Header/footer story kinds from the page sequence (assembler design §5.8).

Per assembled section with pages p1..pn and per story type, H(p) is the
page's story content: the canonical serialization of its story subtrees,
typed values included, `/kind` excluded; a page without one is `∅`. The
kinds follow from the sequence alone:

1. all H(p) equal: one `default`;
2. else H(p1) differs and H(p2..pn) are equal: `first` + `default`;
3. else the stories alternate by physical page parity, after an optional
   first-page exception: `default` = odd pages, `even` = even pages
   (+ `first`);
4. otherwise `STORY_UNRESOLVED`: `default` = H(p1).

A `∅` assignment produces no story, never an empty one. The predicted
`/kind` is never read for the derivation; it is only counted against the
derived kind (`kind_predicted_agrees`). Digit runs in story text are
normalized for the equality test only, never in the stored content.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from unflatten.ast.canonical_json import to_canonical_json

if TYPE_CHECKING:
    from unflatten.assembly.assemble import AssembledNode, PageStory

STORY_UNRESOLVED = "STORY_UNRESOLVED"

_STORY_TYPES = ("header", "footer")
_KIND_POINTER = "/kind"
_KIND_ORDER = ("default", "first", "even")
_DIGIT_RUN = re.compile(r"\d+")
#: XML 1.0 text can never contain NUL, so the placeholder cannot collide.
_DIGIT_PLACEHOLDER = "\x00"


@dataclass(frozen=True, slots=True)
class DerivedStory:
    """One section story: `kind`, and the content page `source_page`
    showed, as its unmodified lifted story nodes."""

    kind: str
    source_page: int
    nodes: tuple[AssembledNode, ...]


@dataclass(frozen=True, slots=True)
class StorySet:
    """The derived stories of one story type in one assembled section.
    `misrepresented_pages` are the pages whose resolved story
    (`resolve_page`) is not the content the page showed."""

    section_id: str
    story_type: str
    pages: tuple[int, ...]
    pattern: str  # all_equal | first_different | odd_even | even_first | unresolved
    stories: tuple[DerivedStory, ...]
    misrepresented_pages: tuple[int, ...]
    story_digit_runs_normalized: int
    kind_predicted_known: int
    kind_predicted_agrees: int
    issue: str | None  # STORY_UNRESOLVED or None


def derive_story_sets(
    sections: Sequence[AssembledNode], stories: Sequence[PageStory],
    *, resource_keys: Mapping[str, str] | None = None,
) -> tuple[StorySet, ...]:
    """One `StorySet` per assembled section and story type, in section
    order, header before footer."""
    nodes_by: dict[tuple[str, str, int], list[AssembledNode]] = {}
    for story in stories:
        key = (story.section_id, story.node.node_type, story.page_ordinal)
        nodes_by.setdefault(key, []).append(story.node)
    return tuple(
        _derive(
            section.assembled_id,
            story_type,
            tuple(sorted({ordinal for ordinal, _ in section.sources})),
            lambda p, s=section.assembled_id, t=story_type: tuple(nodes_by.get((s, t, p), ())),
            resource_keys=resource_keys,
        )
        for section in sections
        for story_type in _STORY_TYPES
    )


def resolve_page(story_set: StorySet, page_ordinal: int) -> DerivedStory | None:
    """The story the derived kinds show on `page_ordinal`: `first` on the
    section's first page, else `even` on an even physical page, else
    `default`; `None` when that kind has no story."""
    return _resolve(story_set.stories, story_set.pages[0], page_ordinal)


def _resolve(
    stories: Sequence[DerivedStory], first_page: int, page_ordinal: int
) -> DerivedStory | None:
    by_kind = {story.kind: story for story in stories}
    if page_ordinal == first_page and "first" in by_kind:
        return by_kind["first"]
    if _physical(page_ordinal) % 2 == 0 and "even" in by_kind:
        return by_kind["even"]
    return by_kind.get("default")


def _physical(page_ordinal: int) -> int:
    return page_ordinal + 1


def _derive(
    section_id: str,
    story_type: str,
    pages: tuple[int, ...],
    nodes_on: Callable[[int], tuple[AssembledNode, ...]],
    *, resource_keys: Mapping[str, str] | None = None,
) -> StorySet:
    content = {p: _content_key(nodes_on(p), resource_keys) for p in pages}
    pattern, sources = _kinds(pages, content)
    derived = tuple(
        DerivedStory(kind, sources[kind], nodes_on(sources[kind]))
        for kind in _KIND_ORDER
        if kind in sources and nodes_on(sources[kind])
    )
    resolved = {p: _resolve(derived, pages[0], p) for p in pages}
    shown = {
        p: content[story.source_page] if story else _content_key((), resource_keys)
        for p, story in resolved.items()
    }
    predicted = [
        (node.typed[_KIND_POINTER], resolved[p])
        for p in pages
        for node in nodes_on(p)
        if _KIND_POINTER in node.typed
    ]
    return StorySet(
        section_id=section_id,
        story_type=story_type,
        pages=pages,
        pattern=pattern,
        stories=derived,
        misrepresented_pages=tuple(p for p in pages if shown[p] != content[p]),
        story_digit_runs_normalized=sum(_digit_runs(node) for p in pages for node in nodes_on(p)),
        kind_predicted_known=len(predicted),
        kind_predicted_agrees=sum(
            story is not None and kind == story.kind for kind, story in predicted
        ),
        issue=STORY_UNRESOLVED if pattern == "unresolved" else None,
    )


def _kinds(pages: tuple[int, ...], content: Mapping[int, bytes]) -> tuple[str, dict[str, int]]:
    """The §5.8 pattern and, per kind, the page whose content it takes."""
    first, rest = pages[0], pages[1:]
    if _uniform(pages, content):
        return "all_equal", {"default": first}
    if _uniform(rest, content):
        return "first_different", {"first": first, "default": rest[0]}
    for pattern, span, extra in (("odd_even", pages, {}), ("even_first", rest, {"first": first})):
        odd = [p for p in span if _physical(p) % 2 == 1]
        even = [p for p in span if _physical(p) % 2 == 0]
        if odd and even and _uniform(odd, content) and _uniform(even, content):
            return pattern, {"default": odd[0], "even": even[0], **extra}
    return "unresolved", {"default": first}


def _uniform(pages: Sequence[int], content: Mapping[int, bytes]) -> bool:
    return len({content[p] for p in pages}) == 1


def _content_key(
    nodes: Sequence[AssembledNode], resource_keys: Mapping[str, str] | None = None,
) -> bytes:
    """H(p) for comparison: story subtrees without ids, sources and the
    story `/kind`, with every digit run in text replaced by one
    placeholder."""
    return to_canonical_json([_serialize(node, resource_keys) for node in nodes])


def _serialize(
    node: AssembledNode, resource_keys: Mapping[str, str] | None = None,
) -> dict[str, object]:
    typed = {
        pointer: value
        for pointer, value in node.typed.items()
        if not (node.node_type in _STORY_TYPES and pointer == _KIND_POINTER)
    }
    serialized = {
        "node_type": node.node_type,
        "text": None if node.text is None else _DIGIT_RUN.sub(_DIGIT_PLACEHOLDER, node.text),
        "typed": typed,
        "children": [_serialize(child, resource_keys) for child in node.children],
    }
    if resource_keys is not None and node.node_type in ("inline_image", "floating_image"):
        serialized["resource_key"] = resource_keys.get(
            node.assembled_id, f"unresolved:{node.assembled_id}",
        )
    return serialized


def _digit_runs(node: AssembledNode) -> int:
    own = len(_DIGIT_RUN.findall(node.text)) if node.text else 0
    return own + sum(_digit_runs(child) for child in node.children)

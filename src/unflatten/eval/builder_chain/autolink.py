"""Opt-in chain policy `autolink.v0` (PROPOSED, owner-approved 2026-09-27,
RFC004-ERRATA-009 draft).

The model has no `hyperlink` head (DEC-043: real training data already
flattens hyperlinks, so there is nothing to learn one from). A run whose
text spells out a URL or e-mail address is rendered as plain text, never
as a clickable link. This policy splits a run's matched span into a
`hyperlink` node (`target.kind`/`target.uri` per RFC001 `hyperlink`, wrapping
one child `run` that carries exactly the matched text and the ORIGINAL
run's style) and keeps the rest of the run's text as plain run(s), only
when the caller asks for it, and counts every link.

Scope: every `run` in the document (sections, notes, stories) except a
run that is already a descendant of a `hyperlink` node (never re-link
inside an existing hyperlink) -- there is no `field` case to exclude,
since RFC001's `field` node carries no children at all.

MATCH (closed, conservative regex, case-insensitive): a URL starts with
`http://`, `https://` or `www.` (never a bare domain) and runs to the
first whitespace/quote/angle-bracket; an e-mail is
`local@domain.tld`. Trailing punctuation (`.,;:!?)]}'"`) is trimmed off
the match one character at a time, so "see http://x.example." links only
"http://x.example" -- the sentence's full stop is never swallowed.
`target.uri` is the matched text, `http://`-prefixed when it started
`www.`; an e-mail's `target.uri` is `mailto:<matched text>`. The VISIBLE
text is never touched -- only structure changes, so the document's text
is byte-identical before and after linking.

Profile refusal: `hyperlink` is a `word-visual-v1`+ node type
(`run._V1_NODE_TYPES`, pinned equal by a test); on the model arm's fixed
`word-core-v0` base profile (`run.model_profile_for`, resolved from the
page's PREDICTED node types before this policy ever runs) inserting one
would build a `V1_FEATURE_IN_V0_PROFILE` refusal, so `apply` takes the
page's resolved `profile` and, on the base profile, counts
`skipped_profile` per match instead of linking it.

Idempotent: a linked run's matched span is now inside a `hyperlink`
node, which the walk never re-enters, so a second `apply` finds nothing
left to link on that span. `sources`: the original run's `sources` are
kept on the FIRST produced piece (literal or hyperlink), every other
piece carries `()` -- same approximation as `page_number_fields`.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from collections import Counter
from typing import Any

from unflatten.assembly.assemble import AssembledDocumentHypothesis, AssembledNode

POLICY_ID = "autolink.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-009.md"
COUNT_PREFIX = "autolink:"
#: Mirrors `run.MODEL_PROFILE` (pinned equal by a test).
BASE_PROFILE = "word-core-v0"

_TRAILING_PUNCT = ".,;:!?)]}'\""
_URL_EMAIL = re.compile(
    r"(?P<url>(?:https?://|www\.)[^\s<>\"']+)"
    r"|(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
    re.IGNORECASE,
)


def policy_record() -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "base_profile": BASE_PROFILE,
        "match": r"(?:https?://|www\.)\S+|[\w.+-]+@[\w.-]+\.\w{2,}",
        "trailing_punctuation_trimmed": _TRAILING_PUNCT,
    }


def apply(
    document: AssembledDocumentHypothesis, *, profile: str
) -> tuple[AssembledDocumentHypothesis, Counter[str]]:
    """`document` with every matched URL/e-mail span turned into a
    `hyperlink`, and one count per outcome. `profile` is the page's
    resolved build profile; on `BASE_PROFILE` nothing is linked and every
    match is counted `skipped_profile` instead."""
    counts: Counter[str] = Counter()
    allow = profile != BASE_PROFILE
    sections = tuple(_walk(node, allow, counts, False) for node in document.sections)
    notes = tuple(_walk(node, allow, counts, False) for node in document.notes)
    stories = tuple(
        dataclasses.replace(story, node=_walk(story.node, allow, counts, False))
        for story in document.stories
    )
    if (
        all(a is b for a, b in zip(sections, document.sections, strict=True))
        and all(a is b for a, b in zip(notes, document.notes, strict=True))
        and all(a.node is b.node for a, b in zip(stories, document.stories, strict=True))
    ):
        return document, counts
    return dataclasses.replace(document, sections=sections, notes=notes, stories=stories), counts


def _walk(
    node: AssembledNode, allow: bool, counts: Counter[str], in_hyperlink: bool
) -> AssembledNode:
    child_in_hyperlink = in_hyperlink or node.node_type == "hyperlink"
    children = _expand_children(node.children, allow, counts, child_in_hyperlink)
    if all(c is o for c, o in zip(children, node.children, strict=True)):
        return node
    return dataclasses.replace(node, children=children)


def _expand_children(
    children: tuple[AssembledNode, ...], allow: bool, counts: Counter[str], in_hyperlink: bool
) -> tuple[AssembledNode, ...]:
    out: list[AssembledNode] = []
    for child in children:
        if child.node_type == "run" and not in_hyperlink:
            out.extend(_linkify_run(child, allow, counts))
        else:
            out.append(_walk(child, allow, counts, in_hyperlink))
    return tuple(out)


def _trim_trailing(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1] in _TRAILING_PUNCT:
        end -= 1
    return end


def _find_matches(text: str) -> list[tuple[int, int, str, str]]:
    matches: list[tuple[int, int, str, str]] = []
    for found in _URL_EMAIL.finditer(text):
        if found.group("url"):
            start, end = found.span("url")
            end = _trim_trailing(text, start, end)
            if end <= start:
                continue
            raw = text[start:end]
            target = raw if raw.lower().startswith(("http://", "https://")) else f"http://{raw}"
            matches.append((start, end, "url", target))
        else:
            start, end = found.span("email")
            end = _trim_trailing(text, start, end)
            if end <= start:
                continue
            matches.append((start, end, "email", f"mailto:{text[start:end]}"))
    return matches


def _linkify_run(run: AssembledNode, allow: bool, counts: Counter[str]) -> list[AssembledNode]:
    text = run.text or ""
    matches = _find_matches(text)
    if not matches:
        return [run]
    if not allow:
        counts[COUNT_PREFIX + "skipped_profile"] += len(matches)
        return [run]
    counter = itertools.count()
    specs: list[tuple[str, int, int, str, str] | tuple[str, int, int]] = []
    pos = 0
    for start, end, kind, target in matches:
        if start > pos:
            specs.append(("literal", pos, start))
        specs.append(("link", start, end, kind, target))
        pos = end
    if pos < len(text):
        specs.append(("literal", pos, len(text)))
    out: list[AssembledNode] = []
    for i, spec in enumerate(specs):
        sources = run.sources if i == 0 else ()
        if spec[0] == "literal":
            _, start, end = spec
            out.append(dataclasses.replace(
                run,
                assembled_id=f"{run.assembled_id}.al{next(counter)}",
                text=text[start:end],
                sources=sources,
            ))
        else:
            _, start, end, kind, target = spec
            link_run = dataclasses.replace(
                run,
                assembled_id=f"{run.assembled_id}.al{next(counter)}",
                text=text[start:end],
                sources=(),
            )
            out.append(AssembledNode(
                assembled_id=f"{run.assembled_id}.al{next(counter)}",
                node_type="hyperlink",
                children=(link_run,),
                text=None,
                typed={
                    "/target/kind": "email" if kind == "email" else "external",
                    "/target/uri": target,
                },
                sources=sources,
            ))
            counts[COUNT_PREFIX + ("emails_linked" if kind == "email" else "urls_linked")] += 1
    return out

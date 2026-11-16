"""Opt-in chain policy `page-number-fields.v0` (PROPOSED, owner-approved
2026-09-27, RFC004-ERRATA-009 draft).

A page-number footer/header is emitted by the model as literal text (the
decoder has no `field`/`page_numbering` head): "Strana 3 z 12", "Page 3 of
12", a bare "3". Word never treats that text as a page number -- it is
wrong on every page but the one the model happened to look at. This
policy replaces the number(s) in a closed set of page-number PATTERNS with
the corresponding `field` node (`field_type` "page" / "num_pages" per
RFC001 §42.2), keeping every literal word as a `run` with its ORIGINAL
run style, only when the caller asks for it, and counts every rewrite.

Scope: header/footer stories only (`document.stories`, `PageStory.node.
node_type in {"header", "footer"}`) -- the body is never touched, so a
document paragraph that happens to read "Page 3 of 12" (a quote, a
citation) is left alone. Within a story the policy walks every descendant
looking for a `paragraph` whose children are ALL `run` nodes (any other
child -- an image, an existing hyperlink or field, a tab -- refuses the
whole paragraph, uncounted: "never touch a paragraph with any other
text"); its full text (the runs' text concatenated, unmodified) is tried
against PATTERNS in order and the first `fullmatch` wins.

PATTERNS (case-insensitive, `\\s+` for the spaces the model may have
mis-spaced; a documented STARTER LIST per the owner's brief -- no cheap
corpus measurement was run here, so this is an assumption, not a
measurement): a bare number; "Strana N"; "Page N"; "Seite N"; "Página N";
"N / M"; "N z M"; "N of M"; "Strana N z M"; "Page N of M". `N` becomes a
`page` field, `M` a `num_pages` field; every other character of the
matched text is kept as literal run(s), one per contributing original
run (a literal span crossing an original run boundary is split so each
piece keeps ITS run's style) -- so a two-run "Page " + "3" input yields
literal "Page " (run 1's style) + a page field, not one run with the
wrong style.

Profile refusal: `field` is a `word-visual-v1`+ node type
(`run._V1_NODE_TYPES`, pinned equal by a test), but the model-arm profile
is fixed by `run.model_profile_for` from the page's PREDICTED node types
BEFORE this policy runs. A page that predicted no v1 feature is fixed at
`word-core-v0` -- inserting a `field` node into it would build a
`V1_FEATURE_IN_V0_PROFILE` refusal on every touched page instead of a
clean strict build, exactly the trap `run.model_profile_for`'s own
2026-09-26 fix was written to avoid for the model's OWN v1 output. So
`apply` takes the page's resolved `profile` and, on the base profile,
counts `skipped_profile` per paragraph that WOULD have matched instead of
rewriting it (a real refusal, never a silently emitted invalid AST).

Idempotent: a rewritten paragraph's children are no longer all `run`
nodes (it now holds `field`s), so a second `apply` never matches it
again. `sources` provenance: the paragraph's original runs' `sources` are
concatenated onto the FIRST produced child (whichever kind); every other
produced child carries `()` -- an approximation (a literal segment or
field carries no source of its own), documented here because
`_assembled_source_owners`-style consumers must never see one source
owned twice.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from collections import Counter
from collections.abc import Iterator
from typing import Any

from unflatten.assembly import stories as st
from unflatten.assembly.assemble import AssembledDocumentHypothesis, AssembledNode

POLICY_ID = "page-number-fields.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-009.md"
COUNT_PREFIX = "page_number_fields:"
STORY_TYPES = ("header", "footer")
#: Mirrors `run.MODEL_PROFILE` (pinned equal by a test): the base profile a
#: page without a predicted v1 feature is fixed at before this policy runs.
BASE_PROFILE = "word-core-v0"

_PATTERN_NAMES: tuple[str, ...] = (
    "bare_number.v0", "strana_n.v0", "page_n.v0", "seite_n.v0", "pagina_n.v0",
    "n_slash_m.v0", "n_z_m.v0", "n_of_m.v0", "strana_n_z_m.v0", "page_n_of_m.v0",
)
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?P<n>\d+)\s*$"),
    re.compile(r"^\s*Strana\s+(?P<n>\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*Page\s+(?P<n>\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*Seite\s+(?P<n>\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*P[áa]gina\s+(?P<n>\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?P<n>\d+)\s*/\s*(?P<m>\d+)\s*$"),
    re.compile(r"^\s*(?P<n>\d+)\s+z\s+(?P<m>\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?P<n>\d+)\s+of\s+(?P<m>\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*Strana\s+(?P<n>\d+)\s+z\s+(?P<m>\d+)\s*$", re.IGNORECASE),
    re.compile(r"^\s*Page\s+(?P<n>\d+)\s+of\s+(?P<m>\d+)\s*$", re.IGNORECASE),
)


def policy_record() -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "story_types": list(STORY_TYPES),
        "base_profile": BASE_PROFILE,
        "patterns": list(_PATTERN_NAMES),
        "field_types": {"n": "page", "m": "num_pages"},
    }


def apply(
    document: AssembledDocumentHypothesis, *, profile: str
) -> tuple[AssembledDocumentHypothesis, Counter[str]]:
    """`document` with every matching header/footer paragraph rewritten into
    `field`s and literal runs, and one count per outcome. `profile` is the
    profile the page was resolved to build under (`run.model_profile_for`);
    on `BASE_PROFILE` nothing is rewritten and every match is counted
    `skipped_profile` instead."""
    counts: Counter[str] = Counter()
    allow = profile != BASE_PROFILE
    new_stories = []
    changed = False
    for story in document.stories:
        if story.node.node_type not in STORY_TYPES:
            new_stories.append(story)
            continue
        node = _walk(story.node, allow, counts)
        if node is story.node:
            new_stories.append(story)
        else:
            new_stories.append(dataclasses.replace(story, node=node))
            changed = True
    if not changed:
        return document, counts
    stories = tuple(new_stories)
    return dataclasses.replace(
        document, stories=stories, story_sets=st.derive_story_sets(document.sections, stories)
    ), counts


def _walk(node: AssembledNode, allow: bool, counts: Counter[str]) -> AssembledNode:
    if node.node_type == "paragraph":
        return _rewrite_paragraph(node, allow, counts)
    children = tuple(_walk(child, allow, counts) for child in node.children)
    if all(c is o for c, o in zip(children, node.children, strict=True)):
        return node
    return dataclasses.replace(node, children=children)


def _rewrite_paragraph(node: AssembledNode, allow: bool, counts: Counter[str]) -> AssembledNode:
    runs = node.children
    if not runs or any(child.node_type != "run" for child in runs):
        return node
    full_text = "".join(run.text or "" for run in runs)
    match = _match(full_text)
    if match is None:
        return node
    if not allow:
        counts[COUNT_PREFIX + "skipped_profile"] += 1
        return node
    new_children = _build_children(node, runs, full_text, match)
    counts[COUNT_PREFIX + "paragraphs_rewritten"] += 1
    counts[COUNT_PREFIX + "page_fields"] += 1
    if match.groupdict().get("m") is not None:
        counts[COUNT_PREFIX + "numpages_fields"] += 1
    return dataclasses.replace(node, children=new_children)


def _match(full_text: str) -> re.Match[str] | None:
    for pattern in _PATTERNS:
        match = pattern.fullmatch(full_text)
        if match is not None:
            return match
    return None


def _segments(full_text: str, match: re.Match[str]) -> list[tuple[int, int, str]]:
    marks = [(*match.span("n"), "page")]
    if match.groupdict().get("m") is not None:
        marks.append((*match.span("m"), "num_pages"))
    marks.sort()
    segments: list[tuple[int, int, str]] = []
    pos = 0
    for start, end, kind in marks:
        if start > pos:
            segments.append((pos, start, "literal"))
        segments.append((start, end, kind))
        pos = end
    if pos < len(full_text):
        segments.append((pos, len(full_text), "literal"))
    return segments


def _run_spans(runs: tuple[AssembledNode, ...]) -> list[tuple[int, int]]:
    spans = []
    pos = 0
    for run in runs:
        length = len(run.text or "")
        spans.append((pos, pos + length))
        pos += length
    return spans


def _slice_literal(
    start: int, end: int, runs: tuple[AssembledNode, ...], spans: list[tuple[int, int]]
) -> Iterator[tuple[int, int, AssembledNode]]:
    for run, (run_start, run_end) in zip(runs, spans, strict=True):
        a, b = max(start, run_start), min(end, run_end)
        if a < b:
            yield a, b, run


def _build_children(
    paragraph: AssembledNode,
    runs: tuple[AssembledNode, ...],
    full_text: str,
    match: re.Match[str],
) -> tuple[AssembledNode, ...]:
    spans = _run_spans(runs)
    segments = _segments(full_text, match)
    all_sources = tuple(source for run in runs for source in run.sources)
    counter = itertools.count()
    placed_sources = False
    out: list[AssembledNode] = []
    for start, end, kind in segments:
        if kind == "literal":
            for piece_start, piece_end, run in _slice_literal(start, end, runs, spans):
                sources = all_sources if not placed_sources else ()
                placed_sources = placed_sources or bool(all_sources)
                out.append(dataclasses.replace(
                    run,
                    assembled_id=f"{paragraph.assembled_id}.pnf{next(counter)}",
                    text=full_text[piece_start:piece_end],
                    sources=sources,
                ))
        else:
            sources = all_sources if not placed_sources else ()
            placed_sources = placed_sources or bool(all_sources)
            out.append(AssembledNode(
                assembled_id=f"{paragraph.assembled_id}.pnf{next(counter)}",
                node_type="field",
                children=(),
                text=None,
                typed={"/field_type": kind},
                sources=sources,
            ))
    return tuple(out)

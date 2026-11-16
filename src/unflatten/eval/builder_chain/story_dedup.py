"""Opt-in chain policy `story-kind-dedup.v0` (RATIFIED 2026-09-26, RFC004-
ERRATA-004 E5; measured 2026-09-26 at step 80 000, 5 of 100 pages).

A section carries at most one header and one footer per Word variant
(`/kind` in `default` / `first` / `even`, DEC-003 §2.1). The model does not
know that rule: it emits two or three stories of one type in one section of
one page (footers `n1`, `n10`, `n74`), and the strict build then refuses the
page with `DUPLICATE_STORY_KIND` at `/sections/0/footers/1`.

Re-kinding the predicted `/kind` cannot fix it (measured on prod, bundle
`112a1841…`): the assembler never reads the predicted kind. Per section,
story type and page it lifts EVERY story root into one `PageStory` each,
`stories.derive_story_sets` folds all of one page's roots into ONE
`DerivedStory` whose kind follows from the page sequence alone (a single
page: pattern `all_equal`, kind `default`), and `ast_builder.attach_stories`
stamps that one kind on every root -- two or three `default` footers, whatever
the heads said. The duplicate is structural, so the fix is structural and
post-assembly, like `recovered_image_materialization`: this policy MERGES the
page's extra stories of one type into its first one and recomputes the story
sets, only when the caller asks for it, and counts every edit so the chain
report can audit it.

Rule, per `(section_id, story type, page_ordinal)` with more than one
`PageStory` in `document.stories` (order = order in `document.stories`, the
assembler's lift order):

1. the FIRST story is kept; its node becomes `dataclasses.replace(first.node,
   children=<first's children + every later story's children, in order>,
   sources=<first's sources + every later story's sources>)`. Text and
   typed values of the first story stay; the later stories' own root
   `typed` (their predicted `/kind` included) is discarded with the root,
   their children and every descendant move over unchanged, assembled ids
   included. `sources` are concatenated so the merged root still owns every
   page piece the removed roots owned (`run._assembled_source_owners`);
2. the later stories leave `document.stories` (`story_dedup:stories_merged`
   per removed story, `story_dedup:children_moved` per child moved);
3. `story_sets` are recomputed with `stories.derive_story_sets(
   document.sections, <new stories>)` -- `document.sections` IS the
   `built_sections` the assembler passed (`assemble._finalize`), so the
   recomputation sees exactly what the assembler would have seen with one
   story per page.

Everything else (`pages`, `sections`, `notes`, `seam_decisions`,
`conflicts`) is the same object. A document with at most one story per
(section, type, page) is returned as the same object with no counts.
Idempotent: a second `apply` merges nothing. The merge is a repair of the
model's output, not gold: a gold page never carries two stories of one type
on one page, so the gold arm is untouched by construction.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import TYPE_CHECKING, Any

from unflatten.assembly import stories as st

if TYPE_CHECKING:
    from unflatten.assembly.assemble import AssembledDocumentHypothesis, PageStory

POLICY_ID = "story-kind-dedup.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-004.md"
COUNT_PREFIX = "story_dedup:"
STORY_TYPES = ("header", "footer")
RULES = (
    "first_page_story_kept.v0",
    "later_page_stories_merged_into_first.v0",
    "story_sets_rederived.v0",
)


def policy_record() -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "story_types": list(STORY_TYPES),
        "merge_key": ["section_id", "story_type", "page_ordinal"],
        "rules": list(RULES),
    }


def apply(
    document: AssembledDocumentHypothesis,
) -> tuple[AssembledDocumentHypothesis, Counter[str]]:
    """`document` with every page's extra stories of one type merged into
    its first, and one count per edit kind; the same `document` object when
    nothing needed merging."""
    counts: Counter[str] = Counter()
    groups: dict[tuple[str, str, int], list[int]] = {}
    for i, story in enumerate(document.stories):
        if story.node.node_type in STORY_TYPES:
            key = (story.section_id, story.node.node_type, story.page_ordinal)
            groups.setdefault(key, []).append(i)
    merged: dict[int, PageStory] = {}
    removed: set[int] = set()
    for indices in groups.values():
        if len(indices) < 2:
            continue
        first = document.stories[indices[0]]
        extra = [document.stories[i] for i in indices[1:]]
        node = first.node
        merged[indices[0]] = dataclasses.replace(
            first,
            node=dataclasses.replace(
                node,
                children=node.children + tuple(c for s in extra for c in s.node.children),
                sources=node.sources + tuple(src for s in extra for src in s.node.sources),
            ),
        )
        removed.update(indices[1:])
        counts[COUNT_PREFIX + "stories_merged"] += len(extra)
        counts[COUNT_PREFIX + "children_moved"] += sum(len(s.node.children) for s in extra)
    if not removed:
        return document, counts
    stories = tuple(
        merged.get(i, story) for i, story in enumerate(document.stories) if i not in removed
    )
    edited = dataclasses.replace(
        document,
        stories=stories,
        story_sets=st.derive_story_sets(document.sections, stories),
    )
    return edited, counts

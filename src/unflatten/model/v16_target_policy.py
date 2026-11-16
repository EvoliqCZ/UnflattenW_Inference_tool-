"""v16 target policy `v16-targets.v1` (DEC-055 V1a/V1b/V1c, loop-data audit
2026-09-29 §6; design `docs/plans/v16-data-fixes-design.md` §1).

A PROJECTION of the training target, like DEC-033: the Canonical AST, the
store, the Observation Sidecar and the fragments are untouched, so
`ast_hash`/`fragment_id`/`derivation_hash` do not move because of this
module. What moves is `training_target_materializer_hash` (the policy's
`descriptor()` enters the materializer descriptor) and with it the
exact-text cache identity. Without `--target-policy` nothing here runs and
every output stays byte-identical.

Rules (each refuses nothing by default except where named):

- **V1a** :func:`grant_whitespace_runs` -- a whitespace-only run with no
  grant on this page gets its whole range when the nearest glyph-bearing
  sibling run on BOTH sides (same parent) owns its adjacent character on
  this page. The aligner never observes whitespace, so the fragment keeps
  saying "not observed"; ownership is inferred here, where the
  `text_unowned` withdrawal is decided (`training_materializer`).
- **V1b** lives in `page_projection.project(..., collapse_withdrawn_runs=True)`.
- **V1c** :func:`unowned_ink` -- measurement only; `unowned_ink_floor_chars`
  is a PLACEHOLDER (`None`), the owner sets the threshold later.
- **I1 fallback** -- :func:`spill_signature` (counted refusal
  `skipped_defect_spill`) and :func:`load_page_exclusions` (counted refusal
  `skipped_defect_listed`), because per-row geometry for a textless table
  row does not exist anywhere in the store (design §1).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from unflatten.dataset.identity import domain_hash, lineage_root_id_real
from unflatten.model.boundary_projector import TextSpanAuthority
from unflatten.model.page_projection import _Tree

V16_TARGET_POLICY_ID = "v16-targets.v1"

#: Audit §3 signature: textless rows trail a table's last observed row and
#: the NEXT page of the document has no observation (>= 4 rows), or only
#: the PREVIOUS page has none (>= 16 rows; 2 of 3 shorter cases were
#: legitimate).
SPILL_MIN_TRAILING_ROWS = 4
SPILL_BACK_MIN_TRAILING_ROWS = 16

#: Skip reasons this policy adds to the cache build's partition. Present in
#: a build's counts only when the policy is selected, so the default build
#: summary keeps its exact key set.
V16_SKIP_COUNTS: tuple[str, ...] = (
    "skipped_defect_listed",
    "skipped_defect_spill",
    "skipped_unowned_ink",
)

_EXCLUSIONS_DOMAIN = "unflatten.v16-page-exclusions/v1"
_SHA_HEX = re.compile(r"^[0-9a-f]{64}$")


class PageExclusionsInvalid(ValueError):
    """An exclusion list row is malformed, duplicated, or names a
    `lineage_root_id` that is not the one its `source_sha256` derives."""


@dataclass(frozen=True)
class PageExclusions:
    """`(lineage_root_id, page_index) -> reason`. The lineage is checked at
    load to be `lineage_root_id_real(source_sha256)`, so matching a page by
    its sample's `lineage_root_id` IS matching by `source_sha256` -- the key
    that survives the T2 store revision, where `document_bundle_id` does
    not."""

    entries: Mapping[tuple[str, int], str]
    sha256: str

    def __len__(self) -> int:
        return len(self.entries)

    def reason(self, lineage_root_id: str, page_index: int) -> str | None:
        return self.entries.get((lineage_root_id, int(page_index)))


def load_page_exclusions(paths: Sequence[Path | str]) -> PageExclusions:
    """Read audit exclusion TSVs (`page_id, source_sha256, lineage_root_id,
    page_index, reason, ...`; `#` lines are comments). The digest covers
    the parsed entries, not the file bytes, so a comment edit is not an
    identity change and a reordered list is the same list."""
    rows: dict[tuple[str, int], tuple[str, str]] = {}
    for path in paths:
        for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
            if not line.strip() or line.startswith("#"):
                continue
            where = f"{path}:{number}"
            cols = line.split("\t")
            if len(cols) < 5:
                raise PageExclusionsInvalid(f"{where}: expected >= 5 tab-separated columns")
            _page_id, source_sha256, lineage, page_text, reason = cols[:5]
            if not _SHA_HEX.match(source_sha256):
                raise PageExclusionsInvalid(
                    f"{where}: source_sha256 {source_sha256!r} is not 64 hex"
                )
            if not page_text.isdigit():
                raise PageExclusionsInvalid(f"{where}: page_index {page_text!r} is not an integer")
            if not reason:
                raise PageExclusionsInvalid(f"{where}: empty reason")
            derived = lineage_root_id_real(source_set_hash=f"sha256:{source_sha256}")
            if lineage != derived:
                raise PageExclusionsInvalid(
                    f"{where}: lineage_root_id {lineage!r} is not the one source_sha256 "
                    f"derives ({derived!r}) -- the list belongs to another corpus or is corrupt"
                )
            key = (lineage, int(page_text))
            if key in rows:
                raise PageExclusionsInvalid(f"{where}: duplicate page {source_sha256}#{page_text}")
            rows[key] = (source_sha256, reason)
    digest = domain_hash(
        _EXCLUSIONS_DOMAIN,
        sorted([src, page, reason] for (_lin, page), (src, reason) in rows.items()),
    )
    return PageExclusions(
        entries={key: reason for key, (_src, reason) in rows.items()}, sha256=digest
    )


@dataclass(frozen=True)
class TargetPolicy:
    """The selected v16 target projection. `descriptor()` is what enters the
    materializer descriptor -- every field that changes a row or the cohort
    is in it."""

    policy_id: str = V16_TARGET_POLICY_ID
    whitespace_run_grant: bool = True
    withdrawn_run_collapse: bool = True
    spill_refusal: bool = True
    #: V1d: an inherited header/footer is emitted in the page's own section
    story_rehome: bool = True
    #: V1e: a withdrawn whitespace-only run is dropped from the page
    withdrawn_whitespace_drop: bool = True
    #: V1c: PLACEHOLDER until the owner picks a threshold -- `None` means
    #: "measure, never refuse" (design §7 point 1).
    unowned_ink_floor_chars: int | None = None
    page_exclusions: PageExclusions | None = field(default=None, compare=False)

    def descriptor(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "whitespace_run_grant": self.whitespace_run_grant,
            "withdrawn_run_collapse": self.withdrawn_run_collapse,
            "story_rehome": self.story_rehome,
            "withdrawn_whitespace_drop": self.withdrawn_whitespace_drop,
            "spill_refusal": (
                {
                    "min_trailing_rows_next_empty": SPILL_MIN_TRAILING_ROWS,
                    "min_trailing_rows_previous_empty": SPILL_BACK_MIN_TRAILING_ROWS,
                }
                if self.spill_refusal
                else None
            ),
            "unowned_ink_floor_chars": self.unowned_ink_floor_chars,
            "unowned_ink_floor_is_placeholder": self.unowned_ink_floor_chars is None,
            "page_exclusions_sha256": (
                self.page_exclusions.sha256 if self.page_exclusions is not None else None
            ),
        }

    def listed_reason(self, lineage_root_id: str, page_index: int) -> str | None:
        if self.page_exclusions is None:
            return None
        return self.page_exclusions.reason(lineage_root_id, page_index)

    def refuses_unowned_ink(self, unowned_ink_chars: int) -> bool:
        return (
            self.unowned_ink_floor_chars is not None
            and unowned_ink_chars > self.unowned_ink_floor_chars
        )


def projection_kwargs(policy: TargetPolicy | None) -> dict[str, Any]:
    """Extra `page_projection.project` keywords under `policy`. Empty for no
    policy, so a pre-v16 build calls `project` with its literal old shape."""
    kwargs: dict[str, Any] = {}
    if policy is not None and policy.withdrawn_run_collapse:
        kwargs["collapse_withdrawn_runs"] = True
    if policy is not None and policy.withdrawn_whitespace_drop:
        kwargs["drop_withdrawn_whitespace_runs"] = True
    return kwargs


def resolve_target_policy(
    policy_id: str | None, *, page_exclusions: PageExclusions | None = None
) -> TargetPolicy | None:
    """`None` -> no policy (today's build, byte-identical). An exclusion
    list without a policy is refused: it would change the cohort while no
    identity field named it."""
    if policy_id is None:
        if page_exclusions is not None:
            raise ValueError(
                "--page-exclusions requires --target-policy: the list must be bound "
                "into the materializer descriptor, never applied unnamed"
            )
        return None
    if policy_id != V16_TARGET_POLICY_ID:
        raise ValueError(f"unknown target policy {policy_id!r} (known: {V16_TARGET_POLICY_ID!r})")
    return TargetPolicy(page_exclusions=page_exclusions)


# ---------------------------------------------------------------------------
# V1a
# ---------------------------------------------------------------------------


def _owns(span: TextSpanAuthority, index: int) -> bool:
    return any(start <= index < end for grant in span.grants for start, end in grant.char_ranges)


def _owns_any(span: TextSpanAuthority) -> bool:
    """At least one non-empty granted range on this page."""
    return any(end > start for grant in span.grants for start, end in grant.char_ranges)


def _is_whitespace_run(span: TextSpanAuthority | None) -> bool:
    return span is not None and bool(span.text) and not span.text.strip()


def grant_whitespace_runs(
    text_spans: Sequence[TextSpanAuthority],
    fragment: Mapping[str, Any],
    structural: Mapping[str, Any],
) -> tuple[tuple[TextSpanAuthority, ...], int]:
    """`(text_spans with V1a grants, runs granted)` -- see module docstring.

    Only a grant with no char range is widened (a whitespace run is always
    `structural`: it has no ink); its masks and evidence stay as the fragment
    slice set them. A run whose slice is not on this page has no authority
    here and is never touched."""
    tree = _Tree(dict(structural))
    present = {entry["node_id"] for entry in fragment.get("slices", ())}
    span_by_node = {span.node_id: span for span in text_spans}

    def glyph_run(ordinal: int) -> TextSpanAuthority | None:
        span = span_by_node.get(tree.node_id_of(ordinal))
        if tree.type_of(ordinal) != "run" or span is None or not span.text.strip():
            return None
        return span

    granted_ids: set[str] = set()
    for span in text_spans:
        if not _is_whitespace_run(span) or span.node_id not in present:
            continue
        if not span.grants or _owns_any(span):
            continue
        ordinal = tree.ordinal_by_node_id.get(span.node_id)
        if ordinal is None:
            continue
        siblings = tree.children.get(tree.parent_ordinal[ordinal], [])
        position = siblings.index(ordinal)
        left = next(
            (s for o in reversed(siblings[:position]) if (s := glyph_run(o)) is not None), None
        )
        right = next((s for o in siblings[position + 1 :] if (s := glyph_run(o)) is not None), None)
        if left is None or right is None:
            continue
        left_tail = len(left.text.rstrip()) - 1
        right_head = len(right.text) - len(right.text.lstrip())
        if _owns(left, left_tail) and _owns(right, right_head):
            granted_ids.add(span.node_id)

    out = tuple(
        replace(
            span,
            grants=tuple(
                replace(grant, char_ranges=((0, len(span.text)),), text_range_owner=True)
                for grant in span.grants
            ),
        )
        if span.node_id in granted_ids
        else span
        for span in text_spans
    )
    return out, len(granted_ids)


# ---------------------------------------------------------------------------
# V1d
# ---------------------------------------------------------------------------

_STORY_TYPES = frozenset({"header", "footer"})


def _story_kind(typed: Mapping[str, Any], node_id: str) -> Any:
    for entry in (typed.get(node_id) or {}).get("fields", ()):
        if entry.get("node_property_pointer") == "/kind":
            return entry.get("value")
    return None


def rehome_inherited_stories(
    fragment: Mapping[str, Any],
    structural: Mapping[str, Any],
    typed: Mapping[str, Any],
) -> tuple[dict[str, str], frozenset[str]]:
    """V1d: `(story id -> section id to emit it in, story-only sections to
    drop)` for one page.

    A section with no body on the page that is present only through its
    headers/footers got there because a LATER section on the page inherits
    them (OOXML: a section without its own story of a type and kind shows
    the previous one; `stories._heir_sections`). DEC-033 §3 makes the page
    a standalone document whose section carries the furniture it shows, so
    each such story moves into the first later section with body on the
    page that inherits it (no section in between has its own story of that
    type and kind; `first` is never inherited), and the emptied section is
    dropped. All-or-nothing per section: a story that cannot move keeps the
    whole section, which the `bodyless_sections` G-A detector then names.
    Iteration 2 sample: 149 of 152 body-less-section stories are
    inherited by a section with body on the same page."""
    tree = _Tree(dict(structural))
    present = {entry["node_id"] for entry in fragment.get("slices", ())}
    sections = [
        o
        for root in tree.children.get(-1, [])
        for o in tree.children.get(root, [])
        if tree.type_of(o) == "section"
    ]

    def stories(section: int) -> list[int]:
        return [c for c in tree.children.get(section, []) if tree.type_of(c) in _STORY_TYPES]

    def has_body(section: int) -> bool:
        return any(
            tree.type_of(c) not in _STORY_TYPES and tree.node_id_of(c) in present
            for c in tree.children.get(section, [])
        )

    own = {
        s: {(tree.type_of(c), _story_kind(typed, tree.node_id_of(c))) for c in stories(s)}
        for s in sections
    }
    rehome: dict[str, str] = {}
    dropped: set[str] = set()
    for i, section in enumerate(sections):
        if tree.node_id_of(section) not in present or has_body(section):
            continue
        here = [c for c in stories(section) if tree.node_id_of(c) in present]
        moves: dict[str, str] = {}
        for story in here:
            type_kind = (tree.type_of(story), _story_kind(typed, tree.node_id_of(story)))
            if type_kind[1] in (None, "first"):
                break
            for later in sections[i + 1 :]:
                if type_kind in own[later]:
                    break
                if tree.node_id_of(later) in present and has_body(later):
                    moves[tree.node_id_of(story)] = tree.node_id_of(later)
                    break
        if here and len(moves) == len(here):
            rehome.update(moves)
            dropped.add(tree.node_id_of(section))
    return rehome, frozenset(dropped)


# ---------------------------------------------------------------------------
# V1c
# ---------------------------------------------------------------------------


def unowned_ink(
    text_spans: Sequence[TextSpanAuthority],
    fragment: Mapping[str, Any],
    structural: Mapping[str, Any],
) -> tuple[int, int]:
    """`(glyph chars, paragraphs)` of visible text with no owner on this
    page, counted only in paragraphs that own no character here at all
    (the audit's `c44fd1fd7f04#1` / `skeleton` shape). A paragraph that
    owns part of its text is a partial alignment, not this defect."""
    tree = _Tree(dict(structural))
    present = {entry["node_id"] for entry in fragment.get("slices", ())}

    def paragraph_of(node_id: str) -> int | None:
        ordinal = tree.ordinal_by_node_id.get(node_id)
        while ordinal is not None and ordinal != -1:
            if tree.type_of(ordinal) == "paragraph":
                return ordinal
            ordinal = tree.parent_ordinal[ordinal]
        return None

    owned_paragraphs: set[int] = set()
    unowned_by_paragraph: dict[int, int] = {}
    for span in text_spans:
        if span.node_id not in present:
            continue
        paragraph = paragraph_of(span.node_id)
        if paragraph is None:
            continue
        if _owns_any(span):
            owned_paragraphs.add(paragraph)
            continue
        glyphs = sum(1 for ch in span.text if not ch.isspace())
        if glyphs:
            unowned_by_paragraph[paragraph] = unowned_by_paragraph.get(paragraph, 0) + glyphs
    counted = {p: n for p, n in unowned_by_paragraph.items() if p not in owned_paragraphs}
    return sum(counted.values()), len(counted)


# ---------------------------------------------------------------------------
# I1 fallback: the spill signature
# ---------------------------------------------------------------------------


def spill_signature(
    fragment: Mapping[str, Any],
    structural: Mapping[str, Any],
    observation_sidecar: Mapping[str, Any],
    page_index: int,
) -> tuple[bool, int]:
    """`(is_spill, trailing textless rows)` for one page.

    Trailing rows = rows of a table present here that follow the table's
    last row with an observation (`page_evidence != "observed"`); the page
    maximum over its tables. The neighbour test reads the sidecar: a page
    with no observation at all is the page `envelope._resolve_pages` rule 1
    emptied (its rows were borrowed onto the previous page)."""
    tree = _Tree(dict(structural))
    evidence = {
        entry["node_id"]: entry.get("page_evidence") for entry in fragment.get("slices", ())
    }
    trailing_max = 0
    for ordinal, node_type in enumerate(tree.node_types):
        if node_type != "table" or tree.node_id_of(ordinal) not in evidence:
            continue
        rows = [
            child
            for child in tree.children.get(ordinal, [])
            if tree.type_of(child) == "row" and tree.node_id_of(child) in evidence
        ]
        trailing = 0
        for row in reversed(rows):
            if evidence[tree.node_id_of(row)] == "observed":
                break
            trailing += 1
        trailing_max = max(trailing_max, trailing)

    pages = observation_sidecar.get("pages", ())

    def empty(index: int) -> bool:
        return 0 <= index < len(pages) and not pages[index].get("observations")

    forward = empty(page_index + 1) and trailing_max >= SPILL_MIN_TRAILING_ROWS
    backward = empty(page_index - 1) and trailing_max >= SPILL_BACK_MIN_TRAILING_ROWS
    return forward or backward, trailing_max

"""Real-data bridge: RFC003 target-bundle text spans + fragment slices ->
`training_materializer.TextSpanAuthority`/`TextRangeGrant`.

**Why this module exists (2026-09-05 review, team lead).** Every real S1
wiring so far called `TrainingTargetMaterializer.materialize(...,
text_spans=())` -- real text data was never missing, nobody had written the
few lines that reshape it. Measured against 60 real `release-v10` single-
page documents (1770 text spans): `target_bundle["text"]["spans"]` already
carries the real text content, and `fragment["slices"]` (the SAME object
already passed to `materialize` as `fragment`) already carries, per node,
exactly the fields `TextRangeGrant` needs -- `char_ranges`,
`text_range_owner`, `observable_mask`, `supervision_mask`, `page_evidence`
(`dataset/fragments/envelope.py:283-325`). `DEC-005` R4-3 scopes S1 as
"core structure/**exact-text**", not structure-only, and names a
"reversible text path" among its exit criteria -- so a real S1 run that
supervises zero text was not an expected deferral, it was this adapter not
existing yet.

**The one hard rule (team lead's condition, verbatim): do not invent
alignment.** `TextAuthorityAlignmentError` is raised, never silently
resolved, for any of the ways real data could disagree with the
one-span-one-owning-slice assumption this module makes:

- a span's `node_id` matches more than one slice (this module has no rule
  for picking one),
- a slice with non-empty `char_ranges` matches zero or more than one span
  by `node_id` -- a granted range with nothing to attach it to, or nothing
  to choose among (design §4.3, "reverse direction"),
- a slice's `char_ranges` fall outside `[0, span["end"])`, or `span["start"]`
  is ever nonzero (measured 0/1770 times on real data -- this module has no
  defined meaning for a nonzero start, so it refuses rather than guess one).

**Multi-page relaxation (design §4.3):** a span's `node_id` matching zero
fragment slices is no longer an error -- a fragment is one page, and a span
whose node lives on a different page is simply out of scope for this one.
It is skipped: no `TextSpanAuthority` is produced for it, and in particular
no *empty* one -- that would recreate the `missing-grant-is-not-empty-text`
defect in a new place.

**How much text this actually supervises, measured, not "virtually
all" (2026-09-05, 401 real single-page `release-v10` train documents,
12443 real text spans):** supervised 10080 (**81.0%**), withdrawn 2363
(**19.0%**); `page_evidence` distribution `observed` 8711 / `unresolved`
2363 / `structural` 1369 (`tests/model/test_masking_pipeline_real_data.py
::test_real_supervision_statistics`, which any caller can rerun on a
different sample). Nearly a fifth of real text spans are withdrawn as
`unresolved` -- the same placement-evidence family as the header-ownership
problem the data side has separately been working (an `unresolved` node's
page could not be confirmed from pixels). 19% is high enough that whether
it can be reduced is a real, open question for later, not settled by this
module -- recorded here so anyone asking starts from a number instead of a
surprise.

This module authorizes no gradient update: it declares dataclasses'
construction only, no `nn.Module`, no weights, no optimizer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from unflatten.model.boundary_projector import TextRangeGrant, TextSpanAuthority


class TextAuthorityAlignmentError(ValueError):
    """A real text span could not be aligned to fragment ownership without
    guessing (see module docstring's "one hard rule"). M1-internal, not an
    RFC004 registry code -- mirrors the CamelCase-exception convention
    `composite_grammar.py`/`composite_vocab.py` already use for the same
    reason (design §9.3)."""


def build_text_span_authorities(
    *,
    document_bundle_id: str,
    target_bundle_hash: str,
    text: dict[str, Any],
    fragment: dict[str, Any],
    ownership_projection_hash: str,
) -> tuple[TextSpanAuthority, ...]:
    """`(item.targets["text"], item.fragment) -> tuple[TextSpanAuthority, ...]`,
    one per real text span, each carrying exactly one `TextRangeGrant` built
    from the fragment slice sharing that span's `node_id`.

    `ownership_projection_hash` is caller-supplied, matching
    `TrainingTargetMaterializer.materialize`'s own
    `structural_ownership_projection_hash`/`structural_projection_hash`
    convention (its docstring: "M1 has no separate hash for those distinct
    from the target bundle and fragment identity themselves, so the caller
    supplies the binding value it wants pinned") -- this module does not
    invent a new hash artifact either.

    Raises `TextAuthorityAlignmentError` per the module docstring's hard
    rule rather than silently resolving an ambiguous or out-of-bounds span.
    """
    fragment_id = fragment.get("fragment_id")
    fragment_schema_version = fragment.get("fragment_schema_version")
    fragment_sequence_index = fragment.get("sequence_index")

    slices_by_node: dict[str, list[dict[str, Any]]] = {}
    for slc in fragment.get("slices", ()):
        slices_by_node.setdefault(slc["node_id"], []).append(slc)

    # Reverse direction (design §4.3): a `run` slice that was actually
    # granted characters must have exactly one span to attach them to.
    # Only `run` nodes ever carry an `op == "text"` event
    # (`targets/events.py:135`); a container's own `char_ranges` (e.g. a
    # paragraph's, aggregated from its descendants) is not a text grant
    # and has no span to match, so it is not this check's business.
    # Checked against every slice up front, independent of the forward
    # loop below, so a granted range with nothing to own it is never
    # silently dropped.
    spans_by_node: dict[str, list[dict[str, Any]]] = {}
    for span in text.get("spans", ()):
        spans_by_node.setdefault(span["node_id"], []).append(span)
    for slc in fragment.get("slices", ()):
        if slc.get("node_type") != "run" or not slc.get("char_ranges"):
            continue
        node_id = slc["node_id"]
        owners = spans_by_node.get(node_id, [])
        if len(owners) != 1:
            raise TextAuthorityAlignmentError(
                f"fragment slice for node {node_id!r} has non-empty char_ranges "
                f"and {len(owners)} matching text span(s); alignment requires "
                "exactly one -- 0 means a granted range with nothing to attach it "
                "to, >1 means this module has no rule for choosing among them"
            )

    authorities: list[TextSpanAuthority] = []
    for span in text.get("spans", ()):
        span_id = span["text_span_id"]
        node_id = span["node_id"]
        if span["start"] != 0:
            raise TextAuthorityAlignmentError(
                f"text span {span_id!r} (node {node_id!r}) has start={span['start']!r}, "
                "not 0 -- this module has no defined meaning for a nonzero span start "
                "and refuses to guess one rather than silently misalign char_ranges"
            )

        matches = slices_by_node.get(node_id, [])
        if not matches:
            # Forward relaxation (design §4.3): this fragment is one page
            # and does not carry the node at all -- out of scope, not an
            # error. Skipped means no TextSpanAuthority at all, never an
            # empty one.
            continue
        if len(matches) != 1:
            raise TextAuthorityAlignmentError(
                f"text span {span_id!r} (node {node_id!r}) matches {len(matches)} "
                "fragment slice(s); alignment requires exactly one -- "
                ">1 means this module has no rule for choosing among them"
            )
        slc = matches[0]

        char_ranges: list[tuple[int, int]] = []
        for cr in slc.get("char_ranges", ()):
            start, end = int(cr["start"]), int(cr["end"])
            if not (0 <= start < end <= span["end"]):
                raise TextAuthorityAlignmentError(
                    f"text span {span_id!r} (node {node_id!r}): fragment char_range "
                    f"[{start}, {end}) is outside the span's own [0, {span['end']}) -- "
                    "the fragment builder and the target bundle disagree about this "
                    "span's extent"
                )
            char_ranges.append((start, end))

        grant = TextRangeGrant(
            owner_fragment_id=fragment_id,
            owner_scope="fragment_local",
            fragment_schema_version=fragment_schema_version,
            fragment_sequence_index=fragment_sequence_index,
            char_ranges=tuple(char_ranges),
            # Text content is extracted verbatim from the AST -- unlike a
            # typed property, there is no "value unknown" state for it
            # (target_bundle.schema.json's `known_mask` applies only to
            # `typed` entries, never to `text.spans`).
            known_mask=True,
            observable_mask=bool(slc.get("observable_mask")),
            supervision_mask=bool(slc.get("supervision_mask")),
            structure_owner=slc.get("structure_owner"),
            typed_target_owner=slc.get("typed_target_owner"),
            text_range_owner=bool(slc.get("text_range_owner")),
            page_evidence=slc.get("page_evidence"),
            ownership_projection_hash=ownership_projection_hash,
        )
        authorities.append(
            TextSpanAuthority(
                text_span_id=span_id,
                node_id=node_id,
                text=span["text"],
                document_bundle_id=document_bundle_id,
                target_bundle_hash=target_bundle_hash,
                grants=(grant,),
            )
        )
    return tuple(authorities)


@dataclass(frozen=True, slots=True)
class TextCoverage:
    """How much of one page's store text a fragment was actually granted
    (D2, `scratchpad/audit-exact-text/`, 2026-09-07): 101/1,176 real cache
    rows had granted/store < 0.5, and 2 rows have `store_chars == 0` (a
    page with no text at all), for which a ratio has no defined meaning."""

    granted_chars: int
    store_chars: int

    @property
    def ratio(self) -> float | None:
        """`granted_chars / store_chars`, or `None` when `store_chars == 0`
        -- a page with zero store characters has nothing to have coverage
        OF, so the ratio is undefined rather than vacuously 1.0 or 0.0."""
        if self.store_chars == 0:
            return None
        return self.granted_chars / self.store_chars

    def below(self, floor: float) -> bool:
        """`True` iff `ratio` is defined and strictly below `floor`. A
        `None` ratio (undefined) never withdraws a page -- that is the
        one rule this method enforces. (A `floor <= 0` clause was removed
        2026-09-07: every caller validates its floor through
        `validate_min_text_coverage`, which refuses anything outside
        `[0, 1]`, and `ratio` is never negative, so `ratio < floor` for
        `floor == 0` is already always `False` without a special case --
        keeping the dead branch would have made this docstring claim a
        rule that no caller can actually reach.)"""
        ratio = self.ratio
        if ratio is None:
            return False
        return ratio < floor


def text_coverage(authorities: Sequence[TextSpanAuthority]) -> TextCoverage:
    """One page's `TextCoverage`, summed across every span's own text
    length (`store_chars`) and every grant's `char_ranges` (`granted_chars`)
    -- the same authority `TrainingTargetMaterializer.materialize` and
    `unowned_text_event_origins` read, not a second, independent count."""
    granted = 0
    store = 0
    for span in authorities:
        store += len(span.text)
        for grant in span.grants:
            granted += sum(end - start for start, end in grant.char_ranges)
    return TextCoverage(granted_chars=granted, store_chars=store)

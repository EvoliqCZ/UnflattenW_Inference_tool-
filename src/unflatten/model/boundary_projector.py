"""M1 scalar boundary projection (design §3, §6).

Split out of `training_materializer.py` because the split follows a real
seam where the *kind* of data changes: RFC003 text-span authority in,
homogeneous `ScalarPolicyRange`s out. The boundary-first projection
algorithm is self-contained and `training_materializer.py` only
consumes its output, so the two halves are separately testable.

Turns RFC003 text-span authority (`TextSpanAuthority`, carrying
`TextRangeGrant`s that mirror the real fragment-slice shape from
`src/unflatten/dataset/fragments/envelope.py`) into `ScalarPolicyRange`s:
homogeneous, atomic scalar ranges with a resolved five-way `effective`
mask (design §6 "Boundary-first projection", §16 criterion 7: "zero
tokens cross a scalar owner/effective-mask boundary" -- guaranteed here
because every emitted range is already homogeneous by construction, and
every downstream token drawn from one range shares that range's whole
`[scalar_start, scalar_end)` locator (design §7: "Multiple Qwen tokens
may share one homogeneous scalar range, but each remains one output/
LossUnit").

This module authorizes no gradient update: it declares only dataclasses
and pure functions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

type OwnerScope = Literal["global_target", "fragment_local", "example_local", "document_scope"]


class TextRangeBoundaryInvalid(ValueError):
    """A text span's ownership/stage grants are gapped, overlapping, or ambiguous.

    M1-internal (§9.3); mirrors the proposed `TEXT_RANGE_BOUNDARY_INVALID`
    registry record (§9.2), which is not yet ratified (RR-3, §9.1).
    """


@dataclass(frozen=True, slots=True)
class TextRangeGrant:
    """One fragment-owned, homogeneous ownership/mask grant over a text span.

    Mirrors the real RFC003 fragment-slice shape
    (`src/unflatten/dataset/fragments/envelope.py`): `structure_owner`,
    `typed_target_owner`, `text_range_owner`, `page_evidence`,
    `observable_mask`, `supervision_mask` are node-level scalars there, so
    one grant applies them uniformly across every `char_ranges` interval
    it lists. The projector still cuts at every interval boundary (design
    §6 step 3), so multiple disjoint `char_ranges` never merge across a
    gap even though their attributes match.
    """

    owner_fragment_id: str | None  # None <=> owner_scope == "global_target"
    owner_scope: OwnerScope
    fragment_schema_version: str | None
    fragment_sequence_index: int | None
    char_ranges: tuple[tuple[int, int], ...]
    known_mask: bool
    observable_mask: bool
    supervision_mask: bool
    structure_owner: bool | None
    typed_target_owner: bool | None
    text_range_owner: bool | None  # drives ownership_enabled
    page_evidence: str | None
    ownership_projection_hash: str


@dataclass(frozen=True, slots=True)
class TextSpanAuthority:
    """One RFC003 text span's immutable authority plus its ownership grants.

    `grants` need not cover all of `text`: characters outside every
    grant's `char_ranges` are owned by a different fragment (or none) and
    are simply out of scope for this materialization -- never emitted,
    never guessed (design §6/§7: "Missing authority is not true").
    """

    text_span_id: str
    node_id: str
    text: str  # already NFC-validated by the caller
    document_bundle_id: str
    target_bundle_hash: str
    grants: tuple[TextRangeGrant, ...]
    #: Additional stage-policy cuts, `(start, end, stage_enabled)`,
    #: half-open, disjoint. Positions outside every entry default to
    #: `default_stage_enabled`.
    stage_ranges: tuple[tuple[int, int, bool], ...] = ()
    default_stage_enabled: bool = True


@dataclass(frozen=True, slots=True)
class OwnershipGrant:
    """Raw RFC003 ownership facts one `ScalarPolicyRange` was projected from.

    Carried alongside the design §3 `ScalarPolicyRange` fields (not part
    of that shape itself) so LossUnit construction can assemble the full
    `Rfc003OwnershipProof` (design §4.6) without a second lookup into
    dataset state -- the accepted variant (A) of that section.
    """

    document_bundle_id: str
    target_bundle_hash: str
    fragment_schema_version: str | None
    fragment_sequence_index: int | None
    structure_owner: bool | None
    typed_target_owner: bool | None
    text_range_owner: bool | None
    page_evidence: str | None
    owned_char_ranges: tuple[tuple[int, int], ...]
    ownership_projection_hash: str


@dataclass(frozen=True, slots=True)
class ScalarPolicyRange:
    text_span_id: str
    node_id: str
    scalar_start: int
    scalar_end: int
    owner_fragment_id: str | None
    owner_scope: OwnerScope
    known_mask: bool
    observable_mask: bool
    supervision_mask: bool
    ownership_enabled: bool
    stage_enabled: bool
    effective: bool
    reason_codes: tuple[str, ...]
    ownership: OwnershipGrant


def compute_effective(
    *,
    known_mask: bool,
    observable_mask: bool,
    supervision_mask: bool,
    ownership_enabled: bool,
    stage_enabled: bool,
) -> tuple[bool, tuple[str, ...]]:
    """The exact five-way conjunction (design §6: ``effective = known AND
    observable AND supervision AND ownership AND stage``), plus the
    ordered `reason_codes` of every failing prerequisite. Every argument
    is a required `bool` -- there is no `None`/missing branch, because a
    missing label must never be silently treated as `True` (or encoded as
    a numeric zero downstream): a caller that does not know a
    prerequisite must pass `False` explicitly.
    """
    reasons: list[str] = []
    if not known_mask:
        reasons.append("not_known")
    if not observable_mask:
        reasons.append("not_observable")
    if not supervision_mask:
        reasons.append("not_supervised")
    if not ownership_enabled:
        reasons.append("not_owned")
    if not stage_enabled:
        reasons.append("not_stage_enabled")
    return (not reasons, tuple(reasons))


class ScalarBoundaryProjector:
    """Boundary-first projection of text-span authority into `ScalarPolicyRange`s
    (design §6 "Boundary-first projection")."""

    def project(self, spans: Iterable[TextSpanAuthority]) -> tuple[ScalarPolicyRange, ...]:
        ranges: list[ScalarPolicyRange] = []
        for span in spans:
            ranges.extend(self._project_span(span))
        return tuple(ranges)

    def _project_span(self, span: TextSpanAuthority) -> list[ScalarPolicyRange]:
        text_len = len(span.text)
        _validate_disjoint(
            span.grants, key=lambda g: g.char_ranges, text_len=text_len, label="grant"
        )
        stage_intervals = tuple((a, b) for a, b, _ in span.stage_ranges)
        _validate_disjoint(
            (span.stage_ranges,),
            key=lambda _: stage_intervals,
            text_len=text_len,
            label="stage_range",
        )

        cut_points: set[int] = {0, text_len}
        for grant in span.grants:
            for a, b in grant.char_ranges:
                cut_points.add(a)
                cut_points.add(b)
        for a, b, _ in span.stage_ranges:
            cut_points.add(a)
            cut_points.add(b)
        ordered = sorted(cut_points)

        atomic: list[ScalarPolicyRange] = []
        for a, b in zip(ordered, ordered[1:]):
            grant = _covering_grant(span.grants, a, b)
            if grant is None:
                continue  # out of scope for this fragment: not emitted, not guessed
            stage_enabled = _covering_stage(span.stage_ranges, a, b, span.default_stage_enabled)
            ownership_enabled = bool(grant.text_range_owner)
            effective, reasons = compute_effective(
                known_mask=grant.known_mask,
                observable_mask=grant.observable_mask,
                supervision_mask=grant.supervision_mask,
                ownership_enabled=ownership_enabled,
                stage_enabled=stage_enabled,
            )
            atomic.append(
                ScalarPolicyRange(
                    text_span_id=span.text_span_id,
                    node_id=span.node_id,
                    scalar_start=a,
                    scalar_end=b,
                    owner_fragment_id=grant.owner_fragment_id,
                    owner_scope=grant.owner_scope,
                    known_mask=grant.known_mask,
                    observable_mask=grant.observable_mask,
                    supervision_mask=grant.supervision_mask,
                    ownership_enabled=ownership_enabled,
                    stage_enabled=stage_enabled,
                    effective=effective,
                    reason_codes=reasons,
                    ownership=OwnershipGrant(
                        document_bundle_id=span.document_bundle_id,
                        target_bundle_hash=span.target_bundle_hash,
                        fragment_schema_version=grant.fragment_schema_version,
                        fragment_sequence_index=grant.fragment_sequence_index,
                        structure_owner=grant.structure_owner,
                        typed_target_owner=grant.typed_target_owner,
                        text_range_owner=grant.text_range_owner,
                        page_evidence=grant.page_evidence,
                        owned_char_ranges=grant.char_ranges,
                        ownership_projection_hash=grant.ownership_projection_hash,
                    ),
                )
            )
        return _merge_adjacent(atomic)


def _validate_disjoint(items: Sequence[Any], *, key, text_len: int, label: str) -> None:
    intervals: list[tuple[int, int]] = []
    for item in items:
        intervals.extend(key(item))
    intervals.sort()
    previous_end = 0
    for a, b in intervals:
        if a < 0 or b > text_len or a >= b:
            raise TextRangeBoundaryInvalid(
                f"{label} interval [{a}, {b}) is out of bounds for text length {text_len}"
            )
        if a < previous_end:
            raise TextRangeBoundaryInvalid(f"{label} intervals overlap at scalar index {a}")
        previous_end = b


def _covering_grant(grants: Sequence[TextRangeGrant], a: int, b: int) -> TextRangeGrant | None:
    # `_validate_disjoint` already rejects every overlapping or duplicate
    # grant interval before this runs, so at most one grant can ever cover
    # [a, b) -- no `len(covering) > 1` branch to guard an input that cannot
    # reach here (Karpathy #2: no error handling for an impossible scenario).
    covering = [g for g in grants if any(lo <= a and b <= hi for lo, hi in g.char_ranges)]
    return covering[0] if covering else None


def _covering_stage(
    stage_ranges: Sequence[tuple[int, int, bool]], a: int, b: int, default: bool
) -> bool:
    # Same reasoning as `_covering_grant`: `_validate_disjoint` already
    # rejects every overlapping or duplicate stage-range interval before
    # this runs, so no `len(covering) > 1` branch is reachable here either.
    covering = [enabled for lo, hi, enabled in stage_ranges if lo <= a and b <= hi]
    return covering[0] if covering else default


def _merge_adjacent(atomic: list[ScalarPolicyRange]) -> list[ScalarPolicyRange]:
    if not atomic:
        return []
    merged = [atomic[0]]
    for current in atomic[1:]:
        last = merged[-1]
        if (
            last.scalar_end == current.scalar_start
            and last.owner_fragment_id == current.owner_fragment_id
            and last.owner_scope == current.owner_scope
            and last.known_mask == current.known_mask
            and last.observable_mask == current.observable_mask
            and last.supervision_mask == current.supervision_mask
            and last.ownership_enabled == current.ownership_enabled
            and last.stage_enabled == current.stage_enabled
            and last.ownership == current.ownership
        ):
            merged[-1] = ScalarPolicyRange(
                text_span_id=last.text_span_id,
                node_id=last.node_id,
                scalar_start=last.scalar_start,
                scalar_end=current.scalar_end,
                owner_fragment_id=last.owner_fragment_id,
                owner_scope=last.owner_scope,
                known_mask=last.known_mask,
                observable_mask=last.observable_mask,
                supervision_mask=last.supervision_mask,
                ownership_enabled=last.ownership_enabled,
                stage_enabled=last.stage_enabled,
                effective=last.effective,
                reason_codes=last.reason_codes,
                ownership=last.ownership,
            )
        else:
            merged.append(current)
    return merged

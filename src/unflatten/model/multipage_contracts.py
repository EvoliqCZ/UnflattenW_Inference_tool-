"""RFC004 §31.3-31.4 boundary-candidate contracts and §48.3/§48.5 multi-page
document-mode coverage — code-only, dormant contracts.

RFC status (CLAUDE.md, 2026-09-27): multi-page and product claims remain
**PROPOSED/BLOCKED**. Nothing here is a learned boundary predictor, nothing
here emits a `Page` AST node (the grammar has none, RFC004 DoD 10:
"grammar nevie vytvoriť Page node"), and nothing here derives an explicit
`PageBreak` from a natural page boundary. This module only gives the
already-RATIFIED §31.3 boundary-candidate shape and the §48.3/§48.5
document-mode page-coverage rule a schema and a validator, plus a thin
transport-order-independence harness (§49.4) over the existing,
RATIFIED, deterministic Track A assembler (`unflatten.assembly.assemble`).
It authorizes no gradient update and predicts nothing.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from unflatten.assembly import assemble as asm
from unflatten.assembly import hypothesis as hy
from unflatten.assembly import policy as pol
from unflatten.model.hashing import domain_hash

#: §31.6-adjacent sentinel: the exact required `parent_local_node_id` of the
#: first `open_context` entry. Not a local node ID, not an AST node, and
#: forbidden anywhere else (§31.3).
FRAGMENT_ROOT_SENTINEL = "fragment_root"

BoundarySide = Literal["left", "right"]
SlotKind = Literal["opened_node", "continuation_proxy"]
BoundaryKind = Literal["start", "middle", "end"]

#: §31.3's exact right-boundary / left-boundary `boundary_kind` domains.
_ALLOWED_BOUNDARY_KIND: dict[BoundarySide, frozenset[str]] = {
    "right": frozenset({"start", "middle"}),
    "left": frozenset({"middle", "end"}),
}


class FragmentBoundaryError(ValueError):
    """A `BoundaryContextEntry` / `ContinuationCandidate` / boundary record
    violates RFC004 §31.3."""


@dataclass(frozen=True, slots=True)
class BoundaryContextEntry:
    """One outer→inner open-context frame (§31.3)."""

    depth: int
    node_type: str
    local_node_id: str
    slot_kind: SlotKind
    parent_local_node_id: str


@dataclass(frozen=True, slots=True)
class ContinuationCandidate:
    """One boundary continuation candidate (§31.3)."""

    local_node_id: str
    node_type: str
    boundary_kind: BoundaryKind
    stack_depth: int
    edge_text_nfc: str
    edge_text_scalar_count: int
    edge_text_hash: str | None
    nonoverlap_proof_hash: str | None
    structural_context_fingerprint: str
    visual_boundary_fingerprint: str
    confidence_ppm: int


@dataclass(frozen=True, slots=True)
class FragmentBoundary:
    """A `left_boundary` or `right_boundary` record (§31.2/§31.3)."""

    side: BoundarySide
    open_context: tuple[BoundaryContextEntry, ...]
    continuation_candidates: tuple[ContinuationCandidate, ...]
    stack_fingerprint: str


#: `domain_hash` appends the §3.1 NUL separator itself; the RFC004 domain
#: string is written WITH a trailing `\0` in prose, but passed here without
#: one (matching `hashing.domain_hash`'s own contract).
_BOUNDARY_STACK_DOMAIN = "rfc004.boundary-stack.v0"


def compute_stack_fingerprint(open_context: Sequence[BoundaryContextEntry]) -> str:
    """`stack_fingerprint = H("rfc004.boundary-stack.v0\\0", ordered
    [{depth, node_type}])` (§31.3). An empty stack hashes the empty array."""
    payload = [{"depth": e.depth, "node_type": e.node_type} for e in open_context]
    return domain_hash(_BOUNDARY_STACK_DOMAIN, payload)


def validate_boundary_context(
    open_context: Sequence[BoundaryContextEntry], *, side: BoundarySide
) -> None:
    """§31.3: `depth` equals the array index, the first entry's
    `parent_local_node_id` is the exact sentinel, every later entry's
    `parent_local_node_id` is the previous entry's `local_node_id`, and
    `left_boundary` slots are all `continuation_proxy`."""
    for index, entry in enumerate(open_context):
        if entry.depth != index:
            raise FragmentBoundaryError(
                f"open_context[{index}].depth={entry.depth} must equal its array index"
            )
        expected_parent = (
            FRAGMENT_ROOT_SENTINEL if index == 0 else open_context[index - 1].local_node_id
        )
        if entry.parent_local_node_id != expected_parent:
            raise FragmentBoundaryError(
                f"open_context[{index}].parent_local_node_id={entry.parent_local_node_id!r} "
                f"must be {expected_parent!r}"
            )
        if side == "left" and entry.slot_kind != "continuation_proxy":
            raise FragmentBoundaryError(
                f"left_boundary open_context[{index}].slot_kind must be 'continuation_proxy', "
                f"got {entry.slot_kind!r}"
            )


def validate_continuation_candidate(
    candidate: ContinuationCandidate,
    open_context: Sequence[BoundaryContextEntry],
    *,
    side: BoundarySide,
) -> None:
    """§31.3: `boundary_kind` domain by side, `stack_depth` indexes an entry
    of the same type, the NFC scalar count is exact, and the edge-text /
    nonoverlap-proof exclusivity holds."""
    allowed = _ALLOWED_BOUNDARY_KIND[side]
    if candidate.boundary_kind not in allowed:
        raise FragmentBoundaryError(
            f"{side}_boundary candidate boundary_kind={candidate.boundary_kind!r} "
            f"must be one of {sorted(allowed)}"
        )
    if not (0 <= candidate.stack_depth < len(open_context)):
        raise FragmentBoundaryError(
            f"stack_depth={candidate.stack_depth} does not index open_context "
            f"(length {len(open_context)})"
        )
    entry = open_context[candidate.stack_depth]
    if entry.node_type != candidate.node_type:
        raise FragmentBoundaryError(
            f"stack_depth={candidate.stack_depth} entry node_type={entry.node_type!r} != "
            f"candidate node_type={candidate.node_type!r}"
        )
    expected_count = len(unicodedata.normalize("NFC", candidate.edge_text_nfc))
    if candidate.edge_text_scalar_count != expected_count:
        raise FragmentBoundaryError(
            f"edge_text_scalar_count={candidate.edge_text_scalar_count} != "
            f"NFC scalar count {expected_count}"
        )
    if candidate.edge_text_nfc == "":
        if candidate.edge_text_scalar_count != 0:
            raise FragmentBoundaryError("empty edge_text_nfc must carry scalar_count 0")
        if candidate.nonoverlap_proof_hash is None:
            raise FragmentBoundaryError(
                "empty edge_text_nfc requires a non-null nonoverlap_proof_hash"
            )
    elif candidate.nonoverlap_proof_hash is not None:
        raise FragmentBoundaryError(
            "non-empty edge_text_nfc forbids a nonoverlap_proof_hash"
        )


def validate_fragment_boundary(boundary: FragmentBoundary) -> None:
    """Full §31.3 validation of one `FragmentBoundary` record."""
    validate_boundary_context(boundary.open_context, side=boundary.side)
    for candidate in boundary.continuation_candidates:
        validate_continuation_candidate(candidate, boundary.open_context, side=boundary.side)
    expected = compute_stack_fingerprint(boundary.open_context)
    if boundary.stack_fingerprint != expected:
        raise FragmentBoundaryError(
            f"stack_fingerprint={boundary.stack_fingerprint!r} != recomputed {expected!r}"
        )


def boundary_candidates_compatible(
    left_candidate: ContinuationCandidate,
    right_candidate: ContinuationCandidate,
    *,
    left_stack_prefix_types: Sequence[str],
    right_stack_prefix_types: Sequence[str],
) -> bool:
    """§31.3's right→left compatibility matrix plus its exact-equal
    requirements. `left_candidate` is a right_boundary candidate of the
    earlier fragment; `right_candidate` is a left_boundary candidate of the
    later fragment. Never uses gold node IDs, AST hashes or target event
    ranges (§31.3's last sentence) — every input here is candidate-local."""
    if left_candidate.boundary_kind not in _ALLOWED_BOUNDARY_KIND["right"]:
        return False
    if right_candidate.boundary_kind not in _ALLOWED_BOUNDARY_KIND["left"]:
        return False
    if left_candidate.node_type != right_candidate.node_type:
        return False
    if left_candidate.stack_depth != right_candidate.stack_depth:
        return False
    if (
        left_candidate.structural_context_fingerprint
        != right_candidate.structural_context_fingerprint
    ):
        return False
    if left_candidate.visual_boundary_fingerprint != right_candidate.visual_boundary_fingerprint:
        return False
    depth = left_candidate.stack_depth
    if tuple(left_stack_prefix_types[: depth + 1]) != tuple(right_stack_prefix_types[: depth + 1]):
        return False
    return _text_claims_match(left_candidate, right_candidate)


def _text_claims_match(left: ContinuationCandidate, right: ContinuationCandidate) -> bool:
    if left.edge_text_scalar_count == 0 and right.edge_text_scalar_count == 0:
        return (
            left.nonoverlap_proof_hash is not None
            and left.nonoverlap_proof_hash == right.nonoverlap_proof_hash
        )
    return (
        left.edge_text_nfc == right.edge_text_nfc
        and left.edge_text_scalar_count == right.edge_text_scalar_count
        and left.edge_text_hash == right.edge_text_hash
    )


# ---------------------------------------------------------------------------
# §48.3/§48.5 document-mode page coverage
# ---------------------------------------------------------------------------

DocumentMode = Literal["full_document", "bounded_multi_page"]
_FRAGMENTED_MODES: frozenset[str] = frozenset({"full_document", "bounded_multi_page"})


class DocumentModeCoverageError(ValueError):
    """A fragment set does not cover its document under the declared mode."""


@dataclass(frozen=True, slots=True)
class FragmentPageSpan:
    """The page ordinals one `FragmentHypothesis` covers (§31.2's
    `sequence_index`/`page_ordinals`), stripped to what page-coverage needs."""

    sequence_index: int
    page_ordinals: tuple[int, ...]


def validate_document_mode_page_coverage(
    mode: DocumentMode, fragments: Sequence[FragmentPageSpan]
) -> None:
    """§48.2/§48.3: `full_document` is one bounded hierarchical context (one
    fragment, contiguous from page 0); `bounded_multi_page` is ordered
    overlapping windows whose UNION must still cover every page 0..max with
    no gap (§31.4 lets `core`/`overlap` regions repeat evidence, but never
    lets a page go missing) -- both are Track B's model side, never derived
    from RFC003 gold ownership labels (§31.4)."""
    if mode not in _FRAGMENTED_MODES:
        raise DocumentModeCoverageError(
            f"mode must be one of {sorted(_FRAGMENTED_MODES)}, got {mode!r}"
        )
    if not fragments:
        raise DocumentModeCoverageError("no fragments to validate coverage over")
    ordered = sorted(fragments, key=lambda f: f.sequence_index)
    sequence_indices = [f.sequence_index for f in ordered]
    if len(set(sequence_indices)) != len(sequence_indices):
        raise DocumentModeCoverageError(f"duplicate sequence_index values: {sequence_indices}")
    if mode == "full_document" and len(fragments) != 1:
        raise DocumentModeCoverageError(
            f"full_document requires exactly one fragment, got {len(fragments)}"
        )
    covered = sorted({ordinal for f in ordered for ordinal in f.page_ordinals})
    if not covered:
        raise DocumentModeCoverageError("no fragment declares any page_ordinal")
    if covered[0] != 0:
        raise DocumentModeCoverageError(f"page coverage must start at 0, got {covered[0]}")
    gaps = [b for a, b in zip(covered, covered[1:]) if b != a + 1]
    if gaps:
        raise DocumentModeCoverageError(f"page coverage has a gap before page(s) {gaps}")


# ---------------------------------------------------------------------------
# §49.4 transport-order independence harness over the real Track A assembler
# ---------------------------------------------------------------------------


def assemble_transport_order_independent(
    pages: Sequence[hy.PageHypothesis], policy: pol.JoinPolicy, *, profile: str
) -> asm.AssembledDocumentHypothesis:
    """§49.4: "transport poradie fragment records nesmie meniť výsledok,
    ak ich sequence_index a contents sú rovnaké". `assemble` itself requires
    its input pre-sorted by `page_ordinal`; this is the sort a real
    transport-order-independent caller must do before calling it, so
    receiving pages in ANY order yields the same
    `AssembledDocumentHypothesis` as receiving them already ordered."""
    ordered = sorted(pages, key=lambda page: page.input.page_ordinal)
    return asm.assemble(ordered, policy, profile=profile)

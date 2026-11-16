"""M1 materialized training tensors: `TrainingTargetMaterializer` (design
§3, §4.6, §7, part of §8's tensor shapes).

Owns the design's four largest acceptance criteria (§16): criterion 7
(zero tokens cross a scalar owner/effective-mask boundary), criterion 8
(exactly one LossUnit/family per contribution, zero duplicate or unowned
locators), criterion 9 (the five-way `effective` conjunction is exact,
proved by an exhaustive prerequisite flip plus a synthetic no-model
autograd check), criterion 10 (different batch padding leaves unpadded
ids/modes/domains/LossUnits/decoded text unchanged).

The boundary-projection algorithm (`ScalarBoundaryProjector`,
`ScalarPolicyRange`, criterion 7) lives in `boundary_projector.py`, and the
LossUnit shape plus its fail-closed authority verification (criterion 8)
lives in `loss_unit.py` -- both split out because each split follows a
real seam where the *kind* of data changes (RFC003 authority ->
`ScalarPolicyRange` -> verified `LossUnit` -> tensors), which is what
makes the parts separately testable. Both are re-exported here so this
module remains the one callers/tests import for the full materializer
surface.

Not the exact-text cache writer (`exact_text_cache.py`, a separate task)
and does not populate `runtime_limits_policy`/`s1_run_manifest` binding
(`run_binding.py`, also separate). Ships no head, no `nn.Module`, no
weights, no optimizer -- only dataclasses and pure tensor construction.

Input contract: structural authority is the existing `structural`/
`fragment` dict shape already consumed by `unflatten.model.targets`/
`encode_fragment` (RFC003 `targets/events.py` events, fragment `slices`
from `dataset/fragments/envelope.py`); text authority is
`TextSpanAuthority` (`boundary_projector.py`), typed rather than a raw
dict like every sibling M1 module (`composite_grammar.py`,
`capability_profile.py`), one per RFC003 text span, carrying
`TextRangeGrant`s that mirror the real fragment-slice shape. RFC003's
mask projection (`targets/masks.py`) assesses typed-property pointers,
not Run text, so `observable_mask`/`supervision_mask` for text are
node-level scalars carried on the grant, exactly as real fragment slices
carry them; the projector still partitions at every `char_ranges` cut so
no token can straddle an owned/unowned boundary (design §6 step 3).

Never writes back to RFC003 state (hard constraint 6): every input above
is read, none is mutated, nothing under `src/unflatten/dataset/` is
imported for write.

**D1 bug fix (2026-09-07, audit `scratchpad/audit-exact-text/`).** A text
span can have non-empty text but zero owning `char_ranges` across every
grant (measured: 3,971 whitespace-only runs, 425/1,176 cache rows,
14,524 characters, `page_evidence="structural"`,
`supervision_mask=True`, `char_ranges=[]` -- the aligner assigned them no
owner at all). `ScalarBoundaryProjector` correctly emits zero `L_text`
atoms for such a span (no grant covers it), but its structural
TEXT_BEGIN/TEXT_END wrapper pair was never withdrawn, so it was the ONLY
signal at that position -- effective, unmasked, teaching the model that a
real run has empty text (13.4% of supervised text-or-end decisions after
TEXT_BEGIN in the `-c4` cohort; rendered effect: adjacent words fused with
no separating space). `unowned_text_event_origins` now identifies exactly
these text events (non-empty span text, no grant with any `char_ranges`)
from the same authority `materialize` was called with, and
`_structural_loss_unit` withdraws both wrapper positions with
`reason_codes=("text_unowned",)`. Rejected alternative: teaching
`ScalarBoundaryProjector` to emit a synthetic zero-length, always-withdrawn
`L_text` range for an unowned span instead -- rejected because it would
manufacture a `LossUnit`/locator for a range that has no authority at all
(`TextSpanAuthority`'s own contract: "Missing authority is not true"),
where withdrawing the existing structural wrapper unit needs no new unit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from unflatten.model import events_vocab, page_projection
from unflatten.model.boundary_projector import (
    OwnerScope,
    ScalarBoundaryProjector,
    ScalarPolicyRange,
    TextRangeBoundaryInvalid,
    TextRangeGrant,
    TextSpanAuthority,
    compute_effective,
)
from unflatten.model.capability_profile import STANDALONE_PAGE_PROFILE_IDS
from unflatten.model.composite_grammar import CompositeGrammarController
from unflatten.model.composite_vocab import (
    CompositeVocabulary,
    StructuralRef,
    TokenRef,
)
from unflatten.model.encode_fragment import (
    encode_fragment_with_origins,
    encode_standalone_fragment_with_origins,
)
from unflatten.model.exact_text import QwenTokenizerLike, encode_text
from unflatten.model.hashing import domain_hash
from unflatten.model.leader_projection import (
    CANONICAL_LEN,
    LEADER_CHARS,
    LEADER_POLICY_VERSION,
    MIN_RUN,
    canonicalize_leaders,
)
from unflatten.model.loss_unit import (
    TEXT_CONTINUATION_PRESENTATION,
    TEXT_UNOWNED,
    EffectiveLossMask,
    EventRangeLocator,
    Locator,
    LossFamily,
    LossUnit,
    LossUnitAuthorityMismatch,
    LossUnitBranchInvalid,
    Rfc003OwnershipProof,
    TextRangeLocator,
    assert_no_duplicate_or_unowned_locators,
    build_loss_unit,
    loss_unit_payload,
    verify_loss_unit_authority,
)
from unflatten.model.targets import present_supervised_event_indices, supervised_event_indices
from unflatten.model.v16_target_policy import projection_kwargs

__all__ = [
    "OwnerScope",
    "ScalarBoundaryProjector",
    "ScalarPolicyRange",
    "TextRangeBoundaryInvalid",
    "TextRangeGrant",
    "TextSpanAuthority",
    "compute_effective",
    "EffectiveLossMask",
    "EventRangeLocator",
    "Locator",
    "LossFamily",
    "LOSS_FAMILY_IDS",
    "LossUnit",
    "LossUnitAuthorityMismatch",
    "LossUnitBranchInvalid",
    "Rfc003OwnershipProof",
    "TextRangeLocator",
    "assert_no_duplicate_or_unowned_locators",
    "build_loss_unit",
    "loss_unit_payload",
    "verify_loss_unit_authority",
    "MaterializedSequence",
    "TrainingTargetMaterializer",
    "continuation_text_event_origins",
    "unowned_text_event_origins",
]

_MATERIALIZER_DOMAIN = "rfc005.training-target-materializer/v1"
_MATERIALIZER_IMPLEMENTATION_DOMAIN = "rfc005.training-target-materializer-impl/v1"

TRAINING_TARGET_MATERIALIZER_SCHEMA_VERSION = "1.1.0"
#: 1.0.0 -> 1.1.0 (2026-09-07, D1): `_structural_loss_unit` now withdraws
#: an unowned text event's wrapper pair (see module docstring). Bumping
#: `MATERIALIZER_VERSION` changes `materializer_implementation_hash` and
#: therefore `training_target_materializer_hash` / cache identity -- any
#: run manifest binding it must be re-frozen (RFC005 §29.6).
#: 1.1.0 -> 1.2.0 (2026-09-09): DEC-016 B's canonicalisation and the
#: typed-property channel share this bump, so the rebuild they both force
#: is one rebuild rather than two. The bump is a DECLARATION that the
#: materialized targets differ: `materializer_implementation_hash` reads
#: this string and nothing about behaviour, so a behaviour change left
#: without a bump passes it. What catches that direction is the
#: withdrawal decision digest pinned in
#: `tests/model/test_build_exact_text_cache.py`.
#: 1.2.0 -> 1.3.0 (2026-09-10, DEC-MP-3): `encode_fragment.py` now presents
#: a text continuation this page holds a `char_ranges` grant for even when
#: the page does not own the event's structure (multi-page text-projection
#: design §4.1). Reserved for exactly this change by DEC-022; a third
#: semantic change must not hide under a version string that already
#: declares two others (DEC-016 B and the typed channel share 1.2.0).
#: 1.3.0 -> 1.4.0 (2026-09-13, S3): `L_typed` joins
#: `SUPPORTED_LOSS_FAMILIES` (RFC005 §27.5): the typed property heads are
#: wired into the candidate model, so the §9.3 coverage axes read the
#: typed-property features as unmeasured, not as a derived zero. A
#: declaration, not a new materialised family: the token stream still packs
#: only L_structure / L_text atoms, so `LOSS_FAMILY_IDS` and the `LossFamily`
#: literal stay put and `loss_unit_schema_hash` (cache-identity field 19)
#: does not move.
MATERIALIZER_VERSION = "1.4.0"
#: DEC-033 / RFC003-ERRATA-002 (2026-09-17): `materialize` gains the
#: standalone-page structural path (`encode_standalone_fragment_with_
#: origins`, `present_supervised_event_indices`, no `TEXT_CONTINUATION_
#: PRESENTATION`), scoped to `capability_profile.STANDALONE_V1_PROFILE_ID`
#: exactly like `composite_grammar.py`'s DEC-033 bounds are scoped to it
#: (`CompositeGrammarController._dec033_bounds_active`). Every OTHER
#: profile must keep reproducing its v12 targets exactly, so
#: `materializer_implementation_hash`/`build_materializer_descriptor` read
#: this version only when `capability_profile_id ==
#: STANDALONE_V1_PROFILE_ID`; every other id (including `None`, the
#: pre-DEC-033 call shape) reads `MATERIALIZER_VERSION` unchanged, so
#: `training_target_materializer_hash` for those profiles does not move.
#: 2.0.1 (2026-09-18): per-story trailing drop, note floor, run-cap floor fix (`page_projection`).
#: 2.1.0 (2026-09-19): T12 form fill-leader canonicalization (`leader_projection`).
STANDALONE_MATERIALIZER_VERSION = "2.1.0"
BOUNDARY_PROJECTION_POLICY_VERSION = "1.0.0"

#: Design §4.6: M1 implements only `rfc003_supervised`.
SUPPORTED_LOSS_UNIT_KINDS: tuple[str, ...] = ("rfc003_supervised",)
#: Design §7/RR-1 variant (B): the family split is an M1 artifact
#: constraint bound by this tuple, not yet a ratified RFC005 taxonomy.
#: `L_typed` is declared at S3 (RFC005 §27.5) because the typed property
#: heads are wired into the candidate model; it is NOT packed in the token
#: stream -- typed supervision is carried by the heads, not by `LossUnit`
#: rows -- so it is absent from `_LOSS_FAMILY_ID` on purpose.
#: `L_alignment` (RFC005 §27.3) is declared the same way, pinned per DEC-043
#: §0 / `docs/plans/spec-node-visual-aligner-design.md` §9 BEFORE the v15
#: cache rebuild's bindings are produced: `stage_policy_hash` is a cache
#: identity key, so a `L_alignment` weight added after that rebuild would
#: force a second one. The `NodeVisualAligner` head is not wired yet -- the
#: family is declared, not packed, exactly like `L_typed` above.
SUPPORTED_LOSS_FAMILIES: tuple[str, ...] = (
    "L_structure",
    "L_text",
    "L_typed",
    "L_alignment",
)

type Mode = Literal["structure", "text", "byte", "terminated"]

_MODE_ID: dict[Mode, int] = {"structure": 0, "text": 1, "byte": 2, "terminated": 3}
_LOSS_FAMILY_ID: dict[LossFamily, int] = {"L_structure": 0, "L_text": 1}

#: Public alias for `_LOSS_FAMILY_ID`: `loss_family_ids` is stored as a
#: uint8 tensor, so a consumer reporting per-family numbers needs the same
#: mapping this module packed it with -- read from here, never restated.
LOSS_FAMILY_IDS = _LOSS_FAMILY_ID

# ---------------------------------------------------------------------------
# Structural token/origin extraction (§7: "Structural units reuse
# encode_fragment_with_origins"; document-level case has no such helper,
# so it is reimplemented here rather than modifying encode.py.)
# ---------------------------------------------------------------------------


def _document_level_tokens_with_origins(
    structural: dict[str, Any],
) -> tuple[list[str], list[int | None]]:
    """Mirrors `encode.encode_structural`, additionally tracking each
    token's origin document-event index (`None` for `END_DOCUMENT`, which
    owns no structural event -- exactly like `encode_fragment_with_origins`
    treats envelope-only tokens)."""
    tokens: list[str] = []
    origins: list[int | None] = []
    leaf_open_indices: set[int] = set()
    node_type_by_open_index: dict[int, str] = {}

    for event in structural["events"]:
        op = event["op"]
        index = event["index"]
        if op == "open":
            node_type = event["node_type"]
            node_type_by_open_index[index] = node_type
            if node_type in events_vocab.LEAF_TOKEN:
                leaf_open_indices.add(index)
                tokens.append(events_vocab.LEAF_TOKEN[node_type])
                origins.append(index)
            elif node_type in events_vocab.OPEN_TOKEN:
                tokens.append(events_vocab.OPEN_TOKEN[node_type])
                origins.append(index)
            else:
                raise ValueError(
                    f"node type {node_type!r} has no token in the event alphabet "
                    f"(event index {index})"
                )
        elif op == "text":
            tokens.extend([events_vocab.TEXT_BEGIN, events_vocab.TEXT_END])
            origins.extend([index, index])
        elif op == "close":
            open_index = event["open_index"]
            if open_index in leaf_open_indices:
                continue
            tokens.append(events_vocab.CLOSE_TOKEN[node_type_by_open_index[open_index]])
            origins.append(index)
        else:
            raise ValueError(f"unknown structural op {op!r}")

    tokens.append(events_vocab.END_DOCUMENT)
    origins.append(None)
    return tokens, origins


#: Public alias for `_document_level_tokens_with_origins` (2026-09-07):
#: `s1_runner` needs this exact document-level token/origin extraction and
#: had been reaching into the private name directly. The private name keeps
#: working for existing in-module callers/tests.
document_level_tokens_with_origins = _document_level_tokens_with_origins


def _node_id_by_origin(structural: dict[str, Any]) -> dict[int, str]:
    return {event["index"]: event["node_id"] for event in structural["events"]}


def _text_span_origin(structural: dict[str, Any]) -> dict[str, int]:
    return {
        event["text_span_id"]: event["index"]
        for event in structural["events"]
        if event["op"] == "text"
    }


def unowned_text_event_origins(
    structural: dict[str, Any], text_spans: Sequence[TextSpanAuthority]
) -> frozenset[int]:
    """Document-event indices of every `op == "text"` event whose span has
    non-empty text but no grant with any `char_ranges` at all (D1, module
    docstring). Public so `s1_runner.composite_expected_missing_mask`
    mirrors this exact rule from the same source authority, rather than
    reading it back off already-conjoined `LossUnit`/mask output.

    A span with zero owning `char_ranges` gets zero `L_text` atoms from
    `ScalarBoundaryProjector` (no grant covers `[0, len(text))`), so its
    structural TEXT_BEGIN/TEXT_END wrapper pair -- identified here by
    document-event index -- is the only place withdrawal can attach.
    """
    span_by_id = {span.text_span_id: span for span in text_spans}
    origins: set[int] = set()
    for event in structural["events"]:
        if event["op"] != "text":
            continue
        span = span_by_id.get(event.get("text_span_id"))
        if span is None or not span.text:
            continue
        if any(grant.char_ranges for grant in span.grants):
            continue
        origins.add(event["index"])
    return frozenset(origins)


def continuation_text_event_origins(
    structural: dict[str, Any], text_spans: Sequence[TextSpanAuthority]
) -> frozenset[int]:
    """Document-event indices of every `op == "text"` event whose span is
    *locally granted* -- has at least one grant with non-empty `char_ranges`
    on this page (design DEC-MP-1, §12.6). `_structural_loss_unit` appends
    `TEXT_CONTINUATION_PRESENTATION` next to `not_supervised` for exactly
    these origins: `origin not in supervised AND origin is a locally-granted
    text event`. Public for the same reason as `unowned_text_event_origins`
    -- so a mirror can read the same rule from the raw authority instead of
    reverse-engineering it from `LossUnit` output.
    """
    span_by_id = {span.text_span_id: span for span in text_spans}
    origins: set[int] = set()
    for event in structural["events"]:
        if event["op"] != "text":
            continue
        span = span_by_id.get(event.get("text_span_id"))
        if span is None:
            continue
        if any(grant.char_ranges for grant in span.grants):
            origins.add(event["index"])
    return frozenset(origins)


# ---------------------------------------------------------------------------
# Materialized sequence (design §3 MaterializedSequence)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MaterializedSequence:
    packed_token_ids: torch.Tensor  # int64 [T]
    packed_input_ids: torch.Tensor  # int64 [T]
    effective_loss_mask: torch.Tensor  # bool [T]
    loss_family_ids: torch.Tensor  # uint8 [T]
    controller_mode_ids: torch.Tensor  # uint8 [T]
    structural_allowed_row_ptr: torch.Tensor  # int32 [T+1]
    structural_allowed_local_ids: torch.Tensor  # uint8 [N]
    loss_unit_index: torch.Tensor  # int32 [T]
    loss_units: tuple[LossUnit, ...]


@dataclass(frozen=True, slots=True)
class TrainingTargetMaterializer:
    """Design §3 `TrainingTargetMaterializer` -- bound to one vocabulary,
    one composite grammar controller, and one capability profile."""

    vocabulary: CompositeVocabulary
    controller: CompositeGrammarController
    composite_grammar_hash: str
    capability_profile_id: str
    capability_profile_hash: str
    text_encoding_budget_ops: int
    tokenizer: QwenTokenizerLike
    materializer_hash: str
    head_registry_hash: str
    #: `v16_target_policy.TargetPolicy` or `None` (every pre-v16 caller). Only
    #: `withdrawn_run_collapse` is read here (the internal `project` call);
    #: V1a grants are applied by the caller to `text_spans` BEFORE
    #: `materialize`, because the caller's own `project` needs them too.
    target_policy: Any = None

    def materialize(
        self,
        *,
        document_bundle_id: str,
        target_bundle_hash: str,
        structural: dict[str, Any],
        fragment: dict[str, Any] | None,
        text_spans: Sequence[TextSpanAuthority],
        mask_projection_hash: str,
        structural_ownership_projection_hash: str,
        structural_projection_hash: str,
        pad_to: int | None = None,
        dropped_node_ids: frozenset[str] | None = None,
        rehome: Mapping[str, str] | None = None,
    ) -> MaterializedSequence:
        """Materialize one page/document sample into a `MaterializedSequence`.

        `structural_ownership_projection_hash`/`structural_projection_hash`
        stand in for the RFC003 structural ownership/derivation artifacts:
        M1 has no separate hash for those distinct from the target bundle
        and fragment identity themselves, so the caller supplies the
        binding value it wants pinned (design §4.6's requirement is that
        *some* hash is bound and verified, not that a new artifact exists).

        `dropped_node_ids`, under `STANDALONE_V1_PROFILE_ID` only: the
        caller's own `page_projection.project(fragment, structural,
        text_spans)` result, given `None` means "compute it here" (every
        existing caller). `row_for_page` (T6 wiring) computes it ONCE and
        passes it in, so the ledger it derives from that one call and the
        set this function withdraws by both come from the identical
        projection -- never two separately-computed sets that could drift.

        `rehome` (`v16-targets.v1` V1d, standalone profiles only): the
        caller's `v16_target_policy.rehome_inherited_stories` map, passed to
        the encoder with the story-only sections already in
        `dropped_node_ids`; `None` (every other caller) emits nothing moved.
        """
        # DEC-033 §5/§6 (design §4): the page-normalization projection runs
        # BEFORE any token is emitted, once, so the encoder and the
        # supervision reading agree on exactly which nodes this page
        # drops -- a dropped node is absent from both, never masked.
        resolved_dropped_node_ids: frozenset[str] = frozenset()
        if fragment is not None and self.capability_profile_id in STANDALONE_PAGE_PROFILE_IDS:
            if dropped_node_ids is not None:
                resolved_dropped_node_ids = dropped_node_ids
            else:
                resolved_dropped_node_ids, _ledger_row = page_projection.project(
                    fragment, structural, text_spans, **projection_kwargs(self.target_policy)
                )
        dropped_node_ids = resolved_dropped_node_ids

        (
            composite_atoms,
            node_id_by_origin,
            fragment_id,
            fragment_schema_version,
            fragment_sequence_index,
        ) = self._structural_atoms(
            structural, fragment, dropped_node_ids=dropped_node_ids, rehome=rehome
        )
        policy_ranges = ScalarBoundaryProjector().project(text_spans)
        composite_atoms = self._interleave_text(
            composite_atoms, structural, text_spans, policy_ranges
        )

        if fragment is None:
            supervised = None
        elif self.capability_profile_id in STANDALONE_PAGE_PROFILE_IDS:
            # RFC003-ERRATA-002 E1: supervision is presence-and-observability
            # on THIS page, not page-of-owner-only.
            supervised = present_supervised_event_indices(
                fragment, structural, dropped_node_ids=dropped_node_ids
            )
        else:
            supervised = supervised_event_indices(fragment)
        unowned_text_origins = unowned_text_event_origins(structural, text_spans)
        continuation_text_origins = continuation_text_event_origins(structural, text_spans)

        state = self.controller.fresh(self.capability_profile_id)
        packed_ids: list[int] = []
        modes: list[int] = []
        row_ptr: list[int] = [0]
        allowed_local_ids: list[int] = []
        loss_units: list[LossUnit] = []
        mask: list[bool] = []
        family_ids: list[int] = []

        for i, atom in enumerate(composite_atoms):
            domain = self.controller.allowed(state)
            modes.append(_MODE_ID[domain.mode])
            allowed_local_ids.extend(domain.structural_local_ids)
            row_ptr.append(len(allowed_local_ids))

            packed_ids.append(self.vocabulary.pack(atom.token))
            family_ids.append(_LOSS_FAMILY_ID[atom.loss_family])

            if atom.loss_family == "L_structure":
                unit, effective = self._structural_loss_unit(
                    index=i,
                    atom=atom,
                    fragment=fragment,
                    fragment_id=fragment_id,
                    fragment_schema_version=fragment_schema_version,
                    fragment_sequence_index=fragment_sequence_index,
                    node_id_by_origin=node_id_by_origin,
                    supervised=supervised,
                    unowned_text_origins=unowned_text_origins,
                    continuation_text_origins=continuation_text_origins,
                    document_bundle_id=document_bundle_id,
                    target_bundle_hash=target_bundle_hash,
                    mask_projection_hash=mask_projection_hash,
                    structural_projection_hash=structural_projection_hash,
                    structural_ownership_projection_hash=structural_ownership_projection_hash,
                )
            else:
                if atom.policy_range_index is None:
                    raise ValueError(
                        f"atom at index {i} has loss_family {atom.loss_family!r} but no "
                        "policy_range_index -- every non-L_structure atom must carry the "
                        "text-authority range it was projected from"
                    )
                policy_range = policy_ranges[atom.policy_range_index]
                unit = self._text_loss_unit(
                    index=i, policy_range=policy_range, mask_projection_hash=mask_projection_hash
                )
                effective = policy_range.effective

            loss_units.append(unit)
            mask.append(effective)
            state = self.controller.step(state, atom.token)

        assert_no_duplicate_or_unowned_locators(loss_units)

        bos_local_id = events_vocab.TOKEN_IDS[events_vocab.END_DOCUMENT]
        bos_id = self.vocabulary.pack(StructuralRef(local_id=bos_local_id))
        input_ids = [bos_id, *packed_ids[:-1]]

        seq_len = len(packed_ids)
        total_len = pad_to if pad_to is not None else seq_len
        if total_len < seq_len:
            raise ValueError(f"pad_to {total_len} is shorter than the {seq_len}-token sequence")
        pad_count = total_len - seq_len

        return MaterializedSequence(
            packed_token_ids=_padded(packed_ids, pad_count, bos_id, torch.int64),
            packed_input_ids=_padded(input_ids, pad_count, bos_id, torch.int64),
            effective_loss_mask=_padded(mask, pad_count, False, torch.bool),
            loss_family_ids=_padded(family_ids, pad_count, 0, torch.uint8),
            controller_mode_ids=_padded(modes, pad_count, _MODE_ID["terminated"], torch.uint8),
            structural_allowed_row_ptr=_padded(
                row_ptr, pad_count, row_ptr[-1], torch.int32
            ),
            structural_allowed_local_ids=torch.tensor(allowed_local_ids, dtype=torch.uint8)
            if allowed_local_ids
            else torch.zeros(0, dtype=torch.uint8),
            loss_unit_index=_padded(list(range(seq_len)), pad_count, -1, torch.int32),
            loss_units=tuple(loss_units),
        )

    def _structural_loss_unit(
        self,
        *,
        index: int,
        atom: _Atom,
        fragment: dict[str, Any] | None,
        fragment_id: str | None,
        fragment_schema_version: str | None,
        fragment_sequence_index: int | None,
        node_id_by_origin: dict[int, str],
        supervised: set[int] | None,
        unowned_text_origins: frozenset[int],
        continuation_text_origins: frozenset[int],
        document_bundle_id: str,
        target_bundle_hash: str,
        mask_projection_hash: str,
        structural_projection_hash: str,
        structural_ownership_projection_hash: str,
    ) -> tuple[LossUnit, bool]:
        origin = atom.source_event_index
        supervised_here = origin is None or (supervised is None or origin in supervised)
        ownership_enabled = True
        if not supervised_here:
            # `not_supervised` keeps precedence over the unowned-text check
            # below: an already-withdrawn unit's payload (7,967 measured on
            # `-c4`) must stay byte-identical, so its `loss_unit_id` does
            # not change under this fix.
            effective = False
            reasons = ("not_supervised",)
            # Appended, not substituted (DEC-MP-1): a continuation delimiter
            # is already withdrawn by `not_supervised` -- the origin's text
            # event has no `structural_event_ranges` on this page -- but it
            # is also a locally-granted text event (the page presents text
            # it does not structurally own). Naming that second, narrower
            # reason is what makes the withdrawal countable instead of
            # indistinguishable from any other unresolved placement. A
            # single-page document never satisfies both conjuncts, so
            # `continuation_text_origins` is empty there and this branch is
            # unreached (T-10).
            #
            # Unreachable under `STANDALONE_V1_PROFILE_ID` by construction,
            # not just by data: `encode_standalone_fragment_with_origins`
            # never presents a node's text without also owning its
            # structure (RFC003-ERRATA-002 E1 -- "presented but not owned"
            # is a `RESUME_NODE` continuation-window concept this profile
            # does not have), so the second reason code would be dead
            # weight on every withdrawn unit this profile ever produces.
            if (
                self.capability_profile_id not in STANDALONE_PAGE_PROFILE_IDS
                and origin is not None
                and origin in continuation_text_origins
            ):
                reasons = (*reasons, TEXT_CONTINUATION_PRESENTATION)
        elif origin is not None and origin in unowned_text_origins:
            effective = False
            ownership_enabled = False
            reasons = (TEXT_UNOWNED,)
        else:
            effective = True
            reasons = ()
        node_id = "envelope" if origin is None else node_id_by_origin[origin]
        event_start = origin if origin is not None else index
        unit = build_loss_unit(
            output_locator=f"token/{index}",
            loss_family="L_structure",
            stage_enabled=True,
            owner_scope="fragment_local" if fragment is not None else "global_target",
            owner_fragment_id=fragment_id,
            locator=EventRangeLocator(event_start=event_start, event_end=event_start + 1),
            authority_target_hash=target_bundle_hash,
            projection_hash=structural_projection_hash,
            mask_projection_hash=mask_projection_hash,
            ownership_proof=Rfc003OwnershipProof(
                document_bundle_id=document_bundle_id,
                target_bundle_hash=target_bundle_hash,
                fragment_id=fragment_id,
                fragment_schema_version=fragment_schema_version,
                fragment_sequence_index=fragment_sequence_index,
                node_id=node_id,
                structure_owner=None,
                typed_target_owner=None,
                text_range_owner=None,
                page_evidence=None,
                owned_char_ranges=(),
                ownership_projection_hash=structural_ownership_projection_hash,
            ),
            effective_mask=EffectiveLossMask(
                mask_projection_hash=mask_projection_hash,
                known_mask=True,
                observable_mask=True,
                supervision_mask=supervised_here,
                ownership_enabled=ownership_enabled,
                stage_enabled=True,
                effective=effective,
                reason_codes=reasons,
            ),
            composite_vocabulary_hash=self.vocabulary.artifact_hash,
            composite_grammar_hash=self.composite_grammar_hash,
            capability_profile_hash=self.capability_profile_hash,
            head_registry_hash=self.head_registry_hash,
        )
        return unit, effective

    def _text_loss_unit(
        self, *, index: int, policy_range: ScalarPolicyRange, mask_projection_hash: str
    ) -> LossUnit:
        ownership = policy_range.ownership
        return build_loss_unit(
            output_locator=f"token/{index}",
            loss_family="L_text",
            stage_enabled=policy_range.stage_enabled,
            owner_scope=policy_range.owner_scope,
            owner_fragment_id=policy_range.owner_fragment_id,
            locator=TextRangeLocator(
                text_span_id=policy_range.text_span_id,
                scalar_start=policy_range.scalar_start,
                scalar_end=policy_range.scalar_end,
            ),
            authority_target_hash=ownership.target_bundle_hash,
            projection_hash=ownership.ownership_projection_hash,
            mask_projection_hash=mask_projection_hash,
            ownership_proof=Rfc003OwnershipProof(
                document_bundle_id=ownership.document_bundle_id,
                target_bundle_hash=ownership.target_bundle_hash,
                fragment_id=policy_range.owner_fragment_id,
                fragment_schema_version=ownership.fragment_schema_version,
                fragment_sequence_index=ownership.fragment_sequence_index,
                node_id=policy_range.node_id,
                structure_owner=ownership.structure_owner,
                typed_target_owner=ownership.typed_target_owner,
                text_range_owner=ownership.text_range_owner,
                page_evidence=ownership.page_evidence,
                owned_char_ranges=ownership.owned_char_ranges,
                ownership_projection_hash=ownership.ownership_projection_hash,
            ),
            effective_mask=EffectiveLossMask(
                mask_projection_hash=mask_projection_hash,
                known_mask=policy_range.known_mask,
                observable_mask=policy_range.observable_mask,
                supervision_mask=policy_range.supervision_mask,
                ownership_enabled=policy_range.ownership_enabled,
                stage_enabled=policy_range.stage_enabled,
                effective=policy_range.effective,
                reason_codes=policy_range.reason_codes,
            ),
            composite_vocabulary_hash=self.vocabulary.artifact_hash,
            composite_grammar_hash=self.composite_grammar_hash,
            capability_profile_hash=self.capability_profile_hash,
            head_registry_hash=self.head_registry_hash,
        )

    def _structural_atoms(
        self,
        structural: dict[str, Any],
        fragment: dict[str, Any] | None,
        *,
        dropped_node_ids: frozenset[str] = frozenset(),
        rehome: Mapping[str, str] | None = None,
    ) -> tuple[list[_Atom], dict[int, str], str | None, str | None, int | None]:
        if fragment is not None:
            if self.capability_profile_id in STANDALONE_PAGE_PROFILE_IDS:
                tokens, origins = encode_standalone_fragment_with_origins(
                    fragment, structural, dropped_node_ids=dropped_node_ids, rehome=rehome
                )
            else:
                tokens, origins = encode_fragment_with_origins(fragment, structural)
            fragment_id = fragment.get("fragment_id")
            fragment_schema_version = fragment.get("fragment_schema_version")
            fragment_sequence_index = fragment.get("sequence_index")
        else:
            tokens, origins = _document_level_tokens_with_origins(structural)
            fragment_id = None
            fragment_schema_version = None
            fragment_sequence_index = None

        node_id_by_origin = _node_id_by_origin(structural)
        atoms = [
            _Atom(
                token=StructuralRef(local_id=events_vocab.TOKEN_IDS[name]),
                loss_family="L_structure",
                source_event_index=origin,
                policy_range_index=None,
            )
            for name, origin in zip(tokens, origins)
        ]
        return (
            atoms,
            node_id_by_origin,
            fragment_id,
            fragment_schema_version,
            fragment_sequence_index,
        )

    def _interleave_text(
        self,
        atoms: list[_Atom],
        structural: dict[str, Any],
        text_spans: Sequence[TextSpanAuthority],
        policy_ranges: tuple[ScalarPolicyRange, ...],
    ) -> list[_Atom]:
        span_by_id = {span.text_span_id: span for span in text_spans}
        text_span_origin = _text_span_origin(structural)
        ranges_by_span: dict[str, list[int]] = {}
        for idx, pr in enumerate(policy_ranges):
            ranges_by_span.setdefault(pr.text_span_id, []).append(idx)

        result: list[_Atom] = []
        i = 0
        text_begin_local = events_vocab.TOKEN_IDS[events_vocab.TEXT_BEGIN]
        while i < len(atoms):
            atom = atoms[i]
            result.append(atom)
            is_text_begin = (
                isinstance(atom.token, StructuralRef) and atom.token.local_id == text_begin_local
            )
            if is_text_begin and atom.source_event_index is not None:
                origin = atom.source_event_index
                span_id = next(
                    (sid for sid, o in text_span_origin.items() if o == origin), None
                )
                if span_id is not None and span_id in span_by_id:
                    span = span_by_id[span_id]
                    for range_idx in ranges_by_span.get(span_id, []):
                        pr = policy_ranges[range_idx]
                        range_text = span.text[pr.scalar_start : pr.scalar_end]
                        if self.capability_profile_id in STANDALONE_PAGE_PROFILE_IDS:
                            range_text, _runs, _removed = canonicalize_leaders(range_text)
                        tokens = encode_text(
                            range_text,
                            vocabulary=self.vocabulary,
                            tokenizer=self.tokenizer,
                            budget_ops=self.text_encoding_budget_ops,
                        )
                        for token in tokens:
                            result.append(
                                _Atom(
                                    token=token,
                                    loss_family="L_text",
                                    source_event_index=None,
                                    policy_range_index=range_idx,
                                )
                            )
                # next atom in `atoms` must be the matching TEXT_END; leave
                # it to the outer loop to append unchanged.
            i += 1
        return result


@dataclass(frozen=True, slots=True)
class _Atom:
    token: TokenRef
    loss_family: LossFamily
    source_event_index: int | None
    policy_range_index: int | None


def _padded(values: list[Any], pad_count: int, fill: Any, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values + [fill] * pad_count, dtype=dtype)


# ---------------------------------------------------------------------------
# Materializer descriptor artifact (design §4.6)
# ---------------------------------------------------------------------------


def _materializer_version(capability_profile_id: str | None) -> str:
    """`STANDALONE_MATERIALIZER_VERSION` under DEC-033's profile, else the
    unchanged `MATERIALIZER_VERSION` every other profile (and the
    pre-DEC-033 no-argument call shape) has always read."""
    if capability_profile_id in STANDALONE_PAGE_PROFILE_IDS:
        return STANDALONE_MATERIALIZER_VERSION
    return MATERIALIZER_VERSION


def materializer_implementation_hash(capability_profile_id: str | None = None) -> str:
    return domain_hash(
        _MATERIALIZER_IMPLEMENTATION_DOMAIN,
        {"materializer_version": _materializer_version(capability_profile_id)},
    )


#: Declared only by a descriptor that carries `target_policy` (schema 1.2.0);
#: every other descriptor keeps `TRAINING_TARGET_MATERIALIZER_SCHEMA_VERSION`
#: so no pre-v16 `training_target_materializer_hash` moves.
TARGET_POLICY_MATERIALIZER_SCHEMA_VERSION = "1.2.0"


def build_materializer_descriptor(
    *,
    loss_unit_schema_hash: str,
    composite_vocabulary_hash: str,
    composite_grammar_hash: str,
    capability_profile_hash: str,
    capability_profile_id: str | None = None,
    target_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """`target_policy` is `v16_target_policy.TargetPolicy.descriptor()`
    (DEC-055 V1a/V1b/V1c + audit exclusions/spill): it projects the target
    through `page_projection`, which only the standalone-page profiles run,
    so any other profile is refused rather than silently ignoring it."""
    if target_policy is not None and capability_profile_id not in STANDALONE_PAGE_PROFILE_IDS:
        raise ValueError(
            f"target_policy {target_policy.get('policy_id')!r} needs a standalone-page "
            f"capability profile, got {capability_profile_id!r}"
        )
    result: dict[str, Any] = {
        "training_target_materializer_schema_version": TRAINING_TARGET_MATERIALIZER_SCHEMA_VERSION,
        "materializer_version": _materializer_version(capability_profile_id),
        "implementation_hash": materializer_implementation_hash(capability_profile_id),
        "loss_unit_schema_hash": loss_unit_schema_hash,
        "boundary_projection_policy_version": BOUNDARY_PROJECTION_POLICY_VERSION,
        "composite_vocabulary_hash": composite_vocabulary_hash,
        "composite_grammar_hash": composite_grammar_hash,
        "capability_profile_hash": capability_profile_hash,
        "supported_loss_unit_kinds": list(SUPPORTED_LOSS_UNIT_KINDS),
        "supported_loss_families": list(SUPPORTED_LOSS_FAMILIES),
    }
    if capability_profile_id in STANDALONE_PAGE_PROFILE_IDS:
        result["leader_projection"] = {
            "policy_version": LEADER_POLICY_VERSION,
            "chars": sorted(LEADER_CHARS),
            "min_run": MIN_RUN,
            "canonical_len": CANONICAL_LEN,
        }
    if target_policy is not None:
        result["training_target_materializer_schema_version"] = (
            TARGET_POLICY_MATERIALIZER_SCHEMA_VERSION
        )
        result["target_policy"] = dict(target_policy)
    return result


def materializer_descriptor_hash(payload: dict[str, Any]) -> str:
    return domain_hash(_MATERIALIZER_DOMAIN, payload)

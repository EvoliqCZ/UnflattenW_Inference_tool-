"""M1 composite controller: wrapper state machine over TokenRef streams
(design §3, §5).

This module wraps -- never forks -- the existing structural pushdown
grammar (:mod:`unflatten.model.grammar`, unchanged per design §11) with
two additional modes for Qwen text and byte-fallback frames, so a single
immutable ``allowed``/``step`` controller governs the whole composite
alphabet: :class:`~unflatten.model.composite_vocab.StructuralRef`,
:class:`~unflatten.model.composite_vocab.QwenRef`,
:class:`~unflatten.model.composite_vocab.ByteControlRef` and
:class:`~unflatten.model.composite_vocab.ByteValueRef`.

State machine (design §5)::

    structure --Structural(TEXT_BEGIN)--> text
    text --Qwen(allowlisted)--> text
    text --BYTE_BEGIN--> byte
    byte --BYTE_00..BYTE_FF (valid UTF-8 prefix)--> byte
    byte --BYTE_END (exactly one scalar)--> text
    text --Structural(TEXT_END)--> structure
    structure --accepted END_DOCUMENT/END_FRAGMENT--> terminated

Qwen and byte-frame tokens never enter :func:`grammar.step`; only
:class:`~unflatten.model.composite_vocab.StructuralRef` tokens do (design
§3: "Qwen/byte payload tokens never enter grammar.step; existing
structural TEXT_BEGIN/TEXT_END do").

Legal domains are compact by construction (design §16 criterion 6's
sibling requirement, "compact legal domains" in the task brief): the
structure mode exposes at most 71 structural local ids, the text mode
exposes a single vocabulary hash pointer rather than materializing the
~152k-entry Qwen allowlist, and the byte mode exposes a handful of
half-open `[lo, hi)` byte-value ranges plus one `BYTE_END` boolean, never
a per-value enumeration of up to 256 ids. No
domain ever enumerates the full composite vocabulary.

One rule here is a BOUND, not grammar: ``MAX_CONSECUTIVE_EMPTY_PARAGRAPHS``
(DEC-016 variant C) caps how many bare empty paragraphs may close
back-to-back, because greedy decode over the gold distribution otherwise
emits them until it runs out of steps. It lives in the grammar state (never
in a decode-loop counter) so ``allowed``, ``step`` and every replay of a
stream see the same legality. It bounds one uninterrupted RUN, not a
decode: three ways a decode escapes it are reproduced and recorded in
``tests/model/test_composite_grammar_empty_paragraph_bound.py``. It treats
a symptom -- the cure is DEC-016 variant B, which changes the supervision
itself.

``CompositeGrammarState`` deviates from the design's illustrative
``byte_payload_length: int`` field by storing the actual byte prefix
collected so far (at most 4 bytes) rather than only its length. A bare
length cannot support design §5's "invalid prefixes, overlong encodings,
surrogates, and values above U+10FFFF are rejected incrementally": the
legal continuation range for the second byte of a 3- or 4-byte sequence
depends on the *value* of the leading byte (e.g. ``E0`` requires
``A0..BF`` to exclude overlong forms; ``ED`` requires ``80..9F`` to
exclude surrogates), which only the byte values themselves carry. The
stored prefix is still tiny (<=4 ints) and is never exposed outside this
module's compact ``AllowedDomain``.

``CompositeGrammarController.fresh`` takes a ``capability_profile_id``
exactly as the design's Protocol (§3) specifies, but the controller
itself is constructed already bound to one profile id, one structural
(AST) profile and one set of capability-excluded structural local ids --
mirroring design §4.4's "one explicit S1 capability profile, never
DEFAULT_PROFILE". ``fresh`` therefore validates that the caller's stated
id matches the controller's bound id (fail-closed identity check) rather
than dynamically resolving an id to a profile; the schema-backed
resolution of a ``model_capability_profile`` artifact into a structural
profile and an excluded-node-type set is `capability_profile.py`'s job
(a separate, not-yet-built task) and is out of scope here.

M1-internal exceptions here are CamelCase and are not RFC004 registry
codes (design §9.1, §9.3) -- no unratified code may cross into an Issue
envelope or a `primary_code` until RR-3 is ratified. `ByteFrameInvalid`
is imported from :mod:`unflatten.model.exact_text` rather than redefined,
so one condition (design §9.2's proposed `BYTE_FRAME_INVALID`) maps to
exactly one exception class across both modules.

This module authorizes no gradient update: it declares only dataclasses,
a Literal alias and pure functions/methods -- no `nn.Module`, no weights,
no optimizer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import jsonschema.protocols
import jsonschema.validators

from unflatten.model import events_vocab, grammar
from unflatten.model.capability_profile import (
    STANDALONE_PAGE_PROFILE_IDS,
    CapabilityProfile,
)
from unflatten.model.composite_vocab import (
    ByteControlRef,
    ByteValueRef,
    CompositeTokenIdInvalid,
    CompositeVocabulary,
    QwenRef,
    StructuralRef,
    TokenRef,
)
from unflatten.model.exact_text import ByteFrameInvalid
from unflatten.model.hashing import domain_hash
from unflatten.model.runtime_limits import budget_policy_hash

type Mode = Literal["structure", "text", "byte", "terminated"]

#: Design §4.9 registry row `rfc004.composite-grammar/v1`. Distinct from
#: `rfc004.grammar-controller/v1` (`grammar.py`), which hashes the base
#: structural controller this descriptor *includes* unchanged.
_COMPOSITE_GRAMMAR_DOMAIN = "rfc004.composite-grammar/v1"

COMPOSITE_GRAMMAR_SCHEMA_VERSION = "1.2.0"

#: The pre-DEC-033 schema shape (no `max_consecutive_tabs`), kept verbatim
#: for every capability profile other than `STANDALONE_V1_PROFILE_ID` so
#: those descriptors stay byte-identical to the pre-DEC-033 ones
#: (RFC003-ERRATA-002 / RFC004-ERRATA-002's promise that old profiles keep
#: reproducing their v12 targets, which v12 bindings/caches pin by
#: `composite_grammar_hash`). See "Scope" above `MAX_CONSECUTIVE_EMPTY_
#: PARAGRAPHS`.
_LEGACY_COMPOSITE_GRAMMAR_SCHEMA_VERSION = "1.1.0"

#: The version of the `structure <-> text <-> byte` transition policy this
#: module implements (design §5's state machine).
MODE_TRANSITION_POLICY_VERSION = "1.0.0"

#: The version of the byte-frame policy this module implements: the
#: `BYTE_BEGIN / BYTE_VALUE* / BYTE_END` frame and its incremental
#: shortest-form UTF-8 acceptance (design §5).
BYTE_FRAME_POLICY_VERSION = "1.0.0"

#: The version of the empty-paragraph bound this module implements. Moves
#: whenever the *meaning* of the bound moves; the bound's VALUE travels
#: separately in the descriptor, so raising the number alone still changes
#: `composite_grammar_hash`.
EMPTY_PARAGRAPH_POLICY_VERSION = "2.0.0"

#: DEC-016 variant C's original policy version, kept verbatim for every
#: profile other than `STANDALONE_V1_PROFILE_ID` -- same reproducibility
#: reason as `_LEGACY_COMPOSITE_GRAMMAR_SCHEMA_VERSION` above.
_LEGACY_EMPTY_PARAGRAPH_POLICY_VERSION = "1.0.0"

#: DEC-033 / RFC003-ERRATA-002 / RFC004-ERRATA-002 (R3), superseding
#: DEC-016 variant C: a BOUND on one run of empty paragraphs. It is not a
#: fix for the cause, and it is not a guarantee that a decode terminates
#: (see "what this bound does NOT do" below).
#:
#: DEC-016 variant C bounded shape A only -- a bare `PARAGRAPH_OPEN`
#: immediately followed by `PARAGRAPH_CLOSE` -- because shape B
#: (`PARAGRAPH_OPEN, RUN_OPEN, TEXT_BEGIN, TEXT_END, RUN_CLOSE,
#: PARAGRAPH_CLOSE`, a paragraph holding one empty run, visually identical
#: to shape A) reset the counter and was a recorded, open escape (escape
#: three, `test_composite_grammar_empty_paragraph_bound.py`). DEC-033's
#: page-projection layer (design §4, R4) now caps the *visually* empty run
#: at 3 before the grammar ever sees a stream, counting shape A and shape B
#: together (a paragraph is visually empty iff it carries no text and no
#: image, so a paragraph with only `tab`/`line_break` counts too, R4). The
#: grammar bound is therefore raised one above that guaranteed ceiling --
#: 4, cap+1 semantics -- and widened to close escape three: `PARAGRAPH_OPEN`
#: sets `paragraph_pending_empty`; it stays `True` through
#: `RUN_OPEN`/`TEXT_BEGIN`/`TEXT_END`/`RUN_CLOSE` and is cleared by any
#: scalar (a Qwen token or a completed byte-fallback frame) or by any other
#: structural token, so both shapes now count toward the same run.
#:
#: The `+ 1` is not headroom beyond the projection's own guarantee, it is
#: the bound's shape: the constraint acts on `PARAGRAPH_OPEN`, emitted
#: before anyone can know whether the paragraph will end up empty, so a
#: bound of N forbids opening ANY paragraph -- empty or not -- once N empty
#: ones have closed in a row. Post-projection gold is *held* to at most 3
#: in a row by `page_projection._assert_run_bounds`, which refuses a page
#: whose `max_empty_run_after` exceeds the cap, so a bound of 4 never
#: refuses valid data. That was for a while an assumption rather than an
#: enforced invariant, and it was false: until RFC004-ERRATA-002 E1's
#: correction (18.9.) the container floor ran after the interior cap and
#: could restore a paragraph the cap had just dropped, with no re-check --
#: measured at 6 pages of the v13 train build with a surviving run of 4,
#: consuming this bound's entire headroom. A longer run is a projection
#: defect, now refused at the projection itself and, failing that, by this
#: bound (`CompositeGrammarViolation` at cache build / replay) -- never
#: silently truncated.
#:
#: **What this bound does NOT do.** It bounds one uninterrupted run of
#: visually-empty paragraphs. It does not bound a decode: escapes one and
#: two below still apply (recorded as negative results, not failures, in
#: `tests/model/test_composite_grammar_empty_paragraph_bound.py`):
#: (1) a `PAGE_BREAK` between runs resets the counter for one token,
#: (2) `SECTION_CLOSE` then `SECTION_OPEN` starts a fresh section.
#: A fourth escape -- a chain of empty RUNS inside ONE paragraph, which no
#: bound on paragraphs reaches -- remains open; DEC-016's canonicalisation
#: (variant B) is the eventual cure for both, not this bound.
#:
#: **Scope.** RFC003-ERRATA-002 / RFC004-ERRATA-002 scope DEC-033's page
#: projection -- and therefore this widened bound -- to the
#: `capability_profile.STANDALONE_V1_PROFILE_ID` ("s2-standalone-page-v1")
#: capability profile only: that is the one profile the projection runs
#: under (`capability_profile.build_capability_profile` docstring). A
#: controller bound to any OTHER `capability_profile_id` -- including the
#: pre-DEC-033 `s2-typed-widened-v1` -- must keep reproducing its v12
#: targets exactly, so it applies `_LEGACY_MAX_CONSECUTIVE_EMPTY_PARAGRAPHS`
#: (shape A only, DEC-016 C's original semantics) and no `TAB` bound at
#: all, via `CompositeGrammarController._dec033_bounds_active`.
MAX_CONSECUTIVE_EMPTY_PARAGRAPHS = 4

#: DEC-016 variant C's original value (53, cap+1 over the pre-projection
#: shape-A-only maximum), kept verbatim for every capability profile other
#: than `STANDALONE_V1_PROFILE_ID` -- see "Scope" above.
_LEGACY_MAX_CONSECUTIVE_EMPTY_PARAGRAPHS = 53

#: DEC-033 / RFC003-ERRATA-002 / RFC004-ERRATA-002 (R3): the projection
#: enforces `max_tab_run_after <= 3` (`page_projection._assert_run_bounds`,
#: reading what survives the cap, not a clamped count), so the bound sits at the
#: measured ceiling itself, not one above it -- unlike the empty-paragraph
#: bound, `TAB` is a leaf whose value is fully known at emission, so the
#: 4th consecutive `TAB` can be refused directly rather than deferred to
#: the next `PARAGRAPH_OPEN`. Reset by any non-`TAB` token. Scoped to
#: `STANDALONE_V1_PROFILE_ID` exactly like the bound above; every other
#: profile has no TAB bound at all (its pre-DEC-033 behaviour).
MAX_CONSECUTIVE_TABS = 3

_TOKEN_NAME_BY_ID: Mapping[int, str] = MappingProxyType(
    {local_id: name for name, local_id in events_vocab.TOKEN_IDS.items()}
)

_PARAGRAPH_OPEN = "PARAGRAPH_OPEN"
_PARAGRAPH_CLOSE = "PARAGRAPH_CLOSE"
_PARAGRAPH_OPEN_LOCAL_ID = events_vocab.TOKEN_IDS[_PARAGRAPH_OPEN]

#: Structural tokens a shape-B (visually empty) run may carry between
#: `PARAGRAPH_OPEN` and `PARAGRAPH_CLOSE` without breaking
#: `paragraph_pending_empty` (design R3).
_RUN_OPEN = "RUN_OPEN"
_RUN_CLOSE = "RUN_CLOSE"
_PARAGRAPH_PENDING_PASSTHROUGH = frozenset(
    {_RUN_OPEN, events_vocab.TEXT_BEGIN, events_vocab.TEXT_END, _RUN_CLOSE}
)

_TAB = "TAB"
_TAB_LOCAL_ID = events_vocab.TOKEN_IDS[_TAB]

class CompositeGrammarViolation(ValueError):
    """A token was not legal in the current composite mode/domain/capability.

    M1-internal (§9.3); mirrors the proposed `COMPOSITE_GRAMMAR_VIOLATION`
    registry record (§9.2), which is not yet ratified (RR-3, §9.1).
    """


class IncompleteCompositeStream(ValueError):
    """A token stream ended without reaching a valid terminator.

    Design §16 criterion 6: "every materialized stream reaches a valid
    document/fragment terminator or a typed failure; zero
    truncate-and-accept". This exception IS that typed failure -- a
    stream that ran out of tokens before :class:`CompositeGrammarState`
    reached ``mode == "terminated"`` is never returned as a shorter
    accepted result.
    """


def _token_name(local_id: int) -> str:
    """Structural local id -> token name; total for the 71 valid ids."""
    try:
        return _TOKEN_NAME_BY_ID[local_id]
    except KeyError:
        raise CompositeTokenIdInvalid(
            f"structural local_id {local_id} is not a valid token id"
        ) from None


#: Half-open `[lo, hi)` byte-value ranges; `()` means no `BYTE_VALUE` is legal.
type ByteRanges = tuple[tuple[int, int], ...]


def _byte_frame_status(prefix: tuple[int, ...]) -> tuple[ByteRanges, bool]:
    """Legal next `BYTE_VALUE` ranges and whether `BYTE_END` is legal now.

    Implements RFC 3629's well-formed-UTF-8 byte table (Unicode §3.9,
    "Well-Formed UTF-8 Byte Sequences"): the return value depends on the
    *leading* byte of `prefix` and how many continuation bytes have
    already been accepted, so overlong encodings, surrogates (U+D800..
    U+DFFF) and code points above U+10FFFF are excluded by the legal
    ranges themselves rather than checked after the fact. Returned as
    half-open ranges, never a per-value enumeration of up to 256 ids --
    this is also what `AllowedDomain.allowed_byte_ranges` reports, so
    `allowed()` and `step()` read the exact same ranges (design §16
    criterion 5's falsifiability requirement: the two must never diverge).
    """
    if not prefix:
        # 1-byte ASCII scalars, then 2/3/4-byte sequence leads; C0-C1 and
        # F5-FF are never valid lead bytes (excluded by these exact bounds).
        return ((0x00, 0x80), (0xC2, 0xE0), (0xE0, 0xF0), (0xF0, 0xF5)), False

    lead = prefix[0]
    if lead < 0x80:
        return (), True  # single-byte scalar is already complete
    if 0xC2 <= lead <= 0xDF:
        continuation_ranges = [(0x80, 0xC0)]
    elif lead == 0xE0:
        continuation_ranges = [(0xA0, 0xC0), (0x80, 0xC0)]  # excludes overlong
    elif 0xE1 <= lead <= 0xEC:
        continuation_ranges = [(0x80, 0xC0), (0x80, 0xC0)]
    elif lead == 0xED:
        continuation_ranges = [(0x80, 0xA0), (0x80, 0xC0)]  # excludes surrogates
    elif 0xEE <= lead <= 0xEF:
        continuation_ranges = [(0x80, 0xC0), (0x80, 0xC0)]
    elif lead == 0xF0:
        continuation_ranges = [(0x90, 0xC0), (0x80, 0xC0), (0x80, 0xC0)]  # excludes overlong
    elif 0xF1 <= lead <= 0xF3:
        continuation_ranges = [(0x80, 0xC0), (0x80, 0xC0), (0x80, 0xC0)]
    elif lead == 0xF4:
        continuation_ranges = [(0x80, 0x90), (0x80, 0xC0), (0x80, 0xC0)]  # excludes > U+10FFFF
    else:
        # Unreachable: step() only ever admits a lead byte drawn from the
        # empty-prefix branch above, which already excludes C0-C1/F5-FF.
        raise AssertionError(f"invalid UTF-8 lead byte 0x{lead:02X} reached grammar state")

    position = len(prefix) - 1
    if position >= len(continuation_ranges):
        return (), True  # frame already holds its full expected length
    return (continuation_ranges[position],), False


def _value_in_ranges(value: int, ranges: ByteRanges) -> bool:
    return any(lo <= value < hi for lo, hi in ranges)


@dataclass(frozen=True, slots=True)
class CompositeGrammarState:
    """Immutable composite state; `step` returns a new instance, never mutates."""

    structural_state: grammar.GrammarState
    mode: Mode
    byte_payload: tuple[int, ...] = ()
    #: How many empty paragraphs have closed back-to-back immediately
    #: before this state (DEC-016 C). Reset by ANY other token, so the
    #: count is always over one uninterrupted run, never a per-document
    #: total -- gold pages carry up to 213 empty paragraphs in total but
    #: at most 52 in a row, and a per-document budget would refuse them.
    empty_paragraph_run: int = 0
    #: True between a `PARAGRAPH_OPEN` and the first token after it: the
    #: paragraph currently open has received nothing yet, so its
    #: `PARAGRAPH_CLOSE` would make it empty.
    paragraph_pending_empty: bool = False
    #: How many `TAB` tokens have been accepted back-to-back immediately
    #: before this state (design R3). Reset by any non-`TAB` token.
    tab_run: int = 0


@dataclass(frozen=True, slots=True)
class AllowedDomain:
    """Compact legal-token domain for one `CompositeGrammarState` (design §3).

    Never a materialized set of every legal packed id: `structural_local_ids`
    holds at most 71 ints, `qwen_allowlist_hash` is a hash pointer into the
    `CompositeVocabulary` artifact (the ~152k allowlisted ids live there,
    not here), and `allowed_byte_ranges` is at most a handful of half-open
    `[lo, hi)` intervals -- never the up to 256 byte values enumerated one
    by one.
    """

    mode: Mode
    structural_local_ids: tuple[int, ...]
    qwen_allowlist_hash: str | None
    allow_text_end: bool
    allow_byte_begin: bool
    allowed_byte_ranges: ByteRanges
    allow_byte_end: bool


_TERMINATED_DOMAIN = AllowedDomain(
    mode="terminated",
    structural_local_ids=(),
    qwen_allowlist_hash=None,
    allow_text_end=False,
    allow_byte_begin=False,
    allowed_byte_ranges=(),
    allow_byte_end=False,
)


@dataclass(frozen=True, slots=True)
class CompositeGrammarController:
    """Wraps one structural profile + capability narrowing + vocabulary.

    Bound to exactly one `capability_profile_id` (design §4.4: "Required
    explicit S1 capability profile... no DEFAULT_PROFILE"). `fresh` checks
    the caller's stated id against this binding rather than resolving it
    dynamically -- see the module docstring for why capability-profile
    resolution itself is out of this module's scope.
    """

    vocabulary: CompositeVocabulary
    capability_profile_id: str
    structural_profile: str
    capability_excluded_local_ids: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if not self.capability_profile_id:
            raise CompositeGrammarViolation("capability_profile_id must be non-empty")
        if self.structural_profile not in grammar.PROFILES:
            raise CompositeGrammarViolation(
                f"unknown structural_profile {self.structural_profile!r}; "
                f"supported: {sorted(grammar.PROFILES)}"
            )

    @property
    def _dec033_bounds_active(self) -> bool:
        """Whether the DEC-033 widened bounds (`MAX_CONSECUTIVE_EMPTY_
        PARAGRAPHS`/`MAX_CONSECUTIVE_TABS`) apply, vs. every other
        profile's pre-DEC-033 behaviour (see the "Scope" module comment
        above `MAX_CONSECUTIVE_EMPTY_PARAGRAPHS`). Identified by
        `capability_profile_id` -- the identity this controller already
        binds to and fail-closed-checks in `fresh` -- rather than by
        `structural_profile` (`s2-standalone-page-v1` and
        `s2-typed-widened-v1` share the same `word-visual-v1.1` AST
        profile, so only the capability profile id distinguishes them) or
        by `scope` on a resolved `CapabilityProfile` (an open string field
        with no closed domain, and this controller is never guaranteed to
        be handed one -- `fresh`'s own docstring notes capability-profile
        resolution is out of this module's scope).
        """
        return self.capability_profile_id in STANDALONE_PAGE_PROFILE_IDS

    def fresh(self, capability_profile_id: str) -> CompositeGrammarState:
        """Initial state; `capability_profile_id` must match this controller's binding."""
        if capability_profile_id != self.capability_profile_id:
            raise CompositeGrammarViolation(
                f"capability_profile_id {capability_profile_id!r} does not match "
                f"this controller's bound profile {self.capability_profile_id!r}"
            )
        return CompositeGrammarState(
            structural_state=grammar.fresh(self.structural_profile),
            mode="structure",
            byte_payload=(),
        )

    def allowed(self, state: CompositeGrammarState) -> AllowedDomain:
        """The exact (compact) set of tokens `step` accepts from `state`."""
        if state.mode == "terminated":
            return _TERMINATED_DOMAIN
        if state.mode == "structure":
            return AllowedDomain(
                mode="structure",
                structural_local_ids=self._structural_domain(state),
                qwen_allowlist_hash=None,
                allow_text_end=False,
                allow_byte_begin=False,
                allowed_byte_ranges=(),
                allow_byte_end=False,
            )
        if state.mode == "text":
            return AllowedDomain(
                mode="text",
                structural_local_ids=(),
                qwen_allowlist_hash=self.vocabulary.artifact_hash,
                allow_text_end=True,
                allow_byte_begin=True,
                allowed_byte_ranges=(),
                allow_byte_end=False,
            )
        byte_ranges, end_legal = _byte_frame_status(state.byte_payload)
        return AllowedDomain(
            mode="byte",
            structural_local_ids=(),
            qwen_allowlist_hash=None,
            allow_text_end=False,
            allow_byte_begin=False,
            allowed_byte_ranges=byte_ranges,
            allow_byte_end=end_legal,
        )

    def step(self, state: CompositeGrammarState, token: TokenRef) -> CompositeGrammarState:
        """Apply one token; raises `CompositeGrammarViolation`/`ByteFrameInvalid`
        unless it is allowed. Never mutates `state`; always returns a new instance."""
        if state.mode == "terminated":
            raise CompositeGrammarViolation(
                "no tokens are legal once the stream has terminated"
            )
        if state.mode == "structure":
            return self._step_structure(state, token)
        if state.mode == "text":
            return self._step_text(state, token)
        return self._step_byte(state, token)

    # -- internals ----------------------------------------------------

    def _structural_domain(self, state: CompositeGrammarState) -> tuple[int, ...]:
        names = grammar.allowed_events(state.structural_state)
        ids = {events_vocab.TOKEN_IDS[name] for name in names}
        ids -= self.capability_excluded_local_ids
        empty_paragraph_bound = (
            MAX_CONSECUTIVE_EMPTY_PARAGRAPHS
            if self._dec033_bounds_active
            else _LEGACY_MAX_CONSECUTIVE_EMPTY_PARAGRAPHS
        )
        if state.empty_paragraph_run >= empty_paragraph_bound:
            bounded = ids - {_PARAGRAPH_OPEN_LOCAL_ID}
            # The bound narrows the domain; it may never empty it. Where the
            # structural grammar (or a capability profile) leaves
            # `PARAGRAPH_OPEN` as the ONLY continuation, forbidding it would
            # turn a legal gold stream into an un-continuable one -- exactly
            # what this bound must not do -- and would surface as
            # `DecodeVocabularyExhausted` rather than as a stop.
            if bounded:
                ids = bounded
        # No TAB bound at all outside `STANDALONE_V1_PROFILE_ID` (see the
        # "Scope" module comment above `MAX_CONSECUTIVE_TABS`).
        if self._dec033_bounds_active and state.tab_run >= MAX_CONSECUTIVE_TABS:
            bounded = ids - {_TAB_LOCAL_ID}
            if bounded:
                ids = bounded
        return tuple(sorted(ids))

    def _empty_paragraph_counters(
        self, state: CompositeGrammarState, name: str
    ) -> tuple[int, bool]:
        """`(empty_paragraph_run, paragraph_pending_empty)` after `name`.

        Under `STANDALONE_V1_PROFILE_ID` (design R3/R4), a visually empty
        paragraph is `PARAGRAPH_OPEN` followed by `PARAGRAPH_CLOSE` with
        nothing but empty runs in between -- shape A (immediately
        adjacent) or shape B (`RUN_OPEN, TEXT_BEGIN, TEXT_END, RUN_CLOSE`
        around no scalar). Those four run-internal tokens leave
        `paragraph_pending_empty` untouched, so a shape-B run counts
        exactly like shape A; any scalar clears it (see `_step_text`/
        `_step_byte`), and any OTHER structural token (a `TAB`, an
        `INLINE_IMAGE`, a nested container) makes the paragraph non-empty
        and, being none of the above, also breaks the run.

        Every other profile keeps DEC-016 variant C's original semantics
        (shape A only): `RUN_OPEN`/`TEXT_BEGIN`/`TEXT_END`/`RUN_CLOSE`
        fall through to the same reset as any other structural token, so
        shape B (and real content) always breaks the run, never counts.
        """
        if name == _PARAGRAPH_OPEN:
            return state.empty_paragraph_run, True
        if name == _PARAGRAPH_CLOSE:
            if state.paragraph_pending_empty:
                return state.empty_paragraph_run + 1, False
            return 0, False
        if self._dec033_bounds_active and name in _PARAGRAPH_PENDING_PASSTHROUGH:
            return state.empty_paragraph_run, state.paragraph_pending_empty
        return 0, False

    def _step_structure(
        self, state: CompositeGrammarState, token: TokenRef
    ) -> CompositeGrammarState:
        if not isinstance(token, StructuralRef):
            raise CompositeGrammarViolation(
                f"{token!r} is not legal in structure mode: only StructuralRef tokens are"
            )
        name = _token_name(token.local_id)
        if token.local_id not in self._structural_domain(state):
            raise CompositeGrammarViolation(
                f"structural token {name!r} is not legal here "
                f"(mode={state.mode!r}, capability-excluded or grammar-illegal)"
            )
        new_structural = grammar.step(state.structural_state, name)
        run, pending = self._empty_paragraph_counters(state, name)
        tab_run = state.tab_run + 1 if name == _TAB else 0
        if name == events_vocab.TEXT_BEGIN:
            return replace(
                state,
                structural_state=new_structural,
                mode="text",
                empty_paragraph_run=run,
                paragraph_pending_empty=pending,
                tab_run=tab_run,
            )
        mode: Mode = "terminated" if not grammar.allowed_events(new_structural) else "structure"
        return replace(
            state,
            structural_state=new_structural,
            mode=mode,
            empty_paragraph_run=run,
            paragraph_pending_empty=pending,
            tab_run=tab_run,
        )

    def _step_text(
        self, state: CompositeGrammarState, token: TokenRef
    ) -> CompositeGrammarState:
        if isinstance(token, QwenRef):
            if token.native_id not in self.vocabulary.qwen_text_allowlist:
                raise CompositeGrammarViolation(
                    f"Qwen native_id {token.native_id} is not in the allowlist"
                )
            # A Qwen token is a scalar (design R3): it clears
            # `paragraph_pending_empty` even though it never enters
            # `grammar.step` (design §3).
            return replace(state, paragraph_pending_empty=False)
        if isinstance(token, StructuralRef):
            name = _token_name(token.local_id)
            if name != events_vocab.TEXT_END:
                raise CompositeGrammarViolation(
                    f"only TEXT_END is legal as a structural token in text mode, got {name!r}"
                )
            new_structural = grammar.step(state.structural_state, name)
            return replace(state, structural_state=new_structural, mode="structure")
        if isinstance(token, ByteControlRef) and token.kind == "byte_begin":
            return replace(state, mode="byte", byte_payload=())
        raise CompositeGrammarViolation(f"{token!r} is not legal in text mode")

    def _step_byte(
        self, state: CompositeGrammarState, token: TokenRef
    ) -> CompositeGrammarState:
        byte_ranges, end_legal = _byte_frame_status(state.byte_payload)
        if isinstance(token, ByteControlRef) and token.kind == "byte_end":
            if not end_legal:
                raise ByteFrameInvalid(
                    f"BYTE_END is not legal after incomplete frame prefix "
                    f"{state.byte_payload!r}"
                )
            # `BYTE_END` completes exactly one byte-fallback scalar (module
            # docstring's state machine): it clears `paragraph_pending_empty`
            # the same as a Qwen scalar (design R3).
            return replace(
                state, mode="text", byte_payload=(), paragraph_pending_empty=False
            )
        if isinstance(token, ByteValueRef):
            value = token.value
            if not _value_in_ranges(value, byte_ranges):
                raise ByteFrameInvalid(
                    f"byte value 0x{value:02X} is not a legal continuation of "
                    f"frame prefix {state.byte_payload!r}"
                )
            return replace(state, byte_payload=state.byte_payload + (value,))
        raise CompositeGrammarViolation(f"{token!r} is not legal in byte mode")


def replay(
    controller: CompositeGrammarController,
    capability_profile_id: str,
    tokens: Iterable[TokenRef],
) -> CompositeGrammarState:
    """Step `controller` through the whole `tokens` stream from a fresh state.

    Design §16 criterion 6's negative control: a stream that does not reach
    `mode == "terminated"` raises `IncompleteCompositeStream` rather than
    returning whatever prefix was consumed.
    """
    state = controller.fresh(capability_profile_id)
    for token in tokens:
        state = controller.step(state, token)
    if state.mode != "terminated":
        raise IncompleteCompositeStream(
            f"stream ended in mode {state.mode!r} without reaching a valid "
            "document/fragment terminator; a truncated stream is never accepted"
        )
    return state


# ---------------------------------------------------------------------------
# Composite grammar descriptor artifact (design §4.5)
# ---------------------------------------------------------------------------
#
# The descriptor is derived from the live surfaces that decide what this
# controller permits, never restated: the base structural controller hash
# from `grammar.controller_hash()` (itself derived from the controller
# modules' source at run time), the structural token map from
# `events_vocab.TOKEN_MAP_HASH`, the vocabulary from the controller's own
# bound `CompositeVocabulary`, and the encoding budget from the frozen
# runtime limits policy. The two policy versions below are the versions of
# the policies *this module implements*, so they live with the
# implementation the way `MATERIALIZER_VERSION` lives with the
# materializer -- a mode-transition or byte-frame semantics change is a
# change to this file and must move them.


def _validate_composite_grammar_document(document: Mapping[str, object]) -> None:
    errors = list(_composite_grammar_validator().iter_errors(document))
    if errors:
        raise CompositeGrammarViolation(
            f"composite_grammar schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )


@cache
def _composite_grammar_validator() -> jsonschema.protocols.Validator:
    path = Path(__file__).parent / "schemas" / "composite_grammar.schema.json"
    with open(path, encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate_composite_grammar(document: Mapping[str, object]) -> None:
    """Validate one composite grammar descriptor against its closed schema."""
    _validate_composite_grammar_document(document)


def build_composite_grammar_descriptor(
    *,
    controller: CompositeGrammarController,
    profile: CapabilityProfile,
    runtime_limits_policy: Mapping[str, object],
) -> dict[str, object]:
    """The §4.5 descriptor for `controller`, validated on the way out.

    `profile` must be the profile `controller` is bound to: the descriptor
    names one `capability_profile_hash`, and a descriptor claiming a
    profile whose id the controller does not enforce would bind a grammar
    to a capability set it never applied. Checked rather than assumed,
    because the controller carries the id but not the hash.
    """
    if profile.profile_id != controller.capability_profile_id:
        raise CompositeGrammarViolation(
            f"profile {profile.profile_id!r} does not match the controller's bound "
            f"capability_profile_id {controller.capability_profile_id!r} -- the "
            "descriptor would name a capability profile this controller never applied"
        )
    # Every field that carries a bound/policy is emitted in the shape
    # actually IN EFFECT for this controller (see the "Scope" module
    # comment above `MAX_CONSECUTIVE_EMPTY_PARAGRAPHS`): under
    # `STANDALONE_V1_PROFILE_ID` this is DEC-033's 1.2.0 shape (widened
    # empty-paragraph bound, a TAB bound, `empty_paragraph_policy_version`
    # 2.0.0); every other profile gets the 1.1.0 shape verbatim -- the
    # exact fields a pre-DEC-033 descriptor for that profile always had --
    # so `composite_grammar_hash` for an old profile is unaffected by this
    # module ever having grown DEC-033's bounds at all.
    active = controller._dec033_bounds_active
    document: dict[str, object] = {
        "composite_grammar_schema_version": (
            COMPOSITE_GRAMMAR_SCHEMA_VERSION
            if active
            else _LEGACY_COMPOSITE_GRAMMAR_SCHEMA_VERSION
        ),
        "base_structural_controller_hash": grammar.controller_hash(),
        "structural_token_map_hash": events_vocab.TOKEN_MAP_HASH,
        "composite_vocabulary_hash": controller.vocabulary.artifact_hash,
        "capability_profile_hash": profile.profile_hash,
        "mode_transition_policy_version": MODE_TRANSITION_POLICY_VERSION,
        "byte_frame_policy_version": BYTE_FRAME_POLICY_VERSION,
        "empty_paragraph_policy_version": (
            EMPTY_PARAGRAPH_POLICY_VERSION if active else _LEGACY_EMPTY_PARAGRAPH_POLICY_VERSION
        ),
        "max_consecutive_empty_paragraphs": (
            MAX_CONSECUTIVE_EMPTY_PARAGRAPHS
            if active
            else _LEGACY_MAX_CONSECUTIVE_EMPTY_PARAGRAPHS
        ),
        "budget_policy_hash": budget_policy_hash(runtime_limits_policy),
    }
    if active:
        document["max_consecutive_tabs"] = MAX_CONSECUTIVE_TABS
    validate_composite_grammar(document)
    return document


def composite_grammar_hash(document: Mapping[str, object]) -> str:
    """`composite_grammar_hash` over a valid §4.5 descriptor."""
    validate_composite_grammar(document)
    return domain_hash(_COMPOSITE_GRAMMAR_DOMAIN, dict(document))

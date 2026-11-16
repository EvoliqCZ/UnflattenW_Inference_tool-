"""S2 exit-gate measurement module -- phase B (design §3 F2, plan task 4).

Phase A (`decode.py`, not this module) owns the model and the reference
text is not visible to it (design §3 F2, RFC005 §36.1: scorer targets
become available only "az po sealed prediction manifest"). Phase B is the
other half of that split: it takes phase A's sealed artefact plus the
eval cache and turns them into per-page and cohort-level measurements. It
never runs a decode and never needs a GPU (design §8).

**The seal (§36.1, governance M4).** Every `sealed.PageDecodeResult` must
carry `inference_mode` and `input_parity_proof_hash`; an absent field is a
refusal (`SealIncomplete`), never a default, and `inference_mode` must be
exactly `"free_running"` (`InferenceModeInvalid`) -- "a selection or gate
record with another mode is invalid" (RFC005 §36.1). A page that reached a
valid mode terminator additionally carries `prediction_ir_hash` plus the
`accepted_tokens`/`node_states`/... fields it was computed over; this
module calls `sealed.recompute_prediction_ir_hash` (the phase A/B shared
contract in `sealed.py`, torch-free by design) and refuses on mismatch
(`SealMismatch`) before reading anything else about the page. This module
never redefines the seal, the hash domain, or the JSON shape -- both are
`sealed.py`'s, imported, not restated.

**No normalisation (RFC005 §37.2).** Every comparison in this module
(exact match, CER, WER, invalid-success text comparison) runs on the
hypothesis and reference strings exactly as `exact_text.decode_tokens`
produced them. `slices.is_diacritic` uses NFD internally to *classify* a
character; that NFD form is never compared, concatenated into a reported
string, or substituted for the original scalar anywhere in this module.

**Every reported value carries its `Denominator`** (§36.4): eligible,
attempted and scored counts, each as a row count AND an independent
lineage-root count (governance M6/§12.2/§39.5 -- row count and lineage
count can differ), plus missing/failed/partial/review counts. `Metric`
cannot be constructed without a `Denominator` -- this is a type, not a
convention.

**Rates use the attempted denominator (governance M6).** A page whose
sealed outcome is not `"terminated"`, or that has no sealed page at all,
is scored with an empty-string hypothesis against its real reference --
it stays in the cohort's rate as a full error rather than silently
leaving the denominator. The **scored-only** subset rate is reported
separately, explicitly labelled `..._diagnostic`, and must never be read
as the criterion value (§36.4:2430).

This module's own `eligible`/`attempted` columns are pass-through equal:
score.py receives whatever cohort of `(sealed page, cache row)` pairs it
is given and does not itself know about the upstream 449 -> 437 filter
accounting (grammar-refused / capability-excluded / text-encoding-failed
/ over-length / the ratified text-coverage floor) -- that reconciliation
is `report.py`'s job (plan task 5), one layer up, over multiple cohorts
including pages this module never sees.

Cohort keying is by `document_bundle_id` alone, matching the real
contract (`sealed.PageDecodeResult`/`sealed.CohortPage` carry no separate
page index): every cohort bundle is a genuinely single-page document
(§36.1:2398), so `ExactTextCacheRow.page_index` is always `0` and is never
part of the key here.

Authorizes no gradient update: pure functions and frozen dataclasses only.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from unflatten.eval.s2_gate import sealed
from unflatten.eval.s2_gate.distance import ErrorRate, character_error_rate, word_error_rate
from unflatten.eval.s2_gate.slices import CONTROL_EVENT_TOKENS, is_diacritic
from unflatten.model import composite_vocab as cv
from unflatten.model import exact_text as et
from unflatten.model import header_length_cap as hlc
from unflatten.model import page_start_policy as psp
from unflatten.model import s1_runner
from unflatten.model import s1_runner_inputs as sri
from unflatten.model import story_retry as sr_policy
from unflatten.model.capability_profile import CapabilityProfile
from unflatten.model.exact_text_cache import ExactTextCacheRow
from unflatten.model.prediction_ir import RawDocumentHypothesis

#: RFC005 §36.1: every prediction manifest carries its own `inference_mode`;
#: this module's cohort is exclusively free-running canary pages. Equal to
#: `sealed.INFERENCE_MODE`, not a second spelling of it.
REQUIRED_INFERENCE_MODE = sealed.INFERENCE_MODE


# ---------------------------------------------------------------------------
# Approximate text match (DEC-030) -- a normalised CER, reported as a
# second, non-diagnostic criterion-2 metric alongside the un-normalised
# `exact_match_rate`. §37.2's "no normalisation" governs the RAW compare
# (`exact_match`/`cer`/`wer` above); DEC-030 is a later, explicit decision
# to ALSO report a second, tolerant metric -- the two are not in tension,
# they answer different questions.
# ---------------------------------------------------------------------------

#: DEC-030 D2: a page's normalised CER of at most 2 wrong characters per
#: 100 counts as an approximate match.
APPROX_MATCH_MAX_CER = 0.02

#: DEC-030 D1 rule 1: a run of >= 3 chars from this class (leading dots,
#: ellipsis, middle dot, hyphen/en dash/em dash) collapses to one `~`.
_LEADER_RUN_RE = re.compile(r"[._…·\-–—]{3,}")

#: DEC-030 D1 rule 3: whitespace removed after normalisation, `str.isspace()`
#: plus NBSP (U+00A0) and NARROW NO-BREAK SPACE (U+202F) -- both are
#: already `str.isspace() == True` in Python, but named explicitly per the
#: decision text rather than relying on that being true forever.
_APPROX_MATCH_EXTRA_SPACE_CHARS = "  "


def normalize_for_approx_match(text: str) -> str:
    """DEC-030 D1's three normalisation rules, in order: (1) a run of
    `>= 3` leader-dot-class chars collapses to one `~`; (2) remaining `_`
    are removed; (3) every remaining whitespace char (`str.isspace()` or
    NBSP/NNBSP) is removed."""
    normalized = _LEADER_RUN_RE.sub("~", text)
    normalized = normalized.replace("_", "")
    return "".join(
        c for c in normalized if not (c.isspace() or c in _APPROX_MATCH_EXTRA_SPACE_CHARS)
    )


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class SealRefused(ValueError):
    """Phase B refuses to score a `sealed.PageDecodeResult` (RFC005 §36.1,
    governance M4)."""


class SealIncomplete(SealRefused):
    """`inference_mode`, `input_parity_proof_hash`, or (for a terminated
    outcome) `prediction_ir_hash`/`accepted_tokens`/`node_states` is
    absent. An absent field is a refusal, not a default."""


class InferenceModeInvalid(SealRefused):
    """`inference_mode` is present but is not `"free_running"`."""


class SealMismatch(SealRefused):
    """The recomputed `prediction_ir_hash` differs from the sealed one --
    the artefact was altered after phase A sealed it, or was built from a
    different fragment than the one it claims."""


class PageStartOracleSealed(SealRefused):
    """The page's `page_start_pattern_source` starts with `gold`: its story
    pattern came from the reference, so the decode is a diagnostic oracle and
    can never be a selection or gate artefact (DEC-061, design D-6)."""


class PageStartPolicyUnratified(SealRefused):
    """The page was decoded under a page-start policy whose sha256 differs
    from the committed policy file, or whose committed status is not
    `RATIFIED` (DEC-061, design D-6)."""


class TextSpanUnterminated(ValueError):
    """A `TEXT_BEGIN` in a token stream has no matching `TEXT_END`."""


# ---------------------------------------------------------------------------
# Sealed artefact contract -- `sealed.PageDecodeResult`/
# `sealed.SealedCohortArtefact` (phase A's output, phase B's input). Defined
# in `sealed.py`, the shared torch-free phase A/B contract; not redefined
# here (the exact mismatch a second local spelling caused before is the bug
# `sealed.py` exists to make impossible).
# ---------------------------------------------------------------------------


# HeadBundle.sha256 is a bare hexdigest; the producer emits `head:<64 hex>`, the only accepted form.
_HEAD_SOURCE_RE = re.compile(r"head:[0-9a-f]{64}")


def _refuse_page_start_seal(page: sealed.PageDecodeResult) -> None:
    """DEC-061 (PROPOSED), design D-6: refuse a page decoded with a gold-derived
    story pattern, or under a page-start policy that is not the committed,
    `RATIFIED` one. The status is read ONLY from the committed policy file
    (`load_page_start_policy()`), never from `page.detail` -- a forged
    `"page_start_policy_status": "RATIFIED"` there must not open the gate.

    Honest-actor limit: `detail` is not part of `recompute_prediction_ir_hash`,
    so deleting the whole `page_start_*` block from `detail` leaves the seal
    valid and this check silent. It guards against mistakes, not against
    deliberate tampering (same weakness as loop-guard identity today); the
    second barrier is that oracle outputs live only under their own root.
    A seal without a page-start block passes."""
    if page.detail is None:
        return
    location = page.document_bundle_id
    try:
        payload = json.loads(page.detail)
    except json.JSONDecodeError as exc:
        raise SealIncomplete(
            f"{location}: sealed detail is not JSON ({exc}) -- an unreadable detail cannot be "
            "shown to carry no page-start block, so it is a refusal"
        ) from exc
    if not isinstance(payload, dict) or payload.get("page_start_policy") is None:
        return
    source = payload.get("page_start_pattern_source")
    if isinstance(source, str) and source.startswith("gold"):
        raise PageStartOracleSealed(
            f"{location}: page_start_pattern_source {source!r} is reference-derived -- an "
            "oracle decode is diagnostic only and cannot be scored as a gate artefact"
        )
    if payload.get("page_start_applied") and not (
        isinstance(source, str) and _HEAD_SOURCE_RE.fullmatch(source)
    ):
        raise SealIncomplete(
            f"{location}: page_start_pattern_source {source!r} is not 'head:<64 hex>' -- "
            "an applied pattern must name the head bundle it came from"
        )
    committed = psp.load_page_start_policy()
    sealed_sha = payload.get("page_start_policy_sha256")
    if sealed_sha != committed.sha256 or committed.status != "RATIFIED":
        raise PageStartPolicyUnratified(
            f"{location}: page-start policy sha256 {sealed_sha!r} "
            f"vs committed {committed.sha256!r} with committed status {committed.status!r} -- "
            "only a RATIFIED policy file that the seal matches may be scored"
        )


class StoryLengthCapUnratified(SealRefused):
    """The page was decoded under a `header-length-cap` policy whose sha256 differs
    from the committed policy file, or whose committed status is not `RATIFIED`."""


def _refuse_story_length_cap_seal(page: sealed.PageDecodeResult) -> None:
    """Same rule as `_refuse_page_start_seal` for `header-length-cap.v0`: the status is
    read only from the committed policy file, never from `page.detail`. A seal
    without a cap block passes."""
    if page.detail is None:
        return
    try:
        payload = json.loads(page.detail)
    except json.JSONDecodeError:
        return  # `_refuse_page_start_seal` already refused an unreadable detail
    if not isinstance(payload, dict) or payload.get("story_length_cap") is None:
        return
    committed = hlc.load_header_length_cap_policy()
    sealed_sha = payload.get("story_length_cap_sha256")
    if sealed_sha != committed.sha256 or committed.status != "RATIFIED":
        raise StoryLengthCapUnratified(
            f"{page.document_bundle_id}: header-length-cap policy sha256 {sealed_sha!r} "
            f"vs committed {committed.sha256!r} with committed status {committed.status!r} -- "
            "only a RATIFIED policy file that the seal matches may be scored"
        )


class StoryRetryUnratified(SealRefused):
    """The page was decoded under a `story-retry` policy whose sha256 differs from the
    committed policy file, or whose committed status is not `RATIFIED`."""


def _refuse_story_retry_seal(page: sealed.PageDecodeResult) -> None:
    """Same rule as `_refuse_story_length_cap_seal` for `story-retry.v0`. A seal without a
    retry block passes."""
    if page.detail is None:
        return
    try:
        payload = json.loads(page.detail)
    except json.JSONDecodeError:
        return  # `_refuse_page_start_seal` already refused an unreadable detail
    if not isinstance(payload, dict) or payload.get("story_retry") is None:
        return
    committed = sr_policy.load_story_retry_policy()
    sealed_sha = payload.get("story_retry_sha256")
    if sealed_sha != committed.sha256 or committed.status != "RATIFIED":
        raise StoryRetryUnratified(
            f"{page.document_bundle_id}: story-retry policy sha256 {sealed_sha!r} "
            f"vs committed {committed.sha256!r} with committed status {committed.status!r} -- "
            "only a RATIFIED policy file that the seal matches may be scored"
        )


def verify_seal(page: sealed.PageDecodeResult) -> None:
    """Refuse `page` if its seal is incomplete, its `inference_mode` is
    not `"free_running"`, or its recomputed `prediction_ir_hash` does not
    match the sealed one. Raises; returns `None` on success."""
    location = page.document_bundle_id
    if page.inference_mode is None or page.input_parity_proof_hash is None:
        raise SealIncomplete(
            f"{location}: sealed artefact is missing inference_mode or "
            "input_parity_proof_hash (RFC005 §36.1) -- an absent field is a "
            "refusal, not a default"
        )
    if page.inference_mode != REQUIRED_INFERENCE_MODE:
        raise InferenceModeInvalid(
            f"{location}: inference_mode {page.inference_mode!r} != "
            f"{REQUIRED_INFERENCE_MODE!r} -- a selection or gate record with "
            "another mode is invalid (RFC005 §36.1)"
        )
    _refuse_page_start_seal(page)
    _refuse_story_length_cap_seal(page)
    _refuse_story_retry_seal(page)
    if page.outcome == "terminated":
        if page.prediction_ir_hash is None:
            raise SealIncomplete(
                f"{location}: outcome is 'terminated' but prediction_ir_hash is "
                "missing -- a terminated page must be sealed"
            )
        try:
            recomputed = sealed.recompute_prediction_ir_hash(page)
        except ValueError as exc:
            raise SealIncomplete(f"{location}: {exc}") from exc
        if recomputed != page.prediction_ir_hash:
            raise SealMismatch(
                f"{location}: recomputed prediction_ir_hash {recomputed!r} != sealed "
                f"{page.prediction_ir_hash!r} -- the artefact was altered after sealing"
            )


# ---------------------------------------------------------------------------
# Text assembly -- two segmenters, one per source (design §6, plan task 4)
# ---------------------------------------------------------------------------


def _unpack_all(token_ids: Sequence[int], vocabulary: cv.CompositeVocabulary) -> list[cv.TokenRef]:
    return [vocabulary.unpack(int(t)) for t in token_ids]


def assemble_hypothesis_text(
    token_ids: Sequence[int],
    *,
    vocabulary: cv.CompositeVocabulary,
    tokenizer: et.QwenTokenizerLike,
) -> str:
    """Segment a hypothesis's packed token stream on `TEXT_BEGIN`/
    `TEXT_END` and decode each span with `exact_text.decode_tokens`,
    concatenating the spans in stream order (plan task 4 point 2). An
    unterminated `TEXT_BEGIN` is a typed failure, not a truncated result.
    """
    text_begin = CONTROL_EVENT_TOKENS["TEXT_BEGIN"]
    text_end = CONTROL_EVENT_TOKENS["TEXT_END"]
    refs = _unpack_all(token_ids, vocabulary)
    spans: list[str] = []
    i = 0
    n = len(refs)
    while i < n:
        ref = refs[i]
        if isinstance(ref, cv.StructuralRef) and ref.local_id == text_begin:
            j = i + 1
            inner: list[cv.TokenRef] = []
            terminated = False
            while j < n:
                candidate = refs[j]
                if isinstance(candidate, cv.StructuralRef) and candidate.local_id == text_end:
                    terminated = True
                    break
                inner.append(candidate)
                j += 1
            if not terminated:
                raise TextSpanUnterminated(
                    f"TEXT_BEGIN at token index {i} has no matching TEXT_END"
                )
            spans.append(
                et.decode_tokens(tuple(inner), vocabulary=vocabulary, tokenizer=tokenizer)
            )
            i = j + 1
            continue
        i += 1
    return "".join(spans)


def assemble_reference_text(
    row: ExactTextCacheRow,
    *,
    vocabulary: cv.CompositeVocabulary,
    tokenizer: et.QwenTokenizerLike,
) -> str:
    """Assemble the gold page string from `row.packed_token_ids`, using
    `row.text_scalar_start`/`row.text_scalar_end` (int32 `[T]`, `-1` for
    non-text positions) to find each maximal contiguous run of text-token
    positions, and decoding each run with `exact_text.decode_tokens`
    (design §6, plan task 4 point 3). The scalar values themselves are
    not consumed here -- `decode_tokens` recovers scalars from the token
    types it groups internally; `-1` vs. not-`-1` is the boundary signal
    this function needs.
    """
    token_ids = row.packed_token_ids.tolist()
    starts = row.text_scalar_start.tolist()
    n = len(token_ids)
    spans: list[str] = []
    i = 0
    while i < n:
        if starts[i] == -1:
            i += 1
            continue
        j = i
        run: list[cv.TokenRef] = []
        while j < n and starts[j] != -1:
            run.append(vocabulary.unpack(int(token_ids[j])))
            j += 1
        spans.append(et.decode_tokens(tuple(run), vocabulary=vocabulary, tokenizer=tokenizer))
        i = j
    return "".join(spans)


# ---------------------------------------------------------------------------
# The two declared slices, as measurements (slices.py declares; this
# module measures). Deliberately count-based, not positional: no edit-
# distance alignment is recovered anywhere in this package (`distance.py`
# returns a cost only, by design -- plan task 1, no external dependency),
# so a slice fidelity here is "how much of the reference's occurrences of
# this slice appear anywhere in the hypothesis", not "at the right
# position". Reported as a named diagnostic alongside CER/WER, never a
# positional substitute for either (criterion 4 is `blocked` regardless,
# `slices.py` module docstring).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SliceFidelity:
    """A presence/multiset-based fidelity measure for one declared slice.

    `rate = matched / reference_count`, `None` when `reference_count` is
    0 and `hypothesis_count` is not -- mirroring `ErrorRate`'s
    empty-reference rule without reusing its edit-distance-shaped fields.
    Two empty sides is the trivial match, `rate = 1.0`.
    """

    matched: int
    reference_count: int
    hypothesis_count: int
    rate: float | None
    reason: str | None = None


def _multiset_fidelity(reference: Counter, hypothesis: Counter) -> SliceFidelity:
    reference_count = sum(reference.values())
    hypothesis_count = sum(hypothesis.values())
    matched = sum(min(count, hypothesis.get(key, 0)) for key, count in reference.items())
    if reference_count == 0:
        return SliceFidelity(
            matched=matched,
            reference_count=0,
            hypothesis_count=hypothesis_count,
            rate=1.0 if hypothesis_count == 0 else None,
            reason=None if hypothesis_count == 0 else "empty_reference",
        )
    return SliceFidelity(
        matched=matched,
        reference_count=reference_count,
        hypothesis_count=hypothesis_count,
        rate=matched / reference_count,
    )


def diacritic_slice(reference_text: str, hypothesis_text: str) -> SliceFidelity:
    """Multiset fidelity over characters `slices.is_diacritic` classifies
    as diacritic-bearing. Classification consumes an NFD view internally;
    the compared/counted characters are the ORIGINAL scalars from
    `reference_text`/`hypothesis_text`, never the NFD form (§37.2)."""
    reference_counter = Counter(c for c in reference_text if is_diacritic(c))
    hypothesis_counter = Counter(c for c in hypothesis_text if is_diacritic(c))
    return _multiset_fidelity(reference_counter, hypothesis_counter)


def control_event_slice(
    reference_token_ids: Sequence[int],
    hypothesis_token_ids: Sequence[int],
    *,
    vocabulary: cv.CompositeVocabulary,
) -> SliceFidelity:
    """Multiset fidelity over occurrences of `slices.CONTROL_EVENT_TOKENS`
    in the two RAW token streams (not the decoded text -- control events
    carry no text of their own, `slices.py` module docstring)."""
    control_local_ids = frozenset(CONTROL_EVENT_TOKENS.values())

    def counts(token_ids: Sequence[int]) -> Counter[int]:
        counter: Counter[int] = Counter()
        for token_id in token_ids:
            ref = vocabulary.unpack(int(token_id))
            if isinstance(ref, cv.StructuralRef) and ref.local_id in control_local_ids:
                counter[ref.local_id] += 1
        return counter

    return _multiset_fidelity(counts(reference_token_ids), counts(hypothesis_token_ids))


# ---------------------------------------------------------------------------
# Replay validity (gate 7, replayed from the sealed hypothesis -- CPU-only
# grammar automaton, no model forward: `s1_runner.gate_7_grammar_
# constrained_valid_tree` only ever touches `controller`/`tokens`) and
# invalid success (M1's derivation, governance M1: NOT this module's own
# substitute definition -- the existing gate is reused as-is).
# ---------------------------------------------------------------------------


def replay_gate7(
    page: sealed.PageDecodeResult,
    *,
    vocabulary: cv.CompositeVocabulary,
    capability_profile: CapabilityProfile,
) -> s1_runner.GateResult:
    """Replay `page`'s accepted tokens through a fresh grammar controller
    via the existing `gate_7_from_hypothesis` (governance M1: this module
    does not redefine gate 7's replay, only supplies its input from the
    sealed artefact instead of a live decode).

    `capability_profile` is required keyword-only with no default and is
    forwarded to `gate_7_from_hypothesis` verbatim. The page must replay
    under the SAME profile the cache it was decoded from actually names
    (`score_cohort`'s caller resolves this from
    `sealed.SealedCohortArtefact.eval_cache_identity`, 2026-09-10 review
    item B), never a profile hardcoded independently of it."""
    if page.outcome != "terminated" or page.accepted_tokens is None:
        return sri.gate_7_from_hypothesis(
            None, vocabulary=vocabulary, capability_profile=capability_profile
        )
    hypothesis = RawDocumentHypothesis(
        accepted_tokens=page.accepted_tokens, node_states=(), terminated=True
    )
    return sri.gate_7_from_hypothesis(
        hypothesis, vocabulary=vocabulary, capability_profile=capability_profile
    )


def is_invalid_success(
    *, outcome: str, gate7_status: str, hypothesis_text: str, reference_text: str
) -> bool:
    """`terminated == True` while gate 7's replay fails or the decoded
    text differs from the reference, EXACTLY as stored (no normalisation)
    -- RFC005 §37.1 `false_strict_valid`, §14.6, derived per governance
    M1's ruling (`docs/plans/spec-s2-exit-gate-metrics-design.md` §4)."""
    if outcome != "terminated":
        return False
    return gate7_status == "fail" or hypothesis_text != reference_text


# ---------------------------------------------------------------------------
# Denominator and Metric -- a type, not a convention (§36.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Denominator:
    """§36.4's per-value accounting block. Every reported `Metric` binds
    one of these; a `Metric` cannot be constructed without it.

    Row counts AND independent-lineage counts, both, at every level
    (governance M6/§12.2/§39.5): a lineage may contribute more than one
    row to `attempted`/`scored`, so the row count and the lineage count
    can legitimately differ, and neither substitutes for the other.
    """

    eligible_row_count: int
    eligible_lineage_count: int
    attempted_row_count: int
    attempted_lineage_count: int
    scored_row_count: int
    scored_lineage_count: int
    missing_count: int
    failed_count: int
    partial_count: int
    review_count: int

    def __post_init__(self) -> None:
        if not (
            self.eligible_row_count >= self.attempted_row_count >= self.scored_row_count >= 0
        ):
            raise ValueError(
                "Denominator requires eligible_row_count >= attempted_row_count >= "
                f"scored_row_count >= 0, got {self.eligible_row_count}, "
                f"{self.attempted_row_count}, {self.scored_row_count}"
            )
        if not (
            self.eligible_lineage_count
            >= self.attempted_lineage_count
            >= self.scored_lineage_count
            >= 0
        ):
            raise ValueError(
                "Denominator requires eligible_lineage_count >= "
                "attempted_lineage_count >= scored_lineage_count >= 0, got "
                f"{self.eligible_lineage_count}, {self.attempted_lineage_count}, "
                f"{self.scored_lineage_count}"
            )
        accounted = (
            self.scored_row_count
            + self.missing_count
            + self.failed_count
            + self.partial_count
            + self.review_count
        )
        if accounted != self.attempted_row_count:
            raise ValueError(
                "Denominator rows do not reconcile: scored"
                f"({self.scored_row_count}) + missing({self.missing_count}) + "
                f"failed({self.failed_count}) + partial({self.partial_count}) + "
                f"review({self.review_count}) = {accounted} != "
                f"attempted_row_count({self.attempted_row_count})"
            )


@dataclass(frozen=True, slots=True)
class Metric:
    """A reported numeric value bound to its `Denominator` (§36.4). This
    is the structural requirement design and governance both call the
    single most important shape in this module: there is no constructor
    path to a bare float."""

    value: float | None
    denominator: Denominator
    reason: str | None = None


def _rate_metric(numerator: int, denominator_count: int, denom: Denominator) -> Metric:
    if denominator_count == 0:
        return Metric(value=None, denominator=denom, reason="empty_denominator")
    return Metric(value=numerator / denominator_count, denominator=denom)


def _micro_error_rate_metric(rates: Sequence[ErrorRate], denom: Denominator) -> Metric:
    total_distance = sum(rate.distance for rate in rates)
    total_reference_length = sum(rate.reference_length for rate in rates)
    if total_reference_length == 0:
        return Metric(value=None, denominator=denom, reason="empty_reference_cohort")
    return Metric(value=total_distance / total_reference_length, denominator=denom)


def _slice_fidelity_metric(slices: Sequence[SliceFidelity], denom: Denominator) -> Metric:
    total_matched = sum(s.matched for s in slices)
    total_reference = sum(s.reference_count for s in slices)
    if total_reference == 0:
        return Metric(value=None, denominator=denom, reason="empty_reference_cohort")
    return Metric(value=total_matched / total_reference, denominator=denom)


# ---------------------------------------------------------------------------
# Per-page scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PageScore:
    document_bundle_id: str
    page_index: int
    lineage_root_id: str
    outcome: str
    status: Literal["scored", "failed", "missing"]
    hypothesis_text: str
    reference_text: str
    exact_match: bool
    cer: ErrorRate
    wer: ErrorRate
    #: DEC-030: CER over `normalize_for_approx_match`-normalised text.
    #: `None` unless the page is `"terminated"` and `status == "scored"`;
    #: even then it is still `None` if the normalised reference is empty
    #: and the normalised hypothesis is not (undefined rate, same
    #: `distance.character_error_rate` empty-reference rule the raw `cer`
    #: field already follows).
    approx_cer: float | None
    #: DEC-030 D2: True only for a terminated, scored page whose
    #: `approx_cer <= APPROX_MATCH_MAX_CER`.
    approx_match: bool
    diacritic_slice: SliceFidelity
    control_event_slice: SliceFidelity
    replay_status: str
    invalid_success: bool
    #: `None` unless hypothesis assembly itself raised a typed failure
    #: (`TextNotNfc`/`TextInvalidUnicode`/`ByteFrameInvalid`/
    #: `TextSpanUnterminated`) -- the exception's class name, so a failed
    #: page's cause is visible without re-decoding it (must_fix, code
    #: review 2026-09-07: a non-NFC hypothesis must not raise out of
    #: `score_page`, §36.4).
    hypothesis_decode_failure: str | None = None


def score_page(
    page: sealed.PageDecodeResult | None,
    row: ExactTextCacheRow,
    *,
    lineage_root_id: str,
    vocabulary: cv.CompositeVocabulary,
    tokenizer: et.QwenTokenizerLike,
    capability_profile: CapabilityProfile,
) -> PageScore:
    """Score one page. `page is None` means phase A recorded no sealed
    page at all for `row` -- treated as a `"missing"` error, scored with
    an empty-string hypothesis against the real reference so it stays in
    every rate as a full error (§36.4) rather than silently vanishing.
    `row` is always the identity source (`document_bundle_id`/`page_index`)
    -- `sealed.PageDecodeResult` carries no page index of its own, since
    every cohort bundle is a genuinely single-page document.

    `outcome` is `"missing"` here (should_fix, code review 2026-09-07),
    never `"not_terminated"`: `"not_terminated"` is gate 13's own
    silent-truncation branch name (decode.py) for a page that WAS decoded
    and failed to reach a mode terminator; a page phase A never sealed at
    all was never attempted, which is a different failure
    (`report.criterion_3`'s `silent_truncation_count` reads `outcome ==
    "not_terminated"` and must not count a page that never ran a decode).

    Refuses via `verify_seal` before reading anything else about `page`.
    """
    reference_text = assemble_reference_text(row, vocabulary=vocabulary, tokenizer=tokenizer)
    reference_token_ids = row.packed_token_ids.tolist()
    document_bundle_id = row.document_bundle_id
    page_index = row.page_index

    hypothesis_decode_failure: str | None = None

    if page is None:
        outcome = "missing"
        status: Literal["scored", "failed", "missing"] = "missing"
        gate7_status = "blocked"
        hypothesis_text = ""
        accepted_tokens: tuple[int, ...] = ()
    else:
        verify_seal(page)
        outcome = page.outcome
        gate7_status = replay_gate7(
            page, vocabulary=vocabulary, capability_profile=capability_profile
        ).status
        if outcome == "terminated":
            assert page.accepted_tokens is not None  # verify_seal already checked
            accepted_tokens = page.accepted_tokens
            try:
                hypothesis_text = assemble_hypothesis_text(
                    accepted_tokens, vocabulary=vocabulary, tokenizer=tokenizer
                )
                status = "scored"
            except (
                et.TextNotNfc,
                et.TextInvalidUnicode,
                et.ByteFrameInvalid,
                TextSpanUnterminated,
            ) as exc:
                # A free-running arm can emit a grammar-legal span that is
                # not valid Unicode/NFC text (§37.2, must_fix code review
                # 2026-09-07) -- the untrained baseline is the arm most
                # likely to. One such page must stay in the denominator as
                # a full error (§36.4), never end the whole cohort.
                status = "failed"
                hypothesis_text = ""
                hypothesis_decode_failure = f"{type(exc).__name__}: {exc}"
        else:
            status = "failed"
            accepted_tokens = ()
            hypothesis_text = ""

    cer = character_error_rate(hypothesis_text, reference_text)
    wer = word_error_rate(hypothesis_text, reference_text)
    exact_match = hypothesis_text == reference_text
    if outcome == "terminated" and status == "scored":
        approx_cer = character_error_rate(
            normalize_for_approx_match(hypothesis_text), normalize_for_approx_match(reference_text)
        ).rate
    else:
        approx_cer = None
    approx_match = approx_cer is not None and approx_cer <= APPROX_MATCH_MAX_CER
    invalid = is_invalid_success(
        outcome=outcome,
        gate7_status=gate7_status,
        hypothesis_text=hypothesis_text,
        reference_text=reference_text,
    )
    return PageScore(
        document_bundle_id=document_bundle_id,
        page_index=page_index,
        lineage_root_id=lineage_root_id,
        outcome=outcome,
        status=status,
        hypothesis_text=hypothesis_text,
        reference_text=reference_text,
        exact_match=exact_match,
        cer=cer,
        wer=wer,
        approx_cer=approx_cer,
        approx_match=approx_match,
        diacritic_slice=diacritic_slice(reference_text, hypothesis_text),
        control_event_slice=control_event_slice(
            reference_token_ids, accepted_tokens, vocabulary=vocabulary
        ),
        replay_status=gate7_status,
        invalid_success=invalid,
        hypothesis_decode_failure=hypothesis_decode_failure,
    )


# ---------------------------------------------------------------------------
# Cohort-level aggregate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CohortScore:
    denominator: Denominator
    exact_match_rate: Metric
    #: DEC-030: the fraction of attempted rows that are an approximate
    #: match (`PageScore.approx_match`), same denominator as
    #: `exact_match_rate`.
    approx_text_match_rate: Metric
    character_error_rate: Metric
    word_error_rate: Metric
    diacritic_slice: Metric
    control_event_slice: Metric
    invalid_success_count: int
    pages: tuple[PageScore, ...]
    #: Named diagnostic per §36.4:2430 -- exact-match rate over the
    #: `"scored"` subset only. Never the criterion value, never the
    #: headline (governance M6).
    scored_subset_exact_match_rate_diagnostic: Metric


def score_cohort(
    sealed_pages: Sequence[sealed.PageDecodeResult],
    rows_by_key: Mapping[str, ExactTextCacheRow],
    *,
    lineage_by_bundle: Mapping[str, str],
    vocabulary: cv.CompositeVocabulary,
    tokenizer: et.QwenTokenizerLike,
    capability_profile: CapabilityProfile,
) -> CohortScore:
    """Score every row in `rows_by_key`, matched by `document_bundle_id`
    against `sealed_pages` (`sealed.PageDecodeResult` carries no separate
    page index -- module docstring).

    `rows_by_key` is this call's attempted set -- score.py's own
    eligible/attempted columns are always equal (module docstring); a row
    with no matching sealed page is scored as `"missing"`, never dropped.
    Refuses if any `sealed_pages` entry has no matching row, or if
    `lineage_by_bundle` does not cover every row's bundle id.

    `capability_profile` is required keyword-only with no default and is
    forwarded to every `score_page`/`replay_gate7` call. The caller
    (`cli._score_artefact`, 2026-09-10 review item B) resolves it from the
    sealed artefact's own `eval_cache_identity.capability_profile_hash`, so
    every page in one cohort replays under the profile that cohort's cache
    actually names.
    """
    if not rows_by_key:
        raise ValueError("score_cohort requires at least one row")

    pages_by_key = {p.document_bundle_id: p for p in sealed_pages}
    unknown_pages = set(pages_by_key) - set(rows_by_key)
    if unknown_pages:
        raise ValueError(
            f"sealed_pages contains {len(unknown_pages)} page(s) with no matching "
            f"cache row: {sorted(unknown_pages)[:5]}"
        )
    unknown_lineage = {
        row.document_bundle_id
        for row in rows_by_key.values()
        if row.document_bundle_id not in lineage_by_bundle
    }
    if unknown_lineage:
        raise ValueError(
            f"lineage_by_bundle is missing {len(unknown_lineage)} bundle id(s), e.g. "
            f"{sorted(unknown_lineage)[:5]} -- every attempted row must carry its "
            "lineage_root_id (RFC005 §12.2/§39.5)"
        )

    page_scores = [
        score_page(
            pages_by_key.get(key),
            row,
            lineage_root_id=lineage_by_bundle[row.document_bundle_id],
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            capability_profile=capability_profile,
        )
        for key, row in rows_by_key.items()
    ]

    attempted_row_count = len(page_scores)
    attempted_lineage_count = len({p.lineage_root_id for p in page_scores})
    scored = [p for p in page_scores if p.status == "scored"]
    scored_row_count = len(scored)
    scored_lineage_count = len({p.lineage_root_id for p in scored})
    failed_count = sum(1 for p in page_scores if p.status == "failed")
    missing_count = sum(1 for p in page_scores if p.status == "missing")

    denominator = Denominator(
        eligible_row_count=attempted_row_count,
        eligible_lineage_count=attempted_lineage_count,
        attempted_row_count=attempted_row_count,
        attempted_lineage_count=attempted_lineage_count,
        scored_row_count=scored_row_count,
        scored_lineage_count=scored_lineage_count,
        missing_count=missing_count,
        failed_count=failed_count,
        partial_count=0,
        review_count=0,
    )

    exact_match_rate = _rate_metric(
        sum(1 for p in page_scores if p.exact_match), attempted_row_count, denominator
    )
    approx_text_match_rate = _rate_metric(
        sum(1 for p in page_scores if p.approx_match), attempted_row_count, denominator
    )
    cer_metric = _micro_error_rate_metric([p.cer for p in page_scores], denominator)
    wer_metric = _micro_error_rate_metric([p.wer for p in page_scores], denominator)
    diacritic_metric = _slice_fidelity_metric(
        [p.diacritic_slice for p in page_scores], denominator
    )
    control_event_metric = _slice_fidelity_metric(
        [p.control_event_slice for p in page_scores], denominator
    )
    scored_subset_exact_match_rate_diagnostic = _rate_metric(
        sum(1 for p in scored if p.exact_match), scored_row_count, denominator
    )

    return CohortScore(
        denominator=denominator,
        exact_match_rate=exact_match_rate,
        approx_text_match_rate=approx_text_match_rate,
        character_error_rate=cer_metric,
        word_error_rate=wer_metric,
        diacritic_slice=diacritic_metric,
        control_event_slice=control_event_metric,
        invalid_success_count=sum(1 for p in page_scores if p.invalid_success),
        pages=tuple(page_scores),
        scored_subset_exact_match_rate_diagnostic=scored_subset_exact_match_rate_diagnostic,
    )

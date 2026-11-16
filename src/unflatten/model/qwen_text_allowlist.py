"""The pinned-tokenizer `classify` callable, its independent verifier, and
the Component Lock binding (design §4.3, §4.5, §4.7).

`composite_vocab.build_qwen_text_allowlist` is the builder: it partitions
`range(native_vocab_size)` given a `classify` callable. This module supplies
the callable for the real Qwen tokenizer, checks the result against
acceptance criteria A1-A10 **without reusing the classifier**, and chains the
two resulting hashes into the two Component Lock components.

Three constraints from the design, each load-bearing:

- **No-entry ids are found by LOOKUP.** One batched
  `convert_ids_to_tokens(range(native_vocab_size))`; `None` means the
  tokenizer has no entry for that id. `id >= len(tokenizer)` is exactly the
  numeric-range inference `composite_vocab` forbids ("Safety is not inferred
  from numeric range or spelling alone"). That the result comes out as the
  contiguous padding tail is a *verified consequence* (A4b), not the
  mechanism.
- **`native_vocab_size` is the model's embedding width** (151936 on the
  pinned candidate), never `len(tokenizer)` (151669). The composite head is
  built over `packed_ranges(sdpa_candidate.vocab_size)`; an allowlist built
  at the other number indexes silently wrong logit columns (§4.6).
- **The verifier is independent of the builder** and expresses acceptance as
  counts, so a misclassification surfaces as a failing equality rather than
  a judgement call. A10 extends that to the rules themselves: a rule that
  matches zero ids denies nothing while reading as if it does, and nothing
  but a count can see it.

No new `DENIAL_REASONS` member: the 267-id padding tail is `unknown_token`
(the honest "the tokenizer has no entry for this id"), and the two
populations that could merge under that label are kept apart by A4b's set
equality, not by a wider enum -- see §4.3.

This module authorizes no gradient update: pure functions, no `nn.Module`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from unflatten.model import composite_vocab as cv

__all__ = [
    "QWEN_CLASSIFY_RULES",
    "AllowlistAcceptance",
    "ClassifyRule",
    "QwenTextBinding",
    "TextTokenAllowlistUnacceptable",
    "build_qwen_text_binding",
    "make_qwen_classify",
    "verify_qwen_text_allowlist",
]

#: Version stamped on the two Component Lock components this module builds.
COMPONENT_VERSION = "1.0.0"

_CHAT_CONTROL = re.compile(r"^<\|im_.*\|>$")
_VISION_CONTROL = re.compile(r"^(<\|vision_.*\|>|<\|image_pad\|>|<\|video_pad\|>)$")
#: Design §4.3's table spells this row `<|tool_*|>`, which matches NOTHING on
#: the pinned tokenizer: its tool controls are `<tool_call>`, `</tool_call>`,
#: `<tool_response>`, `</tool_response>` -- no pipes, and two of them closing
#: tags. Corrected to the measured spellings 2026-09-05; the row's intent
#: (tool controls are denied as `tool_control`) is unchanged, and all four
#: were denied under the old spelling too, as `unclassified_added_token`.
_TOOL_CONTROL = re.compile(r"^</?tool_[a-z0-9_]+>$")


@dataclass(frozen=True, slots=True)
class ClassifyRule:
    """One rule of the §4.3 classification table, as data.

    Exists so `verify_qwen_text_allowlist` can count how many ids each rule
    actually matched. A rule that matches zero ids is *invisible* otherwise:
    it reads as correct, denies nothing, and no test fails -- which is how
    `<|tool_*|>` survived governance three times and was caught only by
    reading. A10 turns that into a counting failure.

    `inert_reason` is the escape hatch, and it is deliberately a written,
    reviewable declaration rather than silence: a rule that legitimately
    matches nothing on a given tokenizer (a chat-control spelling a future
    revision drops, say) must say so and say why. A rule declared inert that
    then DOES match also fails A10 -- a stale declaration is the same defect
    pointing the other way.
    """

    reason: str
    description: str
    inert_reason: str | None = None


#: The classification rules A10 counts. Ordered as design §4.3's table.
#:
#: `control_bearing_text` is deliberately NOT here: it is the decode pass
#: (§4.4), not a classify rule, and A6 reports its count precisely because a
#: zero there retires a premise and must be looked at rather than raise.
QWEN_CLASSIFY_RULES: tuple[ClassifyRule, ...] = (
    ClassifyRule("unknown_token", "the tokenizer has no entry for this id (batched lookup)"),
    ClassifyRule("special_token", "id in tokenizer.all_special_ids"),
    ClassifyRule("chat_control", "added token matching <|im_*|>"),
    ClassifyRule(
        "vision_control", "added token matching <|vision_*|>, <|image_pad|>, <|video_pad|>"
    ),
    ClassifyRule("tool_control", "added token matching <tool_*> or </tool_*>"),
    ClassifyRule("pad_token", "id == tokenizer.pad_token_id"),
    ClassifyRule(
        "unclassified_added_token",
        "an added-vocab token no rule above recognized (fail-closed catch-all)",
    ),
)


class QwenTokenizerLike(Protocol):
    """The tokenizer surface classification and verification need.

    Deliberately narrow (mirroring `exact_text.QwenTokenizerLike`): every
    test but the pinned-tokenizer one runs against a stub, so the fast
    suite never downloads or loads a real tokenizer.
    """

    all_special_ids: list[int]
    pad_token_id: int | None
    vocab_size: int

    def __len__(self) -> int: ...

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str | None]: ...

    def get_added_vocab(self) -> dict[str, int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class TextTokenAllowlistUnacceptable(ValueError):
    """An allowlist payload fails one of the acceptance criteria A1-A10.

    Distinct from `composite_vocab.TextTokenAllowlistInvalid`, which is the
    builder's own refusal: this one is the independent verifier's verdict on
    a finished payload. M1-internal (design §9.3).
    """


def _no_entry_ids(tokenizer: QwenTokenizerLike, native_vocab_size: int) -> set[int]:
    """The ids the tokenizer has no entry for -- ONE batched lookup."""
    tokens = tokenizer.convert_ids_to_tokens(list(range(native_vocab_size)))
    return {native_id for native_id, token in enumerate(tokens) if token is None}


def _metadata_ids(tokenizer: QwenTokenizerLike) -> set[int]:
    return set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())


def make_qwen_classify(
    tokenizer: QwenTokenizerLike, *, native_vocab_size: int
) -> Callable[[int], tuple[str, ...] | None]:
    """Build the `classify` callable `build_qwen_text_allowlist` needs, per
    design §4.3's table.

    The no-entry lookup runs once, here, over the whole range; the returned
    callable is then a pure set/dict membership test per id.

    Reasons are unioned, not exclusive, so the pad id (which is also a
    special token) legitimately carries two. The added-vocab catch-all
    `unclassified_added_token` fires only for an added token that no rule
    above it recognized *at all* -- an added token that is a declared
    special token is recognized, and labelling it "unclassified" as well
    would hide the genuinely unrecognized ones it exists to surface.
    """
    no_entry = _no_entry_ids(tokenizer, native_vocab_size)
    special_ids = set(tokenizer.all_special_ids)
    added_by_id = {native_id: token for token, native_id in tokenizer.get_added_vocab().items()}
    pad_token_id = tokenizer.pad_token_id

    def classify(native_id: int) -> tuple[str, ...] | None:
        if native_id in no_entry:
            return ("unknown_token",)
        reasons: set[str] = set()
        if native_id in special_ids:
            reasons.add("special_token")
        token = added_by_id.get(native_id)
        if token is not None:
            if _CHAT_CONTROL.match(token):
                reasons.add("chat_control")
            elif _VISION_CONTROL.match(token):
                reasons.add("vision_control")
            elif _TOOL_CONTROL.match(token):
                reasons.add("tool_control")
            elif not reasons:
                reasons.add("unclassified_added_token")
        if pad_token_id is not None and native_id == pad_token_id:
            reasons.add("pad_token")
        return tuple(sorted(reasons)) if reasons else None

    return classify


@dataclass(frozen=True, slots=True)
class AllowlistAcceptance:
    """What A1-A10 counted. Returned so a caller can record the numbers in a
    run report -- A6 in particular is *reported*, not thresholded, because a
    zero there retires design §4.2's premise and must be looked at.
    """

    native_vocab_size: int
    allowed_count: int
    denied_count: int
    metadata_id_count: int
    no_entry_id_count: int
    control_bearing_only_count: int
    no_entry_empty_decode_count: int
    #: A10: how many denied ids each `ClassifyRule` accounts for. A rule
    #: reading 0 here has stopped matching anything the payload records.
    classify_rule_counts: Mapping[str, int]


def verify_qwen_text_allowlist(
    payload: Mapping[str, Any],
    *,
    tokenizer: QwenTokenizerLike,
    native_vocab_size: int,
    rules: Sequence[ClassifyRule] = QWEN_CLASSIFY_RULES,
) -> AllowlistAcceptance:
    """Check acceptance criteria A1-A10 against `payload` and return the
    counts. Raises `TextTokenAllowlistUnacceptable` on the first failure.

    Every set this compares against is recomputed from `tokenizer` here;
    nothing is read back out of the payload and believed, and
    `make_qwen_classify` is never called. `native_vocab_size` is the
    caller's own `sdpa_candidate.vocab_size` -- A3 is what makes an
    allowlist built at the wrong width a loud failure instead of a silently
    wrong logit index (§4.6).

    `rules` is the classification table A10 counts against. It defaults to
    `QWEN_CLASSIFY_RULES`, the pinned tokenizer's; a different tokenizer with
    a different table passes its own rather than being measured against
    rules that were never meant to fire for it.
    """
    # A8 first: the criteria below index the payload's fields, and a payload
    # that is not even shaped like one should say so rather than KeyError.
    try:
        cv.validate_qwen_text_allowlist(payload)
    except cv.TextTokenAllowlistInvalid as exc:
        raise TextTokenAllowlistUnacceptable(f"A8: {exc}") from exc

    allowed = set(payload["allowed_native_ids"])
    denied_reasons = {entry["native_id"]: entry["reasons"] for entry in payload["denied"]}
    denied = set(denied_reasons)

    if payload["native_vocab_size"] != native_vocab_size:
        raise TextTokenAllowlistUnacceptable(
            f"A3: payload native_vocab_size {payload['native_vocab_size']} != "
            f"{native_vocab_size} -- an allowlist built at another width packs "
            "ids at offsets the composite head does not share (design §4.6)"
        )

    if len(allowed) + len(denied) != native_vocab_size:
        raise TextTokenAllowlistUnacceptable(
            f"A1: {len(allowed)} allowed + {len(denied)} denied != {native_vocab_size}"
        )
    if allowed & denied or allowed | denied != set(range(native_vocab_size)):
        raise TextTokenAllowlistUnacceptable(
            f"A2: allowed and denied are not a partition of range({native_vocab_size}) "
            f"-- {len(allowed & denied)} id(s) in both, "
            f"{len(set(range(native_vocab_size)) - (allowed | denied))} in neither"
        )

    for native_id, reasons in denied_reasons.items():
        if not reasons or not set(reasons) <= cv.DENIAL_REASONS:
            raise TextTokenAllowlistUnacceptable(
                f"A7: native_id {native_id} has reasons {reasons!r}, which is empty "
                f"or outside the closed DENIAL_REASONS set"
            )

    metadata_ids = _metadata_ids(tokenizer)
    no_entry = _no_entry_ids(tokenizer, native_vocab_size)

    non_control = {
        i for i, reasons in denied_reasons.items() if reasons != ["control_bearing_text"]
    }
    if non_control != metadata_ids | no_entry:
        raise TextTokenAllowlistUnacceptable(
            f"A4a: ids denied for a reason other than control_bearing_text "
            f"({len(non_control)}) are not exactly the tokenizer's metadata ids "
            f"({len(metadata_ids)}) plus its no-entry ids ({len(no_entry)}): "
            f"{len(non_control - (metadata_ids | no_entry))} unexpected, "
            f"{len((metadata_ids | no_entry) - non_control)} missing"
        )

    unknown_only = {i for i, reasons in denied_reasons.items() if reasons == ["unknown_token"]}
    expected_tail = set(range(len(tokenizer), native_vocab_size))
    if unknown_only != no_entry or no_entry != expected_tail:
        raise TextTokenAllowlistUnacceptable(
            f"A4b: the unknown_token-only ids ({len(unknown_only)}) must be exactly "
            f"the tokenizer's no-entry ids ({len(no_entry)}) and exactly the padding "
            f"tail [{len(tokenizer)}, {native_vocab_size}) ({len(expected_tail)}) -- "
            "a real UNK id, or a no-entry id anywhere but the tail, changes what "
            "this reason means (design §4.3)"
        )

    metadata_block = set(range(tokenizer.vocab_size, len(tokenizer)))
    if metadata_ids != metadata_block:
        raise TextTokenAllowlistUnacceptable(
            f"A4c: the tokenizer's {len(metadata_ids)} metadata ids are not the "
            f"contiguous block [{tokenizer.vocab_size}, {len(tokenizer)}) "
            f"({len(metadata_block)} ids)"
        )

    if payload["control_text_check"] != "applied":
        raise TextTokenAllowlistUnacceptable(
            f"A5: control_text_check is {payload['control_text_check']!r}; a payload "
            "built without a tokenizer never ran the decode check and cannot answer "
            "for control-bearing text"
        )

    empty_decodes = sum(1 for native_id in sorted(no_entry) if tokenizer.decode([native_id]) == "")
    if empty_decodes != len(no_entry):
        raise TextTokenAllowlistUnacceptable(
            f"A9: {len(no_entry) - empty_decodes} of {len(no_entry)} no-entry ids "
            "decode to something other than '' -- the measurement that let the "
            "decode pass run unguarded no longer holds (design §4.4)"
        )

    rule_counts = {
        rule.reason: sum(1 for reasons in denied_reasons.values() if rule.reason in reasons)
        for rule in rules
    }
    silently_inert = sorted(
        rule.reason for rule in rules if rule.inert_reason is None and rule_counts[rule.reason] == 0
    )
    stale_declarations = sorted(
        rule.reason for rule in rules if rule.inert_reason is not None and rule_counts[rule.reason]
    )
    if silently_inert or stale_declarations:
        raise TextTokenAllowlistUnacceptable(
            f"A10: rule(s) {silently_inert} matched no id and are not declared inert; "
            f"rule(s) {stale_declarations} are declared inert but did match -- a rule "
            "that cannot fire denies nothing while reading as if it does, and only a "
            "count can see that"
        )

    return AllowlistAcceptance(
        native_vocab_size=native_vocab_size,
        allowed_count=len(allowed),
        denied_count=len(denied),
        metadata_id_count=len(metadata_ids),
        no_entry_id_count=len(no_entry),
        control_bearing_only_count=len(denied) - len(non_control),
        no_entry_empty_decode_count=empty_decodes,
        classify_rule_counts=rule_counts,
    )


@dataclass(frozen=True, slots=True)
class QwenTextBinding:
    """The §4.7 chain, built once: allowlist payload -> its hash -> the
    composite vocabulary payload -> its hash -> the two Component Lock
    components.

    Deliberately does NOT verify itself: `verify_qwen_text_allowlist` is the
    acceptance authority and a builder that checks its own output proves
    nothing. Callers run it on `allowlist`.
    """

    tokenizer_artifact_set_hash: str
    allowlist: dict[str, Any]
    qwen_text_allowlist_hash: str
    composite_vocabulary: dict[str, Any]
    composite_vocabulary_hash: str
    components: tuple[dict[str, Any], ...]


def build_qwen_text_binding(
    *,
    tokenizer: QwenTokenizerLike,
    tokenizer_artifact_set_hash: str,
    native_vocab_size: int,
) -> QwenTextBinding:
    """Build the allowlist for `tokenizer` and everything that hangs off it.

    `native_vocab_size` must be the candidate's `vocab_size` (the embedding
    width), not `len(tokenizer)` -- see §4.6.
    """
    allowlist = cv.build_qwen_text_allowlist(
        tokenizer_artifact_set_hash=tokenizer_artifact_set_hash,
        native_vocab_size=native_vocab_size,
        classify=make_qwen_classify(tokenizer, native_vocab_size=native_vocab_size),
        tokenizer=tokenizer,
    )
    allowlist_hash = cv.qwen_text_allowlist_hash(allowlist)
    vocabulary, _ranges = cv.build_composite_vocabulary(
        tokenizer_artifact_set_hash=tokenizer_artifact_set_hash,
        text_allowlist_hash=allowlist_hash,
        native_vocab_size=native_vocab_size,
    )
    vocabulary_hash = cv.composite_vocabulary_hash(vocabulary)
    return QwenTextBinding(
        tokenizer_artifact_set_hash=tokenizer_artifact_set_hash,
        allowlist=allowlist,
        qwen_text_allowlist_hash=allowlist_hash,
        composite_vocabulary=vocabulary,
        composite_vocabulary_hash=vocabulary_hash,
        # In UTF-8 byte order of `kind`, so splicing these into a lock's
        # `components` array keeps §39.3's canonical ordering local.
        components=(
            {
                "kind": "composite_vocabulary",
                "version": COMPONENT_VERSION,
                "semantic_hash": vocabulary_hash,
            },
            {
                "kind": "text_tokenizer",
                "version": COMPONENT_VERSION,
                # Raw SHA-256 over the tokenizer's own bytes, not a semantic
                # hash -- the schema keeps the two distinct on purpose.
                "artifact_sha256": tokenizer_artifact_set_hash,
            },
        ),
    )

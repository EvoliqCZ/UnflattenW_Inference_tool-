"""M1 exact-text codec: Unicode policy, Qwen-first tokenization, one-scalar
byte fallback (design §3, §6).

This module is the scalar-exact heart of M1: it turns already-NFC text into
a sequence of :class:`~unflatten.model.composite_vocab.TokenRef` (Qwen refs
interleaved with ``BYTE_BEGIN``/``BYTE_VALUE``/``BYTE_END`` frames) and back,
such that ``decode_tokens(encode_text(text)) == text`` for every scalar of
every text accepted by the Unicode/control policy (design §16 criterion 2),
and every accepted byte frame is exactly one canonical UTF-8 scalar (design
§16 criterion 5).

It builds on :mod:`unflatten.model.composite_vocab` (`TokenRef`, packing,
`CompositeVocabulary`) and does not reimplement or modify it. It does not
implement the higher-level `ExactTextCodec.encode_range` (design §3), which
additionally consumes a `ScalarPolicyRange` from RFC003 boundary/ownership
projection -- that belongs to the training materializer, not this module.

Unicode policy (design §6):

1. Input must already be NFC; a difference from ``unicodedata.normalize``
   means the stored authority is malformed/stale, not a rewrite target.
2. Lone surrogates (U+D800-U+DFFF) are rejected -- they are not scalars.
3. U+0009 TAB, U+000A LF and U+000D CR are illegal Run substitutes: tabs and
   explicit breaks are structural nodes, never text scalars.
4. Every other scalar accepted by upstream AST validation is preserved
   unchanged; this module performs no case fold, cleanup, autocorrect or
   translation beyond the required NFC equality check.

The codec never emits U+FFFD: every decode path either produces an exact
scalar or raises a typed exception (`ByteFrameInvalid`, `TextInvalidUnicode`,
`TextNotNfc`) -- UTF-8 decoding is always strict, never a lossy/replacing mode.

M1-internal exceptions here are CamelCase and are not RFC004 registry codes
(design §9.1, §9.3) -- no unratified code may cross into an Issue envelope or
a `primary_code` until RR-3 is ratified.

This module authorizes no gradient update: it declares only a Protocol,
dataclasses-free pure functions and exceptions -- no `nn.Module`, no
weights, no optimizer.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Protocol

from unflatten.model.composite_vocab import (
    ByteControlRef,
    ByteValueRef,
    CompositeVocabulary,
    QwenRef,
    TokenRef,
)

#: Illegal Run control substitutes (design §6 step 4): tabs and explicit
#: breaks are structural nodes, never encoded as text scalars.
_ILLEGAL_RUN_CONTROLS: frozenset[int] = frozenset({0x09, 0x0A, 0x0D})

_SURROGATE_LOW = 0xD800
_SURROGATE_HIGH = 0xDFFF


class TextNotNfc(ValueError):
    """Text is not exactly its own NFC form.

    M1-internal (§9.3); mirrors the proposed `TEXT_NOT_NFC` registry record
    (§9.2), which is not yet ratified (RR-3, §9.1).
    """


class TextInvalidUnicode(ValueError):
    """Non-scalar, forbidden Run control, or otherwise invalid input/decoded text.

    M1-internal (§9.3); mirrors `TEXT_INVALID_UNICODE`, which already exists
    in RFC004 §42.4 -- this class is the internal exception, not the
    registry code itself (§9.3 naming convention).
    """


class ByteFrameInvalid(ValueError):
    """A byte frame is not exactly one valid canonical UTF-8 scalar.

    M1-internal (§9.3); mirrors the proposed `BYTE_FRAME_INVALID` registry
    record (§9.2), which is not yet ratified (RR-3, §9.1). Covers
    unterminated frames, malformed UTF-8, multi-scalar frames, and
    non-canonical (overlong/surrogate/out-of-range) encodings.
    """


class TextTokenizerRoundtripFailed(ValueError):
    """A Qwen candidate did not round-trip exactly under no-cleanup decode.

    M1-internal (§9.3); mirrors `TEXT_TOKENIZER_ROUNDTRIP_FAILED`, which
    already exists in RFC004 §42.4. `encode_text` never raises this itself
    -- a failed candidate is silently rejected in favor of the next
    candidate or the byte fallback (design §6) -- but it is available for
    callers that need to report a hard round-trip failure (e.g. gate code).
    """


class TextEncodingBudgetExceeded(ValueError):
    """The hash-bound per-range operation budget (design §6) was exhausted.

    M1-internal (§9.3); mirrors the proposed `TEXT_ENCODING_BUDGET_EXCEEDED`
    registry record (§9.2). Raised instead of silently switching
    segmentation strategy -- exceeding the budget is a typed failure, never
    a route change.
    """


class QwenTokenizerLike(Protocol):
    """Minimal duck-typed Qwen tokenizer interface the codec needs.

    Deliberately narrower than any concrete tokenizer class: the backbone
    and its tokenizer are swappable (CLAUDE.md "Backbone is swappable"), and
    keeping this Protocol minimal lets tests use a fast, offline fake rather
    than loading real tokenizer artifacts. Implementations must apply no
    special tokens and no cleanup -- `encode`/`decode` are exact-text
    primitives, not chat-template or generation helpers.
    """

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


def require_nfc(text: str) -> str:
    """Validate `text` is exactly its own NFC form; never rewrites it.

    Design §6 Unicode policy step 2: "Normalize once to NFC and require
    equality to stored authority. Difference is malformed/stale authority,
    not permission to rewrite it."
    """
    normalized = unicodedata.normalize("NFC", text)
    if normalized != text:
        raise TextNotNfc(
            f"text ({len(text)} scalars) is not NFC-normalized; the stored "
            "authority must already be NFC, this function does not rewrite it"
        )
    return text


def _is_surrogate(codepoint: int) -> bool:
    return _SURROGATE_LOW <= codepoint <= _SURROGATE_HIGH


def validate_scalars(text: str) -> None:
    """Reject lone surrogates and illegal Run control substitutes (design §6 steps 3-4).

    Every other scalar is left untouched -- this function narrows, it never
    rewrites (step 6: "No normalization beyond NFC, case fold, cleanup,
    autocorrect, or translation").
    """
    for index, ch in enumerate(text):
        codepoint = ord(ch)
        if _is_surrogate(codepoint):
            raise TextInvalidUnicode(
                f"lone surrogate U+{codepoint:04X} at scalar index {index}"
            )
        if codepoint in _ILLEGAL_RUN_CONTROLS:
            raise TextInvalidUnicode(
                f"illegal Run control U+{codepoint:04X} at scalar index {index}; "
                "tabs and explicit breaks are structural nodes, not text scalars"
            )


def _encode_byte_fallback_scalar(scalar: str) -> tuple[TokenRef, ...]:
    """Encode exactly one Unicode scalar as a `BYTE_BEGIN .. BYTE_END` frame."""
    data = scalar.encode("utf-8")
    return (
        (ByteControlRef(kind="byte_begin"),)
        + tuple(ByteValueRef(value=b) for b in data)
        + (ByteControlRef(kind="byte_end"),)
    )


def encode_text(
    text: str,
    *,
    vocabulary: CompositeVocabulary,
    tokenizer: QwenTokenizerLike,
    budget_ops: int,
) -> tuple[TokenRef, ...]:
    """Qwen-first, one-scalar-fallback encode of `text` (design §6).

    For each position, tries the largest allowlisted, exactly-round-tripping
    Qwen candidate (leftmost-longest); falls back to a byte frame for
    exactly one scalar otherwise. `text` must already be NFC and contain no
    lone surrogate or illegal Run control (`require_nfc`, `validate_scalars`
    are applied first).

    `budget_ops` bounds the number of tokenizer probes (design §6: "A
    hash-bound operation budget prevents quadratic denial-of-service").
    Each candidate probe (one `tokenizer.encode` call) consumes one op;
    exceeding the budget raises `TextEncodingBudgetExceeded` rather than
    silently changing segmentation.
    """
    require_nfc(text)
    validate_scalars(text)

    atoms: list[TokenRef] = []
    n = len(text)
    i = 0
    ops = 0
    while i < n:
        chosen: tuple[int, list[int]] | None = None
        for j in range(n, i, -1):
            ops += 1
            if ops > budget_ops:
                raise TextEncodingBudgetExceeded(
                    f"encoding budget of {budget_ops} ops exceeded at scalar "
                    f"index {i} (text length {n})"
                )
            candidate_ids = tokenizer.encode(text[i:j])
            if not candidate_ids:
                continue
            if not all(tid in vocabulary.qwen_text_allowlist for tid in candidate_ids):
                continue
            if tokenizer.decode(candidate_ids) != text[i:j]:
                continue
            chosen = (j, candidate_ids)
            break
        if chosen is not None:
            j, candidate_ids = chosen
            atoms.extend(QwenRef(native_id=tid) for tid in candidate_ids)
            i = j
        else:
            atoms.extend(_encode_byte_fallback_scalar(text[i]))
            i += 1
    return tuple(atoms)


def _decode_byte_frame(data: bytes, *, frame_start_index: int) -> str:
    """Decode one byte frame's payload; must be exactly one canonical UTF-8 scalar."""
    if not data:
        raise ByteFrameInvalid(f"empty byte frame at token index {frame_start_index}")
    try:
        decoded = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ByteFrameInvalid(
            f"byte frame at token index {frame_start_index} is not valid UTF-8: {exc}"
        ) from exc
    if len(decoded) != 1:
        raise ByteFrameInvalid(
            f"byte frame at token index {frame_start_index} decodes to "
            f"{len(decoded)} scalars, expected exactly 1"
        )
    codepoint = ord(decoded)
    if _is_surrogate(codepoint):
        raise ByteFrameInvalid(
            f"byte frame at token index {frame_start_index} decodes to lone "
            f"surrogate U+{codepoint:04X}"
        )
    if decoded.encode("utf-8") != data:
        # CPython's strict UTF-8 decoder already rejects overlong encodings,
        # surrogate halves and out-of-range code points, but the design
        # names the re-encode-compare check explicitly (criterion 5) as the
        # canonical-form guarantee; keep it so this does not silently rely
        # on decoder-internal behavior alone.
        raise ByteFrameInvalid(
            f"byte frame at token index {frame_start_index} is not canonical UTF-8"
        )
    return decoded


def decode_tokens(
    tokens: Sequence[TokenRef],
    *,
    vocabulary: CompositeVocabulary,
    tokenizer: QwenTokenizerLike,
) -> str:
    """Decode a `Qwen`/byte-frame token sequence back to NFC text (design §6).

    Consecutive `QwenRef` tokens are decoded together with the exact pinned
    tokenizer settings; each byte frame is decoded independently as one
    scalar. Disallowed Qwen IDs, malformed/unterminated frames, invalid
    UTF-8, multi-scalar frames, any token outside a Qwen run or byte frame,
    non-NFC output, and a lone surrogate or illegal Run control scalar
    (`validate_scalars`, the same check `encode_text` applies on input) are
    all typed failures -- never U+FFFD, never a truncated result.
    """
    pieces: list[str] = []
    qwen_run: list[int] = []

    def flush_qwen_run() -> None:
        if qwen_run:
            pieces.append(tokenizer.decode(list(qwen_run)))
            qwen_run.clear()

    n = len(tokens)
    i = 0
    while i < n:
        token = tokens[i]
        if isinstance(token, QwenRef):
            if token.native_id not in vocabulary.qwen_text_allowlist:
                raise TextInvalidUnicode(
                    f"denied Qwen native_id {token.native_id} at token index {i}"
                )
            qwen_run.append(token.native_id)
            i += 1
            continue
        flush_qwen_run()
        if isinstance(token, ByteControlRef) and token.kind == "byte_begin":
            j = i + 1
            byte_values: list[int] = []
            next_token = tokens[j] if j < n else None
            while isinstance(next_token, ByteValueRef):
                byte_values.append(next_token.value)
                j += 1
                next_token = tokens[j] if j < n else None
            if not (isinstance(next_token, ByteControlRef) and next_token.kind == "byte_end"):
                raise ByteFrameInvalid(
                    f"unterminated byte frame starting at token index {i}"
                )
            pieces.append(_decode_byte_frame(bytes(byte_values), frame_start_index=i))
            i = j + 1
            continue
        raise ByteFrameInvalid(
            f"unexpected token {token!r} at index {i}: not inside a Qwen run "
            "or a well-formed byte frame"
        )
    flush_qwen_run()

    result = "".join(pieces)
    normalized = unicodedata.normalize("NFC", result)
    if normalized != result:
        raise TextNotNfc(f"decoded text ({len(result)} scalars) is not NFC")
    validate_scalars(result)
    return result


def validate_byte_frame_scalars(tokens: Sequence[TokenRef]) -> None:
    """Reject any byte frame in `tokens` that decodes to a lone surrogate or
    illegal Run control scalar (design §6 steps 2-3), without requiring a
    `QwenTokenizerLike`.

    For a caller that only has a packed `TokenRef` stream and no tokenizer
    -- e.g. `exact_text_cache.verify_row_replay`, which replays a stream
    through the grammar controller and never decodes Qwen runs -- this is
    the Run-control/surrogate check `decode_tokens` applies to its result,
    scoped to byte frames only. `QwenRef` tokens are skipped: their scalar
    content is a tokenizer-decode concern this function has no tokenizer to
    resolve, not a byte-frame concern.
    """
    n = len(tokens)
    i = 0
    while i < n:
        token = tokens[i]
        if isinstance(token, ByteControlRef) and token.kind == "byte_begin":
            j = i + 1
            byte_values: list[int] = []
            next_token = tokens[j] if j < n else None
            while isinstance(next_token, ByteValueRef):
                byte_values.append(next_token.value)
                j += 1
                next_token = tokens[j] if j < n else None
            if not (isinstance(next_token, ByteControlRef) and next_token.kind == "byte_end"):
                raise ByteFrameInvalid(
                    f"unterminated byte frame starting at token index {i}"
                )
            validate_scalars(_decode_byte_frame(bytes(byte_values), frame_start_index=i))
            i = j + 1
            continue
        i += 1
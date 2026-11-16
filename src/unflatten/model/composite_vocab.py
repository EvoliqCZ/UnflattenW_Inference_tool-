"""M1 composite vocabulary: TokenRef union, packing, and the Qwen text
allowlist / composite vocabulary artifacts (design §3, §4.2, §4.3).

This module wraps the existing 71-token structural map
(:mod:`unflatten.model.events_vocab`) rather than reimplementing it — the
structural alphabet, its ids, version and golden hash are untouched (design
§11: "No semantic/API/format change"). It adds a disjoint, deterministic
packing of the union `Structural | Qwen | ByteControl | ByteValue` into one
integer id space, plus the artifact builders/hashes for the closed Qwen
text-ID allowlist and the composite vocabulary.

Packing (design §3):

    Structural(local)  -> local
    Qwen(native)       -> 71 + native
    ByteControl(begin) -> 71 + qwen_vocab_size + 0
    ByteControl(end)   -> 71 + qwen_vocab_size + 1
    ByteValue(value)   -> 71 + qwen_vocab_size + 2 + value

`pack`/`unpack` are total inverses for valid refs; invalid values raise
typed errors, never clamp or map to an unknown-token placeholder.

M1-internal exceptions here are CamelCase and are not RFC004 registry
codes (§9.1, §9.3) — no unratified code may cross into an Issue envelope
or a `primary_code` until RR-3 is ratified.

This module authorizes no gradient update: it declares only dataclasses
and pure functions, no `nn.Module`, no weights, no optimizer.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal, Protocol

import jsonschema.protocols
import jsonschema.validators

from unflatten.model import events_vocab
from unflatten.model.hashing import domain_hash

_SCHEMA_DIR = Path(__file__).parent / "schemas"

_ALLOWLIST_DOMAIN = "rfc004.qwen-text-allowlist/v1"
_VOCABULARY_DOMAIN = "rfc004.composite-vocabulary/v1"

QWEN_TEXT_ALLOWLIST_SCHEMA_VERSION = "1.2.0"  # MINOR: added control_text_check
COMPOSITE_VOCABULARY_SCHEMA_VERSION = "1.0.0"

#: The classification policy `build_qwen_text_allowlist` applies by default.
#: Bumped 1.0.0 -> 1.1.0 when construction started denying decoded control
#: text (design §4.2 "Why `control_bearing_text` exists"), and 1.1.0 -> 1.2.0
#: when the artifact started declaring whether that decode check ran
#: (design §4.2 "The check has an undeclared skip mode"). An artifact built
#: under an older policy is not comparable to one built under this one, and
#: `tokenizer_artifact_set_hash` alone would not reveal that.
QWEN_TEXT_ALLOWLIST_CLASSIFICATION_POLICY_VERSION = "1.2.0"

#: Whether `build_qwen_text_allowlist` decoded each candidate id and applied
#: the `control_bearing_text` check, or skipped it for lack of a tokenizer
#: (design §4.2 "The check has an undeclared skip mode"). Declared on the
#: payload itself so a weaker, tokenizer-less artifact is never
#: indistinguishable from a fully-checked one except by rebuilding it.
type ControlTextCheckMode = Literal["applied", "skipped"]
CONTROL_TEXT_CHECK_MODES: frozenset[str] = frozenset({"applied", "skipped"})

#: Closed denial-reason vocabulary (design §4.2). Classification must land
#: on one of these; ambiguous/unrecognized metadata aborts construction.
DENIAL_REASONS: frozenset[str] = frozenset(
    {
        "special_token",
        "added_control",
        "chat_control",
        "vision_control",
        "tool_control",
        "pad_token",
        "unknown_token",
        "unclassified_added_token",
        "control_bearing_text",
    }
)

#: U+0009 TAB, U+000A LF, U+000D CR -- illegal Run substitutes (design §6)
#: and structural controls (RFC004 :1322-1337). A native id whose decoded
#: text carries one of these is denied regardless of its metadata.
_CONTROL_BEARING_CHARS = frozenset("\t\n\r")


class _NativeIdDecoder(Protocol):
    """Minimal decode-only duck type `build_qwen_text_allowlist` needs to
    check decoded text for control characters (design §4.2).

    Structurally compatible with `exact_text.QwenTokenizerLike`, but defined
    locally rather than imported: `exact_text` imports from this module, so
    importing back would cycle.
    """

    def decode(self, ids: Sequence[int]) -> str: ...


def _decodes_to_control_bearing_text(tokenizer: _NativeIdDecoder, native_id: int) -> bool:
    decoded = tokenizer.decode([native_id])
    return any(ch in _CONTROL_BEARING_CHARS for ch in decoded)

STRUCTURAL_COUNT: int = len(events_vocab.TOKEN_IDS)  # 71, pinned by the structural map
BYTE_CONTROL_COUNT = 2
BYTE_VALUE_COUNT = 256

_BYTE_CONTROL_OFFSET: Mapping[ByteControlKind, int] = {"byte_begin": 0, "byte_end": 1}
_BYTE_CONTROL_KIND_BY_OFFSET: Mapping[int, ByteControlKind] = {0: "byte_begin", 1: "byte_end"}


class CompositeTokenIdInvalid(ValueError):
    """A `TokenRef` or packed integer id is not valid for a `PackedRanges`.

    M1-internal (§9.3); mirrors the proposed `COMPOSITE_TOKEN_ID_INVALID`
    registry record (§9.2), which is not yet ratified (RR-3, §9.1).
    """


class TextTokenAllowlistInvalid(ValueError):
    """The Qwen allow/deny partition or its schema is invalid.

    M1-internal (§9.3); mirrors the proposed `TEXT_TOKEN_ALLOWLIST_INVALID`
    registry record (§9.2), which is not yet ratified (RR-3, §9.1).
    """


class CompositeVocabularyInvalid(ValueError):
    """A composite vocabulary payload fails its closed schema (§4.3)."""


# ---------------------------------------------------------------------------
# TokenRef union
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StructuralRef:
    local_id: int  # 0 <= id < STRUCTURAL_COUNT; existing events_vocab.TOKEN_IDS


@dataclass(frozen=True, slots=True)
class QwenRef:
    native_id: int


type ByteControlKind = Literal["byte_begin", "byte_end"]


@dataclass(frozen=True, slots=True)
class ByteControlRef:
    kind: ByteControlKind


@dataclass(frozen=True, slots=True)
class ByteValueRef:
    value: int  # 0..255


type TokenRef = StructuralRef | QwenRef | ByteControlRef | ByteValueRef


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PackedRanges:
    structural_start: int
    structural_count: int
    qwen_start: int
    qwen_count: int
    byte_control_start: int
    byte_control_count: int
    byte_value_start: int
    byte_value_count: int
    total_count: int


def packed_ranges(qwen_vocab_size: int) -> PackedRanges:
    """Compute the disjoint packed ranges for a given Qwen native vocab size."""
    if qwen_vocab_size < 0:
        raise CompositeTokenIdInvalid(
            f"qwen_vocab_size must be >= 0, got {qwen_vocab_size}"
        )
    structural_start = 0
    qwen_start = structural_start + STRUCTURAL_COUNT
    byte_control_start = qwen_start + qwen_vocab_size
    byte_value_start = byte_control_start + BYTE_CONTROL_COUNT
    total_count = byte_value_start + BYTE_VALUE_COUNT
    return PackedRanges(
        structural_start=structural_start,
        structural_count=STRUCTURAL_COUNT,
        qwen_start=qwen_start,
        qwen_count=qwen_vocab_size,
        byte_control_start=byte_control_start,
        byte_control_count=BYTE_CONTROL_COUNT,
        byte_value_start=byte_value_start,
        byte_value_count=BYTE_VALUE_COUNT,
        total_count=total_count,
    )


def pack(ref: TokenRef, ranges: PackedRanges) -> int:
    """Pack a `TokenRef` into its integer id under `ranges`. Total for valid refs."""
    if isinstance(ref, StructuralRef):
        if not (0 <= ref.local_id < ranges.structural_count):
            raise CompositeTokenIdInvalid(
                f"structural local_id {ref.local_id} out of range "
                f"[0, {ranges.structural_count})"
            )
        return ranges.structural_start + ref.local_id
    if isinstance(ref, QwenRef):
        if not (0 <= ref.native_id < ranges.qwen_count):
            raise CompositeTokenIdInvalid(
                f"qwen native_id {ref.native_id} out of range [0, {ranges.qwen_count})"
            )
        return ranges.qwen_start + ref.native_id
    if isinstance(ref, ByteControlRef):
        offset = _BYTE_CONTROL_OFFSET.get(ref.kind)
        if offset is None:
            raise CompositeTokenIdInvalid(f"unknown byte control kind {ref.kind!r}")
        return ranges.byte_control_start + offset
    if isinstance(ref, ByteValueRef):
        if not (0 <= ref.value < ranges.byte_value_count):
            raise CompositeTokenIdInvalid(
                f"byte value {ref.value} out of range [0, {ranges.byte_value_count})"
            )
        return ranges.byte_value_start + ref.value
    raise CompositeTokenIdInvalid(f"unknown TokenRef type: {type(ref)!r}")


def unpack(packed_id: int, ranges: PackedRanges) -> TokenRef:
    """Unpack an integer id into its `TokenRef` under `ranges`. Total for valid ids."""
    if not (0 <= packed_id < ranges.total_count):
        raise CompositeTokenIdInvalid(
            f"packed id {packed_id} out of range [0, {ranges.total_count})"
        )
    if packed_id < ranges.qwen_start:
        return StructuralRef(local_id=packed_id - ranges.structural_start)
    if packed_id < ranges.byte_control_start:
        return QwenRef(native_id=packed_id - ranges.qwen_start)
    if packed_id < ranges.byte_value_start:
        offset = packed_id - ranges.byte_control_start
        return ByteControlRef(kind=_BYTE_CONTROL_KIND_BY_OFFSET[offset])
    return ByteValueRef(value=packed_id - ranges.byte_value_start)


@dataclass(frozen=True, slots=True)
class CompositeVocabulary:
    """Concrete `CompositeVocabulary` (design §3 Protocol)."""

    artifact_hash: str
    ranges: PackedRanges
    qwen_text_allowlist: frozenset[int]
    tokenizer_artifact_set_hash: str

    def pack(self, ref: TokenRef) -> int:
        return pack(ref, self.ranges)

    def unpack(self, packed_id: int) -> TokenRef:
        return unpack(packed_id, self.ranges)


# ---------------------------------------------------------------------------
# Closed Qwen text-ID allowlist (design §4.2)
# ---------------------------------------------------------------------------


def build_qwen_text_allowlist(
    *,
    tokenizer_artifact_set_hash: str,
    native_vocab_size: int,
    classify: Callable[[int], Sequence[str] | None],
    tokenizer: _NativeIdDecoder | None = None,
    classification_policy_version: str = QWEN_TEXT_ALLOWLIST_CLASSIFICATION_POLICY_VERSION,
) -> dict[str, Any]:
    """Partition `range(native_vocab_size)` into allowed/denied native ids.

    `classify(native_id)` must return `None` for an ordinary allowed text id,
    or a non-empty sequence of closed `DENIAL_REASONS` for a denied id.
    Every id in range receives exactly one classification; ambiguous or
    unrecognized classification aborts construction rather than guessing
    (design §4.2 — "Safety is not inferred from numeric range or spelling
    alone").

    When `tokenizer` is given, every candidate id is additionally decoded
    and denied with `control_bearing_text` if its text carries U+0009,
    U+000A or U+000D — independent of, and unioned with, whatever `classify`
    returns. `classify`-only metadata cannot see this: Qwen's byte-level BPE
    is expected to hold ordinary, unremarked merge tokens spelling a bare
    tab or newline (design §4.2 "Why `control_bearing_text` exists"). Without
    a tokenizer this decode-based check cannot run and is skipped, not
    treated as a denial. The payload records which happened in
    `control_text_check` (`"applied"` if `tokenizer` was given, else
    `"skipped"`) so the two are never indistinguishable except by
    rebuilding (design §4.2 "The check has an undeclared skip mode").
    """
    if native_vocab_size < 0:
        raise TextTokenAllowlistInvalid(
            f"native_vocab_size must be >= 0, got {native_vocab_size}"
        )
    allowed: list[int] = []
    denied: list[dict[str, Any]] = []
    for native_id in range(native_vocab_size):
        reasons = classify(native_id)
        reason_set: set[str] = set()
        if reasons is not None:
            reason_set = set(reasons)
            if not reason_set:
                raise TextTokenAllowlistInvalid(
                    f"native_id {native_id} has empty denial reasons; "
                    "ambiguous classification is forbidden"
                )
            unknown = reason_set - DENIAL_REASONS
            if unknown:
                raise TextTokenAllowlistInvalid(
                    f"native_id {native_id} has unrecognized denial reasons: "
                    f"{sorted(unknown)}"
                )
        if tokenizer is not None and _decodes_to_control_bearing_text(tokenizer, native_id):
            reason_set.add("control_bearing_text")
        if not reason_set:
            allowed.append(native_id)
            continue
        denied.append({"native_id": native_id, "reasons": sorted(reason_set)})
    control_text_check: ControlTextCheckMode = "applied" if tokenizer is not None else "skipped"
    return {
        "qwen_text_allowlist_schema_version": QWEN_TEXT_ALLOWLIST_SCHEMA_VERSION,
        "tokenizer_artifact_set_hash": tokenizer_artifact_set_hash,
        "native_vocab_size": native_vocab_size,
        "allowed_native_ids": allowed,
        "denied": denied,
        "classification_policy_version": classification_policy_version,
        "control_text_check": control_text_check,
    }


def qwen_text_allowlist_hash(payload: Mapping[str, Any]) -> str:
    return domain_hash(_ALLOWLIST_DOMAIN, payload)


# ---------------------------------------------------------------------------
# Composite vocabulary (design §4.3)
# ---------------------------------------------------------------------------


def build_composite_vocabulary(
    *,
    tokenizer_artifact_set_hash: str,
    text_allowlist_hash: str,
    native_vocab_size: int,
    packing_policy_version: str = "1.0.0",
    text_encoding_policy_version: str = "1.0.0",
    greedy_tie_break: str = "lowest_packed_id",
) -> tuple[dict[str, Any], PackedRanges]:
    """Build the composite vocabulary artifact payload and its `PackedRanges`."""
    ranges = packed_ranges(native_vocab_size)
    payload: dict[str, Any] = {
        "composite_vocabulary_schema_version": COMPOSITE_VOCABULARY_SCHEMA_VERSION,
        "structural": {
            "token_map_version": events_vocab.TOKEN_MAP_VERSION,
            "token_map_hash": events_vocab.TOKEN_MAP_HASH,
            "count": ranges.structural_count,
            "packed_start": ranges.structural_start,
        },
        "qwen": {
            "tokenizer_artifact_set_hash": tokenizer_artifact_set_hash,
            "text_allowlist_hash": text_allowlist_hash,
            "native_vocab_size": native_vocab_size,
            "packed_start": ranges.qwen_start,
        },
        "byte_fallback": {
            "policy_version": "1.0.0",
            "encoding": "UTF-8",
            "scalar_normalization": "NFC",
            "frame": ["BYTE_BEGIN", "BYTE_VALUE*", "BYTE_END"],
            "byte_controls": [["BYTE_BEGIN", 0], ["BYTE_END", 1]],
            "byte_value_count": ranges.byte_value_count,
            "packed_start": ranges.byte_control_start,
        },
        "packing_policy_version": packing_policy_version,
        "text_encoding_policy_version": text_encoding_policy_version,
        "greedy_tie_break": greedy_tie_break,
    }
    return payload, ranges


def composite_vocabulary_hash(payload: Mapping[str, Any]) -> str:
    return domain_hash(_VOCABULARY_DOMAIN, payload)


def make_composite_vocabulary(
    *,
    tokenizer_artifact_set_hash: str,
    allowlist_payload: Mapping[str, Any],
) -> CompositeVocabulary:
    """Assemble a `CompositeVocabulary` from an already-built allowlist payload."""
    native_vocab_size = allowlist_payload["native_vocab_size"]
    text_allowlist_hash = qwen_text_allowlist_hash(allowlist_payload)
    vocab_payload, ranges = build_composite_vocabulary(
        tokenizer_artifact_set_hash=tokenizer_artifact_set_hash,
        text_allowlist_hash=text_allowlist_hash,
        native_vocab_size=native_vocab_size,
    )
    artifact_hash = composite_vocabulary_hash(vocab_payload)
    allowlist = frozenset(allowlist_payload["allowed_native_ids"])
    return CompositeVocabulary(
        artifact_hash=artifact_hash,
        ranges=ranges,
        qwen_text_allowlist=allowlist,
        tokenizer_artifact_set_hash=tokenizer_artifact_set_hash,
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


@cache
def _validator(name: str) -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{name}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate_qwen_text_allowlist(payload: Mapping[str, Any]) -> None:
    """Validate `payload` against `qwen_text_allowlist.schema.json`."""
    errors = list(_validator("qwen_text_allowlist").iter_errors(payload))
    if errors:
        raise TextTokenAllowlistInvalid(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )


def validate_composite_vocabulary(payload: Mapping[str, Any]) -> None:
    """Validate `payload` against `composite_vocabulary.schema.json`."""
    errors = list(_validator("composite_vocabulary").iter_errors(payload))
    if errors:
        raise CompositeVocabularyInvalid(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )

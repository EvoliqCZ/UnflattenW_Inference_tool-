"""Deterministic CBOR encoder — profile ``rfc8949-core-det-nfc-int-v0`` (RFC004 §3.1).

RFC 8949 §4.2.1 Core Deterministic Encoding Requirements, narrowed:

- allowed values: integers within the basic CBOR argument range (major type
  0/1, i.e. ``-2**64 <= v <= 2**64 - 1``), ``True``/``False``/``None``
  (major 7), UTF-8 text strings (major 3), byte strings (major 2), lists
  (major 4) and str-keyed dicts (major 5);
- every item is definite-length in the shortest possible head form;
- text strings (values and map keys) are Unicode NFC-normalized on encode;
- map keys are unique text strings sorted bytewise-lexicographically over
  their *encoded* bytes (RFC 8949 §4.2.1) — two source keys that collide
  after NFC normalization are rejected, never silently merged;
- arrays preserve declared order and are never sorted by the encoder;
- floats, tags, ``undefined``, NaN/Inf, indefinite lengths, non-str map
  keys and any other Python type raise :class:`CborProfileError` — the
  encoder rejects values outside the data model instead of coercing them.

Integer boundary policy: an integer outside ``[-2**64, 2**64 - 1]`` cannot
be encoded in an 8-byte argument and would require tag 2/3 bignums, which
the profile forbids — it is rejected with :class:`CborProfileError`.
RFC004 identity projections never need such values.

Only ``bytes`` (not ``bytearray``/``memoryview``), ``list`` (not ``tuple``)
and ``dict`` are accepted, to keep the accepted data model exact.
"""

from __future__ import annotations

import unicodedata
from typing import Any

PROFILE_ID = "rfc8949-core-det-nfc-int-v0"

_MAX_ARG = 2**64 - 1

# Container nesting cap: hostile/degenerate inputs surface as
# CborProfileError instead of RecursionError.
_MAX_DEPTH = 64


class CborProfileError(ValueError):
    """A value is outside the rfc8949-core-det-nfc-int-v0 data model."""


def encode(value: Any) -> bytes:
    """Encode one value as deterministic CBOR bytes per the §3.1 profile."""
    return _encode(value, 0)


def _encode(value: Any, depth: int) -> bytes:
    if value is False:
        return b"\xf4"
    if value is True:
        return b"\xf5"
    if value is None:
        return b"\xf6"
    if type(value) is int:
        return _encode_int(value)
    if type(value) is str:
        return _encode_str(value)
    if type(value) is bytes:
        return _head(2, len(value)) + value
    if type(value) is list:
        if depth >= _MAX_DEPTH:
            raise CborProfileError(f"container nesting exceeds {_MAX_DEPTH} levels")
        return _head(4, len(value)) + b"".join(_encode(item, depth + 1) for item in value)
    if type(value) is dict:
        if depth >= _MAX_DEPTH:
            raise CborProfileError(f"container nesting exceeds {_MAX_DEPTH} levels")
        return _encode_map(value, depth)
    raise CborProfileError(
        f"value of type {type(value).__name__} is outside the {PROFILE_ID} data model: {value!r}"
    )


def _encode_str(value: str) -> bytes:
    try:
        data = unicodedata.normalize("NFC", value).encode("utf-8")
    except UnicodeEncodeError as exc:
        # Lone surrogates are not encodable UTF-8 and thus outside major
        # type 3 — a profile violation, never a leaked UnicodeEncodeError.
        raise CborProfileError(f"text string is not valid UTF-8: {exc}") from exc
    return _head(3, len(data)) + data


def _encode_int(value: int) -> bytes:
    if 0 <= value <= _MAX_ARG:
        return _head(0, value)
    if -(2**64) <= value < 0:
        return _head(1, -1 - value)
    raise CborProfileError(
        f"integer {value} exceeds the basic CBOR argument range "
        f"[-2**64, 2**64 - 1]; bignum tags are forbidden by {PROFILE_ID}"
    )


def _encode_map(value: dict[Any, Any], depth: int) -> bytes:
    entries: dict[bytes, bytes] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise CborProfileError(
                f"map keys must be text strings, got {type(key).__name__}: {key!r}"
            )
        encoded_key = _encode_str(key)
        if encoded_key in entries:
            raise CborProfileError(f"duplicate map key after NFC normalization: {key!r}")
        entries[encoded_key] = _encode(item, depth + 1)
    return _head(5, len(entries)) + b"".join(key + item for key, item in sorted(entries.items()))


def _head(major: int, argument: int) -> bytes:
    if argument < 24:
        return bytes([(major << 5) | argument])
    for additional_info, size in ((24, 1), (25, 2), (26, 4), (27, 8)):
        if argument < 1 << (8 * size):
            return bytes([(major << 5) | additional_info]) + argument.to_bytes(size, "big")
    raise CborProfileError(f"argument {argument} exceeds the 8-byte CBOR head")

"""Canonical JSON serialization (RFC001 §35.1).

UTF-8, lexicographically sorted object keys, no insignificant whitespace,
stable array order, no floating-point values (no `NaN`/`Infinity`, no
binary floats for lengths — RFC001 §9.4, §12.1). This is the no-float
subset of RFC 8785 (JCS); it is implemented directly rather than via an
external dependency because the Canonical AST never contains floats.

Object keys are sorted by UTF-16 code-unit sequence, per RFC 8785 §3.2.3
— NOT by raw Unicode code point. The two orderings diverge for keys
outside the Basic Multilingual Plane: a supplementary-plane character
(encoded as a UTF-16 surrogate pair, first code unit 0xD800-0xDBFF)
sorts *before* BMP characters in the 0xE000-0xFFFF range, even though
its code point is numerically larger. `json.dumps(sort_keys=True)`
sorts by Python `str` comparison (code point), which gets this
backwards. The recursive encoder therefore sorts keys explicitly, and
the `json.dumps` fast path is gated (`_fast_encodable`) to documents
whose keys hold no code unit at or above U+D800 — the range where the
two orderings can differ.
"""

from __future__ import annotations

import json
from typing import Any


class CanonicalJsonError(ValueError):
    """Raised when a value cannot be represented in canonical JSON."""


def to_canonical_json(doc: Any) -> bytes:
    """Serialize `doc` to canonical JSON bytes per RFC001 §35.1.

    Raises CanonicalJsonError if `doc` contains any float (including
    NaN/Infinity) at any depth, or a string that cannot be encoded as
    UTF-8 (e.g. an unpaired UTF-16 surrogate). Float rejection needs no
    dedicated tree walk: `float` is absent from `_FAST_TYPES`, so any
    float sends its whole document down `_encode`, which raises the
    moment it reaches that leaf in `_encode_scalar` — one traversal, not
    two.
    """
    try:
        if _fast_encodable(doc):
            return json.dumps(
                doc,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        return _encode(doc)
    except UnicodeEncodeError as exc:
        raise CanonicalJsonError(
            f"cannot encode string as UTF-8 (unpaired surrogate?): {exc}"
        ) from exc


#: Types the fast path may hand to `json.dumps`. Matched by identity, not
#: `isinstance`: an `int` subclass with its own `__str__` would encode as
#: `str(value)` under `_encode_scalar` and as `int.__repr__` under
#: `json.dumps`, and a `tuple` — which `_encode_scalar` refuses — would
#: silently become a JSON array. Both take the slow path and keep their
#: current behaviour exactly.
_FAST_TYPES = frozenset({dict, list, str, int, bool, type(None)})


def _fast_encodable(value: Any) -> bool:
    """May `json.dumps` produce byte-identical output for `value`?

    `json.dumps(sort_keys=True)` sorts by Python string comparison (code
    point) where RFC 8785 §3.2.3 requires UTF-16 code-unit order. The two
    agree exactly while no key contains a code unit at or above U+D800,
    so that — conservatively, over whole keys — is what this checks.
    Escaping needs no check: `_encode_string` IS `json.dumps` with the
    same arguments.

    A false answer costs one wasted traversal and the recursive encoder;
    a false *positive* would change every identity in the project, so
    every judgement here is made in the safe direction.
    """
    if type(value) not in _FAST_TYPES:
        return False
    if type(value) is dict:
        return all(
            type(key) is str
            and (key.isascii() or max(key) < "\ud800")
            and _fast_encodable(item)
            for key, item in value.items()
        )
    if type(value) is list:
        return all(_fast_encodable(item) for item in value)
    return True


def _encode(value: Any) -> bytes:
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise CanonicalJsonError(f"object key must be a string, got {type(key).__name__}")
        items = sorted(value.items(), key=lambda kv: _utf16_sort_key(kv[0]))
        parts = [_encode_string(key) + b":" + _encode(item) for key, item in items]
        return b"{" + b",".join(parts) + b"}"
    if isinstance(value, list):
        return b"[" + b",".join(_encode(item) for item in value) + b"]"
    return _encode_scalar(value)


def _utf16_sort_key(key: str) -> bytes:
    """Bytes whose lexicographic order matches UTF-16 code-unit order.

    UTF-16BE encoding puts the most significant byte of each code unit
    first, so comparing the encoded bytes is equivalent to comparing the
    sequence of UTF-16 code units numerically.
    """
    return key.encode("utf-16-be", "surrogatepass")


def _encode_string(value: str) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def _encode_scalar(value: Any) -> bytes:
    if value is None:
        return b"null"
    if isinstance(value, bool):
        return b"true" if value else b"false"
    if isinstance(value, int):
        return str(value).encode("ascii")
    if isinstance(value, str):
        return _encode_string(value)
    if isinstance(value, float):
        raise CanonicalJsonError(f"floating-point value not allowed in canonical JSON: {value!r}")
    raise CanonicalJsonError(f"unsupported value type in canonical JSON: {type(value)!r}")

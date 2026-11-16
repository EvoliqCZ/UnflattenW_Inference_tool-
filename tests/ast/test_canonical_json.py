import math

import pytest

from unflatten.ast.canonical_json import (
    CanonicalJsonError,
    _fast_encodable,
    _utf16_sort_key,
    to_canonical_json,
)


def test_to_canonical_json_sorts_object_keys():
    result = to_canonical_json({"b": 1, "a": 2})
    assert result == b'{"a":2,"b":1}'


def test_to_canonical_json_nested_objects_sort_recursively():
    result = to_canonical_json({"z": {"y": 1, "x": 2}, "a": 1})
    assert result == b'{"a":1,"z":{"x":2,"y":1}}'


def test_to_canonical_json_no_insignificant_whitespace():
    result = to_canonical_json({"a": [1, 2, 3], "b": "x"})
    assert b" " not in result
    assert b"\n" not in result


def test_to_canonical_json_preserves_array_order():
    result = to_canonical_json({"a": [3, 1, 2]})
    assert result == b'{"a":[3,1,2]}'


def test_to_canonical_json_returns_utf8_bytes():
    result = to_canonical_json({"a": "héllo é"})
    assert isinstance(result, bytes)
    assert result.decode("utf-8") == '{"a":"héllo é"}'


def test_to_canonical_json_rejects_float():
    with pytest.raises(CanonicalJsonError):
        to_canonical_json({"a": 1.5})


def test_to_canonical_json_rejects_nan():
    with pytest.raises(CanonicalJsonError):
        to_canonical_json({"a": math.nan})


def test_to_canonical_json_rejects_infinity():
    with pytest.raises(CanonicalJsonError):
        to_canonical_json({"a": math.inf})


def test_to_canonical_json_rejects_float_nested_in_array():
    with pytest.raises(CanonicalJsonError):
        to_canonical_json({"a": [1, 2.0]})


def test_to_canonical_json_accepts_int_and_bool():
    result = to_canonical_json({"a": 1, "b": True, "c": None})
    assert result == b'{"a":1,"b":true,"c":null}'


def test_to_canonical_json_deterministic_across_calls():
    doc = {"b": 1, "a": {"y": 2, "x": [1, 2, 3]}}
    assert to_canonical_json(doc) == to_canonical_json(doc)


def test_to_canonical_json_sorts_keys_by_utf16_code_unit_not_code_point():
    # RFC 8785 (JCS) requires UTF-16 code-unit ordering. "😀" (U+1F600) is
    # a surrogate pair starting with the code unit 0xD83D, which sorts
    # *before* the BMP character "￿" (U+FFFF) in UTF-16 order even though
    # U+1F600 > U+FFFF as a raw Unicode code point. json.dumps(sort_keys=
    # True) sorts by Python str comparison (code point), which would get
    # this backwards.
    result = to_canonical_json({"\U0001f600": 1, "￿": 2})
    assert result == '{"\U0001f600":1,"￿":2}'.encode("utf-8")


def test_to_canonical_json_surrogate_pair_key_before_high_bmp_key():
    keys_in_expected_order = ["\U0001f600", "￿"]
    doc = {"￿": "b", "\U0001f600": "a"}
    result = to_canonical_json(doc).decode("utf-8")
    assert result.index(keys_in_expected_order[0]) < result.index(keys_in_expected_order[1])


def test_to_canonical_json_wraps_lone_surrogate_as_canonical_json_error():
    with pytest.raises(CanonicalJsonError):
        to_canonical_json({"a": "\ud800"})


def test_to_canonical_json_wraps_lone_surrogate_key_as_canonical_json_error():
    # The value-side twin is above. A lone surrogate in a KEY takes a
    # different route: `_fast_encodable` rejects it at the `max(key) <
    # "\ud800"` guard, the slow path sorts it through `surrogatepass`,
    # and `_encode_string` is where the UTF-8 encode finally fails.
    with pytest.raises(CanonicalJsonError):
        to_canonical_json({"\ud800": 1})


def test_fast_path_rejects_keys_whose_two_orderings_disagree():
    # The whole fast path rests on one claim: `json.dumps(sort_keys=True)`
    # (code-point order) and RFC 8785 §3.2.3 (UTF-16 code-unit order)
    # agree for every dict `_fast_encodable` accepts. A differential test
    # against `_encode` cannot check this — a key the gate rejects sends
    # both sides down the same slow path — so both halves are pinned here.
    straddling = {"\U0001f600": 1, "\uffff": 2}
    keys = list(straddling)

    # Half one: the two orderings genuinely disagree for this pair.
    assert sorted(keys) == ["\uffff", "\U0001f600"]
    assert sorted(keys, key=_utf16_sort_key) == ["\U0001f600", "\uffff"]

    # Half two: the gate refuses to let that dict reach `json.dumps`.
    assert _fast_encodable(straddling) is False
    # Either half of the pair alone is refused too — the gate judges keys
    # one at a time and conservatively, not by comparing them.
    assert _fast_encodable({"\U0001f600": 1}) is False
    assert _fast_encodable({"\uffff": 2}) is False
    # And the largest key the gate does accept is below the surrogate block.
    assert _fast_encodable({"\ud7ff": 1}) is True


# ------------------------------------------------- reference equivalence --
# `to_canonical_json` takes a stdlib-encoder fast path for the documents
# that can use it (str keys with no code unit at or above U+D800, exact
# JSON scalar types) and falls back to `_encode` otherwise. The two must
# agree byte for byte: every identity in the project — `ast_hash`,
# `target_bundle_hash`, `release_root_hash` — is a digest of this output.


def _reference(doc):
    """The recursive encoder, bypassing the fast path."""
    from unflatten.ast.canonical_json import _encode

    return _encode(doc)


@pytest.mark.parametrize(
    "doc",
    [
        {},
        [],
        {"a": 1},
        {"b": [1, {"d": None, "c": True}], "a": "x"},
        {"": ""},
        {"\u00e4": 1, "a": 2, "Z": 3, "": 4},
        {"a": "\\ \" \n \t \u0000"},
        {"a": "h\u00e9llo \U0001f600"},
        {"n": -12345678901234567890},
        {"\U0001f600": 1, "\uffff": 2},
        [[[[{"a": [1, 2, [3]]}]]]],
    ],
)
def test_to_canonical_json_matches_the_reference_encoder(doc):
    assert to_canonical_json(doc) == _reference(doc)


def test_to_canonical_json_rejects_a_non_string_key():
    with pytest.raises(CanonicalJsonError):
        to_canonical_json({1: "a"})


def test_to_canonical_json_rejects_an_unsupported_type():
    with pytest.raises(CanonicalJsonError):
        to_canonical_json({"a": (1, 2)})


def test_to_canonical_json_matches_the_reference_for_an_int_subclass():
    class Weird(int):
        def __str__(self):  # a subclass may disagree with int.__repr__
            return "weird"

    assert to_canonical_json({"a": Weird(3)}) == _reference({"a": Weird(3)})

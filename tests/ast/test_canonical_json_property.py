"""Property-based equivalence of the fast path and the reference encoder.

`to_canonical_json` picks a stdlib fast path for the documents that can
use it. Nothing about any identity in this project survives the two
disagreeing, so they are compared on arbitrary documents — including the
ones both must refuse.

What this test can and cannot do: an *ordering* divergence is out of its
reach by construction, because any key the gate rejects sends both sides
down the same slow path and the comparison is then trivially true. Its
power is over the other direction — a document the gate wrongly accepts,
or an input the two paths refuse differently. The gate's ordering claim
is pinned directly instead, by
`test_fast_path_rejects_keys_whose_two_orderings_disagree` in
`test_canonical_json.py`.

`st.text()`'s default alphabet is `characters(codec="utf-8")`, which can
never draw a surrogate, so "the ones both must refuse" is only true if
the refused inputs are mixed in on purpose: lone surrogates (no UTF-8
encoding), floats, and non-`str` object keys.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from unflatten.ast.canonical_json import _encode, to_canonical_json

#: Characters that sit on the boundaries this module cares about: the two
#: ends of the surrogate block (never encodable as UTF-8, in a key or a
#: value), a supplementary-plane character and a high-BMP one (the pair
#: whose two sort orders disagree), and the code points immediately below
#: and above the surrogate block (the last key the gate accepts, and the
#: first it does not).
_edge_chars = st.sampled_from(
    [
        "\ud800",
        "\udbff",
        "\udc00",
        "\udfff",
        "\U0001f600",
        "\U00010000",
        "\uffff",
        "\ud7ff",
        "\ue000",
    ]
)
_strings = st.text(max_size=6) | st.lists(_edge_chars, max_size=3).map("".join)
#: Non-`str` keys are refused by both paths and must be refused alike.
_keys = _strings | st.integers() | st.booleans() | st.none()
_scalars = st.none() | st.booleans() | st.integers() | st.floats() | _strings
_docs = st.recursive(
    _scalars,
    lambda child: st.lists(child, max_size=4) | st.dictionaries(_keys, child, max_size=4),
    max_leaves=60,
)


def _reference(doc):
    """What `to_canonical_json` would return with no fast path at all.

    `_encode` alone already refuses a float the moment it reaches
    `_encode_scalar` at that leaf — no separate pre-pass needed."""
    return _encode(doc)


@settings(max_examples=2000, deadline=None)
@given(_docs)
def test_to_canonical_json_matches_the_reference_encoder_on_arbitrary_documents(doc):
    try:
        expected = _reference(doc)
    except (UnicodeEncodeError, ValueError):
        expected = None
    try:
        got = to_canonical_json(doc)
    except ValueError:
        got = None
    assert got == expected


@settings(max_examples=100, deadline=None)
@given(_docs, st.integers(min_value=1, max_value=200), st.booleans())
def test_to_canonical_json_matches_the_reference_encoder_on_deeply_nested_documents(
    doc, depth, wrap_in_lists
):
    # `st.recursive` bottoms out around six levels whatever `max_leaves`
    # says, so depth is supplied explicitly. Both paths recurse per level;
    # 200 stays well inside either one's headroom.
    for _ in range(depth):
        doc = [doc] if wrap_in_lists else {"k": doc}
    try:
        expected = _reference(doc)
    except (UnicodeEncodeError, ValueError):
        expected = None
    try:
        got = to_canonical_json(doc)
    except ValueError:
        got = None
    assert got == expected

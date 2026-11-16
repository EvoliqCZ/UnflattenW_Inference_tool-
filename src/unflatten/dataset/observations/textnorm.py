"""Pairing-key normalization for the aligner (spec-zarovnanie §0g).

The PDF text layer and the AST differ in ways that are invisible to a
reader but fatal to exact matching: a non-breaking space or a soft
hyphen where a line wrapped, a ligature where the font rendered one, an
uppercase glyph where a caps-styled run stores the lowercase source.
This module computes the KEY both sides are matched on — never the
stored text. The key is a positional 1:1 transform of the word: the
returned ``key_to_raw`` map says which raw index each key character
came from, and the aligner keeps its boxes in the original offset
space (normalization is for pairing only, the same rule as 0a).

Transforms, and nothing else:

* ``invisible`` — a character drops out of the key when ``ch.isspace()``
  or its Unicode category is ``Zs``/``Cc``/``Cf`` (non-breaking space,
  zero-width joiner, soft hyphen, BOM, DEL). Both sides.
* ``lig`` — the ligatures U+FB00–FB06 are NFKC-expanded per character
  (``ff`` ``fi`` ``fl`` ``ffi`` ``ffl``). Both sides. NO global NFKC or
  NFKD: that would also rewrite ², ℃ and friends, which the measured
  control set (50 ``train_ready`` documents: 0 gained / 0 lost / 0
  moved for this key) does not want touched.
* ``caps`` — AST side only, per character inside the run's
  ``all_caps``/``small_caps`` spans: the renderer paints caps runs
  uppercase, the AST stores the source text. :func:`apply_caps`
  applies it; a word crossing a caps / non-caps boundary is uppercased
  only in its caps part, never forced whole.

Order: caps → invisible/lig (the order the A/B measurement validated).

``casefold`` is deliberately NOT in the key: on the same control set it
moved 4 nodes — a silent relocation, which is a false match — and a
test in the suite pins that out.
"""

from __future__ import annotations

import unicodedata

#: The only NFKC expansions applied: the Latin ligature block
#: U+FB00–FB06, per character. ``unicodedata.normalize("NFKC", ch)`` on
#: a single character is the exact per-character expansion.
_LIGATURE_FIRST = 0xFB00
_LIGATURE_LAST = 0xFB06


def _is_invisible(ch: str) -> bool:
    """A character that carries no glyph and no key character:
    whitespace, or a separator / control / format character
    (``Zs``/``Cc``/``Cf``) — NBSP, ZWJ, soft hyphen, BOM, DEL."""
    return ch.isspace() or unicodedata.category(ch) in ("Zs", "Cc", "Cf")


def normalize_for_matching(text: str) -> tuple[str, list[int]]:
    """The pairing key of ``text`` and its map back to raw indices.

    Returns ``(key, key_to_raw)`` where ``key_to_raw[i]`` is the index
    in ``text`` of the source character of the ``i``-th key character.
    A ligature expansion maps every key character it produces onto the
    one raw character it came from, so a key can be longer than the
    word it came from.

    Idempotent on its own output: the key carries no invisible
    character and no unexpanded ligature, so
    ``normalize_for_matching(key)`` returns the same key with the
    identity map.
    """
    key_chars: list[str] = []
    key_to_raw: list[int] = []
    for raw_index, ch in enumerate(text):
        if _is_invisible(ch):
            continue
        if _LIGATURE_FIRST <= ord(ch) <= _LIGATURE_LAST:
            expanded = unicodedata.normalize("NFKC", ch)
        else:
            expanded = ch
        for key_char in expanded:
            key_chars.append(key_char)
            key_to_raw.append(raw_index)
    return "".join(key_chars), key_to_raw


def apply_caps(word: str, word_start: int, caps_spans: list[tuple[int, int]]) -> str:
    """The word as the renderer painted it.

    ``word`` sits at ``[word_start, word_start + len(word))`` of the
    paragraph text; only the characters whose position lies inside an
    ``all_caps``/``small_caps`` run span are uppercased. A word crossing
    a caps / non-caps boundary changes only its caps part — the whole
    word is never forced.
    """
    if not caps_spans:
        return word
    chars = list(word)
    for index, ch in enumerate(chars):
        position = word_start + index
        for start, end in caps_spans:
            if start <= position < end:
                chars[index] = ch.upper()
                break
    return "".join(chars)

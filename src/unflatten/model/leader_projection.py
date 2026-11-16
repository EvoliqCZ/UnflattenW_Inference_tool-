"""Canonicalization of form fill-leader runs (DEC-036 point 2, design §5b).

A "fill leader" is a maximal run of one identical character from a closed
list, repeated at least 4 times. Examples:

- ``________`` (underscore, often used for signature lines)
- ``…………`` (horizontal ellipsis, form fields)
- ``........`` (period, dot leaders in tables of contents)
- ``--------`` (hyphen-minus, ruled lines)

In V1, these are canonicalized to exactly 4 repetitions before the text
is tokenized for training. The page image shows the full run, but the
target shows exactly 4 tokens. The model learns "a fill leader → one
canonical leader token", and later stages can expand it to the page width.

This is a model-side target projection only, exactly like DEC-016 B
(empty-paragraph and TAB caps). It never touches the AST, the sidecar,
or the observation layer. The aligner and renderer still see and compare
the full-length runs. Only the tokenized training targets change.

Scope: only ``STANDALONE_V1_PROFILE_ID`` (standalone pages under DEC-033).
Every other profile's targets are byte-identical to their pre-T12 state.

Policy (V1, ratified in DEC-036 / design §5b):

- ``LEADER_CHARS = {"_", "…", ".", "-"}``
- A run is maximal and within one ``ScalarPolicyRange`` (never crosses a
  policy-range boundary, since each range is encoded separately).
- Runs shorter than 4 are left alone (they are ordinary punctuation).
- The canonical form is exactly 4 identical characters.
- Mixed-character runs, spaced patterns (``. . .``), and characters outside
  the closed list are never touched.

Edge cases explicitly tested (T12):

- in-context fusion with a neighbour: ``:____``, ``(____)``, ``____.``, ``....5``
  are allowed; the guarantee is only "no two consecutive identical leader
  tokens" after tokenization, not "exactly one token per leader".
- runs of exactly 4 are unchanged (already canonical).
- runs at the start or end of text are handled correctly.
- the return value carries how many runs were canonicalized and how many
  characters were removed (ledger counters).
"""

from __future__ import annotations

LEADER_CHARS: frozenset[str] = frozenset({"_", "\u2026", ".", "-"})
MIN_RUN: int = 4
CANONICAL_LEN: int = 4
LEADER_POLICY_VERSION: str = "1.0.0"


def canonicalize_leaders(text: str) -> tuple[str, int, int]:
    """Canonicalize fill-leader runs in ``text``.

    Returns:
        A tuple ``(canonical_text, runs_canonicalized, chars_removed)``:

        - ``canonical_text``: text where each maximal run of length ≥ 4 of
          a single character from ``LEADER_CHARS`` is replaced by exactly 4
          copies of that character.
        - ``runs_canonicalized``: how many such runs were found and shortened.
        - ``chars_removed``: how many characters were removed in total (sum over
          each shortened run: ``original_length - 4``).

    Runs of length exactly 4 are counted as canonicalized (they satisfy the
    invariant), but they contribute 0 to ``chars_removed``.
    """
    if not text:
        return text, 0, 0

    chars = list(text)
    n = len(chars)
    runs_canonicalized = 0
    chars_removed = 0
    out: list[str] = []
    i = 0

    while i < n:
        c = chars[i]
        if c not in LEADER_CHARS:
            out.append(c)
            i += 1
            continue

        run_start = i
        while i < n and chars[i] == c:
            i += 1
        run_len = i - run_start

        if run_len >= MIN_RUN:
            runs_canonicalized += 1
            out.append(c * CANONICAL_LEN)
            if run_len > CANONICAL_LEN:
                chars_removed += run_len - CANONICAL_LEN
        else:
            out.append(c * run_len)

    return "".join(out), runs_canonicalized, chars_removed

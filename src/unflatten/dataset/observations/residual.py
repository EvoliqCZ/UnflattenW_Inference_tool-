"""The residual pass (RFC003 design `spec-aligner-spans-notes-images`
§3.2/§4.2): a corridor-bounded, unique-run claim over PDF words no
paragraph's body-pass match spelled, run after the body, story and
frame passes and before fragments are collected.

Only two classes of unmatched paragraph are attempted: table cells
(`table_id is not None`) and paragraphs carrying a `note_ref`
(`note_refs`). Both are lost collaterally today — the table path
verifies whole units (§3.1) and a `note_ref`'s marker glyph breaks the
body pass's exact whole-paragraph match (§4.1) — so recovering their
own, otherwise-unique text is gains-only: nothing this pass claims was
ever a candidate for an existing match, because it only looks at words
no paragraph has already spelled.

**Corridor.** Bounded by the nearest MATCHED neighbour before and after
the target in AST preorder (a stand-in for "(page, y)" — reading order
already tracks preorder for body and table content), skipping a
same-table neighbour whose row range overlaps the target's own (row
cells sit side by side, not in stream order).

**Claim rule.** A target's key must be spelled by exactly one
contiguous run of available (unowned, unexcluded) words inside its
corridor; any other shape (0 or >=2 runs, or a run contested by another
target claimed in the same round) is refused, not approximated.

**Rounds.** A claim marks its paragraph `matched`, which narrows a
neighbouring target's corridor on the next round. Deterministic: AST
preorder, no RNG, at most `MAX_ROUNDS` rounds.

**Single-owner invariant.** `claim_single_owner` (aligner.py) is the
only way a claim is registered — this module never mutates `owners`
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from unflatten.dataset.observations.aligner import (
    AlignerPolicy,
    _consume,
    _keyed_words,
    _Paragraph,
    _span_at,
    _StreamIndex,
    _StreamWord,
    _Walker,
    claim_single_owner,
)
from unflatten.dataset.observations.textnorm import normalize_for_matching

#: Smallest keyed-character length a target is attempted at (§3.2 point
#: 2). Frozen by T0 (m2a-v8) G3 leave-one-out precision census.
#: The G3 gate failed for all lengths below 187 due to repetitive text in
#: the corpus (form fields, repeated headers). Setting this high eliminates
#: the risk of residual_pass mis-assigning words by finding an identical
#: (but incorrect) match elsewhere in the document.
MIN_KEY_CHARS = 187  # frozen by T0 G3, m2a-v8

#: Round budget (§3.2 point 5). Frozen by T0 (m2a-v8).
MAX_ROUNDS = 4  # frozen by T0 G3, m2a-v8


@dataclass
class ResidualResult:
    """What one `residual_pass` call did, for the coverage counters."""

    claimed: list[str] = field(default_factory=list)
    refused_ambiguous: int = 0
    refused_contested: int = 0
    not_found: int = 0
    rounds: int = 0
    invariant_skips: int = 0
    note_marker_matches: int = 0
    claimed_words: list[_StreamWord] = field(default_factory=list)


def _row_range(paragraph: _Paragraph) -> tuple[int, int] | None:
    if paragraph.row_index is None:
        return None
    span = paragraph.row_span or 1
    return (paragraph.row_index, paragraph.row_index + span - 1)


def _rows_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _is_row_neighbor(target: _Paragraph, candidate: _Paragraph) -> bool:
    """Is `candidate` a cell of the same table row (range) as `target`?
    Cells of one row sit side by side, not in stream order, so they
    must not bound the corridor (§3.2 point 3)."""
    if target.table_id is None or candidate.table_id != target.table_id:
        return False
    target_row = _row_range(target)
    candidate_row = _row_range(candidate)
    if target_row is None or candidate_row is None:
        return False
    return _rows_overlap(target_row, candidate_row)


def _corridor_bounds(
    ordered: list[_Paragraph],
    target_index: int,
    word_position: dict[int, int],
    stream_len: int,
) -> tuple[int, int]:
    target = ordered[target_index]
    lo = 0
    for i in range(target_index - 1, -1, -1):
        candidate = ordered[i]
        if not candidate.matched or not candidate.spelled_words:
            continue
        if _is_row_neighbor(target, candidate):
            continue
        lo = word_position[id(candidate.spelled_words[-1])] + 1
        break
    hi = stream_len
    for i in range(target_index + 1, len(ordered)):
        candidate = ordered[i]
        if not candidate.matched or not candidate.spelled_words:
            continue
        if _is_row_neighbor(target, candidate):
            continue
        hi = word_position[id(candidate.spelled_words[0])]
        break
    return lo, hi


def _is_target(paragraph: _Paragraph) -> bool:
    if paragraph.matched or paragraph.has_field:
        return False
    return paragraph.table_id is not None or bool(paragraph.note_refs)


def _marker_key(marker_text: str) -> str:
    key, _key_to_raw = normalize_for_matching(marker_text)
    return key


def _augmented_with_markers(
    paragraph: _Paragraph, keyed: list[tuple[str, int, int]]
) -> list[tuple[str, int, int]]:
    """`keyed` with each `note_ref` marker key spliced in at its seam
    (§4.2 tier 1) — a zero-length ``(key, offset, offset)`` phantom
    entry belonging to no AST word.

    A seam can fall INSIDE one keyed word, not only between two: the
    AST text concatenates the runs either side of the `note_ref` with
    no separator (``"word" + note_ref + "more"`` is the one word
    ``"wordmore"``), so the word carrying the seam is split there. This
    assumes the key is the same length as the raw text at the cut
    point, which holds outside a ligature or invisible character
    landing exactly on the seam.
    """
    pieces = list(keyed)
    for offset, marker_text in sorted(paragraph.note_refs, key=lambda item: item[0]):
        marker_key = _marker_key(marker_text)
        split: list[tuple[str, int, int]] = []
        inserted = False
        for key, start, end in pieces:
            if not inserted and start < offset < end:
                cut = offset - start
                split.append((key[:cut], start, offset))
                if marker_key:
                    split.append((marker_key, offset, offset))
                split.append((key[cut:], offset, end))
                inserted = True
            else:
                split.append((key, start, end))
        if not inserted and marker_key:
            insert_at = next(
                (i for i, (_k, start, _e) in enumerate(split) if start >= offset), len(split)
            )
            split.insert(insert_at, (marker_key, offset, offset))
        pieces = split
    return pieces


def _target_forms(
    paragraph: _Paragraph, *, doubled_markers: bool = False
) -> list[tuple[str, list[tuple[str, int, int]]]]:
    """Candidate ``(key, ast_words)`` forms to try, in order (§4.2):

    1. the plain keys — a failure elsewhere may have been collateral;
    2. tier 1: each `note_ref` marker key inserted at its seam;
    3. with ``doubled_markers`` (policy `v16-align.v1`): each marker key
       inserted twice -- LibreOffice paints the automatic note number AND
       the renderer's `customMarkFollows` text, glued (`news3636`).

    Tier 2 (a displaced marker word skipped anywhere in the run) is not
    a fixed key string — `residual_pass` tries it separately with
    `_find_tier2_runs`.
    """
    keyed = _keyed_words(paragraph)
    plain_key = "".join(key for key, _start, _end in keyed)
    forms = [(plain_key, keyed)]
    if paragraph.note_refs:
        augmented = _augmented_with_markers(paragraph, keyed)
        tier1_key = "".join(key for key, _start, _end in augmented)
        if tier1_key != plain_key:
            forms.append((tier1_key, augmented))
            if doubled_markers:
                twice = _augmented_with_markers(paragraph, augmented)
                forms.append(("".join(key for key, _start, _end in twice), twice))
    return forms


def _find_page_split_runs(key: str, corridor: list[_StreamWord]) -> list[list[_StreamWord]]:
    """Exact note-marker forms across adjacent pages, with unique placement."""
    from unflatten.dataset.observations.aligner import _spans_across_page_breaks

    found: dict[tuple[int, ...], list[_StreamWord]] = {}
    consumed = bytearray(len(corridor))
    for start, word in enumerate(corridor):
        if not word.key or not key.startswith(word.key):
            continue
        pieces = _spans_across_page_breaks([key], corridor, start, consumed)
        if pieces is None:
            continue
        words = [w for first, end in pieces for w in corridor[first:end]]
        if _is_reading_order(words, same_line=True):
            found[tuple(id(w) for w in words)] = words
        if len(found) > 1:
            break
    return list(found.values())


def _is_reading_order(words: list[_StreamWord], *, same_line: bool = False) -> bool:
    """Does this run of words never move UP the page from one word to
    the next?

    A corridor bounded only by the nearest MATCHED neighbour can stay
    wide open when both cells either side of a stray glyph are
    themselves unmatched (§3.2's own worked case: a stray glyph glued
    to a genuinely later line, verified by exact characters all the
    same). A run whose words visit a later line before an earlier one
    is not a reading-order run and is dropped before it is ever
    counted as a candidate — the corridor is a stand-in for "(page,
    y)" and this is the cheapest part of that check that is still
    always safe to apply.

    With ``same_line`` (policy `v16-align.v1`) an upward step between two
    words whose vertical extents overlap is a baseline shift on one line
    (a raised note mark, a smaller font), not a return to an earlier line."""
    for earlier, later in zip(words, words[1:]):
        if later.page_index != earlier.page_index:
            continue
        if later.bbox_ppm[1] < earlier.bbox_ppm[1]:
            if (
                same_line
                and later.bbox_ppm[1] < earlier.bbox_ppm[3]
                and later.bbox_ppm[3] > earlier.bbox_ppm[1]
            ):
                continue
            return False
    return True


def _find_runs(
    target_key: str, words: list[_StreamWord], *, same_line: bool = False
) -> list[list[_StreamWord]]:
    if not target_key:
        return []
    index = _StreamIndex(words)
    runs: list[list[_StreamWord]] = []
    for start in index.candidate_starts(target_key, 0):
        span = _span_at(target_key, words, start)
        if span is not None and _is_reading_order(words[span[0] : span[1]], same_line=same_line):
            runs.append(words[span[0] : span[1]])
    return runs


def _find_tier2_runs(
    ast_words: list[tuple[str, int, int]],
    marker_keys: frozenset[str],
    words: list[_StreamWord],
    *,
    same_line: bool = False,
) -> list[tuple[list[_StreamWord], list[_StreamWord]]]:
    """Tier 2 (§4.2 point 3): the plain keys, where the run may
    additionally contain any number of available words whose key
    EQUALS one of the paragraph's marker keys — a marker read by
    ``pdftotext`` as its own word elsewhere in the line, not glued to
    an AST word.

    Returns ``(real_words, marker_words)`` per candidate run: the real
    words are handed to `_consume` against ``ast_words`` unchanged (so
    the marker never gets a char range); the marker words are only
    removed from the available pool.

    A stream word is only ever skipped when accepting it as the
    target's own next character(s) would break the match — a marker
    key that also happens to be the target's own next word is real
    text, not a displaced marker, and is consumed normally.
    """
    target = "".join(key for key, _start, _end in ast_words)
    if not target:
        return []
    runs: list[tuple[list[_StreamWord], list[_StreamWord]]] = []
    for start in range(len(words)):
        accumulated = ""
        real: list[_StreamWord] = []
        markers: list[_StreamWord] = []
        consumed_in_order: list[_StreamWord] = []
        index = start
        while index < len(words) and len(accumulated) < len(target):
            word = words[index]
            consumed_in_order.append(word)
            if word.key in marker_keys and not target.startswith(accumulated + word.key):
                markers.append(word)
                index += 1
                continue
            accumulated += word.key
            real.append(word)
            index += 1
            if accumulated == target:
                if _is_reading_order(consumed_in_order, same_line=same_line):
                    runs.append((real, markers))
                break
            if not target.startswith(accumulated):
                break
    return runs


def residual_pass(
    walker: _Walker,
    stream: list[_StreamWord],
    excluded_boxes_by_page: dict[int, list[tuple[int, int, int, int]]],
    margins_by_page: dict[int, list[tuple[int, int]]],
    *,
    max_rounds: int = MAX_ROUNDS,
    min_key_chars: int = MIN_KEY_CHARS,
    strict: bool = False,
    policy: AlignerPolicy | None = None,
) -> ResidualResult:
    """Claim unmatched table-cell and `note_ref` paragraphs whose exact
    text is a unique, uncontested run of words no paragraph has spelled
    (§3.2/§4.2). Mutates only paragraphs with ``matched == False`` (via
    `_consume`); never reads or writes a matched paragraph's words.
    """
    owners: dict[int, str] = {}
    for paragraph in walker.paragraphs:
        for word in paragraph.spelled_words:
            owners[id(word)] = paragraph.node_id

    word_position = {id(word): index for index, word in enumerate(stream)}

    def is_excluded(word: _StreamWord) -> bool:
        cx = (word.bbox_ppm[0] + word.bbox_ppm[2]) // 2
        cy = (word.bbox_ppm[1] + word.bbox_ppm[3]) // 2
        for x0, y0, x1, y1 in excluded_boxes_by_page.get(word.page_index, ()):
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return True
        for y0, y1 in margins_by_page.get(word.page_index, ()):
            if y0 <= cy <= y1:
                return True
        return False

    ordered = sorted(walker.paragraphs, key=lambda p: p.preorder)
    result = ResidualResult()
    if policy is not None and policy.note_marker_tolerance and policy.page_break_split:
        from unflatten.dataset.observations.cell_exact import recover_complete_cells

        recover_complete_cells(ordered, stream, owners, is_excluded=is_excluded,
                               result=result, strict=strict)
    # policy `v16-align.v1` (A8): note-marker forms only -- the plain form
    # and every table-cell target keep `min_key_chars`
    tolerant = policy is not None and policy.note_marker_tolerance
    marker_floor = policy.note_marker_min_key_chars if tolerant else min_key_chars

    for round_index in range(max_rounds):
        # (target, ast_words, real_words, marker_words, is_marker_form)
        pending: list[
            tuple[
                _Paragraph,
                list[tuple[str, int, int]],
                list[_StreamWord],
                list[_StreamWord],
                bool,
            ]
        ] = []
        round_ambiguous = 0
        round_not_found = 0

        for target_index, target in enumerate(ordered):
            if not _is_target(target):
                continue
            lo, hi = _corridor_bounds(ordered, target_index, word_position, len(stream))
            corridor = [
                word for word in stream[lo:hi] if id(word) not in owners and not is_excluded(word)
            ]
            claim: (
                tuple[list[tuple[str, int, int]], list[_StreamWord], list[_StreamWord], bool]
                | None
            )
            claim = None
            ambiguous = False
            forms = _target_forms(target, doubled_markers=tolerant)
            for form_index, (key, ast_words) in enumerate(forms):
                if len(key) < (min_key_chars if form_index == 0 else marker_floor):
                    continue
                runs = _find_runs(key, corridor, same_line=tolerant)
                if not runs and form_index > 0 and tolerant and policy.page_break_split:
                    runs = _find_page_split_runs(key, corridor)
                if len(runs) == 1:
                    claim = (ast_words, runs[0], [], form_index > 0)
                    break
                if len(runs) > 1:
                    ambiguous = True
                    break
            if claim is None and not ambiguous and target.note_refs:
                keyed = _keyed_words(target)
                plain_key = "".join(key for key, _s, _e in keyed)
                if len(plain_key) >= marker_floor:
                    marker_keys = frozenset(
                        mk for _offset, marker_text in target.note_refs
                        if (mk := _marker_key(marker_text))
                    )
                    tier2_runs = _find_tier2_runs(
                        keyed, marker_keys, corridor, same_line=tolerant
                    )
                    if len(plain_key) < min_key_chars:
                        # below the plain floor a run is only anchored by
                        # the marker it skipped (A8)
                        tier2_runs = [(real, mk) for real, mk in tier2_runs if mk]
                    if len(tier2_runs) == 1:
                        real, markers = tier2_runs[0]
                        claim = (keyed, real, markers, True)
                    elif len(tier2_runs) > 1:
                        ambiguous = True
            if claim is not None:
                pending.append((target, claim[0], claim[1], claim[2], claim[3]))
            elif ambiguous:
                round_ambiguous += 1
            else:
                round_not_found += 1

        word_claimants: dict[int, list[str]] = {}
        for target, _ast_words, real, markers, _is_marker in pending:
            for word in (*real, *markers):
                word_claimants.setdefault(id(word), []).append(target.node_id)

        round_contested = 0
        claimed_this_round = False
        for target, ast_words, real, markers, is_marker in pending:
            all_words = (*real, *markers)
            if any(len(word_claimants[id(word)]) > 1 for word in all_words):
                round_contested += 1
                continue
            if not claim_single_owner(target.node_id, list(all_words), owners, strict=strict):
                result.invariant_skips += 1
                continue
            _consume(target, ast_words, real)
            target.spelled_words.extend(markers)
            result.claimed.append(target.node_id)
            result.claimed_words.extend(all_words)
            if is_marker:
                result.note_marker_matches += 1
            claimed_this_round = True

        result.rounds = round_index + 1
        result.refused_ambiguous = round_ambiguous
        result.refused_contested = round_contested
        result.not_found = round_not_found
        if not claimed_this_round:
            break

    return result


__all__ = ["ResidualResult", "residual_pass", "MIN_KEY_CHARS", "MAX_ROUNDS"]

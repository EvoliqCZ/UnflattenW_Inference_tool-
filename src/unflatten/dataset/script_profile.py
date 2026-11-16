"""Writing-script profile of a Canonical AST.

Corpus admission needs to know which script a document is written in:
the imported real corpus turned out to be 52 % Arabic, and RTL documents
behave differently enough (bidi, positional shaping, mirrored alignment,
and a PDF text layer that carries presentation forms instead of source
characters) that they are a separate modelling problem from LTR ones.

The AST carries no language field — `Run.language` is optional and the
importer rarely sees one — so the profile is derived from the run text
itself. This is a coarse codepoint-range classifier, not a language
identifier: it answers "which alphabet", never "which language".
"""

from __future__ import annotations

from collections import Counter
from typing import Any

#: Returned when no run text carries a classifiable character.
UNDETERMINED = "undetermined"

#: (first, last) inclusive codepoint ranges per script. Digits, spaces
#: and punctuation are deliberately unclassified — they carry no script
#: signal and would otherwise drown a short document in "latin".
SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "latin": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)),
    "greek": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F)),
    "hebrew": ((0x0590, 0x05FF), (0xFB1D, 0xFB4F)),
    # Arabic proper, supplements, and the presentation forms a PDF text
    # layer emits after shaping.
    "arabic": (
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ),
    "devanagari": ((0x0900, 0x097F),),
    "thai": ((0x0E00, 0x0E7F),),
    "cjk": ((0x3040, 0x30FF), (0x4E00, 0x9FFF), (0xAC00, 0xD7AF)),
}

#: Right-to-left scripts — the ones whose PDF text layer does not round
#: trip to source characters (see RUNBOOK "RTL coverage").
RTL_SCRIPTS: frozenset[str] = frozenset({"arabic", "hebrew"})

_LOOKUP: tuple[tuple[int, int, str], ...] = tuple(
    sorted(
        (lo, hi, name)
        for name, ranges in SCRIPT_RANGES.items()
        for lo, hi in ranges
    )
)


def script_of(char: str) -> str | None:
    """Script name for `char`, or None when it carries no script signal."""
    code = ord(char)
    for lo, hi, name in _LOOKUP:
        if lo <= code <= hi:
            return name
        if code < lo:
            break
    return None


def _run_texts(node: Any, out: list[str]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "run" and isinstance(node.get("text"), str):
            out.append(node["text"])
        for value in node.values():
            _run_texts(value, out)
    elif isinstance(node, list):
        for value in node:
            _run_texts(value, out)


def script_histogram(ast: dict[str, Any]) -> dict[str, int]:
    """Character counts per script across every run in `ast`."""
    texts: list[str] = []
    _run_texts(ast, texts)
    counts: Counter[str] = Counter()
    for text in texts:
        for char in text:
            name = script_of(char)
            if name is not None:
                counts[name] += 1
    return dict(counts)


def dominant_script(ast: dict[str, Any]) -> str:
    """The most frequent script in `ast`, or `UNDETERMINED`.

    Ties break on script name so the result is deterministic."""
    counts = script_histogram(ast)
    if not counts:
        return UNDETERMINED
    return max(sorted(counts), key=lambda name: counts[name])

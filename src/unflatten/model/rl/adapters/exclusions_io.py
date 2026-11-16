"""Decision-file status reader for the exclusion registry (torch-free).

A ``declared_absent`` entry is an attestation, so the decision that carries it
must itself be ratified. ``docs/decisions/DEC-*.md`` headers are not uniform
(``| Stav | **DECIDED** ...``, ``**Stav: DECIDED ...**``, ``**Status: ...**``), so
the status is the first closed-vocabulary keyword on the first header line (in
the first ``HEADER_LINES`` lines) that carries the word ``Stav`` or ``Status``.
No such line, or a status outside ``RATIFIED_STATUSES``, is a refusal -- a
missing status is never read as ratified.
"""

from __future__ import annotations

import re
from typing import Final

HEADER_LINES: Final = 40
RATIFIED_STATUSES: Final = frozenset({"RATIFIED", "DECIDED"})
KNOWN_STATUSES: Final = (*sorted(RATIFIED_STATUSES), "PROPOSED", "OPEN", "REJECTED", "SUPERSEDED")
_LABEL_RE = re.compile(r"\b(?:stav|status)\b", re.IGNORECASE)
_KEYWORD_RE = re.compile(r"\b(" + "|".join(KNOWN_STATUSES) + r")\b")


def decision_status(text: str) -> str | None:
    """The declared status keyword of a decision document, or ``None``."""
    for line in text.splitlines()[:HEADER_LINES]:
        if not _LABEL_RE.search(line):
            continue
        match = _KEYWORD_RE.search(line)
        if match is not None:
            return match.group(1)
    return None

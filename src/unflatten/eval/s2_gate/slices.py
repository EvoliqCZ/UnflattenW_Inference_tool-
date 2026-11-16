"""Two declared sub-population slices for the S2 exit gate (design §5).

Neither declaration exists anywhere else in the repo. Both are named
constants with their reasoning here, both versioned by
`SLICE_POLICY_VERSION`, and both must be reported alongside any measurement
that uses them so a later reader knows what was actually sliced.

**Criterion 4 status — `blocked`, not measured-and-passed.** RFC005
§12.2 makes critical slices (which slices matter, and their regression
margins) a frozen-policy field. Nothing has pinned that policy yet, so
this module can report the two slices' numbers, but the S2 exit-gate
*criterion* built on them (design §4, item 4) must resolve to `blocked`
exactly like criterion 6 — there is no ratified threshold or margin these
numbers can be compared against. A caller must not read a slice number
computed here as a pass/fail verdict (governance review §11 M2).

- **Control event** = the eight single-token leaf controls of
  `unflatten.model.events_vocab.LEAF_TYPES` (`column_break`, `field`,
  `floating_image`, `inline_image`, `line_break`, `note_ref`, `page_break`,
  `tab`) plus `TEXT_BEGIN`/`TEXT_END`. These are the events that carry
  visible layout meaning with no text of their own, so a model can score
  well on running text while losing all of them. `CONTROL_EVENT_TOKENS`
  resolves every name to its id through `events_vocab.TOKEN_IDS` at
  *import time*, so a rename in the token map breaks this module's import
  (and its tests) instead of silently emptying the slice.

- **Diacritic character** = a character whose NFD decomposition contains a
  combining mark (Unicode category `Mn`), while the character itself is
  not a combining mark. Computed with `unicodedata` only — no lookup
  table, no language assumption.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from types import MappingProxyType

from unflatten.model import events_vocab

#: Version of the two slice declarations below. Bump this if either
#: declaration's membership rule changes, so a report can name exactly
#: which policy produced its slice numbers.
SLICE_POLICY_VERSION = "1.0.0"

_CONTROL_TOKEN_STRINGS: Mapping[str, str] = MappingProxyType(
    {
        **{leaf_type: events_vocab.LEAF_TOKEN[leaf_type] for leaf_type in events_vocab.LEAF_TYPES},
        "TEXT_BEGIN": events_vocab.TEXT_BEGIN,
        "TEXT_END": events_vocab.TEXT_END,
    }
)

#: Declared name -> token id, resolved through `events_vocab.TOKEN_IDS` at
#: import time. A `KeyError` here (e.g. from a token-map rename) fails the
#: import loudly instead of leaving the slice silently empty.
CONTROL_EVENT_TOKENS: Mapping[str, int] = MappingProxyType(
    {
        name: events_vocab.TOKEN_IDS[token_string]
        for name, token_string in _CONTROL_TOKEN_STRINGS.items()
    }
)


def is_diacritic(ch: str) -> bool:
    """True if `ch` carries a diacritic: its NFD decomposition contains a
    combining mark (category `Mn`), and `ch` itself is not one.

    A bare combining mark (e.g. U+0301 COMBINING ACUTE ACCENT on its own)
    is excluded by the second condition — it has no base character to
    modify, so counting it as "a diacritic-bearing character" would be
    wrong.
    """
    if len(ch) != 1:
        raise ValueError(f"is_diacritic expects a single character, got {ch!r}")
    if unicodedata.category(ch) == "Mn":
        return False
    decomposed = unicodedata.normalize("NFD", ch)
    return any(unicodedata.category(part) == "Mn" for part in decomposed)

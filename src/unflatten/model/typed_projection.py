"""Typed value projection: the rules that keep decoded typed values inside
what the AST accepts (25.9., found when no v15b page reached a strict build).

Torch-free and pure over plain values, so the same rules serve two callers:
`typed_decode` (every new decode) and the builder chain's opt-in
`headless_defaults` policy (sealed artefacts decoded before these rules
existed). One definition, no drift between the two.
"""

from __future__ import annotations

from typing import Any

from unflatten.model import ast_required

#: The palette's trailing class label for "no colour" (`typed_head_config.
#: palette_domain`); it decodes to JSON null, never the string.
NULL_COLOUR_LABEL = "null"

#: OPTIONAL section keys (absent = off, the ~98 % case) whose leaves have
#: heads but whose PRESENCE has none: the leaf heads are supervised only where
#: the object exists, so emitting them built line numbers and a page frame on
#: every page (25.9.). Omitted until a presence head exists.
OPTIONAL_WITHOUT_PRESENCE_HEAD: dict[str, tuple[str, ...]] = {
    "section": ("/line_numbering", "/page_borders"),
}

FIRST_LINE = "/layout/indent/first_line_mpt"
HANGING = "/layout/indent/hanging_mpt"


def clamp_integer(node_type: str, pointer: str, value: int) -> int:
    """`value` projected into the schema's minimum/maximum for the leaf: an
    unbounded regression decoded -762 for a spacing (minimum 0) and the schema
    refused the whole page."""
    low, high = ast_required.integer_bounds(node_type, pointer)
    if low is not None:
        value = max(value, low)
    if high is not None:
        value = min(value, high)
    return value


def optional_objects_without_presence_head(node_type: str, pointers: Any) -> set[str]:
    """The members of `pointers` under an optional object with no presence head."""
    prefixes = OPTIONAL_WITHOUT_PRESENCE_HEAD.get(node_type, ())
    return {pointer for pointer in pointers if pointer.startswith(prefixes)} if prefixes else set()


def one_first_line_offset(values: dict[str, Any]) -> bool:
    """First-line indent and hanging are one signed first-line offset; the
    semantic validator refuses both non-zero. Two independent heads can predict
    both, so keep the net offset: positive stays `first_line_mpt`, negative
    becomes `hanging_mpt`. Mutates `values`; returns whether it changed them."""
    first_line, hanging = values.get(FIRST_LINE), values.get(HANGING)
    if isinstance(first_line, int) and isinstance(hanging, int) and first_line and hanging:
        net = first_line - hanging
        values[FIRST_LINE], values[HANGING] = max(net, 0), max(-net, 0)
        return True
    return False

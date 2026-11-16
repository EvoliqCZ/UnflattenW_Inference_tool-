"""RFC004 §18.2 structural event alphabet + versioned token map.

The decoder token alphabet is NOT canonical dataset authority (§18.1) —
it projects deterministically onto the RFC003 structural stream
(:mod:`unflatten.model.encode` is the inverse bridge). The concrete token
map is versioned and hash-bound so a Component Lock can pin it (§18.2):
``TOKEN_MAP_HASH = domain_hash("rfc004.token-map.v0", token_map_projection())``.

Alphabet contents (71 tokens):

- ``<TYPE>_OPEN`` / ``<TYPE>_CLOSE`` for the 15 container node types,
- single-token leaf controls ``TAB``, ``LINE_BREAK``, ``PAGE_BREAK``,
  ``COLUMN_BREAK``, ``INLINE_IMAGE`` (grammar-visible leaf events, §18.6;
  they expand to ``open`` + immediate ``close`` in the RFC003 stream),
- ``TEXT_BEGIN`` / ``TEXT_END`` bracketing one text span on the current run,
- ``END_DOCUMENT`` (full-document termination claim, no structural event),
- ``FRAGMENT_BEGIN`` / ``END_FRAGMENT`` (fragment envelope controls),
- ``RESUME_NODE(type)`` / ``SUSPEND_NODE(type)`` boundary claims for the
  14 resumable container types; ``text_frame`` adds only OPEN/CLOSE in
  token-map v0.3.0, as ERRATA-004 fixes the total alphabet at 71.

ONE alphabet spans both capability profiles. §18.2 fixes a *minimum*
("minimalny event alphabet musi vediet reprezentovat"), and §18.4 makes
the Grammar Controller a function of "aktivneho capability profile" — so
profile eligibility is the controller's decision, expressed as a mask
over this alphabet, exactly as §18.5's word "maskovane" says. Two
alphabets would instead make the `word-core-v0` types *absent*, which is
a different (and untestable) claim, and would fork ``TOKEN_MAP_HASH`` into
two Component Lock entries for one model.

:data:`PROFILE_NODE_TYPES` therefore carries the per-profile eligible node
type sets; :func:`node_types` is the accessor. ``word-extended-v2``
features (shapes, equations and complex fields) have no tokens at all —
those are unrepresentable by construction (DEC-003 §5).

Token ids are stable small integers assigned from the canonical
(lexicographic) sort of the token strings; any drift in alphabet, ids or
version changes ``TOKEN_MAP_HASH`` and breaks the golden-pin test.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from unflatten.model.hashing import domain_hash

TOKEN_MAP_VERSION = "0.3.0"
_TOKEN_MAP_DOMAIN = "rfc004.token-map.v0"

# Container node types = node types that own a stack frame (RFC001 §15).
V0_CONTAINER_TYPES: tuple[str, ...] = (
    "cell",
    "document",
    "list",
    "list_item",
    "paragraph",
    "row",
    "run",
    "section",
    "table",
)

# Leaf control node types (RFC004 §18.6).
V0_LEAF_TYPES: tuple[str, ...] = (
    "column_break",
    "inline_image",
    "line_break",
    "page_break",
    "tab",
)

# `word-visual-v1` additions (RFC001 §42.2, DEC-003 §1). Container vs leaf
# is read off the schema, not the name: `header`/`footer`/`footnote`/
# `endnote` hold `children` of story blocks, `hyperlink` holds inline
# children; `note_ref`, `floating_image` and `field` have no `children`
# key at all.
V1_CONTAINER_TYPES: tuple[str, ...] = (
    "endnote",
    "footer",
    "footnote",
    "header",
    "hyperlink",
)
V1_1_CONTAINER_TYPES: tuple[str, ...] = ("text_frame",)
V1_LEAF_TYPES: tuple[str, ...] = (
    "field",
    "floating_image",
    "note_ref",
)

CONTAINER_TYPES: tuple[str, ...] = tuple(
    sorted(V0_CONTAINER_TYPES + V1_CONTAINER_TYPES + V1_1_CONTAINER_TYPES)
)
LEAF_TYPES: tuple[str, ...] = tuple(sorted(V0_LEAF_TYPES + V1_LEAF_TYPES))

#: Node types each capability profile may emit (§18.4 "aktivny capability
#: profile"). The keys are exactly the schema's `profile` enum.
PROFILE_NODE_TYPES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "word-core-v0": frozenset(V0_CONTAINER_TYPES) | frozenset(V0_LEAF_TYPES),
        "word-visual-v1": frozenset(V0_CONTAINER_TYPES + V1_CONTAINER_TYPES)
        | frozenset(LEAF_TYPES),
        "word-visual-v1.1": frozenset(CONTAINER_TYPES) | frozenset(LEAF_TYPES),
    }
)


def node_types(profile: str) -> frozenset[str]:
    """Node types eligible under `profile`."""
    try:
        return PROFILE_NODE_TYPES[profile]
    except KeyError:
        raise ValueError(
            f"unknown profile {profile!r}; supported: {sorted(PROFILE_NODE_TYPES)}"
        ) from None

OPEN_TOKEN: Mapping[str, str] = MappingProxyType({t: f"{t.upper()}_OPEN" for t in CONTAINER_TYPES})
CLOSE_TOKEN: Mapping[str, str] = MappingProxyType(
    {t: f"{t.upper()}_CLOSE" for t in CONTAINER_TYPES}
)
LEAF_TOKEN: Mapping[str, str] = MappingProxyType({t: t.upper() for t in LEAF_TYPES})
# ERRATA-004 fixes the alphabet growth at two structural tokens (69 -> 71):
# text_frame can open/close in a window but has no boundary continuation
# claim in this token-map revision.
RESUMABLE_CONTAINER_TYPES = tuple(t for t in CONTAINER_TYPES if t != "text_frame")
RESUME_TOKEN: Mapping[str, str] = MappingProxyType(
    {t: f"RESUME_NODE({t})" for t in RESUMABLE_CONTAINER_TYPES}
)
SUSPEND_TOKEN: Mapping[str, str] = MappingProxyType(
    {t: f"SUSPEND_NODE({t})" for t in RESUMABLE_CONTAINER_TYPES}
)

TEXT_BEGIN = "TEXT_BEGIN"
TEXT_END = "TEXT_END"
END_DOCUMENT = "END_DOCUMENT"
FRAGMENT_BEGIN = "FRAGMENT_BEGIN"
END_FRAGMENT = "END_FRAGMENT"

TOKENS: frozenset[str] = frozenset(
    (
        *OPEN_TOKEN.values(),
        *CLOSE_TOKEN.values(),
        *LEAF_TOKEN.values(),
        *RESUME_TOKEN.values(),
        *SUSPEND_TOKEN.values(),
        TEXT_BEGIN,
        TEXT_END,
        END_DOCUMENT,
        FRAGMENT_BEGIN,
        END_FRAGMENT,
    )
)

# Stable integer ids from the canonical lexicographic sort.
TOKEN_IDS: Mapping[str, int] = MappingProxyType({name: i for i, name in enumerate(sorted(TOKENS))})


def token_map_projection() -> dict[str, Any]:
    """Hash projection of the versioned token map (id-ordered pairs)."""
    return {
        "token_map_version": TOKEN_MAP_VERSION,
        "tokens": [[name, token_id] for name, token_id in sorted(TOKEN_IDS.items())],
    }


TOKEN_MAP_HASH: str = domain_hash(_TOKEN_MAP_DOMAIN, token_map_projection())

"""XML 1.0 character legality (shared by the importer boundary and the
renderer guard — RFC001's premise that the Canonical AST is
deterministically renderable requires every string that can reach
`w:t` to be valid XML character data).

XML 1.0 `Char` production (excluding the surrogate range, which Python
strings cannot hold as a single codepoint anyway): `\\t`, `\\n`, `\\r`
and `\\x20-\\uD7FF`, `\\uE000-\\uFFFD`, `\\U00010000-\\U0010FFFF`. The
practical forbidden set for text produced by this codebase is the C0
control range minus the three legal whitespace controls:
`\\x00-\\x08`, `\\x0b`, `\\x0c`, `\\x0e-\\x1f`.
"""

from __future__ import annotations

import re

#: `\t` (\x09), `\n` (\x0a), `\r` (\x0d) are legal XML whitespace and
#: MUST NOT be stripped.
ILLEGAL_XML_CHARS_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")


def find_illegal_xml_char(text: str) -> str | None:
    """First XML-1.0-illegal character in `text`, or None."""
    match = ILLEGAL_XML_CHARS_RE.search(text)
    return match.group(0) if match else None


def strip_illegal_xml_chars(text: str) -> tuple[str, bool]:
    """Remove XML-1.0-illegal control characters from `text`.

    Returns `(clean_text, had_illegal_chars)`. `\\t`/`\\n`/`\\r` are
    never touched.
    """
    clean = ILLEGAL_XML_CHARS_RE.sub("", text)
    return clean, clean != text

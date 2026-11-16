"""Deterministic XML serialization backend (RFC002 "Determinism guarantees").

Wraps `lxml.etree` with a fixed OOXML Transitional namespace map, a
single XML declaration policy, and no pretty-printing. Element and
attribute order are never reordered by this module — callers (the
`ooxml/` builders) are responsible for inserting children/attributes
in the deterministic order the spec requires; `lxml.etree` preserves
insertion order for both.
"""

from __future__ import annotations

from lxml import etree

# RFC002 "Package structure": fixed prefix -> URI map for OOXML Transitional.
NSMAP: dict[str, str] = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
}

_XML_DECLARATION = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'


def qn(prefix: str, local: str) -> str:
    """Build a Clark-notation tag/attribute name for `prefix:local`."""
    return f"{{{NSMAP[prefix]}}}{local}"


def make_root(prefix: str, local: str, used_prefixes: list[str]) -> etree._Element:
    """Create a root element declaring only the namespaces actually used.

    `used_prefixes` MUST include `prefix` itself; order is preserved in
    the emitted `xmlns:*` attributes for determinism.
    """
    nsmap = {p: NSMAP[p] for p in used_prefixes}
    root = etree.Element(qn(prefix, local), nsmap=nsmap)
    if "wps" in used_prefixes:
        root.set(qn("mc", "Ignorable"), "wps")
    return root


def serialize(root: etree._Element) -> bytes:
    """Serialize `root` to bytes with the fixed OOXML declaration policy.

    No pretty-print, no timestamps, no rsid. Single declaration format:
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` followed
    by CRLF, matching native Word part encoding.
    """
    body = etree.tostring(root, xml_declaration=False, encoding="UTF-8")
    return _XML_DECLARATION + body

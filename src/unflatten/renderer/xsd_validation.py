"""OOXML schema validation (RFC002 validation layer V4).

Validates a generated package part's XML bytes against the official
ECMA-376 Transitional XSDs vendored under `vendor/ecma376/transitional/`
(see that directory's README). Only `wml.xsd`-rooted parts are in V0
scope (`document.xml`, `numbering.xml`, `styles.xml`, `settings.xml`,
`fontTable.xml`); a part name outside that set is not validated here
(`validate_part_against_xsd` returns `[]`).

The compiled `XMLSchema` is process-wide and built once — `lxml`
schema compilation walks and links the whole `wml.xsd` import graph
(dml-main, dml-wordprocessingDrawing, dml-picture, shared-*, ...),
which is too expensive to repeat per part or per compile.
"""

from __future__ import annotations

import re
from pathlib import Path

from lxml import etree

from unflatten.ast.issues import Issue
from unflatten.renderer.xml_writer import NSMAP

_VENDOR_ROOT = Path(__file__).resolve().parents[3] / "vendor" / "ecma376" / "transitional"
_WML_XSD = _VENDOR_ROOT / "wml.xsd"

# RFC002 "Package structure": every wml.xsd-rooted part -> its root
# local name (all in the `w:` namespace).
_PART_ROOT_ELEMENT: dict[str, str] = {
    "/word/document.xml": "document",
    "/word/numbering.xml": "numbering",
    "/word/styles.xml": "styles",
    "/word/settings.xml": "settings",
    "/word/fontTable.xml": "fonts",
    "/word/footnotes.xml": "footnotes",
    "/word/endnotes.xml": "endnotes",
}

# `word-visual-v1` emits one part per header/footer story, so these are
# numbered rather than fixed names (`hdr`/`ftr`, wml.xsd).
_NUMBERED_PART_ROOT_ELEMENT: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^/word/header\d+\.xml$"), "hdr"),
    (re.compile(r"^/word/footer\d+\.xml$"), "ftr"),
)


def wml_root_element(part_name: str) -> str | None:
    """Root element local name `part_name` must validate as, or `None`
    when the part is not rooted in `wml.xsd` at all.
    """
    fixed = _PART_ROOT_ELEMENT.get(part_name)
    if fixed is not None:
        return fixed
    for pattern, local_name in _NUMBERED_PART_ROOT_ELEMENT:
        if pattern.match(part_name):
            return local_name
    return None

_schema: etree.XMLSchema | None = None


def _get_schema() -> etree.XMLSchema:
    global _schema
    if _schema is None:
        schema_doc = etree.parse(str(_WML_XSD))
        _schema = etree.XMLSchema(schema_doc)
    return _schema


def validate_part_against_xsd(part_name: str, xml_bytes: bytes) -> list[Issue]:
    """Validate one package part's XML bytes against `wml.xsd`.

    `part_name` is the package-absolute path (leading `/`, e.g.
    `/word/document.xml`). Parts `wml_root_element` does not recognize
    are not wml.xsd-rooted (`[Content_Types].xml`, `.rels`, media,
    `docProps/*`) and are not validated here — an empty list, not an
    error.
    """
    if wml_root_element(part_name) is None:
        return []
    try:
        doc = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as exc:
        return [
            Issue(
                code="XSD_VALIDATION_ERROR",
                severity="error",
                node_id=None,
                path=part_name,
                message=f"part is not well-formed XML: {exc}",
                details={"part": part_name},
            )
        ]

    schema = _get_schema()
    validation_doc = _mce_preprocess_ignorable(doc)
    if schema.validate(validation_doc):
        return []
    issues: list[Issue] = []
    for error in schema.error_log:  # type: ignore[attr-defined]
        issues.append(
            Issue(
                code="XSD_VALIDATION_ERROR",
                severity="error",
                node_id=None,
                path=part_name,
                message=f"line {error.line}: {error.message}",
                details={"part": part_name, "line": error.line, "column": error.column},
            )
        )
    return issues


def _mce_preprocess_ignorable(doc: etree._Element) -> etree._Element:
    """Remove unsupported namespaces named by ``mc:Ignorable``.

    ECMA-376 validation happens after Markup Compatibility preprocessing.
    The vendored Transitional schemas intentionally do not define Office's
    ``wps`` extension, so validating the raw extension tree would reject a
    conforming package before the schema gets the required fallback view.
    """
    ignorable_attr = f"{{{NSMAP['mc']}}}Ignorable"
    prefixes = doc.get(ignorable_attr, "").split()
    if not prefixes:
        return doc
    namespaces = {doc.nsmap[prefix] for prefix in prefixes if prefix in doc.nsmap}
    projected = etree.fromstring(etree.tostring(doc))
    projected.attrib.pop(ignorable_attr, None)
    for element in list(projected.iter()):
        if etree.QName(element).namespace in namespaces:
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)
            continue
        for name in list(element.attrib):
            if etree.QName(name).namespace in namespaces:
                del element.attrib[name]
    return projected


def validate_all_parts(parts: dict[str, bytes]) -> list[Issue]:
    """Validate every wml.xsd-rooted part in `parts` (name -> XML bytes).

    `parts` keys are package-relative names without a leading slash
    (as produced by the compiler's `PackagePart.name`); this function
    normalizes to the leading-slash form `_PART_ROOT_ELEMENT` uses.
    """
    issues: list[Issue] = []
    for name, data in parts.items():
        part_name = name if name.startswith("/") else f"/{name}"
        issues.extend(validate_part_against_xsd(part_name, data))
    return issues

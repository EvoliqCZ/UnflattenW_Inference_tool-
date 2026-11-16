# ECMA-376 5th Edition — Part 4 Transitional XML Schemas

Source: https://ecma-international.org/publications-and-standards/standards/ecma-376/
(`ECMA-376-4_5th_edition_december_2016.zip` → `OfficeOpenXML-XMLSchema-Transitional.zip`), downloaded 2026-08-30.

These are the authoritative OOXML Transitional XSDs used by the renderer test suite
(RFC002 validation layer **V4 — OOXML schema validation**): every generated part of a
`.docx` (document.xml, numbering.xml, …) must validate against `transitional/wml.xsd`
and its imports via `lxml.XMLSchema`.

## Local modifications (the only ones)

- `transitional/xml.xsd` added — the W3C schema for the `xml:` namespace
  (https://www.w3.org/2001/xml.xsd), which the ECMA distribution imports without
  shipping.
- `wml.xsd` and `shared-math.xsd`: the bare
  `<xsd:import namespace="http://www.w3.org/XML/1998/namespace"/>` was given
  `schemaLocation="xml.xsd"` so the schemas compile offline (libxml2 does not
  resolve schemaLocation-less imports). No other content was changed.

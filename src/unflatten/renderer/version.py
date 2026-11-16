"""Renderer version constant (RFC002 "Versioning").

Recorded in `RendererResult.renderer_version` and the Package Manifest;
bump on any change that can alter compiled output bytes for the same
input (ordering fix, mapping change, determinism fix, ...).
"""

from __future__ import annotations

# 0.2.0 — `word-visual-v1` (DEC-003): header/footer story parts,
# footnotes/endnotes, anchored images, hyperlinks, nested tables,
# whitelisted fields and advanced section settings. `word-core-v0`
# output bytes are unchanged, but the capability set the manifest
# records is not, and m2a-v3 must be able to tell the two apart.
# 0.3.0 — `word-visual-v1.1` adds deterministic anchored text frames.
# 0.3.1 — explicit ordered-list instance starts prevent LibreOffice
# continuing counters across independent lists sharing an abstract definition.
RENDERER_VERSION = "0.3.1"

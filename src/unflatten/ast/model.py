"""Typed root document model with from_dict/to_dict round-trip.

Design note: only the root object (RFC001 §10) is mirrored as a typed
frozen dataclass. `sections`, `resources` and `extensions` are kept as
plain dicts/lists rather than deeply-typed dataclasses for every one of
the ~14 node types (paragraph, run, table, cell, ...). The JSON Schema
(`schemas/canonical_ast.schema.json`) and `semantic_validator.py` are
the authoritative structural contract for node bodies; duplicating that
shape into a parallel dataclass hierarchy here would be speculative
code with no current consumer. See the implementation report for this
explicit deviation from the literal task wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Root Canonical AST object (RFC001 §10)."""

    ast_version: str
    profile: str
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    resources: list[dict[str, Any]] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    extensions: dict[str, Any] = field(default_factory=dict)
    #: word-visual-v1 footnote stories. Optional-when-empty like every
    #: other key of that profile extension, so a document without
    #: footnotes round-trips byte-identically and keeps its `ast_hash`.
    footnotes: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        for field_name in ("ast_version", "profile", "document_id"):
            if field_name not in data:
                raise ValueError(f"from_dict: missing required root field '{field_name}'")
        return cls(
            ast_version=data["ast_version"],
            profile=data["profile"],
            document_id=data["document_id"],
            metadata=data.get("metadata", {}),
            resources=data.get("resources", []),
            sections=data.get("sections", []),
            extensions=data.get("extensions", {}),
            footnotes=data.get("footnotes", []),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ast_version": self.ast_version,
            "profile": self.profile,
            "document_id": self.document_id,
            "metadata": self.metadata,
            "resources": self.resources,
            "sections": self.sections,
            "extensions": self.extensions,
        }
        if self.footnotes:
            result["footnotes"] = self.footnotes
        return result

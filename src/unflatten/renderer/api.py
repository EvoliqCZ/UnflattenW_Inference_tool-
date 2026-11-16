"""Public renderer contract types (RFC002 "Public API" / "Sidecar artifacts").

`compile_docx` (in `compiler.py`) is the sole entry point; this module
only defines the frozen option/result value objects it takes and
returns, reusing `unflatten.ast.issues.Issue` per RFC002's shared
Issue schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from unflatten.ast.issues import Issue
from unflatten.renderer.fonts import DEFAULT_FALLBACK_MAP

FontPolicyMode = Literal["require_exact", "allow_fallback"]
RendererStatus = Literal["success", "success_with_warnings", "failed"]


@dataclass(frozen=True)
class FontPolicy:
    """Governs how requested fonts outside the pinned pack are handled.

    `require_exact`: a font not in the pilot pack is a compile error
    (`FONT_NOT_IN_PACK`). `allow_fallback`: `fallback_map` substitutes
    a pack font and emits a `FONT_FALLBACK` warning with
    requested/used in `Issue.details`.

    Default is `allow_fallback`: the golden corpus (and any realistic
    document) requests common proprietary fonts (Arial, Times New
    Roman, ...) that are never literally in the pilot pack, so
    `require_exact` would reject virtually all real input by default.
    Callers building a strictly pinned TRAIN_READY pipeline should opt
    into `require_exact` explicitly.
    """

    mode: FontPolicyMode = "allow_fallback"
    fallback_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FALLBACK_MAP))
    #: Pinned substitute for a requested font that is neither in the pack
    #: nor in `fallback_map`. Real corpora request a long tail of faces
    #: (544 distinct unknown names across the imported corpus) that no map
    #: enumerates; without a default those documents cannot compile at
    #: all. Every use is still reported as `FONT_FALLBACK` (`via:
    #: "default"`). Set to `None` to restore the strict map-only
    #: behaviour. `require_exact` ignores this field.
    fallback_default: str | None = "DejaVu Sans"


@dataclass(frozen=True)
class RendererOptions:
    """Configuration for a single `compile_docx` call (RFC002 §"Public API")."""

    profile: str = "word-core-v0"
    strict: bool = True
    deterministic: bool = True
    ooxml_conformance: str = "transitional"
    compatibility_mode: str = "office2013_plus"
    font_policy: FontPolicy = field(default_factory=FontPolicy)
    validate_package: bool = True
    validate_xsd: bool = True
    emit_manifest: bool = False
    max_nodes: int = 100_000
    max_tree_depth: int = 64
    max_text_chars: int = 10_000_000
    max_table_cells: int = 100_000


@dataclass(frozen=True)
class RendererResult:
    """Outcome of a `compile_docx` call (RFC002 §"Public API")."""

    status: RendererStatus
    output_path: Path | None
    ast_hash: str | None
    package_hash: str | None
    renderer_version: str
    profile: str
    ast_version: str | None = None
    issues: list[Issue] = field(default_factory=list)
    approximations: list[Issue] = field(default_factory=list)
    unsupported: list[Issue] = field(default_factory=list)
    parts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

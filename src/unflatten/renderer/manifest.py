"""Package Manifest sidecar (RFC002 "Sidecar artifacts" — Package Manifest).

Not part of the `.docx` package itself; a separate, deterministic JSON
document describing *how* the package was built, for dataset
provenance (RFC003 persists this per training sample). Built from an
already-produced `RendererResult` plus the `RendererOptions` that
produced it — the manifest must reflect the actual configuration used
(`ooxml_conformance`, `compatibility_mode`, font policy), never
restate renderer-wide literals that could silently drift out of sync
with what `compile_docx` was actually called with.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from unflatten.renderer.api import FontPolicy, RendererOptions, RendererResult
from unflatten.renderer.fonts import PILOT_FONT_PACK


def compute_font_pack_hash(font_policy: FontPolicy) -> str:
    """Deterministic `sha256:` digest identifying `font_policy`'s effect
    on rendering, for dataset reproducibility (RFC002 "Fonts" —
    `font_pack_hash`).

    Projection (documented here since it is not otherwise observable
    from the hash alone): canonical JSON of
    `{"pilot_pack": sorted(PILOT_FONT_PACK), "mode": font_policy.mode,
    "fallback_map": font_policy.fallback_map sorted by key,
    "fallback_default": font_policy.fallback_default}`, UTF-8
    encoded, `sort_keys=True`, `separators=(",", ":")` (no whitespace).
    Two `FontPolicy`s with the same mode/fallback_map hash identically
    regardless of dict insertion order; a pilot-pack change (new/removed
    font family) always changes the hash even if no fixture happens to
    use the added/removed family.
    """
    payload = {
        "pilot_pack": sorted(PILOT_FONT_PACK),
        "mode": font_policy.mode,
        "fallback_map": dict(sorted(font_policy.fallback_map.items())),
        # The pinned default substitute is part of the rendering effect:
        # two policies differing only in it produce different pixels.
        "fallback_default": font_policy.fallback_default,
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def build_package_manifest(result: RendererResult, options: RendererOptions) -> dict[str, Any]:
    """Build the manifest dict for a successful (or failed) `result`.

    `ooxml_conformance`/`compatibility_mode`/`font_policy`/
    `font_pack_hash` are read from `options` (the actual configuration
    used for this compile), never hardcoded. `parts`/`issues` are
    always deterministic list orderings (sorted part names, issues in
    `result.issues` order) so `json.dumps(..., sort_keys=True)` on this
    dict is itself byte-stable for a given `result`/`options`.
    """
    issue_counts = {
        "info": 0,
        "warning": 0,
        "error": 0,
        "fatal": 0,
    }
    for issue in result.issues:
        issue_counts[issue.severity] = issue_counts.get(issue.severity, 0) + 1

    return {
        "renderer_version": result.renderer_version,
        "ast_version": result.ast_version,
        "ast_hash": result.ast_hash,
        "package_hash": result.package_hash,
        "profile": result.profile,
        "ooxml_conformance": options.ooxml_conformance,
        "compatibility_mode": options.compatibility_mode,
        "font_policy": {
            "mode": options.font_policy.mode,
            "fallback_map": dict(sorted(options.font_policy.fallback_map.items())),
            "fallback_default": options.font_policy.fallback_default,
        },
        "font_pack_hash": compute_font_pack_hash(options.font_policy),
        "status": result.status,
        "parts": [
            {"name": name, "sha256": digest} for name, digest in sorted(result.parts.items())
        ],
        "issue_counts": issue_counts,
        "issues": [
            {"code": issue.code, "severity": issue.severity, "node_id": issue.node_id}
            for issue in result.issues
        ],
    }


def write_package_manifest(
    result: RendererResult, output_path: Path, options: RendererOptions
) -> Path:
    """Write `build_package_manifest(result, options)` as JSON next to
    `output_path`.

    Sidecar file name is `<output_path>.manifest.json`; returns the
    manifest path. Deterministic: sorted keys, fixed separators, `\n`
    trailing newline, no wall-clock timestamps.
    """
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest = build_package_manifest(result, options)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return manifest_path

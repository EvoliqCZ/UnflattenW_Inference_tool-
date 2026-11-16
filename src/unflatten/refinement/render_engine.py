"""Injectable render boundary for the Relative Fidelity Verifier
(RFC006 §25-26).

`RenderEngine` is a `Protocol` so unit tests inject a deterministic stub
and never need LibreOffice; a real implementation compiles through
`unflatten.renderer.compiler.compile_docx` and rasterizes through the
existing LibreOffice path (`unflatten.dataset.migration.m2b`,
`unflatten.eval.render_pipeline`) exactly like the RFC002 renderer's own
open-probe/eval tooling. Building the real implementation and its pixel
comparison channels (RFC006 §26.5 — edge alignment, SSIM, ...) is out of
scope for W3-12: no source pixels or comparison-profile infrastructure
exists in this package yet, and inventing pixel metrics without them
would be exactly the kind of silent assumption Karpathy principle 1
forbids. `RealRenderEngine` below marks that boundary and is exercised
only by the slow, LibreOffice-gated test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from unflatten.renderer.api import RendererOptions
from unflatten.renderer.compiler import compile_docx


@dataclass(frozen=True)
class RenderObservation:
    """The minimal per-AST render evidence the Verifier's hard gates and
    metric vector need (RFC006 §26.2/§26.3/§26.5, simplified — see module
    docstring for what is deliberately not modeled: real pixel comparison
    channels)."""

    render_ready: bool
    page_count: int
    profile: str
    clipping_overflow_count: int
    unsupported_fallback_count: int
    #: Lower-is-better integer measurements the caller supplies, keyed by
    #: comparison channel name. Tier 1 (targeted) and Tier 2/3
    #: (protected/global) channels are both plain entries here; the
    #: Verifier is told which key is the targeted one per call.
    metrics: dict[str, int]


class RenderEngine(Protocol):
    """Renders one canonical AST into one `RenderObservation`."""

    def render(self, ast: dict[str, Any]) -> RenderObservation: ...


class RealRenderEngine:
    """Compiles `ast` through the real RFC002 renderer.

    Only exercises `compile_docx` (`render_ready`/`page_count` proxy via
    compile success); does not open LibreOffice or produce pixel
    metrics — those channels are out of scope here (module docstring).
    Marked for the slow/optional test only.
    """

    def render(self, ast: dict[str, Any]) -> RenderObservation:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            result = compile_docx(ast, Path(tmp) / "doc.docx", RendererOptions(strict=True))
        render_ready = result.status != "failed"
        return RenderObservation(
            render_ready=render_ready,
            page_count=0,
            profile=ast.get("profile", ""),
            clipping_overflow_count=0,
            unsupported_fallback_count=0,
            metrics={},
        )

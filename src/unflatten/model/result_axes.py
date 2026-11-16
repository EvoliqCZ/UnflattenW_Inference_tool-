"""RFC004 §6.6 three independent success axes.

The Model Result Envelope carries three orthogonal statuses; the runtime
must never mix technical AST validity, renderer readiness and the product
auto-accept decision. Confidence is diagnostics only (§6.6, §36): it must
never change any status — enforced here *by construction*: no field,
method or parameter in this module accepts a confidence value.

Validation is Issue-based (RFC001 §50 model): :meth:`ResultAxes.validate`
returns ``list[Issue]`` (empty means valid) and never raises on invalid
combinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from unflatten.ast.issues import Issue

AstStatus = Literal["strict_valid", "review_candidate", "partial", "failed"]
AST_STATUS: frozenset[str] = frozenset({"strict_valid", "review_candidate", "partial", "failed"})

RendererStatus = Literal["not_run", "render_ready", "approximation_required", "failed"]
RENDERER_STATUS: frozenset[str] = frozenset(
    {"not_run", "render_ready", "approximation_required", "failed"}
)

AcceptanceStatus = Literal["not_evaluated", "auto_accept", "review_required", "rejected"]
ACCEPTANCE_STATUS: frozenset[str] = frozenset(
    {"not_evaluated", "auto_accept", "review_required", "rejected"}
)


@dataclass(frozen=True)
class ResultAxes:
    """One (ast, renderer, acceptance) status triple."""

    ast_status: str
    renderer_status: str
    acceptance_status: str

    def validate(self) -> list[Issue]:
        """Check enum membership and cross-axis invariants (§6.6, §35)."""
        issues: list[Issue] = []
        if self.ast_status not in AST_STATUS:
            issues.append(_issue("MODEL_AST_STATUS_UNKNOWN", "/ast_status", self.ast_status))
        if self.renderer_status not in RENDERER_STATUS:
            issues.append(
                _issue("MODEL_RENDERER_STATUS_UNKNOWN", "/renderer_status", self.renderer_status)
            )
        if self.acceptance_status not in ACCEPTANCE_STATUS:
            issues.append(
                _issue(
                    "MODEL_ACCEPTANCE_STATUS_UNKNOWN",
                    "/acceptance_status",
                    self.acceptance_status,
                )
            )
        if issues:
            return issues
        # §35: the renderer may only run on a strict-valid AST; every other
        # ast.status permits only renderer.status = not_run (reason
        # ast_not_strict).
        if self.ast_status != "strict_valid" and self.renderer_status != "not_run":
            issues.append(
                Issue(
                    code="MODEL_RENDERER_ON_NON_STRICT_AST",
                    severity="error",
                    node_id=None,
                    path="/renderer_status",
                    message=(
                        f"renderer_status {self.renderer_status!r} requires "
                        f"ast_status 'strict_valid', got {self.ast_status!r}; "
                        "only 'not_run' is allowed"
                    ),
                )
            )
        # §33/§34: a non-strict AST (review candidate, partial, failed) can
        # never be auto-accepted.
        if self.ast_status != "strict_valid" and self.acceptance_status == "auto_accept":
            issues.append(
                Issue(
                    code="MODEL_AUTO_ACCEPT_ON_NON_STRICT_AST",
                    severity="error",
                    node_id=None,
                    path="/acceptance_status",
                    message=(
                        "acceptance_status 'auto_accept' requires ast_status "
                        f"'strict_valid', got {self.ast_status!r}"
                    ),
                )
            )
        # §6.6 exact-product-success semantics: auto_accept asserts the
        # end product is deliverable, which requires an actually
        # render-ready result — never a skipped or degraded render.
        if self.acceptance_status == "auto_accept" and self.renderer_status != "render_ready":
            issues.append(
                Issue(
                    code="MODEL_AUTO_ACCEPT_WITHOUT_RENDER_READY",
                    severity="error",
                    node_id=None,
                    path="/acceptance_status",
                    message=(
                        "acceptance_status 'auto_accept' requires renderer_status "
                        f"'render_ready', got {self.renderer_status!r}"
                    ),
                )
            )
        return issues

    def is_exact_product_success(self) -> bool:
        """§6.6: exact product success = strict_valid + render_ready + auto_accept."""
        return (
            self.ast_status == "strict_valid"
            and self.renderer_status == "render_ready"
            and self.acceptance_status == "auto_accept"
        )


def _issue(code: str, path: str, value: str) -> Issue:
    return Issue(
        code=code,
        severity="error",
        node_id=None,
        path=path,
        message=f"unknown status value {value!r}",
    )

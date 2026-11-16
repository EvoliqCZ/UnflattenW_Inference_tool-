"""`PromotionEnvelope` (RFC004 §6.6/§35.3/§35.4, design §2.1/§2.2, T5a/T5b of
`spec-ast-builder-t5-promotion` P5/P6).

Wraps a `ast_builder.BuildResult` (never extends `result_axes.ResultAxes`
itself -- design D5, REJECTED: `ResultAxes` is the ratified confidence-free
three-axis carrier and stays untouched) into the closed `ast`/`renderer`
branch shapes §35.3/§35.4 describe, plus the `failure_report` §42.7 the AST
axis needs whenever it is not `strict_valid`.

**Scope of this module (design §2.1):** only the `strict_valid | failed`
AST branch is modeled -- `review_candidate`/`partial` have no producer here
(design §2.1 REJECTED: modeling `AstReview` now would be a half branch,
without `validation_report`/`ArtifactRef` §35.3 requires the whole of).
Rendering is opt-in (`RenderRequest`): `promote` always repeats RFC002 S1
on `result.ast` via `_reconfirm_s1` regardless of whether a render was
requested (design §2.2 D6), and, when a `RenderRequest` is given and the
AST axis is `strict_valid`, wires it through `compile_docx`
(`_render`/`map_renderer_result`, P6/T5b). Without a request, or when the
AST axis is not `strict_valid`, `renderer.status` stays `not_run`; its
`reason` follows §35.4's rule that a non-strict AST forces
`ast_not_strict` regardless of whether a render was even requested
(`ResultAxes.validate`'s own cross-axis rule, mirrored here so an
unrendered failed build's envelope is never internally inconsistent about
why nothing rendered).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.ast.canonicalizer import CanonicalizationError, canonicalize
from unflatten.ast.hashing import ast_file_hash as compute_ast_file_hash
from unflatten.ast.hashing import ast_hash as compute_ast_hash
from unflatten.ast.issues import Issue
from unflatten.model import ast_builder as ab
from unflatten.model import model_issue_registry as mir
from unflatten.model.hashing import domain_hash
from unflatten.model.result_axes import ResultAxes
from unflatten.renderer.api import RendererOptions, RendererResult
from unflatten.renderer.compiler import compile_docx

RendererReason = Literal[
    "not_requested",
    "ast_not_strict",
    "capability_approximation",
    "renderer_error",
    "capability_unsupported",
]

#: Design §4.9 registry row `rfc004.renderer-options/v0` (plan P6/U15).
#: Owned here: no other module computes a `renderer_options_hash`.
_RENDERER_OPTIONS_DOMAIN = "rfc004.renderer-options/v0"

#: §35.4: "unknown warning code is not exact-safe" -- fail-closed until an
#: owner registers a code here (design §2.2 D6, plan P6, Q3). A
#: `success_with_warnings` result whose warnings are not all in this set
#: never becomes `render_ready` (R7).
EXACT_SAFE_WARNING_CODES: frozenset[str] = frozenset()

_LIMIT_EXCEEDED_PREFIX = "LIMIT_EXCEEDED_"


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """Opt-in render request (design §2.1). `options` is wired into
    `compile_docx` via `_render` when `promote` is called with a request
    and the AST axis is `strict_valid`."""

    output_path: Path
    options: RendererOptions = RendererOptions()


@dataclass(frozen=True, slots=True)
class _AssetResourceProvider:
    """`renderer.resources.ResourceProvider` over `BuildResult.assets`
    (design §2.2 U16): renders straight from the binder's already-resolved
    bytes, never round-tripping through a directory on disk. Structural
    typing satisfies the `ResourceProvider` `Protocol` without inheriting
    it."""

    assets: Mapping[str, bytes]

    def open(self, path: str) -> bytes:
        try:
            return self.assets[path]
        except KeyError:
            raise FileNotFoundError(f"no bound asset at path {path!r}") from None


@dataclass(frozen=True, slots=True)
class ConsistencyMismatch:
    """`map_renderer_result`'s AST-axis outcome (design §2.2 table row 2):
    RFC002's own S1 (`canonicalize` inside `compile_docx`,
    `compiler.py:87-89`) rejected an AST `promote`'s own re-run already
    called strict_valid, or a `LIMIT_EXCEEDED_*`-free renderer failure with
    `ast_hash is None` reached `map_renderer_result` directly. Never a
    renderer failure (R15) -- the caller must turn this into
    `VALIDATOR_CONSISTENCY_MISMATCH` on the AST axis."""

    inner_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AstBranch:
    """§35.3, restricted to the two columns this envelope models
    (`strict_valid` | `failed`)."""

    status: Literal["strict_valid", "failed"]
    canonical_ast: dict[str, Any] | None
    ast_hash: str | None
    ast_file_hash: str | None
    resources: tuple[dict[str, Any], ...]
    assets: Mapping[str, bytes]
    builder_report: dict[str, Any]
    failure_report: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class RendererBranch:
    """§35.4 internal projection. `renderer_gate_report`/`renderer_result`
    are opaque dicts here (design §2.1: a projection, not the full public
    envelope -- Q9)."""

    status: Literal["not_run", "render_ready", "approximation_required", "failed"]
    reason: RendererReason | None
    renderer_gate_report: dict[str, Any] | None
    renderer_result: dict[str, Any] | None
    docx_path: Path | None
    failure_report: dict[str, Any] | None


def _envelope_issue(path: str, message: str) -> Issue:
    return Issue(
        code="RESULT_ENVELOPE_INVALID", severity="error", node_id=None, path=path, message=message
    )


@dataclass(frozen=True, slots=True)
class PromotionEnvelope:
    axes: ResultAxes
    ast: AstBranch
    renderer: RendererBranch
    renderer_options_hash: str | None

    def validate(self) -> list[Issue]:
        """`ResultAxes.validate()` plus the §35.3/§35.4 nullability rules
        design §2.1 pins, each as its own guard (so removing any single
        one is exactly one failing mutant, plan P5)."""
        issues = list(self.axes.validate())
        issues.extend(self._ast_branch_issues())
        issues.extend(self._renderer_branch_issues())
        return issues

    def _ast_branch_issues(self) -> list[Issue]:
        issues: list[Issue] = []
        a = self.ast
        if a.status == "strict_valid":
            if a.canonical_ast is None or a.ast_hash is None or a.ast_file_hash is None:
                issues.append(
                    _envelope_issue(
                        "/ast", "strict_valid requires canonical_ast/ast_hash/ast_file_hash"
                    )
                )
            if a.failure_report is not None:
                issues.append(
                    _envelope_issue(
                        "/ast/failure_report", "strict_valid must not carry a failure_report"
                    )
                )
        elif a.status == "failed":
            if a.canonical_ast is not None or a.ast_hash is not None or a.ast_file_hash is not None:
                issues.append(
                    _envelope_issue(
                        "/ast", "failed requires canonical_ast/ast_hash/ast_file_hash to be None"
                    )
                )
            if a.failure_report is None:
                issues.append(
                    _envelope_issue("/ast/failure_report", "failed requires a failure_report")
                )
        return issues

    def _renderer_branch_issues(self) -> list[Issue]:
        issues: list[Issue] = []
        r = self.renderer
        if r.status == "not_run":
            if r.reason not in ("not_requested", "ast_not_strict"):
                issues.append(
                    _envelope_issue(
                        "/renderer/reason",
                        f"not_run requires reason in {{not_requested, ast_not_strict}}, "
                        f"got {r.reason!r}",
                    )
                )
            if r.renderer_gate_report is not None:
                issues.append(
                    _envelope_issue(
                        "/renderer/renderer_gate_report",
                        "not_run requires renderer_gate_report to be None",
                    )
                )
            if r.renderer_result is not None:
                issues.append(
                    _envelope_issue(
                        "/renderer/renderer_result", "not_run requires renderer_result to be None"
                    )
                )
            if r.docx_path is not None:
                issues.append(
                    _envelope_issue("/renderer/docx_path", "not_run requires docx_path to be None")
                )
            if r.failure_report is not None:
                issues.append(
                    _envelope_issue(
                        "/renderer/failure_report", "not_run requires failure_report to be None"
                    )
                )
        elif r.status == "render_ready":
            if r.reason is not None:
                issues.append(
                    _envelope_issue("/renderer/reason", "render_ready requires reason to be None")
                )
            if r.renderer_gate_report is None or r.renderer_result is None or r.docx_path is None:
                issues.append(
                    _envelope_issue(
                        "/renderer",
                        "render_ready requires renderer_gate_report/renderer_result/docx_path",
                    )
                )
            if r.failure_report is not None:
                issues.append(
                    _envelope_issue(
                        "/renderer/failure_report", "render_ready must not carry a failure_report"
                    )
                )
        elif r.status == "approximation_required":
            if r.reason != "capability_approximation":
                issues.append(
                    _envelope_issue(
                        "/renderer/reason",
                        f"approximation_required requires reason 'capability_approximation', "
                        f"got {r.reason!r}",
                    )
                )
            if r.renderer_gate_report is None:
                issues.append(
                    _envelope_issue(
                        "/renderer/renderer_gate_report",
                        "approximation_required requires renderer_gate_report",
                    )
                )
            # RFC004:2978 literally: the DOCX was physically written but the
            # envelope never carries it here (design §2.1, R14 mutant).
            if r.renderer_result is not None:
                issues.append(
                    _envelope_issue(
                        "/renderer/renderer_result",
                        "approximation_required requires renderer_result to be None",
                    )
                )
            if r.docx_path is not None:
                issues.append(
                    _envelope_issue(
                        "/renderer/docx_path",
                        "approximation_required requires docx_path to be None",
                    )
                )
            if r.failure_report is not None:
                issues.append(
                    _envelope_issue(
                        "/renderer/failure_report",
                        "approximation_required must not carry a failure_report",
                    )
                )
        elif r.status == "failed":
            if r.reason not in ("renderer_error", "capability_unsupported"):
                issues.append(
                    _envelope_issue(
                        "/renderer/reason",
                        f"failed requires reason in {{renderer_error, capability_unsupported}}, "
                        f"got {r.reason!r}",
                    )
                )
            if r.renderer_gate_report is None:
                issues.append(
                    _envelope_issue(
                        "/renderer/renderer_gate_report", "failed requires renderer_gate_report"
                    )
                )
            if r.failure_report is None:
                issues.append(
                    _envelope_issue("/renderer/failure_report", "failed requires a failure_report")
                )
            if r.renderer_result is not None:
                issues.append(
                    _envelope_issue(
                        "/renderer/renderer_result", "failed requires renderer_result to be None"
                    )
                )
            if r.docx_path is not None:
                issues.append(
                    _envelope_issue("/renderer/docx_path", "failed requires docx_path to be None")
                )
        return issues


def _model_issue(code: str, *, details: dict[str, Any], message: str) -> dict[str, Any]:
    """Build one closed ModelIssue dict (§42.1 shape) for a code this module
    raises itself -- never sourced from a `BuilderIssue`, so it cannot go
    through `mir.to_model_issue`."""
    record = mir.MODEL_ISSUE_REGISTRY[code]
    mir.validate_details(code, details)
    return {
        "code": code,
        "category": record.category,
        "severity": record.severity,
        "stage": record.stage,
        "page_ordinal": None,
        "fragment_sequence_index": None,
        "local_node_id": None,
        "node_id": None,
        "node_property_pointer": None,
        "ast_pointer": None,
        "message": message,
        "details": details,
        "recoverability": record.recoverability,
        "policy_action": record.policy_action,
    }


def _consistency_mismatch_failure_report(inner_codes: Sequence[str]) -> dict[str, Any]:
    issue = _model_issue(
        "VALIDATOR_CONSISTENCY_MISMATCH",
        details={"inner_codes": list(inner_codes)},
        message="S1 re-run rejected an AST the build reported as strict_valid",
    )
    return mir.failure_report("ast", [issue])


def _renderer_gate_failure_report(renderer_codes: Sequence[str]) -> dict[str, Any]:
    issue = _model_issue(
        "RENDERER_GATE_FAILED",
        details={"renderer_codes": list(renderer_codes)},
        message="the renderer rejected the strict Canonical AST",
    )
    return mir.failure_report("renderer", [issue])


def _reconfirm_s1(ast: dict[str, Any]) -> dict[str, Any] | ConsistencyMismatch:
    """Design §2.2 D6: `promote` always repeats RFC002 S1 on `result.ast`,
    even when no render is requested -- a refusal or a hash mismatch is an
    AST-axis defect (§34), not merely a renderer concern (R5). Must
    actually re-derive a fresh canonical tree and compare hashes, not
    compare `ast` against itself (the R5 mutant this guards against).

    Returns the freshly reconfirmed tree on success, never `ast` itself
    (should_fix, verify FAIL): `ast_hash` excludes `metadata` (ERRATA-001
    E1), so two trees differing only in `metadata` -- e.g. a NFD title
    `canonicalize` would NFC-normalize (`canonicalizer.py:112-117`) --
    compare hash-equal here even though they are not byte-identical. The
    caller must promote this reconfirmed tree, not `ast`, or the
    envelope's `canonical_ast`/`ast_file_hash` would be computed over a
    tree S1 never actually re-confirmed as canonical."""
    try:
        reconfirmed = canonicalize(ast)
    except CanonicalizationError as exc:
        return ConsistencyMismatch(inner_codes=tuple(i.code for i in exc.issues))
    if compute_ast_hash(reconfirmed) != compute_ast_hash(ast):
        return ConsistencyMismatch(inner_codes=("AST_HASH_MISMATCH",))
    return reconfirmed


def _renderer_options_projection(options: RendererOptions) -> dict[str, Any]:
    return {
        "profile": options.profile,
        "strict": options.strict,
        "deterministic": options.deterministic,
        "ooxml_conformance": options.ooxml_conformance,
        "compatibility_mode": options.compatibility_mode,
        "font_policy_mode": options.font_policy.mode,
        "font_policy_fallback_map": dict(sorted(options.font_policy.fallback_map.items())),
        "font_policy_fallback_default": options.font_policy.fallback_default,
        "validate_package": options.validate_package,
        "validate_xsd": options.validate_xsd,
        "max_nodes": options.max_nodes,
        "max_tree_depth": options.max_tree_depth,
        "max_text_chars": options.max_text_chars,
        "max_table_cells": options.max_table_cells,
    }


def renderer_options_hash(options: RendererOptions) -> str:
    """`domain_hash` over the caller-supplied `RendererOptions` (design
    §2.1/U15, plan P6). `options` itself is never mutated (U17)."""
    return domain_hash(_RENDERER_OPTIONS_DOMAIN, _renderer_options_projection(options))


def _renderer_result_projection(rr: RendererResult) -> dict[str, Any]:
    """§35.4: status/ast_hash/package_hash/renderer_version/profile,
    without `output_path`/`metrics` (design §2.1)."""
    return {
        "status": rr.status,
        "ast_hash": rr.ast_hash,
        "package_hash": rr.package_hash,
        "renderer_version": rr.renderer_version,
        "profile": rr.profile,
    }


def _has_only_safe_warnings(rr: RendererResult) -> bool:
    warning_codes = {i.code for i in rr.issues if i.severity == "warning"}
    return warning_codes <= EXACT_SAFE_WARNING_CODES


def _has_no_blocking_issue(rr: RendererResult) -> bool:
    """§35.4 `render_ready` literally: "no RendererResult/PackageManifest
    issue has severity error or fatal". `_has_only_safe_warnings` alone
    cannot enforce this -- it only inspects `severity == "warning"` issues,
    so a caller-supplied `RendererOptions(strict=False)` that lets
    `compile_docx` write a DOCX and report `success_with_warnings` despite
    an error-severity issue (`compiler.py:107-124`, e.g. `strict=True`'s own
    XSD/resource gate turned off) would otherwise reach `render_ready` with
    a blocking issue still present (verify FAIL R16). `rr.status ==
    "success"` already implies `rr.issues == []` (`compiler.py:131`), so
    this only ever bites the `success_with_warnings` arm."""
    return not any(i.severity in ("error", "fatal") for i in rr.issues)


def _gate_report(
    rr: RendererResult,
    *,
    s0_status: Literal["passed", "failed"],
    s1_status: Literal["passed", "failed", "not_run"],
    s2_s10_status: Literal["passed", "failed", "not_run"],
    options_hash: str | None,
) -> dict[str, Any]:
    """RFC002 stage projection (design §2.1/U15): `S0_limits`/`S1_validate`/
    `S2_S10` each `passed | failed | not_run`, issue codes grouped by
    severity, renderer version and the options hash that produced `rr`."""
    by_severity: dict[str, list[str]] = {}
    for issue in rr.issues:
        by_severity.setdefault(issue.severity, []).append(issue.code)
    return {
        "S0_limits": s0_status,
        "S1_validate": s1_status,
        "S2_S10": s2_s10_status,
        "issue_codes_by_severity": {k: sorted(v) for k, v in by_severity.items()},
        "renderer_version": rr.renderer_version,
        "renderer_options_hash": options_hash,
    }


def map_renderer_result(
    rr: RendererResult, *, options_hash: str | None = None
) -> RendererBranch | ConsistencyMismatch:
    """Pure mapping from `compile_docx`'s `RendererResult` to the internal
    `RendererBranch` shape (design §2.2), evaluated in the design table's
    order."""
    codes = [i.code for i in rr.issues]
    if rr.status == "failed" and rr.ast_hash is None:
        if any(code.startswith(_LIMIT_EXCEEDED_PREFIX) for code in codes):
            gate = _gate_report(
                rr,
                s0_status="failed",
                s1_status="not_run",
                s2_s10_status="not_run",
                options_hash=options_hash,
            )
            return RendererBranch(
                status="failed",
                reason="renderer_error",
                renderer_gate_report=gate,
                renderer_result=None,
                docx_path=None,
                failure_report=_renderer_gate_failure_report(codes),
            )
        # `canonicalize`'s own refusal inside `compile_docx` -- the AST
        # axis's problem (R15), never surfaced as a renderer failure.
        return ConsistencyMismatch(inner_codes=tuple(codes))

    if rr.status == "failed":
        reason: RendererReason = "capability_unsupported" if rr.unsupported else "renderer_error"
        gate = _gate_report(
            rr,
            s0_status="passed",
            s1_status="passed",
            s2_s10_status="failed",
            options_hash=options_hash,
        )
        return RendererBranch(
            status="failed",
            reason=reason,
            renderer_gate_report=gate,
            renderer_result=None,
            docx_path=None,
            failure_report=_renderer_gate_failure_report(codes),
        )

    gate = _gate_report(
        rr,
        s0_status="passed",
        s1_status="passed",
        s2_s10_status="passed",
        options_hash=options_hash,
    )
    if rr.status == "success" or (
        _has_only_safe_warnings(rr)
        and _has_no_blocking_issue(rr)
        and not rr.approximations
        and not rr.unsupported
    ):
        return RendererBranch(
            status="render_ready",
            reason=None,
            renderer_gate_report=gate,
            renderer_result=_renderer_result_projection(rr),
            docx_path=rr.output_path,
            failure_report=None,
        )
    # `rr.status` is not "failed" here (handled above), so this is either a
    # genuine capability approximation (FONT_FALLBACK, an unregistered
    # warning) or -- only reachable via `RendererOptions(strict=False)` --
    # a `success_with_warnings` result that still carries an error/fatal
    # issue. §35.4 only names 4 renderer states and only constrains
    # `render_ready` against blocking issues; it never says a non-`failed`
    # RendererResult must become the envelope's own `failed`. Mapping it to
    # `failed/renderer_error` here would mean `promote()` invents a renderer
    # failure the renderer itself never reported (`rr.status` says
    # otherwise) and would strand the DOCX that `compile_docx` already
    # wrote. `approximation_required` is the closest ratified state that
    # still refuses `render_ready` without asserting a failure that did not
    # happen (design §2.2 row "success_with_warnings ... -> approximation_
    # required" already covers the ordinary warning case; this reuses the
    # same branch for the blocking-issue case rather than adding a new one).
    return RendererBranch(
        status="approximation_required",
        reason="capability_approximation",
        renderer_gate_report=gate,
        renderer_result=None,
        docx_path=None,
        failure_report=None,
    )


def _render(
    canonical_ast: dict[str, Any], render: RenderRequest, assets: Mapping[str, bytes]
) -> tuple[RendererBranch, str | None, ConsistencyMismatch | None]:
    """Runs `compile_docx` and maps its result. Returns `(branch,
    options_hash, mismatch)`; `mismatch` is non-`None` only in the
    theoretically unreachable case `map_renderer_result` still types as
    possible (R15 doc) -- `_reconfirm_s1` already guards `promote`'s own
    call site against ever reaching it in practice."""
    provider = _AssetResourceProvider(assets)
    rr = compile_docx(canonical_ast, render.output_path, render.options, provider)
    options_hash = renderer_options_hash(render.options)
    mapped = map_renderer_result(rr, options_hash=options_hash)
    if isinstance(mapped, ConsistencyMismatch):
        return _renderer_branch_for("failed"), None, mapped
    return mapped, options_hash, None


def _renderer_branch_for(ast_status: str) -> RendererBranch:
    """Used when `promote` will not call `compile_docx` at all -- no
    `RenderRequest` was given, or the AST axis is not `strict_valid` (§35.4:
    the renderer may only run for `ast.status = strict_valid`). `reason`
    follows §35.4's own rule -- `ast_not_strict` whenever the AST axis is
    not `strict_valid`, `not_requested` otherwise."""
    reason: RendererReason = "ast_not_strict" if ast_status != "strict_valid" else "not_requested"
    return RendererBranch(
        status="not_run",
        reason=reason,
        renderer_gate_report=None,
        renderer_result=None,
        docx_path=None,
        failure_report=None,
    )


def _strict_ast_branch(result: ab.BuildResult, canonical_ast: dict[str, Any]) -> AstBranch:
    return AstBranch(
        status="strict_valid",
        canonical_ast=canonical_ast,
        ast_hash=compute_ast_hash(canonical_ast),
        ast_file_hash=compute_ast_file_hash(to_canonical_json(canonical_ast)),
        resources=tuple(canonical_ast.get("resources", ())),
        assets=result.assets,
        builder_report=dict(result.report),
        failure_report=None,
    )


def _failed_ast_branch(result: ab.BuildResult, failure_report: dict[str, Any]) -> AstBranch:
    return AstBranch(
        status="failed",
        canonical_ast=None,
        ast_hash=None,
        ast_file_hash=None,
        resources=(),
        assets=result.assets,
        builder_report=dict(result.report),
        failure_report=failure_report,
    )


def _ast_branch_from_build(
    result: ab.BuildResult,
) -> tuple[AstBranch, Literal["strict_valid", "failed"]]:
    """The AST axis, including the always-on S1 re-run (design §2.2 D6,
    R5): a `built` result whose own `canonicalize()` re-run refuses it or
    disagrees on the hash is `VALIDATOR_CONSISTENCY_MISMATCH`, never
    `strict_valid`, regardless of whether a render was ever requested."""
    if result.status == "built":
        assert result.ast is not None
        reconfirmed = _reconfirm_s1(result.ast)
        if not isinstance(reconfirmed, ConsistencyMismatch):
            return _strict_ast_branch(result, reconfirmed), "strict_valid"
        report = _consistency_mismatch_failure_report(reconfirmed.inner_codes)
        return _failed_ast_branch(result, report), "failed"
    model_issues = [mir.to_model_issue(issue) for issue in result.issues]
    report = mir.failure_report("ast", model_issues)
    return _failed_ast_branch(result, report), "failed"


def promote(result: ab.BuildResult, *, render: RenderRequest | None = None) -> PromotionEnvelope:
    """`result` must be a `BuildResult` -- a
    `review_candidate.ReviewCandidateResult` (never canonicalized, never
    hashed, §33.7 DECIDED) is rejected by type (R10). `render` is opt-in
    (design §2.1); the AST axis's own S1 re-run (§2.2) runs either way."""
    if not isinstance(result, ab.BuildResult):
        raise TypeError(
            f"promote() requires an ast_builder.BuildResult, got {type(result).__name__}"
        )

    ast_branch, ast_status = _ast_branch_from_build(result)
    options_hash: str | None = None

    if ast_status != "strict_valid" or render is None:
        renderer_branch = _renderer_branch_for(ast_status)
    else:
        assert ast_branch.canonical_ast is not None
        renderer_branch, options_hash, mismatch = _render(
            ast_branch.canonical_ast, render, result.assets
        )
        if mismatch is not None:
            ast_status = "failed"
            report = _consistency_mismatch_failure_report(mismatch.inner_codes)
            ast_branch = _failed_ast_branch(result, report)
            options_hash = None

    axes = ResultAxes(
        ast_status=ast_status,
        renderer_status=renderer_branch.status,
        acceptance_status="not_evaluated",
    )
    return PromotionEnvelope(
        axes=axes, ast=ast_branch, renderer=renderer_branch, renderer_options_hash=options_hash
    )

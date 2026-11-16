"""Diagnostic Word stage (design D4/D18, G1/G2/G4/G5/G9; plan T5, P8/P9).

`build_diagnostic_docx` re-projects the ALREADY-sealed Prediction IR hypothesis
through the service's one ratified chain -- the opt-in policies
`headless-defaults.v0` (RFC004-ERRATA-003), `table-geometry.v0`,
`story-kind-dedup.v0` and `recovered-image-materialization.v0`
(RFC004-ERRATA-004; DEC-044 Dodatok B/D), and since DEC-077 (2026-10-16,
RFC004-ERRATA-011) `run-typography-snap.v0`, `image-region-snap.v0`,
`tab-stops-from-regions.v0` and `leader-fill.v1` (which replaces `leader-fill.v0`
of DEC-059 in the service chain), always all of them, never per request,
in `run.process_model_page`'s own chain order -- and the strict AST
builder/renderer; it never re-decodes. Its import surface is the closed
allowlist `run.process_model_page` / `decode.seal_page` / the policy
modules' `apply`/`POLICY_ID`/`POLICY_STATUS`: no direct import of `unflatten.renderer`
and no call to `compile_docx` anywhere in this module
(`test_renderer_separation.py`'s positive assert for this file). The renderer
only ever runs inside `promotion.promote`, itself only reachable through
`run.process_model_page`.

**G1, the IR-safety rule this module exists to keep:** every failure here --
typed page outcome, a builder/renderer axis that never reaches
`render_ready`, or a genuinely unexpected exception -- is caught at THIS
module's boundary (`except Exception` is deliberate here: the diagnostic
Word chain is exactly the "map internal errors to a public outcome" boundary
`.claude/rules/error-handling.md` describes) and turned into a
`DocxStageResult(produced=False, ...)`. The caller (`worker.py`) always still
has the already-sealed Prediction IR bytes; this module never touches them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from unflatten.eval.builder_chain import headless_defaults as hd
from unflatten.eval.builder_chain import image_region_snap as irs
from unflatten.eval.builder_chain import leader_fill_v1 as lf1
from unflatten.eval.builder_chain import recovered_image_materialization as rim
from unflatten.eval.builder_chain import run_typography_snap as rts
from unflatten.eval.builder_chain import story_dedup as sd
from unflatten.eval.builder_chain import tab_stops_from_regions as tsr
from unflatten.eval.builder_chain import table_geometry as tg
from unflatten.eval.builder_chain.run import process_model_page, process_model_pages
from unflatten.eval.s2_gate import sealed
from unflatten.eval.s2_gate.decode import seal_page

_PAGE_STREAM_OUTCOMES = frozenset({"PageStreamInvalid", "AssemblyInvariantViolation"})

#: The chain policies the service applies, in the order `run.process_model_page`
#: applies them (DEC-044 Dodatok B/D, DEC-059, DEC-077): headless-defaults ->
#: run-typography-snap -> image-region-snap -> tab-stops-from-regions -> table-geometry ->
#: (assemble) -> story-dedup -> leader-fill.v1 -> recovered-image. Readiness requires every
#: one RATIFIED, and `docx_chain["policies"]` reports every one, so adding a
#: policy here is a governance change, not a flag.
SERVICE_CHAIN_POLICIES = (hd, rts, irs, tsr, tg, sd, lf1, rim)
#: Both entry points apply the same policies with their original page rasters.
MULTI_PAGE_CHAIN_POLICIES = SERVICE_CHAIN_POLICIES
#: `ChainRow` field carrying each policy's counts (`None` there means "off").
_ROW_COUNTS_FIELD = {
    hd.POLICY_ID: "policy_counts",
    rts.POLICY_ID: "run_typography_snap_counts",
    irs.POLICY_ID: "image_region_snap_counts",
    tsr.POLICY_ID: "tab_stops_counts",
    tg.POLICY_ID: "geometry_counts",
    sd.POLICY_ID: "story_dedup_counts",
    lf1.POLICY_ID: "leader_fill_v1_counts",
    rim.POLICY_ID: "recovered_image_counts",
}


class InMemoryRasterSource:
    """The one page raster a diagnostic job ever owns (design G2/G4 step 2):
    served from memory, never from disk. `run._recover_resources` is the
    sole caller, and calls `.read(document_bundle_id, page_index)` at most
    once per region prediction -- refuses any other bundle id or page
    index outright rather than silently returning the one page it has."""

    def __init__(self, job_id: str, png_bytes: bytes) -> None:
        self._job_id = job_id
        self._png_bytes = png_bytes

    def read(self, document_bundle_id: str, page_index: int) -> bytes:
        if document_bundle_id != self._job_id:
            raise ValueError("InMemoryRasterSource refuses a raster read for a foreign bundle")
        if isinstance(page_index, bool) or not isinstance(page_index, int) or page_index != 0:
            raise ValueError("InMemoryRasterSource serves only the one page at index 0")
        return self._png_bytes


@dataclass(frozen=True, slots=True)
class DocxStageResult:
    """The diagnostic Word chain's outcome (design D4/G9). `docx_bytes` is
    set if and only if `produced` is `True`; `docx_chain` follows the G9
    table (`None` only when the chain never ran at all -- never returned by
    this module, which is only ever called once a chain run was requested)."""

    produced: bool
    docx_bytes: bytes | None
    reason_codes: tuple[str, ...]
    docx_chain: dict[str, Any]


def _loop_guard_detail(
    raw_issues: tuple[Any, ...], *, prefill_chunk_tokens: int | None = None
) -> str | None:
    """`seal_page`'s `detail` (design G4): the loop_guard issue entry a
    terminated decode carries on `RawDocumentHypothesis.issues`, JSON-encoded
    in the same shape `s2_gate.decode._loop_guard_detail` uses, plus (W0-11)
    `prefill_chunk_tokens` when that decode used chunked prefill -- so a
    diagnostic Word output produced under chunking is distinguishable from
    one produced by the single-pass prefix. `None` only when the decode
    carried no loop-guard issue AND chunking was off (today's behaviour,
    unchanged)."""
    payload: dict[str, Any] = {}
    for issue in raw_issues:
        if not isinstance(issue, dict):
            continue
        kind = issue.get("kind")
        if kind not in ("loop_guard_v1", "loop_guard_v2", "loop_guard_v3"):
            continue
        payload["loop_guard"] = {"loop_guard_v1": "v1", "loop_guard_v2": "v2"}.get(kind, "v3")
        payload["loop_guard_counters"] = issue.get("counters")
        if kind == "loop_guard_v3":
            payload["loop_guard_v3_policy_sha256"] = issue.get("policy_sha256")
            payload["loop_guard_v3_arm"] = issue.get("arm")
        if kind in ("loop_guard_v2", "loop_guard_v3"):
            payload["loop_guard_caps_sha256"] = issue.get("caps_sha256")
            payload["loop_guard_interventions"] = issue.get("interventions")
        break
    if prefill_chunk_tokens is not None:
        payload["prefill_chunk_tokens"] = prefill_chunk_tokens
    return json.dumps(payload, sort_keys=True) if payload else None


def diagnostic_docx_capability(runtime: Any) -> bool:
    """D10's fail-closed conjunction, pure over an already-loaded runtime --
    no I/O, no side effects. A runtime this cannot fully inspect (a missing
    attribute) reads as `False`, never raises: capability is proven present,
    never assumed."""
    try:
        decode_policy = runtime.decode_policy
        return bool(
            runtime.tokenizer is not None
            and decode_policy.typed_decode
            and decode_policy.region_decode
            # v3 (DEC-055, worker override only) wraps v2 unchanged
            and decode_policy.loop_guard in ("v2", "v3")
            and runtime.eval_cache_identity
            and runtime.component_lock_hash
            and runtime.selection_policy_hash
            and all(policy.POLICY_STATUS == "RATIFIED" for policy in SERVICE_CHAIN_POLICIES)
        )
    except AttributeError:
        return False


def _policy_entries(
    counts_by_policy: dict[str, dict[str, int]],
    *,
    policies: tuple[Any, ...] = SERVICE_CHAIN_POLICIES,
) -> list[dict[str, Any]]:
    """`docx_chain["policies"]` (DEC-044 Dodatok B): one entry per applied
    policy, `counts` always a dict (`{}` = ran, nothing to edit)."""
    return [
        {
            "policy_id": policy.POLICY_ID,
            "policy_status": policy.POLICY_STATUS,
            "counts": counts_by_policy[policy.POLICY_ID],
        }
        for policy in policies
    ]


def _collapse_invisible_empty_paragraphs(
    page: sealed.PageDecodeResult,
) -> tuple[sealed.PageDecodeResult, dict[str, int]]:
    """Reconcile invisible model spacers before diagnostic assembly only.

    A childless paragraph has no run font. With single spacing, Word gives
    it a reader-default full line in addition to its explicit before/after
    spacing. Repeated invisible model spacers can therefore push a page's
    final text into a mostly empty, vertically centered continuation page.

    Require a direct section child, no content of any kind, implicit single
    spacing and at least 95% `abstain_not_visible` confidence. Keep the node,
    its before/after spacing and all other predictions; only its implicit
    line box becomes one twip. Explicit line heights and visible, uncertain,
    table, story and content-bearing paragraphs retain their semantics.
    The sealed input is never mutated. Counts are attached to the diagnostic
    chain separately from the common builder policies.
    """
    states = {state.local_node_id: state for state in getattr(page, "node_states", ()) or ()}
    if not states:
        return page, {}
    parents = {state.parent_local_node_id for state in states.values()}
    regions = {region.local_node_id: region for region in getattr(page, "region_predictions", ())}
    values = {
        (prediction["local_node_id"], prediction["pointer"]): prediction["value"]
        for prediction in getattr(page, "typed_predictions", ())
    }
    mode_pointer = "/layout/spacing/line/mode"
    eligible: set[str] = set()
    for node_id, state in states.items():
        parent = states.get(state.parent_local_node_id)
        region = regions.get(node_id)
        if (
            state.node_type == "paragraph"
            and parent is not None and parent.node_type == "section"
            and node_id not in parents
            and region is not None and region.node_type == "paragraph"
            and region.alignment_status == "abstain_not_visible"
            and not region.regions and region.confidence_ppm >= 950_000
            and values.get((node_id, mode_pointer)) == "single"
        ):
            eligible.add(node_id)
    if not eligible:
        return page, {}
    predictions = []
    for prediction in page.typed_predictions:
        if prediction["local_node_id"] in eligible and prediction["pointer"] == mode_pointer:
            prediction = {**prediction, "value": "exact"}
        predictions.append(prediction)
    for node_id in sorted(eligible):
        predictions.append({
            "head_id": "diagnostic.invisible-empty-paragraph-line.v0",
            "local_node_id": node_id, "node_type": "paragraph",
            "pointer": "/layout/spacing/line/value_mpt", "value": 50,
        })
    return replace(page, typed_predictions=tuple(predictions)), {
        "invisible_empty_paragraph_line_collapsed": len(eligible),
    }


def _require_policies_ran(
    row: Any, *, policies: tuple[Any, ...] = SERVICE_CHAIN_POLICIES
) -> dict[str, dict[str, int]]:
    """Every policy the service requested must have run: `run.py` reports an
    applied policy as a dict (`{}` when it edited nothing) and an off policy
    as `None`. A `None` here is a broken chain, never "nothing to edit"."""
    counts_by_policy: dict[str, dict[str, int]] = {}
    for policy in policies:
        counts = getattr(row, _ROW_COUNTS_FIELD[policy.POLICY_ID], None)
        if counts is None:
            raise ValueError(f"chain policy {policy.POLICY_ID} was requested but did not run")
        counts_by_policy[policy.POLICY_ID] = dict(counts)
    return counts_by_policy


def _docx_chain_from_row(
    row: Any,
    *,
    policies: tuple[Any, ...] = SERVICE_CHAIN_POLICIES,
    error_class: str | None = None,
    reconstruction_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """G9's table, projected from a `ChainRow` (`run.process_model_page`/
    `run.process_model_pages`) whose policies `_require_policies_ran`
    already checked."""
    counts_by_policy = _require_policies_ran(row, policies=policies)
    chain: dict[str, Any] = {
        "policy_id": hd.POLICY_ID,
        "policy_status": hd.POLICY_STATUS,
        "page_outcome": row.page_outcome,
    }
    if row.policy_counts:
        chain["policy_counts"] = row.policy_counts
    chain["policies"] = _policy_entries(counts_by_policy, policies=policies)
    if row.profile is not None:
        chain["profile"] = row.profile
    if row.envelope is not None:
        chain["ast_status"] = row.envelope.axes.ast_status
        chain["renderer_status"] = row.envelope.renderer.status
        codes: set[str] = set()
        for branch_name in ("ast", "renderer"):
            branch = getattr(row.envelope, branch_name, None)
            report = getattr(branch, "failure_report", None) or {}
            for issue in report.get("issues", ()):
                pointer = issue.get("node_property_pointer") or issue.get("ast_pointer") or "/"
                codes.add(f"{issue['code']}|{pointer}")
                for code in issue.get("details", {}).get("renderer_codes", ()):
                    codes.add(f"{code}|{pointer}")
        if codes:
            chain["failure_codes"] = sorted(codes)
    if getattr(row, "assembler_summary", None) is not None:
        chain["assembler"] = row.assembler_summary
    if error_class is not None:
        chain["error_class"] = error_class
    _record_layout_reconstruction(chain, reconstruction_counts)
    return chain


def _record_layout_reconstruction(
    chain: dict[str, Any], counts: Mapping[str, int] | None,
) -> None:
    if counts:
        chain["reconstruction"] = {
            "policy_id": "invisible-empty-paragraph-line.v0",
            "counts": dict(sorted(counts.items())),
        }


def _partial_docx_chain(
    page: Any, exc: Exception, *, reconstruction_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """P8: partial policy counts for `DOCX_CHAIN_ERROR`, recomputed here
    (never inside `run.py`, which stays unchanged) via second, pure
    `hd.apply` / `tg.apply` calls over the SAME page `seal_page` already
    sealed, in chain order. `story-kind-dedup.v0` and
    `recovered-image-materialization.v0` both edit the assembled document,
    which a failed chain never yields: on, nothing edited (`{}`), exactly as
    `run.py` reports an unreached page. Without a page no policy ever ran, so
    no counts and no `policies` are reported."""
    chain: dict[str, Any] = {
        "policy_id": hd.POLICY_ID,
        "policy_status": hd.POLICY_STATUS,
        "page_outcome": "error",
        "error_class": type(exc).__name__,
    }
    if page is not None:
        try:
            edited, hd_counted = hd.apply(page)
            edited, rts_counted = rts.apply(edited)
            edited, tsr_counted = tsr.apply(edited)
            _, tg_counted = tg.apply(edited)
        except Exception:
            return chain
        hd_counts = dict(sorted(hd_counted.items()))
        if hd_counts:
            chain["policy_counts"] = hd_counts
        # `image-region-snap.v0` needs the page raster, which this boundary does
        # not hold: reported as "on, nothing edited" like the post-assembly policies.
        chain["policies"] = _policy_entries({
            hd.POLICY_ID: hd_counts,
            rts.POLICY_ID: dict(sorted(rts_counted.items())),
            irs.POLICY_ID: {},
            tsr.POLICY_ID: dict(sorted(tsr_counted.items())),
            tg.POLICY_ID: dict(sorted(tg_counted.items())),
            sd.POLICY_ID: {},
            lf1.POLICY_ID: {},
            rim.POLICY_ID: {},
        })
    _record_layout_reconstruction(chain, reconstruction_counts)
    return chain


def build_diagnostic_docx(
    raw: Any,
    prepared: Any,
    png_bytes: bytes,
    *,
    runtime: Any,
    job_id: str,
    work_dir: Path,
    max_docx_bytes: int,
    expected_prediction_ir_hash: str,
) -> DocxStageResult:
    """Run the diagnostic Word chain over an already-sealed hypothesis.

    `expected_prediction_ir_hash` (plan P9): `seal_page`'s own
    `prediction_ir_hash` must equal the Prediction IR artifact this worker
    attempt already published -- any mismatch is `DOCX_CHAIN_ERROR`, never a
    silently-diverging Word document."""
    page: Any = None
    reconstruction_counts: dict[str, int] = {}
    try:
        page = seal_page(
            document_bundle_id=job_id,
            hypothesis=raw,
            eval_cache_identity=runtime.eval_cache_identity,
            visual_features=prepared.visual_features,
            component_lock_hash=runtime.component_lock_hash,
            selection_policy_hash=runtime.selection_policy_hash,
            decode_fn_identity="decode_free_running",
            max_steps=runtime.decode_policy.max_steps,
            detail=_loop_guard_detail(
                raw.issues, prefill_chunk_tokens=runtime.decode_policy.prefill_chunk_tokens
            ),
        )
        if page.prediction_ir_hash != expected_prediction_ir_hash:
            raise ValueError(
                "seal_page prediction_ir_hash does not match the published Prediction IR"
            )
        page, reconstruction_counts = _collapse_invisible_empty_paragraphs(page)
        raster_source = InMemoryRasterSource(job_id, png_bytes)
        row = process_model_page(
            page,
            vocabulary=runtime.vocabulary,
            tokenizer=runtime.tokenizer,
            raster_source=raster_source,
            render_dir=work_dir,
            headless_defaults=True,
            run_typography_snap=True,
            image_region_snap=True,
            tab_stops_from_regions=True,
            table_geometry=True,
            story_dedup=True,
            leader_fill_v1=True,
            recovered_image_materialization=True,
        )
        _require_policies_ran(row)
    except Exception as exc:  # module boundary (design D4/G1): IR is never lost
        return DocxStageResult(
            produced=False,
            docx_bytes=None,
            reason_codes=("DOCX_CHAIN_ERROR",),
            docx_chain=_partial_docx_chain(
                page, exc, reconstruction_counts=reconstruction_counts,
            ),
        )

    if row.page_outcome in _PAGE_STREAM_OUTCOMES:
        return DocxStageResult(
            produced=False,
            docx_bytes=None,
            reason_codes=("DOCX_PAGE_STREAM_INVALID",),
            docx_chain=_docx_chain_from_row(row, reconstruction_counts=reconstruction_counts),
        )
    if row.page_outcome != "terminated":
        return DocxStageResult(
            produced=False,
            docx_bytes=None,
            reason_codes=("DOCX_PAGE_NOT_TERMINATED",),
            docx_chain=_docx_chain_from_row(row, reconstruction_counts=reconstruction_counts),
        )
    envelope = row.envelope
    if envelope is None or envelope.axes.ast_status != "strict_valid":
        return DocxStageResult(
            produced=False,
            docx_bytes=None,
            reason_codes=("DOCX_AST_NOT_STRICT",),
            docx_chain=_docx_chain_from_row(row, reconstruction_counts=reconstruction_counts),
        )
    substituted = _font_substitution_only(envelope.renderer)
    docx_path = envelope.renderer.docx_path
    if substituted:
        # the approximation branch never carries the path (RFC004:2978); the
        # chain wrote it where `_render_request` names it: `<job_id>.docx`
        docx_path = work_dir / f"{job_id.replace('/', '__')}.docx"
        if not docx_path.is_file():
            docx_path = None
    elif envelope.renderer.status != "render_ready":
        docx_path = None
    if docx_path is None:
        return DocxStageResult(
            produced=False,
            docx_bytes=None,
            reason_codes=("DOCX_RENDER_NOT_READY",),
            docx_chain=_docx_chain_from_row(row, reconstruction_counts=reconstruction_counts),
        )

    try:
        work_root = work_dir.resolve(strict=True)
        resolved = docx_path.resolve(strict=True)
        resolved.relative_to(work_root)
        if not resolved.is_file():
            raise ValueError("diagnostic docx path is not a regular file under the work dir")
        size = resolved.stat().st_size
        if size > max_docx_bytes:
            return DocxStageResult(
                produced=False,
                docx_bytes=None,
                reason_codes=("DOCX_OUTPUT_TOO_LARGE",),
                docx_chain=_docx_chain_from_row(row, reconstruction_counts=reconstruction_counts),
            )
        docx_bytes = resolved.read_bytes()
    except (OSError, ValueError) as exc:
        return DocxStageResult(
            produced=False,
            docx_bytes=None,
            reason_codes=("DOCX_CHAIN_ERROR",),
            docx_chain=_docx_chain_from_row(
                row, error_class=type(exc).__name__, reconstruction_counts=reconstruction_counts,
            ),
        )
    chain = _docx_chain_from_row(row, reconstruction_counts=reconstruction_counts)
    reasons: tuple[str, ...] = ("HEADLESS_DEFAULTS_V0_APPLIED",)
    if substituted:
        chain["renderer_approximations"] = sorted(FONT_SUBSTITUTION_CODES)
        reasons += ("DOCX_FONT_SUBSTITUTED",)
    return DocxStageResult(
        produced=True, docx_bytes=docx_bytes, reason_codes=reasons, docx_chain=chain,
    )


#: DEC-044 Dodatok C (owner 2026-09-26): the only renderer approximation a
#: published diagnostic Word may carry is a metric-compatible font
#: substitution (Calibri -> Carlito). Anything else stays withheld.
FONT_SUBSTITUTION_CODES = frozenset({"FONT_FALLBACK"})


def _font_substitution_only(renderer: Any) -> bool:
    """`approximation_required` whose every renderer issue is a font
    substitution; an empty or unknown gate report is never read as one."""
    if getattr(renderer, "status", None) != "approximation_required":
        return False
    gate = getattr(renderer, "renderer_gate_report", None) or {}
    codes = [code for codes in gate.get("issue_codes_by_severity", {}).values() for code in codes]
    return bool(codes) and set(codes) <= FONT_SUBSTITUTION_CODES


@dataclass(frozen=True, slots=True)
class PageAttempt:
    """One page's decode attempt, already timed by the worker (design B1).
    `raw`/`prepared` are present iff the page's decode actually terminated;
    `error_code` is present iff it did not (`raw is None`)."""

    index: int
    seconds: float
    raw: Any | None = None
    prepared: Any | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class PageResult:
    """One page's outcome for the service's `pages[]` status field."""

    index: int
    state: str  # "succeeded" | "failed"
    error_code: str | None
    decode_steps: int | None
    seconds: float


@dataclass(frozen=True, slots=True)
class MultiPageDocxStageResult:
    """The multi-page counterpart of `DocxStageResult` (design B1): `pages`
    always carries one `PageResult` per INPUT page, in order, regardless of
    whether the merged document was produced -- a failed merge still reports
    every page's own decode outcome."""

    produced: bool
    docx_bytes: bytes | None
    reason_codes: tuple[str, ...]
    docx_chain: dict[str, Any]
    pages: tuple[PageResult, ...]


def build_diagnostic_docx_multi(
    attempts: Sequence[PageAttempt],
    *,
    runtime: Any,
    job_id: str,
    work_dir: Path,
    max_docx_bytes: int,
    page_rasters: Mapping[int, bytes] | None = None,
) -> MultiPageDocxStageResult:
    """DEC-044 Dodatok D §1 (B1): the multi-page counterpart of
    `build_diagnostic_docx`. Each already-decoded page is sealed (`seal_page`,
    same as the single-page chain) independently; a page whose decode never
    terminated, or whose own `seal_page` call fails, becomes a `"failed"`
    `PageResult` and a non-`"terminated"` `sealed.PageDecodeResult` for
    `run.process_model_pages` -- which itself turns that into a placeholder
    paragraph rather than dropping the page (module boundary, same G1 IR
    safety rule as the single-page chain: a page failure here never touches
    the already-published per-page Prediction IR artifacts)."""
    if not attempts:
        raise ValueError("build_diagnostic_docx_multi requires at least one page attempt")
    sealed_pages: list[Any] = []
    page_results: list[PageResult] = []
    for attempt in sorted(attempts, key=lambda a: a.index):
        bundle_id = f"{job_id}#{attempt.index}"
        if attempt.raw is None:
            page_results.append(
                PageResult(attempt.index, "failed", attempt.error_code, None, attempt.seconds)
            )
            sealed_pages.append(
                sealed.PageDecodeResult(
                    document_bundle_id=bundle_id, outcome=attempt.error_code or "failed"
                )
            )
            continue
        try:
            page = seal_page(
                document_bundle_id=bundle_id,
                hypothesis=attempt.raw,
                eval_cache_identity=runtime.eval_cache_identity,
                visual_features=attempt.prepared.visual_features,
                component_lock_hash=runtime.component_lock_hash,
                selection_policy_hash=runtime.selection_policy_hash,
                decode_fn_identity="decode_free_running",
                max_steps=runtime.decode_policy.max_steps,
                detail=_loop_guard_detail(
                    attempt.raw.issues,
                    prefill_chunk_tokens=runtime.decode_policy.prefill_chunk_tokens,
                ),
            )
        except Exception:  # module boundary (design D4/G1): one page's own error
            page_results.append(
                PageResult(attempt.index, "failed", "DOCX_CHAIN_ERROR", None, attempt.seconds)
            )
            sealed_pages.append(
                sealed.PageDecodeResult(document_bundle_id=bundle_id, outcome="error")
            )
            continue
        sealed_pages.append(page)
        page_results.append(
            PageResult(
                attempt.index, "succeeded", None, len(attempt.raw.accepted_tokens), attempt.seconds
            )
        )
    pages_tuple = tuple(page_results)
    reconstruction_counts: dict[str, int] = {}
    try:
        reconstructed_pages = []
        for page in sealed_pages:
            reconstructed, counts = _collapse_invisible_empty_paragraphs(page)
            reconstructed_pages.append(reconstructed)
            for name, count in counts.items():
                reconstruction_counts[name] = reconstruction_counts.get(name, 0) + count
        row = process_model_pages(
            reconstructed_pages,
            vocabulary=runtime.vocabulary,
            tokenizer=runtime.tokenizer,
            document_id=job_id,
            render_dir=work_dir,
            headless_defaults=True,
            run_typography_snap=True,
            tab_stops_from_regions=True,
            table_geometry=True,
            story_dedup=True,
            leader_fill_v1=True,
            page_rasters=page_rasters,
            image_region_snap=True,
            recovered_image_materialization=True,
        )
        chain_kwargs: dict[str, Any] = {
            "policies": MULTI_PAGE_CHAIN_POLICIES,
            "reconstruction_counts": reconstruction_counts,
        }
        if row.page_outcome in _PAGE_STREAM_OUTCOMES:
            return MultiPageDocxStageResult(
                produced=False, docx_bytes=None, reason_codes=("DOCX_PAGE_STREAM_INVALID",),
                docx_chain=_docx_chain_from_row(row, **chain_kwargs), pages=pages_tuple,
            )
        if row.page_outcome != "terminated":
            return MultiPageDocxStageResult(
                produced=False, docx_bytes=None, reason_codes=("DOCX_PAGE_NOT_TERMINATED",),
                docx_chain=_docx_chain_from_row(row, **chain_kwargs), pages=pages_tuple,
            )
        envelope = row.envelope
        if envelope is None or envelope.axes.ast_status != "strict_valid":
            return MultiPageDocxStageResult(
                produced=False, docx_bytes=None, reason_codes=("DOCX_AST_NOT_STRICT",),
                docx_chain=_docx_chain_from_row(row, **chain_kwargs), pages=pages_tuple,
            )
        substituted = _font_substitution_only(envelope.renderer)
        docx_path = envelope.renderer.docx_path
        if substituted:
            docx_path = work_dir / f"{job_id.replace('/', '__')}.docx"
            if not docx_path.is_file():
                docx_path = None
        elif envelope.renderer.status != "render_ready":
            docx_path = None
        if docx_path is None:
            return MultiPageDocxStageResult(
                produced=False, docx_bytes=None, reason_codes=("DOCX_RENDER_NOT_READY",),
                docx_chain=_docx_chain_from_row(row, **chain_kwargs), pages=pages_tuple,
            )
        try:
            work_root = work_dir.resolve(strict=True)
            resolved = docx_path.resolve(strict=True)
            resolved.relative_to(work_root)
            if not resolved.is_file():
                raise ValueError("diagnostic docx path is not a regular file under the work dir")
            size = resolved.stat().st_size
            if size > max_docx_bytes:
                return MultiPageDocxStageResult(
                    produced=False, docx_bytes=None, reason_codes=("DOCX_OUTPUT_TOO_LARGE",),
                    docx_chain=_docx_chain_from_row(row, **chain_kwargs), pages=pages_tuple,
                )
            docx_bytes = resolved.read_bytes()
        except (OSError, ValueError) as exc:
            return MultiPageDocxStageResult(
                produced=False, docx_bytes=None, reason_codes=("DOCX_CHAIN_ERROR",),
                docx_chain=_docx_chain_from_row(
                    row, error_class=type(exc).__name__, **chain_kwargs
                ),
                pages=pages_tuple,
            )
        chain = _docx_chain_from_row(row, **chain_kwargs)
        reasons: tuple[str, ...] = ("HEADLESS_DEFAULTS_V0_APPLIED",)
        if substituted:
            chain["renderer_approximations"] = sorted(FONT_SUBSTITUTION_CODES)
            reasons += ("DOCX_FONT_SUBSTITUTED",)
        return MultiPageDocxStageResult(
            produced=True, docx_bytes=docx_bytes, reason_codes=reasons, docx_chain=chain,
            pages=pages_tuple,
        )
    except Exception as exc:  # module boundary: sealed per-page IR remains usable
        return MultiPageDocxStageResult(
            produced=False,
            docx_bytes=None,
            reason_codes=("DOCX_CHAIN_ERROR",),
            docx_chain={
                "policy_id": hd.POLICY_ID,
                "policy_status": hd.POLICY_STATUS,
                "page_outcome": "error",
                "error_class": type(exc).__name__,
            },
            pages=pages_tuple,
        )

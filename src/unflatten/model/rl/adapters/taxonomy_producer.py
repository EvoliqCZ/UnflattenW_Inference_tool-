"""Train-cohort failure-taxonomy producer (S4.5, DEC-047 addendum A v3 item 8).

The ONE place under ``rl/`` that opens a real checkpoint for **inference only**:
``torch.no_grad``, no optimiser, no backward, no ``torch.optim`` (a test scans
this module's source). Coding is allowed; execution over a real checkpoint needs
an owner-approved window recorded for that exact ``checkpoint_id``.

Order inside :func:`produce_train_failure_report`, each step a named refusal:

1. the approved window names EXACTLY this checkpoint id (ancestry-derived
   ``ckpt_...``, never a file sha) and ``now`` is inside it -- before the
   output directory exists and before any weight is read;
2. the output directory is created and must be new (never overwritten);
3. the candidate is opened, then every page is decoded free-running under
   argmax, sealed per page index ``k`` (:mod:`taxonomy_assembly`), and one
   taxonomy report is built per ``k`` with ``inputs.decode_precision`` set from
   the window (fp8-vs-bf16 stays visible in the report);
4. :func:`~unflatten.model.rl.selection.build_selection` over the reports.

Only the output directory is written (atomic JSON). No S4 ledger, no ``docs/``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from unflatten.eval.failure_taxonomy import report as ft
from unflatten.eval.s2_gate import sealed
from unflatten.model import checkpoint_ancestry as ca
from unflatten.model.rl import selection as sel
from unflatten.model.rl.adapters import run_control
from unflatten.model.rl.adapters import taxonomy_assembly as ta

WINDOW_KEYS: Final = frozenset(
    {"checkpoint_id", "window_start", "window_end", "approved_by", "precision"}
)
PRECISIONS: Final = ("bf16", "fp8", "int8", "int4")
SELECTION_FILE: Final = "selection.json"
SUMMARY_FILE: Final = "summary.json"
PROGRESS_FILE: Final = "progress.json"
#: report ``inputs.visual_dtype`` of a decoder that does not name its dtype (a test seam)
UNSPECIFIED_VISUAL_DTYPE: Final = "unspecified"


class ProducerRefused(ValueError):
    """The approval window, the output directory or the inputs refuse this run."""


class _StopRequested(Exception):
    """Internal: the injected stop poll fired between two pages."""


@dataclass(frozen=True)
class ApprovedWindow:
    checkpoint_id: str
    window_start: datetime
    window_end: datetime
    approved_by: str
    precision: str


@dataclass(frozen=True)
class ProducerResult:
    status: str  # "complete" | "stopped"
    reports: tuple[dict[str, Any], ...]
    selection: sel.FrozenSelection | None
    summary: dict[str, Any]
    out_dir: Path


def _parse_time(value: Any, key: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ProducerRefused(f"window record {key} {value!r} is not ISO-8601") from exc
    if stamp.tzinfo is None:
        raise ProducerRefused(f"window record {key} {value!r} carries no time zone")
    return stamp


def load_window_record(path: Path | str) -> ApprovedWindow:
    """The owner's approved-window record (JSON, ``docs/runs/`` style), closed key set."""
    try:
        doc = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise ProducerRefused(f"window record {path} is unreadable: {exc}") from exc
    if not isinstance(doc, dict) or set(doc) != WINDOW_KEYS:
        got = sorted(doc) if isinstance(doc, dict) else type(doc).__name__
        raise ProducerRefused(
            f"window record keys must be exactly {sorted(WINDOW_KEYS)}, got {got}"
        )
    for key in ("checkpoint_id", "approved_by"):
        if not isinstance(doc[key], str) or not doc[key]:
            raise ProducerRefused(f"window record {key} must be a non-empty string")
    if doc["precision"] not in PRECISIONS:
        raise ProducerRefused(f"window record precision {doc['precision']!r} not in {PRECISIONS}")
    start = _parse_time(doc["window_start"], "window_start")
    end = _parse_time(doc["window_end"], "window_end")
    if end <= start:
        raise ProducerRefused("window record window_end is not after window_start")
    return ApprovedWindow(doc["checkpoint_id"], start, end, doc["approved_by"], doc["precision"])


def require_window(window: ApprovedWindow, *, checkpoint_id: str, now: datetime) -> None:
    """Refuse unless the window names ``checkpoint_id`` and ``now`` is in [start, end)."""
    if window.checkpoint_id != checkpoint_id:
        raise ProducerRefused(
            f"approved window names checkpoint {window.checkpoint_id!r}, the opened "
            f"checkpoint is {checkpoint_id!r}"
        )
    if not window.window_start <= now < window.window_end:
        raise ProducerRefused(
            f"now {now.isoformat()} is outside the approved window "
            f"[{window.window_start.isoformat()}, {window.window_end.isoformat()})"
        )


def resolve_checkpoint_id(
    checkpoint_path: Path | str,
    *,
    expect_base_ancestry_hash: str,
    expect_s1_run_manifest_hash: str | None,
) -> str:
    """The ancestry-derived checkpoint id from the verified sidecar (file bytes hashed,
    ``torch.load`` not called)."""
    path = Path(checkpoint_path)
    try:
        own = json.loads(ca.sidecar_path(path).read_text()).get("ancestry_hash", "")
    except (OSError, ValueError, AttributeError):
        own = ""  # verify_sidecar raises with the real reason
    doc = ca.verify_sidecar(
        path,
        expect_base_ancestry_hash=expect_base_ancestry_hash,
        expect_s1_run_manifest_hash=expect_s1_run_manifest_hash,
        embedded_ancestry_hash=own,
    )
    return doc["checkpoint"]["checkpoint_id"]


def _write_json(path: Path, body: Any) -> None:
    if path.exists():
        raise ProducerRefused(f"{path} exists: an output artefact is never overwritten")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(body, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _selection_body(selection: sel.FrozenSelection) -> dict[str, Any]:
    if selection.ready:
        return sel.selection_document(selection)
    return {**selection.payload(), "report_hash": selection.report_hash}


def _tracked_decoder(
    decode_fn: Callable[[Any], sealed.PageDecodeResult],
    *,
    stop_requested: Callable[[], bool],
    progress_path: Path,
    clock: Callable[[], datetime],
    outcomes: Counter[str],
) -> Callable[[Any], sealed.PageDecodeResult]:
    def tracked(page: Any) -> sealed.PageDecodeResult:
        if stop_requested():
            raise _StopRequested
        result = decode_fn(page)
        outcomes[result.outcome] += 1
        seen = sum(outcomes.values())
        run_control.write_progress(
            progress_path,
            {
                "step": seen,
                "update_step": 0,
                "pages_seen": seen,
                "ledger_records": 0,
                "rollouts": 0,
                "gate_failed_count": 0,
            },
            clock=lambda: clock().isoformat(),
        )
        return result

    return tracked


def produce_train_failure_report(
    *,
    window: ApprovedWindow,
    checkpoint_id: str,
    open_candidate: Callable[[], Any],
    make_decoder: Callable[[Any], Callable[[Any], sealed.PageDecodeResult]],
    pages: Iterable[Any],
    page_indices: Sequence[int],
    cache_identity: Mapping[str, Any],
    load_gold: Callable[[Collection[str], int], Mapping[str, Any]],
    bundle_lineage: Mapping[str, str],
    allowed_lineage_ids: Collection[str],
    budget: int,
    out_dir: Path | str,
    stop_requested: Callable[[], bool] = lambda: False,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    decode_inputs: Mapping[str, Any] | None = None,
) -> ProducerResult:
    """Decode ``pages`` (train only) with the opened candidate and write the
    per-``k`` taxonomy reports, the frozen selection and a summary into ``out_dir``."""
    require_window(window, checkpoint_id=checkpoint_id, now=clock())
    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ProducerRefused(f"output directory {out} exists: never overwritten") from exc
    page_list = list(pages)
    outcomes: Counter[str] = Counter()
    decoder = make_decoder(open_candidate())
    # the dtype the decoder was built with; a seam decoder that does not say is "unspecified"
    visual_dtype = getattr(decoder, "visual_dtype", None)
    if not isinstance(visual_dtype, str):
        visual_dtype = UNSPECIFIED_VISUAL_DTYPE
    tracked = _tracked_decoder(
        decoder,
        stop_requested=stop_requested,
        progress_path=out / PROGRESS_FILE,
        clock=clock,
        outcomes=outcomes,
    )
    base = {
        "checkpoint_id": checkpoint_id,
        "decode_precision": window.precision,
        "approved_by": window.approved_by,
        "role": ta.TRAIN_TAXONOMY_ROLE,
        "page_indices": sorted(set(page_indices)),
        "pages_requested": len(page_list),
    }
    try:
        artefacts = ta.assemble_train_sealed_artefacts(
            tracked,
            page_list,
            checkpoint_id=checkpoint_id,
            cache_identity=cache_identity,
            page_indices=page_indices,
        )
    except _StopRequested:
        summary = {
            **base,
            "status": "stopped",
            "pages_attempted": sum(outcomes.values()),
            "outcomes": dict(sorted(outcomes.items())),
        }
        _write_json(out / SUMMARY_FILE, summary)
        return ProducerResult("stopped", (), None, summary, out)
    reports: list[dict[str, Any]] = []
    missing_gold: dict[str, list[str]] = {}
    for k, artefact in zip(base["page_indices"], artefacts, strict=True):
        bundles = [p.document_bundle_id for p in artefact.pages]
        gold = load_gold(bundles, k)
        report = ft.build_report(
            artefact,
            gold,
            inputs={
                "page_index": k,
                "decode_precision": window.precision,
                "visual_dtype": visual_dtype,
                **dict(decode_inputs or {}),  # absent when no decode flag is on
            },
        )
        _write_json(out / f"report_k{k}.json", report)
        missing_gold[str(k)] = list(report["accounting"]["gold_missing_bundle_ids"])
        reports.append(report)
    selection = sel.build_selection(
        reports,
        policy_parent_checkpoint_id=checkpoint_id,
        allowed_lineage_ids=allowed_lineage_ids,
        bundle_lineage=bundle_lineage,
        budget=budget,
    )
    _write_json(out / SELECTION_FILE, _selection_body(selection))
    summary = {
        **base,
        "status": "complete",
        "pages_attempted": sum(outcomes.values()),
        "pages_decoded": outcomes.get("terminated", 0),
        "max_steps_hits": outcomes.get("max_steps_exceeded", 0),
        "outcomes": dict(sorted(outcomes.items())),
        "missing_gold_bundle_ids": missing_gold,
        "report_hashes": [r["report_hash"] for r in reports],
        "selection_status": selection.status,
        "selection_lineages": len(selection.entries),
    }
    _write_json(out / SUMMARY_FILE, summary)
    return ProducerResult("complete", tuple(reports), selection, summary, out)


@dataclass(frozen=True)
class KeyPage:
    """A ``(bundle, k)`` page. ``key`` serves the assembly, ``page_key`` the
    ``build_visual`` port (``RlPage`` shape)."""

    key: tuple[str, int]

    @property
    def page_key(self) -> tuple[str, int]:
        return self.key


def make_free_running_page_decoder(
    opened: Any,
    *,
    build_visual: Callable[[Any], Any],
    eval_cache_identity: Mapping[str, Any],
    component_lock_hash: str,
    selection_policy_hash: str,
    max_steps: int,
    device: Any,
    decode_fn: Callable[..., Any] | None = None,
    visual_dtype: Any = None,
    decode_overrides: Mapping[str, Any] | None = None,
) -> Callable[[Any], sealed.PageDecodeResult]:
    """The real per-page decode: free-running, constrained greedy (argmax), under
    ``torch.no_grad`` with the policy in ``eval`` mode -- built over ``opened``
    (an ``OpenedCandidate``) and sealed exactly as ``s2_gate.decode_cohort`` does
    (terminated -> ``seal_page``; every other outcome carries its evidence).
    ``decode_overrides`` are extra ``DecodeConfig`` fields (typed/region decode, loop
    guard); ``None`` is the bare config of old."""
    import torch

    from unflatten.eval.s2_gate import decode as dm
    from unflatten.model.free_running_decode import DecodeConfig, decode_free_running

    run = decode_free_running if decode_fn is None else decode_fn
    dtype = torch.bfloat16 if visual_dtype is None else visual_dtype
    opened.policy.eval()
    dtype_name = str(dtype).removeprefix("torch.")

    def decode(page: Any) -> sealed.PageDecodeResult:
        bundle = page.key[0]
        with torch.no_grad():
            visual = build_visual(page)
            levels = visual.deepstack_visual_embeds
            kwargs: dict[str, Any] = {
                "model": opened.policy,
                "controller": opened.controller,
                "vocabulary": opened.vocabulary,
                "visual_features": visual.visual_features.to(dtype).to(device),
                "image_grid_thw": visual.image_grid_thw.to(device),
                "image_token_id": opened.image_token_id,
                "config": DecodeConfig(
                    capability_profile_id=opened.capability_profile_id,
                    max_steps=max_steps,
                    **dict(decode_overrides or {}),
                ),
            }
            if levels is not None:
                kwargs["deepstack_visual_embeds"] = tuple(lv.to(dtype).to(device) for lv in levels)
            outcome, hypothesis, wiring = dm.decode_page(run, kwargs)
            common = {
                "decode_fn_identity": wiring.get("decode_fn"),
                "max_steps": wiring.get("max_steps"),
                "decode_fn_inner_identity": wiring.get("decode_fn_inner_identity"),
                "detail": dm._decode_detail(wiring),
            }
            if outcome == "terminated" and hypothesis is not None and hypothesis.node_states:
                return dm.seal_page(
                    document_bundle_id=bundle,
                    hypothesis=hypothesis,
                    eval_cache_identity=eval_cache_identity,
                    visual_features=kwargs["visual_features"],
                    component_lock_hash=component_lock_hash,
                    selection_policy_hash=selection_policy_hash,
                    **common,
                )
            parity = dm.compute_input_parity_proof_hash(
                document_bundle_id=bundle,
                eval_cache_identity=eval_cache_identity,
                visual_features=kwargs["visual_features"],
                component_lock_hash=component_lock_hash,
                selection_policy_hash=selection_policy_hash,
            )
            return sealed.PageDecodeResult(
                document_bundle_id=bundle,
                outcome="empty_hypothesis" if outcome == "terminated" else outcome,
                inference_mode=sealed.INFERENCE_MODE,
                input_parity_proof_hash=parity,
                scope=sealed._SCOPE_KIND,
                unterminated_prefix=wiring.get("unterminated_prefix"),
                **common,
            )

    setattr(decode, "visual_dtype", dtype_name)  # noqa: B010 - read by the producer's report
    return decode

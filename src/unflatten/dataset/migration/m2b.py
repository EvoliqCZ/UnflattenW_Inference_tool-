"""M2b — render enrichment of the M2a store (RFC003 §17 branch 2).

For every ``index.jsonl`` bundle not yet in ``renders.jsonl``: load the
canonical AST + resources from ``objects/`` (sha-verified reads via
``gates.store.read_object``), compile a strict RFC002 DOCX (+ package
manifest), render a PDF with headless LibreOffice, rasterize page PNGs
via pdftoppm, and derive the Observation Sidecar, Canonical Target
Bundle, ``single_page`` Page Fragments and the RFC003 §29 font
substitution mask projection — all stored content-addressed in the same
``objects/`` store.

``renders.jsonl`` is the append-only enrichment ledger (one line per
bundle, joined on ``document_bundle_id``); ``index.jsonl`` itself is
never rewritten. Compile/render failures for imported real documents are
signal, not crashes: they land as typed ``render_status="failed"``
records with issue codes and the run continues. Sidecar alignment
mismatches are recorded honestly (``matched < total``) and the render is
kept.

LibreOffice renders run across a process pool of ``workers`` (default 1
= sequential), each worker converting through its own LibreOffice
profile — a *shared* profile serializes on a lock and flakes. Pending
bundles are grouped into batches of ``--batch-size`` (default 50) so one
LibreOffice invocation converts many documents: LibreOffice startup, not
conversion, dominates the per-document cost (docs/RUNBOOK.md). A batch
that comes back short — a timeout, or fewer PDFs than documents
submitted — is transparently retried one-by-one, so one poison document
costs only its own failure, never the rest of its batch. The parent
process stays the only ledger appender. A single-run lockfile
(``renders.jsonl.lock`` with the holder's pid; stale locks from dead
pids are removed) guards against concurrent appenders.

Resume semantic (last-wins): ``renders.jsonl`` is append-only, so a
bundle id may appear on multiple lines — the LAST line per bundle id is
authoritative (``migration.m2c``'s join uses the same rule). On resume,
bundles whose last row is ``rendered`` or ``failed`` are skipped;
``skipped`` rows are re-attempted by default (a new line supersedes the
old one). ``render=None`` auto-detects the LibreOffice probe but ABORTS
when the probe fails — writing a whole store of ``skipped`` rows on a
probe flake requires the explicit ``--allow-skipped``. ``--fresh``
backs up and clears ``renders.jsonl`` before the run (for renderer
profile changes, e.g. a font-pack bump, that invalidate old rows).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.ast.canonicalizer import CanonicalizationError, canonicalize
from unflatten.dataset.fragments.envelope import build_fragments, validate_fragments
from unflatten.dataset.gates.store import StoreError, read_object
from unflatten.dataset.identity import domain_hash
from unflatten.dataset.observations.pdf_text import PdfTextExtractionError
from unflatten.dataset.observations.sidecar import (
    ObservationSidecarInvalid,
    build_observation_sidecar,
)
from unflatten.dataset.targets.bundle import build_target_bundle, validate_target_bundle
from unflatten.dataset.targets.masks import (
    build_mask_projection,
    validate_mask_projection,
)
from unflatten.libreoffice import profile_dir as libreoffice_profile_dir
from unflatten.renderer.api import RendererOptions
from unflatten.renderer.compiler import compile_docx
from unflatten.renderer.manifest import compute_font_pack_hash
from unflatten.renderer.probe import probe_available
from unflatten.renderer.resources import DirResourceProvider

#: 0.2.0: `--align-only` (design §2) — re-derives the Observation Sidecar
#: from a STORED pdf, never re-invoking LibreOffice.
MIGRATION_VERSION = "0.2.0"

_ALIGN_DOC_TIMEOUT_DEFAULT = 600

#: Modules whose code decides what the derived artifacts contain. Their
#: source digests form the render profile's ``derivation_hash``: without
#: it, a ledger row says WHICH font pack and DPI produced it but nothing
#: about the alignment or derivation code, so an aligner fix left old
#: rows looking final on a normal resume (the operator had to remember
#: ``--fresh``). Source bytes rather than a hand-maintained version
#: constant, because a version constant is exactly the claim nobody
#: verifies.
_DERIVATION_MODULES: tuple[str, ...] = (
    "unflatten.dataset.enums",
    "unflatten.dataset.fragments.envelope",
    "unflatten.dataset.fragments.identity",
    "unflatten.dataset.identity",
    "unflatten.dataset.observations.aligner",
    "unflatten.dataset.observations.cell_exact",
    "unflatten.dataset.observations.frames",
    "unflatten.dataset.observations.images",
    "unflatten.dataset.observations.pdf_text",
    "unflatten.dataset.observations.reading_order",
    "unflatten.dataset.observations.residual",
    "unflatten.dataset.observations.sidecar",
    "unflatten.dataset.observations.stories",
    "unflatten.dataset.observations.textnorm",
    "unflatten.dataset.targets.bundle",
    "unflatten.dataset.targets.events",
    "unflatten.dataset.targets.masks",
    "unflatten.dataset.targets.properties",
)

_DERIVATION_HASH_DOMAIN = "rfc003.m2b-derivation/v1"
FRAGMENT_WINDOW_POLICY = "single_page"
TARGET_LABEL_ORIGIN = "docx_importer_resolved"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

_LO_ROOT = Path.home() / "unflatten-probe"
_LO_TIMEOUT_SECONDS = 120
#: Default number of documents batched into one LibreOffice invocation
#: (`--batch-size`). Startup dominates a single conversion (~13.5 s of
#: ~14.5 s total, measured 2026-09-03 under load — see docs/RUNBOOK.md);
#: batching amortizes it across N documents instead of paying it per
#: document. 50 is an ESTIMATE, not a measurement: bigger amortizes
#: startup further but grows the blast radius of a single fallback replay
#: when the batch comes back short (see `_convert_batch_with_fallback`).
_DEFAULT_BATCH_SIZE = 50
#: Per-batch conversion budget: a flat startup allowance plus a generous
#: per-document allowance, so a batch that is merely slow (not stuck)
#: doesn't spuriously trip the fallback path and throw away the
#: amortization batching exists for. A batch that genuinely hangs still
#: times out and falls back to one-by-one conversion.
_LO_BATCH_STARTUP_BUDGET_SECONDS = 30
_LO_BATCH_PER_DOC_SECONDS = 15
#: Rasterization gets its own, larger budget: cost scales with page count
#: AND with the square of the DPI, so a long document at 300 DPI can take
#: minutes where the same document at 96 DPI took seconds.
_RASTER_TIMEOUT_SECONDS = 600
#: Default DPI of the pdftoppm page rasters; the sidecar's page pixel
#: dimensions are computed at the same DPI (matches the synthetic
#: factory). Overridable per run via ``--render-dpi``.
#:
#: 96 DPI puts an A4 page at 794x1123 px. A document VLM that merges a
#: 16 px patch grid 2x1 therefore sees ~8.75 mm of paper per visual
#: token — about four characters of 11 pt text, which no head can
#: separate afterwards. 200-300 DPI is what text tasks need; 96 is
#: enough for coarse structure signal (the Stage-0 probe).
#:
#: Changing it does NOT invalidate stored geometry: sidecar boxes are
#: `bbox_ppm` (millionths of a page), resolution-independent by design.
#: Only `width_px`/`height_px` and the rasters themselves change.
_RENDER_DPI = 96

_LO_VERSION: str | None = None

#: renders.jsonl statuses that are final on resume; a last row with any
#: other status (i.e. "skipped") is re-attempted by default.
_RESUME_FINAL_STATUSES: frozenset[str] = frozenset({"rendered", "failed"})


class M2bError(RuntimeError):
    """A run-level refusal (probe unavailable, lock held) — never a
    per-bundle failure, which lands as a typed renders.jsonl record."""


class AlignOnlyDriftError(RuntimeError):
    """Align-only's compare-before-store [M1] (design §2.1 step 4) caught
    real drift: recomputed target-bundle/mask-projection content differs
    from the previously released artifact. Never silently overwritten —
    the whole document becomes a typed align-only failure instead."""

    def __init__(self, code: str, expected: dict[str, Any], actual: dict[str, Any]) -> None:
        super().__init__(code)
        self.code = code
        self.expected = expected
        self.actual = actual


# ------------------------------------------------------------ store helpers --


def default_aligner_policy() -> str | None:
    """The aligner policy a render of THIS store revision uses: the v16
    rules (`aligner.V16_ALIGNER_POLICY`, docs/runs/target-defects-root-cause-
    2026-09-30.md) once `m2a.MIGRATION_VERSION` moves past the pre-I2
    revisions -- the same binding as `m2a.drop_empty_text_frames` (I2), so a
    v16 store cannot be rendered without it and today's store never with it.
    Recorded as `render_profile.aligner_policy` and as the sidecar's
    `aligner_policy` (schema 0.5.0)."""
    from unflatten.dataset.migration import m2a
    from unflatten.dataset.observations.aligner import V16_ALIGNER_POLICY

    return V16_ALIGNER_POLICY.policy_id if m2a.drop_empty_text_frames() else None


def _store(out_dir: Path, data: bytes) -> tuple[str, str]:
    """Atomic content-addressed write to ``objects/<sha256>`` (M2a pattern)."""
    hex_digest = hashlib.sha256(data).hexdigest()
    path = out_dir / "objects" / hex_digest
    tmp = path.with_name(f"{hex_digest}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hex_digest, f"objects/{hex_digest}"


def _artifact_ref(ref: str, hex_digest: str, size_bytes: int, mime_type: str) -> dict[str, Any]:
    return {
        "ref": ref,
        "sha256": f"sha256:{hex_digest}",
        "size_bytes": size_bytes,
        "mime_type": mime_type,
    }


def _store_ref(out_dir: Path, data: bytes, mime_type: str) -> dict[str, Any]:
    hex_digest, ref = _store(out_dir, data)
    return _artifact_ref(ref, hex_digest, len(data), mime_type)


def _would_be_ref(data: bytes, mime_type: str) -> dict[str, Any]:
    """The `_store_ref` a write WOULD produce, without writing — lets
    align-only's compare-before-store [M1] compare content before ever
    touching the store."""
    hex_digest = hashlib.sha256(data).hexdigest()
    return _artifact_ref(f"objects/{hex_digest}", hex_digest, len(data), mime_type)


# ------------------------------------------------- LibreOffice render path --
# Adapted from unflatten.dataset.synthetic.factory: staged under
# ~/unflatten-probe/ (the snap LibreOffice cannot read /tmp or
# dot-directories), one shared profile, strictly sequential calls.


def _staging_dir(tag: str) -> Path:
    return _LO_ROOT / "m2b" / f"{tag}-{os.getpid()}-{uuid.uuid4().hex}"


def _libreoffice_version() -> str:
    global _LO_VERSION
    if _LO_VERSION is None:
        try:
            proc = subprocess.run(
                ["libreoffice", "--version"], capture_output=True, text=True, timeout=60
            )
            first_line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
            _LO_VERSION = first_line if proc.returncode == 0 and first_line else "unknown"
        except (OSError, subprocess.TimeoutExpired):
            _LO_VERSION = "unknown"
    return _LO_VERSION


def _libreoffice_pdf(docx_bytes: bytes, tag: str, profile_slot: int = 0) -> bytes | None:
    """One headless DOCX→PDF conversion; ``None`` on failure.

    Each parallel worker owns a private ``-env:UserInstallation`` profile
    (``profile_slot``): LibreOffice serializes on a per-profile lock, so
    concurrent conversions are only safe when their profiles differ. Slot
    0 is also the sequential default.
    """
    profile_dir = libreoffice_profile_dir("m2b", slot=profile_slot)
    profile_dir.mkdir(parents=True, exist_ok=True)
    call_dir = _staging_dir(tag)
    work_dir = call_dir / "work"
    out_dir = call_dir / "out"
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        staged = work_dir / "doc.docx"
        staged.write_bytes(docx_bytes)
        cmd = [
            "libreoffice",
            "--headless",
            f"-env:UserInstallation=file://{profile_dir}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            str(staged),
        ]
        try:
            # `start_new_session` puts LibreOffice in its own process
            # group. The `libreoffice` launcher is only a shell wrapper:
            # killing it (what `subprocess.run(timeout=...)` does) leaves
            # the real `soffice.bin` grandchild alive, holding this
            # profile's lock FOREVER — every later conversion routed to
            # the same slot then fails. Observed in the wild: one hung
            # document poisoned a worker slot and 36 orphans piled up on
            # it. Killing the whole group is what actually stops it.
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError:
            return None
        try:
            proc.communicate(timeout=_LO_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            return None
        finally:
            _kill_stale_soffice(profile_dir)
        pdf_path = out_dir / "doc.pdf"
        if proc.returncode != 0 or not pdf_path.exists():
            return None
        return pdf_path.read_bytes()
    finally:
        shutil.rmtree(call_dir, ignore_errors=True)


def _batch_timeout_seconds(n: int) -> int:
    return _LO_BATCH_STARTUP_BUDGET_SECONDS + n * _LO_BATCH_PER_DOC_SECONDS


def _libreoffice_pdf_batch(
    docx_bytes_list: list[bytes], tag: str, profile_slot: int = 0
) -> list[bytes | None]:
    """Convert many DOCX to PDF through ONE headless LibreOffice invocation.

    Positional results: index ``i`` is ``None`` when input ``i`` produced
    no PDF — either that one document's conversion failed (LibreOffice
    keeps going past a single bad file) or the whole invocation
    timed out / errored, in which case every entry is ``None``.

    Same per-profile isolation as `_libreoffice_pdf` (``profile_slot``)
    and the same process-group kill on timeout, for the same reason: the
    ``libreoffice`` launcher is a shell wrapper, so a plain timeout-kill
    leaves ``soffice.bin`` alive holding the profile lock.

    Deliberately does not itself retry a partial batch — that is
    `_convert_batch_with_fallback`'s job, so a batch that comes back
    short is retried one-by-one rather than accepted partially.
    """
    n = len(docx_bytes_list)
    if n == 0:
        return []
    profile_dir = libreoffice_profile_dir("m2b", slot=profile_slot)
    profile_dir.mkdir(parents=True, exist_ok=True)
    call_dir = _staging_dir(tag)
    work_dir = call_dir / "work"
    out_dir = call_dir / "out"
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        names = [f"doc-{i:04d}" for i in range(n)]
        staged_paths = []
        for name, data in zip(names, docx_bytes_list):
            path = work_dir / f"{name}.docx"
            path.write_bytes(data)
            staged_paths.append(path)
        cmd = [
            "libreoffice",
            "--headless",
            f"-env:UserInstallation=file://{profile_dir}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            *(str(p) for p in staged_paths),
        ]
        try:
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError:
            return [None] * n
        try:
            proc.communicate(timeout=_batch_timeout_seconds(n))
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            return [None] * n
        finally:
            _kill_stale_soffice(profile_dir)
        if proc.returncode != 0:
            return [None] * n
        return [
            (out_dir / f"{name}.pdf").read_bytes()
            if (out_dir / f"{name}.pdf").exists()
            else None
            for name in names
        ]
    finally:
        shutil.rmtree(call_dir, ignore_errors=True)


def _convert_batch_with_fallback(
    docx_bytes_list: list[bytes], doc_tags: list[str], profile_slot: int = 0
) -> list[bytes | None]:
    """Batch-convert, falling back to one-by-one conversion on any
    batch-level failure.

    Failure isolation is the whole point of batching rather than a
    long-lived UNO daemon (see docs/RUNBOOK.md): a batch that times out
    or comes back with fewer PDFs than documents submitted must not cost
    the other N-1 documents their conversion, so on ANY missing entry the
    entire batch is reconverted through the existing single-document path
    `_libreoffice_pdf` — one poison document then costs only its own
    failure, at the price of reconverting the batch's healthy documents
    once more (a rare event, not the common path).
    """
    n = len(docx_bytes_list)
    if n == 0:
        return []
    if n == 1:
        return [_libreoffice_pdf(docx_bytes_list[0], doc_tags[0], profile_slot)]
    batch_tag = f"batch-{uuid.uuid4().hex[:8]}"
    results = _libreoffice_pdf_batch(docx_bytes_list, batch_tag, profile_slot)
    if len(results) == n and all(pdf is not None for pdf in results):
        return results
    return [
        _libreoffice_pdf(data, doc_tags[i], profile_slot)
        for i, data in enumerate(docx_bytes_list)
    ]


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    """SIGKILL the converter's whole process group, then reap it."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.communicate(timeout=10)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass


def _kill_stale_soffice(profile_dir: Path) -> None:
    """Kill any `soffice` still holding `profile_dir` after a conversion.

    Conversions on one profile are strictly sequential (a slot belongs to
    exactly one worker), so once a call returns nothing may legitimately
    still hold its profile. LibreOffice nonetheless leaves the process
    running in some failure modes; left alone it locks the slot out for
    the rest of the run.
    """
    marker = str(profile_dir)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if b"soffice" not in cmdline or marker.encode() not in cmdline:
            continue
        try:
            os.kill(int(entry.name), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _pdf_pages_png(
    pdf_bytes: bytes, tag: str, render_dpi: int = _RENDER_DPI
) -> list[bytes]:
    """Rasterize per-page PNGs via pdftoppm (empty list when unavailable)."""
    if shutil.which("pdftoppm") is None:
        return []
    call_dir = _staging_dir(f"{tag}-pages")
    try:
        call_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = call_dir / "doc.pdf"
        pdf_path.write_bytes(pdf_bytes)
        try:
            proc = subprocess.run(
                ["pdftoppm", "-png", "-r", str(render_dpi),
                 str(pdf_path), str(call_dir / "page")],
                capture_output=True,
                timeout=_RASTER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            # A pathological page must cost that document its rasters, not
            # the run: the caller records RENDER_PAGES_MISSING.
            return []
        if proc.returncode != 0:
            return []
        return [p.read_bytes() for p in sorted(call_dir.glob("page-*.png"))]
    finally:
        shutil.rmtree(call_dir, ignore_errors=True)


# ------------------------------------------------------ derived enrichments --


def _build_observation(
    canonical: dict[str, Any],
    pdf_bytes: bytes,
    options: RendererOptions,
    store_dir: Path,
    artifacts: dict[str, Any],
    render_dpi: int = _RENDER_DPI,
    aligner_policy: str | None = None,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """Build + store the Observation Sidecar bound to the stored PDF.

    Returns ``(observation_record, skip_reason, sidecar)`` — exactly one
    of record/skip_reason is non-None. Alignment mismatches are typed
    issue codes next to honest coverage counts, never a failure."""
    renderer_meta = {
        "name": "libreoffice",
        "version": _libreoffice_version(),
        "dpi": render_dpi,
        "font_pack_hash": compute_font_pack_hash(options.font_policy),
    }
    try:
        build = build_observation_sidecar(
            canonical, pdf_bytes, renderer_meta,
            **({"aligner_policy": aligner_policy} if aligner_policy is not None else {}),
        )
    except PdfTextExtractionError as exc:
        return None, f"pdf_text_extraction_failed: {exc}", None
    except ObservationSidecarInvalid as exc:
        # sidecar.py is in _DERIVATION_MODULES, this module is not: the
        # validity check and the issue-code filter both live there, so
        # neither can drift out of step with derivation_hash (design §0a).
        codes = ",".join(i.code for i in exc.issues)
        return None, f"sidecar_invalid:{codes}", None
    ref = _store_ref(store_dir, to_canonical_json(build.sidecar), "application/json")
    artifacts["observation_sidecar"] = ref
    return {
        "sidecar": ref,
        "coverage": build.coverage(),
        "reading_order_switched_pages": list(build.reading_order_switched_pages),
        "issue_codes": build.issue_codes(),
        "issue_details": build.issue_details(),
    }, None, build.sidecar


def _build_targets(
    canonical: dict[str, Any],
    sidecar: dict[str, Any],
    store_dir: Path,
    artifacts: dict[str, Any],
    *,
    document_bundle_id: str | None = None,
    ast_file_hash: str | None = None,
    expected: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Derive + store the target bundle and single_page fragments.

    ``document_bundle_id`` / ``ast_file_hash`` close the RFC003 §26/§27
    identity: the bundle and every fragment bind the DocumentBundle and
    the exact stored AST file they were derived from.

    Unlike the synthetic factory (where a validation failure is a factory
    bug), imported real documents may produce non-derivable targets —
    that is a typed outcome ``(None, codes)``, not a crash.

    ``expected`` (align-only's compare-before-store [M1]): when given, the
    recomputed target bundle is compared to it by sha256 BEFORE any write.
    Unchanged content reuses ``expected`` as-is — no write, no new object.
    Different content raises :class:`AlignOnlyDriftError` and stores
    nothing, including fragments: a drifted bundle is a released-artifact
    hazard, not a partial update. Fragments still count toward
    ``document_bundle_id``/``ast_file_hash`` identity every time they ARE
    written; unlike the bundle they are always freshly stored, because a
    realignment is exactly what changes their content (design §2.1 step 4)."""
    try:
        bundle = build_target_bundle(
            canonical,
            label_origin=TARGET_LABEL_ORIGIN,
            document_bundle_id=document_bundle_id,
            ast_file_hash=ast_file_hash,
        )
        fragments, fragment_index = build_fragments(
            canonical,
            bundle,
            sidecar,
            FRAGMENT_WINDOW_POLICY,
            document_bundle_id=document_bundle_id,
        )
    except ValueError:
        return None, ["FRAGMENT_DERIVATION_FAILED"]
    issues = validate_target_bundle(bundle) + validate_fragments(
        fragments, fragment_index, bundle, sidecar
    )
    if issues:
        return None, sorted({i.code for i in issues})

    bundle_data = to_canonical_json(bundle)
    if expected is not None:
        actual_ref = _would_be_ref(bundle_data, "application/json")
        if actual_ref["sha256"] != expected["sha256"]:
            raise AlignOnlyDriftError("ALIGN_ONLY_TARGET_BUNDLE_DRIFT", expected, actual_ref)
        bundle_artifact = expected
    else:
        bundle_artifact = _store_ref(store_dir, bundle_data, "application/json")
    artifacts["target_bundle"] = bundle_artifact
    fragment_refs = [
        _store_ref(store_dir, to_canonical_json(fragment), "application/json")
        for fragment in fragments
    ]
    stored_index = {
        **fragment_index,
        "entries": [
            {**entry, "fragment": ref}
            for entry, ref in zip(fragment_index["entries"], fragment_refs)
        ],
    }
    index_artifact = _store_ref(store_dir, to_canonical_json(stored_index), "application/json")
    artifacts["fragment_index"] = index_artifact
    return {
        "target_bundle": bundle_artifact,
        "fragment_index": index_artifact,
        "fragment_count": fragment_index["fragment_count"],
        "window_policy": FRAGMENT_WINDOW_POLICY,
    }, []


def _build_mask_projection(
    canonical: dict[str, Any],
    options: RendererOptions,
    store_dir: Path,
    artifacts: dict[str, Any],
    targets: dict[str, Any],
    *,
    expected: dict[str, Any] | None = None,
) -> list[str]:
    """Derive + store the RFC003 §29 font-substitution mask projection.

    Built from the SAME canonicalized AST and the SAME `font_policy` the
    compile used — the renderer substitutes faces outside the pilot pack,
    so the page shows a family the AST does not name and
    ``/style/font_family`` must not be supervised for those runs.

    Pure function of AST + policy: no render input, hence no dependency on
    the render succeeding. A derivation failure is a typed code on the
    kept record, never a crash.

    ``expected``: align-only's compare-before-store [M1], same contract as
    :func:`_build_targets` — unchanged content reuses ``expected`` without
    a write; drifted content raises :class:`AlignOnlyDriftError`."""
    try:
        projection = build_mask_projection(
            canonical, options.font_policy, strict=options.strict
        )
        issues = validate_mask_projection(projection)
    except (KeyError, TypeError, ValueError):
        return ["MASK_PROJECTION_FAILED"]
    if issues:
        return sorted({i.code for i in issues})
    data = to_canonical_json(projection)
    if expected is not None:
        actual_ref = _would_be_ref(data, "application/json")
        if actual_ref["sha256"] != expected["sha256"]:
            raise AlignOnlyDriftError("ALIGN_ONLY_MASK_DRIFT", expected, actual_ref)
        ref = expected
    else:
        ref = _store_ref(store_dir, data, "application/json")
    artifacts["mask_projection"] = ref
    targets["mask_projection"] = ref
    targets["masked_font_runs"] = projection["counts"]["affected_runs"]
    return []


# ----------------------------------------------------------- per-bundle run --


def _failed(record: dict[str, Any], stage: str, codes: list[str]) -> dict[str, Any]:
    record["render_status"] = "failed"
    record["stage"] = stage
    record["codes"] = sorted(set(codes))
    return record


def _compile_one(
    store_dir: Path,
    index_record: dict[str, Any],
    stage_root: Path,
    render_profile: dict[str, Any],
) -> tuple[dict[str, Any], bytes | None, dict[str, Any] | None, RendererOptions | None]:
    """Steps 1-2 of enrichment: load, canonicalize, strict RFC002 compile.

    Returns ``(record, docx_bytes, canonical, options)``. On any failure
    ``record`` is already a typed ``failed`` row and ``docx_bytes`` is
    ``None`` — the caller must not attempt to render it. Split out of the
    former ``_enrich_one`` so a batch can compile every document before
    the (single, shared) LibreOffice conversion call.
    """
    envelope = index_record["envelope"]
    bundle_id = envelope["document_bundle_id"]
    record: dict[str, Any] = {
        "m2b_version": MIGRATION_VERSION,
        "document_bundle_id": bundle_id,
        "render_status": "skipped",
        "stage": None,
        "codes": [],
        "artifacts": {},
        "observation": None,
        "observation_skip_reason": None,
        "targets": None,
        # renderer profile note: rows rendered under a different
        # font_pack_hash are not comparable (see --fresh)
        "render_profile": dict(render_profile),
    }

    # 1. sha-verified loads from the content-addressed store; a malformed
    # manifest/entry (missing key, non-JSON object) is a typed load
    # failure, never a crash — the run continues
    try:
        stored_ast = json.loads(read_object(store_dir, envelope["ast"]))
        manifest = json.loads(read_object(store_dir, envelope["resources_manifest"]))
        resources: dict[str, bytes] = {}
        for entry in manifest:
            # M2a manifest rows carry no mime_type; ArtifactRef needs one
            ref = {
                "ref": entry["ref"],
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
                "mime_type": entry.get("mime_type", "application/octet-stream"),
            }
            resources[entry["path"]] = read_object(store_dir, ref)
    except StoreError as exc:
        return _failed(record, "load", [exc.code]), None, None, None
    except (KeyError, TypeError, json.JSONDecodeError):
        return _failed(record, "load", ["LOAD_RECORD_MALFORMED"]), None, None, None

    # M2a-imported ASTs keep importer node ids (``i*``); targets and
    # sidecar derivation require preorder ``n*`` ids, so re-canonicalize.
    try:
        canonical = canonicalize(stored_ast)
    except CanonicalizationError as exc:
        return _failed(record, "canonicalize", [i.code for i in exc.issues]), None, None, None

    # 2. strict RFC002 compile (pilot font pack with fallbacks — the same
    # RendererOptions the synthetic factory uses)
    stage = stage_root / bundle_id
    stage.mkdir(parents=True, exist_ok=True)
    try:
        for path, data in resources.items():
            target = stage / "resources" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        docx_path = stage / "doc.docx"
        options = RendererOptions(strict=True, emit_manifest=True)
        result = compile_docx(
            canonical,
            docx_path,
            options,
            resources=DirResourceProvider(stage / "resources") if resources else None,
        )
        if result.status == "failed":
            return (
                _failed(record, "compile", [i.code for i in result.issues]),
                None,
                None,
                None,
            )
        docx_bytes = docx_path.read_bytes()
        package_manifest_bytes = docx_path.with_suffix(".docx.manifest.json").read_bytes()
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    record["artifacts"]["docx"] = _store_ref(store_dir, docx_bytes, DOCX_MIME)
    record["artifacts"]["package_manifest"] = _store_ref(
        store_dir, package_manifest_bytes, "application/json"
    )
    return record, docx_bytes, canonical, options


def _finish_one(
    record: dict[str, Any],
    canonical: dict[str, Any],
    options: RendererOptions,
    tag: str,
    pdf_bytes: bytes | None,
    store_dir: Path,
    render_dpi: int,
    bundle_id: str,
    ast_file_hash: str,
    aligner_policy: str | None = None,
) -> dict[str, Any]:
    """Steps 3-4 of enrichment: page rasters + sidecar + targets/masks,
    given the PDF bytes a (possibly batched) LibreOffice conversion
    already produced for this document (``None`` on render failure)."""
    if pdf_bytes is None:
        return _failed(record, "render", ["RENDER_PDF_FAILED"])
    record["render_status"] = "rendered"
    record["artifacts"]["pdf"] = _store_ref(store_dir, pdf_bytes, "application/pdf")
    record["artifacts"]["pages"] = [
        _store_ref(store_dir, page_bytes, "image/png")
        for page_bytes in _pdf_pages_png(pdf_bytes, tag=tag, render_dpi=render_dpi)
    ]
    if not record["artifacts"]["pages"]:
        # pdftoppm failed/produced no pages — the render is kept, but the
        # gap is a typed warning code, never silent
        record["codes"] = sorted(set(record["codes"]) | {"RENDER_PAGES_MISSING"})

    observation, skip_reason, sidecar = _build_observation(
        canonical, pdf_bytes, options, store_dir, record["artifacts"], render_dpi,
        **({"aligner_policy": aligner_policy} if aligner_policy is not None else {}),
    )
    record["observation"] = observation
    record["observation_skip_reason"] = skip_reason
    if sidecar is not None:
        targets, target_codes = _build_targets(
            canonical,
            sidecar,
            store_dir,
            record["artifacts"],
            document_bundle_id=bundle_id,
            ast_file_hash=ast_file_hash,
        )
        record["targets"] = targets
        record["codes"] = sorted(set(record["codes"]) | set(target_codes))
        if targets is not None:
            record["codes"] = sorted(
                set(record["codes"])
                | set(
                    _build_mask_projection(
                        canonical, options, store_dir, record["artifacts"], targets
                    )
                )
            )
    return record


def _blank_failed_record(
    bundle_id: str, render_profile: dict[str, Any], stage: str, exc: Exception
) -> dict[str, Any]:
    """A typed ``failed`` row for an UNEXPECTED exception (Defect A): every
    other per-document failure in this pipeline already lands as one of
    these without aborting the run, so an exception escaping ``_compile_one``
    / ``_finish_one`` must be no different — a single bad document must
    never kill an 18,000-document, multi-hour job. The exception type and
    message are recorded in full (``error_detail``) so the failure is loud
    in the ledger, never a silent gap; ``codes`` stays a single low-
    cardinality bucket so ``failure_codes`` aggregation is still meaningful.
    """
    record = _failed(
        {
            "m2b_version": MIGRATION_VERSION,
            "document_bundle_id": bundle_id,
            "artifacts": {},
            "observation": None,
            "observation_skip_reason": None,
            "targets": None,
            "render_profile": dict(render_profile),
        },
        stage,
        [f"{stage.upper()}_UNEXPECTED_ERROR"],
    )
    record["error_detail"] = {"exception_type": type(exc).__name__, "message": str(exc)}
    return record


def _safe_compile_one(
    store_dir: Path,
    index_record: dict[str, Any],
    stage_root: Path,
    render_profile: dict[str, Any],
) -> tuple[dict[str, Any], bytes | None, dict[str, Any] | None, RendererOptions | None]:
    """``_compile_one`` isolated against an unexpected exception. Never
    swallows ``KeyboardInterrupt``/``SystemExit`` — the run stays
    stoppable."""
    bundle_id = index_record.get("envelope", {}).get("document_bundle_id", "unknown")
    try:
        return _compile_one(store_dir, index_record, stage_root, render_profile)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # isolation boundary — see _blank_failed_record
        return _blank_failed_record(bundle_id, render_profile, "compile", exc), None, None, None


def _safe_finish_one(
    record: dict[str, Any],
    canonical: dict[str, Any],
    options: RendererOptions,
    tag: str,
    pdf_bytes: bytes | None,
    store_dir: Path,
    render_dpi: int,
    bundle_id: str,
    ast_file_hash: str,
    render_profile: dict[str, Any],
) -> dict[str, Any]:
    """``_finish_one`` isolated against an unexpected exception — same
    principle as ``_safe_compile_one``, one level later in the pipeline."""
    try:
        return _finish_one(
            record, canonical, options, tag, pdf_bytes, store_dir, render_dpi,
            bundle_id, ast_file_hash, render_profile.get("aligner_policy"),
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # isolation boundary — see _blank_failed_record
        return _blank_failed_record(bundle_id, render_profile, "finish", exc)


def _enrich_batch(
    store_dir: Path,
    index_records: list[dict[str, Any]],
    stage_root: Path,
    do_render: bool,
    render_profile: dict[str, Any],
    profile_slot: int = 0,
) -> list[dict[str, Any]]:
    """Enrich a batch of bundles: compile every DOCX first, convert them
    all through ONE LibreOffice invocation (with automatic per-document
    fallback on any batch failure), then finish each document
    individually — exactly the four steps ``_enrich_one`` used to run per
    document, with step 3 now shared across the batch.

    Compile and finish are each run through an isolation wrapper
    (``_safe_compile_one``/``_safe_finish_one``): an unexpected exception
    on one document must not kill the batch (Defect A) — LibreOffice
    conversion itself is already isolated by
    ``_convert_batch_with_fallback``; this is the same principle for our
    own Python.
    """
    compiled = [
        _safe_compile_one(store_dir, rec, stage_root, render_profile) for rec in index_records
    ]
    if not do_render:
        return [c[0] for c in compiled]

    render_dpi = int(render_profile.get("render_dpi", _RENDER_DPI))
    convert_idx = [i for i, c in enumerate(compiled) if c[1] is not None]
    docx_list = [compiled[i][1] for i in convert_idx]
    doc_tags = [compiled[i][0]["artifacts"]["docx"]["sha256"][7:23] for i in convert_idx]
    pdf_results = _convert_batch_with_fallback(docx_list, doc_tags, profile_slot)
    pdf_by_idx = dict(zip(convert_idx, pdf_results))
    tag_by_idx = dict(zip(convert_idx, doc_tags))

    records: list[dict[str, Any]] = []
    for i, (record, docx_bytes, canonical, options) in enumerate(compiled):
        if docx_bytes is None:
            records.append(record)
            continue
        envelope = index_records[i]["envelope"]
        records.append(
            _safe_finish_one(
                record,
                canonical,
                options,
                tag_by_idx[i],
                pdf_by_idx.get(i),
                store_dir,
                render_dpi,
                envelope["document_bundle_id"],
                envelope["ast"]["sha256"],
                render_profile,
            )
        )
    return records


def _module_digest(module: Any) -> str:
    """sha256 of a module's source file."""
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def derivation_hash() -> str:
    """Identity of the code that derives sidecar, targets and fragments.

    Deliberately computed from the modules as they are AT RUN TIME, so a
    change anywhere in :data:`_DERIVATION_MODULES` — an aligner fix, a
    fragment schema field — produces a different hash without anyone
    having to remember to bump anything.
    """
    payload = {
        name: _module_digest(importlib.import_module(name))
        for name in _DERIVATION_MODULES
    }
    return domain_hash(_DERIVATION_HASH_DOMAIN, payload)


# ------------------------------------------------------- align-only (§2) ----
# Re-derives the Observation Sidecar (+ fragments) for an already-rendered
# bundle from its STORED pdf — never re-invoking LibreOffice. Exists
# because an aligner fix invalidates every stored sidecar without
# invalidating a single render, and re-rendering 18,803 documents through
# LibreOffice to fix a Python bug in the aligner is not that fix's cost.


def _align_base_record(
    bundle_id: str, source_row: dict[str, Any], render_profile: dict[str, Any]
) -> dict[str, Any]:
    """The record shape shared by every align-only outcome (success AND
    [A-MUST-A] failure): `docx`/`package_manifest`/`pdf`/`pages` carried
    over from the source row as provenance, `target_bundle`/
    `mask_projection` carried over as the compare-before-store baseline
    (overwritten with themselves on success — see `_build_targets` /
    `_build_mask_projection` — never rewritten on failure)."""
    source_artifacts = source_row.get("artifacts") or {}
    return {
        "m2b_version": MIGRATION_VERSION,
        "document_bundle_id": bundle_id,
        "render_status": "rendered",
        "stage": None,
        # design §2.1 step 5: RENDER_PAGES_MISSING is about the render,
        # not the alignment — it survives a realignment; nothing else does
        "codes": sorted(set(source_row.get("codes") or []) & {"RENDER_PAGES_MISSING"}),
        "artifacts": {
            key: source_artifacts[key]
            for key in (
                "docx", "package_manifest", "pdf", "pages",
                "target_bundle", "mask_projection",
            )
            if key in source_artifacts
        },
        "observation": None,
        "observation_skip_reason": None,
        "targets": None,
        "render_profile": dict(render_profile),
    }


def _align_failure(
    bundle_id: str,
    source_row: dict[str, Any],
    render_profile: dict[str, Any],
    code: str,
    error_detail: dict[str, Any] | None,
    renderer_meta: dict[str, Any] | None,
    source_sidecar_ref: dict[str, Any] | None,
) -> dict[str, Any]:
    """[A-MUST-A] failure shape (design §2.1): `render_status` stays
    "rendered" (truthful — the render is fine, only realignment failed) so
    the gate engine quarantines it via `OBSERVATION_MISSING`, never
    `RENDER_SKIPPED`; `observation_skip_reason` carries the code so the
    reason is visible without decoding `align_only.failure`; the document
    is repeatable (`align_only.failure != null` triggers re-attempt by
    default, design §2.1 "Pravidlo re-attemptu")."""
    record = _align_base_record(bundle_id, source_row, render_profile)
    record["observation_skip_reason"] = f"align_only:{code}"
    prior_align = source_row.get("align_only") or {}
    record["align_only"] = {
        "source_derivation_hash": (
            prior_align.get("source_derivation_hash")
            if prior_align.get("failure") is not None
            else (source_row.get("render_profile") or {}).get("derivation_hash")
        ),
        "source_sidecar": (
            source_sidecar_ref
            if source_sidecar_ref is not None
            else (
                prior_align.get("source_sidecar")
                or (source_row.get("observation") or {}).get("sidecar")
            )
        ),
        "renderer": renderer_meta if renderer_meta is not None else prior_align.get("renderer"),
        "failure": {"code": code, "error_detail": error_detail},
    }
    return record


def _realign_one(
    store_dir: Path,
    index_record: dict[str, Any],
    source_row: dict[str, Any],
    render_profile: dict[str, Any],
) -> tuple[dict[str, Any], bool, bool]:
    """Re-derive one bundle's Observation Sidecar (+ fragments) from its
    STORED pdf. Target bundle and mask projection are compare-before-store
    [M1]: unchanged content reuses the source ref without a write; drifted
    content is a typed failure, never a silent overwrite of a released
    artifact (`AlignOnlyDriftError`, caught here).

    Returns ``(record, target_bundle_unchanged, mask_projection_unchanged)``.
    """
    envelope = index_record["envelope"]
    bundle_id = envelope["document_bundle_id"]
    source_artifacts = source_row.get("artifacts") or {}
    prior_align = source_row.get("align_only") or {}
    is_retry = prior_align.get("failure") is not None

    def fail(
        code: str,
        error_detail: dict[str, Any] | None = None,
        renderer_meta: dict[str, Any] | None = None,
        source_sidecar_ref: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool, bool]:
        return (
            _align_failure(
                bundle_id, source_row, render_profile, code, error_detail,
                renderer_meta, source_sidecar_ref,
            ),
            False,
            False,
        )

    # renderer_meta / source_sidecar_ref: from the prior align-only failure
    # on a retry (design §2.1 "Pravidlo re-attemptu" — a retry never has an
    # `observation` to read a sidecar from), otherwise from the stored
    # sidecar the source render produced (sidecar.py `renderer` key).
    if is_retry:
        renderer_meta = prior_align.get("renderer")
        source_sidecar_ref = prior_align.get("source_sidecar")
        source_derivation_hash = prior_align.get("source_derivation_hash")
        # A timeout row can carry no renderer meta because the alarm fired
        # before `_realign_one` returned one. Its `source_sidecar` still
        # names the stored sidecar, so recover the renderer meta from there.
        if not renderer_meta and source_sidecar_ref is not None:
            try:
                stored_sidecar = json.loads(read_object(store_dir, source_sidecar_ref))
            except StoreError as exc:
                return fail(f"LOAD_{exc.code}", {"message": str(exc)})
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                return fail("LOAD_RECORD_MALFORMED", {"message": str(exc)})
            renderer_meta = stored_sidecar.get("renderer")
    else:
        source_sidecar_ref = (source_row.get("observation") or {}).get("sidecar")
        try:
            stored_sidecar = json.loads(read_object(store_dir, source_sidecar_ref))
        except StoreError as exc:
            return fail(f"LOAD_{exc.code}", {"message": str(exc)})
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            return fail("LOAD_RECORD_MALFORMED", {"message": str(exc)})
        renderer_meta = stored_sidecar.get("renderer")
        source_derivation_hash = (source_row.get("render_profile") or {}).get("derivation_hash")

    if not renderer_meta:
        return fail("LOAD_RENDERER_META_MISSING", None, None, source_sidecar_ref)

    render_dpi = render_profile.get("render_dpi", _RENDER_DPI)
    if renderer_meta.get("dpi") != render_dpi:
        return fail(
            "ALIGN_ONLY_PROFILE_MISMATCH",
            {"renderer_dpi": renderer_meta.get("dpi"), "render_dpi": render_dpi},
            renderer_meta,
            source_sidecar_ref,
        )

    # AST + PDF: sha-verified loads, never a fresh LibreOffice conversion
    try:
        stored_ast = json.loads(read_object(store_dir, envelope["ast"]))
        pdf_bytes = read_object(store_dir, source_artifacts["pdf"])
    except StoreError as exc:
        return fail(f"LOAD_{exc.code}", {"message": str(exc)}, renderer_meta, source_sidecar_ref)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        return fail(
            "LOAD_RECORD_MALFORMED", {"message": str(exc)}, renderer_meta, source_sidecar_ref
        )

    try:
        canonical = canonicalize(stored_ast)
    except CanonicalizationError as exc:
        return fail(
            "LOAD_CANONICALIZATION_FAILED",
            {"codes": [i.code for i in exc.issues]},
            renderer_meta,
            source_sidecar_ref,
        )

    options = RendererOptions(strict=True, emit_manifest=True)
    aligner_policy = render_profile.get("aligner_policy")
    try:
        build = build_observation_sidecar(
            canonical, pdf_bytes, renderer_meta,
            **({"aligner_policy": aligner_policy} if aligner_policy is not None else {}),
        )
    except PdfTextExtractionError as exc:
        return fail(
            "ALIGN_ONLY_PDF_TEXT_EXTRACTION_FAILED",
            {"message": str(exc)},
            renderer_meta,
            source_sidecar_ref,
        )
    except ObservationSidecarInvalid as exc:
        codes = ",".join(i.code for i in exc.issues) or "UNKNOWN"
        return fail(f"OBSERVATION_{codes}", None, renderer_meta, source_sidecar_ref)

    # sidecar + fragments: always freshly stored — a realignment is
    # exactly what changes their content (design §2.1 step 4)
    record = _align_base_record(bundle_id, source_row, render_profile)
    sidecar_ref = _store_ref(store_dir, to_canonical_json(build.sidecar), "application/json")
    record["artifacts"]["observation_sidecar"] = sidecar_ref
    record["observation"] = {
        "sidecar": sidecar_ref,
        "coverage": build.coverage(),
        "reading_order_switched_pages": list(build.reading_order_switched_pages),
        "issue_codes": build.issue_codes(),
        "issue_details": build.issue_details(),
    }

    ast_file_hash = envelope["ast"]["sha256"]
    expected_bundle = source_artifacts.get("target_bundle")
    expected_mask = source_artifacts.get("mask_projection")
    try:
        targets, target_codes = _build_targets(
            canonical,
            build.sidecar,
            store_dir,
            record["artifacts"],
            document_bundle_id=bundle_id,
            ast_file_hash=ast_file_hash,
            expected=expected_bundle,
        )
    except AlignOnlyDriftError as exc:
        return fail(
            exc.code, {"expected": exc.expected, "actual": exc.actual},
            renderer_meta, source_sidecar_ref,
        )
    record["codes"] = sorted(set(record["codes"]) | set(target_codes))
    record["targets"] = targets
    target_unchanged = expected_bundle is not None and targets is not None
    mask_unchanged = False
    if targets is not None:
        try:
            mask_codes = _build_mask_projection(
                canonical, options, store_dir, record["artifacts"], targets,
                expected=expected_mask,
            )
        except AlignOnlyDriftError as exc:
            return fail(
                exc.code, {"expected": exc.expected, "actual": exc.actual},
                renderer_meta, source_sidecar_ref,
            )
        record["codes"] = sorted(set(record["codes"]) | set(mask_codes))
        mask_unchanged = expected_mask is not None

    record["align_only"] = {
        "source_derivation_hash": source_derivation_hash,
        "source_sidecar": source_sidecar_ref,
        "renderer": renderer_meta,
        "failure": None,
    }
    return record, target_unchanged, mask_unchanged


class _AlignDocTimeout(Exception):
    """One document exceeded its `--doc-timeout` wall-clock budget."""


def _align_alarm(_signum: int, _frame: Any) -> None:
    raise _AlignDocTimeout("align-only document budget exceeded")


def _safe_realign_one(
    store_dir: Path,
    index_record: dict[str, Any],
    source_row: dict[str, Any],
    render_profile: dict[str, Any],
    doc_timeout: int,
) -> tuple[dict[str, Any], bool, bool]:
    """`_realign_one` isolated against an unexpected exception (never
    swallows `KeyboardInterrupt`/`SystemExit`) AND against a document that
    never returns, via SIGALRM (design §2.1 step 6, pattern from
    `scripts/partial_ready_promotion.py:_recompute`). One poison document
    must cost only itself, never the run — the same principle `enrich_renders`
    already applies through `_safe_compile_one`/`_safe_finish_one`."""
    bundle_id = index_record.get("envelope", {}).get("document_bundle_id", "unknown")
    if doc_timeout:
        signal.signal(signal.SIGALRM, _align_alarm)
        signal.alarm(doc_timeout)
    try:
        return _realign_one(store_dir, index_record, source_row, render_profile)
    except _AlignDocTimeout:
        return (
            _align_failure(
                bundle_id, source_row, render_profile, "ALIGN_TIMEOUT",
                {"doc_timeout": doc_timeout}, None, None,
            ),
            False,
            False,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # isolation boundary — see _align_failure
        return (
            _align_failure(
                bundle_id, source_row, render_profile, "ALIGN_UNEXPECTED_ERROR",
                {"exception_type": type(exc).__name__, "message": str(exc)}, None, None,
            ),
            False,
            False,
        )
    finally:
        if doc_timeout:
            signal.alarm(0)


#: Set once per `realign_renders` call, alongside `_ALIGN_STORE_DIR` /
#: `_ALIGN_RENDER_PROFILE` / `_ALIGN_DOC_TIMEOUT`: a fork-context pool
#: worker takes only an INDEX (`_align_worker_index`) — these globals are
#: what the fork already inherited, never re-pickled per task (pattern
#: from scripts/census/multipage_admission_repeat.py:_worker_measure_index).
_ALIGN_JOBS: list[tuple[dict[str, Any], dict[str, Any]]] = []
_ALIGN_STORE_DIR: Path | None = None
_ALIGN_RENDER_PROFILE: dict[str, Any] = {}
_ALIGN_DOC_TIMEOUT: int = _ALIGN_DOC_TIMEOUT_DEFAULT


def _align_worker_init() -> None:
    """PDEATHSIG: a killed parent must not leave orphan workers burning
    CPU forever (same pattern as scripts/aligner_store_gate.py)."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGKILL)
    except OSError:
        pass


def _align_worker_index(index: int) -> tuple[dict[str, Any], bool, bool]:
    """Process-pool entry point: takes only `index` into the module-global
    `_ALIGN_JOBS` the fork already inherited."""
    index_record, source_row = _ALIGN_JOBS[index]
    assert _ALIGN_STORE_DIR is not None
    return _safe_realign_one(
        _ALIGN_STORE_DIR, index_record, source_row, _ALIGN_RENDER_PROFILE, _ALIGN_DOC_TIMEOUT
    )


def _iter_realigned(
    indices: list[int], worker_count: int
) -> Iterator[tuple[dict[str, Any], bool, bool]]:
    """Same streaming/ordering contract as `_iter_enriched`: results arrive
    in submission order so the ledger stays deterministic, and an early
    exit tears the pool down with `cancel_futures`."""
    if worker_count <= 1:
        for index in indices:
            yield _align_worker_index(index)
        return
    ctx = mp.get_context("fork")
    pool = ProcessPoolExecutor(
        max_workers=worker_count, mp_context=ctx, initializer=_align_worker_init
    )
    try:
        yield from pool.map(_align_worker_index, indices, chunksize=8)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _select_align_only_candidates(
    joined: list[tuple[dict[str, Any], dict[str, Any] | None]],
    current_derivation: str,
    only_bundles: set[str] | None,
    aligner_policy: str | None = None,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], int, int, int]:
    """design §2.1/§2.3 selection: a `rendered` row with a stored
    observation and a stale `derivation_hash`, OR a failed align-only row
    (re-attempted by default). Everything else is left as-is, named.

    Returns ``(candidates, skipped_already_done, not_rendered_left_as_is,
    no_stored_sidecar_left_as_is)``.
    """
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    skipped_already_done = not_rendered_left_as_is = no_stored_sidecar_left_as_is = 0
    for index_record, source_row in joined:
        bundle_id = index_record["envelope"]["document_bundle_id"]
        if only_bundles is not None and bundle_id not in only_bundles:
            continue
        if source_row is None or source_row.get("render_status") != "rendered":
            not_rendered_left_as_is += 1
            continue
        align_only = source_row.get("align_only") or {}
        failed_before = align_only.get("failure") is not None
        if failed_before:
            candidates.append((index_record, source_row))
            continue
        if source_row.get("observation") is None:
            no_stored_sidecar_left_as_is += 1
            continue
        source_profile = source_row.get("render_profile") or {}
        if (
            source_profile.get("derivation_hash") == current_derivation
            and source_profile.get("aligner_policy") == aligner_policy
        ):
            skipped_already_done += 1
            continue
        candidates.append((index_record, source_row))
    return candidates, skipped_already_done, not_rendered_left_as_is, no_stored_sidecar_left_as_is


def _infer_align_only_render_dpi(
    candidates: list[tuple[dict[str, Any], dict[str, Any]]],
) -> int:
    """Infer the run's ``render_dpi`` from the candidate source rows.

    align-only never re-renders — it re-derives from STORED pdfs — so the
    run's ``render_dpi`` must be the DPI those renders actually used, not
    the fresh-render default: a default that disagrees with the stored
    sidecars makes `_realign_one`'s ``ALIGN_ONLY_PROFILE_MISMATCH`` guard
    reject every document (the T17 shape — a store rendered at 300 DPI
    re-aligned under the 96 default failed 100 % of its canary).

    - non-retry source row: ``render_profile.render_dpi``
    - retry row (``align_only.failure is not None``): the failed run may
      have written its own, wrong profile before the guard rejected it,
      so ``align_only.renderer.dpi`` is authoritative when present; a
      timeout row can carry no renderer meta and falls back to the
      retry row's own ``render_profile.render_dpi``
    - no candidates: ``_RENDER_DPI`` (nothing to disagree with)
    - more than one distinct DPI, or a missing/invalid DPI on any
      candidate: fail closed with :class:`M2bError` naming what was seen
      — never guess which DPI to write
    """
    if not candidates:
        return _RENDER_DPI
    seen: set[int] = set()
    invalid: list[str] = []
    for index_record, source_row in candidates:
        bundle_id = index_record["envelope"]["document_bundle_id"]
        align_only = source_row.get("align_only") or {}
        profile_dpi = (source_row.get("render_profile") or {}).get("render_dpi")
        if align_only.get("failure") is not None:
            dpi = (align_only.get("renderer") or {}).get("dpi")
            if not (isinstance(dpi, int) and not isinstance(dpi, bool) and dpi > 0):
                dpi = profile_dpi
        else:
            dpi = profile_dpi
        if isinstance(dpi, int) and not isinstance(dpi, bool) and dpi > 0:
            seen.add(dpi)
        else:
            invalid.append(f"{bundle_id}: {dpi!r}")
    if invalid:
        raise M2bError(
            "m2b --align-only: candidates with a missing/invalid render DPI "
            f"({'; '.join(invalid)}) — refusing to guess the run's "
            f"render_dpi (valid DPIs seen: {sorted(seen) or 'none'})"
        )
    if len(seen) > 1:
        raise M2bError(
            "m2b --align-only: candidates imply more than one render DPI "
            f"({', '.join(str(d) for d in sorted(seen))}) — refusing to "
            "guess which stored render the sidecars were derived from; "
            "re-run over a homogeneous candidate set (--only-bundles)"
        )
    return seen.pop()


def _append_align_summary(store_dir: Path, run_entry: dict[str, Any]) -> None:
    """Append one run to `<store>/align-only-summary.json` (design §2.6,
    §3.2 "nie do run-summary.json — m2a ho prepisuje"). Atomic full
    rewrite, like `_write_page_outcomes_jsonl` elsewhere: the file is
    small (one dict per run), so a full rewrite is cheap and never leaves
    a torn tail the way append-in-place could."""
    path = store_dir / "align-only-summary.json"
    if path.is_file():
        data = json.loads(path.read_text())
    else:
        data = {"runs": []}
    data["runs"].append(run_entry)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    os.replace(tmp, path)


def realign_renders(
    store_dir: Path,
    *,
    limit: int | None = None,
    workers: int | None = None,
    progress_every: int = 100,
    only_bundles: set[str] | None = None,
    doc_timeout: int = _ALIGN_DOC_TIMEOUT_DEFAULT,
    source_store: str | None = None,
    aligner_policy: str | None = None,
) -> dict[str, Any]:
    """`--align-only`: re-derive the Observation Sidecar (+ fragments) for
    every `rendered` bundle whose `derivation_hash` is stale, from the
    STORED pdf — never re-invoking LibreOffice (design §2). A failed
    align-only row (`align_only.failure != null`) is re-attempted by
    default, the same way `enrich_renders` re-attempts `skipped` rows.

    Target bundle and mask projection are compare-before-store [M1]:
    identical content reuses the source ref (no write); drifted content is
    a typed, repeatable failure — never a silent overwrite of a released
    artifact.

    The run's ``render_dpi`` is NOT the fresh-render default: it is
    inferred from the candidate source rows (their
    ``render_profile.render_dpi``, or ``align_only.renderer.dpi`` for a
    failed retry row whose own profile may be stale) and must be
    single-valued — candidates implying more than one DPI, or no valid
    DPI, abort fail-closed (M2bError) before the lock is taken or a row
    is written.
    """
    # imported here, not at module load: m2c does not import m2b, but a
    # module-level import would still make m2b's import graph depend on
    # the whole gate/split/release stack it otherwise has no reason to
    # touch (design §2.1: "m2c neimportuje m2b" — the acyclic side of that
    # statement is this import direction, not its absence).
    from unflatten.dataset.migration.m2c import join_store_records

    store_dir = Path(store_dir)
    renders_path = store_dir / "renders.jsonl"
    lock_path = store_dir / "renders.jsonl.lock"
    # [M3]: before `_acquire_lock` and before repairing the torn tail — an
    # in-place write onto a still-hardlinked ledger corrupts the SOURCE
    # store too, so this must happen before ANY write, including the lock.
    _refuse_if_hardlinked(renders_path)
    if lock_path.exists():
        _refuse_if_hardlinked(lock_path)

    worker_count = max(1, workers or 1)

    # Selection and the run's `render_dpi` inference happen BEFORE the
    # lock: a candidate set that implies more than one DPI — or none at
    # all — must fail closed without even creating the lockfile, let
    # alone appending a row. Reading the store unlocked is safe: the lock
    # guards the APPEND, a concurrent live appender is still refused at
    # `_acquire_lock` below, and a torn trailing line is tolerated by
    # `join_store_records` exactly as it is after the post-lock
    # `_repair_torn_tail`.
    current_derivation = derivation_hash()
    # The target ledger is stale by definition after an aligner change and is
    # re-derived by m2b_targets from the sidecars written here; the drift check
    # in `_build_targets` still compares every bundle to it.
    joined = join_store_records(store_dir, refuse_stale_target_ledger=False)
    (
        candidates,
        skipped_already_done,
        not_rendered_left_as_is,
        no_stored_sidecar_left_as_is,
    ) = _select_align_only_candidates(joined, current_derivation, only_bundles, aligner_policy)
    if only_bundles is not None:
        found = {ir["envelope"]["document_bundle_id"] for ir, _sr in candidates}
        found |= {
            ir["envelope"]["document_bundle_id"]
            for ir, sr in joined
            if sr is not None and ir["envelope"]["document_bundle_id"] in only_bundles
        }
        missing = only_bundles - found
        if missing:
            raise M2bError(
                f"m2b --align-only: {len(missing)} requested bundle id(s) "
                f"have no renders.jsonl row (first: {sorted(missing)[0]})"
            )
    if limit is not None:
        candidates = candidates[:limit]

    # align-only never re-renders: the run's `render_dpi` must be the DPI
    # the STORED renders used, not the fresh-render default — a default
    # that disagrees with the stored sidecars is exactly what
    # `_realign_one`'s `ALIGN_ONLY_PROFILE_MISMATCH` guard rejects for
    # every document (the T17 shape).
    render_profile = {
        "font_pack_hash": compute_font_pack_hash(
            RendererOptions(strict=True, emit_manifest=True).font_policy
        ),
        "render_dpi": _infer_align_only_render_dpi(candidates),
        "derivation_hash": current_derivation,
        # design §2.2: distinguishes align-only rows from a fresh
        # LibreOffice render
        "alignment_source": "stored_pdf",
    }
    if aligner_policy is not None:
        render_profile["aligner_policy"] = aligner_policy

    _acquire_lock(lock_path)
    try:
        _repair_torn_tail(renders_path)

        global _ALIGN_JOBS, _ALIGN_STORE_DIR, _ALIGN_RENDER_PROFILE, _ALIGN_DOC_TIMEOUT
        _ALIGN_JOBS = candidates
        _ALIGN_STORE_DIR = store_dir
        _ALIGN_RENDER_PROFILE = render_profile
        _ALIGN_DOC_TIMEOUT = doc_timeout

        realigned = 0
        failures: dict[str, int] = {}
        failure_bundles: list[str] = []
        target_bundle_unchanged = mask_projection_unchanged = 0
        retried = retried_succeeded = 0
        source_derivation_hashes_seen: set[str] = set()
        coverage_ratios: list[float] = []
        with_sidecar = full_coverage = zero_text_node_docs = 0
        started_at = datetime.now(UTC)
        started = time.monotonic()
        try:
            with open(renders_path, "a") as out_f:
                for position, (record, target_unchanged, mask_unchanged) in enumerate(
                    _iter_realigned(list(range(len(candidates))), worker_count), start=1
                ):
                    _, source_row = candidates[position - 1]
                    was_retry = (source_row.get("align_only") or {}).get("failure") is not None
                    if was_retry:
                        retried += 1
                    source_derivation = (source_row.get("render_profile") or {}).get(
                        "derivation_hash"
                    )
                    if source_derivation:
                        source_derivation_hashes_seen.add(source_derivation)
                    align_only = record.get("align_only") or {}
                    if align_only.get("failure") is not None:
                        code = align_only["failure"]["code"]
                        failures[code] = failures.get(code, 0) + 1
                        failure_bundles.append(record["document_bundle_id"])
                    else:
                        realigned += 1
                        if was_retry:
                            retried_succeeded += 1
                        if target_unchanged:
                            target_bundle_unchanged += 1
                        if mask_unchanged:
                            mask_projection_unchanged += 1
                        observation = record.get("observation")
                        if observation is not None:
                            with_sidecar += 1
                            cov = observation["coverage"]
                            total = cov["total_text_nodes"]
                            if total == 0:
                                zero_text_node_docs += 1
                            else:
                                coverage_ratios.append(cov["matched_text_nodes"] / total)
                                if cov["matched_text_nodes"] == total:
                                    full_coverage += 1
                    out_f.write(json.dumps(record, sort_keys=True) + "\n")
                    out_f.flush()
                    if progress_every and position % progress_every == 0:
                        elapsed = time.monotonic() - started
                        print(
                            f"[m2b --align-only] {position}/{len(candidates)} "
                            f"realigned={realigned} failed={sum(failures.values())} "
                            f"retried={retried} ({elapsed / position:.1f}s/doc)",
                            flush=True,
                        )
        finally:
            _ALIGN_JOBS = []
    finally:
        lock_path.unlink(missing_ok=True)

    elapsed = time.monotonic() - started
    processed = len(candidates)
    summary = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "derivation_hash": current_derivation,
        "source_derivation_hashes_seen": sorted(source_derivation_hashes_seen),
        "m2b_version": MIGRATION_VERSION,
        "source_store": source_store,
        "processed": processed,
        "realigned": realigned,
        "skipped_already_done": skipped_already_done,
        "not_rendered_left_as_is": not_rendered_left_as_is,
        "no_stored_sidecar_left_as_is": no_stored_sidecar_left_as_is,
        "failures": dict(sorted(failures.items(), key=lambda kv: -kv[1])),
        "failure_bundles": failure_bundles,
        "target_bundle_unchanged": target_bundle_unchanged,
        "mask_projection_unchanged": mask_projection_unchanged,
        "coverage": {
            "with_sidecar": with_sidecar,
            "full_coverage": full_coverage,
            "zero_text_node_docs": zero_text_node_docs,
            "mean_ratio": (
                sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else None
            ),
            "min_ratio": min(coverage_ratios) if coverage_ratios else None,
        },
        "seconds_per_doc": round(elapsed / processed, 2) if processed else None,
        "doc_timeout": doc_timeout,
        "retried": retried,
        "retried_succeeded": retried_succeeded,
    }
    _append_align_summary(store_dir, summary)
    return summary


# ------------------------------------------------------------------ driver --


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover — pid exists, other user
        return True
    return True


def _repair_torn_tail(renders_path: Path) -> None:
    """A killed run can leave a torn trailing line in the ledger; appending
    onto it would create mid-file corruption that readers cannot skip."""
    if renders_path.exists() and renders_path.stat().st_size:
        with open(renders_path, "rb") as tail_f:
            tail_f.seek(-1, os.SEEK_END)
            if tail_f.read(1) != b"\n":
                with open(renders_path, "a") as fix_f:
                    fix_f.write("\n")
                print(
                    "[m2b] repaired torn trailing line in renders.jsonl",
                    flush=True,
                )


def _refuse_if_hardlinked(path: Path) -> None:
    """[M3] Refuse a `renders.jsonl` (or its `.lock`) that still shares an
    inode with another store's file — the shape a `cp -al` clone (design
    §3.1) leaves before the hardlink is broken. Writing through it (an
    in-place ``write_text``/append) would corrupt the SOURCE store's
    ledger too, since both names point at the same bytes on disk."""
    if path.exists() and os.stat(path).st_nlink > 1:
        raise M2bError(
            f"m2b --align-only: {path} has st_nlink > 1 (likely a `cp -al` "
            "clone still sharing the source store's inode, design §3.1) — "
            "writing here would silently corrupt the source ledger too. "
            "Break the hardlink first (cp --remove-destination the file "
            "onto itself, or rm the stray copy)."
        )


def _acquire_lock(lock_path: Path) -> None:
    """Single-run lockfile with pid; a dead holder's lock is stale."""
    if lock_path.exists():
        try:
            holder = int(lock_path.read_text().strip())
        except ValueError:
            holder = None
        if holder is not None and _pid_alive(holder):
            raise M2bError(
                f"m2b: {lock_path} is held by live pid {holder} — another "
                "run is appending to renders.jsonl; refusing to start"
            )
        print(f"[m2b] removing stale lock {lock_path} (pid {holder} is dead)", flush=True)
        lock_path.unlink()
    lock_path.write_text(str(os.getpid()))


def _worker_task(
    args: tuple[Path, list[dict[str, Any]], Path, bool, dict[str, Any], int],
) -> list[dict[str, Any]]:
    """Process-pool entry point: one BATCH of bundles, one private LO
    profile slot, one shared LibreOffice conversion call.

    Safe to run concurrently: the object store is content-addressed with
    atomic ``os.replace`` writes, staging dirs carry pid+uuid, and each
    worker converts through its own LibreOffice profile. The ledger is
    written only by the parent, so there is a single appender.
    """
    store_dir, index_records, stage_root, do_render, render_profile, _slot = args
    # The LibreOffice profile belongs to the WORKER PROCESS, never to the
    # task: `pool.map` hands the next queued task to whichever worker is
    # free, so a slot derived from the task index can be picked up by a
    # second worker while the first is still converting — two concurrent
    # conversions on one profile then fight over its lock (observed: five
    # soffice processes on profile-2, and a 7-9 % failure rate on the
    # slower RTL corpora, where overruns make the overlap likely). Keying
    # on the pid gives every worker a private, stable profile; workers are
    # long-lived, so the profile is created once and reused.
    return _enrich_batch(
        store_dir, index_records, stage_root, do_render, render_profile, os.getpid()
    )


def _iter_enriched(
    tasks: list[tuple[Path, list[dict[str, Any]], Path, bool, dict[str, Any], int]],
    worker_count: int,
) -> Iterator[dict[str, Any]]:
    """Stream enriched records (flattened, one per document), sequentially
    or across a process pool of batch tasks.

    Results arrive in submission order so the ledger stays deterministic.
    On an early exit (circuit breaker) the pool is torn down with
    ``cancel_futures`` so queued work does not keep the run alive.
    """
    if worker_count <= 1:
        for task in tasks:
            yield from _worker_task(task)
        return
    pool = ProcessPoolExecutor(max_workers=worker_count)
    try:
        for batch_result in pool.map(_worker_task, tasks, chunksize=1):
            yield from batch_result
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _load_resume_state(
    renders_path: Path,
    *,
    retry_failed: bool = False,
    derivation: str | None = None,
    aligner_policy: str | None = None,
) -> tuple[set[str], int, set[str], int]:
    """Last-wins per bundle id (append-only ledger — duplicates expected
    after a skipped-row re-attempt).

    Returns ``(final bundle ids, duplicate count, font pack hashes seen,
    stale-derivation count)``. With ``retry_failed`` the ``failed`` status
    stops being final, so a transient outage (e.g. a mid-run LibreOffice
    crash that failed a long tail of bundles) is recoverable without
    discarding the whole ledger.

    A row whose ``render_profile.derivation_hash`` is not ``derivation``
    was produced by different derivation code and is NOT final: it drops
    out of the resume set and is recomputed, so correctness does not
    depend on the operator remembering ``--fresh``. The same holds for a
    row whose ``render_profile.aligner_policy`` is not ``aligner_policy``.
    """
    last_status: dict[str, Any] = {}
    last_profile: dict[str, Any] = {}
    last_derivation: dict[str, Any] = {}
    last_policy: dict[str, Any] = {}
    duplicates = 0
    if renders_path.exists():
        for line in renders_path.read_text().splitlines():
            try:
                row = json.loads(line)
                bundle_id = row["document_bundle_id"]
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if bundle_id in last_status:
                duplicates += 1
            last_status[bundle_id] = row.get("render_status")
            profile = row.get("render_profile") or {}
            if not isinstance(profile, dict):
                profile = {}
            last_derivation[bundle_id] = profile.get("derivation_hash")
            last_policy[bundle_id] = profile.get("aligner_policy")
            if profile.get("font_pack_hash"):
                # the DPI is part of the profile: rasters at a different
                # resolution are not comparable training inputs
                last_profile[bundle_id] = (
                    profile["font_pack_hash"],
                    profile.get("render_dpi", _RENDER_DPI),
                )
    if duplicates:
        print(
            f"[m2b] {renders_path}: {duplicates} duplicate bundle row(s) — "
            "last row wins (resume semantic)",
            flush=True,
        )
    final_statuses = (
        frozenset({"rendered"}) if retry_failed else _RESUME_FINAL_STATUSES
    )
    done = {b for b, s in last_status.items() if s in final_statuses}
    # the font-pack/DPI comparison keeps seeing every final row: a mixed
    # ledger stays an explicit abort, not a silent recompute
    profiles = {last_profile[b] for b in done if b in last_profile}
    stale = {
        b
        for b in done
        if last_derivation.get(b) != derivation or last_policy.get(b) != aligner_policy
    }
    if stale:
        print(
            f"[m2b] {len(stale)} row(s) derived by different derivation code "
            "— recomputing (stale derivation_hash)",
            flush=True,
        )
    return done - stale, duplicates, profiles, len(stale)


def enrich_renders(
    store_dir: Path,
    *,
    limit: int | None = None,
    workers: int | None = None,
    render: bool | None = None,
    progress_every: int = 100,
    allow_skipped: bool = False,
    fresh: bool = False,
    retry_failed: bool = False,
    allow_mixed_profile: bool = False,
    max_consecutive_render_failures: int = 25,
    only_bundles: set[str] | None = None,
    render_dpi: int = _RENDER_DPI,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    aligner_policy: str | None = None,
) -> dict[str, Any]:
    """Enrich un-rendered M2a bundles; returns a run summary dict.

    ``workers`` sets the render parallelism (default 1 = sequential).
    Each worker converts through its own LibreOffice ``UserInstallation``
    profile, which is what makes concurrency safe — a *shared* profile
    serializes on a lock and produces flaky failures. The parent stays
    the sole ledger appender. Historical note: earlier revisions kept
    renders strictly sequential
    (shared profile lock). ``render=None`` auto-detects the probe but
    ABORTS (``M2bError``) when the probe fails, unless
    ``allow_skipped=True`` — one probe flake must never mass-write
    ``skipped`` rows. ``render=False`` explicitly compiles DOCX only
    (records land ``skipped``). ``fresh=True`` backs up ``renders.jsonl``
    to ``renders.jsonl.bak.<unix-ts>`` and clears it before the run.

    ``batch_size`` groups up to that many pending bundles into ONE
    LibreOffice invocation (docs/RUNBOOK.md: startup dominates a single
    conversion, ~13.5 s of ~14.5 s measured 2026-09-03). Each worker still
    owns a private profile; batches, not documents, now round-robin
    across profile slots. A batch that comes back short (timeout, or
    fewer PDFs than documents submitted) is transparently retried
    one-by-one so a single poison document never costs the rest of the
    batch (`_convert_batch_with_fallback`).

    ``only_bundles`` re-enriches exactly those bundle ids regardless of
    resume state — for rebuilding rows whose stored targets went stale
    (e.g. a canonicalizer fix changed the canonical ``ast_hash``).
    """
    store_dir = Path(store_dir)
    index_path = store_dir / "index.jsonl"
    renders_path = store_dir / "renders.jsonl"
    if render is None:
        do_render = probe_available()
        if not do_render and not allow_skipped:
            raise M2bError(
                "m2b: LibreOffice render probe failed with render auto-detect "
                "on — refusing to write 'skipped' rows for the whole store. "
                "Fix the render toolchain, or pass --allow-skipped "
                "(allow_skipped=True) to compile DOCX only."
            )
    else:
        do_render = render

    worker_count = max(1, workers or 1)
    lock_path = store_dir / "renders.jsonl.lock"
    _acquire_lock(lock_path)
    try:
        backup: str | None = None
        if fresh and renders_path.exists():
            backup_path = renders_path.with_name(
                f"renders.jsonl.bak.{int(time.time())}"
            )
            shutil.copy2(renders_path, backup_path)
            renders_path.unlink()
            backup = str(backup_path)
            print(f"[m2b] --fresh: renders.jsonl backed up to {backup_path}", flush=True)

        done, duplicate_rows, ledger_pack_hashes, stale_derivation_rows = (
            _load_resume_state(
                renders_path,
                retry_failed=retry_failed,
                derivation=derivation_hash(),
                aligner_policy=aligner_policy,
            )
        )

        pending: list[dict[str, Any]] = []
        skipped_already_done = 0
        with open(index_path) as f:
            for line in f:
                rec = json.loads(line)
                bundle_id = rec["envelope"]["document_bundle_id"]
                if only_bundles is not None:
                    # targeted re-enrichment: the ledger's last-wins resume
                    # semantic makes the appended rows supersede the stale
                    # ones, so `done` must not filter these out.
                    if bundle_id in only_bundles:
                        pending.append(rec)
                    continue
                if bundle_id in done:
                    skipped_already_done += 1
                    continue
                pending.append(rec)
        if only_bundles is not None:
            missing = only_bundles - {
                r["envelope"]["document_bundle_id"] for r in pending
            }
            if missing:
                raise M2bError(
                    f"m2b: {len(missing)} requested bundle id(s) are not in "
                    f"{index_path} (first: {sorted(missing)[0]})"
                )
        if limit is not None:
            pending = pending[:limit]

        render_profile = {
            "font_pack_hash": compute_font_pack_hash(
                RendererOptions(strict=True, emit_manifest=True).font_policy
            ),
            # part of the profile identity: rasters at a different DPI are
            # not comparable training inputs, the same way a different font
            # pack is not a comparable render
            "render_dpi": render_dpi,
            # identity of the derivation code (aligner, sidecar builder,
            # target/fragment derivation): rows produced by other code are
            # not comparable and are recomputed on resume
            "derivation_hash": derivation_hash(),
            # design §2.2: distinguishes a normal render from an
            # `--align-only` re-derivation off a STORED pdf
            "alignment_source": "fresh_render",
        }
        if aligner_policy is not None:
            render_profile["aligner_policy"] = aligner_policy
        current_pack_hash = render_profile.get("font_pack_hash")
        current = (current_pack_hash, render_dpi)
        stale_packs = {p for p in ledger_pack_hashes if p != current}
        if stale_packs and not allow_mixed_profile:
            stale = sorted(stale_packs)[0]
            raise M2bError(
                "renders.jsonl holds rows rendered under a different render "
                f"profile (font pack {stale[0]} at {stale[1]} DPI vs current "
                f"{current_pack_hash} at {render_dpi} DPI); re-run with "
                "--fresh (recommended) or --allow-mixed-profile."
            )
        if stale_packs:
            print(
                f"[m2b] WARNING: mixed render-profile ledger ({len(stale_packs)} "
                "stale profile(s)) — records are not profile-comparable",
                flush=True,
            )
        status_counts = {"rendered": 0, "failed": 0, "skipped": 0}
        failure_codes: dict[str, int] = {}
        warning_codes: dict[str, int] = {}
        coverage_ratios: list[float] = []
        with_sidecar = full_coverage = zero_text_node_docs = 0
        rendered_without_targets = 0
        started = time.monotonic()
        consecutive_render_failures = 0
        stage_root = store_dir / "_m2b_stage"
        try:
            _repair_torn_tail(renders_path)
            with open(renders_path, "a") as out_f:
                batch_len = max(1, batch_size)
                batches = [
                    pending[i : i + batch_len] for i in range(0, len(pending), batch_len)
                ]
                tasks = [
                    (
                        store_dir,
                        batch,
                        stage_root,
                        do_render,
                        render_profile,
                        offset % worker_count,
                    )
                    for offset, batch in enumerate(batches)
                ]
                for position, record in enumerate(
                    _iter_enriched(tasks, worker_count), start=1
                ):
                    status_counts[record["render_status"]] += 1
                    if record["render_status"] == "failed":
                        # Circuit breaker: a mid-run LibreOffice outage would
                        # otherwise write final `failed` rows for the whole
                        # remaining corpus. A long unbroken run of *render*
                        # failures means the environment died, not the data.
                        if "RENDER_PDF_FAILED" in record["codes"]:
                            consecutive_render_failures += 1
                        else:
                            consecutive_render_failures = 0
                        if (
                            max_consecutive_render_failures
                            and consecutive_render_failures
                            >= max_consecutive_render_failures
                        ):
                            out_f.write(json.dumps(record, sort_keys=True) + "\n")
                            out_f.flush()
                            raise M2bError(
                                f"{consecutive_render_failures} consecutive render "
                                "failures — aborting (LibreOffice outage?). Fix the "
                                "environment and resume with --retry-failed."
                            )
                        for code in record["codes"]:
                            failure_codes[code] = failure_codes.get(code, 0) + 1
                    else:
                        consecutive_render_failures = 0
                        # warnings (target derivation codes, pages missing)
                        # never mix into failure_codes
                        for code in record["codes"]:
                            warning_codes[code] = warning_codes.get(code, 0) + 1
                    if record["render_status"] == "rendered" and record["targets"] is None:
                        rendered_without_targets += 1
                    if record["observation"] is not None:
                        with_sidecar += 1
                        cov = record["observation"]["coverage"]
                        total = cov["total_text_nodes"]
                        if total == 0:
                            zero_text_node_docs += 1
                        else:
                            coverage_ratios.append(cov["matched_text_nodes"] / total)
                            if cov["matched_text_nodes"] == total:
                                full_coverage += 1
                    out_f.write(json.dumps(record, sort_keys=True) + "\n")
                    out_f.flush()
                    if progress_every and position % progress_every == 0:
                        elapsed = time.monotonic() - started
                        print(
                            f"[m2b] {position}/{len(pending)} "
                            f"rendered={status_counts['rendered']} "
                            f"failed={status_counts['failed']} "
                            f"({elapsed / position:.1f}s/doc)",
                            flush=True,
                        )
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
    finally:
        lock_path.unlink(missing_ok=True)

    elapsed = time.monotonic() - started
    return {
        "migration_version": MIGRATION_VERSION,
        "store": str(store_dir),
        "render_profile": render_profile,
        "fresh_backup": backup,
        "processed": len(pending),
        "rendered": status_counts["rendered"],
        "failed": status_counts["failed"],
        "skipped": status_counts["skipped"],
        "skipped_already_done": skipped_already_done,
        "duplicate_ledger_rows": duplicate_rows,
        "stale_derivation_rows": stale_derivation_rows,
        "rendered_without_targets": rendered_without_targets,
        "failure_codes": dict(sorted(failure_codes.items(), key=lambda kv: -kv[1])),
        "warning_codes": dict(sorted(warning_codes.items(), key=lambda kv: -kv[1])),
        "coverage": {
            "with_sidecar": with_sidecar,
            "full_coverage": full_coverage,
            "zero_text_node_docs": zero_text_node_docs,
            "mean_ratio": (
                sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else None
            ),
            "min_ratio": min(coverage_ratios) if coverage_ratios else None,
        },
        "seconds_total": round(elapsed, 2),
        "seconds_per_doc": round(elapsed / len(pending), 2) if pending else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.dataset.migration.m2b", description=__doc__
    )
    parser.add_argument("store_dir", type=Path, help="M2a store root (index.jsonl + objects/)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="render parallelism (default 1 = sequential); each worker "
             "converts through its own LibreOffice profile",
    )
    parser.add_argument(
        "--allow-skipped",
        action="store_true",
        help="permit a compile-only run (skipped rows) when the render probe fails",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="back up and clear renders.jsonl before the run (renderer profile change)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="re-attempt rows recorded as failed (recovery after an outage)",
    )
    parser.add_argument(
        "--allow-mixed-profile",
        action="store_true",
        help="permit resuming a ledger rendered under a different font pack",
    )
    # `default=None` on the render-only switches below (design §2.3, "set_
    # defaults(None)") is deliberate: it is what lets `--align-only`
    # distinguish "the user passed this" from "this is just the normal
    # default" and refuse the combination, rather than silently ignoring a
    # render switch that can never apply to a run that never renders.
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=None,
        help=f"pdftoppm raster DPI (default {_RENDER_DPI}); part of the render profile",
    )
    parser.add_argument(
        "--only-bundles",
        type=Path,
        default=None,
        help="file of bundle ids (one per line) to re-enrich regardless of resume state",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            f"pending bundles grouped into one LibreOffice invocation "
            f"(default {_DEFAULT_BATCH_SIZE}, an ESTIMATE not a measurement — "
            "larger amortizes LibreOffice startup further but grows the cost "
            "of a fallback replay when a batch comes back short)"
        ),
    )
    parser.add_argument(
        "--align-only",
        action="store_true",
        help=(
            "re-derive the Observation Sidecar (+ fragments) for already-"
            "rendered bundles from the STORED pdf — never re-invokes "
            "LibreOffice (design §2). Refuses --fresh and every render-only "
            "switch (--render-dpi/--batch-size/--allow-skipped/"
            "--allow-mixed-profile/--retry-failed)."
        ),
    )
    parser.add_argument(
        "--doc-timeout",
        type=int,
        default=_ALIGN_DOC_TIMEOUT_DEFAULT,
        help=f"--align-only per-document wall-clock budget in seconds "
             f"(default {_ALIGN_DOC_TIMEOUT_DEFAULT}); repeatable on timeout",
    )
    parser.add_argument(
        "--source-store",
        type=str,
        default=None,
        help="--align-only provenance: name of the store these renders came from",
    )
    args = parser.parse_args(argv)

    if args.align_only:
        render_switches_touched = (
            args.fresh
            or args.render_dpi is not None
            or args.batch_size is not None
            or args.allow_skipped
            or args.allow_mixed_profile
            or args.retry_failed
        )
        if render_switches_touched:
            parser.error(
                "--align-only refuses --fresh and render-only switches "
                "(--render-dpi/--batch-size/--allow-skipped/"
                "--allow-mixed-profile/--retry-failed) — it never renders"
            )
    if args.render_dpi is None:
        args.render_dpi = _RENDER_DPI
    if args.batch_size is None:
        args.batch_size = _DEFAULT_BATCH_SIZE

    only_bundles = None
    if args.only_bundles is not None:
        only_bundles = {
            line.strip()
            for line in args.only_bundles.read_text().splitlines()
            if line.strip()
        }
        if not only_bundles:
            parser.error(f"--only-bundles: {args.only_bundles} holds no bundle ids")

    if args.align_only:
        summary = realign_renders(
            args.store_dir,
            only_bundles=only_bundles,
            limit=args.limit,
            workers=args.workers,
            doc_timeout=args.doc_timeout,
            source_store=args.source_store,
            aligner_policy=default_aligner_policy(),
        )
    else:
        summary = enrich_renders(
            args.store_dir,
            only_bundles=only_bundles,
            render_dpi=args.render_dpi,
            limit=args.limit,
            workers=args.workers,
            allow_skipped=args.allow_skipped,
            retry_failed=args.retry_failed,
            allow_mixed_profile=args.allow_mixed_profile,
            fresh=args.fresh,
            batch_size=args.batch_size,
            aligner_policy=default_aligner_policy(),
        )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

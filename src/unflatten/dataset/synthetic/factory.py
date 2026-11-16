"""RFC003 §17 branch-1 synthetic factory (D2 smoke scale).

Per document: canonical AST JSON, strict RFC002 DOCX (+ package manifest),
optional LibreOffice PDF render + pdftoppm page PNGs + Observation Sidecar
(RFC001 §37, built from the stored PDF's text layer — per-RenderView under
the §22.2 frozen-pixels rule), a Canonical Target Bundle + `single_page`
Page Fragments derived over the sidecar (RFC003 §26–§27), and a §14
DocumentBundle envelope validated by `envelope_validator.validate_bundle`.

Output layout (content-addressed, release-root relative):

    out_dir/
      objects/<sha256-hex>     raw artifact bytes (AST, manifest, DOCX, ...)
      index.jsonl              one JCS record per line: the §14 DocumentBundle
                               envelope plus factory-local fields next to it
                               ({"envelope": ..., "render_status": ...})
      run_manifest.json        factory-local run record (renders by hash)

Determinism contract (§20/§22.2): AST and DOCX objects are byte-identical
across runs for the same recipes + indices. PDF/page-PNG bytes are NOT
guaranteed byte-identical across runs or hosts ("frozen pixels" rule
§22.2: accepted page images are frozen as immutable artifacts; a release
must never silently re-render) — the factory records them by hash only.

LibreOffice renders are staged under `~/unflatten-probe/` like
`unflatten.renderer.probe` (the snap LibreOffice here cannot read `/tmp`
or dot-directories), each with a **per-process** `-env:UserInstallation`
profile, so a factory run concurrent with another process — or with the
test suite — does not contend on the profile lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.ast.canonicalizer import canonicalize
from unflatten.ast.hashing import ast_file_hash, ast_hash
from unflatten.dataset import identity
from unflatten.dataset.envelope_validator import validate_bundle
from unflatten.dataset.fragments.envelope import build_fragments, validate_fragments
from unflatten.dataset.observations.pdf_text import PdfTextExtractionError
from unflatten.dataset.observations.sidecar import build_observation_sidecar
from unflatten.dataset.synthetic import recipes as recipes_module
from unflatten.dataset.synthetic.generator import generate_document
from unflatten.dataset.synthetic.recipes import (
    DATASET_NAMESPACE,
    GENERATOR_VERSION,
    IMAGE_ROLE,
    Recipe,
    pyramid_level,
)
from unflatten.dataset.targets.bundle import build_target_bundle, validate_target_bundle
from unflatten.libreoffice import profile_dir as libreoffice_profile_dir
from unflatten.renderer.api import RendererOptions
from unflatten.renderer.compiler import compile_docx
from unflatten.renderer.manifest import compute_font_pack_hash
from unflatten.renderer.probe import probe_available
from unflatten.renderer.resources import DirResourceProvider

_LO_ROOT = Path.home() / "unflatten-probe"
_LO_TIMEOUT_SECONDS = 120
#: Rasterization gets its own, larger budget than the DOCX->PDF
#: conversion: cost scales with page count AND with the square of the
#: DPI, so a document that rasterizes in seconds at 96 DPI can take
#: minutes at 300 (same reasoning as m2b.py:125-127).
_RASTER_TIMEOUT_SECONDS = 600

#: Default DPI of the pdftoppm page rasters below; the Observation
#: Sidecar's page pixel dimensions are computed at the same DPI.
#: Overridable per run via ``run_factory(render_dpi=...)`` / ``--render-dpi``.
_RENDER_DPI = 96

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@dataclass
class DocumentRecord:
    """Factory-local per-document record (not an RFC003 envelope)."""

    recipe_id: str
    root_instance_index: int
    document_bundle_id: str
    ast_hash: str
    artifacts: dict[str, Any]
    render_status: str  # "rendered" | "skipped" | "failed"
    #: sidecar ref + coverage stats + issue codes, or None with a reason
    observation: dict[str, Any] | None = None
    observation_skip_reason: str | None = None
    #: target bundle + single_page fragment refs (RFC003 §26/§27), or None
    #: when no sidecar exists (targets are derived only over rendered pages)
    targets: dict[str, Any] | None = None


@dataclass
class FactoryReport:
    out_dir: Path
    documents: list[DocumentRecord] = field(default_factory=list)

    @property
    def rendered_count(self) -> int:
        return sum(1 for d in self.documents if d.render_status == "rendered")


def _store(out_dir: Path, data: bytes) -> tuple[str, str]:
    """Write `data` to `objects/<sha256>`; returns (hex, relative ref).

    Crash-safe: always writes to a per-process tmp file and `os.replace`s
    it over the object path (atomic, idempotent) — a truncated file left
    by a killed run can never survive as a content-addressed object.
    """
    hex_digest = hashlib.sha256(data).hexdigest()
    path = out_dir / "objects" / hex_digest
    tmp = path.with_name(f"{hex_digest}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hex_digest, f"objects/{hex_digest}"


def _staging_dir(tag: str) -> Path:
    """A LibreOffice/pdftoppm staging dir unique per process AND call
    (`<tag>-<pid>-<uuid>`), so concurrent factory runs — or factory +
    tests — never collide on staging paths. The LibreOffice *profile*
    lock is no longer a cross-process limitation either: `_docx_to_pdf`
    takes a per-process profile.
    """
    return _LO_ROOT / "synthetic" / f"{tag}-{os.getpid()}-{uuid.uuid4().hex}"


def _artifact_ref(ref: str, hex_digest: str, size_bytes: int, mime_type: str) -> dict[str, Any]:
    return {
        "ref": ref,
        "sha256": f"sha256:{hex_digest}",
        "size_bytes": size_bytes,
        "mime_type": mime_type,
    }


def _libreoffice_version() -> str:
    """`libreoffice --version` first line, cached per process; "unknown"
    when unavailable (the sidecar still binds to the stored PDF bytes)."""
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


_LO_VERSION: str | None = None


def _libreoffice_pdf(docx_bytes: bytes, tag: str) -> bytes | None:
    """One headless-LibreOffice DOCX→PDF conversion.

    Returns the PDF bytes, or None when the conversion failed. Staged
    under `~/unflatten-probe/synthetic/<tag>/` (see module docstring).

    The `-env:UserInstallation` profile is **per process**: LibreOffice
    serializes on a per-profile lock, so two processes sharing one
    profile make the loser fail with returncode 0 and empty stderr —
    indistinguishable from a genuinely unconvertible document, which is
    why it survived as "flaky" rather than being read as a bug. Same
    defect and same remedy as `renderer/probe.py` and `migration/m2b.py`;
    this call path was missed when those two were fixed.
    """
    profile_dir = libreoffice_profile_dir("synthetic", process_id=os.getpid())
    profile_dir.mkdir(parents=True, exist_ok=True)
    call_dir = _staging_dir(tag)
    work_dir = call_dir / "work"
    out_dir = call_dir / "out"
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        staged = work_dir / "doc.docx"
        staged.write_bytes(docx_bytes)
        try:
            proc = subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    f"-env:UserInstallation=file://{profile_dir}",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(out_dir),
                    str(staged),
                ],
                capture_output=True,
                text=True,
                timeout=_LO_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        pdf_path = out_dir / "doc.pdf"
        if proc.returncode != 0 or not pdf_path.exists():
            return None
        return pdf_path.read_bytes()
    finally:
        shutil.rmtree(call_dir, ignore_errors=True)


def _pdf_pages_png(
    pdf_bytes: bytes, tag: str, render_dpi: int = _RENDER_DPI
) -> list[bytes]:
    """Rasterize `pdf_bytes` to per-page PNGs at `render_dpi` via pdftoppm
    (empty list when pdftoppm is unavailable or times out).

    `render_dpi` reaches the rasterizer here and the Observation Sidecar in
    `_build_observation`; the two must never be able to disagree, or the
    sidecar records a DPI the pixels were never rendered at.
    """
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
            # A pathological page costs that document its rasters, not the
            # whole run (m2b.py:424-432). Unreachable at 96 DPI; reachable
            # the moment DPI is raised.
            return []
        if proc.returncode != 0:
            return []
        return [p.read_bytes() for p in sorted(call_dir.glob("page-*.png"))]
    finally:
        shutil.rmtree(call_dir, ignore_errors=True)


def _bundle_envelope(
    *,
    recipe: Recipe,
    root_instance_index: int,
    doc_ast_hash: str,
    doc_ast_file_hash: str,
    manifest_records: list[dict[str, Any]],
    ast_ref: dict[str, Any],
    manifest_ref: dict[str, Any],
    rendered: bool,
) -> dict[str, Any]:
    root_key = identity.synthetic_root_key(
        dataset_namespace=DATASET_NAMESPACE,
        generator_version=GENERATOR_VERSION,
        recipe_id=recipe.recipe_id,
        recipe_hash=recipe.hash(),
        root_instance_index=root_instance_index,
    )
    lineage_root_id = identity.lineage_root_id_synthetic(synthetic_root_key=root_key)
    # Semantic manifest hash (distinct from the manifest *file* hash in the
    # ArtifactRef) — domain projection over the sorted resource records.
    resource_manifest_hash = identity.domain_hash(
        "rfc003.resource-manifest/v1", manifest_records
    )
    label_revision_hash = identity.domain_hash(
        "rfc003.label-revision/v1",
        {
            "label_method": "synthetic_generator",
            "label_grade": "synthetic_exact",
            "generator_version": GENERATOR_VERSION,
            "recipe_hash": recipe.hash(),
            "root_instance_index": root_instance_index,
        },
    )
    bundle_id = identity.document_bundle_id(
        lineage_root_id=lineage_root_id,
        ast_hash=doc_ast_hash,
        ast_file_hash=doc_ast_file_hash,
        resource_manifest_hash=resource_manifest_hash,
        label_revision_hash=label_revision_hash,
    )
    return {
        "document_bundle_schema_version": "0.1.0",
        "document_bundle_id": bundle_id,
        "lineage_root_id": lineage_root_id,
        "ast_hash": doc_ast_hash,
        "ast_file_hash": doc_ast_file_hash,
        "resource_manifest_hash": resource_manifest_hash,
        "label_revision_hash": label_revision_hash,
        "source_origin": "synthetic_ast",
        "label_method": "synthetic_generator",
        "label_grade": "synthetic_exact",
        "processing_state": "rendered" if rendered else "compiled",
        "verification_tier": "machine_verified",
        "disposition": "active",
        "revision_id": "rev_000001",
        "supersedes_revision_id": None,
        "ast": ast_ref,
        "resources_manifest": manifest_ref,
    }


def run_factory(
    recipes: list[Recipe] | tuple[Recipe, ...],
    count_per_recipe: int,
    out_dir: Path,
    *,
    resource_bytes: dict[str, bytes] | None = None,
    render: bool | None = None,
    render_dpi: int = _RENDER_DPI,
) -> FactoryReport:
    """Produce `count_per_recipe` documents per recipe into `out_dir`.

    `render=None` auto-detects the LibreOffice probe; `False` skips PDF
    renders entirely (bundles land as `processing_state="compiled"`).
    `render_dpi` is the pdftoppm raster DPI AND the DPI the Observation
    Sidecar records; it does not enter document identity (`_bundle_envelope`
    reads no renderer metadata) but it does move the sidecar hash and every
    `fragment_id` derived over it, which is the correct signal.
    See the module docstring for the layout and the §22.2 frozen-pixels
    determinism caveat on PDF/page bytes.
    """
    out_dir = Path(out_dir)
    (out_dir / "objects").mkdir(parents=True, exist_ok=True)
    stage_root = out_dir / "_stage"
    do_render = probe_available() if render is None else render

    report = FactoryReport(out_dir=out_dir)
    index_lines: list[bytes] = []
    try:
        for recipe in recipes:
            for index in range(count_per_recipe):
                record = _produce_one(
                    recipe, index, out_dir, stage_root, resource_bytes, do_render,
                    render_dpi,
                )
                report.documents.append(record)
                index_lines.append(record.artifacts.pop("_envelope_jcs"))
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)

    (out_dir / "index.jsonl").write_bytes(b"".join(line + b"\n" for line in index_lines))
    run_manifest = {
        "dataset_namespace": DATASET_NAMESPACE,
        "generator_version": GENERATOR_VERSION,
        "count_per_recipe": count_per_recipe,
        # RFC003 §47.4: DPI is part of the recorded render profile, not
        # just of the command line — two runs at different DPI must be
        # distinguishable from their manifests alone. `font_policy` is
        # never overridden by the factory, so the default policy is the
        # one every document was compiled under.
        "render_profile": {
            "render_dpi": render_dpi,
            "font_pack_hash": compute_font_pack_hash(RendererOptions().font_policy),
        },
        "recipes": [{"recipe_id": r.recipe_id, "recipe_hash": r.hash()} for r in recipes],
        "documents": [
            {
                "recipe_id": d.recipe_id,
                "root_instance_index": d.root_instance_index,
                "document_bundle_id": d.document_bundle_id,
                "ast_hash": d.ast_hash,
                "render_status": d.render_status,
                "observation": d.observation,
                "observation_skip_reason": d.observation_skip_reason,
                "targets": d.targets,
                "artifacts": d.artifacts,
            }
            for d in report.documents
        ],
    }
    (out_dir / "run_manifest.json").write_bytes(to_canonical_json(run_manifest))
    return report


def _produce_one(
    recipe: Recipe,
    index: int,
    out_dir: Path,
    stage_root: Path,
    resource_bytes: dict[str, bytes] | None,
    do_render: bool,
    render_dpi: int = _RENDER_DPI,
) -> DocumentRecord:
    ast, resources = generate_document(recipe, index, resource_bytes)
    canonical = canonicalize(ast)
    ast_bytes = to_canonical_json(canonical)
    doc_ast_hash = ast_hash(canonical)
    doc_ast_file_hash = ast_file_hash(ast_bytes)
    ast_hex, ast_ref_path = _store(out_dir, ast_bytes)
    artifacts: dict[str, Any] = {
        "ast": _artifact_ref(ast_ref_path, ast_hex, len(ast_bytes), "application/json")
    }

    # Resource files + resources manifest. Mime types come from the AST
    # resource declarations, never hardcoded.
    mime_by_path = {r["path"]: r["mime_type"] for r in ast.get("resources", [])}
    manifest_records = sorted(
        (
            {
                "path": path,
                "sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
                "size_bytes": len(data),
                "mime_type": mime_by_path.get(path, "application/octet-stream"),
            }
            for path, data in resources.items()
        ),
        key=lambda r: r["path"],
    )
    for path, data in resources.items():
        hex_digest, ref = _store(out_dir, data)
        artifacts.setdefault("resources", []).append(
            {
                "path": path,
                **_artifact_ref(
                    ref, hex_digest, len(data),
                    mime_by_path.get(path, "application/octet-stream"),
                ),
            }
        )
    manifest_bytes = to_canonical_json(manifest_records)
    manifest_hex, manifest_ref_path = _store(out_dir, manifest_bytes)
    manifest_ref = _artifact_ref(
        manifest_ref_path, manifest_hex, len(manifest_bytes), "application/json"
    )
    artifacts["resources_manifest"] = manifest_ref

    # Strict DOCX compile (manifest on) via a per-document staging dir.
    stage = stage_root / f"{recipe.recipe_id}-{index}"
    stage.mkdir(parents=True, exist_ok=True)
    for path, data in resources.items():
        target = stage / "resources" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    docx_path = stage / "doc.docx"
    # The AST's own profile, so the package manifest stops asserting a
    # word-core-v0 render of a visual-profile document. `options.profile`
    # reaches `RendererResult.profile` and the package manifest only
    # (compiler.py:145, manifest.py:76) — never the package bytes — so the
    # stored DOCX and the gate's build-determinism recompile are unaffected.
    options = RendererOptions(
        profile=canonical["profile"], strict=True, emit_manifest=True
    )
    result = compile_docx(
        canonical,
        docx_path,
        options,
        resources=DirResourceProvider(stage / "resources") if resources else None,
    )
    if result.status == "failed":
        raise RuntimeError(
            f"strict compile failed for recipe {recipe.recipe_id!r} index {index} "
            f"(factory bug): " + "; ".join(i.code for i in result.issues)
        )
    docx_bytes = docx_path.read_bytes()
    docx_hex, docx_ref = _store(out_dir, docx_bytes)
    artifacts["docx"] = _artifact_ref(docx_ref, docx_hex, len(docx_bytes), DOCX_MIME)
    package_manifest_bytes = (
        docx_path.with_suffix(".docx.manifest.json").read_bytes()
    )
    pm_hex, pm_ref = _store(out_dir, package_manifest_bytes)
    artifacts["package_manifest"] = _artifact_ref(
        pm_ref, pm_hex, len(package_manifest_bytes), "application/json"
    )

    # Sequential LibreOffice render (see §22.2 frozen-pixels caveat).
    render_status = "skipped"
    observation: dict[str, Any] | None = None
    observation_skip_reason: str | None = "render_skipped"
    targets: dict[str, Any] | None = None
    if do_render:
        pdf_bytes = _libreoffice_pdf(docx_bytes, tag=docx_hex[:16])
        if pdf_bytes is None:
            render_status = "failed"
            observation_skip_reason = "render_failed"
        else:
            render_status = "rendered"
            pdf_hex, pdf_ref = _store(out_dir, pdf_bytes)
            artifacts["pdf"] = _artifact_ref(pdf_ref, pdf_hex, len(pdf_bytes), "application/pdf")
            pages = []
            for page_bytes in _pdf_pages_png(
                pdf_bytes, tag=docx_hex[:16], render_dpi=render_dpi
            ):
                page_hex, page_ref = _store(out_dir, page_bytes)
                pages.append(_artifact_ref(page_ref, page_hex, len(page_bytes), "image/png"))
            artifacts["pages"] = pages
            observation, observation_skip_reason, sidecar = _build_observation(
                canonical, pdf_bytes, options, out_dir, artifacts, render_dpi=render_dpi
            )
            if sidecar is not None:
                targets = _build_targets(canonical, sidecar, out_dir, artifacts)

    return _finish_record(
        recipe=recipe,
        index=index,
        doc_ast_hash=doc_ast_hash,
        doc_ast_file_hash=doc_ast_file_hash,
        manifest_records=manifest_records,
        manifest_ref=manifest_ref,
        artifacts=artifacts,
        render_status=render_status,
        observation=observation,
        observation_skip_reason=observation_skip_reason,
        targets=targets,
    )


def _build_observation(
    canonical: dict[str, Any],
    pdf_bytes: bytes,
    options: RendererOptions,
    out_dir: Path,
    artifacts: dict[str, Any],
    render_dpi: int = _RENDER_DPI,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """Build + store the Observation Sidecar for a rendered document.

    Returns ``(observation_record, skip_reason, sidecar)`` — exactly one
    of record/skip_reason is non-None; ``sidecar`` is the built sidecar
    dict (for target/fragment derivation) or None on failure. The
    sidecar binds to the stored PDF artifact (frozen pixels, RFC003
    §22.2); alignment problems are typed issues recorded as codes next
    to the coverage stats, never silent."""
    renderer_meta = {
        "name": "libreoffice",
        "version": _libreoffice_version(),
        "dpi": render_dpi,
        "font_pack_hash": compute_font_pack_hash(options.font_policy),
    }
    try:
        build = build_observation_sidecar(canonical, pdf_bytes, renderer_meta)
    except PdfTextExtractionError as exc:
        return None, f"pdf_text_extraction_failed: {exc}", None
    sidecar_bytes = to_canonical_json(build.sidecar)
    sidecar_hex, sidecar_ref = _store(out_dir, sidecar_bytes)
    ref = _artifact_ref(sidecar_ref, sidecar_hex, len(sidecar_bytes), "application/json")
    artifacts["observation_sidecar"] = ref
    return {
        "sidecar": ref,
        "coverage": build.coverage(),
        "reading_order_switched_pages": list(build.reading_order_switched_pages),
        "issue_codes": sorted({issue.code for issue in build.issues}),
    }, None, build.sidecar


#: V0 factory fragment window policy (RFC003 §26.4 / OPEN-003-007).
FRAGMENT_WINDOW_POLICY = "single_page"


def _build_targets(
    canonical: dict[str, Any],
    sidecar: dict[str, Any],
    out_dir: Path,
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Derive + store the Canonical Target Bundle and single_page fragments.

    Both are mechanically derived views (never hand-fixed, RFC003 §27/§26);
    a validation failure here is a factory bug, not data to quarantine.
    Returns the ``targets`` record referenced from index.jsonl."""
    bundle = build_target_bundle(canonical)
    fragments, fragment_index = build_fragments(
        canonical, bundle, sidecar, FRAGMENT_WINDOW_POLICY
    )
    issues = validate_target_bundle(bundle) + validate_fragments(
        fragments, fragment_index, bundle, sidecar
    )
    if issues:
        raise RuntimeError(
            "factory-derived targets failed validation (factory bug): "
            + "; ".join(f"{i.code}@{i.path}" for i in issues)
        )

    bundle_bytes = to_canonical_json(bundle)
    bundle_hex, bundle_ref = _store(out_dir, bundle_bytes)
    bundle_artifact = _artifact_ref(
        bundle_ref, bundle_hex, len(bundle_bytes), "application/json"
    )
    artifacts["target_bundle"] = bundle_artifact

    fragment_refs = []
    for fragment in fragments:
        fragment_bytes = to_canonical_json(fragment)
        fragment_hex, fragment_ref = _store(out_dir, fragment_bytes)
        fragment_refs.append(
            _artifact_ref(fragment_ref, fragment_hex, len(fragment_bytes), "application/json")
        )
    # stored fragment index = builder index + per-entry ArtifactRef
    stored_index = {
        **fragment_index,
        "entries": [
            {**entry, "fragment": ref}
            for entry, ref in zip(fragment_index["entries"], fragment_refs)
        ],
    }
    index_bytes = to_canonical_json(stored_index)
    index_hex, index_ref = _store(out_dir, index_bytes)
    index_artifact = _artifact_ref(index_ref, index_hex, len(index_bytes), "application/json")
    artifacts["fragment_index"] = index_artifact

    return {
        "target_bundle": bundle_artifact,
        "fragment_index": index_artifact,
        "fragment_count": fragment_index["fragment_count"],
        "window_policy": FRAGMENT_WINDOW_POLICY,
    }


def _finish_record(
    *,
    recipe: Recipe,
    index: int,
    doc_ast_hash: str,
    doc_ast_file_hash: str,
    manifest_records: list[dict[str, Any]],
    manifest_ref: dict[str, Any],
    artifacts: dict[str, Any],
    render_status: str,
    observation: dict[str, Any] | None,
    observation_skip_reason: str | None,
    targets: dict[str, Any] | None,
) -> DocumentRecord:
    envelope = _bundle_envelope(
        recipe=recipe,
        root_instance_index=index,
        doc_ast_hash=doc_ast_hash,
        doc_ast_file_hash=doc_ast_file_hash,
        manifest_records=manifest_records,
        ast_ref=artifacts["ast"],
        manifest_ref=manifest_ref,
        rendered=render_status == "rendered",
    )
    issues = validate_bundle(envelope)
    if issues:
        raise RuntimeError(
            "factory-built bundle envelope failed validation (factory bug): "
            + "; ".join(f"{i.code}@{i.path}" for i in issues)
        )
    # index.jsonl record: the §14 envelope plus factory-local fields NEXT
    # TO it (never inside — the envelope schema stays unchanged), so a
    # failed render is distinguishable from a skipped one, and a missing
    # sidecar always carries an explicit reason (no silent loss).
    artifacts["_envelope_jcs"] = to_canonical_json(
        {
            "envelope": envelope,
            # DEC-013 §31 pyramid level: an index-row field, never an envelope
            # property (the envelope schema is ratified, additionalProperties
            # false) and never an identity input.
            "pyramid_level": pyramid_level(recipe.recipe_id),
            "render_status": render_status,
            "observation": observation,
            "observation_skip_reason": observation_skip_reason,
            "targets": targets,
        }
    )
    return DocumentRecord(
        recipe_id=recipe.recipe_id,
        root_instance_index=index,
        document_bundle_id=envelope["document_bundle_id"],
        ast_hash=doc_ast_hash,
        artifacts=artifacts,
        render_status=render_status,
        observation=observation,
        observation_skip_reason=observation_skip_reason,
        targets=targets,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.dataset.synthetic.factory",
        description="RFC003 D2 deterministic synthetic factory (smoke scale).",
    )
    parser.add_argument("out_dir", type=Path, help="output directory (created if missing)")
    parser.add_argument(
        "--smoke", action="store_true", help="smoke scale: 17 documents per recipe (119 total)"
    )
    parser.add_argument(
        "--count", type=int, default=2, help="documents per recipe (default 2; --smoke wins)"
    )
    parser.add_argument(
        "--image-asset",
        type=Path,
        # Resolved against the repo root, not the CWD, so the default
        # works no matter where the CLI is invoked from.
        default=Path(__file__).resolve().parents[4] / "tests/data/assets/tiny_2x2.png",
        help="PNG used by the inline-image recipe (default: repo test asset)",
    )
    parser.add_argument(
        "--no-render", action="store_true", help="skip LibreOffice PDF renders"
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=_RENDER_DPI,
        help=f"pdftoppm raster DPI (default {_RENDER_DPI}); part of the render profile",
    )
    parser.add_argument(
        "--roster",
        choices=("core", "d", "f", "all"),
        default="core",
        help=(
            "recipe roster (default core). The default is deliberately "
            "opt-out: the gate and split fixtures build from it, so new "
            "tiers must never enter them implicitly."
        ),
    )
    args = parser.parse_args(argv)

    png_bytes = args.image_asset.read_bytes()
    # Rosters resolve through `recipes.ROSTERS`. The `choices` tuple above
    # and that registry live in two modules and can drift, so a name the
    # registry does not offer fails loudly instead of raising KeyError.
    rosters = recipes_module.ROSTERS
    if args.roster not in rosters:
        parser.error(
            f"roster {args.roster!r} is not available: "
            f"recipes.ROSTERS offers {sorted(rosters)}"
        )
    recipes = rosters[args.roster](png_bytes)
    count = 17 if args.smoke else args.count
    report = run_factory(
        recipes,
        count,
        args.out_dir,
        resource_bytes={IMAGE_ROLE: png_bytes},
        render=False if args.no_render else None,
        render_dpi=args.render_dpi,
    )
    summary = {
        "out_dir": str(report.out_dir),
        "documents": len(report.documents),
        "rendered": report.rendered_count,
        "failed_renders": sum(1 for d in report.documents if d.render_status == "failed"),
        "skipped_renders": sum(1 for d in report.documents if d.render_status == "skipped"),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

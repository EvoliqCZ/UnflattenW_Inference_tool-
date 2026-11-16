"""D5 eligibility engine — RFC003 §34.1 universal hard gates over one
factory `index.jsonl` record (envelope + render_status + observation +
targets), reading artifacts from the content-addressed object store
with sha256 verification on every read (`gates.store`).

100 % rule (§34/§34.6): `train_ready` requires ALL applicable gates to
pass — there is no threshold. A failed gate makes the record
`ineligible` with `disposition = quarantine` and §45 reason codes; a
compiled-but-unrendered record (`render_status = "skipped"`) takes the
downgrade path to `partial_ready` (reason `RENDER_SKIPPED`).

Text conservation path (documented decision): the engine ALWAYS takes
the deterministic-recompile path — it recompiles the stored canonical
AST via RFC002 `compile_docx` and runs `verify_text_conservation`
against the recompiled package, using the recompile's own
`technical_boundary_paragraph_positions` metric. The stored-manifest
path is not available today because (a) the factory `index.jsonl`
record carries no DOCX/package-manifest ArtifactRef and (b) the stored
package manifest does not record the renderer metrics the verifier
needs. The recompile is deterministic (RFC002), so it reproduces the
stored package byte-for-byte for an untampered record; the recompile
result also feeds the no-silent-loss accounting (§34.1-15) and the
approximation allowlist gate (§34.1-16).

M2a+renders records (built by ``migration.m2c``) are evaluated with the
same gates: record-embedded render-lane ``artifacts`` refs take
precedence over the run manifest, the rights/privacy gates READ the
record's explicit ``rights`` block (basis DEC-001-option-A;
``privacy_review: pending`` passes with a recorded
PRIVACY_REVIEW_PENDING warning — dev inventory only; the training path
refuses that record at the admission boundary, see ``dataset.samples``).

The integrity gate also enforces the record shape (`render_status` is
the closed set rendered|skipped|failed — anything else is
`RECORD_MALFORMED`, never coerced), defensively parses the resources
manifest (non-list top level, missing/duplicate `path` entries are
integrity failures, never crashes), and hash-verifies the run-manifest
render-lane artifacts (DOCX, package manifest, PDF, page images) when
the run manifest records refs for the record's document — a rendered
record with missing refs fails integrity. The `build_determinism` gate
compares the recompiled package hash against the stored DOCX
(`NONDETERMINISTIC_BUILD` on mismatch).

Gates implemented vs §34.1: 1 integrity, 2 rights, 3 privacy, 4 AST
schema, 5 AST semantics, 6 profile, 8 canonicalization idempotence,
9/10/11 renderer preflight+compile+package (all enforced inside the
strict `compile_docx` recompile), 12 open/render probe evidence
(`render_status`, folded into the compile gate), 13 text conservation,
15 no silent loss, 16 approximations allowlist. Deferred (module has no
inputs for them yet): 7 resource decode probing beyond MIME/hash/size
(hash+size+MIME verified via the store), 14 structure conservation as a
separate stream (partially covered by text events + target-bundle
validation), 17 split integrity and 18 lineage completeness (split and
leakage engines are separate D5 deliverables).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.ast.canonicalizer import canonicalize
from unflatten.ast.hashing import ast_file_hash, ast_hash
from unflatten.ast.issues import Issue
from unflatten.ast.schema_validator import validate_schema
from unflatten.ast.semantic_validator import validate_semantics
from unflatten.dataset.envelope_validator import validate_bundle
from unflatten.dataset.fragments.envelope import validate_fragments
from unflatten.dataset.gates.result import (
    GateCheck,
    GateResult,
    validate_gate_result,
)
from unflatten.dataset.gates.store import StoreError, read_object
from unflatten.dataset.gates.task_eligibility import task_eligibility
from unflatten.dataset.observations.sidecar import validate_observation_sidecar
from unflatten.dataset.targets.bundle import validate_target_bundle
from unflatten.dataset.targets.masks import validate_mask_projection
from unflatten.renderer.api import RendererOptions, RendererResult
from unflatten.renderer.compiler import compile_docx
from unflatten.renderer.text_extract import verify_text_conservation

#: Capability profiles a document may declare and still pass the gate.
#: `word-visual-v1` is admitted from DEC-003 §1 on; `word-core-v0` stays
#: admissible because the `m2a-v2` store is deliberately kept comparable
#: (DEC-003 §3) and its documents must remain gateable. Anything else —
#: `word-extended-v2`, a typo, a missing key — is still PROFILE_UNSUPPORTED:
#: the set is closed, it just has two members now.
SUPPORTED_PROFILES: frozenset[str] = frozenset(
    {"word-core-v0", "word-visual-v1", "word-visual-v1.1"}
)
AST_VERSION_PREFIX = "0.1."

#: Closed set of valid record render states — anything else (including a
#: missing field) is RECORD_MALFORMED, never silently coerced.
RENDER_STATUSES: tuple[str, ...] = ("rendered", "skipped", "failed")

#: run_manifest render-lane artifact keys hash-verified by the integrity
#: gate; a rendered record must additionally record pdf + pages refs.
_RENDER_LANES: tuple[str, ...] = ("docx", "package_manifest", "pdf")

#: Stable gate ids, in report order (§34.1 subset — see module docstring).
GATE_IDS: tuple[str, ...] = (
    "integrity",
    "rights",
    "privacy",
    "ast_schema",
    "ast_semantics",
    "profile",
    "canonicalization",
    "compile",
    "build_determinism",
    "text_conservation",
    "no_silent_loss",
    "approximations",
    "observation",
    "targets",
)

#: §45 reason code recorded when the named gate fails.
_GATE_REASON: dict[str, str] = {
    "integrity": "OBJECT_INTEGRITY_FAILED",
    "rights": "RIGHTS_UNKNOWN",
    "privacy": "PRIVACY_REVIEW_REQUIRED",
    "ast_schema": "AST_INVALID",
    "ast_semantics": "AST_INVALID",
    "profile": "PROFILE_UNSUPPORTED",
    "canonicalization": "NONDETERMINISTIC_BUILD",
    "compile": "RENDER_FAILED",
    "build_determinism": "NONDETERMINISTIC_BUILD",
    "text_conservation": "TEXT_CONSERVATION_FAILED",
    "no_silent_loss": "VISIBLE_CONTENT_DROPPED",
    "approximations": "APPROXIMATION_NOT_ALLOWED",
    "observation": "MAPPING_INCOMPLETE",
    "targets": "TARGETS_INVALID",
}

#: RFC002 approximation issue codes allowlisted for TRAIN_READY (§34.1-16).
APPROXIMATION_ALLOWLIST: frozenset[str] = frozenset({"FONT_FALLBACK"})

_SYNTHETIC_BASIS = {
    "privacy_class": "synthetic",
    "training_allowed": True,
    "basis": "synthetic_corpus_auto_pass",
}

#: privacy_class values the gate-result details schema accepts.
_PRIVACY_CLASSES: frozenset[str] = frozenset(
    {
        "synthetic",
        "public_non_personal",
        "public_personal",
        "internal_non_sensitive",
        "internal_restricted",
        "sensitive_personal",
        "prohibited",
    }
)

#: privacy_review states that pass the privacy gate outright; "pending"
#: passes with a recorded PRIVACY_REVIEW_PENDING warning (documented dev
#: release-candidate decision — the review is tracked, not skipped).
_PRIVACY_REVIEW_PASS: frozenset[str] = frozenset({"approved", "not_required"})


def _issue(code: str, path: str, message: str, severity: str = "error") -> Issue:
    return Issue(code=code, severity=severity, node_id=None, path=path, message=message)


def _store_issue(exc: StoreError, path: str) -> Issue:
    return _issue(exc.code, path, str(exc))


def _all_node_ids(canonical: dict[str, Any]) -> set[str]:
    """All AST node ids under `sections` (the renderer accounting scope)."""
    ids: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("id"), str):
                ids.add(node["id"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(canonical.get("sections", []))
    return ids


class _DictResourceProvider:
    """RFC002 ResourceProvider over resource bytes read from the store."""

    def __init__(self, blobs: dict[str, bytes]) -> None:
        self._blobs = blobs

    def open(self, path: str) -> bytes:
        if path not in self._blobs:
            raise FileNotFoundError(f"resource {path!r} not present in object store")
        return self._blobs[path]


class _Evaluation:
    """Single-record evaluation state; one instance per evaluate_document."""

    def __init__(self, record: dict[str, Any], out_dir: Path) -> None:
        self.record = record
        self.out_dir = Path(out_dir)
        self.envelope: dict[str, Any] = record.get("envelope") or {}
        # kept verbatim — validity is checked by the integrity gate
        # (RECORD_MALFORMED), never coerced to a default
        self.render_status: Any = record.get("render_status")
        self.checks: dict[str, GateCheck] = {}
        # loaded artifacts
        self.ast_bytes: bytes | None = None
        self.resources: dict[str, bytes] | None = None
        self.sidecar_bytes: bytes | None = None
        self.bundle_bytes: bytes | None = None
        self.index_bytes: bytes | None = None
        self.mask_bytes: bytes | None = None
        # derived state
        self.ast_doc: dict[str, Any] | None = None
        self.canonical: dict[str, Any] | None = None
        self.compile_result: RendererResult | None = None
        self.docx_bytes: bytes | None = None
        self.stored_docx_sha256: str | None = None
        self.sidecar: dict[str, Any] | None = None
        # imported (docx_importer) rendered record with matched < total
        # coverage: partial supervision, downgraded — never quarantined
        self.observation_partial = False

    def _set(
        self,
        gate_id: str,
        status: str,
        issues: list[Issue] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.checks[gate_id] = GateCheck(
            gate_id=gate_id, status=status, issues=issues or [], details=details
        )

    def _from_issues(self, gate_id: str, issues: list[Issue]) -> None:
        self._set(gate_id, "fail" if issues else "pass", issues)

    def _read(self, ref: Any, path: str, issues: list[Issue]) -> bytes | None:
        try:
            return read_object(self.out_dir, ref)
        except StoreError as exc:
            issues.append(_store_issue(exc, path))
            return None

    # --- gates -----------------------------------------------------------

    def gate_integrity(self) -> None:
        """§34.1-1: every referenced object exists and its sha256 matches.

        Also validates the §14 envelope itself — refs are only followed
        from a schema-valid envelope. Resource MIME/hash/size (§34.1-7)
        are verified here via the resources manifest records; the record
        shape (`render_status` closed set) and the run-manifest
        render-lane artifacts (DOCX, package manifest, PDF, page images)
        are verified here too."""
        malformed: list[Issue] = []
        if self.render_status not in RENDER_STATUSES:
            malformed.append(
                _issue(
                    "RECORD_MALFORMED",
                    "/render_status",
                    f"render_status {self.render_status!r} is not one of "
                    f"{RENDER_STATUSES}",
                )
            )
        issues = list(validate_bundle(self.envelope)) if self.envelope else [
            _issue("ENVELOPE_MISSING", "/envelope", "record has no envelope")
        ]
        if issues:
            self._set("integrity", "fail", malformed + issues)
            return
        issues = malformed

        self.ast_bytes = self._read(self.envelope["ast"], "/envelope/ast", issues)
        manifest_bytes = self._read(
            self.envelope["resources_manifest"], "/envelope/resources_manifest", issues
        )
        if manifest_bytes is not None:
            try:
                records = json.loads(manifest_bytes)
            except ValueError as exc:
                records = None
                issues.append(
                    _issue(
                        "RESOURCE_MANIFEST_PARSE_ERROR",
                        "/envelope/resources_manifest",
                        f"resources manifest is not valid JSON: {exc}",
                    )
                )
            if records is not None and not isinstance(records, list):
                records = None
                issues.append(
                    _issue(
                        "RESOURCE_MANIFEST_MALFORMED",
                        "/envelope/resources_manifest",
                        "resources manifest top level is not a list",
                    )
                )
            if records is not None:
                self.resources = {}
                seen_paths: set[str] = set()
                for k, rec in enumerate(records):
                    path = rec.get("path") if isinstance(rec, dict) else None
                    if not isinstance(path, str) or not path:
                        issues.append(
                            _issue(
                                "RESOURCE_MANIFEST_MALFORMED",
                                f"/resources_manifest/{k}",
                                "resource record has no string 'path'",
                            )
                        )
                        continue
                    if path in seen_paths:
                        issues.append(
                            _issue(
                                "RESOURCE_MANIFEST_DUPLICATE_PATH",
                                f"/resources_manifest/{k}",
                                f"duplicate resource path {path!r} "
                                "(last-wins overwrite hazard)",
                            )
                        )
                        continue
                    seen_paths.add(path)
                    ref = {
                        "ref": f"objects/{str(rec.get('sha256', '')).removeprefix('sha256:')}",
                        "sha256": rec.get("sha256"),
                        "size_bytes": rec.get("size_bytes"),
                        # pre-m2a-v2 manifest rows carry no mime_type —
                        # default, never fail integrity on it
                        "mime_type": rec.get("mime_type") or "application/octet-stream",
                    }
                    data = self._read(ref, f"/resources_manifest/{k}", issues)
                    if data is not None:
                        self.resources[path] = data

        issues.extend(self._verify_render_artifacts())

        observation = self.record.get("observation")
        if observation is not None:
            self.sidecar_bytes = self._read(
                observation["sidecar"], "/observation/sidecar", issues
            )
        targets = self.record.get("targets")
        if targets is not None:
            self.bundle_bytes = self._read(
                targets["target_bundle"], "/targets/target_bundle", issues
            )
            self.index_bytes = self._read(
                targets["fragment_index"], "/targets/fragment_index", issues
            )
            mask_ref = targets.get("mask_projection")
            if mask_ref is not None:
                self.mask_bytes = self._read(
                    mask_ref, "/targets/mask_projection", issues
                )
        self._from_issues("integrity", issues)

    def _verify_render_artifacts(self) -> list[Issue]:
        """§34.1-1 over the run-manifest render lane: when the run
        manifest records artifact refs for this record's document they
        are hash-verified; a rendered record with missing refs is an
        integrity failure. The stored DOCX sha256 is kept for the
        build-determinism gate."""
        issues: list[Issue] = []
        # record-embedded render-lane refs (M2b renders.jsonl shape) take
        # precedence; factory records carry theirs in run_manifest.json
        record_artifacts = self.record.get("artifacts")
        artifacts: dict[str, Any] = (
            dict(record_artifacts) if isinstance(record_artifacts, dict) else {}
        )
        manifest_path = self.out_dir / "run_manifest.json"
        if not artifacts and manifest_path.is_file():
            try:
                run_manifest = json.loads(manifest_path.read_bytes())
            except ValueError:
                run_manifest = {}
            for doc in run_manifest.get("documents", []):
                if (
                    isinstance(doc, dict)
                    and doc.get("document_bundle_id")
                    == self.envelope.get("document_bundle_id")
                ):
                    artifacts = doc.get("artifacts") or {}
                    break
        if self.render_status == "rendered":
            for lane in (*_RENDER_LANES, "pages"):
                if lane not in artifacts:
                    issues.append(
                        _issue(
                            "RENDER_ARTIFACT_REF_MISSING",
                            f"/run_manifest/artifacts/{lane}",
                            f"rendered document records no {lane!r} artifact ref",
                        )
                    )
            # the KEY existing is not the model input existing: a
            # rendered document with zero page images has nothing to
            # feed a model, and every task view's input is a page image
            recorded_pages = artifacts.get("pages")
            if "pages" in artifacts and (
                not isinstance(recorded_pages, list) or not recorded_pages
            ):
                issues.append(
                    _issue(
                        "RENDER_PAGES_EMPTY",
                        "/run_manifest/artifacts/pages",
                        "rendered document records "
                        f"{recorded_pages!r} page images — the model input "
                        "is empty",
                    )
                )
        for lane in _RENDER_LANES:
            ref = artifacts.get(lane)
            if ref is None:
                continue
            data = self._read(ref, f"/run_manifest/artifacts/{lane}", issues)
            if lane == "docx" and data is not None:
                self.stored_docx_sha256 = ref.get("sha256")
        pages = artifacts.get("pages")
        if isinstance(pages, list):
            for k, page in enumerate(pages):
                self._read(page, f"/run_manifest/artifacts/pages/{k}", issues)
        return issues

    def gate_rights_privacy(self) -> None:
        """§34.1-2/-3: explicit rights + privacy decisions. Synthetic
        corpus records auto-pass with an explicit recorded basis
        (privacy_class synthetic, training_allowed true) — the gates
        exist and record why, they are never silently skipped (§50).

        An EXPLICIT record-level rights block always takes precedence
        over the synthetic auto-pass: a recorded rights/privacy decision
        (e.g. an explicit block) must never be shadowed by an origin
        heuristic."""
        rights = self.record.get("rights")
        if isinstance(rights, dict):
            self._gate_rights_block(rights)
            return
        if self.envelope.get("source_origin") == "synthetic_ast":
            self._set("rights", "pass", details=dict(_SYNTHETIC_BASIS))
            self._set("privacy", "pass", details=dict(_SYNTHETIC_BASIS))
            return
        self._set(
            "rights",
            "fail",
            [
                _issue(
                    "RIGHTS_DECISION_MISSING",
                    "/envelope/source_origin",
                    "no rights decision store exists for non-synthetic origins yet",
                )
            ],
        )
        self._set(
            "privacy",
            "fail",
            [
                _issue(
                    "PRIVACY_REVIEW_MISSING",
                    "/envelope/source_origin",
                    "no privacy review store exists for non-synthetic origins yet",
                )
            ],
        )

    def _gate_rights_block(self, rights: dict[str, Any]) -> None:
        """§34.1-2/-3 over an explicit record-level rights block (M2a
        shape, basis DEC-001-option-A): the gate READS the recorded
        decision — training_allowed must be True with non-null evidence
        and a known privacy_class. `privacy_review: "pending"` passes
        the privacy gate with a recorded PRIVACY_REVIEW_PENDING warning
        (documented dev release-candidate decision, never a silent
        skip); any other unreviewed state fails it."""
        evidence = rights.get("evidence")
        privacy_class = rights.get("privacy_class")
        issues: list[Issue] = []
        if rights.get("training_allowed") is not True:
            issues.append(
                _issue(
                    "RIGHTS_TRAINING_NOT_ALLOWED",
                    "/rights/training_allowed",
                    f"rights block records training_allowed="
                    f"{rights.get('training_allowed')!r}",
                )
            )
        if not isinstance(evidence, str) or not evidence:
            issues.append(
                _issue(
                    "RIGHTS_EVIDENCE_MISSING",
                    "/rights/evidence",
                    f"rights block carries no evidence (got {evidence!r})",
                )
            )
        if privacy_class not in _PRIVACY_CLASSES:
            issues.append(
                _issue(
                    "RIGHTS_PRIVACY_CLASS_UNKNOWN",
                    "/rights/privacy_class",
                    f"privacy_class {privacy_class!r} is not in the "
                    "pinned registry",
                )
            )
        if issues:
            self._set("rights", "fail", issues)
        else:
            details = {
                "privacy_class": privacy_class,
                "training_allowed": True,
                "basis": evidence,
            }
            self._set("rights", "pass", details=details)

        review = rights.get("privacy_review")
        basis_details = None
        if not issues:
            basis_details = {
                "privacy_class": privacy_class,
                "training_allowed": True,
                "basis": f"{evidence} privacy_review={review}",
            }
        # hard privacy-class blocks — a recorded training_allowed=true can
        # NEVER override these classes (§34.1-3)
        if privacy_class == "prohibited":
            self._set(
                "privacy",
                "fail",
                [
                    _issue(
                        "PRIVACY_PROHIBITED",
                        "/rights/privacy_class",
                        "privacy_class 'prohibited' never trains, regardless "
                        "of training_allowed or review state",
                    )
                ],
            )
            return
        if privacy_class == "sensitive_personal":
            self._set(
                "privacy",
                "fail",
                [
                    _issue(
                        "PRIVACY_SENSITIVE_REVIEW_REQUIRED",
                        "/rights/privacy_class",
                        "privacy_class 'sensitive_personal' is blocked "
                        "pending a dedicated sensitive-data review, "
                        "regardless of training_allowed",
                    )
                ],
            )
            return
        if privacy_class == "internal_restricted":
            if review == "approved":
                self._set("privacy", "pass", details=basis_details)
            else:
                self._set(
                    "privacy",
                    "fail",
                    [
                        _issue(
                            "PRIVACY_REVIEW_MISSING",
                            "/rights/privacy_review",
                            "privacy_class 'internal_restricted' passes only "
                            f"with an approved privacy review (got {review!r})",
                        )
                    ],
                )
            return
        if review in _PRIVACY_REVIEW_PASS:
            self._set("privacy", "pass", details=basis_details)
        elif review == "pending":
            # RFC005 §9.2 classifies an incomplete privacy decision as
            # BLOCKED for training. The gate still passes the record into
            # the dev inventory and RECORDS why; the training path reads
            # this issue code and refuses (`samples.require_privacy_review`)
            # — the rejection lives at the admission boundary, not in a
            # new eligibility tier.
            self._set(
                "privacy",
                "pass",
                [
                    _issue(
                        "PRIVACY_REVIEW_PENDING",
                        "/rights/privacy_review",
                        "privacy review is pending — dev/research inventory "
                        "only (RFC005 §9.2/§9.3), never a training admission",
                        severity="warning",
                    )
                ],
                details=basis_details,
            )
        else:
            self._set(
                "privacy",
                "fail",
                [
                    _issue(
                        "PRIVACY_REVIEW_MISSING",
                        "/rights/privacy_review",
                        f"privacy_review {review!r} is neither approved, "
                        "not_required nor pending",
                    )
                ],
            )

    def gate_ast(self) -> None:
        """§34.1-4/-5/-6: AST schema, semantics, profile pin."""
        if self.ast_bytes is None:
            self._set("ast_schema", "not_applicable")
            self._set("ast_semantics", "not_applicable")
            self._set("profile", "not_applicable")
            return
        try:
            self.ast_doc = json.loads(self.ast_bytes)
        except ValueError as exc:
            self._set(
                "ast_schema",
                "fail",
                [_issue("AST_PARSE_ERROR", "/ast", f"stored AST is not valid JSON: {exc}")],
            )
            self._set("ast_semantics", "not_applicable")
            self._set("profile", "not_applicable")
            return

        schema_issues = validate_schema(self.ast_doc)
        self._from_issues("ast_schema", schema_issues)
        if schema_issues:
            self._set("ast_semantics", "not_applicable")
            self.ast_doc = None  # downstream gates need a schema-valid AST
        else:
            self._from_issues("ast_semantics", validate_semantics(self.ast_doc))
            if self.checks["ast_semantics"].status == "fail":
                self.ast_doc = None

        profile_issues: list[Issue] = []
        doc = json.loads(self.ast_bytes)
        if doc.get("profile") not in SUPPORTED_PROFILES:
            profile_issues.append(
                _issue(
                    "PROFILE_UNSUPPORTED",
                    "/ast/profile",
                    f"profile {doc.get('profile')!r} is not one of the supported "
                    f"{sorted(SUPPORTED_PROFILES)}",
                )
            )
        version = doc.get("ast_version")
        if not isinstance(version, str) or not version.startswith(AST_VERSION_PREFIX):
            profile_issues.append(
                _issue(
                    "AST_VERSION_UNSUPPORTED",
                    "/ast/ast_version",
                    f"ast_version {version!r} is outside the pinned "
                    f"{AST_VERSION_PREFIX}x line",
                )
            )
        self._from_issues("profile", profile_issues)

    def gate_canonicalization(self) -> None:
        """§34.1-8: canonicalize twice => identical; stored form is
        canonical; envelope hashes recompute from the stored bytes."""
        if self.ast_doc is None or self.ast_bytes is None:
            self._set("canonicalization", "not_applicable")
            return
        issues: list[Issue] = []
        canonical = canonicalize(self.ast_doc)
        if canonicalize(canonical) != canonical:
            issues.append(
                _issue(
                    "CANONICALIZATION_NOT_IDEMPOTENT",
                    "/ast",
                    "canonicalize(canonicalize(ast)) differs from canonicalize(ast)",
                )
            )
        # ONE AST identity per bundle (design D1): the stored bytes ARE
        # the canonical serialization and the envelope hashes recompute
        # from that same tree — no label_method exemption. A store whose
        # tree hashes to something other than its canonical form names
        # one identity in the envelope while targets derive from another.
        if to_canonical_json(canonical) != self.ast_bytes:
            issues.append(
                _issue(
                    "STORED_AST_NOT_CANONICAL",
                    "/ast",
                    "stored AST bytes are not the canonical serialization",
                )
            )
        if ast_hash(canonical) != self.envelope.get("ast_hash"):
            issues.append(
                _issue(
                    "AST_HASH_MISMATCH",
                    "/envelope/ast_hash",
                    "envelope ast_hash does not recompute from the stored AST",
                )
            )
        if ast_file_hash(self.ast_bytes) != self.envelope.get("ast_file_hash"):
            issues.append(
                _issue(
                    "AST_FILE_HASH_MISMATCH",
                    "/envelope/ast_file_hash",
                    "envelope ast_file_hash does not recompute from the stored bytes",
                )
            )
        self._from_issues("canonicalization", issues)
        if not issues:
            self.canonical = canonical

    def gate_compile(self) -> None:
        """§34.1-9..12: strict deterministic recompile (preflight, RFC002
        compile, OPC/OOXML package + XSD validation all run inside
        `compile_docx` with strict options), plus the recorded render
        probe evidence — `render_status = "failed"` fails the gate;
        `"skipped"` only takes the downgrade path (module docstring)."""
        issues: list[Issue] = []
        if self.render_status == "failed":
            issues.append(
                _issue(
                    "RENDER_FAILED",
                    "/render_status",
                    "the factory recorded a failed render for this document",
                )
            )
        if self.canonical is None:
            self._set("compile", "fail" if issues else "not_applicable", issues)
            return
        with tempfile.TemporaryDirectory(prefix="unflatten-gates-") as tmp:
            docx_path = Path(tmp) / "recompiled.docx"
            try:
                result = compile_docx(
                    self.canonical,
                    docx_path,
                    RendererOptions(strict=True),
                    resources=_DictResourceProvider(self.resources or {}),
                )
            except Exception as exc:  # renderer bugs must not crash the gate run
                issues.append(
                    _issue("COMPILE_ERROR", "/ast", f"recompile raised: {exc}")
                )
                self._set("compile", "fail", issues)
                return
            if result.status == "failed":
                issues.append(
                    _issue(
                        "COMPILE_FAILED",
                        "/ast",
                        "strict recompile failed: "
                        + "; ".join(i.code for i in result.issues),
                    )
                )
            else:
                self.compile_result = result
                self.docx_bytes = docx_path.read_bytes()
        self._from_issues("compile", issues)

    def gate_build_determinism(self) -> None:
        """§22/§34.1: the strict recompile must reproduce the stored
        DOCX package byte-for-byte — a hash mismatch against the
        hash-verified stored package is a NONDETERMINISTIC_BUILD."""
        if self.docx_bytes is None or self.stored_docx_sha256 is None:
            self._set("build_determinism", "not_applicable")
            return
        actual = f"sha256:{hashlib.sha256(self.docx_bytes).hexdigest()}"
        issues = (
            []
            if actual == self.stored_docx_sha256
            else [
                _issue(
                    "RECOMPILED_PACKAGE_HASH_MISMATCH",
                    "/run_manifest/artifacts/docx",
                    f"recompiled package hash {actual} does not match the "
                    f"stored package {self.stored_docx_sha256}",
                )
            ]
        )
        self._from_issues("build_determinism", issues)

    def gate_text_conservation(self) -> None:
        """§34.1-13: 100 % visible text event conservation, via the
        deterministic recompile path (module docstring)."""
        if self.canonical is None or self.docx_bytes is None or self.compile_result is None:
            self._set("text_conservation", "not_applicable")
            return
        issues = verify_text_conservation(
            self.canonical,
            self.docx_bytes,
            self.compile_result.metrics.get("technical_boundary_paragraph_positions", ()),
            # Story parts need the PER-PART form. An empty header must still
            # carry one `<w:p/>` and `footnotes.xml` must carry the separator
            # and continuationSeparator notes — OOXML requires them and the
            # AST correctly models none of them. Passing only the legacy
            # body-level positions marked 2,355 `word-visual-v1` documents
            # ineligible for text loss that never happened.
            self.compile_result.metrics.get("technical_paragraph_paths"),
        )
        self._from_issues("text_conservation", list(issues))

    def gate_no_silent_loss(self) -> None:
        """§34.1-15: recompile metrics rendered ∪ unsupported must cover
        every AST node id — nothing may be silently dropped (§6.4)."""
        if self.canonical is None or self.compile_result is None:
            self._set("no_silent_loss", "not_applicable")
            return
        rendered = set(self.compile_result.metrics.get("rendered_node_ids", []))
        unsupported = {
            i.node_id for i in self.compile_result.unsupported if i.node_id is not None
        }
        missing = sorted(_all_node_ids(self.canonical) - rendered - unsupported)
        issues = (
            [
                _issue(
                    "NODE_NOT_ACCOUNTED",
                    "/ast",
                    "node ids neither rendered nor accounted as unsupported: "
                    + ", ".join(missing),
                )
            ]
            if missing
            else []
        )
        self._from_issues("no_silent_loss", issues)

    def gate_approximations(self) -> None:
        """§34.1-16: only allowlisted approximations; any renderer
        `UNSUPPORTED_*` drop is likewise not allowlisted content."""
        if self.compile_result is None:
            self._set("approximations", "not_applicable")
            return
        issues = [
            _issue(
                "APPROXIMATION_NOT_ALLOWED",
                "/ast",
                f"approximation {i.code!r} is not allowlisted",
            )
            for i in self.compile_result.approximations
            if i.code not in APPROXIMATION_ALLOWLIST
        ]
        issues += [
            _issue(
                "UNSUPPORTED_CONTENT",
                "/ast",
                f"renderer reported unsupported content: {i.code}",
            )
            for i in self.compile_result.unsupported
        ]
        self._from_issues("approximations", issues)

    def gate_observation(self) -> None:
        """Observation gate: sidecar present, schema/semantic valid, and
        coverage matched == total (full_ast eligibility, §34.3/§24)."""
        if self.render_status != "rendered":
            self._set("observation", "not_applicable")
            return
        observation = self.record.get("observation")
        if observation is None or self.sidecar_bytes is None:
            reason = self.record.get("observation_skip_reason")
            self._set(
                "observation",
                "fail",
                [
                    _issue(
                        "OBSERVATION_MISSING",
                        "/observation",
                        f"rendered record has no usable observation sidecar "
                        f"(skip reason: {reason!r})",
                    )
                ],
            )
            return
        issues: list[Issue] = []
        try:
            sidecar = json.loads(self.sidecar_bytes)
        except ValueError as exc:
            self._set(
                "observation",
                "fail",
                [
                    _issue(
                        "OBSERVATION_PARSE_ERROR",
                        "/observation/sidecar",
                        f"stored sidecar is not valid JSON: {exc}",
                    )
                ],
            )
            return
        issues.extend(validate_observation_sidecar(sidecar))
        coverage = observation.get("coverage", {})
        coverage_incomplete = coverage.get("matched_text_nodes") != coverage.get(
            "total_text_nodes"
        )
        # alignment diagnostics recorded by the factory are surfaced as
        # warnings; the hard criterion is validity + full coverage
        warnings = [
            _issue(
                "OBSERVATION_ALIGNMENT_DIAGNOSTIC",
                "/observation",
                f"factory recorded alignment issue code {code}",
                severity="warning",
            )
            for code in observation.get("issue_codes", [])
        ]
        # Nothing on these pages was ever observed: every node's placement
        # would be inferred, so the pages carry no scored position. The
        # gate still records the sidecar as valid — but a document with
        # zero observed placement is not training evidence for any task,
        # and `task_eligibility` reads this marker.
        matched = coverage.get("matched_text_nodes")
        total = coverage.get("total_text_nodes")
        if isinstance(matched, int) and isinstance(total, int) and total and not matched:
            warnings.append(
                _issue(
                    "OBSERVATION_NO_PLACEMENT",
                    "/observation/coverage",
                    f"0 of {total} text nodes were observed — every "
                    "placement on these pages would be inferred, so the "
                    "document carries no observed supervision",
                    severity="warning",
                )
            )
        importer = self.envelope.get("label_method") == "docx_importer"
        if coverage_incomplete and importer and not issues:
            # imported real documents (docx_importer): matched < total is
            # partial supervision (RFC003 §33) — retained as a warning
            # marker and downgraded to partial_ready by run(), never
            # quarantined; full_ast eligibility stays out of reach.
            # Synthetic records keep the strict matched == total rule.
            self.observation_partial = True
            warnings.append(
                _issue(
                    "OBSERVATION_COVERAGE_INCOMPLETE",
                    "/observation/coverage",
                    f"matched_text_nodes {coverage.get('matched_text_nodes')!r} != "
                    f"total_text_nodes {coverage.get('total_text_nodes')!r} — "
                    "imported document, partial supervision (MAPPING_INCOMPLETE)",
                    severity="warning",
                )
            )
            self._set("observation", "pass", warnings)
            self.sidecar = sidecar
            return
        if coverage_incomplete:
            # Synthetic records keep the strict rule, and under
            # `word-visual-v1` that now includes stories: header, footer
            # and note text counts toward `total_text_nodes` (the
            # aligner walks stories via `observations.stories`). A
            # synthetic recipe whose story the aligner cannot place
            # therefore FAILS here rather than degrading to
            # partial_ready — deliberately: a generated document whose
            # own text cannot be found in its own render is a broken
            # recipe, not evidence to salvage. Today's
            # `word_core_v0_recipes` carry no stories; the first v1
            # recipe is where this is met.
            issues.append(
                _issue(
                    "OBSERVATION_COVERAGE_INCOMPLETE",
                    "/observation/coverage",
                    f"matched_text_nodes {coverage.get('matched_text_nodes')!r} != "
                    f"total_text_nodes {coverage.get('total_text_nodes')!r}",
                )
            )
        if issues:
            self._set("observation", "fail", issues + warnings)
        else:
            self._set("observation", "pass", warnings)
            self.sidecar = sidecar

    def gate_targets(self) -> None:
        """Targets gate: Canonical Target Bundle + fragments validate
        cleanly against the AST hash and (valid) sidecar (§26/§27)."""
        if self.render_status != "rendered":
            self._set("targets", "not_applicable")
            return
        targets = self.record.get("targets")
        if targets is None:
            if self.observation_partial:
                # imported doc under partial supervision: targets may be
                # underivable — absent evidence, not a failure
                self._set("targets", "not_applicable")
                return
            self._set(
                "targets",
                "fail",
                [
                    _issue(
                        "TARGETS_MISSING",
                        "/targets",
                        "rendered record carries no target bundle",
                    )
                ],
            )
            return
        if self.sidecar is None or self.bundle_bytes is None or self.index_bytes is None:
            # fragment validation is defined over a valid sidecar/bundle
            self._set("targets", "not_applicable")
            return
        issues: list[Issue] = []
        try:
            bundle = json.loads(self.bundle_bytes)
            stored_index = json.loads(self.index_bytes)
        except ValueError as exc:
            self._set(
                "targets",
                "fail",
                [
                    _issue(
                        "TARGETS_PARSE_ERROR",
                        "/targets",
                        f"stored target artifact is not valid JSON: {exc}",
                    )
                ],
            )
            return
        # One AST identity (design D1): the envelope ast_hash IS the
        # canonical tree's hash, and the target bundle binds to it.
        if bundle.get("ast_hash") != self.envelope.get("ast_hash"):
            issues.append(
                _issue(
                    "TARGET_BUNDLE_HASH_MISMATCH",
                    "/targets/target_bundle",
                    "target bundle ast_hash differs from the canonical AST hash",
                )
            )
        issues.extend(validate_target_bundle(bundle))
        fragments: list[dict[str, Any]] = []
        for k, entry in enumerate(stored_index.get("entries", [])):
            data = self._read(
                entry.get("fragment"), f"/targets/fragment_index/entries/{k}", issues
            )
            if data is None:
                continue
            try:
                fragments.append(json.loads(data))
            except ValueError as exc:
                issues.append(
                    _issue(
                        "TARGETS_PARSE_ERROR",
                        f"/targets/fragment_index/entries/{k}",
                        f"stored fragment is not valid JSON: {exc}",
                    )
                )
        if not issues:
            issues.extend(
                validate_fragments(fragments, stored_index, bundle, self.sidecar)
            )
        mask_issues = self._mask_issues(targets)
        blocking = [i for i in mask_issues if i.severity != "warning"]
        if issues or blocking:
            self._set("targets", "fail", issues + mask_issues)
        else:
            self._set("targets", "pass", mask_issues)

    def _mask_issues(self, targets: dict[str, Any]) -> list[Issue]:
        """§29 font-substitution mask: when the record records one it must
        parse and validate (the integrity gate already hash-verified the
        bytes). An ABSENT mask is recorded as a warning, not a failure:
        the synthetic factory emits no mask and the M2b backfill keeps
        its own ledger, so absence here is a join gap, not evidence that
        substituted fonts are supervised."""
        if targets.get("mask_projection") is None:
            return [
                _issue(
                    "MASK_PROJECTION_ABSENT",
                    "/targets/mask_projection",
                    "record carries no font-substitution mask projection",
                    severity="warning",
                )
            ]
        if self.mask_bytes is None:
            return []  # unreadable: already an integrity failure
        try:
            projection = json.loads(self.mask_bytes)
        except ValueError as exc:
            return [
                _issue(
                    "TARGETS_PARSE_ERROR",
                    "/targets/mask_projection",
                    f"stored mask projection is not valid JSON: {exc}",
                )
            ]
        return validate_mask_projection(projection)

    # --- result ----------------------------------------------------------

    def run(self) -> GateResult:
        self.gate_integrity()
        self.gate_rights_privacy()
        self.gate_ast()
        self.gate_canonicalization()
        self.gate_compile()
        self.gate_build_determinism()
        self.gate_text_conservation()
        self.gate_no_silent_loss()
        self.gate_approximations()
        self.gate_observation()
        self.gate_targets()

        report = [self.checks[gate_id] for gate_id in GATE_IDS]
        failed = [check for check in report if check.status == "fail"]
        if failed:
            eligibility, disposition = "ineligible", "quarantine"
            reasons = {_GATE_REASON[check.gate_id] for check in failed}
            if any(
                issue.code == "RECORD_MALFORMED"
                for check in failed
                for issue in check.issues
            ):
                reasons.add("RECORD_MALFORMED")
            reason_codes = sorted(reasons)
        elif self.render_status == "rendered":
            if self.observation_partial:
                # imported doc, matched < total: partial supervision —
                # downgrade with the MAPPING_INCOMPLETE marker retained
                eligibility, disposition = "partial_ready", "active"
                reason_codes = ["MAPPING_INCOMPLETE"]
            else:
                eligibility, disposition, reason_codes = "train_ready", "active", []
        else:
            # compiled but never rendered: §34.1 render/observation
            # evidence is absent, not failed — downgrade, don't quarantine
            eligibility, disposition = "partial_ready", "active"
            reason_codes = ["RENDER_SKIPPED"]
        return GateResult(
            document_bundle_id=self.envelope.get("document_bundle_id") or "unknown",
            eligibility=eligibility,
            disposition=disposition,
            reason_codes=reason_codes,
            gate_report=report,
        )


def evaluate_document(record: dict[str, Any], out_dir: Path) -> GateResult:
    """Evaluate the §34.1 universal gates for one index.jsonl record."""
    return _Evaluation(record, out_dir).run()


def _node_type_counts(record: dict[str, Any], out_dir: Path) -> dict[str, int]:
    """The record's node-type histogram, read from the stored AST.

    An unreadable AST yields an empty histogram: the record's gates have
    already failed on it, and a failed gate makes every task view
    ineligible regardless of the histogram."""
    try:
        ast_doc = json.loads(read_object(out_dir, record["envelope"]["ast"]))
    except (KeyError, StoreError, ValueError):
        return {}
    counts: dict[str, int] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("type"), str) and isinstance(value.get("id"), str):
                counts[value["type"]] = counts.get(value["type"], 0) + 1
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(ast_doc.get("sections", []))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.dataset.gates.engine",
        description="RFC003 D5 eligibility engine over a factory output directory.",
    )
    parser.add_argument("out_dir", type=Path, help="factory output directory")
    args = parser.parse_args(argv)

    records = [
        json.loads(line)
        for line in (args.out_dir / "index.jsonl").read_bytes().splitlines()
    ]
    lines: list[bytes] = []
    eligibility_counts = {
        "train_ready": 0,
        "eval_ready": 0,
        "partial_ready": 0,
        "ineligible": 0,
    }
    quarantine = 0
    for record in records:
        result = evaluate_document(record, args.out_dir)
        serialized = result.to_dict()
        schema_issues = validate_gate_result(serialized)
        if schema_issues:  # engine bug, never data to quarantine
            raise RuntimeError(
                "engine produced a schema-invalid gate result: "
                + "; ".join(f"{i.code}@{i.path}" for i in schema_issues)
            )
        # §33.4: eligibility is per TaskView, and the release identity
        # covers it — attached AFTER schema validation, as a projection
        # of the validated result, never as new evidence.
        serialized["task_eligibility"] = task_eligibility(
            serialized, _node_type_counts(record, args.out_dir)
        )
        lines.append(to_canonical_json(serialized))
        eligibility_counts[result.eligibility] += 1
        quarantine += result.disposition == "quarantine"

    (args.out_dir / "gate-report.jsonl").write_bytes(
        b"".join(line + b"\n" for line in lines)
    )
    print(
        json.dumps(
            {
                "documents": len(records),
                "eligibility": eligibility_counts,
                "quarantine": quarantine,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from unflatten.service.auth import CAPABILITIES
from unflatten.service.errors import ProblemInvalid, validate_problem_response

REQUESTED_OUTPUTS = frozenset({"prediction_ir", "diagnostic_docx", "docx"})
ACCEPTED_OUTPUTS = frozenset({"prediction_ir", "diagnostic_docx"})
#: DEC-044 Dodatok D (C): side-by-side preview artifact roles, only ever
#: present alongside a diagnostic_docx job (never prediction_ir-only).
_PREVIEW_ROLES = frozenset({"source_preview", "docx_preview"})
JOB_STATES = frozenset({"queued", "running", "succeeded", "failed", "cancelled", "expired"})
JOB_STAGES = frozenset(
    {
        "admission_complete",
        # DEC-044 Dodatok D §1 fix (2026-09-27): live model variant
        # switching -- entered once per job, a no-op when the requested
        # variant is already resident.
        "loading_model",
        "pdf_inspection",
        "pdf_rasterization",
        "png_validation",
        "model_input_projection",
        "feature_extraction",
        "model_decode",
        "prediction_ir_sealing",
        "ast_promotion",
        "renderer_gate",
        "artifact_finalization",
    }
)
DIAGNOSTIC_REASON_CODES = frozenset(
    {
        "HEADLESS_DEFAULTS_V0_APPLIED",
        "DIAGNOSTIC_DOCX_NOT_REQUESTED",
        "DOCX_PAGE_NOT_TERMINATED",
        "DOCX_PAGE_STREAM_INVALID",
        "DOCX_AST_NOT_STRICT",
        "DOCX_RENDER_NOT_READY",
        "DOCX_OUTPUT_TOO_LARGE",
        "DOCX_CHAIN_ERROR",
        # DEC-044 Dodatok C: published with a metric-compatible font substitution
        "DOCX_FONT_SUBSTITUTED",
        # DEC-044 Dodatok D (C): a preview (source_preview/docx_preview) failed
        # to render; the job still succeeds with its diagnostic_docx.
        "PREVIEW_RENDER_FAILED",
    }
)
#: Every artifact role a job may publish (`job_artifacts.role` in schema.sql).
ARTIFACT_ROLES = ACCEPTED_OUTPUTS | _PREVIEW_ROLES
#: `model_commands.kind`/`.state`/`.phase` (schema.sql v7, §4.1).
MODEL_COMMAND_KINDS = frozenset({"download", "load", "delete"})
MODEL_COMMAND_STATES = frozenset({"queued", "running", "succeeded", "failed"})
MODEL_COMMAND_PHASES = frozenset({"downloading", "verifying", "loading", "deleting"})
#: `models_response` variant `download.state` (design §3.8).
CACHED_LOAD_FAILURE_CODES = frozenset({
    "MODEL_VARIANT_DOES_NOT_FIT", "MODEL_ENGINE_UNAVAILABLE", "INTERNAL_ERROR"
})
DOWNLOAD_STATES = frozenset({"absent", "downloading", "verifying", "loading", "ready", "failed"})
#: Reason a variant's `fits` is `false` (design §3.8, A4).
FITS_REASONS = frozenset({"compute_capability", "vram"})
#: `conversion_status`/`job_summary` `input_kind` (design §3.2, A10).
INPUT_KINDS = frozenset({"pdf", "image"})
#: `InferenceEngine` implementations a worker may run a variant under.
ENGINE_IDS = frozenset({"native"})
#: Why a non-native `engine_id` is unavailable on this worker (design §17.5).
ENGINE_UNAVAILABLE_REASONS = frozenset({"worker_not_published"})
#: `models_response` 1.2 variant `availability` (owner request 2026-09-29):
#: only `published` can be downloaded, loaded or selected; `unpublished` is a
#: known precision whose HF repo is not public yet (DEC-050), `planned` a
#: model that does not exist yet (Unflatten W MoE).
VARIANT_AVAILABILITIES = frozenset({"published", "unpublished", "planned"})

#: DEC-052 design §5.1: one closed field set per named service-local
#: contract, read by `_closed_mapping` below and by `schemas.py`'s Pydantic
#: models (`test_schema_contract_parity.py` checks the two never drift).
_SUBMISSION_RESPONSE_FIELDS = frozenset(
    {
        "schema_version",
        "job_id",
        "state",
        "requested_output",
        "claim_scope",
        "product_quality_claim",
        "status_url",
    }
)
_DIAGNOSTIC_OUTCOME_FIELDS = frozenset(
    {
        "prediction_ir_status",
        "claim_scope",
        "canonical_ast_produced",
        "docx_produced",
        "reason_codes",
        "docx_chain",
    }
)
_CONVERSION_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "job_id",
        "state",
        "stage",
        "attempt",
        "requested_output",
        "claim_scope",
        "product_quality_claim",
        "diagnostic_outcome",
        "artifacts",
        "error",
        # DEC-044 Dodatok D §1: the model variant this job ran (or will
        # run) under; null when no models config is configured.
        "model_variant",
        # DEC-044 Dodatok D §1 (B1): multi-page progress; all null for a
        # job that never needed per-page decode (single-page, unchanged).
        "pages_total",
        "pages_done",
        "current_page",
        "pages",
        # DEC-052 design §3.2/§3.3 (1.1): free-text label, input
        # classification, engine provenance, and lifecycle timestamps.
        "label",
        "input_kind",
        "engine",
        "created_at",
        "updated_at",
        "terminal_at",
        "expires_at",
    }
)
#: `conversion_status` states in which `terminal_at` must be set (design §3.3).
_CONVERSION_STATUS_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "expired"})
#: `conversion_status` states in which `expires_at` may be set (design §3.3).
_CONVERSION_STATUS_EXPIRING_STATES = frozenset({"succeeded", "expired"})
#: `errors.validate_problem_response`'s closed field superset (optional
#: fields are present only for the codes that carry them).
_PROBLEM_FIELDS = frozenset(
    {"type", "code", "title", "status", "retryable", "blockers", "max_pages", "retry_after_seconds"}
)
#: `job_summary` (design §3.4): one `GET /v1/conversions` list row.
_JOB_SUMMARY_FIELDS = frozenset(
    {
        "job_id", "state", "stage", "requested_output", "model_variant", "label",
        "input_kind", "created_at", "updated_at", "terminal_at", "expires_at",
        "pages_total", "pages_done", "docx_produced", "error_code",
    }
)
#: `job_list` (design §3.4): `GET /v1/conversions` response envelope.
_JOB_LIST_FIELDS = frozenset({"schema_version", "items", "next_cursor"})
#: `settings_response` (design §3.10): `GET /v1/settings`.
_SETTINGS_RESPONSE_FIELDS = frozenset(
    {"schema_version", "service_version", "session", "default_variant", "sandbox"}
)
_SETTINGS_SERVICE_VERSION_FIELDS = frozenset({"git_sha", "api_version"})
_SETTINGS_SESSION_FIELDS = frozenset({"idle_seconds", "max_seconds"})
_SETTINGS_DEFAULT_VARIANT_FIELDS = frozenset({"configured", "override", "effective", "source"})
_SETTINGS_DEFAULT_VARIANT_SOURCES = frozenset({"user", "config", "auto"})
_SETTINGS_SANDBOX_FIELDS = frozenset({"rootful"})
#: `artifacts_manifest` (design §3.6): `artifacts.zip`'s `MANIFEST.json`.
_ARTIFACTS_MANIFEST_FIELDS = frozenset(
    {"schema_version", "job_id", "generated_at", "files", "omitted_roles"}
)
_ARTIFACTS_MANIFEST_FILE_FIELDS = frozenset(
    {"path", "role", "page_index", "sha256", "size_bytes", "mime_type"}
)
_ARTIFACTS_MANIFEST_OMITTED_FIELDS = frozenset({"role", "reason"})
_ARTIFACTS_MANIFEST_OMITTED_REASONS = frozenset({"capability_denied"})
#: Mirrors `dataset.artifact_ref._SHA256_RE`.
_SHA256_HEX_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
#: DEC-052 design §2.3: `GET/POST /v1/session` and `POST /v1/setup` share
#: this shape. Every field is a required key; `principal_id`/`capabilities`/
#: `auth_kind`/`expires_at` are nullable rather than optional (§0's "required
#: key, nullable value" pattern, same as `ConversionStatus`'s multi-page fields).
_SESSION_INFO_FIELDS = frozenset(
    {
        "schema_version",
        "authenticated",
        "setup_required",
        "setup_tokenless",
        "principal_id",
        "capabilities",
        "auth_kind",
        "expires_at",
    }
)
_ISO8601_UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
#: `conversion_status` 1.1's `created_at`/`updated_at`/`terminal_at`/`expires_at`
#: (design §3.3): millisecond precision, unlike `session_info.expires_at` above.
_ISO8601_UTC_MS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

_DOCX_CHAIN_REQUIRED = frozenset({"policy_id", "policy_status", "page_outcome"})
_DOCX_CHAIN_OPTIONAL = frozenset(
    {
        "policy_counts", "ast_status", "renderer_status", "failure_codes", "error_class",
        # DEC-044 Dodatok B (2026-09-26): every applied chain policy with its
        # counts, and the AST profile the page was built under (ERRATA-004 E4).
        "policies", "profile",
        # DEC-044 Dodatok C: the renderer approximations a published Word carries
        "renderer_approximations",
        # DEC-044 Dodatok D §1 (B1): the assembler's seam-decision summary,
        # multi-page jobs only.
        "assembler",
    }
)
_DOCX_CHAIN_PROFILES = frozenset({"word-core-v0", "word-visual-v1", "word-visual-v1.1"})
_DOCX_CHAIN_ALL = _DOCX_CHAIN_REQUIRED | _DOCX_CHAIN_OPTIONAL
_FAILURE_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*\|/\S*$")


class ContractInvalid(ValueError):
    """A service-local response violates its closed diagnostic contract."""


def validate_contract(payload: object, contract: str) -> dict[str, Any]:
    """Validate one named closed service contract and return a shallow copy."""
    if contract == "submission_response":
        return _submission_response(payload)
    if contract == "diagnostic_outcome":
        return _diagnostic_outcome(payload)
    if contract == "conversion_status":
        return _conversion_status(payload)
    if contract == "models_response":
        return _models_response(payload)
    if contract == "model_variant_entry":
        return _model_variant_entry(payload)
    if contract == "model_command_accepted":
        return _model_command_accepted(payload)
    if contract == "session_info":
        return _session_info(payload)
    if contract == "job_summary":
        return _job_summary(payload)
    if contract == "job_list":
        return _job_list(payload)
    if contract == "settings_response":
        return _settings_response(payload)
    if contract == "artifacts_manifest":
        return _artifacts_manifest(payload)
    if contract == "problem":
        try:
            return validate_problem_response(payload)
        except ProblemInvalid as exc:
            raise ContractInvalid(str(exc)) from exc
    raise ContractInvalid(f"unknown service-local contract {contract!r}")


def _closed_mapping(payload: object, fields: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ContractInvalid(f"contract fields must be exactly {', '.join(sorted(fields))}")
    if any(not isinstance(key, str) for key in payload):
        raise ContractInvalid("contract keys must be strings")
    return payload


def _submission_response(payload: object) -> dict[str, Any]:
    value = _closed_mapping(payload, _SUBMISSION_RESPONSE_FIELDS)
    job_id = value["job_id"]
    if (
        value["schema_version"] != "1.0"
        or not isinstance(job_id, str)
        or not job_id.startswith("job_")
        or value["state"] != "queued"
        or value["requested_output"] not in ACCEPTED_OUTPUTS
        or value["claim_scope"] != "diagnostic"
        or value["product_quality_claim"] is not False
        or value["status_url"] != f"/v1/conversions/{job_id}"
    ):
        raise ContractInvalid("submission response violates the diagnostic-only contract")
    return dict(value)


def _docx_chain_policies(policies: object) -> None:
    """`docx_chain.policies`: one closed record per applied policy, every
    one RATIFIED (a PROPOSED policy never ships through the service),
    counts a map of non-negative integers (empty when nothing was edited)."""
    if not isinstance(policies, list) or not policies:
        raise ContractInvalid("docx_chain.policies must be a non-empty array")
    for entry in policies:
        if not isinstance(entry, Mapping) or set(entry) != {
            "policy_id", "policy_status", "counts"
        }:
            raise ContractInvalid(
                "docx_chain.policies entries must be closed maps of "
                "policy_id, policy_status, counts"
            )
        if not isinstance(entry["policy_id"], str) or not entry["policy_id"]:
            raise ContractInvalid("docx_chain.policies[].policy_id must be a non-empty string")
        if entry["policy_status"] != "RATIFIED":
            raise ContractInvalid("docx_chain.policies[].policy_status must be RATIFIED")
        counts = entry["counts"]
        if not isinstance(counts, Mapping):
            raise ContractInvalid("docx_chain.policies[].counts must be an object")
        for key, count in counts.items():
            if not isinstance(key, str) or type(count) is not int or count < 0:
                raise ContractInvalid("docx_chain.policies[].counts must map strings to counts")


def _docx_chain_assembler(assembler: object) -> None:
    """`docx_chain.assembler` (DEC-044 Dodatok D §1, B1): the assembler's
    own seam-decision summary, multi-page jobs only."""
    if not isinstance(assembler, Mapping) or set(assembler) != {
        "joins_accepted", "joins_refused", "refusal_codes"
    }:
        raise ContractInvalid(
            "docx_chain.assembler must be a closed map of "
            "joins_accepted/joins_refused/refusal_codes"
        )
    if type(assembler["joins_accepted"]) is not int or assembler["joins_accepted"] < 0:
        raise ContractInvalid("docx_chain.assembler.joins_accepted must be a non-negative integer")
    if type(assembler["joins_refused"]) is not int or assembler["joins_refused"] < 0:
        raise ContractInvalid("docx_chain.assembler.joins_refused must be a non-negative integer")
    codes = assembler["refusal_codes"]
    if not isinstance(codes, Mapping):
        raise ContractInvalid("docx_chain.assembler.refusal_codes must be an object")
    for key, count in codes.items():
        if not isinstance(key, str) or type(count) is not int or count < 0:
            raise ContractInvalid(
                "docx_chain.assembler.refusal_codes must map strings to non-negative counts"
            )


def _docx_chain(payload: object) -> None:
    if payload is None:
        return
    if not isinstance(payload, Mapping):
        raise ContractInvalid("docx_chain must be null or an object")
    keys = set(payload)
    if not _DOCX_CHAIN_REQUIRED <= keys <= _DOCX_CHAIN_ALL:
        raise ContractInvalid(
            f"docx_chain must be a closed map with required keys "
            f"{', '.join(sorted(_DOCX_CHAIN_REQUIRED))}"
        )
    if any(not isinstance(key, str) for key in payload):
        raise ContractInvalid("docx_chain keys must be strings")
    if payload["policy_id"] != "headless-defaults.v0":
        raise ContractInvalid("docx_chain.policy_id must be headless-defaults.v0")
    if payload["policy_status"] != "RATIFIED":
        raise ContractInvalid("docx_chain.policy_status must be RATIFIED")
    if not isinstance(payload["page_outcome"], str) or not payload["page_outcome"]:
        raise ContractInvalid("docx_chain.page_outcome must be a non-empty string")
    for optional_key in ("ast_status", "renderer_status"):
        if optional_key in payload and not isinstance(payload[optional_key], str):
            raise ContractInvalid(f"docx_chain.{optional_key} must be a string when present")
    if "policies" in payload:
        _docx_chain_policies(payload["policies"])
    if "assembler" in payload:
        _docx_chain_assembler(payload["assembler"])
    if "profile" in payload and payload["profile"] is not None:
        if payload["profile"] not in _DOCX_CHAIN_PROFILES:
            raise ContractInvalid("docx_chain.profile must be a known AST profile or null")
    if "policy_counts" in payload:
        counts = payload["policy_counts"]
        if not isinstance(counts, Mapping) or not counts:
            raise ContractInvalid("docx_chain.policy_counts must be a non-empty object")
        for key, count in counts.items():
            if not isinstance(key, str) or type(count) is not int or count < 0:
                raise ContractInvalid(
                    "docx_chain.policy_counts values must be non-negative integers"
                )
    if "failure_codes" in payload:
        codes = payload["failure_codes"]
        if not isinstance(codes, list):
            raise ContractInvalid("docx_chain.failure_codes must be an array")
        for code in codes:
            if not isinstance(code, str) or _FAILURE_CODE_PATTERN.match(code) is None:
                raise ContractInvalid(
                    "docx_chain.failure_codes entries must be '<CODE>|/pointer'"
                )
    if "error_class" in payload and (
        not isinstance(payload["error_class"], str) or not payload["error_class"]
    ):
        raise ContractInvalid("docx_chain.error_class must be a non-empty string when present")


def _diagnostic_outcome(payload: object) -> dict[str, Any]:
    value = _closed_mapping(payload, _DIAGNOSTIC_OUTCOME_FIELDS)
    reasons = value["reason_codes"]
    docx_chain = value["docx_chain"]
    if (
        value["prediction_ir_status"] not in {"sealed", "not_produced"}
        or value["claim_scope"] != "diagnostic"
        or type(value["canonical_ast_produced"]) is not bool
        or type(value["docx_produced"]) is not bool
        or not isinstance(reasons, list)
        or any(reason not in DIAGNOSTIC_REASON_CODES for reason in reasons)
        or len(reasons) != len(set(reasons))
    ):
        raise ContractInvalid("diagnostic outcome violates the non-product boundary")
    _docx_chain(docx_chain)
    renderer_ready = isinstance(docx_chain, Mapping) and (
        docx_chain.get("renderer_status") == "render_ready"
        or (
            docx_chain.get("renderer_status") == "approximation_required"
            and docx_chain.get("renderer_approximations") == ["FONT_FALLBACK"]
            and "DOCX_FONT_SUBSTITUTED" in reasons
        )
    )
    docx_ready = "HEADLESS_DEFAULTS_V0_APPLIED" in reasons and renderer_ready
    if value["docx_produced"] and not docx_ready:
        raise ContractInvalid(
            "docx_produced=true requires HEADLESS_DEFAULTS_V0_APPLIED and a render_ready chain"
        )
    ast_strict = isinstance(docx_chain, Mapping) and docx_chain.get("ast_status") == "strict_valid"
    if value["canonical_ast_produced"] != ast_strict:
        raise ContractInvalid(
            "canonical_ast_produced must equal docx_chain.ast_status == 'strict_valid'"
        )
    return dict(value)


_PAGE_STATES = frozenset({"succeeded", "failed"})


def _page_reports(pages: object) -> None:
    if not isinstance(pages, (list, tuple)) or not pages:
        raise ContractInvalid("pages must be a non-empty array")
    seen_indices: set[int] = set()
    for entry in pages:
        if not isinstance(entry, Mapping) or set(entry) != {
            "index", "state", "error_code", "decode_steps", "seconds"
        }:
            raise ContractInvalid(
                "pages entries must be closed maps of index/state/error_code/"
                "decode_steps/seconds"
            )
        index = entry["index"]
        if type(index) is not int or index < 0 or index in seen_indices:
            raise ContractInvalid("pages[].index must be a unique non-negative integer")
        seen_indices.add(index)
        if entry["state"] not in _PAGE_STATES:
            raise ContractInvalid("pages[].state must be succeeded or failed")
        if entry["error_code"] is not None and not isinstance(entry["error_code"], str):
            raise ContractInvalid("pages[].error_code must be a string or null")
        if (entry["state"] == "failed") != (entry["error_code"] is not None):
            raise ContractInvalid("pages[].error_code must be set iff state is failed")
        steps = entry["decode_steps"]
        if steps is not None and (type(steps) is not int or steps < 0):
            raise ContractInvalid("pages[].decode_steps must be a non-negative integer or null")
        seconds = entry["seconds"]
        if type(seconds) not in (int, float) or seconds < 0:
            raise ContractInvalid("pages[].seconds must be a non-negative number")


def _artifact_entries(artifacts: object, allowed_roles: frozenset[str]) -> None:
    """`artifacts` is either the legacy flat list of role strings (a
    single-page job, byte-identical to before B1) or a list of page-scoped
    `{role, page_index, url}` objects (DEC-044 Dodatok D §1); never mixed."""
    if not isinstance(artifacts, list):
        raise ContractInvalid("artifacts must be an array")
    if not artifacts or isinstance(artifacts[0], str):
        if any(not isinstance(a, str) or a not in allowed_roles for a in artifacts):
            raise ContractInvalid("artifacts role strings must be known accepted outputs")
        if len(artifacts) != len(set(artifacts)):
            raise ContractInvalid("artifacts must not repeat a role")
        return
    seen: set[tuple[str, int | None]] = set()
    for entry in artifacts:
        if not isinstance(entry, Mapping) or set(entry) != {"role", "page_index", "url"}:
            raise ContractInvalid("artifacts entries must be closed maps of role/page_index/url")
        if entry["role"] not in allowed_roles:
            raise ContractInvalid("artifacts[].role must be a known accepted output")
        page_index = entry["page_index"]
        if page_index is not None and (type(page_index) is not int or page_index < 0):
            raise ContractInvalid("artifacts[].page_index must be a non-negative integer or null")
        if not isinstance(entry["url"], str) or not entry["url"].startswith("/"):
            raise ContractInvalid("artifacts[].url must be a same-origin path")
        key = (entry["role"], page_index)
        if key in seen:
            raise ContractInvalid("artifacts must not repeat a (role, page_index) pair")
        seen.add(key)


def _artifact_roles(artifacts: list[Any]) -> set[str]:
    if not artifacts or isinstance(artifacts[0], str):
        return set(artifacts)
    return {entry["role"] for entry in artifacts}


def _conversion_status(payload: object) -> dict[str, Any]:
    value = _closed_mapping(payload, _CONVERSION_STATUS_FIELDS)
    job_id, artifacts = value["job_id"], value["artifacts"]
    requested_output = value["requested_output"]
    allowed_artifact_roles = (
        {"prediction_ir"}
        if requested_output == "prediction_ir"
        else ACCEPTED_OUTPUTS | _PREVIEW_ROLES
    )
    if (
        value["schema_version"] != "1.1"
        or not isinstance(job_id, str)
        or not job_id.startswith("job_")
        or value["state"] not in JOB_STATES
        or value["stage"] not in JOB_STAGES
        or type(value["attempt"]) is not int
        or value["attempt"] < 0
        or requested_output not in ACCEPTED_OUTPUTS
        or value["claim_scope"] != "diagnostic"
        or value["product_quality_claim"] is not False
        or (value["model_variant"] is not None and not isinstance(value["model_variant"], str))
        or (value["label"] is not None and not isinstance(value["label"], str))
        or (value["input_kind"] is not None and value["input_kind"] not in INPUT_KINDS)
        or (value["engine"] is not None and value["engine"] not in ENGINE_IDS)
    ):
        raise ContractInvalid("conversion status violates the diagnostic-only contract")
    _conversion_status_timestamps(value)
    # T42 (design §3.3): `engine` (∈ ENGINE_IDS, checked above) is set iff
    # `model_variant` is set -- `create_job` now takes it from the resolved
    # variant's own `engine` field, so it is no longer always `"native"`.
    if (value["model_variant"] is not None) != (value["engine"] is not None):
        raise ContractInvalid("conversion status engine must be set iff model_variant is set")
    _artifact_entries(artifacts, allowed_artifact_roles)
    _multi_page_fields(value)
    diagnostic_outcome = value["diagnostic_outcome"]
    if diagnostic_outcome is not None:
        validated = _diagnostic_outcome(diagnostic_outcome)
        if validated["docx_produced"] != ("diagnostic_docx" in _artifact_roles(artifacts)):
            raise ContractInvalid(
                "docx_produced must equal whether diagnostic_docx is in artifacts"
            )
    if value["error"] is not None:
        try:
            validate_problem_response(value["error"])
        except ValueError as exc:
            raise ContractInvalid("conversion status contains an invalid problem") from exc
    return dict(value)


def _conversion_status_timestamps(value: Mapping[str, Any]) -> None:
    """DEC-052 design §3.3: `created_at`/`updated_at` are always set, ISO-8601
    UTC with millisecond precision; `terminal_at` is set iff the job reached a
    terminal state; `expires_at` may be set only once artifacts can expire."""
    created_at, updated_at = value["created_at"], value["updated_at"]
    terminal_at, expires_at = value["terminal_at"], value["expires_at"]
    if (
        not isinstance(created_at, str)
        or _ISO8601_UTC_MS_PATTERN.match(created_at) is None
        or not isinstance(updated_at, str)
        or _ISO8601_UTC_MS_PATTERN.match(updated_at) is None
    ):
        raise ContractInvalid("conversion status created_at/updated_at must be ISO-8601 UTC ms")
    if updated_at < created_at:
        raise ContractInvalid("conversion status updated_at must not precede created_at")
    if terminal_at is not None and (
        not isinstance(terminal_at, str) or _ISO8601_UTC_MS_PATTERN.match(terminal_at) is None
    ):
        raise ContractInvalid("conversion status terminal_at must be ISO-8601 UTC ms or null")
    if (terminal_at is not None) != (value["state"] in _CONVERSION_STATUS_TERMINAL_STATES):
        raise ContractInvalid("conversion status terminal_at must be set iff state is terminal")
    if expires_at is not None:
        if not isinstance(expires_at, str) or _ISO8601_UTC_MS_PATTERN.match(expires_at) is None:
            raise ContractInvalid("conversion status expires_at must be ISO-8601 UTC ms or null")
        if value["state"] not in _CONVERSION_STATUS_EXPIRING_STATES:
            raise ContractInvalid(
                "conversion status expires_at may be set only for succeeded or expired jobs"
            )


def _multi_page_fields(value: Mapping[str, Any]) -> None:
    pages_total, pages_done, current_page, pages = (
        value["pages_total"], value["pages_done"], value["current_page"], value["pages"],
    )
    all_none = pages_total is None and pages_done is None and current_page is None and pages is None
    if all_none:
        return
    if (
        type(pages_total) is not int or pages_total < 1
        or type(pages_done) is not int or not (0 <= pages_done <= pages_total)
        or type(current_page) is not int or not (0 <= current_page < pages_total)
    ):
        raise ContractInvalid("multi-page progress counts are inconsistent")
    if isinstance(pages, (list, tuple)) and not pages and pages_done == 0:
        # A running job before its first page finishes has no page report yet.
        return
    _page_reports(pages)


_MODELS_RESPONSE_FIELDS = frozenset(
    {
        "schema_version",
        "default_variant",
        "default_variant_source",
        "resident_variant",
        "device",
        "variants",
    }
)
_MODELS_RESPONSE_DEVICE_FIELDS = frozenset(
    {"total_vram_mib", "free_vram_mib", "compute_capability", "observed_at"}
)
_MODEL_VARIANT_ENTRY_FIELDS = frozenset(
    {
        "variant_id",
        "label",
        "precision",
        "kv_cache",
        "bundle_manifest_sha256",
        "estimated_peak_vram_mib",
        "fits",
        "resident",
        "source_kind",
        "downloaded",
        "download_size_bytes",
        "is_default",
        "fits_reason",
        "engine",
        "engine_available",
        "engine_unavailable_reason",
        "download",
        "availability",
    }
)
#: `models_response.variants[].download` (design §3.8), `null` for a local
#: `bundle` variant (`source_kind == "local"`) and for every variant that is
#: not `published` (1.2: nothing to download yet).
_MODEL_VARIANT_DOWNLOAD_FIELDS = frozenset(
    {"state", "bytes_done", "bytes_total", "error_code", "command_id"}
)
#: `ModelCommandAccepted` (design §3.9): the `POST/DELETE /v1/models/{variant_id}/*`
#: response shape.
_MODEL_COMMAND_ACCEPTED_FIELDS = frozenset(
    {
        "schema_version",
        "command_id",
        "kind",
        "variant_id",
        "state",
        "phase",
        "bytes_done",
        "bytes_total",
        "error_code",
        "requested_at",
        "started_at",
        "finished_at",
    }
)


def _model_variant_download(payload: object) -> dict[str, Any]:
    value = _closed_mapping(payload, _MODEL_VARIANT_DOWNLOAD_FIELDS)
    if (
        value["state"] not in DOWNLOAD_STATES
        or (
            value["bytes_done"] is not None
            and (type(value["bytes_done"]) is not int or value["bytes_done"] < 0)
        )
        or (
            value["bytes_total"] is not None
            and (type(value["bytes_total"]) is not int or value["bytes_total"] < 0)
        )
        or (value["error_code"] is not None and not isinstance(value["error_code"], str))
        or (value["command_id"] is not None and not isinstance(value["command_id"], str))
    ):
        raise ContractInvalid("models response variant download field is invalid")
    return dict(value)


def _model_command_accepted(payload: object) -> dict[str, Any]:
    """Mirrors `schemas.ModelCommandAccepted` (design §3.9): the
    `POST/DELETE /v1/models/{variant_id}/*` response body."""
    value = _closed_mapping(payload, _MODEL_COMMAND_ACCEPTED_FIELDS)
    phase = value["phase"]
    if (
        value["schema_version"] != "1.0"
        or not isinstance(value["command_id"], str)
        or not value["command_id"]
        or value["kind"] not in MODEL_COMMAND_KINDS
        or not isinstance(value["variant_id"], str)
        or not value["variant_id"]
        or value["state"] not in MODEL_COMMAND_STATES
        or (phase is not None and phase not in MODEL_COMMAND_PHASES)
        or (
            value["bytes_done"] is not None
            and (type(value["bytes_done"]) is not int or value["bytes_done"] < 0)
        )
        or (
            value["bytes_total"] is not None
            and (type(value["bytes_total"]) is not int or value["bytes_total"] < 0)
        )
        or (value["error_code"] is not None and not isinstance(value["error_code"], str))
        or not isinstance(value["requested_at"], str)
        or _ISO8601_UTC_MS_PATTERN.match(value["requested_at"]) is None
        or (
            value["started_at"] is not None
            and (
                not isinstance(value["started_at"], str)
                or _ISO8601_UTC_MS_PATTERN.match(value["started_at"]) is None
            )
        )
        or (
            value["finished_at"] is not None
            and (
                not isinstance(value["finished_at"], str)
                or _ISO8601_UTC_MS_PATTERN.match(value["finished_at"]) is None
            )
        )
    ):
        raise ContractInvalid("model command accepted violates its closed contract")
    # `finish_model_command` requires `error_code` iff the command failed
    # (`job_store_aux.py`); mirrored here so a handler can never publish a
    # `failed` command without a reason, or a reason on a live one.
    if (value["error_code"] is not None) != (value["state"] == "failed"):
        raise ContractInvalid(
            "model command accepted error_code must be set iff state is failed"
        )
    return dict(value)


def _models_response_device(payload: object) -> dict[str, Any]:
    value = _closed_mapping(payload, _MODELS_RESPONSE_DEVICE_FIELDS)
    compute_capability = value["compute_capability"]
    if (
        type(value["total_vram_mib"]) not in (int, float)
        or value["total_vram_mib"] < 0
        or type(value["free_vram_mib"]) not in (int, float)
        or value["free_vram_mib"] < 0
        or not isinstance(compute_capability, (list, tuple))
        or len(compute_capability) != 2
        or any(type(part) is not int for part in compute_capability)
        or _ISO8601_UTC_PATTERN.match(value["observed_at"]) is None
    ):
        raise ContractInvalid("models response device field is invalid")
    return dict(value)


def _model_variant_entry(payload: object) -> dict[str, Any]:
    """One `models_response.variants[]` entry; also a standalone contract
    (`validate_contract(entry, "model_variant_entry")`)."""
    value = _closed_mapping(payload, _MODEL_VARIANT_ENTRY_FIELDS)
    availability = value["availability"]
    planned = availability == "planned"
    if (
        not isinstance(value["variant_id"], str)
        or not value["variant_id"]
        or not isinstance(value["label"], str)
        or not value["label"]
        or availability not in VARIANT_AVAILABILITIES
        or value["precision"] not in ({None} if planned else {"bf16", "int8", "fp8", "int4"})
        or value["kv_cache"] not in ({None} if planned else {"bf16", "fp8"})
        or (
            value["bundle_manifest_sha256"] is not None
            and not isinstance(value["bundle_manifest_sha256"], str)
        )
        or (
            value["estimated_peak_vram_mib"] is not None
            and type(value["estimated_peak_vram_mib"]) not in (int, float)
        )
        or value["fits"] not in (True, False, None)
        or type(value["resident"]) is not bool
        or value["source_kind"] not in ({None} if planned else {"hf", "local"})
        or type(value["downloaded"]) is not bool
        or (
            value["download_size_bytes"] is not None
            and (type(value["download_size_bytes"]) is not int or value["download_size_bytes"] < 0)
        )
        or type(value["is_default"]) is not bool
        or value["fits_reason"] not in FITS_REASONS | {None}
        or value["engine"] not in ENGINE_IDS
        or type(value["engine_available"]) is not bool
        or value["engine_unavailable_reason"] not in ENGINE_UNAVAILABLE_REASONS | {None}
    ):
        raise ContractInvalid("models response variant entry field is invalid")
    if (value["fits_reason"] is not None) != (value["fits"] is False):
        raise ContractInvalid("models response variant fits_reason must be set iff fits is false")
    if value["engine"] == "native" and (
        not value["engine_available"] or value["engine_unavailable_reason"] is not None
    ):
        raise ContractInvalid("models response native engine is always available")
    if (value["engine_unavailable_reason"] is not None) == value["engine_available"]:
        raise ContractInvalid(
            "models response engine_unavailable_reason must be set iff engine_available is false"
        )
    if availability != "published":
        # Nothing exists to download, load or select yet: no origin, no
        # local copy, never resident or default, and a planned model has no
        # measured tier to estimate from.
        if (
            value["source_kind"] == "local"
            or value["download"] is not None
            or value["downloaded"]
            or value["download_size_bytes"] is not None
            or value["bundle_manifest_sha256"] is not None
            or value["resident"]
            or value["is_default"]
            or (
                planned
                and (value["estimated_peak_vram_mib"] is not None or value["fits"] is not None)
            )
        ):
            raise ContractInvalid(
                "models response a variant that is not published has nothing downloaded, "
                "resident or default"
            )
        return dict(value)
    download = value["download"]
    if (download is None) != (value["source_kind"] == "local"):
        raise ContractInvalid("models response download is null exactly for a local bundle")
    if download is None:
        if not value["downloaded"]:
            raise ContractInvalid("models response a local bundle variant is always downloaded")
    else:
        validated_download = _model_variant_download(download)
        ready = validated_download["state"] == "ready"
        cached_load_failure = (
            validated_download["state"] == "failed"
            and validated_download["error_code"] in CACHED_LOAD_FAILURE_CODES
        )
        if (ready and not value["downloaded"]) or (
            value["downloaded"] and not ready and not cached_load_failure
        ):
            raise ContractInvalid("models response downloaded conflicts with transfer state")
    return dict(value)


def _models_response(payload: object) -> dict[str, Any]:
    """DEC-044 Dodatok D §1: `GET /v1/models`. `default_variant` and
    `resident_variant` are null exactly when no models config is configured
    (today's single-bundle deployments) or no worker has published state.
    1.2 (2026-09-29) adds the per-variant `availability`."""
    value = _closed_mapping(payload, _MODELS_RESPONSE_FIELDS)
    variants = value["variants"]
    if (
        value["schema_version"] != "1.2"
        or (value["default_variant"] is not None and not isinstance(value["default_variant"], str))
        or (
            value["default_variant_source"] is not None
            and value["default_variant_source"] not in _SETTINGS_DEFAULT_VARIANT_SOURCES
        )
        or (value["default_variant_source"] is None) != (value["default_variant"] is None)
        or (
            value["resident_variant"] is not None
            and not isinstance(value["resident_variant"], str)
        )
        or not isinstance(variants, list)
    ):
        raise ContractInvalid("models response violates its closed contract")
    if value["device"] is not None:
        _models_response_device(value["device"])
    ids = []
    for entry in variants:
        validated_entry = _model_variant_entry(entry)
        ids.append(validated_entry["variant_id"])
        expected_default = validated_entry["variant_id"] == value["default_variant"]
        if validated_entry["is_default"] != expected_default:
            raise ContractInvalid("models response is_default must match default_variant")
    if len(ids) != len(set(ids)):
        raise ContractInvalid("models response variant_id values must be unique")
    if value["default_variant"] is not None and value["default_variant"] not in ids:
        raise ContractInvalid("models response default_variant names no listed variant")
    if value["resident_variant"] is not None and value["resident_variant"] not in ids:
        raise ContractInvalid("models response resident_variant names no listed variant")
    if sum(1 for entry in variants if entry["resident"]) > 1:
        raise ContractInvalid("models response may name at most one resident variant")
    return dict(value)


def _session_info(payload: object) -> dict[str, Any]:
    """DEC-052 design §2.3: `GET/POST /v1/session`, `POST /v1/setup`."""
    value = _closed_mapping(payload, _SESSION_INFO_FIELDS)
    authenticated = value["authenticated"]
    principal_id = value["principal_id"]
    capabilities = value["capabilities"]
    auth_kind = value["auth_kind"]
    expires_at = value["expires_at"]
    if (
        value["schema_version"] != "1.1"
        or type(authenticated) is not bool
        or type(value["setup_required"]) is not bool
        or type(value["setup_tokenless"]) is not bool
    ):
        raise ContractInvalid("session info violates its closed contract")
    if value["setup_tokenless"] and not value["setup_required"]:
        raise ContractInvalid("session info setup_tokenless requires setup_required")
    if authenticated:
        if not isinstance(principal_id, str) or not principal_id:
            raise ContractInvalid("session info principal_id must be set when authenticated")
        if auth_kind not in ("bearer", "session"):
            raise ContractInvalid("session info auth_kind must be bearer or session")
        if (
            not isinstance(capabilities, list)
            or any(not isinstance(c, str) or c not in CAPABILITIES for c in capabilities)
            or len(capabilities) != len(set(capabilities))
            or capabilities != sorted(capabilities)
        ):
            raise ContractInvalid("session info capabilities must be a sorted known subset")
    elif principal_id is not None or auth_kind is not None or capabilities is not None:
        raise ContractInvalid("session info must be all-null when unauthenticated")
    if auth_kind == "session":
        if not isinstance(expires_at, str) or _ISO8601_UTC_PATTERN.match(expires_at) is None:
            raise ContractInvalid("session info expires_at must be ISO-8601 UTC for a session")
    elif expires_at is not None:
        raise ContractInvalid("session info expires_at is valid only for auth_kind session")
    return dict(value)


def _job_summary(payload: object) -> dict[str, Any]:
    """Mirrors `schemas.JobSummary` (design §3.4): one `GET /v1/conversions`
    list row. Reuses `_conversion_status_timestamps` -- the same
    created_at/updated_at/terminal_at/expires_at invariants apply to the same
    `state` enum."""
    value = _closed_mapping(payload, _JOB_SUMMARY_FIELDS)
    job_id = value["job_id"]
    pages_total, pages_done = value["pages_total"], value["pages_done"]
    if (
        not isinstance(job_id, str)
        or not job_id.startswith("job_")
        or value["state"] not in JOB_STATES
        or value["stage"] not in JOB_STAGES
        or value["requested_output"] not in ACCEPTED_OUTPUTS
        or (value["model_variant"] is not None and not isinstance(value["model_variant"], str))
        or (value["label"] is not None and not isinstance(value["label"], str))
        or (value["input_kind"] is not None and value["input_kind"] not in INPUT_KINDS)
        or (pages_total is not None and (type(pages_total) is not int or pages_total < 1))
        or (pages_done is not None and (type(pages_done) is not int or pages_done < 0))
        or (pages_total is None) != (pages_done is None)
        or (
            pages_total is not None and pages_done is not None and pages_done > pages_total
        )
        or (value["docx_produced"] is not None and type(value["docx_produced"]) is not bool)
        or (value["error_code"] is not None and not isinstance(value["error_code"], str))
    ):
        raise ContractInvalid("job summary violates its closed contract")
    _conversion_status_timestamps(value)
    return dict(value)


def _job_list(payload: object) -> dict[str, Any]:
    """Mirrors `schemas.JobList` (design §3.4): the `GET /v1/conversions`
    envelope, cursor-paginated and principal-scoped."""
    value = _closed_mapping(payload, _JOB_LIST_FIELDS)
    items = value["items"]
    if value["schema_version"] != "1.0" or not isinstance(items, list):
        raise ContractInvalid("job list violates its closed contract")
    for entry in items:
        _job_summary(entry)
    next_cursor = value["next_cursor"]
    if next_cursor is not None and not isinstance(next_cursor, str):
        raise ContractInvalid("job list next_cursor must be a string or null")
    return dict(value)


def _settings_response(payload: object) -> dict[str, Any]:
    """Mirrors `schemas.SettingsResponse` (design §3.10): `GET /v1/settings`."""
    value = _closed_mapping(payload, _SETTINGS_RESPONSE_FIELDS)
    if value["schema_version"] != "1.0":
        raise ContractInvalid("settings response violates its closed contract")
    service_version = _closed_mapping(value["service_version"], _SETTINGS_SERVICE_VERSION_FIELDS)
    if (
        not isinstance(service_version["git_sha"], str)
        or not service_version["git_sha"]
        or service_version["api_version"] != "1.1"
    ):
        raise ContractInvalid("settings response service_version is invalid")
    session = _closed_mapping(value["session"], _SETTINGS_SESSION_FIELDS)
    idle_seconds, max_seconds = session["idle_seconds"], session["max_seconds"]
    if (
        type(idle_seconds) is not int
        or idle_seconds <= 0
        or type(max_seconds) is not int
        or max_seconds < idle_seconds
    ):
        raise ContractInvalid("settings response session is invalid")
    default_variant = _closed_mapping(
        value["default_variant"], _SETTINGS_DEFAULT_VARIANT_FIELDS
    )
    configured, override, effective, source = (
        default_variant["configured"],
        default_variant["override"],
        default_variant["effective"],
        default_variant["source"],
    )
    if (
        not isinstance(configured, str)
        or not configured
        or (override is not None and not isinstance(override, str))
        or (effective is not None and not isinstance(effective, str))
        or (source is not None and source not in _SETTINGS_DEFAULT_VARIANT_SOURCES)
        or (effective is None) != (source is None)
        or (source == "user" and effective != override)
    ):
        raise ContractInvalid("settings response default_variant is invalid")
    sandbox = _closed_mapping(value["sandbox"], _SETTINGS_SANDBOX_FIELDS)
    if sandbox["rootful"] is not None and type(sandbox["rootful"]) is not bool:
        raise ContractInvalid("settings response sandbox.rootful must be a boolean or null")
    return dict(value)


def _artifacts_manifest_file(payload: object) -> Mapping[str, Any]:
    value = _closed_mapping(payload, _ARTIFACTS_MANIFEST_FILE_FIELDS)
    page_index = value["page_index"]
    if (
        not isinstance(value["path"], str)
        or not value["path"]
        or value["role"] not in ARTIFACT_ROLES
        or type(page_index) is not int
        or page_index < -1
        or not isinstance(value["sha256"], str)
        or _SHA256_HEX_PATTERN.match(value["sha256"]) is None
        or type(value["size_bytes"]) is not int
        or value["size_bytes"] < 0
        or not isinstance(value["mime_type"], str)
        or not value["mime_type"]
    ):
        raise ContractInvalid("artifacts manifest file violates its closed contract")
    return value


def _artifacts_manifest_omitted_role(payload: object) -> Mapping[str, Any]:
    value = _closed_mapping(payload, _ARTIFACTS_MANIFEST_OMITTED_FIELDS)
    if (
        value["role"] not in ARTIFACT_ROLES
        or value["reason"] not in _ARTIFACTS_MANIFEST_OMITTED_REASONS
    ):
        raise ContractInvalid("artifacts manifest omitted role violates its closed contract")
    return value


def _artifacts_manifest(payload: object) -> dict[str, Any]:
    """Mirrors `schemas.ArtifactsManifest` (design §3.6): `artifacts.zip`'s
    `MANIFEST.json`."""
    value = _closed_mapping(payload, _ARTIFACTS_MANIFEST_FIELDS)
    job_id, files, omitted_roles = value["job_id"], value["files"], value["omitted_roles"]
    if (
        value["schema_version"] != "1.0"
        or not isinstance(job_id, str)
        or not job_id.startswith("job_")
        or not isinstance(value["generated_at"], str)
        or _ISO8601_UTC_MS_PATTERN.match(value["generated_at"]) is None
        or not isinstance(files, list)
        or not isinstance(omitted_roles, list)
    ):
        raise ContractInvalid("artifacts manifest violates its closed contract")
    seen_paths: set[str] = set()
    for entry in files:
        validated_file = _artifacts_manifest_file(entry)
        if validated_file["path"] in seen_paths:
            raise ContractInvalid("artifacts manifest files must not repeat a path")
        seen_paths.add(validated_file["path"])
    seen_roles: set[str] = set()
    for entry in omitted_roles:
        validated_role = _artifacts_manifest_omitted_role(entry)
        if validated_role["role"] in seen_roles:
            raise ContractInvalid("artifacts manifest omitted_roles must not repeat a role")
        seen_roles.add(validated_role["role"])
    return dict(value)


#: DEC-052 design §5.1: the closed field set of every named service-local
#: contract this module validates today (T15 A3: "existujúce 1.0 kontrakty" +
#: `problem`). `schemas.py`'s Pydantic models are checked against this by
#: `test_schema_contract_parity.py::test_model_fields_equal_contract_fields`.
CONTRACT_FIELDS: dict[str, frozenset[str]] = {
    "submission_response": _SUBMISSION_RESPONSE_FIELDS,
    "diagnostic_outcome": _DIAGNOSTIC_OUTCOME_FIELDS,
    "conversion_status": _CONVERSION_STATUS_FIELDS,
    "models_response": _MODELS_RESPONSE_FIELDS,
    "model_variant_entry": _MODEL_VARIANT_ENTRY_FIELDS,
    "model_command_accepted": _MODEL_COMMAND_ACCEPTED_FIELDS,
    "session_info": _SESSION_INFO_FIELDS,
    "job_summary": _JOB_SUMMARY_FIELDS,
    "job_list": _JOB_LIST_FIELDS,
    "settings_response": _SETTINGS_RESPONSE_FIELDS,
    "artifacts_manifest": _ARTIFACTS_MANIFEST_FIELDS,
    "problem": _PROBLEM_FIELDS,
}

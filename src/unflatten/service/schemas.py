"""OpenAPI-only Pydantic models for the closed service-local contracts in
`contracts.py` (DEC-052 design §5.1, T15 API A3).

`contracts.validate_contract` (and, for `problem`, `errors.validate_problem_response`)
remain the single behavioural gate every handler calls -- these models exist only so
FastAPI can generate a typed OpenAPI document. They independently duplicate the closed
field sets and cross-field invariants of the contracts they mirror (never delegate to
`validate_contract`): `test_schema_contract_parity.py` runs the same corpus through
both, so a `@model_validator` missing here turns that test red instead of hiding a
silent divergence.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from unflatten.service.auth import CAPABILITIES
from unflatten.service.contracts import (
    ACCEPTED_OUTPUTS,
    ARTIFACT_ROLES,
    CACHED_LOAD_FAILURE_CODES,
    DIAGNOSTIC_REASON_CODES,
    DOWNLOAD_STATES,
    ENGINE_IDS,
    ENGINE_UNAVAILABLE_REASONS,
    FITS_REASONS,
    INPUT_KINDS,
    JOB_STAGES,
    JOB_STATES,
    MODEL_COMMAND_KINDS,
    MODEL_COMMAND_PHASES,
    MODEL_COMMAND_STATES,
    VARIANT_AVAILABILITIES,
)
from unflatten.service.errors import PROBLEM_CODES
from unflatten.service.readiness import DOCX_BLOCKERS

#: Mirrors `contracts._FAILURE_CODE_PATTERN` (kept private there; duplicated
#: here on purpose, see module docstring).
_FAILURE_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*\|/\S*$")
#: Mirrors `contracts._ISO8601_UTC_PATTERN`.
_ISO8601_UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
#: Mirrors `contracts._ISO8601_UTC_MS_PATTERN`.
_ISO8601_UTC_MS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
#: Mirrors `contracts._CONVERSION_STATUS_TERMINAL_STATES`.
_CONVERSION_STATUS_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "expired"})
#: Mirrors `contracts._CONVERSION_STATUS_EXPIRING_STATES`.
_CONVERSION_STATUS_EXPIRING_STATES = frozenset({"succeeded", "expired"})
#: Mirrors `contracts._DOCX_CHAIN_PROFILES`.
_DOCX_CHAIN_PROFILES = ("word-core-v0", "word-visual-v1", "word-visual-v1.1")
_JOB_ID_PATTERN = re.compile(r"^job_.+$")


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Problem(_Closed):
    """Mirrors `errors.validate_problem_response`'s closed, content-free shape."""

    #: Derived from `code` when omitted (`errors.validate_problem_response`
    #: never requires it on input, only cross-checks it when present).
    type: str | None = None
    code: Literal[*sorted(PROBLEM_CODES)]
    title: str = Field(min_length=1, max_length=200)
    status: int = Field(ge=400, le=599)
    retryable: bool
    blockers: list[Literal[*DOCX_BLOCKERS]] | None = None
    max_pages: int | None = Field(default=None, ge=1)
    retry_after_seconds: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> Problem:
        expected_type = f"urn:unflatten:error:{self.code.lower().replace('_', '-')}"
        if self.type is not None and self.type != expected_type:
            raise ValueError("problem type must be derived from its stable code")
        if self.blockers is not None and self.code != "CAPABILITY_NOT_READY":
            raise ValueError("blockers are valid only for capability readiness problems")
        if self.blockers is not None and len(self.blockers) != len(set(self.blockers)):
            raise ValueError("blockers must not repeat a code")
        if self.max_pages is not None and self.code != "TOO_MANY_PAGES":
            raise ValueError("max_pages is valid only for a TOO_MANY_PAGES problem")
        if self.retry_after_seconds is not None and self.code != "RATE_LIMITED":
            raise ValueError("retry_after_seconds is valid only for a RATE_LIMITED problem")
        return self


class SubmissionResponse(_Closed):
    """Mirrors `contracts._submission_response`."""

    schema_version: Literal["1.0"]
    job_id: str = Field(pattern=_JOB_ID_PATTERN)
    state: Literal["queued"]
    requested_output: Literal[*sorted(ACCEPTED_OUTPUTS)]
    claim_scope: Literal["diagnostic"]
    product_quality_claim: Literal[False]
    status_url: str

    @model_validator(mode="after")
    def _status_url_matches_job_id(self) -> SubmissionResponse:
        if self.status_url != f"/v1/conversions/{self.job_id}":
            raise ValueError("status_url must be derived from job_id")
        return self


class DocxChainPolicy(_Closed):
    policy_id: str = Field(min_length=1)
    policy_status: Literal["RATIFIED"]
    counts: dict[str, int]

    @field_validator("counts")
    @classmethod
    def _counts_non_negative(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("counts must map to non-negative integers")
        return value


class DocxChainAssembler(_Closed):
    joins_accepted: int = Field(ge=0)
    joins_refused: int = Field(ge=0)
    refusal_codes: dict[str, int]

    @field_validator("refusal_codes")
    @classmethod
    def _refusal_codes_non_negative(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("refusal_codes must map to non-negative integers")
        return value


class DocxChain(_Closed):
    """Mirrors `contracts._docx_chain`."""

    policy_id: Literal["headless-defaults.v0"]
    policy_status: Literal["RATIFIED"]
    page_outcome: str = Field(min_length=1)
    ast_status: str | None = None
    renderer_status: str | None = None
    renderer_approximations: list[str] | None = None
    policies: list[DocxChainPolicy] | None = None
    profile: Literal[*_DOCX_CHAIN_PROFILES] | None = None
    policy_counts: dict[str, int] | None = None
    failure_codes: list[str] | None = None
    error_class: str | None = Field(default=None, min_length=1)

    @field_validator("policies")
    @classmethod
    def _policies_non_empty(
        cls, value: list[DocxChainPolicy] | None
    ) -> list[DocxChainPolicy] | None:
        if value is not None and not value:
            raise ValueError("policies must be a non-empty array when present")
        return value

    @field_validator("policy_counts")
    @classmethod
    def _policy_counts_non_empty_non_negative(
        cls, value: dict[str, int] | None
    ) -> dict[str, int] | None:
        if value is not None:
            if not value:
                raise ValueError("policy_counts must be a non-empty object when present")
            if any(count < 0 for count in value.values()):
                raise ValueError("policy_counts values must be non-negative integers")
        return value

    @field_validator("failure_codes")
    @classmethod
    def _failure_codes_pointer_shaped(cls, value: list[str] | None) -> list[str] | None:
        if value is not None:
            for code in value:
                if _FAILURE_CODE_PATTERN.match(code) is None:
                    raise ValueError("failure_codes entries must be '<CODE>|/pointer'")
        return value


class DiagnosticOutcome(_Closed):
    """Mirrors `contracts._diagnostic_outcome`."""

    prediction_ir_status: Literal["sealed", "not_produced"]
    claim_scope: Literal["diagnostic"]
    canonical_ast_produced: bool
    docx_produced: bool
    reason_codes: list[Literal[*sorted(DIAGNOSTIC_REASON_CODES)]]
    docx_chain: DocxChain | None

    @field_validator("reason_codes")
    @classmethod
    def _reason_codes_no_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("reason_codes must not repeat a code")
        return value

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> DiagnosticOutcome:
        chain = self.docx_chain
        renderer_ready = chain is not None and (
            chain.renderer_status == "render_ready"
            or (
                chain.renderer_status == "approximation_required"
                and chain.renderer_approximations == ["FONT_FALLBACK"]
                and "DOCX_FONT_SUBSTITUTED" in self.reason_codes
            )
        )
        docx_ready = "HEADLESS_DEFAULTS_V0_APPLIED" in self.reason_codes and renderer_ready
        if self.docx_produced and not docx_ready:
            raise ValueError(
                "docx_produced=true requires HEADLESS_DEFAULTS_V0_APPLIED and a "
                "render_ready chain"
            )
        ast_strict = chain is not None and chain.ast_status == "strict_valid"
        if self.canonical_ast_produced != ast_strict:
            raise ValueError(
                "canonical_ast_produced must equal docx_chain.ast_status == 'strict_valid'"
            )
        return self


class ArtifactEntry(_Closed):
    role: Literal[*sorted(ARTIFACT_ROLES)]
    #: Required key, nullable value (`_artifact_entries`'s exact key set).
    page_index: int | None = Field(ge=0)
    url: str = Field(pattern=r"^/")


class PageReport(_Closed):
    index: int = Field(ge=0)
    state: Literal["succeeded", "failed"]
    #: Required keys, nullable values (`_page_reports`'s exact key set).
    error_code: str | None
    decode_steps: int | None = Field(ge=0)
    seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _error_code_iff_failed(self) -> PageReport:
        if (self.state == "failed") != (self.error_code is not None):
            raise ValueError("error_code must be set iff state is failed")
        return self


class ConversionStatus(_Closed):
    """Mirrors `contracts._conversion_status`."""

    schema_version: Literal["1.1"]
    job_id: str = Field(pattern=_JOB_ID_PATTERN)
    state: Literal[*sorted(JOB_STATES)]
    stage: Literal[*sorted(JOB_STAGES)]
    attempt: int = Field(ge=0)
    requested_output: Literal[*sorted(ACCEPTED_OUTPUTS)]
    claim_scope: Literal["diagnostic"]
    product_quality_claim: Literal[False]
    diagnostic_outcome: DiagnosticOutcome | None
    artifacts: list[str] | list[ArtifactEntry]
    error: Problem | None
    model_variant: str | None
    #: `_closed_mapping`-validated: null is a valid value, but every field is
    #: still a required key (unlike `docx_chain`'s genuinely optional keys).
    pages_total: int | None = Field(ge=1)
    pages_done: int | None = Field(ge=0)
    current_page: int | None = Field(ge=0)
    pages: list[PageReport] | None
    #: DEC-052 design §3.2/§3.3 (1.1): free-text label, input classification,
    #: engine provenance, and lifecycle timestamps.
    label: str | None
    input_kind: Literal[*sorted(INPUT_KINDS)] | None
    engine: Literal[*sorted(ENGINE_IDS)] | None
    created_at: str = Field(pattern=_ISO8601_UTC_MS_PATTERN)
    updated_at: str = Field(pattern=_ISO8601_UTC_MS_PATTERN)
    terminal_at: str | None = Field(pattern=_ISO8601_UTC_MS_PATTERN)
    expires_at: str | None = Field(pattern=_ISO8601_UTC_MS_PATTERN)

    @staticmethod
    def _artifact_roles(artifacts: list[str] | list[ArtifactEntry]) -> set[str]:
        if not artifacts or isinstance(artifacts[0], str):
            return set(artifacts)  # type: ignore[arg-type]
        return {entry.role for entry in artifacts}  # type: ignore[union-attr]

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> ConversionStatus:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        if (self.terminal_at is not None) != (self.state in _CONVERSION_STATUS_TERMINAL_STATES):
            raise ValueError("terminal_at must be set iff state is terminal")
        if self.expires_at is not None and self.state not in _CONVERSION_STATUS_EXPIRING_STATES:
            raise ValueError("expires_at may be set only for succeeded or expired jobs")
        if (self.model_variant is not None) != (self.engine is not None):
            raise ValueError("engine must be set iff model_variant is set")
        allowed_roles = (
            {"prediction_ir"}
            if self.requested_output == "prediction_ir"
            else set(ARTIFACT_ROLES)
        )
        if not self.artifacts or isinstance(self.artifacts[0], str):
            roles = list(self.artifacts)
            if any(role not in allowed_roles for role in roles):
                raise ValueError("artifacts role strings must be known accepted outputs")
            if len(roles) != len(set(roles)):
                raise ValueError("artifacts must not repeat a role")
        else:
            seen: set[tuple[str, int | None]] = set()
            for entry in self.artifacts:
                assert isinstance(entry, ArtifactEntry)
                if entry.role not in allowed_roles:
                    raise ValueError("artifacts[].role must be a known accepted output")
                key = (entry.role, entry.page_index)
                if key in seen:
                    raise ValueError("artifacts must not repeat a (role, page_index) pair")
                seen.add(key)
        if self.diagnostic_outcome is not None:
            docx_produced_matches = self.diagnostic_outcome.docx_produced == (
                "diagnostic_docx" in self._artifact_roles(self.artifacts)
            )
            if not docx_produced_matches:
                raise ValueError(
                    "docx_produced must equal whether diagnostic_docx is in artifacts"
                )
        all_none = (
            self.pages_total is None
            and self.pages_done is None
            and self.current_page is None
            and self.pages is None
        )
        if not all_none:
            if (
                self.pages_total is None
                or self.pages_done is None
                or self.current_page is None
                or not (0 <= self.pages_done <= self.pages_total)
                or not (0 <= self.current_page < self.pages_total)
            ):
                raise ValueError("multi-page progress counts are inconsistent")
            running_without_reports = (
                isinstance(self.pages, list) and not self.pages and self.pages_done == 0
            )
            if not running_without_reports:
                if not self.pages:
                    raise ValueError("pages must be a non-empty array")
                indices = [page.index for page in self.pages]
                if len(indices) != len(set(indices)):
                    raise ValueError("pages[].index must be unique")
        return self


class ModelVariantDownload(_Closed):
    """Mirrors `contracts._model_variant_download` (design §3.8)."""

    state: Literal[*sorted(DOWNLOAD_STATES)]
    bytes_done: int | None = Field(ge=0)
    bytes_total: int | None = Field(ge=0)
    error_code: str | None
    command_id: str | None


class ModelCommandAccepted(_Closed):
    """Mirrors `contracts._model_command_accepted` (design §3.9)."""

    schema_version: Literal["1.0"]
    command_id: str = Field(min_length=1)
    kind: Literal[*sorted(MODEL_COMMAND_KINDS)]
    variant_id: str = Field(min_length=1)
    state: Literal[*sorted(MODEL_COMMAND_STATES)]
    phase: Literal[*sorted(MODEL_COMMAND_PHASES)] | None
    bytes_done: int | None = Field(ge=0)
    bytes_total: int | None = Field(ge=0)
    error_code: str | None
    requested_at: str = Field(pattern=_ISO8601_UTC_MS_PATTERN)
    started_at: str | None = Field(pattern=_ISO8601_UTC_MS_PATTERN)
    finished_at: str | None = Field(pattern=_ISO8601_UTC_MS_PATTERN)

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> ModelCommandAccepted:
        if (self.error_code is not None) != (self.state == "failed"):
            raise ValueError("error_code must be set iff state is failed")
        return self


class ModelVariantEntry(_Closed):
    """Mirrors `contracts._models_response`'s per-variant entry."""

    variant_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    #: `null` exactly for a `planned` variant (no precision built or measured).
    precision: Literal["bf16", "int8", "fp8", "int4"] | None
    kv_cache: Literal["bf16", "fp8"] | None
    #: Every field below is a required key with a nullable value (`_closed_mapping`).
    bundle_manifest_sha256: str | None
    estimated_peak_vram_mib: float | None
    fits: bool | None
    resident: bool
    #: `null` exactly for a `planned` variant.
    source_kind: Literal["hf", "local"] | None
    downloaded: bool
    download_size_bytes: int | None = Field(ge=0)
    is_default: bool
    fits_reason: Literal[*sorted(FITS_REASONS)] | None
    engine: Literal[*sorted(ENGINE_IDS)]
    engine_available: bool
    engine_unavailable_reason: Literal[*sorted(ENGINE_UNAVAILABLE_REASONS)] | None
    download: ModelVariantDownload | None
    availability: Literal[*sorted(VARIANT_AVAILABILITIES)]

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> ModelVariantEntry:
        planned = self.availability == "planned"
        if (self.precision is None) != planned or (self.kv_cache is None) != planned:
            raise ValueError("precision and kv_cache are null exactly for a planned variant")
        if (self.source_kind is None) != planned:
            raise ValueError("source_kind is null exactly for a planned variant")
        if (self.fits_reason is not None) != (self.fits is False):
            raise ValueError("fits_reason must be set iff fits is false")
        if self.engine == "native" and (
            not self.engine_available or self.engine_unavailable_reason is not None
        ):
            raise ValueError("native engine is always available")
        if (self.engine_unavailable_reason is not None) == self.engine_available:
            raise ValueError("engine_unavailable_reason must be set iff engine_available is false")
        if self.availability != "published":
            if (
                self.source_kind == "local"
                or self.download is not None
                or self.downloaded
                or self.download_size_bytes is not None
                or self.bundle_manifest_sha256 is not None
                or self.resident
                or self.is_default
                or (planned and (self.estimated_peak_vram_mib is not None or self.fits is not None))
            ):
                raise ValueError(
                    "a variant that is not published has nothing downloaded, resident or default"
                )
            return self
        if (self.download is None) != (self.source_kind == "local"):
            raise ValueError("download is null exactly for a local bundle")
        if self.download is None:
            if not self.downloaded:
                raise ValueError("a local bundle variant is always downloaded")
        else:
            ready = self.download.state == "ready"
            cached_load_failure = (
                self.download.state == "failed"
                and self.download.error_code in CACHED_LOAD_FAILURE_CODES
            )
            if (ready and not self.downloaded) or (
                self.downloaded and not ready and not cached_load_failure
            ):
                raise ValueError("downloaded conflicts with transfer state")
        return self


class ModelsResponseDevice(_Closed):
    """Mirrors `contracts._models_response_device` (design §3.8)."""

    total_vram_mib: float = Field(ge=0)
    free_vram_mib: float = Field(ge=0)
    compute_capability: tuple[int, int]
    observed_at: str = Field(pattern=_ISO8601_UTC_PATTERN)


class ModelsResponse(_Closed):
    """Mirrors `contracts._models_response`."""

    schema_version: Literal["1.2"]
    default_variant: str | None
    default_variant_source: Literal["user", "config", "auto"] | None
    resident_variant: str | None
    device: ModelsResponseDevice | None
    variants: list[ModelVariantEntry]

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> ModelsResponse:
        if (self.default_variant_source is None) != (self.default_variant is None):
            raise ValueError("default_variant_source must be set iff default_variant is set")
        ids = [entry.variant_id for entry in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("models response variant_id values must be unique")
        if self.default_variant is not None and self.default_variant not in ids:
            raise ValueError("models response default_variant names no listed variant")
        if self.resident_variant is not None and self.resident_variant not in ids:
            raise ValueError("models response resident_variant names no listed variant")
        if sum(1 for entry in self.variants if entry.resident) > 1:
            raise ValueError("models response may name at most one resident variant")
        for entry in self.variants:
            if entry.is_default != (entry.variant_id == self.default_variant):
                raise ValueError("models response is_default must match default_variant")
        return self


class JobSummary(_Closed):
    """Mirrors `contracts._job_summary`: one `GET /v1/conversions` list row."""

    job_id: str = Field(pattern=_JOB_ID_PATTERN)
    state: Literal[*sorted(JOB_STATES)]
    stage: Literal[*sorted(JOB_STAGES)]
    requested_output: Literal[*sorted(ACCEPTED_OUTPUTS)]
    #: Required keys, nullable values (`_closed_mapping`'s exact key set).
    model_variant: str | None
    label: str | None
    input_kind: Literal[*sorted(INPUT_KINDS)] | None
    created_at: str = Field(pattern=_ISO8601_UTC_MS_PATTERN)
    updated_at: str = Field(pattern=_ISO8601_UTC_MS_PATTERN)
    terminal_at: str | None = Field(pattern=_ISO8601_UTC_MS_PATTERN)
    expires_at: str | None = Field(pattern=_ISO8601_UTC_MS_PATTERN)
    pages_total: int | None = Field(ge=1)
    pages_done: int | None = Field(ge=0)
    docx_produced: bool | None
    error_code: str | None

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> JobSummary:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        if (self.terminal_at is not None) != (self.state in _CONVERSION_STATUS_TERMINAL_STATES):
            raise ValueError("terminal_at must be set iff state is terminal")
        if self.expires_at is not None and self.state not in _CONVERSION_STATUS_EXPIRING_STATES:
            raise ValueError("expires_at may be set only for succeeded or expired jobs")
        if (self.pages_total is None) != (self.pages_done is None):
            raise ValueError("pages_total and pages_done must be set together")
        if (
            self.pages_total is not None
            and self.pages_done is not None
            and self.pages_done > self.pages_total
        ):
            raise ValueError("pages_done must not exceed pages_total")
        return self


class JobList(_Closed):
    """Mirrors `contracts._job_list`: the `GET /v1/conversions` envelope."""

    schema_version: Literal["1.0"]
    items: list[JobSummary]
    next_cursor: str | None


class ArtifactsManifestFile(_Closed):
    """Mirrors `contracts._artifacts_manifest_file`: one `artifacts.zip` member."""

    path: str = Field(min_length=1)
    role: Literal[*sorted(ARTIFACT_ROLES)]
    page_index: int = Field(ge=-1)
    sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    mime_type: str = Field(min_length=1)


class ArtifactsManifestOmittedRole(_Closed):
    """Mirrors `contracts._artifacts_manifest_omitted_role`."""

    role: Literal[*sorted(ARTIFACT_ROLES)]
    reason: Literal["capability_denied"]


class ArtifactsManifest(_Closed):
    """Mirrors `contracts._artifacts_manifest`: `artifacts.zip`'s `MANIFEST.json`
    (design §3.6)."""

    schema_version: Literal["1.0"]
    job_id: str = Field(pattern=_JOB_ID_PATTERN)
    generated_at: str = Field(pattern=_ISO8601_UTC_MS_PATTERN)
    files: list[ArtifactsManifestFile]
    omitted_roles: list[ArtifactsManifestOmittedRole]

    @model_validator(mode="after")
    def _no_repeated_path_or_role(self) -> ArtifactsManifest:
        paths = [entry.path for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("artifacts manifest files must not repeat a path")
        roles = [entry.role for entry in self.omitted_roles]
        if len(roles) != len(set(roles)):
            raise ValueError("artifacts manifest omitted_roles must not repeat a role")
        return self


class SessionInfo(_Closed):
    """Mirrors `contracts._session_info`."""

    schema_version: Literal["1.1"]
    authenticated: bool
    setup_required: bool
    #: Owner 2026-09-29 (DEC-052 addendum): `POST /v1/setup` accepts a
    #: request without a bearer token right now (loopback-only deployment,
    #: setup still required, exactly one principal).
    setup_tokenless: bool
    #: Required keys, nullable values (`_closed_mapping`'s exact key set).
    principal_id: str | None
    capabilities: list[str] | None
    auth_kind: Literal["bearer", "session"] | None
    expires_at: str | None

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> SessionInfo:
        if self.setup_tokenless and not self.setup_required:
            raise ValueError("session info setup_tokenless requires setup_required")
        if self.authenticated:
            if not self.principal_id:
                raise ValueError("session info principal_id must be set when authenticated")
            if self.auth_kind not in ("bearer", "session"):
                raise ValueError("session info auth_kind must be bearer or session")
            if (
                self.capabilities is None
                or any(c not in CAPABILITIES for c in self.capabilities)
                or len(self.capabilities) != len(set(self.capabilities))
                or self.capabilities != sorted(self.capabilities)
            ):
                raise ValueError("session info capabilities must be a sorted known subset")
        elif (
            self.principal_id is not None
            or self.auth_kind is not None
            or self.capabilities is not None
        ):
            raise ValueError("session info must be all-null when unauthenticated")
        if self.auth_kind == "session":
            if self.expires_at is None or _ISO8601_UTC_PATTERN.match(self.expires_at) is None:
                raise ValueError("session info expires_at must be ISO-8601 UTC for a session")
        elif self.expires_at is not None:
            raise ValueError("expires_at is valid only for auth_kind session")
        return self


class SettingsServiceVersion(_Closed):
    """Mirrors `contracts._settings_response`'s nested `service_version`."""

    git_sha: str = Field(min_length=1)
    api_version: Literal["1.1"]


class SettingsSession(_Closed):
    """Mirrors `contracts._settings_response`'s nested `session`."""

    idle_seconds: int = Field(gt=0)
    max_seconds: int

    @model_validator(mode="after")
    def _max_at_least_idle(self) -> SettingsSession:
        if self.max_seconds < self.idle_seconds:
            raise ValueError("max_seconds must be >= idle_seconds")
        return self


class SettingsDefaultVariant(_Closed):
    """Mirrors `contracts._settings_response`'s nested `default_variant`."""

    configured: str = Field(min_length=1)
    override: str | None
    effective: str | None
    source: Literal["user", "config", "auto"] | None

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> SettingsDefaultVariant:
        if (self.effective is None) != (self.source is None):
            raise ValueError("effective must be set iff source is set")
        if self.source == "user" and self.effective != self.override:
            raise ValueError("effective must equal override when source is user")
        return self


class SettingsSandbox(_Closed):
    """Mirrors `contracts._settings_response`'s nested `sandbox`."""

    rootful: bool | None


class SettingsResponse(_Closed):
    """Mirrors `contracts._settings_response` (design §3.10): `GET /v1/settings`."""

    schema_version: Literal["1.0"]
    service_version: SettingsServiceVersion
    session: SettingsSession
    default_variant: SettingsDefaultVariant
    sandbox: SettingsSandbox

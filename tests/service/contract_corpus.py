"""Shared payload corpus for `test_contracts.py` and
`test_schema_contract_parity.py` (T15 API A3, design §5.1).

Payload builders here used to live inline in `test_contracts.py`; moving them
here lets both test modules exercise the exact same valid/invalid shapes,
in particular the cross-invariant rejection cases that a Pydantic model
without the matching `@model_validator` would wrongly accept (e.g.
`docx_produced` without `diagnostic_docx` in `artifacts`, `canonical_ast_produced`
disagreeing with `docx_chain.ast_status`, an unnamed font substitution, a
`failed` page without `error_code`, an unknown `default_variant`, two
`resident` variants).

`CORPUS[name]` is a list of `(label, payload, expected_valid)` triples: every
payload both `contracts.validate_contract(payload, name)` and the matching
Pydantic model in `schemas.py` must agree on (accept iff `expected_valid`).
"""

from __future__ import annotations

from typing import Any


def diagnostic_outcome_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prediction_ir_status": "sealed",
        "claim_scope": "diagnostic",
        "canonical_ast_produced": False,
        "docx_produced": False,
        "reason_codes": ["DIAGNOSTIC_DOCX_NOT_REQUESTED"],
        "docx_chain": None,
    }
    payload.update(overrides)
    return payload


def chain_with_policies() -> dict[str, Any]:
    return {
        "policy_id": "headless-defaults.v0", "policy_status": "RATIFIED",
        "page_outcome": "terminated", "ast_status": "strict_valid",
        "renderer_status": "render_ready", "policy_counts": {"default:run:/language": 1},
        "policies": [
            {"policy_id": "headless-defaults.v0", "policy_status": "RATIFIED",
             "counts": {"default:run:/language": 1}},
            {"policy_id": "table-geometry.v0", "policy_status": "RATIFIED", "counts": {}},
            {"policy_id": "recovered-image-materialization.v0", "policy_status": "RATIFIED",
             "counts": {"recovered_image:crop_zeroed": 2}},
        ],
        "profile": "word-visual-v1",
    }


def published_outcome(chain: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return diagnostic_outcome_payload(canonical_ast_produced=True, docx_chain=chain) | {
        "docx_produced": True, "reason_codes": reasons,
    }


def model_variant_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "variant_id": "fp8",
        "label": "FP8 (fast)",
        "precision": "fp8",
        "kv_cache": "fp8",
        "bundle_manifest_sha256": "sha256:" + "a" * 64,
        "estimated_peak_vram_mib": 6900.0,
        "fits": True,
        "resident": True,
        "source_kind": "local",
        "downloaded": True,
        "download_size_bytes": None,
        "is_default": True,
        "fits_reason": None,
        "engine": "native",
        "engine_available": True,
        "engine_unavailable_reason": None,
        "download": None,
        "availability": "published",
    }
    entry.update(overrides)
    return entry


def unpublished_variant_entry(**overrides: object) -> dict[str, object]:
    """models_response 1.2: a known precision whose HF repo is not public."""
    return model_variant_entry(
        **{
            "variant_id": "fp8-hf",
            "availability": "unpublished",
            "bundle_manifest_sha256": None,
            "resident": False,
            "is_default": False,
            "source_kind": "hf",
            "downloaded": False,
            "download": None,
            **overrides,
        }
    )


def planned_variant_entry(**overrides: object) -> dict[str, object]:
    """models_response 1.2: a model that does not exist yet (Unflatten W MoE)."""
    return unpublished_variant_entry(
        **{
            "variant_id": "moe",
            "availability": "planned",
            "precision": None,
            "kv_cache": None,
            "source_kind": None,
            "estimated_peak_vram_mib": None,
            "fits": None,
            **overrides,
        }
    )


def model_variant_download(**overrides: object) -> dict[str, object]:
    download: dict[str, object] = {
        "state": "downloading",
        "bytes_done": 512,
        "bytes_total": 1024,
        "error_code": None,
        "command_id": "cmd_1",
    }
    download.update(overrides)
    return download


def model_command_accepted_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "command_id": "cmd_1",
        "kind": "download",
        "variant_id": "fp8",
        "state": "running",
        "phase": "downloading",
        "bytes_done": 512,
        "bytes_total": 1024,
        "error_code": None,
        "requested_at": "2026-09-27T09:00:00.000Z",
        "started_at": "2026-09-27T09:00:01.000Z",
        "finished_at": None,
    }
    payload.update(overrides)
    return payload


def multi_page_status_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.1",
        "job_id": "job_abc123",
        "state": "succeeded",
        "stage": "artifact_finalization",
        "attempt": 1,
        "requested_output": "diagnostic_docx",
        "claim_scope": "diagnostic",
        "product_quality_claim": False,
        "diagnostic_outcome": None,
        "artifacts": [
            {"role": "prediction_ir", "page_index": 0,
             "url": "/v1/conversions/job_abc123/artifacts/prediction_ir/0"},
            {"role": "prediction_ir", "page_index": 1,
             "url": "/v1/conversions/job_abc123/artifacts/prediction_ir/1"},
            {"role": "diagnostic_docx", "page_index": None,
             "url": "/v1/conversions/job_abc123/artifacts/diagnostic_docx"},
        ],
        "error": None,
        "model_variant": None,
        "pages_total": 2,
        "pages_done": 2,
        "current_page": 1,
        "pages": [
            {"index": 0, "state": "succeeded", "error_code": None,
             "decode_steps": 12, "seconds": 1.1},
            {"index": 1, "state": "failed", "error_code": "MODEL_DECODE_MAX_STEPS",
             "decode_steps": None, "seconds": 2.0},
        ],
        "label": None,
        "input_kind": "pdf",
        "engine": None,
        "created_at": "2026-09-27T09:00:00.000Z",
        "updated_at": "2026-09-27T09:05:00.000Z",
        "terminal_at": "2026-09-27T09:05:00.000Z",
        "expires_at": "2026-09-27T10:05:00.000Z",
    }
    payload.update(overrides)
    return payload


_SUBMISSION_VALID = {
    "schema_version": "1.0",
    "job_id": "job_abc123",
    "state": "queued",
    "requested_output": "prediction_ir",
    "claim_scope": "diagnostic",
    "product_quality_claim": False,
    "status_url": "/v1/conversions/job_abc123",
}

_DIAGNOSTIC_OUTCOME_READY_CHAIN = {
    "policy_id": "headless-defaults.v0",
    "policy_status": "RATIFIED",
    "page_outcome": "terminated",
    "ast_status": "strict_valid",
    "renderer_status": "render_ready",
}
_DIAGNOSTIC_OUTCOME_PRODUCED = diagnostic_outcome_payload(
    docx_produced=True,
    canonical_ast_produced=True,
    reason_codes=["HEADLESS_DEFAULTS_V0_APPLIED"],
    docx_chain=_DIAGNOSTIC_OUTCOME_READY_CHAIN,
)

_CONVERSION_STATUS_VALID = {
    "schema_version": "1.1",
    "job_id": "job_abc123",
    "state": "succeeded",
    "stage": "prediction_ir_sealing",
    "attempt": 1,
    "requested_output": "prediction_ir",
    "claim_scope": "diagnostic",
    "product_quality_claim": False,
    "diagnostic_outcome": diagnostic_outcome_payload(),
    "artifacts": ["prediction_ir"],
    "error": None,
    "model_variant": None,
    "pages_total": None,
    "pages_done": None,
    "current_page": None,
    "pages": None,
    "label": "board-report.pdf",
    "input_kind": "pdf",
    "engine": None,
    "created_at": "2026-09-27T09:00:00.000Z",
    "updated_at": "2026-09-27T09:05:00.000Z",
    "terminal_at": "2026-09-27T09:05:00.000Z",
    "expires_at": "2026-09-27T10:05:00.000Z",
}
_CONVERSION_STATUS_DIAGNOSTIC_DOCX = {
    **_CONVERSION_STATUS_VALID,
    "stage": "artifact_finalization",
    "requested_output": "diagnostic_docx",
    "model_variant": "fp8",
    "engine": "native",
    "diagnostic_outcome": _DIAGNOSTIC_OUTCOME_PRODUCED,
    "artifacts": ["prediction_ir", "diagnostic_docx"],
}
_CONVERSION_STATUS_NOT_PRODUCED_OUTCOME = diagnostic_outcome_payload(
    reason_codes=["DOCX_RENDER_NOT_READY"],
    docx_chain={
        "policy_id": "headless-defaults.v0",
        "policy_status": "RATIFIED",
        "page_outcome": "terminated",
        "renderer_status": "failed",
    },
)

_MODELS_RESPONSE_DEVICE = {
    "total_vram_mib": 24000.0,
    "free_vram_mib": 12000.0,
    "compute_capability": [8, 9],
    "observed_at": "2026-09-27T09:00:00Z",
}
_MODELS_RESPONSE_VALID = {
    "schema_version": "1.2",
    "default_variant": "fp8",
    "default_variant_source": "config",
    "resident_variant": "fp8",
    "device": _MODELS_RESPONSE_DEVICE,
    "variants": [model_variant_entry()],
}
_MODELS_RESPONSE_UNCONFIGURED = {
    "schema_version": "1.2",
    "default_variant": None,
    "default_variant_source": None,
    "resident_variant": None,
    "device": None,
    "variants": [],
}

def job_summary_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "job_id": "job_abc123",
        "state": "succeeded",
        "stage": "artifact_finalization",
        "requested_output": "diagnostic_docx",
        "model_variant": "fp8",
        "label": "board-report.pdf",
        "input_kind": "pdf",
        "created_at": "2026-09-27T09:00:00.000Z",
        "updated_at": "2026-09-27T09:05:00.000Z",
        "terminal_at": "2026-09-27T09:05:00.000Z",
        "expires_at": "2026-09-27T10:05:00.000Z",
        "pages_total": 2,
        "pages_done": 2,
        "docx_produced": True,
        "error_code": None,
    }
    payload.update(overrides)
    return payload


def session_info_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.1",
        "authenticated": False,
        "setup_required": False,
        "setup_tokenless": False,
        "principal_id": None,
        "capabilities": None,
        "auth_kind": None,
        "expires_at": None,
    }
    payload.update(overrides)
    return payload


_SESSION_INFO_BEARER = session_info_payload(
    authenticated=True,
    principal_id="principal_a",
    capabilities=["read_status", "submit_diagnostic"],
    auth_kind="bearer",
)
_SESSION_INFO_SESSION = session_info_payload(
    authenticated=True,
    principal_id="principal_a",
    capabilities=["read_status", "submit_diagnostic"],
    auth_kind="session",
    expires_at="2026-10-04T12:00:00Z",
)

_MULTI_PAGE_VALID = multi_page_status_payload()
_MULTI_PAGE_RUNNING = multi_page_status_payload(
    state="running", artifacts=[], pages_done=0, current_page=0, pages=[],
    terminal_at=None, expires_at=None,
)
_MULTI_PAGE_FAILED_PAGE = multi_page_status_payload()
_MULTI_PAGE_FAILED_PAGE["pages"] = [dict(page) for page in _MULTI_PAGE_FAILED_PAGE["pages"]]
_MULTI_PAGE_FAILED_PAGE["pages"][1]["error_code"] = None

CORPUS: dict[str, list[tuple[str, dict[str, Any], bool]]] = {
    "submission_response": [
        ("valid_prediction_ir", _SUBMISSION_VALID, True),
        (
            "valid_diagnostic_docx",
            {**_SUBMISSION_VALID, "requested_output": "diagnostic_docx"},
            True,
        ),
        ("invalid_requested_output_docx", {**_SUBMISSION_VALID, "requested_output": "docx"}, False),
        (
            "invalid_status_url_mismatch",
            {**_SUBMISSION_VALID, "status_url": "/v1/conversions/job_other"},
            False,
        ),
    ],
    "diagnostic_outcome": [
        ("valid_not_produced", diagnostic_outcome_payload(), True),
        ("valid_produced", _DIAGNOSTIC_OUTCOME_PRODUCED, True),
        (
            "invalid_produced_without_headless_reason",
            {**_DIAGNOSTIC_OUTCOME_PRODUCED, "reason_codes": ["DOCX_CHAIN_ERROR"]},
            False,
        ),
        (
            # Named cross-invariant: canonical_ast_produced != ast_status == strict_valid.
            "invalid_canonical_ast_mismatch",
            diagnostic_outcome_payload(
                canonical_ast_produced=True,
                docx_chain={
                    "policy_id": "headless-defaults.v0", "policy_status": "RATIFIED",
                    "page_outcome": "terminated",
                },
            ),
            False,
        ),
        (
            "invalid_unknown_reason_code",
            diagnostic_outcome_payload(reason_codes=["NOT_A_REAL_CODE"]),
            False,
        ),
        (
            "valid_named_font_substitution",
            published_outcome(
                {**chain_with_policies(), "renderer_status": "approximation_required",
                 "renderer_approximations": ["FONT_FALLBACK"]},
                ["HEADLESS_DEFAULTS_V0_APPLIED", "DOCX_FONT_SUBSTITUTED"],
            ),
            True,
        ),
        (
            # Named cross-invariant: approximation_required without DOCX_FONT_SUBSTITUTED.
            "invalid_approximation_required_without_font_substituted_reason",
            published_outcome(
                {**chain_with_policies(), "renderer_status": "approximation_required",
                 "renderer_approximations": ["FONT_FALLBACK"]},
                ["HEADLESS_DEFAULTS_V0_APPLIED"],
            ),
            False,
        ),
        (
            "invalid_approximation_required_with_other_approximation",
            published_outcome(
                {**chain_with_policies(), "renderer_status": "approximation_required",
                 "renderer_approximations": ["FONT_FALLBACK", "IMAGE_DOWNSAMPLED"]},
                ["HEADLESS_DEFAULTS_V0_APPLIED", "DOCX_FONT_SUBSTITUTED"],
            ),
            False,
        ),
    ],
    "conversion_status": [
        ("valid_prediction_ir_only", _CONVERSION_STATUS_VALID, True),
        ("valid_diagnostic_docx_scoped", _CONVERSION_STATUS_DIAGNOSTIC_DOCX, True),
        (
            "invalid_diagnostic_docx_artifact_wrong_requested_output",
            {**_CONVERSION_STATUS_DIAGNOSTIC_DOCX, "requested_output": "prediction_ir"},
            False,
        ),
        (
            # Named cross-invariant: docx_produced ⇔ diagnostic_docx in artifacts.
            "invalid_docx_produced_artifact_mismatch",
            {
                **_CONVERSION_STATUS_DIAGNOSTIC_DOCX,
                "diagnostic_outcome": _CONVERSION_STATUS_NOT_PRODUCED_OUTCOME,
            },
            False,
        ),
        ("valid_multi_page", _MULTI_PAGE_VALID, True),
        ("valid_multi_page_running_without_reports", _MULTI_PAGE_RUNNING, True),
        (
            "invalid_multi_page_fewer_reports_than_pages_done",
            multi_page_status_payload(pages_done=2, pages=[]),
            False,
        ),
        (
            "invalid_multi_page_inconsistent_counts",
            multi_page_status_payload(pages_done=3),
            False,
        ),
        (
            # Named cross-invariant: pages[].error_code must be set iff failed.
            "invalid_failed_page_without_error_code",
            _MULTI_PAGE_FAILED_PAGE,
            False,
        ),
        (
            "invalid_duplicate_page_scoped_artifact",
            multi_page_status_payload(
                artifacts=[*_MULTI_PAGE_VALID["artifacts"], _MULTI_PAGE_VALID["artifacts"][0]]
            ),
            False,
        ),
        (
            "invalid_partial_multi_page_fields",
            multi_page_status_payload(pages_total=None),
            False,
        ),
        (
            "invalid_schema_version_1_0",
            {**_CONVERSION_STATUS_VALID, "schema_version": "1.0"},
            False,
        ),
        (
            "invalid_input_kind_unknown_value",
            {**_CONVERSION_STATUS_VALID, "input_kind": "docx"},
            False,
        ),
        (
            "invalid_engine_unknown_value",
            {**_CONVERSION_STATUS_DIAGNOSTIC_DOCX, "engine": "gpu-mainframe"},
            False,
        ),
        (
            # Named cross-invariant (T42): engine is set iff model_variant is set.
            "invalid_engine_set_without_model_variant",
            {**_CONVERSION_STATUS_VALID, "engine": "native"},
            False,
        ),
        (
            "invalid_model_variant_without_engine",
            {**_CONVERSION_STATUS_DIAGNOSTIC_DOCX, "engine": None},
            False,
        ),
        (
            # Named cross-invariant: terminal_at is set iff state is terminal.
            "invalid_terminal_at_missing_for_terminal_state",
            {**_CONVERSION_STATUS_VALID, "terminal_at": None},
            False,
        ),
        (
            "invalid_terminal_at_present_for_active_state",
            {**_MULTI_PAGE_RUNNING, "terminal_at": "2026-09-27T09:05:00.000Z"},
            False,
        ),
        (
            # Named cross-invariant: expires_at only for succeeded/expired.
            "invalid_expires_at_for_non_terminal_state",
            {**_MULTI_PAGE_RUNNING, "expires_at": "2026-09-27T10:05:00.000Z"},
            False,
        ),
        (
            "invalid_updated_at_before_created_at",
            {
                **_CONVERSION_STATUS_VALID,
                "created_at": "2026-09-27T09:05:00.000Z",
                "updated_at": "2026-09-27T09:00:00.000Z",
            },
            False,
        ),
        (
            "invalid_created_at_not_iso8601_ms",
            {**_CONVERSION_STATUS_VALID, "created_at": "2026-09-27T09:00:00Z"},
            False,
        ),
    ],
    "models_response": [
        ("valid_with_variant", _MODELS_RESPONSE_VALID, True),
        ("valid_unconfigured", _MODELS_RESPONSE_UNCONFIGURED, True),
        (
            "invalid_duplicate_variant_ids",
            {
                **_MODELS_RESPONSE_VALID,
                "resident_variant": None,
                "variants": [
                    model_variant_entry(resident=False),
                    model_variant_entry(resident=False),
                ],
            },
            False,
        ),
        (
            # Named cross-invariant: at most one resident variant.
            "invalid_two_resident_variants",
            {
                **_MODELS_RESPONSE_VALID,
                "variants": [
                    model_variant_entry(resident=True),
                    model_variant_entry(variant_id="int4", resident=True, is_default=False),
                ],
            },
            False,
        ),
        (
            # Named cross-invariant: default_variant must name a listed variant.
            "invalid_default_variant_outside_variants",
            {
                **_MODELS_RESPONSE_VALID,
                "default_variant": "does-not-exist",
                "default_variant_source": "config",
                "resident_variant": None,
                "variants": [model_variant_entry(resident=False, is_default=False)],
            },
            False,
        ),
        (
            # Named cross-invariant: is_default must match default_variant (mutation
            # target: "at most one default").
            "invalid_two_default_variants",
            {
                **_MODELS_RESPONSE_VALID,
                "variants": [
                    model_variant_entry(resident=False),
                    model_variant_entry(variant_id="int4", resident=False, is_default=True),
                ],
            },
            False,
        ),
        (
            # Named cross-invariant: fits_reason must be set iff fits is false.
            "invalid_fits_reason_without_false_fits",
            {
                **_MODELS_RESPONSE_VALID,
                "variants": [model_variant_entry(fits_reason="vram")],
            },
            False,
        ),
        (
            # Named cross-invariant: native engine is always available.
            "invalid_native_engine_unavailable",
            {
                **_MODELS_RESPONSE_VALID,
                "variants": [
                    model_variant_entry(
                        engine_available=False, engine_unavailable_reason="worker_not_published"
                    )
                ],
            },
            False,
        ),
        (
            # Named cross-invariant: downloaded must equal download.state == ready.
            "invalid_downloaded_disagrees_with_download_state",
            {
                **_MODELS_RESPONSE_VALID,
                "variants": [
                    model_variant_entry(
                        source_kind="hf",
                        downloaded=True,
                        download=model_variant_download(state="downloading"),
                    )
                ],
            },
            False,
        ),
        (
            # Named cross-invariant: download is null only for a local bundle.
            "invalid_local_bundle_with_download",
            {
                **_MODELS_RESPONSE_VALID,
                "variants": [model_variant_entry(download=model_variant_download(state="ready"))],
            },
            False,
        ),
        (
            "valid_hf_variant_downloading",
            {
                **_MODELS_RESPONSE_VALID,
                "variants": [
                    model_variant_entry(
                        source_kind="hf",
                        downloaded=False,
                        download=model_variant_download(),
                    )
                ],
            },
            True,
        ),
        (
            "valid_catalogue_with_unpublished_and_planned",
            {
                **_MODELS_RESPONSE_VALID,
                "variants": [
                    model_variant_entry(),
                    unpublished_variant_entry(),
                    planned_variant_entry(),
                ],
            },
            True,
        ),
        (
            # Named cross-invariant: a non-published variant is never the default.
            "invalid_default_names_unpublished_variant",
            {
                **_MODELS_RESPONSE_VALID,
                "default_variant": "fp8-hf",
                "variants": [
                    model_variant_entry(is_default=False),
                    unpublished_variant_entry(is_default=True),
                ],
            },
            False,
        ),
        (
            "invalid_schema_version_1_1",
            {**_MODELS_RESPONSE_VALID, "schema_version": "1.1"},
            False,
        ),
    ],
    "model_variant_entry": [
        ("valid_entry", model_variant_entry(), True),
        (
            "valid_hf_entry_downloading",
            model_variant_entry(
                source_kind="hf",
                downloaded=False,
                download=model_variant_download(),
            ),
            True,
        ),
        ("invalid_unknown_precision", model_variant_entry(precision="fp32"), False),
        ("invalid_unknown_kv_cache", model_variant_entry(kv_cache="fp32"), False),
        ("invalid_empty_variant_id", model_variant_entry(variant_id=""), False),
        ("invalid_unknown_engine", model_variant_entry(engine="onnx"), False),
        (
            "invalid_unknown_engine_unavailable_reason",
            model_variant_entry(engine_unavailable_reason="because"),
            False,
        ),
        (
            "invalid_fits_reason_without_false_fits",
            model_variant_entry(fits_reason="vram"),
            False,
        ),
        (
            "invalid_native_engine_marked_unavailable",
            model_variant_entry(
                engine_available=False, engine_unavailable_reason="worker_not_published"
            ),
            False,
        ),
        (
            "invalid_downloaded_disagrees_with_download_state",
            model_variant_entry(
                source_kind="hf",
                downloaded=True,
                download=model_variant_download(state="downloading"),
            ),
            False,
        ),
        (
            "invalid_local_bundle_with_download",
            model_variant_entry(download=model_variant_download(state="ready")),
            False,
        ),
        ("valid_unpublished_entry", unpublished_variant_entry(), True),
        (
            "valid_unpublished_entry_that_does_not_fit",
            unpublished_variant_entry(fits=False, fits_reason="compute_capability"),
            True,
        ),
        ("valid_planned_entry", planned_variant_entry(), True),
        ("invalid_unknown_availability", model_variant_entry(availability="beta"), False),
        (
            "invalid_missing_availability",
            {k: v for k, v in model_variant_entry().items() if k != "availability"},
            False,
        ),
        (
            # Named cross-invariant: nothing to download for a non-published variant.
            "invalid_unpublished_with_download",
            unpublished_variant_entry(download=model_variant_download(state="absent")),
            False,
        ),
        ("invalid_unpublished_downloaded", unpublished_variant_entry(downloaded=True), False),
        ("invalid_unpublished_local", unpublished_variant_entry(source_kind="local"), False),
        ("invalid_unpublished_resident", unpublished_variant_entry(resident=True), False),
        ("invalid_unpublished_default", unpublished_variant_entry(is_default=True), False),
        (
            "invalid_unpublished_with_manifest",
            unpublished_variant_entry(bundle_manifest_sha256="sha256:" + "a" * 64),
            False,
        ),
        ("invalid_unpublished_without_precision", unpublished_variant_entry(precision=None), False),
        (
            "invalid_unpublished_without_source_kind",
            unpublished_variant_entry(source_kind=None),
            False,
        ),
        ("invalid_planned_with_precision", planned_variant_entry(precision="bf16"), False),
        ("invalid_planned_with_source_kind", planned_variant_entry(source_kind="hf"), False),
        (
            "invalid_planned_with_estimate",
            planned_variant_entry(estimated_peak_vram_mib=1.0),
            False,
        ),
        ("invalid_planned_with_fits", planned_variant_entry(fits=True), False),
        ("invalid_published_without_precision", model_variant_entry(precision=None), False),
    ],
    "model_command_accepted": [
        ("valid_running_download", model_command_accepted_payload(), True),
        (
            "valid_queued",
            model_command_accepted_payload(
                state="queued", phase=None, bytes_done=None, bytes_total=None, started_at=None
            ),
            True,
        ),
        (
            "valid_succeeded",
            model_command_accepted_payload(
                state="succeeded", phase=None, started_at=None,
                finished_at="2026-09-27T09:05:00.000Z",
            ),
            True,
        ),
        (
            "valid_failed_load",
            model_command_accepted_payload(
                kind="load", state="failed", phase=None, started_at=None,
                finished_at="2026-09-27T09:05:00.000Z", error_code="MODEL_ENGINE_UNAVAILABLE",
            ),
            True,
        ),
        ("invalid_unknown_kind", model_command_accepted_payload(kind="unload"), False),
        ("invalid_unknown_state", model_command_accepted_payload(state="cancelled"), False),
        ("invalid_unknown_phase", model_command_accepted_payload(phase="uploading"), False),
        ("invalid_empty_command_id", model_command_accepted_payload(command_id=""), False),
        (
            "invalid_requested_at_not_iso8601_ms",
            model_command_accepted_payload(requested_at="2026-09-27T09:00:00Z"),
            False,
        ),
        (
            # Named cross-invariant: error_code must be set iff state is failed.
            "invalid_error_code_without_failed_state",
            model_command_accepted_payload(error_code="MODEL_ENGINE_UNAVAILABLE"),
            False,
        ),
        (
            "invalid_failed_without_error_code",
            model_command_accepted_payload(
                state="failed", phase=None, started_at=None,
                finished_at="2026-09-27T09:05:00.000Z",
            ),
            False,
        ),
    ],
    "session_info": [
        ("valid_unauthenticated", session_info_payload(), True),
        ("valid_unauthenticated_setup_required", session_info_payload(setup_required=True), True),
        ("valid_bearer", _SESSION_INFO_BEARER, True),
        ("valid_session", _SESSION_INFO_SESSION, True),
        (
            # Named cross-invariant: authenticated requires principal_id/auth_kind/capabilities.
            "invalid_authenticated_without_principal_id",
            session_info_payload(authenticated=True, auth_kind="bearer", capabilities=[]),
            False,
        ),
        (
            "invalid_unauthenticated_with_principal_id",
            session_info_payload(principal_id="principal_a"),
            False,
        ),
        (
            "invalid_unknown_capability",
            {**_SESSION_INFO_BEARER, "capabilities": ["not_a_real_capability"]},
            False,
        ),
        (
            "invalid_unsorted_capabilities",
            {**_SESSION_INFO_BEARER, "capabilities": ["submit_diagnostic", "read_status"]},
            False,
        ),
        (
            # Named cross-invariant: expires_at only for auth_kind=session.
            "invalid_bearer_with_expires_at",
            {**_SESSION_INFO_BEARER, "expires_at": "2026-10-04T12:00:00Z"},
            False,
        ),
        (
            "invalid_session_without_expires_at",
            {**_SESSION_INFO_SESSION, "expires_at": None},
            False,
        ),
        (
            "invalid_session_expires_at_not_iso8601",
            {**_SESSION_INFO_SESSION, "expires_at": "2026-10-04 12:00:00"},
            False,
        ),
        (
            # Owner 2026-09-29 (DEC-052 addendum): tokenless setup is offered
            # only while setup is still required.
            "valid_setup_tokenless",
            session_info_payload(setup_required=True, setup_tokenless=True),
            True,
        ),
        (
            "invalid_setup_tokenless_after_setup",
            session_info_payload(setup_required=False, setup_tokenless=True),
            False,
        ),
        (
            "invalid_schema_version_1_0",
            session_info_payload(schema_version="1.0"),
            False,
        ),
        (
            "invalid_missing_setup_tokenless",
            {k: v for k, v in session_info_payload().items() if k != "setup_tokenless"},
            False,
        ),
    ],
    "problem": [
        (
            "valid_basic",
            {"type": "urn:unflatten:error:internal-error", "code": "INTERNAL_ERROR",
             "title": "Internal error", "status": 500, "retryable": False},
            True,
        ),
        (
            "valid_with_blockers",
            {"type": "urn:unflatten:error:capability-not-ready", "code": "CAPABILITY_NOT_READY",
             "title": "Requested output is not ready", "status": 503, "retryable": False,
             "blockers": ["AST_PROMOTER_UNAVAILABLE"]},
            True,
        ),
        (
            "valid_too_many_pages",
            {"type": "urn:unflatten:error:too-many-pages", "code": "TOO_MANY_PAGES",
             "title": "Too many pages", "status": 422, "retryable": False, "max_pages": 20},
            True,
        ),
        (
            "invalid_unknown_code",
            {"type": "urn:unflatten:error:secret-path", "code": "SECRET_PATH",
             "title": "bad", "status": 500, "retryable": False},
            False,
        ),
        (
            "invalid_extra_field",
            {"type": "urn:unflatten:error:internal-error", "code": "INTERNAL_ERROR",
             "title": "Internal error", "status": 500, "retryable": False, "path": "/secret"},
            False,
        ),
        (
            "invalid_max_pages_for_wrong_code",
            {"type": "urn:unflatten:error:request-invalid", "code": "REQUEST_INVALID",
             "title": "bad", "status": 400, "retryable": False, "max_pages": 20},
            False,
        ),
        (
            "invalid_blockers_for_wrong_code",
            {"type": "urn:unflatten:error:request-invalid", "code": "REQUEST_INVALID",
             "title": "bad", "status": 400, "retryable": False,
             "blockers": ["AST_PROMOTER_UNAVAILABLE"]},
            False,
        ),
        (
            "invalid_type_does_not_match_code",
            {"type": "urn:unflatten:error:wrong-urn", "code": "INTERNAL_ERROR",
             "title": "Internal error", "status": 500, "retryable": False},
            False,
        ),
    ],
}

_JOB_SUMMARY_VALID = job_summary_payload()
_JOB_SUMMARY_RUNNING = job_summary_payload(
    state="running", stage="model_decode", terminal_at=None, expires_at=None,
    docx_produced=None, error_code=None, pages_total=None, pages_done=None,
)

CORPUS["job_summary"] = [
    ("valid_succeeded", _JOB_SUMMARY_VALID, True),
    ("valid_running_no_pages", _JOB_SUMMARY_RUNNING, True),
    (
        # Named cross-invariant: terminal_at must be set iff state is terminal.
        "invalid_terminal_at_missing_for_terminal_state",
        {**_JOB_SUMMARY_VALID, "terminal_at": None},
        False,
    ),
    (
        # Named cross-invariant: expires_at only for succeeded/expired.
        "invalid_expires_at_for_non_terminal_state",
        {**_JOB_SUMMARY_RUNNING, "expires_at": "2026-09-27T10:05:00.000Z"},
        False,
    ),
    (
        # Named cross-invariant: pages_total/pages_done travel together.
        "invalid_pages_total_without_pages_done",
        {**_JOB_SUMMARY_VALID, "pages_done": None},
        False,
    ),
    (
        "invalid_pages_done_exceeds_pages_total",
        {**_JOB_SUMMARY_VALID, "pages_total": 1, "pages_done": 2},
        False,
    ),
    (
        "invalid_unknown_input_kind",
        {**_JOB_SUMMARY_VALID, "input_kind": "docx"},
        False,
    ),
    (
        "invalid_updated_at_before_created_at",
        {
            **_JOB_SUMMARY_VALID,
            "created_at": "2026-09-27T09:05:00.000Z",
            "updated_at": "2026-09-27T09:00:00.000Z",
        },
        False,
    ),
]

_JOB_LIST_VALID = {
    "schema_version": "1.0",
    "items": [_JOB_SUMMARY_VALID, _JOB_SUMMARY_RUNNING],
    "next_cursor": "eyJ0IjoxfQ",
}
_JOB_LIST_EMPTY = {"schema_version": "1.0", "items": [], "next_cursor": None}

CORPUS["job_list"] = [
    ("valid_with_items", _JOB_LIST_VALID, True),
    ("valid_empty", _JOB_LIST_EMPTY, True),
    (
        "invalid_item_bad_state",
        {**_JOB_LIST_EMPTY, "items": [{**_JOB_SUMMARY_VALID, "state": "bogus"}]},
        False,
    ),
    (
        "invalid_schema_version",
        {**_JOB_LIST_EMPTY, "schema_version": "2.0"},
        False,
    ),
]

_ARTIFACTS_MANIFEST_FILE = {
    "path": "prediction_ir.json",
    "role": "prediction_ir",
    "page_index": -1,
    "sha256": "sha256:" + "a" * 64,
    "size_bytes": 4,
    "mime_type": "application/json",
}
_ARTIFACTS_MANIFEST_PAGE_FILE = {
    "path": "source_preview/page-000.png",
    "role": "source_preview",
    "page_index": 0,
    "sha256": "sha256:" + "b" * 64,
    "size_bytes": 10,
    "mime_type": "image/png",
}
_ARTIFACTS_MANIFEST_VALID = {
    "schema_version": "1.0",
    "job_id": "job_1",
    "generated_at": "2026-09-27T00:00:00.000Z",
    "files": [_ARTIFACTS_MANIFEST_FILE, _ARTIFACTS_MANIFEST_PAGE_FILE],
    "omitted_roles": [{"role": "diagnostic_docx", "reason": "capability_denied"}],
}
_ARTIFACTS_MANIFEST_EMPTY = {
    "schema_version": "1.0",
    "job_id": "job_1",
    "generated_at": "2026-09-27T00:00:00.000Z",
    "files": [],
    "omitted_roles": [],
}

CORPUS["artifacts_manifest"] = [
    ("valid_with_files_and_omitted_role", _ARTIFACTS_MANIFEST_VALID, True),
    ("valid_empty", _ARTIFACTS_MANIFEST_EMPTY, True),
    (
        "invalid_schema_version",
        {**_ARTIFACTS_MANIFEST_EMPTY, "schema_version": "2.0"},
        False,
    ),
    (
        "invalid_duplicate_file_path",
        {
            **_ARTIFACTS_MANIFEST_EMPTY,
            "files": [_ARTIFACTS_MANIFEST_FILE, _ARTIFACTS_MANIFEST_FILE],
        },
        False,
    ),
    (
        "invalid_duplicate_omitted_role",
        {
            **_ARTIFACTS_MANIFEST_EMPTY,
            "omitted_roles": [
                {"role": "diagnostic_docx", "reason": "capability_denied"},
                {"role": "diagnostic_docx", "reason": "capability_denied"},
            ],
        },
        False,
    ),
    (
        "invalid_omitted_role_unknown_reason",
        {
            **_ARTIFACTS_MANIFEST_EMPTY,
            "omitted_roles": [{"role": "diagnostic_docx", "reason": "expired"}],
        },
        False,
    ),
    (
        "invalid_file_negative_page_index",
        {
            **_ARTIFACTS_MANIFEST_EMPTY,
            "files": [{**_ARTIFACTS_MANIFEST_FILE, "page_index": -2}],
        },
        False,
    ),
    (
        "invalid_file_bad_sha256",
        {
            **_ARTIFACTS_MANIFEST_EMPTY,
            "files": [{**_ARTIFACTS_MANIFEST_FILE, "sha256": "not-a-hash"}],
        },
        False,
    ),
    (
        "invalid_file_unknown_role",
        {
            **_ARTIFACTS_MANIFEST_EMPTY,
            "files": [{**_ARTIFACTS_MANIFEST_FILE, "role": "source_pdf"}],
        },
        False,
    ),
]


def settings_response_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "service_version": {"git_sha": "abc1234", "api_version": "1.1"},
        "session": {"idle_seconds": 43200, "max_seconds": 604800},
        "default_variant": {
            "configured": "auto",
            "override": None,
            "effective": None,
            "source": None,
        },
        "sandbox": {"rootful": None},
    }
    payload.update(overrides)
    return payload


_SETTINGS_RESPONSE_AUTO_RESOLVED = settings_response_payload(
    default_variant={
        "configured": "auto",
        "override": None,
        "effective": "bf16",
        "source": "auto",
    },
    sandbox={"rootful": False},
)
_SETTINGS_RESPONSE_USER_OVERRIDE = settings_response_payload(
    default_variant={
        "configured": "auto",
        "override": "int4",
        "effective": "int4",
        "source": "user",
    },
    sandbox={"rootful": True},
)

CORPUS["settings_response"] = [
    ("valid_unresolved_no_worker_yet", settings_response_payload(), True),
    ("valid_auto_resolved", _SETTINGS_RESPONSE_AUTO_RESOLVED, True),
    ("valid_user_override", _SETTINGS_RESPONSE_USER_OVERRIDE, True),
    (
        "invalid_schema_version",
        {**settings_response_payload(), "schema_version": "2.0"},
        False,
    ),
    (
        "invalid_service_version_api_version",
        settings_response_payload(
            service_version={"git_sha": "abc1234", "api_version": "1.0"}
        ),
        False,
    ),
    (
        "invalid_session_max_less_than_idle",
        settings_response_payload(session={"idle_seconds": 100, "max_seconds": 10}),
        False,
    ),
    (
        # Named cross-invariant: effective must be set iff source is set.
        "invalid_default_variant_effective_without_source",
        settings_response_payload(
            default_variant={
                "configured": "auto",
                "override": None,
                "effective": "bf16",
                "source": None,
            }
        ),
        False,
    ),
    (
        # Named cross-invariant: source "user" must carry effective == override.
        "invalid_default_variant_user_source_disagrees_with_override",
        settings_response_payload(
            default_variant={
                "configured": "auto",
                "override": "int4",
                "effective": "bf16",
                "source": "user",
            }
        ),
        False,
    ),
    (
        "invalid_default_variant_unknown_source",
        settings_response_payload(
            default_variant={
                "configured": "auto",
                "override": None,
                "effective": "bf16",
                "source": "worker",
            }
        ),
        False,
    ),
    (
        "invalid_sandbox_rootful_not_bool_or_null",
        settings_response_payload(sandbox={"rootful": []}),
        False,
    ),
]

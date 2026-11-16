from __future__ import annotations

import pytest
from tests.service.contract_corpus import (
    chain_with_policies,
    diagnostic_outcome_payload,
    multi_page_status_payload,
    published_outcome,
)
from tests.service.contract_corpus import (
    model_variant_entry as _model_variant_entry,
)

from unflatten.service.contracts import ContractInvalid, validate_contract
from unflatten.service.errors import (
    PROBLEM_CODES,
    ProblemInvalid,
    problem_response,
    validate_problem_response,
)


def test_validate_contract_submission_response_closed_and_diagnostic() -> None:
    payload = {
        "schema_version": "1.0",
        "job_id": "job_abc123",
        "state": "queued",
        "requested_output": "prediction_ir",
        "claim_scope": "diagnostic",
        "product_quality_claim": False,
        "status_url": "/v1/conversions/job_abc123",
    }
    assert validate_contract(payload, "submission_response") == payload
    with pytest.raises(ContractInvalid):
        validate_contract({**payload, "ast_status": "strict_valid"}, "submission_response")


def test_validate_contract_submission_response_accepts_diagnostic_docx_output() -> None:
    payload = {
        "schema_version": "1.0",
        "job_id": "job_abc123",
        "state": "queued",
        "requested_output": "diagnostic_docx",
        "claim_scope": "diagnostic",
        "product_quality_claim": False,
        "status_url": "/v1/conversions/job_abc123",
    }
    assert validate_contract(payload, "submission_response") == payload
    with pytest.raises(ContractInvalid):
        validate_contract({**payload, "requested_output": "docx"}, "submission_response")


def test_validate_contract_diagnostic_outcome_refuses_product_or_docx_claim() -> None:
    payload = diagnostic_outcome_payload()
    assert validate_contract(payload, "diagnostic_outcome") == payload
    for key in ("canonical_ast_produced", "docx_produced"):
        with pytest.raises(ContractInvalid):
            validate_contract({**payload, key: True}, "diagnostic_outcome")


def test_validate_contract_diagnostic_outcome_docx_produced_requires_full_chain() -> None:
    ready_chain = {
        "policy_id": "headless-defaults.v0",
        "policy_status": "RATIFIED",
        "page_outcome": "terminated",
        "ast_status": "strict_valid",
        "renderer_status": "render_ready",
    }
    payload = diagnostic_outcome_payload(
        docx_produced=True,
        canonical_ast_produced=True,
        reason_codes=["HEADLESS_DEFAULTS_V0_APPLIED"],
        docx_chain=ready_chain,
    )
    assert validate_contract(payload, "diagnostic_outcome") == payload

    # Missing HEADLESS_DEFAULTS_V0_APPLIED reason code.
    with pytest.raises(ContractInvalid):
        validate_contract({**payload, "reason_codes": ["DOCX_CHAIN_ERROR"]}, "diagnostic_outcome")

    # renderer_status not render_ready.
    not_ready_chain = {**ready_chain, "renderer_status": "failed"}
    with pytest.raises(ContractInvalid):
        validate_contract({**payload, "docx_chain": not_ready_chain}, "diagnostic_outcome")

    # docx_chain missing entirely.
    with pytest.raises(ContractInvalid):
        validate_contract({**payload, "docx_chain": None}, "diagnostic_outcome")


def test_validate_contract_diagnostic_outcome_canonical_ast_matches_chain_status() -> None:
    chain = {
        "policy_id": "headless-defaults.v0",
        "policy_status": "RATIFIED",
        "page_outcome": "terminated",
        "ast_status": "strict_valid",
    }
    payload = diagnostic_outcome_payload(canonical_ast_produced=True, docx_chain=chain)
    assert validate_contract(payload, "diagnostic_outcome") == payload

    with pytest.raises(ContractInvalid):
        validate_contract({**payload, "canonical_ast_produced": False}, "diagnostic_outcome")


def test_validate_contract_diagnostic_outcome_reason_codes_are_closed() -> None:
    with pytest.raises(ContractInvalid):
        validate_contract(
            diagnostic_outcome_payload(reason_codes=["NOT_A_REAL_CODE"]),
            "diagnostic_outcome",
        )


@pytest.mark.parametrize(
    "docx_chain",
    [
        "not-a-mapping",
        {"policy_id": "headless-defaults.v0", "policy_status": "RATIFIED"},  # missing page_outcome
        {
            "policy_id": "headless-defaults.v0",
            "policy_status": "RATIFIED",
            "page_outcome": "terminated",
            "unknown_key": 1,
        },
        {
            "policy_id": "headless-defaults.v0",
            "policy_status": "RATIFIED",
            "page_outcome": "terminated",
            "failure_codes": ["free text without a pointer"],
        },
        {
            "policy_id": "headless-defaults.v0",
            "policy_status": "RATIFIED",
            "page_outcome": "terminated",
            "failure_codes": ["DOCX_RENDER_NOT_READY"],
        },
        {
            "policy_id": "headless-defaults.v0",
            "policy_status": "RATIFIED",
            "page_outcome": "terminated",
            "policy_counts": {"default:run:/style/font_family": -1},
        },
        {
            "policy_id": "headless-defaults.v0",
            "policy_status": "RATIFIED",
            "page_outcome": "terminated",
            "policy_counts": {"default:run:/style/font_family": 1.5},
        },
    ],
)
def test_validate_contract_diagnostic_outcome_docx_chain_closed_map(docx_chain: object) -> None:
    with pytest.raises(ContractInvalid):
        validate_contract(diagnostic_outcome_payload(docx_chain=docx_chain), "diagnostic_outcome")


def test_validate_contract_diagnostic_outcome_docx_chain_accepts_valid_failure_codes() -> None:
    chain = {
        "policy_id": "headless-defaults.v0",
        "policy_status": "RATIFIED",
        "page_outcome": "terminated",
        "failure_codes": ["DOCX_RENDER_NOT_READY|/body/0/runs/0"],
    }
    payload = diagnostic_outcome_payload(docx_chain=chain)
    assert validate_contract(payload, "diagnostic_outcome") == payload


def test_validate_contract_unknown_contract_rejected() -> None:
    with pytest.raises(ContractInvalid):
        validate_contract({}, "rfc004_result_envelope")


def test_validate_contract_conversion_status_is_closed_service_projection() -> None:
    payload = {
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
        "label": None,
        "input_kind": "pdf",
        "engine": None,
        "created_at": "2026-09-27T09:00:00.000Z",
        "updated_at": "2026-09-27T09:05:00.000Z",
        "terminal_at": "2026-09-27T09:05:00.000Z",
        "expires_at": "2026-09-27T10:05:00.000Z",
    }
    assert validate_contract(payload, "conversion_status") == payload
    with pytest.raises(ContractInvalid):
        validate_contract({**payload, "renderer_status": "not_run"}, "conversion_status")


def test_validate_contract_conversion_status_diagnostic_docx_artifact_scoping() -> None:
    ready_chain = {
        "policy_id": "headless-defaults.v0",
        "policy_status": "RATIFIED",
        "page_outcome": "terminated",
        "ast_status": "strict_valid",
        "renderer_status": "render_ready",
    }
    ready_outcome = diagnostic_outcome_payload(
        docx_produced=True,
        canonical_ast_produced=True,
        reason_codes=["HEADLESS_DEFAULTS_V0_APPLIED"],
        docx_chain=ready_chain,
    )
    payload = {
        "schema_version": "1.1",
        "job_id": "job_abc123",
        "state": "succeeded",
        "stage": "artifact_finalization",
        "attempt": 1,
        "requested_output": "diagnostic_docx",
        "claim_scope": "diagnostic",
        "product_quality_claim": False,
        "diagnostic_outcome": ready_outcome,
        "artifacts": ["prediction_ir", "diagnostic_docx"],
        "error": None,
        "model_variant": "fp8",
        "pages_total": None,
        "pages_done": None,
        "current_page": None,
        "pages": None,
        "label": "board-report.pdf",
        "input_kind": "pdf",
        "engine": "native",
        "created_at": "2026-09-27T09:00:00.000Z",
        "updated_at": "2026-09-27T09:05:00.000Z",
        "terminal_at": "2026-09-27T09:05:00.000Z",
        "expires_at": "2026-09-27T10:05:00.000Z",
    }
    assert validate_contract(payload, "conversion_status") == payload

    # diagnostic_docx artifact only permitted when requested for diagnostic_docx output.
    with pytest.raises(ContractInvalid):
        validate_contract(
            {**payload, "requested_output": "prediction_ir"}, "conversion_status"
        )

    # docx_produced must agree with artifacts membership.
    not_produced_outcome = diagnostic_outcome_payload(
        reason_codes=["DOCX_RENDER_NOT_READY"],
        docx_chain={
            "policy_id": "headless-defaults.v0",
            "policy_status": "RATIFIED",
            "page_outcome": "terminated",
            "renderer_status": "failed",
        },
    )
    with pytest.raises(ContractInvalid):
        validate_contract(
            {**payload, "diagnostic_outcome": not_produced_outcome}, "conversion_status"
        )


def test_problem_response_closed_content_free_shape() -> None:
    problem = problem_response(
        code="CAPABILITY_NOT_READY",
        title="Requested output is not ready",
        status=503,
        retryable=False,
        blockers=["AST_PROMOTER_UNAVAILABLE"],
    )
    assert problem == {
        "type": "urn:unflatten:error:capability-not-ready",
        "title": "Requested output is not ready",
        "status": 503,
        "code": "CAPABILITY_NOT_READY",
        "retryable": False,
        "blockers": ["AST_PROMOTER_UNAVAILABLE"],
    }
    assert "detail" not in problem
    assert validate_problem_response(problem) == problem


def test_problem_response_too_many_pages_carries_max_pages() -> None:
    problem = problem_response(
        code="TOO_MANY_PAGES", title="Too many pages", status=422, retryable=False, max_pages=20,
    )
    assert problem["max_pages"] == 20
    assert validate_problem_response(problem) == problem


def test_problem_response_max_pages_rejected_for_any_other_code() -> None:
    with pytest.raises(ProblemInvalid):
        problem_response(
            code="REQUEST_INVALID", title="bad", status=400, retryable=False, max_pages=20,
        )


def test_problem_response_unknown_code_or_extra_field_rejected() -> None:
    assert "INTERNAL_ERROR" in PROBLEM_CODES
    with pytest.raises(ProblemInvalid):
        problem_response(code="SECRET_PATH", title="bad", status=500, retryable=False)
    with pytest.raises(ProblemInvalid):
        problem_response(
            code="INTERNAL_ERROR",
            title="Internal error",
            status=500,
            retryable=False,
            path="/secret",
        )


def test_docx_chain_accepts_the_dec044_dodatok_b_policies_list_and_profile():
    """DEC-044 Dodatok B (26.9.): the chain reports every applied policy and
    the derived AST profile. The first T11 run on prod refused every job as
    INTERNAL_ERROR because this closed contract did not know the two keys."""
    payload = diagnostic_outcome_payload(
        canonical_ast_produced=True, docx_chain=chain_with_policies()
    )
    outcome = validate_contract(payload, "diagnostic_outcome")
    assert outcome["docx_chain"]["profile"] == "word-visual-v1"
    chain = {**chain_with_policies(), "profile": None}
    validate_contract(
        diagnostic_outcome_payload(canonical_ast_produced=True, docx_chain=chain),
        "diagnostic_outcome",
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["policies"].append(
            {"policy_id": "story-kind-dedup.v0", "policy_status": "PROPOSED", "counts": {}}
        ),
        lambda c: c["policies"].append(
            {"policy_id": "", "policy_status": "RATIFIED", "counts": {}}
        ),
        lambda c: c["policies"].append({"policy_id": "x.v0", "policy_status": "RATIFIED"}),
        lambda c: c["policies"].append(
            {"policy_id": "x.v0", "policy_status": "RATIFIED", "counts": {"k": -1}}
        ),
        lambda c: c.__setitem__("policies", []),
        lambda c: c.__setitem__("profile", "word-visual-v9"),
        lambda c: c.__setitem__("profile", 7),
    ],
)
def test_docx_chain_refuses_unratified_or_malformed_policies_and_unknown_profiles(mutate):
    chain = chain_with_policies()
    mutate(chain)
    with pytest.raises(ContractInvalid):
        validate_contract(
            diagnostic_outcome_payload(canonical_ast_produced=True, docx_chain=chain),
            "diagnostic_outcome",
        )


def test_docx_produced_with_a_named_font_substitution_is_valid():
    """DEC-044 Dodatok C (owner 2026-09-26)."""
    chain = {**chain_with_policies(), "renderer_status": "approximation_required",
             "renderer_approximations": ["FONT_FALLBACK"]}
    validate_contract(
        published_outcome(chain, ["HEADLESS_DEFAULTS_V0_APPLIED", "DOCX_FONT_SUBSTITUTED"]),
        "diagnostic_outcome",
    )


@pytest.mark.parametrize(
    ("approximations", "reasons"),
    [
        (["FONT_FALLBACK"], ["HEADLESS_DEFAULTS_V0_APPLIED"]),  # substitution not named
        (["FONT_FALLBACK", "IMAGE_DOWNSAMPLED"],
         ["HEADLESS_DEFAULTS_V0_APPLIED", "DOCX_FONT_SUBSTITUTED"]),
        (None, ["HEADLESS_DEFAULTS_V0_APPLIED", "DOCX_FONT_SUBSTITUTED"]),
    ],
)
def test_docx_produced_with_any_other_approximation_is_refused(approximations, reasons):
    chain = {**chain_with_policies(), "renderer_status": "approximation_required"}
    if approximations is not None:
        chain["renderer_approximations"] = approximations
    with pytest.raises(ContractInvalid):
        validate_contract(published_outcome(chain, reasons), "diagnostic_outcome")


def test_validate_contract_models_response_accepts_well_formed_payload() -> None:
    payload = {
        "schema_version": "1.2",
        "default_variant": "fp8",
        "default_variant_source": "config",
        "resident_variant": "fp8",
        "device": None,
        "variants": [_model_variant_entry()],
    }
    assert validate_contract(payload, "models_response") == payload


def test_validate_contract_models_response_allows_null_defaults_when_unconfigured() -> None:
    payload = {
        "schema_version": "1.2",
        "default_variant": None,
        "default_variant_source": None,
        "resident_variant": None,
        "device": None,
        "variants": [],
    }
    assert validate_contract(payload, "models_response") == payload


def test_validate_contract_models_response_rejects_duplicate_variant_ids() -> None:
    payload = {
        "schema_version": "1.2",
        "default_variant": "fp8",
        "default_variant_source": "config",
        "resident_variant": None,
        "device": None,
        "variants": [_model_variant_entry(resident=False), _model_variant_entry(resident=False)],
    }
    with pytest.raises(ContractInvalid):
        validate_contract(payload, "models_response")


def test_validate_contract_models_response_rejects_two_resident_variants() -> None:
    payload = {
        "schema_version": "1.2",
        "default_variant": "fp8",
        "default_variant_source": "config",
        "resident_variant": "fp8",
        "device": None,
        "variants": [
            _model_variant_entry(resident=True),
            _model_variant_entry(variant_id="int4", resident=True, is_default=False),
        ],
    }
    with pytest.raises(ContractInvalid):
        validate_contract(payload, "models_response")


def test_validate_contract_models_response_rejects_unknown_default_variant() -> None:
    payload = {
        "schema_version": "1.2",
        "default_variant": "does-not-exist",
        "default_variant_source": "config",
        "resident_variant": None,
        "device": None,
        "variants": [_model_variant_entry(resident=False, is_default=False)],
    }
    with pytest.raises(ContractInvalid):
        validate_contract(payload, "models_response")


def test_validate_contract_models_response_rejects_is_default_mismatch() -> None:
    payload = {
        "schema_version": "1.2",
        "default_variant": "fp8",
        "default_variant_source": "config",
        "resident_variant": None,
        "device": None,
        "variants": [_model_variant_entry(resident=False, is_default=False)],
    }
    with pytest.raises(ContractInvalid):
        validate_contract(payload, "models_response")


def test_validate_contract_conversion_status_accepts_multi_page_shape() -> None:
    payload = multi_page_status_payload()
    assert validate_contract(payload, "conversion_status") == payload


def test_validate_contract_conversion_status_accepts_running_multi_page_without_reports() -> None:
    # Live 2026-09-27: the UI polls a running multi-page job while page 0 is
    # still decoding; pages is [] then and the status endpoint returned 500.
    payload = multi_page_status_payload(
        state="running", artifacts=[], pages_done=0, current_page=0, pages=[],
        terminal_at=None, expires_at=None,
    )
    assert validate_contract(payload, "conversion_status") == payload


def test_validate_contract_conversion_status_rejects_fewer_page_reports_than_pages_done() -> None:
    with pytest.raises(ContractInvalid):
        validate_contract(
            multi_page_status_payload(pages_done=2, pages=[]), "conversion_status"
        )


def test_validate_contract_conversion_status_rejects_inconsistent_page_counts() -> None:
    with pytest.raises(ContractInvalid):
        validate_contract(multi_page_status_payload(pages_done=3), "conversion_status")


def test_validate_contract_conversion_status_rejects_failed_page_without_error_code() -> None:
    payload = multi_page_status_payload()
    payload["pages"][1]["error_code"] = None
    with pytest.raises(ContractInvalid):
        validate_contract(payload, "conversion_status")


def test_validate_contract_conversion_status_rejects_duplicate_page_scoped_artifact() -> None:
    payload = multi_page_status_payload()
    payload["artifacts"].append(payload["artifacts"][0])
    with pytest.raises(ContractInvalid):
        validate_contract(payload, "conversion_status")


def test_validate_contract_conversion_status_rejects_partial_multi_page_fields() -> None:
    payload = multi_page_status_payload(pages_total=None)
    with pytest.raises(ContractInvalid):
        validate_contract(payload, "conversion_status")

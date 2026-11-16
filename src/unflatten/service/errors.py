from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from unflatten.service.readiness import DOCX_BLOCKERS

PROBLEM_CODES = frozenset(
    {
        "REQUEST_INVALID",
        "AUTHENTICATION_FAILED",
        "CAPABILITY_DENIED",
        "RESOURCE_NOT_FOUND",
        "JOB_TERMINAL",
        "ARTIFACT_EXPIRED",
        "UPLOAD_TOO_LARGE",
        "UNSUPPORTED_MEDIA_TYPE",
        "IDEMPOTENCY_KEY_REUSED",
        "QUEUE_CAPACITY_EXCEEDED",
        "PDF_SIGNATURE_INVALID",
        "PDF_MALFORMED",
        "PDF_ENCRYPTED",
        "PDF_ACTIVE_CONTENT",
        "PDF_EMBEDDED_FILE",
        "PDF_ANNOTATION_UNSUPPORTED",
        "PDF_HIDDEN_TEXT",
        "PDF_TEXT_VISIBILITY_AMBIGUOUS",
        "UNSUPPORTED_PAGE_COUNT",
        "PDF_PAGE_DIMENSIONS_EXCEEDED",
        "POPPLER_TIMEOUT",
        "POPPLER_RESOURCE_LIMIT",
        "RASTER_OUTPUT_INVALID",
        "RASTER_PIXEL_LIMIT_EXCEEDED",
        "PNG_INVALID",
        "MODEL_INPUT_FORBIDDEN_CHANNEL",
        "RUNTIME_POLICY_MISMATCH",
        "MODEL_COMPONENT_LOCK_MISMATCH",
        "MODEL_CHECKPOINT_UNVERIFIED",
        "MODEL_DECODE_MAX_STEPS",
        "MODEL_DECODE_VOCABULARY_EXHAUSTED",
        "PREDICTION_IR_INVALID",
        "AST_PROMOTION_UNAVAILABLE",
        "AST_PROMOTION_FAILED",
        "RENDERER_FAILED",
        "CAPABILITY_NOT_READY",
        "JOB_CANCELLED",
        "INTERNAL_ERROR",
        # DEC-044 Dodatok D §1: model variant selection (submit-time header).
        "UNKNOWN_MODEL_VARIANT",
        "MODEL_VARIANT_DOES_NOT_FIT",
        # DEC-044 Dodatok D §2: image/multipart submission (B2).
        "TOO_MANY_PAGES",
        "IMAGE_INPUT_INVALID",
        # DEC-052 design §3.0: web session, setup and model-command surfaces.
        "HOST_NOT_ALLOWED",
        "WEB_REQUEST_REJECTED",
        "RATE_LIMITED",
        "SETUP_REQUIRED",
        "SETUP_ALREADY_COMPLETE",
        "JOB_ACTIVE",
        "MODEL_COMMAND_CONFLICT",
        "MODEL_VARIANT_NOT_DOWNLOADABLE",
        "MODEL_ENGINE_UNAVAILABLE",
        "MODEL_DOWNLOAD_FAILED",
        # models_response 1.2 (owner request 2026-09-29): a listed catalogue
        # variant that cannot be downloaded, loaded or selected yet.
        "MODEL_VARIANT_NOT_PUBLISHED",
        "MODEL_VARIANT_PLANNED",
    }
)


class ProblemInvalid(ValueError):
    """A public problem response is open, unknown, or content-bearing."""


def problem_response(**fields: object) -> dict[str, Any]:
    """Build a closed RFC 7807-style response from public, content-free fields."""
    return validate_problem_response(fields)


def validate_problem_response(fields: object) -> dict[str, Any]:
    """Validate and normalize a closed public problem response mapping."""
    if not isinstance(fields, Mapping):
        raise ProblemInvalid("problem response must be an object")
    allowed = {
        "type",
        "code",
        "title",
        "status",
        "retryable",
        "blockers",
        "max_pages",
        "retry_after_seconds",
    }
    unknown = set(fields) - allowed
    missing = {"code", "title", "status", "retryable"} - set(fields)
    if unknown or missing:
        raise ProblemInvalid("problem fields are not the closed service problem contract")
    code = fields["code"]
    title = fields["title"]
    status = fields["status"]
    retryable = fields["retryable"]
    if not isinstance(code, str) or code not in PROBLEM_CODES:
        raise ProblemInvalid("problem code is not in the public service code registry")
    if not isinstance(title, str) or not title or len(title) > 200:
        raise ProblemInvalid("problem title must be a short non-empty string")
    if type(status) is not int or not 400 <= status <= 599:
        raise ProblemInvalid("problem status must be an HTTP error status")
    if type(retryable) is not bool:
        raise ProblemInvalid("problem retryable must be boolean")
    problem_type = f"urn:unflatten:error:{code.lower().replace('_', '-')}"
    if "type" in fields and fields["type"] != problem_type:
        raise ProblemInvalid("problem type must be derived from its stable code")
    result: dict[str, Any] = {
        "type": problem_type,
        "title": title,
        "status": status,
        "code": code,
        "retryable": retryable,
    }
    if "blockers" in fields:
        blockers = _validate_blockers(code, fields["blockers"])
        result["blockers"] = list(blockers)
    if "max_pages" in fields:
        max_pages = fields["max_pages"]
        if code != "TOO_MANY_PAGES" or type(max_pages) is not int or max_pages < 1:
            raise ProblemInvalid("max_pages is valid only for a TOO_MANY_PAGES problem")
        result["max_pages"] = max_pages
    if "retry_after_seconds" in fields:
        retry_after_seconds = fields["retry_after_seconds"]
        if (
            code != "RATE_LIMITED"
            or type(retry_after_seconds) is not int
            or retry_after_seconds < 1
        ):
            raise ProblemInvalid("retry_after_seconds is valid only for a RATE_LIMITED problem")
        result["retry_after_seconds"] = retry_after_seconds
    return result


def _validate_blockers(code: str, blockers: object) -> list[str]:
    if (
        code != "CAPABILITY_NOT_READY"
        or not isinstance(blockers, list)
        or not blockers
        or any(not isinstance(blocker, str) or blocker not in DOCX_BLOCKERS for blocker in blockers)
        or len(blockers) != len(set(blockers))
    ):
        raise ProblemInvalid("blockers are valid only for capability readiness problems")
    return blockers

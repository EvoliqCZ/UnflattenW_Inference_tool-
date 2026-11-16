from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DOCX_BLOCKERS = (
    "PRODUCT_CHECKPOINT_UNAVAILABLE",
    "AST_PROMOTER_UNAVAILABLE",
    "TYPED_PROPERTY_HEADS_UNAVAILABLE",
    "RESOURCE_BRANCH_UNAVAILABLE",
    "RFC004_RENDERER_GATE_UNAVAILABLE",
)

COMPONENT_CODES = (
    "runtime_policy",
    "principal_store",
    "job_store",
    "artifact_store",
    "pdf_sandbox",
    "worker",
)


class ReadinessInvalid(ValueError):
    """Readiness inputs are incomplete, open, or not boolean."""


def job_store_component_statuses(
    job_store: Any, *, now: float, stale_after_seconds: float
) -> dict[str, bool]:
    """Read content-free worker status from the shared durable store."""
    if stale_after_seconds <= 0:
        raise ReadinessInvalid("readiness staleness window must be positive")
    status = job_store.readiness_status()
    if status is None:
        return {"worker": False, "pdf_sandbox": False}
    worker_ready, sandbox_ready, _diagnostic_docx_ready, observed_at = status
    if now >= observed_at + stale_after_seconds:
        return {"worker": False, "pdf_sandbox": False}
    return {"worker": worker_ready, "pdf_sandbox": sandbox_ready}


def diagnostic_docx_status(job_store: Any) -> bool:
    """Read the last published ``diagnostic_docx_ready`` flag regardless of staleness.

    Per design D10/P3: a stale row with ``1`` is still accepted; a missing row or a
    published ``0`` means the capability is not ready.
    """
    status = job_store.readiness_status()
    if status is None:
        return False
    _worker_ready, _sandbox_ready, diagnostic_docx_ready, _observed_at = status
    return diagnostic_docx_ready


def evaluate_readiness(
    *,
    component_statuses: Mapping[str, bool],
    docx_requirements: Mapping[str, bool],
    diagnostic_docx_ready: bool,
) -> dict[str, Any]:
    """Build a content-free operational and DOCX capability readiness report.

    ``docx_requirements`` maps each blocker code to whether that requirement has
    closed. This pure gate does not claim that any current requirement has closed.
    ``diagnostic_docx_ready`` is the last published worker capability flag; it is a
    separate argument, never folded into ``COMPONENT_CODES`` (P4).
    """
    _validate_statuses("component", component_statuses, COMPONENT_CODES)
    _validate_statuses("DOCX requirement", docx_requirements, DOCX_BLOCKERS)
    if type(diagnostic_docx_ready) is not bool:
        raise ReadinessInvalid("diagnostic_docx_ready must be a boolean")
    service_ready = all(component_statuses[code] for code in COMPONENT_CODES)
    blockers = [code for code in DOCX_BLOCKERS if not docx_requirements[code]]
    return {
        "schema_version": "1.0",
        "service_ready": service_ready,
        "components": [
            {"code": code, "ready": component_statuses[code]} for code in COMPONENT_CODES
        ],
        "capabilities": {
            "prediction_ir": service_ready,
            "diagnostic_docx": service_ready and diagnostic_docx_ready,
            "docx": not blockers,
        },
        "docx_blockers": blockers,
    }


def current_docx_requirements() -> dict[str, bool]:
    """Return the fail-closed state: no current DOCX blocker is resolved."""
    return dict.fromkeys(DOCX_BLOCKERS, False)


def _validate_statuses(label: str, statuses: Mapping[str, bool], expected: tuple[str, ...]) -> None:
    if set(statuses) != set(expected):
        raise ReadinessInvalid(f"{label} statuses must contain exactly the closed code set")
    if any(type(statuses[code]) is not bool for code in expected):
        raise ReadinessInvalid(f"{label} statuses must be booleans")

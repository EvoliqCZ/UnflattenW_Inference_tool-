"""GateResult value objects + closed JSON Schema validator (RFC003 §33/§45).

Eligibility, disposition, gate report and reason codes are separate axes
(RFC003 §33 — never one merged status). `reason_codes` uses the §45
registry plus three documented engine extensions (the §45 list is
explicitly a minimum): `OBJECT_INTEGRITY_FAILED` (content-addressed
object unreadable/hash-mismatched), `TARGETS_INVALID` (target bundle or
fragment validation failed), `RENDER_SKIPPED` (downgrade marker for
compiled-but-unrendered records — the §34.1 render evidence is absent,
not failed) and `RECORD_MALFORMED` (the index.jsonl record itself is
malformed, e.g. `render_status` outside the closed
rendered|skipped|failed set).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from unflatten.ast.issues import Issue

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "gate_result.schema.json"

GATE_RESULT_SCHEMA_VERSION = "0.1.0"

#: §45 registry + documented engine extensions (see module docstring).
REASON_CODES: frozenset[str] = frozenset(
    {
        "PROFILE_UNSUPPORTED",
        "VISIBLE_CONTENT_DROPPED",
        "AST_INVALID",
        "RESOURCE_INVALID",
        "FONT_UNAVAILABLE",
        "APPROXIMATION_NOT_ALLOWED",
        "TEXT_CONSERVATION_FAILED",
        "STRUCTURE_CONSERVATION_FAILED",
        "RENDER_FAILED",
        "MAPPING_INCOMPLETE",
        "TABLE_MAPPING_AMBIGUOUS",
        "OBSERVABILITY_AMBIGUOUS",
        "RIGHTS_UNKNOWN",
        "PRIVACY_REVIEW_REQUIRED",
        "SPLIT_COLLISION",
        "NEAR_DUPLICATE_COLLISION",
        "NONDETERMINISTIC_BUILD",
        "OBJECT_INTEGRITY_FAILED",
        "TARGETS_INVALID",
        "RENDER_SKIPPED",
        "RECORD_MALFORMED",
    }
)


@dataclass(frozen=True)
class GateCheck:
    """One named gate outcome: pass / fail / not_applicable + typed issues."""

    gate_id: str
    status: str
    issues: list[Issue] = field(default_factory=list)
    #: rights/privacy basis record (never None for those gates when they
    #: pass — the gate EXISTS and records why, RFC003 §50); None elsewhere.
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "status": self.status,
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "node_id": issue.node_id,
                    "path": issue.path,
                    "message": issue.message,
                }
                for issue in self.issues
            ],
            "details": self.details,
        }


@dataclass(frozen=True)
class GateResult:
    """Record-level eligibility decision derived from the gate report."""

    document_bundle_id: str
    eligibility: str
    disposition: str
    reason_codes: list[str]
    gate_report: list[GateCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_result_schema_version": GATE_RESULT_SCHEMA_VERSION,
            "document_bundle_id": self.document_bundle_id,
            "eligibility": self.eligibility,
            "disposition": self.disposition,
            "reason_codes": list(self.reason_codes),
            "gate_report": [check.to_dict() for check in self.gate_report],
        }


@lru_cache(maxsize=1)
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _pointer(path: Any) -> str:
    return "".join(f"/{p}" for p in path)


def validate_gate_result(doc: Any) -> list[Issue]:
    """Validate a serialized gate result against the closed schema."""
    return [
        Issue(
            code="GATE_RESULT_SCHEMA_VIOLATION",
            severity="error",
            node_id=None,
            path=_pointer(error.absolute_path),
            message=error.message,
        )
        for error in sorted(
            _validator().iter_errors(doc), key=lambda e: str(e.absolute_path)
        )
    ]

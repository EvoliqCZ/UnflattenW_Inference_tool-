from __future__ import annotations

from unflatten.service import openapi_export
from unflatten.service.auth import CAPABILITIES
from unflatten.service.contracts import (
    ARTIFACT_ROLES,
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
)
from unflatten.service.docx_stage import FONT_SUBSTITUTION_CODES
from unflatten.service.errors import PROBLEM_CODES
from unflatten.service.readiness import COMPONENT_CODES, DOCX_BLOCKERS

_GROUP_REGISTRY = {
    "problem_codes": PROBLEM_CODES,
    "diagnostic_reason_codes": DIAGNOSTIC_REASON_CODES,
    "job_states": JOB_STATES,
    "job_stages": JOB_STAGES,
    "docx_blockers": DOCX_BLOCKERS,
    "component_codes": COMPONENT_CODES,
    "renderer_approximations": FONT_SUBSTITUTION_CODES,
    "artifact_roles": ARTIFACT_ROLES,
    "capabilities": CAPABILITIES,
    "model_command_kinds": MODEL_COMMAND_KINDS,
    "model_command_states": MODEL_COMMAND_STATES,
    "model_command_phases": MODEL_COMMAND_PHASES,
    "download_states": DOWNLOAD_STATES,
    "fits_reasons": FITS_REASONS,
    "input_kinds": INPUT_KINDS,
    "engine_ids": ENGINE_IDS,
    "engine_unavailable_reasons": ENGINE_UNAVAILABLE_REASONS,
}


def test_codes_json_groups_equal_registries_sorted() -> None:
    codes = openapi_export.build_codes()
    assert set(codes) == {"schema_version"} | set(_GROUP_REGISTRY)
    assert codes["schema_version"] == "1.0"
    for group, registry in _GROUP_REGISTRY.items():
        assert codes[group] == sorted(registry)

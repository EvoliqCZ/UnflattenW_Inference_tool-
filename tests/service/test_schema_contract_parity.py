"""T15 API A3 (design §5.1): `contracts.CONTRACT_FIELDS` and `schemas.py`'s
Pydantic models must never drift -- neither in their closed field sets nor in
their cross-field invariants. A `@model_validator` missing on the Pydantic
side for a rule `contracts.validate_contract` enforces must turn this file
red (`countable-claims-need-a-counter`: a coherent-looking model is not
proof; only running the corpus both ways is)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from tests.service import contract_corpus

from unflatten.service import schemas
from unflatten.service.contracts import CONTRACT_FIELDS, ContractInvalid, validate_contract

_MODELS: dict[str, type[BaseModel]] = {
    "submission_response": schemas.SubmissionResponse,
    "diagnostic_outcome": schemas.DiagnosticOutcome,
    "conversion_status": schemas.ConversionStatus,
    "models_response": schemas.ModelsResponse,
    "model_variant_entry": schemas.ModelVariantEntry,
    "model_command_accepted": schemas.ModelCommandAccepted,
    "session_info": schemas.SessionInfo,
    "job_summary": schemas.JobSummary,
    "job_list": schemas.JobList,
    "settings_response": schemas.SettingsResponse,
    "artifacts_manifest": schemas.ArtifactsManifest,
    "problem": schemas.Problem,
}


class _Wrong:
    """A value of a type no field in `schemas.py` legitimately accepts --
    guarantees a "wrong type" mutation is rejected regardless of any
    coercion pydantic might otherwise apply to an in-domain type."""


def _verdict(payload: dict[str, Any], name: str) -> bool:
    try:
        validate_contract(payload, name)
        return True
    except ContractInvalid:
        return False


def _model_verdict(model: type[BaseModel], payload: dict[str, Any]) -> bool:
    try:
        model.model_validate(payload)
        return True
    except ValidationError:
        return False


def _mutations(
    payload: dict[str, Any], fields: frozenset[str]
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Missing field, foreign field, wrong type of each present field --
    every mutation is expected to make both sides reject (design §5.1)."""
    for field in sorted(payload):
        mutated = dict(payload)
        del mutated[field]
        yield f"missing:{field}", mutated
    yield "foreign_field", {**payload, "__unflatten_test_unknown_field__": True}
    for field in sorted(payload):
        mutated = dict(payload)
        mutated[field] = _Wrong()
        yield f"wrong_type:{field}", mutated
    assert fields  # every named contract carries at least one field


@pytest.mark.parametrize("name", sorted(CONTRACT_FIELDS))
def test_model_fields_equal_contract_fields(name: str) -> None:
    model = _MODELS[name]
    assert set(model.model_fields) == CONTRACT_FIELDS[name]


@pytest.mark.parametrize("name", sorted(contract_corpus.CORPUS))
def test_verdicts_agree_on_corpus_and_mutations(name: str) -> None:
    model = _MODELS[name]
    for label, payload, expected_valid in contract_corpus.CORPUS[name]:
        assert _verdict(payload, name) is expected_valid, f"{name}:{label} (hand validator)"
        assert _model_verdict(model, payload) is expected_valid, f"{name}:{label} (pydantic)"
    for base_label, base_payload, base_valid in contract_corpus.CORPUS[name]:
        if not base_valid:
            continue
        for mutation_label, mutated in _mutations(base_payload, CONTRACT_FIELDS[name]):
            hand_ok = _verdict(mutated, name)
            model_ok = _model_verdict(model, mutated)
            assert hand_ok == model_ok, (
                f"{name}:{base_label}/{mutation_label} disagree "
                f"(hand validator={hand_ok}, pydantic={model_ok})"
            )


def test_schemas_module_import_not_shadowed_by_data_dir() -> None:
    """`src/unflatten/service/schemas/` (no `__init__.py`, holds
    `runtime_policy.schema.json`) coexists with `schemas.py`. Python must
    resolve `unflatten.service.schemas` to the regular module, not the
    namespace package -- this is the load-bearing assumption behind every
    other test in this file (design §A.1)."""
    assert schemas.__file__ is not None
    assert schemas.__file__.endswith("schemas.py")
    assert hasattr(schemas, "SubmissionResponse")

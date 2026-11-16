from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import unflatten.service.policy as policy_module
from unflatten.service.policy import (
    RuntimePolicyInvalid,
    load_runtime_policy,
    runtime_policy_hash,
    validate_runtime_policy,
)

ROOT = Path(__file__).parents[2]
POLICY_PATH = ROOT / "src/unflatten/service/policies/runtime_policy.json"


def test_runtime_policy_approved_artifact_has_exact_values() -> None:
    assert POLICY_PATH.is_file(), "approved runtime policy artifact must be checked in"
    assert json.loads(POLICY_PATH.read_text(encoding="utf-8")) == {
        "active_jobs_per_principal": 4,
        "crash_recovery_source_raster_seconds": 3600,
        "decode_steps_hard_ceiling": 8192,
        "diagnostic_docx_retention_seconds": 86400,
        "heartbeat_interval_seconds": 15,
        "lease_duration_seconds": 900,
        "max_attempts": 2,
        "max_diagnostic_docx_bytes": 67108864,
        "max_page_height_points": 1440,
        "max_page_width_points": 1440,
        "max_pdf_pages": 100,
        "max_prediction_ir_bytes": 33554432,
        "max_raster_height_pixels": 6000,
        "max_raster_pixels": 36000000,
        "max_raster_width_pixels": 6000,
        "max_sandbox_memory_bytes": 1073741824,
        "max_sandbox_open_files": 64,
        "max_sandbox_output_bytes": 134217728,
        "max_sandbox_processes": 32,
        "max_sandbox_tmpfs_bytes": 268435456,
        "max_upload_bytes": 52428800,
        "metadata_retention_seconds": 2592000,
        "policy_version": "1.3.0",
        "prediction_ir_retention_seconds": 86400,
        "raster_dpi": 300,
        "scanner_cpu_time_seconds": 30,
        "scanner_wall_time_seconds": 60,
        "service_runtime_policy_schema_version": "1.2.0",
        "worker_inference_concurrency": 1,
        "global_active_jobs": 32,
        "max_png_bytes": 134217728,
    }


def test_load_runtime_policy_approved_artifact_validates() -> None:
    policy = load_runtime_policy(POLICY_PATH)
    assert policy["raster_dpi"] == 300


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_validate_runtime_policy_non_closed_document_rejected(mutation: str) -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if mutation == "missing":
        del policy["max_upload_bytes"]
    else:
        policy["surprise"] = 1
    with pytest.raises(RuntimePolicyInvalid):
        validate_runtime_policy(policy)


@pytest.mark.parametrize(
    "field", ["diagnostic_docx_retention_seconds", "max_diagnostic_docx_bytes"]
)
@pytest.mark.parametrize("mutation", ["missing", "zero", "negative"])
def test_validate_runtime_policy_diagnostic_docx_keys_required_positive(
    field: str, mutation: str
) -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if mutation == "missing":
        del policy[field]
    elif mutation == "zero":
        policy[field] = 0
    else:
        policy[field] = -1
    with pytest.raises(RuntimePolicyInvalid):
        validate_runtime_policy(policy)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("heartbeat_interval_seconds", 900),
        ("max_raster_width_pixels", 6001),
        ("max_raster_height_pixels", 6001),
        ("max_raster_pixels", 36000001),
        ("max_sandbox_output_bytes", 268435457),
    ],
)
def test_validate_runtime_policy_invalid_relationship_rejected(field: str, value: int) -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy[field] = value
    with pytest.raises(RuntimePolicyInvalid):
        validate_runtime_policy(policy)


def test_runtime_policy_hash_equivalent_key_order_stable() -> None:
    policy = load_runtime_policy(POLICY_PATH)
    reordered = {key: deepcopy(policy[key]) for key in reversed(policy)}
    digest = runtime_policy_hash(policy)
    assert digest == runtime_policy_hash(reordered)
    assert digest.startswith("sha256:") and len(digest) == 71


def test_runtime_policy_approved_hash_is_pinned() -> None:
    assert getattr(policy_module, "APPROVED_RUNTIME_POLICY_HASH", None) == (
        "sha256:8ab57dc00aac96a67e8465c472a7fbf101b5cd5b91b9555107c7271981008418"
    )


def test_load_runtime_policy_bad_json_contextual_error(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimePolicyInvalid, match="not valid JSON"):
        load_runtime_policy(path)

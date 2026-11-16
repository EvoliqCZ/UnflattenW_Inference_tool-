from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from unflatten.service.api import ApiDependencies, create_app
from unflatten.service.artifact_store import ArtifactStore
from unflatten.service.auth import Principal, PrincipalStore
from unflatten.service.job_store import JobStore
from unflatten.service.models_config import validate_models_config
from unflatten.service.policy import load_runtime_policy, runtime_policy_hash
from unflatten.service.readiness import COMPONENT_CODES

TOKEN = "a" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}
PDF = b"%PDF-1.7\nminimal"

_MODELS_CONFIG = validate_models_config(
    {
        "schema_version": "1.0",
        "default_variant": "bf16",
        "variants": [
            {
                "variant_id": "bf16",
                "label": "BF16 (baseline)",
                "precision": "bf16",
                "kv_cache": "bf16",
                "bundle": "/data/model-bundles/v15b-bf16-bf16",
            },
            {
                "variant_id": "int4",
                "label": "INT4 (small)",
                "precision": "int4",
                "kv_cache": "fp8",
                "bundle": "/data/model-bundles/v15b-int4-fp8",
            },
        ],
    }
)


def _principal_store(*, capabilities: frozenset[str] | None = None) -> PrincipalStore:
    principal = Principal(
        "principal_a",
        capabilities
        or frozenset(
            {
                "submit_diagnostic",
                "read_status",
                "download_prediction_ir",
                "cancel_own_job",
                "download_docx",
            }
        ),
    )
    return PrincipalStore([(hashlib.sha256(TOKEN.encode()).digest(), principal)])


_AUTO_MODELS_CONFIG = validate_models_config(
    {
        "schema_version": "1.0",
        "default_variant": "auto",
        "variants": [
            {
                "variant_id": "bf16",
                "label": "BF16 (baseline)",
                "precision": "bf16",
                "kv_cache": "bf16",
                "bundle": "/data/model-bundles/v15b-bf16-bf16",
            },
        ],
    }
)

_HF_MODELS_CONFIG = validate_models_config(
    {
        "schema_version": "1.0",
        "default_variant": "fp8",
        "variants": [
            {
                "variant_id": "fp8",
                "label": "FP8 (fast)",
                "precision": "fp8",
                "kv_cache": "fp8",
                "source": {
                    "kind": "hf",
                    "repo_id": "unflatten/v15b-fp8-fp8",
                    "revision": "a" * 40,
                    "bundle_manifest_sha256": "sha256:" + "b" * 64,
                },
            },
        ],
    }
)


def _client(
    tmp_path: Path,
    *,
    models_config: dict[str, Any] | None = _MODELS_CONFIG,
    models_cache_dir: Path | None = None,
    capabilities: frozenset[str] | None = None,
) -> TestClient:
    tmp_path.mkdir(parents=True, exist_ok=True)
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    dependencies = ApiDependencies(
        principal_store=_principal_store(capabilities=capabilities),
        job_store=jobs,
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        runtime_policy=policy,
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:" + "1" * 64,
        component_statuses=dict.fromkeys(COMPONENT_CODES, True),
        diagnostic_docx_status=True,
        clock=lambda: 100.0,
        models_config=models_config,
        models_cache_dir=models_cache_dir,
    )
    return TestClient(create_app(dependencies), base_url="http://127.0.0.1")


def _submit(client: TestClient, **headers: str):
    return client.post(
        "/v1/conversions",
        content=PDF,
        headers={
            **AUTH,
            "Content-Type": "application/pdf",
            "Idempotency-Key": "request-1",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
            **headers,
        },
    )


def test_get_models_without_config_returns_empty_unconfigured_response(tmp_path: Path) -> None:
    client = _client(tmp_path, models_config=None)

    response = client.get("/v1/models", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "1.2",
        "default_variant": None,
        "default_variant_source": None,
        "resident_variant": None,
        "device": None,
        "variants": [],
    }


def test_get_models_requires_authentication(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.get("/v1/models").status_code == 401


def test_get_models_lists_configured_variants_with_no_resident_before_worker_publishes(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/models", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["default_variant"] == "bf16"
    assert body["resident_variant"] is None
    ids = {variant["variant_id"] for variant in body["variants"]}
    assert ids == {"bf16", "int4"}
    assert all(variant["fits"] is None for variant in body["variants"])
    assert all(variant["resident"] is False for variant in body["variants"])
    assert all(variant["source_kind"] == "local" for variant in body["variants"])
    assert all(variant["downloaded"] is True for variant in body["variants"])


def test_get_models_reflects_resident_variant_and_fit_once_worker_publishes(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer,
        resident_variant="int4",
        free_vram_mib=1000.0,
        total_vram_mib=8000.0,
        compute_capability=(8, 9),
        now=2.0,
    )

    response = client.get("/v1/models", headers=AUTH)

    body = response.json()
    assert body["resident_variant"] == "int4"
    by_id = {variant["variant_id"]: variant for variant in body["variants"]}
    assert by_id["int4"]["resident"] is True
    # The resident variant always fits, even though its measured free VRAM
    # (a small remainder after loading it) alone would say otherwise.
    assert by_id["int4"]["fits"] is True
    assert by_id["bf16"]["resident"] is False
    # bf16 tier (~11240 MiB * 1.15 margin ~= 12926 MiB) does not fit an
    # 8000 MiB TOTAL card, regardless of what is currently free.
    assert by_id["bf16"]["fits"] is False


def test_submit_without_model_header_uses_default_variant(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client)

    assert response.status_code == 202
    status = client.get(response.json()["status_url"], headers=AUTH)
    assert status.json()["model_variant"] == "bf16"


def test_submit_with_model_header_selects_that_variant(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client, **{"X-Unflatten-Model": "int4"})

    assert response.status_code == 202
    status = client.get(response.json()["status_url"], headers=AUTH)
    assert status.json()["model_variant"] == "int4"


def test_submit_with_unknown_model_variant_is_refused(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client, **{"X-Unflatten-Model": "does-not-exist"})

    assert response.status_code == 422
    assert response.json()["code"] == "UNKNOWN_MODEL_VARIANT"


def test_submit_with_variant_that_does_not_fit_measured_vram_is_refused(tmp_path: Path) -> None:
    client = _client(tmp_path)
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer,
        resident_variant="int4",
        free_vram_mib=1000.0,
        total_vram_mib=8000.0,
        compute_capability=(8, 9),
        now=2.0,
    )

    response = _submit(client, **{"X-Unflatten-Model": "bf16"})

    assert response.status_code == 422
    assert response.json()["code"] == "MODEL_VARIANT_DOES_NOT_FIT"


def test_submit_with_the_resident_variant_is_always_accepted(tmp_path: Path) -> None:
    """Regression: the worker publishes free VRAM AFTER its own model is
    already loaded, so the resident variant's own footprint is never
    'available' by the free-VRAM measurement alone -- it must still be
    accepted (DEC-044 Dodatok D §1 fix, 2026-09-27 field report)."""
    client = _client(tmp_path)
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer,
        resident_variant="int4",
        free_vram_mib=50.0,  # almost nothing free, because int4 is already loaded
        total_vram_mib=8000.0,
        compute_capability=(8, 9),
        now=2.0,
    )

    response = _submit(client, **{"X-Unflatten-Model": "int4"})

    assert response.status_code == 202


def test_get_models_resident_variant_always_fits_even_with_almost_no_free_vram(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer, resident_variant="int4", free_vram_mib=50.0, total_vram_mib=8000.0,
        compute_capability=(8, 9), now=2.0,
    )

    response = client.get("/v1/models", headers=AUTH)

    by_id = {variant["variant_id"]: variant for variant in response.json()["variants"]}
    assert by_id["int4"]["fits"] is True


def test_submit_without_models_config_keeps_todays_single_bundle_behaviour(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path, models_config=None)

    response = _submit(client)

    assert response.status_code == 202
    status = client.get(response.json()["status_url"], headers=AUTH)
    assert status.json()["model_variant"] is None


def test_submit_rejects_repeated_model_header(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/v1/conversions",
        content=PDF,
        headers={
            **AUTH,
            "Content-Type": "application/pdf",
            "Idempotency-Key": "request-1",
            "X-Unflatten-Profile": "word-core-v0",
            "X-Unflatten-Output": "prediction_ir",
        },
    )
    assert response.status_code == 202
    duplicate_header_response = client.post(
        "/v1/conversions",
        content=PDF,
        headers=[
            ("Authorization", AUTH["Authorization"]),
            ("Content-Type", "application/pdf"),
            ("Idempotency-Key", "request-2"),
            ("X-Unflatten-Profile", "word-core-v0"),
            ("X-Unflatten-Output", "prediction_ir"),
            ("X-Unflatten-Model", "bf16"),
            ("X-Unflatten-Model", "int4"),
        ],
    )
    assert duplicate_header_response.status_code == 400


def test_get_models_reports_device_facts_once_worker_publishes(tmp_path: Path) -> None:
    client = _client(tmp_path)
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer, resident_variant="int4", free_vram_mib=1000.0, total_vram_mib=8000.0,
        compute_capability=(8, 9), now=2.0,
    )

    response = client.get("/v1/models", headers=AUTH)

    body = response.json()
    assert body["device"] == {
        "total_vram_mib": 8000.0,
        "free_vram_mib": 1000.0,
        "compute_capability": [8, 9],
        "observed_at": "1970-01-01T00:00:02Z",
    }


def test_get_models_device_is_null_before_any_readiness_publication(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/models", headers=AUTH)

    assert response.json()["device"] is None


def test_get_models_is_default_matches_effective_default(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/models", headers=AUTH)

    body = response.json()
    assert body["default_variant"] == "bf16"
    assert body["default_variant_source"] == "config"
    by_id = {variant["variant_id"]: variant for variant in body["variants"]}
    assert by_id["bf16"]["is_default"] is True
    assert by_id["int4"]["is_default"] is False


def test_get_models_user_override_becomes_the_default(tmp_path: Path) -> None:
    client = _client(tmp_path)
    jobs = client.app.state.dependencies.job_store
    jobs.set_setting("default_variant", "int4", now=1.0)

    response = client.get("/v1/models", headers=AUTH)

    body = response.json()
    assert body["default_variant"] == "int4"
    assert body["default_variant_source"] == "user"
    by_id = {variant["variant_id"]: variant for variant in body["variants"]}
    assert by_id["int4"]["is_default"] is True
    assert by_id["bf16"]["is_default"] is False


def test_get_models_fits_reason_reported_when_variant_does_not_fit(tmp_path: Path) -> None:
    client = _client(tmp_path)
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer, resident_variant="int4", free_vram_mib=1000.0, total_vram_mib=8000.0,
        compute_capability=(8, 9), now=2.0,
    )

    response = client.get("/v1/models", headers=AUTH)

    by_id = {variant["variant_id"]: variant for variant in response.json()["variants"]}
    assert by_id["bf16"]["fits"] is False
    assert by_id["bf16"]["fits_reason"] == "vram"
    assert by_id["int4"]["fits"] is True
    assert by_id["int4"]["fits_reason"] is None


def test_get_models_fits_reason_is_null_before_any_readiness_publication(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/models", headers=AUTH)

    assert all(variant["fits_reason"] is None for variant in response.json()["variants"])


def test_get_models_native_variant_is_always_engine_available(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/models", headers=AUTH)

    for variant in response.json()["variants"]:
        assert variant["engine"] == "native"
        assert variant["engine_available"] is True
        assert variant["engine_unavailable_reason"] is None


def test_get_models_local_bundle_variant_has_no_download_object(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/models", headers=AUTH)

    for variant in response.json()["variants"]:
        assert variant["source_kind"] == "local"
        assert variant["download"] is None


def test_get_models_hf_variant_download_absent_by_default(tmp_path: Path) -> None:
    client = _client(
        tmp_path, models_config=_HF_MODELS_CONFIG, models_cache_dir=tmp_path / "hf-cache"
    )

    response = client.get("/v1/models", headers=AUTH)

    variant = response.json()["variants"][0]
    assert variant["source_kind"] == "hf"
    assert variant["downloaded"] is False
    assert variant["download"] == {
        "state": "absent", "bytes_done": None, "bytes_total": None,
        "error_code": None, "command_id": None,
    }


def test_get_models_hf_variant_download_reports_active_command(tmp_path: Path) -> None:
    client = _client(
        tmp_path, models_config=_HF_MODELS_CONFIG, models_cache_dir=tmp_path / "hf-cache"
    )
    jobs = client.app.state.dependencies.job_store
    jobs.enqueue_model_command(
        kind="download", variant_id="fp8", requested_by="principal_a", now=1.0
    )
    jobs.claim_next_model_command(worker_id="worker_gpu_0", now=2.0)

    response = client.get("/v1/models", headers=AUTH)

    variant = response.json()["variants"][0]
    assert variant["download"]["state"] == "downloading"
    assert variant["download"]["command_id"] is not None


@pytest.mark.parametrize('phase', ['downloading', 'verifying', 'loading'])
def test_get_models_cached_snapshot_with_active_command_reports_progress_without_500(
    tmp_path, monkeypatch, phase,
):
    client = _client(tmp_path, models_config=_HF_MODELS_CONFIG,
                     models_cache_dir=tmp_path / 'hf-cache')
    monkeypatch.setattr('unflatten.service.api.is_variant_downloaded',
                        lambda *a, **k: (True, 1234))
    jobs = client.app.state.dependencies.job_store
    command, _ = jobs.enqueue_model_command(kind='load', variant_id='fp8',
                                            requested_by='principal_a', now=1.0)
    jobs.claim_next_model_command(worker_id='worker_gpu_0', now=2.0)
    jobs.update_model_command_progress(command.command_id, worker_id='worker_gpu_0',
        phase=phase, bytes_done=100, bytes_total=1000, now=3.0)
    response = client.get('/v1/models', headers=AUTH)
    assert response.status_code == 200
    variant = response.json()['variants'][0]
    assert variant['downloaded'] is False
    assert variant['download']['state'] == phase


def test_get_models_hf_variant_download_reports_failed_last_command(tmp_path: Path) -> None:
    client = _client(
        tmp_path, models_config=_HF_MODELS_CONFIG, models_cache_dir=tmp_path / "hf-cache"
    )
    jobs = client.app.state.dependencies.job_store
    record, _ = jobs.enqueue_model_command(
        kind="download", variant_id="fp8", requested_by="principal_a", now=1.0
    )
    jobs.claim_next_model_command(worker_id="worker_gpu_0", now=2.0)
    jobs.finish_model_command(
        record.command_id, worker_id="worker_gpu_0", state="failed", now=3.0,
        error_code="MODEL_CHECKPOINT_UNVERIFIED",
    )

    response = client.get("/v1/models", headers=AUTH)

    variant = response.json()["variants"][0]
    assert variant["download"]["state"] == "failed"
    assert variant["download"]["error_code"] == "MODEL_CHECKPOINT_UNVERIFIED"


def test_submit_native_variant_before_any_readiness_publication_returns_202(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    response = _submit(client, **{"X-Unflatten-Model": "bf16"})

    assert response.status_code == 202


def test_submit_without_model_header_resolves_auto_default(tmp_path: Path) -> None:
    client = _client(tmp_path, models_config=_AUTO_MODELS_CONFIG)
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer, resident_variant=None, free_vram_mib=20000.0, total_vram_mib=20000.0,
        compute_capability=(8, 9), now=2.0,
    )

    response = _submit(client)

    assert response.status_code == 202
    status = client.get(response.json()["status_url"], headers=AUTH)
    assert status.json()["model_variant"] == "bf16"


def test_submit_without_model_header_and_no_auto_fit_returns_422(tmp_path: Path) -> None:
    client = _client(tmp_path, models_config=_AUTO_MODELS_CONFIG)
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer, resident_variant=None, free_vram_mib=1000.0, total_vram_mib=1000.0,
        compute_capability=(8, 9), now=2.0,
    )

    response = _submit(client)

    assert response.status_code == 422
    assert response.json()["code"] == "MODEL_VARIANT_DOES_NOT_FIT"


def test_put_models_default_sets_user_override(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.put(
        "/v1/models/default", json={"variant_id": "int4"}, headers=AUTH
    )

    assert response.status_code == 200
    body = response.json()
    assert body["default_variant"] == "int4"
    assert body["default_variant_source"] == "user"


def test_put_models_default_null_resets_to_config(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.put("/v1/models/default", json={"variant_id": "int4"}, headers=AUTH)

    response = client.put("/v1/models/default", json={"variant_id": None}, headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["default_variant"] == "bf16"
    assert body["default_variant_source"] == "config"


def test_put_models_default_unknown_variant_returns_422(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.put(
        "/v1/models/default", json={"variant_id": "does-not-exist"}, headers=AUTH
    )

    assert response.status_code == 422
    assert response.json()["code"] == "UNKNOWN_MODEL_VARIANT"


def test_put_models_default_does_not_change_policy_hash(tmp_path: Path) -> None:
    client = _client(tmp_path)
    before = client.app.state.dependencies.policy_hash

    response = client.put("/v1/models/default", json={"variant_id": "int4"}, headers=AUTH)

    assert response.status_code == 200
    assert client.app.state.dependencies.policy_hash == before


def test_put_models_default_requires_authentication(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.put("/v1/models/default", json={"variant_id": "int4"})

    assert response.status_code == 401


def test_put_models_default_requires_submit_diagnostic_capability(tmp_path: Path) -> None:
    # Design §3.1: `PUT /v1/models/default` mutates service state, so it
    # requires `submit_diagnostic`, not the read-only `read_status`.
    client = _client(tmp_path, capabilities=frozenset({"read_status"}))

    response = client.put("/v1/models/default", json={"variant_id": "int4"}, headers=AUTH)

    assert response.status_code == 403
    assert response.json()["code"] == "CAPABILITY_DENIED"
    follow_up = client.get("/v1/models", headers=AUTH)
    assert follow_up.json()["default_variant"] == "bf16"


def test_put_models_default_succeeds_with_submit_diagnostic_capability(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path, capabilities=frozenset({"submit_diagnostic"}))

    response = client.put("/v1/models/default", json={"variant_id": "int4"}, headers=AUTH)

    assert response.status_code == 200
    assert response.json()["default_variant"] == "int4"


def test_put_models_default_without_config_returns_422(tmp_path: Path) -> None:
    client = _client(tmp_path, models_config=None)

    response = client.put("/v1/models/default", json={"variant_id": "int4"}, headers=AUTH)

    assert response.status_code == 422
    assert response.json()["code"] == "UNKNOWN_MODEL_VARIANT"


# --- T42: model commands (download/load/delete) and MODEL_ENGINE_UNAVAILABLE ---


def test_post_models_download_enqueues_command_for_hf_variant(tmp_path: Path) -> None:
    client = _client(
        tmp_path, models_config=_HF_MODELS_CONFIG, models_cache_dir=tmp_path / "hf-cache"
    )

    response = client.post("/v1/models/fp8/download", headers=AUTH)

    assert response.status_code == 202
    body = response.json()
    assert body["kind"] == "download"
    assert body["variant_id"] == "fp8"
    assert body["state"] == "queued"
    assert body["command_id"]


def test_post_models_download_is_idempotent_for_same_kind(tmp_path: Path) -> None:
    client = _client(
        tmp_path, models_config=_HF_MODELS_CONFIG, models_cache_dir=tmp_path / "hf-cache"
    )

    first = client.post("/v1/models/fp8/download", headers=AUTH)
    second = client.post("/v1/models/fp8/download", headers=AUTH)

    assert first.status_code == second.status_code == 202
    assert first.json()["command_id"] == second.json()["command_id"]


def test_post_models_download_conflicts_with_a_different_active_kind(tmp_path: Path) -> None:
    client = _client(
        tmp_path, models_config=_HF_MODELS_CONFIG, models_cache_dir=tmp_path / "hf-cache"
    )
    jobs = client.app.state.dependencies.job_store
    jobs.enqueue_model_command(
        kind="load", variant_id="fp8", requested_by="principal_a", now=1.0
    )

    response = client.post("/v1/models/fp8/download", headers=AUTH)

    assert response.status_code == 409
    assert response.json()["code"] == "MODEL_COMMAND_CONFLICT"


def test_post_models_download_local_bundle_variant_returns_422(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/v1/models/bf16/download", headers=AUTH)

    assert response.status_code == 422
    assert response.json()["code"] == "MODEL_VARIANT_NOT_DOWNLOADABLE"


def test_post_models_download_unknown_variant_returns_404(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/v1/models/does-not-exist/download", headers=AUTH)

    assert response.status_code == 404
    assert response.json()["code"] == "RESOURCE_NOT_FOUND"


def test_post_models_download_requires_authentication(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/v1/models/bf16/download")

    assert response.status_code == 401


def test_delete_models_download_resident_variant_returns_409(tmp_path: Path) -> None:
    client = _client(
        tmp_path, models_config=_HF_MODELS_CONFIG, models_cache_dir=tmp_path / "hf-cache"
    )
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer, resident_variant="fp8", free_vram_mib=1000.0, total_vram_mib=8000.0,
        compute_capability=(8, 9), now=2.0,
    )

    response = client.delete("/v1/models/fp8/download", headers=AUTH)

    assert response.status_code == 409
    assert response.json()["code"] == "MODEL_COMMAND_CONFLICT"


def test_delete_models_download_local_bundle_variant_returns_422(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.delete("/v1/models/bf16/download", headers=AUTH)

    assert response.status_code == 422
    assert response.json()["code"] == "MODEL_VARIANT_NOT_DOWNLOADABLE"


def test_delete_models_download_non_resident_variant_enqueues_delete(tmp_path: Path) -> None:
    client = _client(
        tmp_path, models_config=_HF_MODELS_CONFIG, models_cache_dir=tmp_path / "hf-cache"
    )

    response = client.delete("/v1/models/fp8/download", headers=AUTH)

    assert response.status_code == 202
    body = response.json()
    assert body["kind"] == "delete"
    assert body["variant_id"] == "fp8"


def test_post_models_load_enqueues_command(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/v1/models/bf16/load", headers=AUTH)

    assert response.status_code == 202
    body = response.json()
    assert body["kind"] == "load"
    assert body["variant_id"] == "bf16"


def test_post_models_load_conflicts_with_a_different_active_kind(tmp_path: Path) -> None:
    client = _client(tmp_path)
    jobs = client.app.state.dependencies.job_store
    jobs.enqueue_model_command(
        kind="delete", variant_id="bf16", requested_by="principal_a", now=1.0
    )

    response = client.post("/v1/models/bf16/load", headers=AUTH)

    assert response.status_code == 409
    assert response.json()["code"] == "MODEL_COMMAND_CONFLICT"


def test_post_models_load_unknown_variant_returns_404(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/v1/models/does-not-exist/load", headers=AUTH)

    assert response.status_code == 404


def test_post_models_load_variant_that_does_not_fit_returns_422(tmp_path: Path) -> None:
    client = _client(tmp_path)
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer, resident_variant="int4", free_vram_mib=1000.0, total_vram_mib=1000.0,
        compute_capability=(8, 9), now=2.0,
    )

    response = client.post("/v1/models/bf16/load", headers=AUTH)

    assert response.status_code == 422
    assert response.json()["code"] == "MODEL_VARIANT_DOES_NOT_FIT"


def test_list_models_estimate_and_fits_include_static_kv_reserve(tmp_path: Path) -> None:
    """Owner decision 2026-09-29: the worker decodes on a static KV cache by
    default, so `GET /v1/models` costs its `max_steps` preallocation
    (648 MiB) on top of the margin-inflated measured peak. int4's measured
    5189.5 MiB * 1.15 = 5967.9 MiB alone would fit a 6500 MiB card; with the
    reserve (6615.9 MiB) it does not."""
    client = _client(tmp_path)
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer,
        resident_variant=None,
        free_vram_mib=6500.0,
        total_vram_mib=6500.0,
        compute_capability=(8, 9),
        now=2.0,
    )

    body = client.get("/v1/models", headers=AUTH).json()

    by_id = {variant["variant_id"]: variant for variant in body["variants"]}
    assert by_id["bf16"]["estimated_peak_vram_mib"] == 11240.5 * 1.15 + 648.0
    assert by_id["int4"]["estimated_peak_vram_mib"] == 5189.5 * 1.15 + 648.0
    assert by_id["int4"]["fits"] is False
    assert by_id["int4"]["fits_reason"] == "vram"


# Owner request 2026-09-29: catalogue variants that are listed but cannot be
# downloaded or selected yet -- an `unpublished` precision variant (HF repo
# not public, DEC-050) and a `planned` future model (Unflatten W MoE).
_CATALOGUE_MODELS_CONFIG = validate_models_config(
    {
        "schema_version": "1.0",
        "default_variant": "bf16",
        "variants": [
            {
                "variant_id": "bf16",
                "label": "BF16 (baseline)",
                "precision": "bf16",
                "kv_cache": "bf16",
                "bundle": "/data/model-bundles/v15b-bf16-bf16",
            },
            {
                "variant_id": "fp8-hf",
                "label": "FP8",
                "precision": "fp8",
                "kv_cache": "fp8",
                "availability": "unpublished",
            },
            {"variant_id": "moe", "label": "Unflatten W MoE", "availability": "planned"},
        ],
    }
)


def _catalogue_client(tmp_path: Path) -> TestClient:
    return _client(
        tmp_path, models_config=_CATALOGUE_MODELS_CONFIG, models_cache_dir=tmp_path / "hf-cache"
    )


def _publish_device(client: TestClient, *, total_vram_mib: float, cc: tuple[int, int]) -> None:
    jobs = client.app.state.dependencies.job_store
    writer = jobs.claim_readiness_writer(worker_id="worker_gpu_0", now=1.0)
    jobs.publish_model_state(
        writer,
        resident_variant="bf16",
        free_vram_mib=1000.0,
        total_vram_mib=total_vram_mib,
        compute_capability=cc,
        now=2.0,
    )


def test_get_models_lists_unpublished_and_planned_variants_as_not_downloadable(
    tmp_path: Path,
) -> None:
    client = _catalogue_client(tmp_path)

    response = client.get("/v1/models", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "1.2"
    by_id = {variant["variant_id"]: variant for variant in body["variants"]}
    assert [variant["variant_id"] for variant in body["variants"]] == ["bf16", "fp8-hf", "moe"]
    assert by_id["bf16"]["availability"] == "published"
    unpublished = by_id["fp8-hf"]
    assert unpublished["availability"] == "unpublished"
    assert unpublished["source_kind"] == "hf"
    assert unpublished["download"] is None
    assert unpublished["downloaded"] is False
    assert unpublished["bundle_manifest_sha256"] is None
    assert unpublished["precision"] == "fp8"
    assert unpublished["estimated_peak_vram_mib"] == 6817.9 * 1.15 + 648.0
    planned = by_id["moe"]
    assert planned["availability"] == "planned"
    assert planned["source_kind"] is None
    assert planned["download"] is None
    assert planned["downloaded"] is False
    assert planned["precision"] is None
    assert planned["kv_cache"] is None
    assert planned["estimated_peak_vram_mib"] is None
    assert planned["fits"] is None
    assert all(not by_id[vid]["is_default"] for vid in ("fp8-hf", "moe"))


def test_get_models_unpublished_variant_fits_is_estimated_once_device_known(
    tmp_path: Path,
) -> None:
    client = _catalogue_client(tmp_path)
    _publish_device(client, total_vram_mib=12000.0, cc=(8, 6))

    by_id = {v["variant_id"]: v for v in client.get("/v1/models", headers=AUTH).json()["variants"]}

    assert by_id["fp8-hf"]["fits"] is False
    assert by_id["fp8-hf"]["fits_reason"] == "compute_capability"
    # A planned model has no measured tier -- nothing to estimate from.
    assert by_id["moe"]["fits"] is None
    assert by_id["moe"]["fits_reason"] is None


@pytest.mark.parametrize(
    ("variant_id", "code"),
    [("fp8-hf", "MODEL_VARIANT_NOT_PUBLISHED"), ("moe", "MODEL_VARIANT_PLANNED")],
)
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/v1/models/{variant_id}/download"),
        ("delete", "/v1/models/{variant_id}/download"),
        ("post", "/v1/models/{variant_id}/load"),
    ],
)
def test_model_commands_on_non_published_variant_are_refused(
    tmp_path: Path, variant_id: str, code: str, method: str, path: str
) -> None:
    client = _catalogue_client(tmp_path)
    jobs = client.app.state.dependencies.job_store

    response = getattr(client, method)(path.format(variant_id=variant_id), headers=AUTH)

    assert response.status_code == 422
    assert response.json()["code"] == code
    assert jobs.active_model_command(variant_id, "download") is None
    assert jobs.active_model_command(variant_id, "load") is None
    assert jobs.active_model_command(variant_id, "delete") is None


@pytest.mark.parametrize(
    ("variant_id", "code"),
    [("fp8-hf", "MODEL_VARIANT_NOT_PUBLISHED"), ("moe", "MODEL_VARIANT_PLANNED")],
)
def test_put_models_default_on_non_published_variant_is_refused(
    tmp_path: Path, variant_id: str, code: str
) -> None:
    client = _catalogue_client(tmp_path)

    response = client.put("/v1/models/default", json={"variant_id": variant_id}, headers=AUTH)

    assert response.status_code == 422
    assert response.json()["code"] == code
    assert client.app.state.dependencies.job_store.get_setting("default_variant") is None


@pytest.mark.parametrize(
    ("variant_id", "code"),
    [("fp8-hf", "MODEL_VARIANT_NOT_PUBLISHED"), ("moe", "MODEL_VARIANT_PLANNED")],
)
def test_submit_with_non_published_model_header_is_refused(
    tmp_path: Path, variant_id: str, code: str
) -> None:
    client = _catalogue_client(tmp_path)

    response = _submit(client, **{"X-Unflatten-Model": variant_id})

    assert response.status_code == 422
    assert response.json()["code"] == code

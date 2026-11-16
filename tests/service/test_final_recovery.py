import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import PdfImagePlugin
from tests.service.test_api_endpoints import AUTH, _client
from tests.service.test_api_image_input import _png_bytes
from tests.service.test_model_recovery import VARIANT, ReadySandbox, _args, _jobs

from unflatten.service import api, worker_cli
from unflatten.service.models_config import load_models_config

APP = Path(__file__).resolve().parents[2]
FACTS = {"free_vram_mib": 1, "total_vram_mib": 24576, "compute_capability": (9, 0)}


def test_failed_cached_load_is_exposed_by_real_api(tmp_path, monkeypatch):
    client = _client(tmp_path)
    jobs = client.app.state.dependencies.job_store
    config = load_models_config(APP / "deploy/distribution/models.json")
    client.app.state.dependencies = replace(
        client.app.state.dependencies, models_config=config, models_cache_dir=tmp_path / "cache"
    )
    command, _ = jobs.enqueue_model_command(
        kind="load", variant_id=VARIANT, requested_by="worker", now=1
    )
    jobs.claim_next_model_command(worker_id="review", now=2)
    jobs.finish_model_command(
        command.command_id,
        worker_id="review",
        state="failed",
        error_code="MODEL_VARIANT_DOES_NOT_FIT",
        now=3,
    )
    monkeypatch.setattr(api, "is_variant_downloaded", lambda *a, **k: (True, 100))
    monkeypatch.setattr(
        "unflatten.service.model_variants.is_variant_downloaded", lambda *a, **k: (True, 100)
    )
    response = client.get("/v1/models", headers=AUTH)
    assert response.status_code == 200, response.text
    from unflatten.service.schemas import ModelsResponse

    ModelsResponse.model_validate(response.json())
    download = next(v for v in response.json()["variants"] if v["variant_id"] == VARIANT)[
        "download"
    ]
    assert download["state"] == "failed", {
        "actual_public_download": download,
        "actual_store_projection": jobs.last_model_command(VARIANT, "load"),
    }
    assert download["error_code"] == "MODEL_VARIANT_DOES_NOT_FIT"
    assert response.json()["variants"][0]["downloaded"] is True


def test_image_retry_with_same_key_after_timestamp_change_returns_original_job(tmp_path):
    client = _client(tmp_path)
    payload = _png_bytes(2, 2)
    headers = {
        **AUTH,
        "Idempotency-Key": "same-image-retry",
        "X-Unflatten-Profile": "word-core-v0",
        "X-Unflatten-Output": "prediction_ir",
    }
    files = [("page", ("page.png", payload, "image/png"))]
    first_time = time.struct_time((2026, 11, 15, 1, 0, 0, 6, 319, 0))
    later_time = time.struct_time((2026, 11, 15, 1, 0, 2, 6, 319, 0))
    with patch.object(PdfImagePlugin.time, "gmtime", return_value=first_time):
        first = client.post("/v1/conversions", files=files, headers=headers)
    with patch.object(PdfImagePlugin.time, "gmtime", return_value=later_time):
        repeated = client.post("/v1/conversions", files=files, headers=headers)
    assert first.status_code == 202, first.text
    assert repeated.status_code == 202, repeated.text
    assert repeated.json()["job_id"] == first.json()["job_id"]


def test_queued_retry_does_not_hide_cold_gpu_preflight_failure(tmp_path, monkeypatch):
    jobs = _jobs(tmp_path)
    original, _ = jobs.enqueue_model_command(
        kind="load", variant_id=VARIANT, requested_by="worker", now=1
    )
    jobs.claim_next_model_command(worker_id="old-worker", now=2)
    jobs.finish_model_command(
        original.command_id,
        worker_id="old-worker",
        state="failed",
        error_code="MODEL_DOWNLOAD_FAILED",
        now=3,
    )
    jobs.enqueue_model_command(kind="download", variant_id=VARIANT, requested_by="local", now=4)
    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "_measure_own_device_facts", lambda _: FACTS)
    downloads = []
    monkeypatch.setattr(
        "unflatten.service.model_commands._snapshot_download",
        lambda **kw: downloads.append(True) or str(tmp_path / "snapshot"),
    )
    monkeypatch.setattr("unflatten.service.model_commands.verify_bundle", lambda *a, **kw: {})
    for _ in range(2):
        with pytest.raises(SystemExit):
            worker_cli.main(_args(tmp_path))
    assert downloads == [True]
    assert jobs.active_model_command(VARIANT, "download") is None
    assert jobs.last_model_command(VARIANT, "download")["state"] == "succeeded"
    assert jobs.last_model_command(VARIANT, "load")["error_code"] == "MODEL_VARIANT_DOES_NOT_FIT", {
        "active_download_after_two_starts": jobs.active_model_command(VARIANT, "download"),
        "actual_last_load": jobs.last_model_command(VARIANT, "load"),
    }


def test_cold_auto_resolver_checks_free_memory_for_saved_default(tmp_path, monkeypatch):
    config = load_models_config(APP / "deploy/distribution/models.json")
    jobs = _jobs(tmp_path)
    jobs.set_setting("default_variant", VARIANT, now=1)
    monkeypatch.setattr(worker_cli, "_measure_own_device_facts", lambda _: FACTS)
    selected = worker_cli._resolve_auto_variant(
        SimpleNamespace(device="cuda:0"),
        config,
        SimpleNamespace(available=lambda: ("native",)),
        jobs,
    )
    assert selected is None, {"selected": selected, "saved_default": VARIANT, "measured": FACTS}


def test_cold_gpu_failure_finishes_queued_load_without_loading(tmp_path, monkeypatch):
    jobs = _jobs(tmp_path)
    jobs.enqueue_model_command(kind="load", variant_id=VARIANT, requested_by="local", now=1)
    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "_measure_own_device_facts", lambda _: FACTS)

    def forbidden(*args, **kwargs):
        raise AssertionError("cold admission attempted a GPU allocation")

    monkeypatch.setattr(worker_cli, "_load_variant", forbidden)
    with pytest.raises(SystemExit):
        worker_cli.main(_args(tmp_path))
    assert jobs.active_model_command(VARIANT, "load") is None
    assert jobs.last_model_command(VARIANT, "load")["error_code"] == "MODEL_VARIANT_DOES_NOT_FIT"


def test_cold_preflight_never_steals_live_foreign_command(tmp_path, monkeypatch):
    command_time = time.time()
    jobs = _jobs(tmp_path)
    original, _ = jobs.enqueue_model_command(
        kind="download", variant_id=VARIANT, requested_by="local", now=command_time
    )
    jobs.claim_next_model_command(worker_id="live-other", now=command_time)
    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "_measure_own_device_facts", lambda _: FACTS)
    with pytest.raises(SystemExit):
        worker_cli.main(_args(tmp_path))
    active = jobs.active_model_command(VARIANT, "download")
    assert active["state"] == "running" and active["command_id"] == original.command_id


@pytest.mark.parametrize("change", ["page_order", "output"])
def test_image_idempotency_still_rejects_changed_request(tmp_path, change):
    client = _client(tmp_path)
    files = [
        ("page", ("a.png", _png_bytes(2, 3), "image/png")),
        ("page", ("b.png", _png_bytes(3, 2), "image/png")),
    ]
    headers = {
        **AUTH,
        "Idempotency-Key": "changed-image-request",
        "X-Unflatten-Profile": "word-core-v0",
        "X-Unflatten-Output": "prediction_ir",
    }
    first = client.post("/v1/conversions", files=files, headers=headers)
    assert first.status_code == 202
    if change == "page_order":
        files.reverse()
    else:
        headers["X-Unflatten-Output"] = "diagnostic_docx"
    repeated = client.post("/v1/conversions", files=files, headers=headers)
    assert repeated.status_code == 409
    assert repeated.json()["code"] == "IDEMPOTENCY_KEY_REUSED"

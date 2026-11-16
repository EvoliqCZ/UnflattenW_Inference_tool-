"""Smoke test for `fake_worker.py` (T37): the e2e harness's job/model-command
drain loop works end to end with no GPU and no torch import, against a real
`JobStore`/`ArtifactStore` on disk."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests.service.fake_worker import (
    FAKE_HF_MANIFEST_BYTES,
    publish_fake_readiness,
    run_job,
    run_model_command,
)

from unflatten.service.artifact_store import ArtifactStore
from unflatten.service.job_store import JobStore
from unflatten.service.models_config import validate_models_config
from unflatten.service.policy import load_runtime_policy, runtime_policy_hash

_HF_MODELS_CONFIG = {
    "schema_version": "1.0",
    "default_variant": "hf-fp8",
    "variants": [
        {
            "variant_id": "hf-fp8",
            "label": "FP8 (e2e fake)",
            "precision": "fp8",
            "kv_cache": "fp8",
            "source": {
                "kind": "hf",
                "repo_id": "unflatten-e2e/fake-variant",
                "revision": "b" * 40,
                "bundle_manifest_sha256": (
                    "sha256:" + hashlib.sha256(FAKE_HF_MANIFEST_BYTES).hexdigest()
                ),
            },
        },
    ],
}


def _store(tmp_path: Path) -> JobStore:
    policy = load_runtime_policy()
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id="job_1234567890abcdef",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output="diagnostic_docx",
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:components",
        now=100.0,
    )
    return store


def test_run_job_publishes_diagnostic_docx_and_succeeds(tmp_path: Path) -> None:
    store = _store(tmp_path)
    artifacts = ArtifactStore(tmp_path / "artifacts")

    claimed = run_job(store, artifacts, worker_id="fake-worker", now=101.0)

    assert claimed is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None
    assert job.state == "succeeded"
    roles = {artifact.role for artifact in store.list_artifacts(job.job_id)}
    assert roles == {"prediction_ir", "diagnostic_docx", "source_preview", "docx_preview"}


def test_run_job_prediction_ir_only_when_not_requested(tmp_path: Path) -> None:
    policy = load_runtime_policy()
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id="job_1234567890abcdef",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output="prediction_ir",
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:components",
        now=100.0,
    )
    artifacts = ArtifactStore(tmp_path / "artifacts")

    claimed = run_job(store, artifacts, worker_id="fake-worker", now=101.0)

    assert claimed is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    roles = {artifact.role for artifact in store.list_artifacts(job.job_id)}
    assert roles == {"prediction_ir"}


def test_run_job_returns_false_when_queue_is_empty(tmp_path: Path) -> None:
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    artifacts = ArtifactStore(tmp_path / "artifacts")

    assert run_job(store, artifacts, worker_id="fake-worker", now=101.0) is False


def test_run_model_command_completes_with_fake_progress(tmp_path: Path) -> None:
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    store.enqueue_model_command(
        kind="load", variant_id="variant-a", requested_by="test", now=100.0
    )

    claimed = run_model_command(store, worker_id="fake-worker")

    assert claimed is True
    command = store.last_model_command("variant-a", "load")
    assert command is not None and command["state"] == "succeeded"


def test_run_model_command_materializes_hf_snapshot_on_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design §7.6: a 'succeeded' download must leave behind what
    `is_variant_downloaded` (`scan_cache_dir`) looks for -- otherwise `GET
    /v1/models` reports the variant absent again right after the fake
    worker's own progress said it finished, and the UI never offers Load."""
    from unflatten.service.model_variants import is_variant_downloaded

    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps(_HF_MODELS_CONFIG))
    hf_home = tmp_path / "hf-home"
    monkeypatch.setenv("UNFLATTEN_MODELS_CONFIG", str(config_path))
    monkeypatch.setenv("HF_HOME", str(hf_home))

    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    store.enqueue_model_command(
        kind="download", variant_id="hf-fp8", requested_by="test", now=100.0
    )

    claimed = run_model_command(store, worker_id="fake-worker")

    assert claimed is True
    command = store.last_model_command("hf-fp8", "download")
    assert command is not None and command["state"] == "succeeded"
    variant = validate_models_config(_HF_MODELS_CONFIG)["variants"][0]
    downloaded, _size = is_variant_downloaded(variant, cache_dir=hf_home / "hub")
    assert downloaded is True


def test_run_model_command_skips_materialization_without_models_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No `UNFLATTEN_MODELS_CONFIG` set (every e2e backend except the models
    one) -- a `load`/`download` command must still succeed exactly as before
    T46, with no attempt to resolve a variant that does not exist."""
    monkeypatch.delenv("UNFLATTEN_MODELS_CONFIG", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    store.enqueue_model_command(
        kind="load", variant_id="variant-a", requested_by="test", now=100.0
    )

    claimed = run_model_command(store, worker_id="fake-worker")

    assert claimed is True
    command = store.last_model_command("variant-a", "load")
    assert command is not None and command["state"] == "succeeded"


def test_run_model_command_marks_variant_resident_on_successful_load(tmp_path: Path) -> None:
    """A `load` command's success must flip `model_state().resident_variant`
    (`GET /v1/models`'s `resident`/`On GPU` badge) -- the fake progress alone
    was not enough: nothing else in this module ever published a resident
    variant, so the e2e UI's `On GPU` badge never appeared after a real
    Load click (T46)."""
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    writer = publish_fake_readiness(store, worker_id="fake-worker")
    store.enqueue_model_command(
        kind="load", variant_id="variant-a", requested_by="test", now=100.0
    )

    claimed = run_model_command(store, worker_id="fake-worker", writer=writer)

    assert claimed is True
    model_state = store.model_state()
    assert model_state is not None
    assert model_state[0] == "variant-a"


def test_run_model_command_returns_false_when_queue_is_empty(tmp_path: Path) -> None:
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()

    assert run_model_command(store, worker_id="fake-worker") is False


def test_publish_fake_readiness_reports_no_gpu_device_facts(tmp_path: Path) -> None:
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()

    publish_fake_readiness(store, worker_id="fake-worker")

    status = store.readiness_status()
    assert status is not None and status[0] is True  # worker_ready
    model_state = store.model_state()
    assert model_state is not None
    _resident_variant, _free, total_vram_mib, compute_capability = model_state
    assert total_vram_mib == 12288.0
    assert compute_capability == (8, 9)


def test_fake_worker_module_never_imports_torch() -> None:
    """Design constraint (Sec 7.6): the e2e worker double stays GPU/torch-free.
    Checked in a fresh subprocess -- `sys.modules` in-process is contaminated by
    whichever torch-importing test module pytest collected earlier in this run,
    which would make an in-process assertion pass or fail on suite order, not on
    what `fake_worker.py` itself imports."""
    script = (
        "import sys; import tests.service.fake_worker; "
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parent.parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", result.stderr

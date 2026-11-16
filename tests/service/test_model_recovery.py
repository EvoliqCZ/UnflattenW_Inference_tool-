from __future__ import annotations

import gc
import io
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.service.test_worker_cli import _fake_loaded_variant

from unflatten.service import model_commands, worker_cli
from unflatten.service.job_store import JobStore
from unflatten.service.model_variants import resolve_default_variant, variant_download_state
from unflatten.service.models_config import load_models_config
from unflatten.service.policy import load_runtime_policy

ROOT = Path(__file__).resolve().parents[2]
VARIANT = "dense-1.0-bf16"


class ReadySandbox:
    def __init__(self, **kwargs):
        pass

    def require_ready(self):
        pass

    def readiness(self):
        return {"ready": True, "rootless": True, "runtime_kind": "podman"}


def _jobs(tmp_path):
    policy = load_runtime_policy()
    jobs = JobStore(
        tmp_path / "jobs.sqlite3",
        **{
            k: policy[k]
            for k in (
                "lease_duration_seconds",
                "max_attempts",
                "global_active_jobs",
                "active_jobs_per_principal",
            )
        },
    )
    jobs.initialize()
    return jobs


def _args(tmp_path):
    return [
        "--job-db",
        str(tmp_path / "jobs.sqlite3"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--work-root",
        str(tmp_path / "work"),
        "--sandbox-image",
        "sha256:" + "a" * 64,
        "--models-config",
        str(ROOT / "deploy/distribution/models.json"),
        "--model-id",
        "Qwen/Qwen3-VL-4B-Instruct",
        "--revision",
        "deadbeef",
        "--hf-cache-dir",
        str(tmp_path / "cache"),
    ]


@pytest.mark.parametrize("running", [False, True, "racing"])
def test_bootstrap_resumes_queued_download_before_automatic_load(tmp_path, monkeypatch, running):
    jobs = _jobs(tmp_path)
    if running != "racing":
        jobs.enqueue_model_command(kind="download", variant_id=VARIANT, requested_by="local", now=1)
    if running is True:
        jobs.claim_next_model_command(worker_id="dead-worker", now=2)
    if running == "racing":
        enqueue = JobStore.enqueue_model_command
        inserted = []

        def race(self, **kwargs):
            if kwargs["kind"] == "load" and not inserted:
                inserted.append(True)
                enqueue(self, kind="download", variant_id=VARIANT, requested_by="local", now=1)
            return enqueue(self, **kwargs)

        monkeypatch.setattr(JobStore, "enqueue_model_command", race)
    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(
        worker_cli,
        "_measure_own_device_facts",
        lambda _: {
            "free_vram_mib": 100000,
            "total_vram_mib": 100000,
            "compute_capability": (12, 1),
        },
    )
    monkeypatch.setattr(worker_cli, "_publish_model_state", lambda *a, **k: None)
    downloaded = []
    loaded = []
    constructed = []

    def download(**kwargs):
        downloaded.append(True)
        return str(tmp_path / "snapshot")

    def load(*args, **kwargs):
        assert downloaded
        loaded.append(True)
        return _fake_loaded_variant(VARIANT)

    class IdleWorker:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def run_once(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(model_commands, "_snapshot_download", download)
    monkeypatch.setattr(model_commands, "verify_bundle", lambda *a, **k: {})
    monkeypatch.setattr(
        model_commands, "is_variant_downloaded", lambda *a, **k: (bool(downloaded), 1)
    )
    monkeypatch.setattr(worker_cli, "_load_variant", load)
    monkeypatch.setattr(worker_cli, "DiagnosticWorker", IdleWorker)
    assert worker_cli.main(_args(tmp_path)) == 0
    assert len(downloaded) == len(loaded) == len(constructed) == 1
    assert jobs.last_model_command(VARIANT, "download")["state"] == "succeeded"
    assert jobs.last_model_command(VARIANT, "load")["state"] == "succeeded"


@pytest.mark.parametrize("mib", [None, 8192, 12288])
def test_bootstrap_publishes_device_and_durable_fit_failure_before_exit(tmp_path, monkeypatch, mib):
    jobs = _jobs(tmp_path)
    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    facts = (
        None
        if mib is None
        else {"free_vram_mib": mib, "total_vram_mib": mib, "compute_capability": (8, 9)}
    )
    monkeypatch.setattr(worker_cli, "_measure_own_device_facts", lambda _: facts)
    with pytest.raises(SystemExit):
        worker_cli.main(_args(tmp_path))
    command = jobs.last_model_command(VARIANT, "load")
    assert command is not None and command["state"] == "failed"
    assert command["error_code"] == (
        "MODEL_ENGINE_UNAVAILABLE" if mib is None else "MODEL_VARIANT_DOES_NOT_FIT"
    )
    assert (jobs.model_state() is None) == (mib is None)
    if mib is not None:
        assert jobs.model_state()[2] == mib
    assert jobs.readiness_status()[0] is False


def test_cold_auto_selection_uses_free_memory_instead_of_card_capacity():
    config = load_models_config(ROOT / "deploy/distribution/models.json")
    selected, _ = resolve_default_variant(
        config,
        user_default=None,
        model_state={"free_vram_mib": 1, "total_vram_mib": 24576, "compute_capability": (9, 0)},
    )
    assert selected is None


def test_complete_snapshot_does_not_hide_failed_gpu_load(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "unflatten.service.model_variants.is_variant_downloaded", lambda *a, **k: (True, 100)
    )
    variant = load_models_config(ROOT / "deploy/distribution/models.json")["variants"][0]
    result = variant_download_state(
        variant,
        cache_dir=tmp_path,
        active_command=None,
        last_command={
            "kind": "load",
            "state": "failed",
            "error_code": "MODEL_VARIANT_DOES_NOT_FIT",
            "command_id": "cmd_failed",
        },
    )
    assert result["state"] == "failed"
    assert result["error_code"] == "MODEL_VARIANT_DOES_NOT_FIT"


class Model:
    pass


def _loaded(identifier):
    return worker_cli.LoadedVariant(
        Model(),
        True,
        {"component_lock_hash": "sha256:lock"},
        worker_cli.SelectedVariant(identifier, "bf16", "bf16", None),
    )


def _provider(monkeypatch):
    monkeypatch.setattr(worker_cli, "_publish_model_state", lambda *a, **k: None)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    return worker_cli._SwitchingModelProvider(
        args=SimpleNamespace(device="cpu"),
        policy={},
        jobs=object(),
        readiness_writer=object(),
        initial=_loaded("old"),
    )


def test_switch_drops_every_provider_owner_before_loading_replacement(monkeypatch):
    provider = _provider(monkeypatch)
    previous = weakref.ref(provider.ensure_loaded(None))

    def replacement(*args, **kwargs):
        gc.collect()
        assert previous() is None
        return _loaded("new")

    monkeypatch.setattr(worker_cli, "_load_variant", replacement)
    assert provider.ensure_loaded("new") is provider._loaded.model


def test_failed_switch_can_reload_without_attribute_errors(monkeypatch):
    provider = _provider(monkeypatch)

    def failed(*args, **kwargs):
        raise RuntimeError("synthetic OOM")

    monkeypatch.setattr(worker_cli, "_load_variant", failed)
    with pytest.raises(RuntimeError, match="synthetic OOM"):
        provider.ensure_loaded("new")
    monkeypatch.setattr(worker_cli, "_load_variant", lambda *a, **k: _loaded("old"))
    assert provider.ensure_loaded(None) is provider._loaded.model


def test_progress_ignores_file_counts_and_never_regresses_byte_counts():
    progress = []
    bar_type = model_commands.CommandProgress.bound(
        lambda done, total: progress.append((done, total)), min_interval_seconds=0
    )
    byte_bar = bar_type(total=2000000, unit="B", file=io.StringIO())
    byte_bar.update(1000000)
    byte_bar.close()
    file_bar = bar_type(total=2, unit="it", file=io.StringIO())
    file_bar.update(1)
    file_bar.close()
    reconstructed_bar = bar_type(total=2000000, unit="B", file=io.StringIO())
    reconstructed_bar.close()
    assert progress and all(total == 2000000 for _, total in progress)
    assert [done for done, _ in progress] == sorted(done for done, _ in progress)
    assert progress[-1][0] == 1000000


def test_new_worker_does_not_inherit_stale_resident_gpu_facts(tmp_path):
    jobs = _jobs(tmp_path)
    writer = jobs.claim_readiness_writer(worker_id="old", now=100)
    jobs.publish_model_state(
        writer,
        resident_variant=VARIANT,
        free_vram_mib=10,
        total_vram_mib=100000,
        compute_capability=(12, 1),
        now=101,
    )
    jobs.claim_readiness_writer(worker_id="replacement", now=102)
    assert jobs.model_state() is None


@pytest.mark.parametrize("after_job", [False, True])
def test_idle_worker_does_not_keep_model_alive_during_model_command_switch(
    tmp_path, monkeypatch, after_job
):
    from tests.service.test_worker import Candidate, Sandbox, _source, _store

    from unflatten.service.artifact_store import ArtifactStore
    from unflatten.service.model_runtime import DiagnosticModelRuntime
    from unflatten.service.worker import DiagnosticWorker

    monkeypatch.setattr(worker_cli, "_publish_model_state", lambda *a, **k: None)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    calls = []

    def variant(identifier):
        return worker_cli.LoadedVariant(
            DiagnosticModelRuntime(Candidate(calls)),
            True,
            {"component_lock_hash": "sha256:components"},
            worker_cli.SelectedVariant(identifier, "bf16", "bf16", None),
        )

    provider = worker_cli._SwitchingModelProvider(
        args=SimpleNamespace(device="cpu"),
        policy={},
        jobs=object(),
        readiness_writer=object(),
        initial=variant("old"),
    )
    jobs = _store(tmp_path)
    worker = DiagnosticWorker(
        worker_id="worker",
        jobs=jobs,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=Sandbox(calls),
        model_provider=provider,
        source_resolver=lambda _: _source(tmp_path),
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )
    if after_job:
        assert worker.run_once() is True
        assert jobs.get_job("job_1234567890abcdef", "principal_alpha").state == "succeeded"
    previous = weakref.ref(provider.ensure_loaded(None).candidate_runtime)

    def replacement(*args, **kwargs):
        gc.collect()
        assert previous() is None, "idle document worker retained the previous GPU model"
        return variant("new")

    monkeypatch.setattr(worker_cli, "_load_variant", replacement)
    provider.ensure_loaded("new")

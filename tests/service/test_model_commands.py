"""T27 (design §4.2/§4.3): `ModelCommandExecutor` and `CommandProgress`.

`snapshot_download`/`verify_bundle`/`delete_variant_snapshot`/`is_variant_downloaded`
are monkeypatched module-level names in `model_commands` -- no network, no torch,
no GPU."""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from unflatten.service import model_commands as mc
from unflatten.service.job_store import JobStore
from unflatten.service.model_bundle import ModelBundleInvalid

_SHA = "sha256:" + "a" * 64
_REVISION = "b" * 40


def make_store(tmp_path: Path) -> JobStore:
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=10,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    return store


def _hf_variant(variant_id: str = "remote-int4") -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "label": "INT4 (remote)",
        "precision": "int4",
        "kv_cache": "bf16",
        "source": {
            "kind": "hf",
            "repo_id": "unflatten/v15b",
            "revision": _REVISION,
            "bundle_manifest_sha256": _SHA,
        },
    }


def _bundle_variant(variant_id: str = "local-fp8") -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "label": "FP8 (local)",
        "precision": "fp8",
        "kv_cache": "fp8",
        "bundle": "/data/model-bundles/v15b-fp8-fp8",
    }


class _FakeModelProvider:
    def __init__(self, *, side_effect: Callable[[str | None], Any] | None = None) -> None:
        self.calls: list[str | None] = []
        self._side_effect = side_effect

    def ensure_loaded(self, variant_id: str | None) -> Any:
        self.calls.append(variant_id)
        if self._side_effect is not None:
            return self._side_effect(variant_id)
        return object()


def _executor(
    jobs: JobStore,
    *,
    variants: dict[str, dict[str, Any]],
    model_provider: _FakeModelProvider | None = None,
    cache_dir: Path,
    worker_id: str = "worker-1",
    heartbeat_interval_seconds: float = 100.0,
    clock: Callable[[], float] | None = None,
    publish_readiness: Callable[[float], None] | None = None,
) -> mc.ModelCommandExecutor:
    kwargs: dict[str, Any] = dict(
        jobs=jobs,
        worker_id=worker_id,
        cache_dir=cache_dir,
        variants=variants,
        model_provider=model_provider or _FakeModelProvider(),
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    if clock is not None:
        kwargs["clock"] = clock
    if publish_readiness is not None:
        kwargs["publish_readiness"] = publish_readiness
    return mc.ModelCommandExecutor(**kwargs)


def test_run_once_returns_false_when_nothing_queued(tmp_path: Path) -> None:
    jobs = make_store(tmp_path)
    executor = _executor(jobs, variants={}, cache_dir=tmp_path)

    assert executor.run_once() is False


@pytest.mark.parametrize("failure", ["os", "transport", "unauthorized"])
@pytest.mark.parametrize("kind", ["download", "load"])
def test_remote_download_failure_is_terminal_and_does_not_log_credentials(
    tmp_path,
    monkeypatch,
    caplog,
    kind,
    failure,
):
    jobs = make_store(tmp_path)
    variant = _hf_variant()
    provider = _FakeModelProvider()
    monkeypatch.setattr(mc, "is_variant_downloaded", lambda *a, **k: (False, None))

    def fail_download(**kwargs):
        import httpx
        from huggingface_hub.errors import HfHubHTTPError

        message = "private URL/token hf_secret_do_not_echo"
        if failure == "transport":
            raise httpx.ConnectError(message)
        if failure == "unauthorized":
            raise HfHubHTTPError(message, response=httpx.Response(
                401, request=httpx.Request("GET", "https://huggingface.co/test")
            ))
        raise OSError(message)

    monkeypatch.setattr(mc, "_snapshot_download", fail_download)
    jobs.enqueue_model_command(
        kind=kind, variant_id=variant["variant_id"], requested_by="p", now=0.0
    )
    executor = _executor(
        jobs, variants={variant["variant_id"]: variant}, model_provider=provider, cache_dir=tmp_path
    )
    with caplog.at_level(logging.WARNING):
        assert executor.run_once() is True
    last = jobs.last_model_command(variant["variant_id"], kind)
    assert last["state"] == "failed"
    assert last["error_code"] == "MODEL_DOWNLOAD_FAILED"
    assert "hf_secret_do_not_echo" not in caplog.text
    assert provider.calls == []


def test_download_local_bundle_variant_succeeds_without_network(tmp_path: Path) -> None:
    jobs = make_store(tmp_path)
    variant = _bundle_variant()
    jobs.enqueue_model_command(
        kind="download", variant_id=variant["variant_id"], requested_by="p", now=0.0
    )
    executor = _executor(jobs, variants={variant["variant_id"]: variant}, cache_dir=tmp_path)

    assert executor.run_once() is True
    last = jobs.last_model_command(variant["variant_id"], "download")
    assert last["state"] == "succeeded"


@pytest.mark.parametrize(
    ("availability", "code"),
    [("unpublished", "MODEL_VARIANT_NOT_PUBLISHED"), ("planned", "MODEL_VARIANT_PLANNED")],
)
@pytest.mark.parametrize("kind", ["download", "load", "delete"])
def test_command_on_non_published_variant_fails_without_touching_network_or_gpu(
    tmp_path: Path, availability: str, code: str, kind: str
) -> None:
    """Second safety net behind the API's 422: a command row that reached the
    worker anyway (older API, manual SQL) fails with the same public code --
    it never reaches `snapshot_download`, the cache, or the model provider."""
    jobs = make_store(tmp_path)
    variant = {
        "variant_id": "not-yet",
        "label": "not yet",
        "precision": "fp8" if availability == "unpublished" else None,
        "kv_cache": "fp8" if availability == "unpublished" else None,
        "engine": "native",
        "availability": availability,
    }
    jobs.enqueue_model_command(kind=kind, variant_id="not-yet", requested_by="p", now=0.0)
    provider = _FakeModelProvider()
    executor = _executor(
        jobs, variants={"not-yet": variant}, model_provider=provider, cache_dir=tmp_path
    )

    assert executor.run_once() is True
    last = jobs.last_model_command("not-yet", kind)
    assert last["state"] == "failed"
    assert last["error_code"] == code
    assert provider.calls == []


def test_load_undownloaded_variant_runs_download_verify_load_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = make_store(tmp_path)
    variant = _hf_variant()
    variant_id = variant["variant_id"]
    seen: list[tuple[str, str | None]] = []

    def fake_is_downloaded(_variant: Any, *, cache_dir: Path) -> tuple[bool, int | None]:
        return False, None

    def fake_snapshot_download(**kwargs: Any) -> str:
        seen.append(("download_call", jobs.active_model_command(variant_id, "load")["phase"]))
        tqdm_class = kwargs["tqdm_class"]
        bar = tqdm_class(total=10, unit="B", file=io.StringIO())
        bar.update(10)
        bar.close()
        return str(tmp_path / "snapshot")

    def fake_verify_bundle(bundle_dir: Path, **kwargs: Any) -> dict:
        seen.append(("verify_call", jobs.active_model_command(variant_id, "load")["phase"]))
        assert bundle_dir == tmp_path / "snapshot"
        return {}

    provider = _FakeModelProvider(
        side_effect=lambda vid: seen.append(
            ("load_call", jobs.active_model_command(variant_id, "load")["phase"])
        )
    )
    monkeypatch.setattr(mc, "is_variant_downloaded", fake_is_downloaded)
    monkeypatch.setattr(mc, "_snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(mc, "verify_bundle", fake_verify_bundle)

    jobs.enqueue_model_command(kind="load", variant_id=variant_id, requested_by="p", now=0.0)
    executor = _executor(
        jobs, variants={variant_id: variant}, model_provider=provider, cache_dir=tmp_path
    )

    assert executor.run_once() is True
    assert [call for call, _phase in seen] == ["download_call", "verify_call", "load_call"]
    assert [phase for _call, phase in seen] == ["downloading", "verifying", "loading"]
    assert jobs.last_model_command(variant_id, "load")["state"] == "succeeded"
    assert provider.calls == [variant_id]


def test_progress_throttled_to_one_write_per_second_plus_final_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = make_store(tmp_path)
    variant = _hf_variant()
    variant_id = variant["variant_id"]

    def fake_snapshot_download(**kwargs: Any) -> str:
        tqdm_class = kwargs["tqdm_class"]
        bar = tqdm_class(total=100, unit="B", file=io.StringIO())
        bar.update(10)  # t=0.1, within 1s of construction -- no write
        bar.update(10)  # t=0.5, still within 1s -- no write
        bar.update(10)  # t=1.2, >= 1s since last write -- writes 30/100
        bar.update(5)  # t=1.3, just wrote -- no write
        bar.close()  # always writes once more: 35/100
        return str(tmp_path / "snapshot")

    def fake_verify_bundle(bundle_dir: Path, **kwargs: Any) -> dict:
        return {}

    monkeypatch.setattr(mc, "_snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(mc, "verify_bundle", fake_verify_bundle)

    ticks = iter([0.0, 0.1, 0.5, 1.2, 1.3, 1.3])

    def fake_monotonic() -> float:
        return next(ticks, 1.3)

    jobs.enqueue_model_command(kind="download", variant_id=variant_id, requested_by="p", now=0.0)
    executor = mc.ModelCommandExecutor(
        jobs=jobs,
        worker_id="worker-1",
        cache_dir=tmp_path,
        variants={variant_id: variant},
        model_provider=_FakeModelProvider(),
        heartbeat_interval_seconds=100.0,
        progress_clock=fake_monotonic,
    )

    assert executor.run_once() is True
    last = jobs.last_model_command(variant_id, "download")
    assert last["state"] == "succeeded"
    assert last["bytes_done"] == 35
    assert last["bytes_total"] == 100


def test_download_verify_bundle_failure_marks_unverified_and_deletes_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = make_store(tmp_path)
    variant = _hf_variant()
    variant_id = variant["variant_id"]
    deleted: list[str] = []

    monkeypatch.setattr(mc, "_snapshot_download", lambda **kwargs: str(tmp_path / "snapshot"))

    def fake_verify_bundle(bundle_dir: Path, **kwargs: Any) -> dict:
        raise ModelBundleInvalid("manifest mismatch")

    monkeypatch.setattr(mc, "verify_bundle", fake_verify_bundle)
    monkeypatch.setattr(
        mc, "delete_variant_snapshot", lambda v, *, cache_dir: deleted.append(v["variant_id"])
    )

    jobs.enqueue_model_command(kind="download", variant_id=variant_id, requested_by="p", now=0.0)
    executor = _executor(jobs, variants={variant_id: variant}, cache_dir=tmp_path)

    assert executor.run_once() is True
    last = jobs.last_model_command(variant_id, "download")
    assert last["state"] == "failed"
    assert last["error_code"] == "MODEL_CHECKPOINT_UNVERIFIED"
    assert deleted == [variant_id]


def test_delete_resident_variant_fails_with_conflict(tmp_path: Path) -> None:
    jobs = make_store(tmp_path)
    variant = _bundle_variant()
    variant_id = variant["variant_id"]
    writer = jobs.claim_readiness_writer(worker_id="worker-1", now=0.0)
    jobs.publish_model_state(
        writer,
        resident_variant=variant_id,
        free_vram_mib=1000.0,
        total_vram_mib=12000.0,
        compute_capability=(8, 9),
        now=1.0,
    )
    jobs.enqueue_model_command(kind="delete", variant_id=variant_id, requested_by="p", now=2.0)
    executor = _executor(jobs, variants={variant_id: variant}, cache_dir=tmp_path)

    assert executor.run_once() is True
    last = jobs.last_model_command(variant_id, "delete")
    assert last["state"] == "failed"
    assert last["error_code"] == "MODEL_COMMAND_CONFLICT"


def test_delete_non_resident_variant_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = make_store(tmp_path)
    variant = _hf_variant()
    variant_id = variant["variant_id"]
    deleted: list[str] = []
    monkeypatch.setattr(
        mc, "delete_variant_snapshot", lambda v, *, cache_dir: deleted.append(v["variant_id"])
    )
    jobs.enqueue_model_command(kind="delete", variant_id=variant_id, requested_by="p", now=0.0)
    executor = _executor(jobs, variants={variant_id: variant}, cache_dir=tmp_path)

    assert executor.run_once() is True
    last = jobs.last_model_command(variant_id, "delete")
    assert last["state"] == "succeeded"
    assert deleted == [variant_id]


def test_load_oom_returns_model_variant_does_not_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = make_store(tmp_path)
    variant = _bundle_variant()
    variant_id = variant["variant_id"]

    class _FakeOutOfMemoryError(RuntimeError):
        pass

    class _FakeCuda:
        OutOfMemoryError = _FakeOutOfMemoryError

    class _FakeTorch:
        cuda = _FakeCuda

    monkeypatch.setitem(__import__("sys").modules, "torch", _FakeTorch)

    provider = _FakeModelProvider(
        side_effect=lambda vid: (_ for _ in ()).throw(_FakeOutOfMemoryError("no vram"))
    )
    jobs.enqueue_model_command(kind="load", variant_id=variant_id, requested_by="p", now=0.0)
    executor = _executor(
        jobs, variants={variant_id: variant}, model_provider=provider, cache_dir=tmp_path
    )

    assert executor.run_once() is True
    last = jobs.last_model_command(variant_id, "load")
    assert last["state"] == "failed"
    assert last["error_code"] == "MODEL_VARIANT_DOES_NOT_FIT"


def test_load_engine_unavailable_returns_model_engine_unavailable(tmp_path: Path) -> None:
    jobs = make_store(tmp_path)
    variant = _bundle_variant()
    variant_id = variant["variant_id"]

    def side_effect(vid: str | None) -> Any:
        raise mc.EngineUnavailable(f"engine for {vid} is unavailable")

    provider = _FakeModelProvider(side_effect=side_effect)
    jobs.enqueue_model_command(kind="load", variant_id=variant_id, requested_by="p", now=0.0)
    executor = _executor(
        jobs, variants={variant_id: variant}, model_provider=provider, cache_dir=tmp_path
    )

    assert executor.run_once() is True
    last = jobs.last_model_command(variant_id, "load")
    assert last["state"] == "failed"
    assert last["error_code"] == "MODEL_ENGINE_UNAVAILABLE"


def test_load_bundle_invalid_returns_model_checkpoint_unverified(tmp_path: Path) -> None:
    jobs = make_store(tmp_path)
    variant = _bundle_variant()
    variant_id = variant["variant_id"]

    def side_effect(vid: str | None) -> Any:
        raise ModelBundleInvalid("checksum mismatch")

    provider = _FakeModelProvider(side_effect=side_effect)
    jobs.enqueue_model_command(kind="load", variant_id=variant_id, requested_by="p", now=0.0)
    executor = _executor(
        jobs, variants={variant_id: variant}, model_provider=provider, cache_dir=tmp_path
    )

    assert executor.run_once() is True
    last = jobs.last_model_command(variant_id, "load")
    assert last["state"] == "failed"
    assert last["error_code"] == "MODEL_CHECKPOINT_UNVERIFIED"


def test_load_unexpected_runtime_error_returns_internal_error(tmp_path: Path) -> None:
    jobs = make_store(tmp_path)
    variant = _bundle_variant()
    variant_id = variant["variant_id"]

    def side_effect(vid: str | None) -> Any:
        raise RuntimeError("boom")

    provider = _FakeModelProvider(side_effect=side_effect)
    jobs.enqueue_model_command(kind="load", variant_id=variant_id, requested_by="p", now=0.0)
    executor = _executor(
        jobs, variants={variant_id: variant}, model_provider=provider, cache_dir=tmp_path
    )

    assert executor.run_once() is True
    last = jobs.last_model_command(variant_id, "load")
    assert last["state"] == "failed"
    assert last["error_code"] == "INTERNAL_ERROR"


def test_load_failure_logs_only_exception_type(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    jobs = make_store(tmp_path)
    variant = _bundle_variant()
    variant_id = variant["variant_id"]

    def side_effect(vid: str | None) -> Any:
        raise RuntimeError("/secret/path/should/not/leak")

    provider = _FakeModelProvider(side_effect=side_effect)
    jobs.enqueue_model_command(kind="load", variant_id=variant_id, requested_by="p", now=0.0)
    executor = _executor(
        jobs, variants={variant_id: variant}, model_provider=provider, cache_dir=tmp_path
    )

    with caplog.at_level(logging.WARNING):
        executor.run_once()

    assert "RuntimeError" in caplog.text
    assert "/secret/path" not in caplog.text


def test_heartbeat_thread_keeps_readiness_fresh_during_long_command(
    tmp_path: Path,
) -> None:
    jobs = make_store(tmp_path)
    variant = _bundle_variant()
    variant_id = variant["variant_id"]
    publish_calls: list[float] = []

    def slow_ensure_loaded(vid: str | None) -> Any:
        import time as _time

        _time.sleep(0.3)
        return object()

    provider = _FakeModelProvider(side_effect=slow_ensure_loaded)
    jobs.enqueue_model_command(kind="load", variant_id=variant_id, requested_by="p", now=0.0)
    executor = _executor(
        jobs,
        variants={variant_id: variant},
        model_provider=provider,
        cache_dir=tmp_path,
        heartbeat_interval_seconds=0.02,
        publish_readiness=publish_calls.append,
    )

    assert executor.run_once() is True
    assert len(publish_calls) >= 2
    last = jobs.last_model_command(variant_id, "load")
    assert last["state"] == "succeeded"


def test_finish_lost_fence_logs_and_discards(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    jobs = make_store(tmp_path)
    variant = _bundle_variant()
    variant_id = variant["variant_id"]

    def side_effect(vid: str | None) -> Any:
        # Simulate another worker reclaiming the command (stale heartbeat)
        # while this one is still "running" the load. `stale_after=0.0`
        # with a far-future `now` treats the real claim as stale regardless
        # of the executor's own real-clock `heartbeat_at`.
        import time as _time

        far_future = _time.time() + 1_000_000.0
        jobs.requeue_stale_model_commands(now=far_future, stale_after=0.0)
        jobs.claim_next_model_command(worker_id="worker-2", now=far_future + 1.0)
        return object()

    provider = _FakeModelProvider(side_effect=side_effect)
    jobs.enqueue_model_command(kind="load", variant_id=variant_id, requested_by="p", now=0.0)
    executor = _executor(
        jobs, variants={variant_id: variant}, model_provider=provider, cache_dir=tmp_path
    )

    with caplog.at_level(logging.INFO):
        result = executor.run_once()

    assert result is True
    assert "model_command_fence_lost" in caplog.text
    # worker-2's claim is untouched by worker-1's discarded finish.
    active = jobs.active_model_command(variant_id, "load")
    assert active["state"] == "running"


def test_restart_mid_download_requeue_keeps_bytes_then_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = make_store(tmp_path)
    variant = _hf_variant()
    variant_id = variant["variant_id"]
    record, _created = jobs.enqueue_model_command(
        kind="download", variant_id=variant_id, requested_by="p", now=0.0
    )
    jobs.claim_next_model_command(worker_id="worker-1", now=1.0)
    jobs.update_model_command_progress(
        record.command_id,
        worker_id="worker-1",
        phase="downloading",
        bytes_done=500,
        bytes_total=1000,
        now=2.0,
    )

    requeued = jobs.requeue_stale_model_commands(now=100.0, stale_after=10.0)
    assert requeued == 1
    stalled = jobs.active_model_command(variant_id, "download")
    assert stalled["state"] == "queued"
    assert stalled["bytes_done"] == 500

    monkeypatch.setattr(mc, "_snapshot_download", lambda **kwargs: str(tmp_path / "snapshot"))
    monkeypatch.setattr(mc, "verify_bundle", lambda bundle_dir, **kwargs: {})
    executor = _executor(
        jobs, variants={variant_id: variant}, cache_dir=tmp_path, worker_id="worker-2"
    )

    assert executor.run_once() is True
    last = jobs.last_model_command(variant_id, "download")
    assert last["state"] == "succeeded"

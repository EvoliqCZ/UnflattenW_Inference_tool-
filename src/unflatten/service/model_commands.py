"""§4.2/§4.3 `model_commands` executor (T27): claims one queued row from
`ModelCommandMixin` (T14) between worker jobs and runs `download` / `load` /
`delete` to completion, publishing byte progress and heartbeats along the
way.

Deliberately import-light for GPU: this module never imports `torch` at
module scope (`_is_oom_error` imports it lazily, only to classify an
exception already raised by the caller-supplied `ModelProvider`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from unflatten.service.job_store_types import FenceRejected, ModelCommandRecord
from unflatten.service.model_bundle import ModelBundleInvalid, verify_bundle
from unflatten.service.model_variants import delete_variant_snapshot, is_variant_downloaded
from unflatten.service.models_config import PUBLISHED

LOGGER = logging.getLogger(__name__)


class EngineUnavailable(RuntimeError):
    """Raised by a `ModelProvider.ensure_loaded` when the requested variant
    names an inference engine that is not available on this worker (design
    §10's "second failsafe" -- the API already refuses to enqueue such a
    `load`; a worker whose environment changed after startup must still fail
    the command instead of silently loading the wrong engine)."""


class ModelProvider(Protocol):
    """Narrow seam onto `worker_cli._SwitchingModelProvider` (T22 fold):
    the executor never touches a GPU or an `InferenceEngine` itself."""

    def ensure_loaded(self, variant_id: str | None) -> Any: ...


class JobsProtocol(Protocol):
    """The `ModelCommandMixin` + readiness surface this executor needs."""

    def claim_next_model_command(
        self, *, worker_id: str, now: float, kinds: tuple[str, ...] | None = None
    ) -> ModelCommandRecord | None: ...

    def heartbeat_model_command(self, command_id: str, *, worker_id: str, now: float) -> None: ...

    def update_model_command_progress(
        self,
        command_id: str,
        *,
        worker_id: str,
        phase: str,
        now: float,
        bytes_done: int | None = None,
        bytes_total: int | None = None,
    ) -> None: ...

    def finish_model_command(
        self,
        command_id: str,
        *,
        worker_id: str,
        state: str,
        now: float,
        error_code: str | None = None,
    ) -> bool: ...

    def model_state(self) -> tuple[str | None, float, float, tuple[int, int]] | None: ...


def _snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _is_oom_error(exc: BaseException) -> bool:
    """Whether `exc` is a `torch.cuda.OutOfMemoryError` -- checked via a
    lazy import (never at module scope) so this module stays importable
    without `torch`."""
    try:
        import torch
    except ImportError:
        return False
    return isinstance(exc, torch.cuda.OutOfMemoryError)


class CommandProgress(tqdm):
    """§4.3: `snapshot_download(tqdm_class=...)`'s main byte progress bar.

    `huggingface_hub` constructs this class itself (`tqdm_class(total=...,
    ...)`), so a per-download write callback cannot be a constructor
    argument -- `bound()` returns a fresh subclass with the callback baked
    in as a class attribute. `update(n)` relies on base `tqdm` to accumulate
    `self.n`, and writes `bytes_done=self.n, bytes_total=self.total` at most
    once per `_min_interval_seconds` (monotonic); `close()` always writes
    once more regardless of the interval, so the terminal byte count is
    never lost to throttling.
    """

    _write: Callable[[int | None, int | None], None] | None = None
    _min_interval_seconds: float = 1.0
    _monotonic: Callable[[], float] = staticmethod(time.monotonic)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_write_at = self._monotonic()

    def update(self, n: float = 1) -> bool | None:
        result = super().update(n)
        if self._write is not None and getattr(self, "unit", None) == "B":
            now = self._monotonic()
            if now - self._last_write_at >= self._min_interval_seconds:
                self._last_write_at = now
                self._write(self.n, self.total)
        return result

    def close(self) -> None:
        if self._write is not None and getattr(self, "unit", None) == "B":
            self._write(self.n, self.total)
        super().close()

    @classmethod
    def bound(
        cls,
        write: Callable[[int | None, int | None], None],
        *,
        min_interval_seconds: float = 1.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> type[CommandProgress]:
        maximum_done = 0
        maximum_total: int | None = None
        lock = threading.Lock()

        def monotonic_write(done, total):
            nonlocal maximum_done, maximum_total
            with lock:
                maximum_done = max(maximum_done, int(done or 0))
                if total is not None:
                    maximum_total = max(maximum_total or 0, int(total), maximum_done)
                write(maximum_done, maximum_total)

        return type(
            "CommandProgress",
            (cls,),
            {
                "_write": staticmethod(monotonic_write),
                "_min_interval_seconds": min_interval_seconds,
                "_monotonic": staticmethod(monotonic),
            },
        )


class ModelCommandExecutor:
    """§4.2 DECIDED: claim and run exactly one queued `model_commands` row.

    `variants` is the models config's `variant_id -> variant` mapping;
    `model_provider` is the worker's live-switching provider (T22). Wiring
    this into the `worker_cli` loop (calling `run_once()` before
    `worker.run_once()` at startup) is T32's job -- this class only knows
    how to run one command to completion.
    """

    def __init__(
        self,
        *,
        jobs: JobsProtocol,
        worker_id: str,
        cache_dir: Path,
        variants: Mapping[str, Mapping[str, Any]],
        model_provider: ModelProvider,
        heartbeat_interval_seconds: float,
        publish_readiness: Callable[[float], None] = lambda now: None,
        clock: Callable[[], float] = time.time,
        progress_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._jobs = jobs
        self._worker_id = worker_id
        self._cache_dir = cache_dir
        self._variants = variants
        self._model_provider = model_provider
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._publish_readiness = publish_readiness
        self._clock = clock
        self._progress_clock = progress_clock

    def run_once(self, *, kinds: tuple[str, ...] | None = None) -> bool:
        """Claim the oldest allowed queued command; `False` if none waits."""
        if kinds is None:
            claim = self._jobs.claim_next_model_command(
                worker_id=self._worker_id, now=self._clock()
            )
        else:
            claim = self._jobs.claim_next_model_command(
                worker_id=self._worker_id, now=self._clock(), kinds=kinds
            )
        if claim is None:
            return False
        stopped = threading.Event()

        def renew() -> None:
            while not stopped.wait(self._heartbeat_interval_seconds):
                now = self._clock()
                try:
                    self._jobs.heartbeat_model_command(
                        claim.command_id, worker_id=self._worker_id, now=now
                    )
                except FenceRejected:
                    return
                # design §4.2 step 2: readiness must never go stale during a
                # long download/load, so the same tick republishes it.
                self._publish_readiness(now)

        heartbeat = threading.Thread(target=renew, name="model-command-heartbeat", daemon=True)
        heartbeat.start()
        try:
            state, error_code = self._execute(claim)
        finally:
            stopped.set()
            heartbeat.join()
        finished = self._jobs.finish_model_command(
            claim.command_id,
            worker_id=self._worker_id,
            state=state,
            now=self._clock(),
            error_code=error_code,
        )
        if not finished:
            # §4.2 step 4: the command was requeued to another worker after
            # a stale heartbeat while this one was still running -- discard
            # the late result rather than overwrite the new claim.
            LOGGER.info(
                "model_command_fence_lost command_id=%s kind=%s variant_id=%s",
                claim.command_id,
                claim.kind,
                claim.variant_id,
            )
        return True

    def _execute(self, claim: ModelCommandRecord) -> tuple[str, str | None]:
        variant = self._variants[claim.variant_id]
        # Second safety net behind the API's 422 (models_response 1.2): an
        # unpublished/planned variant has no origin to download, load or
        # delete, so its command fails before any network, cache or GPU use.
        availability = variant.get("availability", PUBLISHED)
        if availability == "unpublished":
            return "failed", "MODEL_VARIANT_NOT_PUBLISHED"
        if availability == "planned":
            return "failed", "MODEL_VARIANT_PLANNED"
        if claim.kind == "download":
            return self._run_download(claim, variant)
        if claim.kind == "load":
            return self._run_load(claim, variant)
        return self._run_delete(claim, variant)

    def _progress_writer(self, command_id: str) -> Callable[[int | None, int | None], None]:
        def write(bytes_done: int | None, bytes_total: int | None) -> None:
            try:
                self._jobs.update_model_command_progress(
                    command_id,
                    worker_id=self._worker_id,
                    phase="downloading",
                    bytes_done=bytes_done,
                    bytes_total=bytes_total,
                    now=self._clock(),
                )
            except FenceRejected:
                LOGGER.info("model_command_fence_lost command_id=%s", command_id)

        return write

    def _advance_phase(self, claim: ModelCommandRecord, phase: str) -> None:
        """Move to `phase`, carrying over whatever byte progress the row
        already has -- a bare phase transition must never wipe out the
        download's last known `bytes_done`/`bytes_total` back to unknown."""
        current = self._jobs.active_model_command(claim.variant_id, claim.kind)
        bytes_done = current["bytes_done"] if current is not None else None
        bytes_total = current["bytes_total"] if current is not None else None
        self._jobs.update_model_command_progress(
            claim.command_id,
            worker_id=self._worker_id,
            phase=phase,
            bytes_done=bytes_done,
            bytes_total=bytes_total,
            now=self._clock(),
        )

    def _run_download(
        self, claim: ModelCommandRecord, variant: Mapping[str, Any]
    ) -> tuple[str, str | None]:
        from httpx import HTTPError
        from huggingface_hub.errors import HfHubHTTPError

        if "bundle" in variant:
            # A local bundle variant has nothing to download (T06 contract).
            return "succeeded", None
        source = variant["source"]
        self._advance_phase(claim, "downloading")
        try:
            snapshot_dir = Path(
                _snapshot_download(
                    repo_id=source["repo_id"],
                    revision=source["revision"],
                    cache_dir=self._cache_dir,
                    tqdm_class=CommandProgress.bound(
                        self._progress_writer(claim.command_id), monotonic=self._progress_clock
                    ),
                )
            )
        except (OSError, RuntimeError, HfHubHTTPError, HTTPError) as exc:
            LOGGER.warning(
                "model_command_failed command_id=%s kind=download exc_type=%s",
                claim.command_id,
                type(exc).__name__,
            )
            return "failed", "MODEL_DOWNLOAD_FAILED"
        bundle_dir = snapshot_dir / source["subfolder"] if "subfolder" in source else snapshot_dir
        self._advance_phase(claim, "verifying")
        try:
            verify_bundle(bundle_dir, expected_manifest_sha256=source["bundle_manifest_sha256"])
        except ModelBundleInvalid as exc:
            LOGGER.warning(
                "model_command_failed command_id=%s kind=download exc_type=%s",
                claim.command_id,
                type(exc).__name__,
            )
            # A download that failed verification must never be reported
            # `ready` later -- delete the unverified snapshot (§4.2 step 3).
            delete_variant_snapshot(variant, cache_dir=self._cache_dir)
            return "failed", "MODEL_CHECKPOINT_UNVERIFIED"
        return "succeeded", None

    def _run_load(
        self, claim: ModelCommandRecord, variant: Mapping[str, Any]
    ) -> tuple[str, str | None]:
        downloaded, _size = is_variant_downloaded(variant, cache_dir=self._cache_dir)
        if not downloaded:
            state, error_code = self._run_download(claim, variant)
            if state == "failed":
                return state, error_code
        self._advance_phase(claim, "loading")
        try:
            self._model_provider.ensure_loaded(claim.variant_id)
        except ModelBundleInvalid as exc:
            self._log_load_failure(claim, exc)
            return "failed", "MODEL_CHECKPOINT_UNVERIFIED"
        except EngineUnavailable as exc:
            self._log_load_failure(claim, exc)
            return "failed", "MODEL_ENGINE_UNAVAILABLE"
        except (OSError, ValueError, RuntimeError) as exc:
            self._log_load_failure(claim, exc)
            if _is_oom_error(exc):
                return "failed", "MODEL_VARIANT_DOES_NOT_FIT"
            return "failed", "INTERNAL_ERROR"
        return "succeeded", None

    @staticmethod
    def _log_load_failure(claim: ModelCommandRecord, exc: BaseException) -> None:
        # Only the exception TYPE and the command's own identifiers -- never
        # the message, which can carry a filesystem path or URL.
        LOGGER.warning(
            "model_command_failed command_id=%s kind=load exc_type=%s",
            claim.command_id,
            type(exc).__name__,
        )

    def _run_delete(
        self, claim: ModelCommandRecord, variant: Mapping[str, Any]
    ) -> tuple[str, str | None]:
        model_state = self._jobs.model_state()
        resident_variant_id = model_state[0] if model_state is not None else None
        if resident_variant_id == claim.variant_id:
            # The API already refuses this at enqueue time; the worker is
            # the second safety net against deleting a resident variant.
            return "failed", "MODEL_COMMAND_CONFLICT"
        self._jobs.update_model_command_progress(
            claim.command_id,
            worker_id=self._worker_id,
            phase="deleting",
            now=self._clock(),
        )
        delete_variant_snapshot(variant, cache_dir=self._cache_dir)
        return "succeeded", None

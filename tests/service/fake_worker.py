"""Test-only worker that drains `JobStore`/`model_commands` without a GPU
(T37, design Sec 7.6/Sec 2.5, plan T37).

**A.3 choice (documented per plan §A.3, made inside the 2 h window):**
publish fixture artefacts directly via `JobStore`/`ArtifactStore`, not by
running `DiagnosticWorker.run_once()`. The alternative the plan offered --
"a real `DiagnosticWorker` with `FakeEngine`" -- turns out to be unusable
without editing files this task does not own: `DiagnosticWorker.run_once()`
imports `unflatten.service.worker`, which imports
`unflatten.model.prediction_ir` (for `PredictionIrInvalid`) at module load,
which imports `unflatten.model.node_lifecycle`, which imports `torch` at
module load -- confirmed locally (`ModuleNotFoundError: No module named
'torch'` in a bare `uv sync`). `FakeEngine.seal()` (`tests/service/
fake_engine.py`, owned by stream D) also just returns its input unchanged,
which is not a `PredictionIrArtifact` and fails
`verify_prediction_ir_artifact` before a job could ever reach
`finalize_success` down that path. Building a real one needs
`RawDocumentHypothesis`/`NodeState`, i.e. the same torch import. Editing
`fake_engine.py` is out of scope (file ownership table, plan §B: stream D
owns it). `FakeEngine` (§17.2) is still used here, for the one thing it does
torch-free: realistic per-page stage/progress timing on `decode_pages`. The
published `prediction_ir`/`diagnostic_docx`/preview bytes are canned fixtures
under `tests/data/e2e/` (the `diagnostic.docx` is a real renderer output,
`tests/data/corpus/valid/custom_text_list.json` compiled once offline; the
`prediction_ir.json` fixture is NOT schema-valid -- see its own
`_fixture_note` -- because that validation needs the same torch import this
module avoids). This matches the plan's own documented fallback ("Zalozna
volba je publikovanie fixtures").

Runs against a job store already created and `.initialize()`-d by
`api_cli.py` (this module never creates schema itself, so it never races the
API process for the same tables). No GPU, no torch import.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from pathlib import Path

from tests.service.fake_engine import FakeEngine

from unflatten.service.artifact_store import ArtifactStore
from unflatten.service.contracts import validate_contract
from unflatten.service.inference_engine import PageInput
from unflatten.service.job_store import ArtifactMetadata, JobStore
from unflatten.service.job_store_types import FenceRejected, ModelCommandRecord, ReadinessWriter
from unflatten.service.models_config import load_models_config
from unflatten.service.policy import DEFAULT_RUNTIME_POLICY_PATH, load_runtime_policy

LOGGER = logging.getLogger("fake_worker")

FAKE_HF_MANIFEST_BYTES = b'{"files":{"config.json":{"size_bytes":2}}}\n'

FIXTURE_DIR = Path(__file__).parent.parent / "data" / "e2e"
_DECODE_STEPS = 10
_STEP_SECONDS = 0.1
_DEVICE_FACTS = {"free_vram_mib": 12288.0, "total_vram_mib": 12288.0, "compute_capability": (8, 9)}


class _NullProgress:
    """`EngineProgress` that discards stage/page events (§17.2 protocol) --
    `FakeEngine.decode_pages` still needs one to call, but this loop reports
    its own stages to `JobStore` directly (see `_run_job`)."""

    def enter_stage(self, stage: str) -> None:  # noqa: D102
        return None

    def page_done(self, result: object) -> None:  # noqa: D102
        return None


def _success_outcome() -> dict:
    return validate_contract(
        {
            "prediction_ir_status": "sealed",
            "claim_scope": "diagnostic",
            "canonical_ast_produced": False,
            "docx_produced": True,
            "reason_codes": ["HEADLESS_DEFAULTS_V0_APPLIED"],
            "docx_chain": {
                "policy_id": "headless-defaults.v0",
                "policy_status": "RATIFIED",
                "page_outcome": "produced",
                "renderer_status": "render_ready",
            },
        },
        "diagnostic_outcome",
    )


def _artifact_metadata(role: str, ref, *, page_index: int = -1) -> ArtifactMetadata:
    return ArtifactMetadata(
        role=role,
        ref=ref.ref,
        sha256=ref.sha256.removeprefix("sha256:"),
        size_bytes=ref.size_bytes,
        mime_type=ref.mime_type,
        page_index=page_index,
    )


def run_job(
    jobs: JobStore,
    artifacts: ArtifactStore,
    *,
    worker_id: str,
    now: float,
    engine: FakeEngine | None = None,
) -> bool:
    """Claim and drain at most one queued job. Returns whether one was
    claimed. Mirrors `DiagnosticWorker.run_once()`'s public shape (§ above
    for why it cannot reuse that implementation)."""
    jobs.requeue_expired(now=now)
    lease = jobs.claim_next(worker_id=worker_id, now=now)
    if lease is None:
        return False
    job = jobs.get_claimed_job(lease, now=now)
    engine = engine or FakeEngine()
    jobs.update_stage(lease, stage="feature_extraction", now=now)
    engine.decode_pages(
        [PageInput(key=(job.job_id, 0), model_input=None)], progress=_NullProgress()
    )
    jobs.heartbeat(lease, now=now)
    jobs.update_stage(lease, stage="artifact_finalization", now=now)
    staged = [
        artifacts.stage(
            principal_id=job.principal_id, job_id=job.job_id, execution_id=lease.execution_id,
            role="prediction_ir", mime_type="application/json",
            chunks=((FIXTURE_DIR / "prediction_ir.json").read_bytes(),),
        ),
    ]
    roles = ["prediction_ir"]
    if job.requested_output == "diagnostic_docx":
        staged.append(
            artifacts.stage(
                principal_id=job.principal_id, job_id=job.job_id, execution_id=lease.execution_id,
                role="diagnostic_docx",
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                chunks=((FIXTURE_DIR / "diagnostic.docx").read_bytes(),),
            )
        )
        roles.append("diagnostic_docx")
        preview_files = (
            ("source_preview", "source_preview.png"),
            ("docx_preview", "docx_preview.png"),
        )
        for role, filename in preview_files:
            staged.append(
                artifacts.stage(
                    principal_id=job.principal_id, job_id=job.job_id,
                    execution_id=lease.execution_id, role=role, mime_type="image/png",
                    chunks=((FIXTURE_DIR / filename).read_bytes(),),
                )
            )
            roles.append(role)
    outcome = (
        _success_outcome()
        if job.requested_output == "diagnostic_docx"
        else validate_contract(
            {
                "prediction_ir_status": "sealed",
                "claim_scope": "diagnostic",
                "canonical_ast_produced": False,
                "docx_produced": False,
                "reason_codes": ["DIAGNOSTIC_DOCX_NOT_REQUESTED"],
                "docx_chain": None,
            },
            "diagnostic_outcome",
        )
    )

    def fence(published, publish) -> bool:
        jobs.finalize_success(
            lease,
            artifacts=[_artifact_metadata(role, ref) for role, ref in zip(roles, published)],
            diagnostic_outcome=outcome,
            retention_seconds=3600,
            now=now,
            publish=publish,
        )
        return True

    try:
        artifacts.publish_many(staged, fence, allow_repeated_roles=True)
    except FenceRejected:
        LOGGER.info("fake_worker_fence_lost job_id=%s", lease.job_id)
    return True


def _hf_cache_dir() -> Path | None:
    """Same derivation as `api_cli._resolve_hf_cache_dir` -- the fake worker
    has no `--models-cache-dir` flag of its own, so `HF_HOME` (set by
    `playwright.config.ts` on the one e2e backend that configures model
    variants) is the only lever that lets it agree with the API process on
    where a downloaded variant lives."""
    hf_home = os.environ.get("HF_HOME")
    return Path(hf_home) / "hub" if hf_home else None


def _load_models_variants() -> dict[str, dict] | None:
    """`UNFLATTEN_MODELS_CONFIG` (same env var `api_cli.py` reads), or
    `None` on every e2e backend that never configures model variants --
    unchanged behaviour for `run_model_command`'s no-op case (T37)."""
    config_path = os.environ.get("UNFLATTEN_MODELS_CONFIG")
    if not config_path:
        return None
    config = load_models_config(config_path)
    return {variant["variant_id"]: variant for variant in config["variants"]}


def _materialize_hf_snapshot(source: dict, cache_dir: Path) -> None:
    """Stand-in for the real `ModelCommandExecutor`'s `snapshot_download` +
    `verify_bundle` (design §7.6 'verify/ready'): the e2e harness never
    touches the network, so a 'succeeded' download command must still leave
    behind what `is_variant_downloaded` (`huggingface_hub.scan_cache_dir`)
    looks for -- otherwise `GET /v1/models` reports the variant absent again
    right after this worker's own progress said it finished, and the UI
    would never offer Load."""
    folder = "models--" + source["repo_id"].replace("/", "--")
    snapshot_dir = cache_dir / folder / "snapshots" / source["revision"]
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "config.json").write_bytes(b"{}")
    (snapshot_dir / "manifest.json").write_bytes(FAKE_HF_MANIFEST_BYTES)


def _finish_download_command(
    command: ModelCommandRecord, jobs: JobStore, *, worker_id: str
) -> None:
    """For an `hf`-sourced variant, an extra 'verifying' tick (mirroring the
    real executor's `downloading` -> `verifying` phase transition) then the
    fake snapshot materialization above; a no-op for a local `bundle`
    variant or when no models config is configured at all."""
    variants = _load_models_variants()
    cache_dir = _hf_cache_dir()
    if variants is None or cache_dir is None:
        return
    variant = variants.get(command.variant_id)
    if variant is None or "source" not in variant:
        return
    jobs.update_model_command_progress(
        command.command_id, worker_id=worker_id, phase="verifying",
        bytes_done=_DECODE_STEPS, bytes_total=_DECODE_STEPS, now=time.time(),
    )
    _materialize_hf_snapshot(variant["source"], cache_dir)


def run_model_command(
    jobs: JobStore, *, worker_id: str, writer: ReadinessWriter | None = None
) -> bool:
    """Drain at most one queued model command with a fake 10-step, 100 ms/step
    progress (design §7.6), then mark it succeeded.

    `writer` is the same fenced handle `publish_fake_readiness` returned --
    a successful `load` republishes it as the resident variant (`GET
    /v1/models`'s `resident`/`On GPU` badge), matching the real worker's
    `ModelCommandExecutor` (`model_provider.ensure_loaded` + readiness
    republish). `None` (every caller before T46) skips this, unchanged."""
    command = jobs.claim_next_model_command(worker_id=worker_id, now=time.time())
    if command is None:
        return False
    phase = {"download": "downloading", "load": "loading", "delete": "deleting"}[command.kind]
    for step in range(1, _DECODE_STEPS + 1):
        time.sleep(_STEP_SECONDS)
        jobs.update_model_command_progress(
            command.command_id, worker_id=worker_id, phase=phase,
            bytes_done=step, bytes_total=_DECODE_STEPS, now=time.time(),
        )
    if command.kind in ("download", "load"):
        _finish_download_command(command, jobs, worker_id=worker_id)
    if command.kind == "load" and writer is not None:
        jobs.publish_model_state(
            writer, resident_variant=command.variant_id, now=time.time(), **_DEVICE_FACTS
        )
    jobs.finish_model_command(
        command.command_id, worker_id=worker_id, state="succeeded", now=time.time()
    )
    return True


def publish_fake_readiness(jobs: JobStore, *, worker_id: str) -> ReadinessWriter:
    """Publish worker readiness + device facts with no GPU (design §7.6:
    `total_vram_mib=12288`, `compute_capability=(8, 9)`). Returns the fenced
    writer so `run_model_command` can republish a `load`'s resident variant
    under the same fence."""
    writer = jobs.claim_readiness_writer(worker_id=worker_id, now=time.time())
    jobs.publish_readiness(
        writer, worker_ready=True, sandbox_ready=True, diagnostic_docx_ready=True,
        now=time.time(), sandbox_rootful=False, available_engines=("native",),
    )
    jobs.publish_model_state(
        writer, resident_variant=None, now=time.time(), **_DEVICE_FACTS
    )
    return writer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-db", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--worker-id", default="fake-worker")
    parser.add_argument("--policy", type=Path, default=DEFAULT_RUNTIME_POLICY_PATH)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.2)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    policy = load_runtime_policy(args.policy)
    jobs = JobStore(
        args.job_db,
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    artifacts = ArtifactStore(args.artifact_root)
    writer = publish_fake_readiness(jobs, worker_id=args.worker_id)

    stop = {"requested": False}

    def _handle_signal(_signum: int, _frame: object) -> None:
        stop["requested"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    LOGGER.info("fake_worker_started worker_id=%s job_db=%s", args.worker_id, args.job_db)
    while not stop["requested"]:
        did_command = run_model_command(jobs, worker_id=args.worker_id, writer=writer)
        did_job = run_job(jobs, artifacts, worker_id=args.worker_id, now=time.time())
        if not did_command and not did_job:
            time.sleep(args.poll_interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

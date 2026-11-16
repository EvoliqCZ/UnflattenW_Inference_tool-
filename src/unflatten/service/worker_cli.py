"""Separately supervised CLI for the single-GPU diagnostic worker."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from unflatten.model.candidate_runtime import (
    KV_CACHE_MODES,
    DecodePolicy,
    RuntimeIdentityPin,
    load_candidate_runtime,
)
from unflatten.model.quantization.auto_policy import load_auto_policy, resolve_auto_quantization
from unflatten.model.quantization.spec import QuantizationSpec
from unflatten.service.artifact_store import ArtifactStore
from unflatten.service.device_memory import cuda_memory_info
from unflatten.service.docx_stage import diagnostic_docx_capability
from unflatten.service.engines.native import NativeEngine
from unflatten.service.inference_engine import (
    EngineId,
    EngineRegistry,
    InferenceEngine,
)
from unflatten.service.job_store import JobStore
from unflatten.service.job_store_types import ModelCommandConflict
from unflatten.service.model_bundle import (
    ALLOWLIST_FILENAME,
    BACKBONE_DIRNAME,
    CAPS_FILENAME,
    CHECKPOINT_FILENAME,
    COMPONENT_LOCK_FILENAME,
    DECODE_POLICY_FILENAME,
    ModelBundleInvalid,
    manifest_sha256,
    verify_bundle,
    visual_feature_tap_from_identities,
)
from unflatten.service.model_commands import EngineUnavailable, ModelCommandExecutor
from unflatten.service.model_runtime import DiagnosticModelRuntime
from unflatten.service.model_variants import (
    SERVICE_DEFAULT_KV_CACHE_MODE,
    is_variant_downloaded,
    resolve_default_variant,
    static_kv_reserve_mib,
    variant_fit_reason,
)
from unflatten.service.models_config import (
    ModelsConfigInvalid,
    is_published,
    load_models_config,
)
from unflatten.service.pdf_sandbox import PdfSandbox
from unflatten.service.policy import DEFAULT_RUNTIME_POLICY_PATH, load_runtime_policy
from unflatten.service.retention import JobStoreRetentionRepository, RetentionSweeper
from unflatten.service.worker import DiagnosticWorker

LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fenced diagnostic GPU worker")
    parser.add_argument("--job-db", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_RUNTIME_POLICY_PATH)
    parser.add_argument("--sandbox-image", required=True)
    parser.add_argument("--sandbox-runtime", choices=("docker", "podman"), default="podman")
    parser.add_argument("--sandbox-allow-rootful", action="store_true")
    parser.add_argument(
        "--model-bundle",
        type=Path,
        default=None,
        help="D7 self-contained model bundle directory; mutually exclusive with "
        "--models-config, one of the two is required",
    )
    parser.add_argument(
        "--model-bundle-sha256",
        default=None,
        help="optional pin on the bundle manifest's own sha256 (UNFLATTEN_MODEL_BUNDLE_SHA256); "
        "ignored when --models-config resolves an hf source (its own "
        "bundle_manifest_sha256 is used instead)",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=os.environ.get("UNFLATTEN_MODELS_CONFIG") or None,
        help="DEC-044 Dodatok D §1 models variants config (UNFLATTEN_MODELS_CONFIG); "
        "mutually exclusive with --model-bundle. This worker still loads exactly ONE "
        "resident variant at startup (--variant, default the config's default_variant) "
        "-- it does not switch variants between jobs of a running process.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="variant_id to load from --models-config; defaults to its default_variant",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="cache directory for --models-config hf source downloads "
        "(defaults to huggingface_hub's own default, honouring HF_HOME)",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-id", default="worker_gpu_0")
    parser.add_argument("--idle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=("bf16", "int8", "fp8", "int4", "auto"),
        default=os.environ.get("UNFLATTEN_PRECISION", "bf16"),
        help="decoder weight precision to load (UNFLATTEN_PRECISION); "
        "default bf16 is today's byte-identical behaviour. Must match the "
        "model bundle's own precision identity -- a mismatch is refused. "
        "'auto' resolves from measured free VRAM/compute capability at "
        "startup (quantization.auto_policy) and ignores --kv-cache.",
    )
    parser.add_argument(
        "--kv-cache",
        choices=("bf16", "fp8"),
        default=os.environ.get("UNFLATTEN_KV_CACHE", "bf16"),
        help="KV cache storage dtype (UNFLATTEN_KV_CACHE); default bf16 is "
        "today's byte-identical behaviour",
    )
    parser.add_argument(
        "--kv-cache-mode",
        choices=KV_CACHE_MODES,
        # `or`: an env_file exports a present-but-empty variable.
        default=os.environ.get("UNFLATTEN_KV_CACHE_MODE") or SERVICE_DEFAULT_KV_CACHE_MODE,
        help="what backs the decode's KV cache (UNFLATTEN_KV_CACHE_MODE); default "
        "'static_graph' (owner decisions 2026-09-29: preallocated StaticKVCache plus "
        "piecewise CUDA graphs for the q=1 step, identical output, up to ~650 MiB "
        "more VRAM). 'static' is the same cache without graphs; 'bf16_dynamic' is "
        "the pre-2026-09-29 behaviour; see CandidateRuntime.kv_cache_mode.",
    )
    parser.add_argument(
        "--deepstack",
        action="store_true",
        default=os.environ.get("UNFLATTEN_DEEPSTACK", "") == "1",
        help="DEC-056 (default off, UNFLATTEN_DEEPSTACK=1): run a bundle whose "
        "checkpoint was trained with DeepStack (v16). The bundle's "
        "identities.visual_feature_tap must agree in both directions -- a v15b "
        "bundle is refused with this flag, a DeepStack bundle without it.",
    )
    parser.add_argument(
        "--fp8-shared-input-quant",
        action=argparse.BooleanOptionalAction,
        default=_env_bool_or_raw(_FP8_SHARED_INPUT_QUANT_ENV),
        help="fp8 decoder only: q/k/v and gate/up share one activation "
        f"quantization ({_FP8_SHARED_INPUT_QUANT_ENV}); default ON for an fp8 "
        "decoder, off otherwise (owner 2026-09-30). Output is bit-identical "
        "(docs/runs/fp8-qkv-fusion-2026-09-29.md). An explicit "
        "--fp8-shared-input-quant is refused with a non-fp8 --precision; a non-fp8 "
        "variant resolved at load time (models config, auto) loads unshared and the "
        "load log line says so.",
    )
    parser.add_argument(
        "--loop-guard",
        choices=LOOP_GUARD_OVERRIDES,
        # `or`: an env_file exports a present-but-empty variable.
        default=os.environ.get("UNFLATTEN_LOOP_GUARD") or SERVICE_DEFAULT_LOOP_GUARD,
        help="loop guard for the decode (UNFLATTEN_LOOP_GUARD); default 'v3' with arm "
        "'r2' (DEC-055 frozen criteria passed, docs/runs/loop-guard-v3-2026-09-29.md). "
        "'bundle' keeps the model bundle's own decode_policy value (today 'v2'); 'v2' "
        "forces v2.",
    )
    parser.add_argument(
        "--loop-guard-v3-arm",
        default=os.environ.get("UNFLATTEN_LOOP_GUARD_V3_ARM") or None,
        help="loop guard v3 arm, e.g. 'r1' or 'r2+r4' (UNFLATTEN_LOOP_GUARD_V3_ARM); "
        "only with --loop-guard v3; unset = the service default "
        f"'{SERVICE_DEFAULT_LOOP_GUARD_V3_ARM}'.",
    )
    parser.add_argument(
        "--engines",
        default=os.environ.get("UNFLATTEN_ENGINES", "native"),
        help="comma-separated InferenceEngine ids this worker may run a variant "
        "under (UNFLATTEN_ENGINES); 'native' is always the reference engine. A "
        "non-native engine stays unavailable until its parity gate is proven "
        "(design §17.5).",
    )
    return parser


_FP8_SHARED_INPUT_QUANT_ENV = "UNFLATTEN_FP8_SHARED_INPUT_QUANT"
_ENV_TRUE = frozenset({"1", "true", "yes", "on"})
_ENV_FALSE = frozenset({"", "0", "false", "no", "off"})


def _env_bool_or_raw(name: str) -> bool | str | None:
    """`True`/`False` for a recognised value of environment variable `name`,
    `None` (= the built-in default) when it is unset or empty, else the raw
    string -- argparse never validates a default, so `main` refuses a
    non-bool explicitly."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in _ENV_TRUE:
        return True
    if value in _ENV_FALSE:
        return False
    return raw


#: Loop guards `--loop-guard` may select; "bundle" keeps the bundle's own value.
LOOP_GUARD_OVERRIDES = ("bundle", "v2", "v3")
#: DEC-055: arm r2 passed all three frozen criteria on 42 val_iid pages
#: (`docs/runs/loop-guard-v3-2026-09-29.md`), so the service decodes with it by
#: default; prod deployment stays the owner's step.
SERVICE_DEFAULT_LOOP_GUARD = "v3"
SERVICE_DEFAULT_LOOP_GUARD_V3_ARM = "r2"


def apply_loop_guard_override(
    decode_policy: DecodePolicy, loop_guard: str | None, loop_guard_v3_arm: str | None
) -> DecodePolicy:
    """DEC-055 switch: `None` returns `decode_policy` unchanged (the bundle's
    frozen value); otherwise the loop guard (and v3 arm) are replaced and
    the override is logged, so a v3 worker is never mistaken for the bundle's
    frozen policy."""
    if loop_guard in (None, "bundle"):
        return decode_policy
    if loop_guard == "v3" and loop_guard_v3_arm is None:
        loop_guard_v3_arm = SERVICE_DEFAULT_LOOP_GUARD_V3_ARM
    LOGGER.info(
        "worker_loop_guard_override bundle=%s override=%s v3_arm=%s",
        decode_policy.loop_guard,
        loop_guard,
        loop_guard_v3_arm,
    )
    return replace(decode_policy, loop_guard=loop_guard, loop_guard_v3_arm=loop_guard_v3_arm)


#: `{engine_id: factory}` a worker may name in `--engines`. `native` is always
#: importable and `parity_status="reference"` (§17.5).
_ENGINE_FACTORIES: dict[EngineId, Callable[[], InferenceEngine]] = {
    "native": NativeEngine,
}


def _parse_engines(engines_arg: str) -> tuple[EngineId, ...]:
    """Validate `--engines`/`UNFLATTEN_ENGINES` against the known engine ids,
    preserving the given order. Raises on an empty or unknown name -- a
    worker never silently drops an engine the operator named."""
    names = tuple(name.strip() for name in engines_arg.split(",") if name.strip())
    if not names:
        raise ValueError("--engines must name at least one engine")
    unknown = [name for name in names if name not in _ENGINE_FACTORIES]
    if unknown:
        raise ValueError(f"--engines names unknown engine(s): {unknown}")
    return names


def _build_engine_registry(engines_arg: str) -> EngineRegistry:
    names = _parse_engines(engines_arg)
    return EngineRegistry(factories={name: _ENGINE_FACTORIES[name] for name in names})


def main(argv: list[str] | None = None) -> int:
    """Load one resident model from a verified bundle, then process jobs
    serially until interrupted."""
    args = _parser().parse_args(argv)
    if args.idle_seconds <= 0:
        raise ValueError("idle-seconds must be positive")
    if args.kv_cache_mode not in KV_CACHE_MODES:
        # argparse never checks a default (here: the environment) against
        # `choices`, so a typo in UNFLATTEN_KV_CACHE_MODE lands here.
        raise ValueError(
            "--kv-cache-mode/UNFLATTEN_KV_CACHE_MODE must be one of "
            f"{list(KV_CACHE_MODES)}, got {args.kv_cache_mode!r}"
        )
    if args.fp8_shared_input_quant is not None and not isinstance(
        args.fp8_shared_input_quant, bool
    ):
        raise ValueError(
            f"{_FP8_SHARED_INPUT_QUANT_ENV} must be one of "
            f"{sorted(_ENV_TRUE | _ENV_FALSE)}, got {args.fp8_shared_input_quant!r}"
        )
    if (
        args.fp8_shared_input_quant is True
        and args.precision not in ("fp8", "auto")
        and (args.models_config is None)
    ):
        raise ValueError(
            f"--fp8-shared-input-quant needs an fp8 decoder, got --precision {args.precision!r}"
        )
    if args.loop_guard not in (None, *LOOP_GUARD_OVERRIDES):
        raise ValueError(
            "--loop-guard/UNFLATTEN_LOOP_GUARD must be one of "
            f"{list(LOOP_GUARD_OVERRIDES)}, got {args.loop_guard!r}"
        )
    if args.loop_guard_v3_arm is not None and args.loop_guard != "v3":
        raise ValueError("--loop-guard-v3-arm/UNFLATTEN_LOOP_GUARD_V3_ARM needs --loop-guard v3")
    if args.loop_guard == "v3":
        from unflatten.model.loop_guard_v3 import load_loop_guard_v3_policy, resolve_arm

        resolve_arm(
            load_loop_guard_v3_policy(),
            args.loop_guard_v3_arm or SERVICE_DEFAULT_LOOP_GUARD_V3_ARM,
        )
    if (args.model_bundle is None) == (args.models_config is None):
        raise ValueError("exactly one of --model-bundle or --models-config is required")
    engine_registry = _build_engine_registry(args.engines)
    policy = load_runtime_policy(args.policy)
    sandbox = PdfSandbox(
        policy=policy,
        image=args.sandbox_image,
        runtime=args.sandbox_runtime,
        allow_rootful=args.sandbox_allow_rootful,
    )
    # G7: do not load a model or claim an untrusted PDF job when the OCI
    # runtime is rootful without explicit authorization, unrooted, or running
    # an image with the wrong scanner identity.
    sandbox.require_ready()
    jobs = JobStore(
        args.job_db,
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    # design §4.2 "Reštart workera uprostred": a `running` model command whose
    # heartbeat went stale (the previous worker process died mid-download or
    # mid-load) must be requeued before this process claims anything.
    jobs.requeue_stale_model_commands(
        now=time.time(), stale_after=2 * policy["heartbeat_interval_seconds"]
    )

    readiness_writer = None
    variants_by_id: dict[str, Any] = {}
    if args.models_config is not None:
        config = load_models_config(args.models_config)
        variants_by_id = {variant["variant_id"]: variant for variant in config["variants"]}
        requested_variant_id = args.variant or config["default_variant"]
        if requested_variant_id == "auto":
            # design §4.2 "Štartovací auto-load": an explicit --variant always
            # wins over "auto" (plan note 6 / item A.6) -- this branch is only
            # reached when the operator did not name a variant and either the
            # config's own default_variant or --variant itself is "auto".
            readiness_writer = jobs.claim_readiness_writer(
                worker_id=args.worker_id, now=time.time()
            )
            facts = _measure_own_device_facts(args.device)
            if facts is not None:
                jobs.publish_model_state(
                    readiness_writer, resident_variant=None, now=time.time(), **facts
                )

            def publish_startup(now: float) -> None:
                snapshot = sandbox.readiness()
                jobs.publish_readiness(
                    readiness_writer,
                    worker_ready=False,
                    sandbox_ready=snapshot["ready"] is True,
                    diagnostic_docx_ready=False,
                    now=now,
                    available_engines=engine_registry.available(),
                    sandbox_rootful=(
                        snapshot["rootless"] is False and snapshot["runtime_kind"] == "docker"
                    ),
                )

            publish_startup(time.time())
            bootstrap_provider = _BootstrapModelProvider(args, policy)
            bootstrap_executor = ModelCommandExecutor(
                jobs=jobs,
                worker_id=args.worker_id,
                cache_dir=args.hf_cache_dir,
                variants=variants_by_id,
                model_provider=bootstrap_provider,
                heartbeat_interval_seconds=policy["heartbeat_interval_seconds"],
                publish_readiness=publish_startup,
            )
            # Downloads and cache deletion do not need an admissible GPU.
            # Service them even when the subsequent cold load cannot fit.
            while bootstrap_executor.run_once(kinds=("download", "delete")):
                pass
            resolved_variant_id = _resolve_auto_variant(args, config, engine_registry, jobs)
            if resolved_variant_id is None:
                code = "MODEL_ENGINE_UNAVAILABLE" if facts is None else "MODEL_VARIANT_DOES_NOT_FIT"
                for variant in config["variants"]:
                    if is_published(variant):
                        jobs.record_model_startup_failure(
                            variant_id=variant["variant_id"], error_code=code, now=time.time()
                        )
                raise SystemExit(
                    "--variant auto: no configured variant fits this worker's available "
                    "engines and measured VRAM; see the model status for its error"
                )
            # Resume queued/stale download commands before adding a different
            # startup kind for the same variant. A load already queued by the
            # user may satisfy startup itself.
            automatic_load_queued = False
            while bootstrap_provider.loaded is None:
                if bootstrap_executor.run_once():
                    continue
                if automatic_load_queued:
                    break
                try:
                    jobs.enqueue_model_command(
                        kind="load",
                        variant_id=resolved_variant_id,
                        requested_by="worker",
                        now=time.time(),
                    )
                except ModelCommandConflict:
                    # A UI download may arrive between draining the queue and
                    # this enqueue. Execute it, then retry. A live command
                    # owned by another worker must not cause a busy loop.
                    if not bootstrap_executor.run_once():
                        raise SystemExit("--variant auto: another model command is active")
                    continue
                automatic_load_queued = True
            if bootstrap_provider.loaded is None:
                raise SystemExit(
                    f"--variant auto: startup load of {resolved_variant_id!r} failed -- "
                    "see the model_commands row for its error_code"
                )
            loaded = bootstrap_provider.loaded
            # Do not leave an invisible owner of the initial model in main().
            bootstrap_provider.loaded = None
            del bootstrap_provider, bootstrap_executor
        else:
            loaded = _load_variant(args, policy, selected_variant=None)
    else:
        loaded = _load_variant(args, policy, selected_variant=None)
    model = loaded.model
    docx_capable = loaded.docx_capable

    artifact_store = ArtifactStore(args.artifact_root)
    if readiness_writer is None:
        readiness_writer = jobs.claim_readiness_writer(worker_id=args.worker_id, now=time.time())
    _publish_model_state(jobs, readiness_writer, args.device, loaded.selected_variant)

    def publish_readiness(now: float) -> None:
        sandbox_snapshot = sandbox.readiness()
        model_ready = model_provider is None or model_provider.is_loaded
        jobs.publish_readiness(
            readiness_writer,
            worker_ready=model_ready,
            sandbox_ready=sandbox_snapshot["ready"] is True,
            diagnostic_docx_ready=docx_capable and model_ready,
            now=now,
            available_engines=engine_registry.available(),
            sandbox_rootful=(
                sandbox_snapshot["rootless"] is False
                and sandbox_snapshot["runtime_kind"] == "docker"
            ),
        )

    model_provider: _SwitchingModelProvider | None = None
    model_arg: DiagnosticModelRuntime | None = model
    command_executor: ModelCommandExecutor | None = None
    if args.models_config is not None:
        # DEC-044 Dodatok D §1 fix (2026-09-27): only a `--models-config`
        # worker can ever be asked for a variant other than the one it
        # booted with -- `--model-bundle` keeps the single fixed `model=`.
        model_provider = _SwitchingModelProvider(
            args=args,
            policy=policy,
            jobs=jobs,
            readiness_writer=readiness_writer,
            initial=loaded,
            engine_registry=engine_registry,
        )
        model_arg = None
        # design §4.2: the persistent executor for commands enqueued after
        # startup (download/load/delete via the API) -- run in the main loop
        # before `worker.run_once()`.
        command_executor = ModelCommandExecutor(
            jobs=jobs,
            worker_id=args.worker_id,
            cache_dir=args.hf_cache_dir,
            variants=variants_by_id,
            model_provider=model_provider,
            heartbeat_interval_seconds=policy["heartbeat_interval_seconds"],
            publish_readiness=publish_readiness,
        )

    component_lock_hash = loaded.identities["component_lock_hash"]
    # The provider (or `model_arg`) now owns the resident model. main()'s own
    # references would keep the boot model on the GPU for the whole process,
    # so a later switch could never free it (live 2026-09-27: switching
    # fp8 -> int4 -> fp8 ran out of memory with the old model still resident).
    del loaded, model
    worker = DiagnosticWorker(
        worker_id=args.worker_id,
        jobs=jobs,
        artifacts=artifact_store,
        sandbox=sandbox,
        model=model_arg,
        model_provider=model_provider,
        policy=policy,
        component_lock_hash=component_lock_hash,
        work_root=args.work_root,
        readiness_heartbeat=publish_readiness,
        diagnostic_docx_capable=docx_capable,
    )
    sweeper = RetentionSweeper(
        store=artifact_store,
        repository=JobStoreRetentionRepository(
            jobs=jobs, store=artifact_store, work_root=args.work_root
        ),
        policy=policy,
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        while True:
            sweeper.sweep(now=time.time())
            # A worker that loses sandbox readiness must not claim a job. The
            # invocation gate provides a second check for changes after this
            # probe and before PDF execution.
            if sandbox.readiness()["ready"] is not True:
                publish_readiness(time.time())
                time.sleep(args.idle_seconds)
                continue
            if command_executor is not None:
                # design §4.2: model commands run before the job claim so a
                # queued switch takes effect for the *next* job, not the one
                # already in flight.
                command_executor.run_once()
            if model_provider is not None and not model_provider.is_loaded:
                publish_readiness(time.time())
                time.sleep(args.idle_seconds)
                continue
            claimed = worker.run_once()
            if claimed:
                _log_gpu_memory()
            else:
                time.sleep(args.idle_seconds)
    except KeyboardInterrupt:
        return 0


@dataclass(frozen=True, slots=True)
class SelectedVariant:
    """A `--models-config` variant resolved to a local bundle directory."""

    variant_id: str
    precision: str
    kv_cache: str
    manifest_sha256: str | None
    #: (T22, design §17.5) the `InferenceEngine` id this variant names in the
    #: models config, default "native" -- carried through so
    #: `_SwitchingModelProvider` can refuse a switch fail-closed instead of
    #: silently loading the wrong engine.
    engine: EngineId = "native"


def _resolve_model_bundle_dir(
    args: argparse.Namespace, *, variant_id: str | None = None
) -> tuple[Path, SelectedVariant | None]:
    """`--model-bundle` keeps today's behaviour unchanged. `--models-config`
    resolves ONE variant (`variant_id`, else `--variant`, else the config's
    `default_variant`) to a bundle directory and returns it alongside the
    variant identity `_resolve_quantization` and the manifest sha256 pin
    need.

    (design §4.2 "Sieť", T32) an `hf` source variant must already be
    downloaded locally -- this function never calls `snapshot_download`
    itself. A not-yet-downloaded `hf` variant is refused fail-closed; only a
    `download`/`load` model command through `ModelCommandExecutor` may fetch
    one (the `--variant auto` startup path in `main()` does exactly that
    before the first call here)."""
    if args.models_config is None:
        return args.model_bundle, None
    config = load_models_config(args.models_config)
    resolved_variant_id = variant_id or args.variant or config["default_variant"]
    variants = {variant["variant_id"]: variant for variant in config["variants"]}
    if resolved_variant_id not in variants:
        raise ValueError(
            f"models config {args.models_config} names no variant {resolved_variant_id!r}"
        )
    variant = variants[resolved_variant_id]
    if not is_published(variant):
        raise ModelsConfigInvalid(
            f"models config variant {resolved_variant_id!r} is not published "
            f"(availability {variant['availability']!r}); it has nothing to load"
        )
    if "bundle" in variant:
        bundle_dir = Path(variant["bundle"])
        manifest_sha256_pin = None
    else:
        source = variant["source"]
        downloaded, _size = is_variant_downloaded(variant, cache_dir=args.hf_cache_dir)
        if not downloaded:
            raise ModelsConfigInvalid(
                f"models config variant {resolved_variant_id!r} source is not downloaded "
                "locally; a 'load' model command must download it first"
            )
        bundle_dir = _hf_cached_snapshot_dir(source, cache_dir=args.hf_cache_dir)
        manifest_sha256_pin = source["bundle_manifest_sha256"]
    return bundle_dir, SelectedVariant(
        variant_id=resolved_variant_id,
        precision=variant["precision"],
        kv_cache=variant["kv_cache"],
        manifest_sha256=manifest_sha256_pin,
        engine=variant["engine"],
    )


def _hf_cached_snapshot_dir(source: Mapping[str, Any], *, cache_dir: Path | None) -> Path:
    """Resolve an already-downloaded `hf` source variant to its local cache
    snapshot directory -- reads the same `scan_cache_dir` view
    `is_variant_downloaded` already checked and never calls
    `snapshot_download` (design §4.2 "Sieť")."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir(cache_dir)
    for repo in info.repos:
        if repo.repo_id != source["repo_id"]:
            continue
        for revision in repo.revisions:
            if revision.commit_hash == source["revision"]:
                snapshot_dir = revision.snapshot_path
                return snapshot_dir / source["subfolder"] if "subfolder" in source else snapshot_dir
    raise ModelsConfigInvalid(
        f"hf source {source['repo_id']!r}@{source['revision']!r} vanished from the local "
        "cache between the download check and the snapshot lookup"
    )


@dataclass(frozen=True, slots=True)
class LoadedVariant:
    """One resolved, verified, loaded model, ready to hand to the worker."""

    model: DiagnosticModelRuntime
    docx_capable: bool
    identities: dict
    selected_variant: SelectedVariant | None


def bind_bundle_loop_guard_caps(model_bundle_dir: Path) -> None:
    """Decode with the loop guard v2 caps the bundle carries. `verify_bundle`
    has already checked that file against the manifest's pinned sha256; the
    runtime otherwise reads the caps file of the code tree, which is only
    right for a bundle whose caps happen to equal that file."""
    from unflatten.model import loop_guard_v2_caps

    caps_path = model_bundle_dir / CAPS_FILENAME
    if not caps_path.is_file():
        raise FileNotFoundError(f"model bundle has no {CAPS_FILENAME}: {caps_path}")
    loop_guard_v2_caps.CAPS_PATH = caps_path


def bind_bundle_backbone(
    model_bundle_dir: Path, identities: Mapping[str, Any], *, model_id: str, revision: str
) -> None:
    """DEC-050 §3: a bundle that carries `backbone/` is self-contained. The
    loaders resolve the base model as `model_id@revision` from the HF hub
    cache, so give them a cache view whose snapshot IS the bundle's
    `backbone/` (files already verified by `verify_bundle`). A bundle
    without a carried backbone keeps using the HF cache as before."""
    import tempfile

    import huggingface_hub.constants as hf_constants

    if identities.get("backbone") is None:
        return
    backbone_dir = model_bundle_dir / BACKBONE_DIRNAME
    if not backbone_dir.is_dir():
        raise FileNotFoundError(
            f"model bundle has no {BACKBONE_DIRNAME}/ directory: {backbone_dir}"
        )
    cache_root = Path(tempfile.mkdtemp(prefix="unflatten-bundle-hf-"))
    snapshot = cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(backbone_dir.resolve(), target_is_directory=True)
    hf_constants.HF_HUB_CACHE = str(cache_root)


def require_bundle_deepstack_matches(identities: Mapping[str, Any], *, deepstack: bool) -> None:
    """DEC-056 point 2, before any weight is loaded: the bundle's
    `identities.visual_feature_tap` (absent == `pooler_output`, every v15b
    bundle) must match the worker's `--deepstack`. `load_candidate_runtime`
    re-checks the checkpoint's own stamped tap after this."""
    tap = visual_feature_tap_from_identities(identities)
    if deepstack != (tap != "pooler_output"):
        raise ModelBundleInvalid(
            f"bundle visual_feature_tap {tap!r} but the worker runs with "
            f"--deepstack={deepstack} -- a bundle trained without DeepStack never runs "
            "with it, and one trained with it never without"
        )


def _load_variant(
    args: argparse.Namespace,
    policy,
    *,
    selected_variant: SelectedVariant | None,
) -> LoadedVariant:
    """Resolve (if `selected_variant` is None, from `--model-bundle` or
    `--models-config`'s default), verify, and load ONE variant. Shared by
    startup and `_SwitchingModelProvider` so a later switch runs exactly the
    same verification path as the first load -- never a shortcut."""
    if selected_variant is None:
        model_bundle_dir, selected_variant = _resolve_model_bundle_dir(args)
    else:
        model_bundle_dir, selected_variant = _resolve_model_bundle_dir(
            args, variant_id=selected_variant.variant_id
        )
    quantization = _resolve_quantization(args, selected_variant)
    # --model-bundle-sha256 pins the single --model-bundle; it must never be
    # applied to another variant's bundle (live 2026-09-27: switching fp8 -> int4
    # verified the int4 bundle against the fp8 pin and failed ModelBundleInvalid).
    if selected_variant is not None:
        expected_manifest_sha256 = selected_variant.manifest_sha256
    else:
        expected_manifest_sha256 = args.model_bundle_sha256
    manifest = verify_bundle(
        model_bundle_dir,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_precision_spec_hash=quantization.spec_hash,
    )
    identities = manifest["identities"]
    require_bundle_deepstack_matches(identities, deepstack=args.deepstack)
    bind_bundle_loop_guard_caps(model_bundle_dir)
    bind_bundle_backbone(
        model_bundle_dir,
        identities,
        model_id=args.model_id,
        revision=args.revision,
    )
    # Only an fp8 decoder can share its input quantization. An explicit
    # non-fp8 --precision was already refused in `main`; a non-fp8 variant
    # resolved here (models-config switch, auto) loads unshared instead --
    # a switch has already freed the resident model, so refusing would
    # leave the worker with none. The log line below records both values.
    fp8_shared_input_quant = (
        args.fp8_shared_input_quant is not False and quantization.decoder_precision == "fp8"
    )
    decode_block = _load_decode_policy_block(model_bundle_dir)
    decode_policy = DecodePolicy(
        loop_guard=decode_block["decode"].get("loop_guard"),
        max_steps=decode_block["decode"]["max_steps"],
        typed_decode=decode_block["decode"]["typed_decode"],
        region_decode=decode_block["decode"]["region_decode"],
        selection_policy_hash=decode_block["selection_policy_hash"],
        # W0-11: absent in every bundle written before this flag existed --
        # `.get` keeps loading them with today's byte-identical decode.
        prefill_chunk_tokens=decode_block["decode"].get("prefill_chunk_tokens"),
    )
    decode_policy = apply_loop_guard_override(
        decode_policy, getattr(args, "loop_guard", None), getattr(args, "loop_guard_v3_arm", None)
    )
    checkpoint_path = model_bundle_dir / CHECKPOINT_FILENAME
    if selected_variant is not None and selected_variant.manifest_sha256 is not None:
        # Verified HF snapshots use links into the immutable blob cache. Resolve
        # only this pinned source; the runtime still O_NOFOLLOW-opens the regular
        # blob and checks its raw SHA-256 before deserializing any weights.
        checkpoint_path = checkpoint_path.resolve(strict=True)
    candidate = load_candidate_runtime(
        checkpoint_path=checkpoint_path,
        component_lock_path=model_bundle_dir / COMPONENT_LOCK_FILENAME,
        allowlist_path=model_bundle_dir / ALLOWLIST_FILENAME,
        tokenizer_artifact_set_hash=identities["tokenizer_artifact_set_hash"],
        pin=RuntimeIdentityPin(
            model_id=args.model_id,
            revision=args.revision,
            attention_implementation=args.attention_implementation,
            checkpoint_sha256=identities["checkpoint_sha256"],
            component_lock_hash=identities["component_lock_hash"],
        ),
        device=args.device,
        decode_policy=decode_policy,
        caps_sha256_pin=identities["loop_guard_v2_caps_sha256"],
        decode_steps_hard_ceiling=policy["decode_steps_hard_ceiling"],
        quantization=quantization,
        kv_cache_mode=args.kv_cache_mode,
        **({"deepstack": True} if args.deepstack else {}),
        fp8_shared_input_quant=fp8_shared_input_quant,
    )
    model = DiagnosticModelRuntime(candidate)
    docx_capable = diagnostic_docx_capability(model)
    LOGGER.info(
        "worker_model_bundle_loaded bundle=%s manifest_sha256=%s diagnostic_docx_ready=%s "
        "variant=%s kv_cache_mode=%s kv_cache_effective=%s "
        "fp8_shared_input_quant=%s fp8_shared_input_quant_requested=%s",
        model_bundle_dir,
        manifest_sha256(model_bundle_dir),
        docx_capable,
        selected_variant.variant_id if selected_variant is not None else None,
        args.kv_cache_mode,
        candidate.kv_cache_effective(),
        fp8_shared_input_quant,
        args.fp8_shared_input_quant is True,
    )
    return LoadedVariant(
        model=model,
        docx_capable=docx_capable,
        identities=identities,
        selected_variant=selected_variant,
    )


def _measure_own_device_facts(device: str) -> dict[str, Any] | None:
    """This process's own measured VRAM/compute capability, `None` without a
    CUDA device. `--variant auto` resolves against this (design §4.2
    "Štartovací auto-load": "vlastné merania") rather than
    `jobs.model_state()`, which is unpublished before any worker has ever
    run."""
    import torch

    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = cuda_memory_info(device)
    capability = torch.cuda.get_device_capability(device)
    return {
        "free_vram_mib": free_bytes / (1024 * 1024),
        "total_vram_mib": total_bytes / (1024 * 1024),
        "compute_capability": capability,
    }


def _resolve_auto_variant(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    engine_registry: EngineRegistry,
    jobs: JobStore,
) -> str | None:
    """`--variant auto` (design §4.2 + §17.5/§17.6 item A.6): resolve among
    variants whose engine is available on this worker's `--engines`, using
    this process's own measured device facts. `resolve_default_variant`'s
    signature takes one `config` and has no engine concept, so the
    unavailable-engine variants are filtered out of `config['variants']`
    before the call (plan note 6) rather than inside it. Returns `None` when
    nothing fits or there is no CUDA device to measure."""
    available = set(engine_registry.available())
    candidates = [variant for variant in config["variants"] if variant["engine"] in available]
    if not candidates:
        return None
    device_facts = _measure_own_device_facts(args.device)
    if device_facts is None:
        return None
    variant_id, _source = resolve_default_variant(
        {**config, "variants": candidates},
        user_default=jobs.get_setting("default_variant"),
        model_state=device_facts,
    )
    if variant_id is None:
        return None
    chosen = next(v for v in candidates if v["variant_id"] == variant_id)
    if variant_fit_reason(
        chosen["precision"], chosen["kv_cache"],
        free_vram_mib=device_facts["free_vram_mib"],
        compute_capability=device_facts["compute_capability"],
    ) is not None:
        return None
    return variant_id


class _BootstrapModelProvider:
    """One-shot `ModelProvider` for the `--variant auto` startup `load`
    command (design §4.2 "Štartovací auto-load"): runs the real
    `_load_variant` verification chain exactly once, through
    `ModelCommandExecutor`, and remembers the result so `main()` can hand it
    straight to `DiagnosticWorker`/`_SwitchingModelProvider` instead of
    loading the model twice."""

    def __init__(self, args: argparse.Namespace, policy: Mapping[str, Any]) -> None:
        self._args = args
        self._policy = policy
        self.loaded: LoadedVariant | None = None

    def ensure_loaded(self, variant_id: str | None) -> DiagnosticModelRuntime:
        self.loaded = _load_variant(
            self._args,
            self._policy,
            selected_variant=SelectedVariant(
                variant_id=variant_id, precision="", kv_cache="", manifest_sha256=None
            ),
        )
        return self.loaded.model


class _SwitchingModelProvider:
    """DEC-044 Dodatok D §1 fix (2026-09-27): the real live-switching
    `ModelProvider` for a `--models-config` worker. Assumes every variant in
    one models config shares the same `component_lock_hash` (same base
    checkpoint, only quantization differs) -- `DiagnosticWorker` pins that
    hash once at construction, so a config mixing architectures would need
    a design change here, not just a new variant entry.

    (T22 fold, design §17.5): tracks the resident model as a `(variant_id,
    engine)` pair via an `EngineRegistry`. Switching to a variant whose
    `engine` is not in the registry's `available()` set -- or names anything
    other than 'native', the only engine `_load_variant` loads -- is refused
    fail-closed rather than silently running the wrong engine."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        policy,
        jobs: JobStore,
        readiness_writer,
        initial: LoadedVariant,
        engine_registry: EngineRegistry | None = None,
    ) -> None:
        self._args = args
        self._policy = policy
        self._jobs = jobs
        self._readiness_writer = readiness_writer
        self._loaded: LoadedVariant | None = initial
        self._fallback_variant = (
            initial.selected_variant.variant_id if initial.selected_variant is not None else None
        )
        self._component_lock_hash = initial.identities["component_lock_hash"]
        self._engine_registry = engine_registry or EngineRegistry(
            factories={"native": NativeEngine}
        )
        self._resident: tuple[str | None, InferenceEngine] | None = self._wrap(initial)

    def _wrap(self, loaded: LoadedVariant) -> tuple[str | None, InferenceEngine]:
        variant = loaded.selected_variant
        variant_id = variant.variant_id if variant is not None else None
        engine_id: EngineId = variant.engine if variant is not None else "native"
        if engine_id not in self._engine_registry.available():
            raise ValueError(
                f"variant {variant_id!r} names engine {engine_id!r}, which is not "
                "available on this worker (--engines)"
            )
        if engine_id != "native":
            raise ValueError(
                f"variant {variant_id!r} names engine {engine_id!r}; only 'native' "
                "variant loading is implemented"
            )
        return (variant_id, NativeEngine.from_loaded(loaded.model))

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    def ensure_loaded(self, variant_id: str | None) -> DiagnosticModelRuntime:
        current_id = self._resident[0] if self._resident is not None else None
        if self._loaded is not None and (variant_id is None or variant_id == current_id):
            return self._loaded.model
        target = variant_id or self._fallback_variant
        if target is None:
            raise EngineUnavailable("no model variant selected")
        LOGGER.info("worker_model_switch_started from_variant=%s to_variant=%s", current_id, target)
        started = time.time()
        self._free_resident_model()
        loaded = _load_variant(
            self._args,
            self._policy,
            selected_variant=SelectedVariant(
                variant_id=target, precision="", kv_cache="", manifest_sha256=None
            ),
        )
        if loaded.identities["component_lock_hash"] != self._component_lock_hash:
            raise ValueError(
                f"variant {target!r} has a different component_lock_hash than the worker "
                "was constructed with; a models config must name variants of one architecture"
            )
        resident = self._wrap(loaded)
        self._loaded = loaded
        self._resident = resident
        self._fallback_variant = target
        self._publish_state()
        LOGGER.info(
            "worker_model_switch_finished from_variant=%s to_variant=%s seconds=%.1f",
            current_id,
            target,
            time.time() - started,
        )
        return loaded.model

    def _free_resident_model(self) -> None:
        resident = self._resident
        self._resident = None
        self._loaded = None
        if resident is not None:
            resident[1].unload()
        self._publish_state()

    def _publish_state(self) -> None:
        _publish_model_state(
            self._jobs,
            self._readiness_writer,
            self._args.device,
            self._loaded.selected_variant if self._loaded is not None else None,
        )


def _publish_model_state(
    jobs: JobStore, readiness_writer, device: str, selected_variant: SelectedVariant | None
) -> None:
    """Best-effort device facts publication (design GET /v1/models): a
    process without a CUDA device (e.g. a CPU-only smoke test) simply never
    publishes -- `JobStore.model_state()` then correctly reports unknown
    rather than a fabricated zero."""
    import torch

    if not torch.cuda.is_available():
        return
    free_bytes, total_bytes = cuda_memory_info(device)
    capability = torch.cuda.get_device_capability(device)
    jobs.publish_model_state(
        readiness_writer,
        resident_variant=selected_variant.variant_id if selected_variant is not None else None,
        free_vram_mib=free_bytes / (1024 * 1024),
        total_vram_mib=total_bytes / (1024 * 1024),
        compute_capability=capability,
        now=time.time(),
    )


def _resolve_quantization(
    args: argparse.Namespace, selected_variant: SelectedVariant | None
) -> QuantizationSpec:
    """`--precision bf16/int8/fp8/int4` picks that spec directly (today's
    behaviour, byte-identical for the `bf16` default). `--precision auto`
    instead measures THIS process's real free VRAM and compute capability
    (`torch.cuda.mem_get_info`/`get_device_capability` on `args.device`) and
    resolves through `quantization.auto_policy` -- general across GPUs, not
    keyed on one card. `--kv-cache` is ignored under `auto` (the policy
    decides both halves together). A `--models-config` selected variant fixes
    the precision/kv_cache directly, ignoring `--precision`/`--kv-cache`."""
    if selected_variant is not None:
        return QuantizationSpec(
            decoder_precision=selected_variant.precision,
            kv_cache_dtype=selected_variant.kv_cache,
        )
    if args.precision != "auto":
        return QuantizationSpec(decoder_precision=args.precision, kv_cache_dtype=args.kv_cache)
    import torch

    free_bytes, total_bytes = cuda_memory_info(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    spec = resolve_auto_quantization(
        free_vram_bytes=free_bytes,
        total_vram_bytes=total_bytes,
        compute_capability=capability,
        extra_reserve_mib=static_kv_reserve_mib(args.kv_cache_mode, load_auto_policy()),
    )
    LOGGER.info(
        "worker_auto_precision_resolved decoder_precision=%s kv_cache_dtype=%s "
        "free_vram_mib=%.1f total_vram_mib=%.1f compute_capability=%s",
        spec.decoder_precision,
        spec.kv_cache_dtype,
        free_bytes / (1024 * 1024),
        total_bytes / (1024 * 1024),
        capability,
    )
    return spec


def _load_decode_policy_block(bundle_dir: Path) -> dict:
    import json

    return json.loads((bundle_dir / DECODE_POLICY_FILENAME).read_text(encoding="utf-8"))


def _log_gpu_memory() -> None:
    """P6: `worker_gpu_memory max_allocated_mib=…` after each claimed job --
    the number the D16/D11 acceptance run closes VRAM OPEN against."""
    import torch

    if not torch.cuda.is_available():
        return
    max_allocated_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    LOGGER.info("worker_gpu_memory max_allocated_mib=%.1f", max_allocated_mib)


if __name__ == "__main__":
    raise SystemExit(main())

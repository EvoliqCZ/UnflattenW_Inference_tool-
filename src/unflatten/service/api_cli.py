"""Uvicorn entry point for the private diagnostic API process."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from unflatten.service.api import ApiDependencies, WebSettings, create_app
from unflatten.service.artifact_store import ArtifactStore
from unflatten.service.auth import load_principal_store
from unflatten.service.job_store import JobStore
from unflatten.service.model_bundle import load_manifest, verify_bundle
from unflatten.service.models_config import load_models_config
from unflatten.service.policy import load_runtime_policy, runtime_policy_hash
from unflatten.service.readiness import (
    COMPONENT_CODES,
    diagnostic_docx_status,
    job_store_component_statuses,
)


def _env_flag(name: str) -> bool:
    """`1` on, `0`/empty/unset off; anything else is refused rather than
    guessed (a typo must never silently switch a security flag either way)."""
    value = os.environ.get(name, "")
    if value not in ("", "0", "1"):
        raise ValueError(f"{name} must be 0, 1 or empty, got {value!r}")
    return value == "1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the private Unflatten diagnostic API")
    parser.add_argument("--principal-file", type=Path, required=True)
    parser.add_argument("--job-store", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=None)
    identity_source = parser.add_mutually_exclusive_group(required=True)
    identity_source.add_argument(
        "--model-bundle",
        type=Path,
        help="D7 model bundle directory -- only its manifest's "
        "identities.component_lock_hash is ever read (design D9 §6.1); the API "
        "never loads a model and never imports torch",
    )
    identity_source.add_argument(
        "--bootstrap-model-manifest",
        type=Path,
        help=(
            "Pinned catalog metadata for serving setup before model download; "
            "never marks the model ready"
        ),
    )
    parser.add_argument(
        "--model-bundle-sha256",
        default=None,
        help="optional pin on the bundle manifest's own sha256",
    )
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--models-config",
        type=Path,
        default=os.environ.get("UNFLATTEN_MODELS_CONFIG") or None,
        help="DEC-044 Dodatok D §1 models variants config (UNFLATTEN_MODELS_CONFIG); "
        "absent keeps today's single-bundle GET /v1/models response (empty, unconfigured)",
    )
    parser.add_argument(
        "--models-cache-dir",
        type=Path,
        default=None,
        help="local HF cache directory GET /v1/models scans for hf source "
        "variants' 'downloaded' status, and the base-model pin check verifies "
        "against; defaults to HF_HOME/hub if HF_HOME is set (T43: this is also "
        "the cache_dir verify_bundle() uses for the base weights pin)",
    )
    parser.add_argument(
        "--web-ui",
        choices=["legacy", "spa", "none"],
        default=os.environ.get("UNFLATTEN_WEB_UI", "legacy"),
        help="design §6.1: 'legacy' serves today's single UI page, 'spa' "
        "serves the built SPA, 'none' runs headless with API and Swagger only "
        "(UNFLATTEN_WEB_UI)",
    )
    parser.add_argument(
        "--web-allowed-hosts",
        default=os.environ.get("UNFLATTEN_WEB_ALLOWED_HOSTS", "127.0.0.1,localhost,[::1]"),
        help="comma-separated Host header allowlist, port never included "
        "(design §2.5, UNFLATTEN_WEB_ALLOWED_HOSTS)",
    )
    parser.add_argument(
        "--web-session-idle-seconds",
        type=int,
        default=int(os.environ.get("UNFLATTEN_WEB_SESSION_IDLE_SECONDS", "43200")),
        help="design §2.2 (UNFLATTEN_WEB_SESSION_IDLE_SECONDS)",
    )
    parser.add_argument(
        "--web-session-max-seconds",
        type=int,
        default=int(os.environ.get("UNFLATTEN_WEB_SESSION_MAX_SECONDS", "604800")),
        help="design §2.2; must be >= --web-session-idle-seconds or the "
        "process refuses to start (UNFLATTEN_WEB_SESSION_MAX_SECONDS)",
    )
    parser.add_argument(
        "--web-auth-file",
        type=Path,
        default=None,
        help="defaults to <dir of --job-store>/web-auth.json (design §2.1)",
    )
    parser.add_argument(
        "--web-setup-tokenless",
        action="store_true",
        default=_env_flag("UNFLATTEN_SETUP_TOKENLESS"),
        help="first-run POST /v1/setup needs no bootstrap bearer token; set ONLY "
        "when the port is published on loopback (owner 2026-09-29, DEC-052 "
        "addendum; run-local.sh/dev-native.sh decide it, UNFLATTEN_SETUP_TOKENLESS=1)",
    )
    parser.add_argument(
        "--git-sha",
        default=os.environ.get("UNFLATTEN_GIT_SHA", "unknown"),
        help="design §3.10: reported in GET /v1/settings (UNFLATTEN_GIT_SHA)",
    )
    return parser


def _resolve_hf_cache_dir(explicit: Path | None) -> Path | None:
    """`--models-cache-dir`, else `HF_HOME`/`hub` -- huggingface_hub's own
    cache functions (`try_to_load_from_cache`, `scan_cache_dir`) treat their
    `cache_dir` argument as the HF_HUB_CACHE-style hub directory, not the
    HF_HOME parent it lives under (HF_HUB_CACHE defaults to `HF_HOME/hub`).
    Passing bare `HF_HOME` straight through as `cache_dir` looks one level
    too high and never finds a real snapshot."""
    if explicit is not None:
        return explicit
    hf_home = os.environ.get("HF_HOME")
    return Path(hf_home) / "hub" if hf_home else None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # A CLI pin lets the launcher give API and worker one explicit identity.
    # Preserve the environment fallback for direct deployments; an exported
    # empty value is deliberately equivalent to no pin.
    expected_manifest_sha256 = args.model_bundle_sha256 or os.environ.get(
        "UNFLATTEN_MODEL_BUNDLE_SHA256"
    )
    hf_cache_dir = _resolve_hf_cache_dir(args.models_cache_dir)
    if args.bootstrap_model_manifest is not None:
        if args.models_config is None:
            raise ValueError("bootstrap manifest requires --models-config")
        catalog = load_models_config(args.models_config)
        variants = [v for v in catalog["variants"] if v["availability"] == "published"]
        if len(variants) != 1 or "source" not in variants[0]:
            raise ValueError("bootstrap manifest requires one pinned HF catalog variant")
        expected = variants[0]["source"]["bundle_manifest_sha256"]
        actual = "sha256:" + hashlib.sha256(args.bootstrap_model_manifest.read_bytes()).hexdigest()
        if actual != expected or (
            expected_manifest_sha256 and expected_manifest_sha256 != expected
        ):
            raise ValueError("bootstrap manifest does not match the catalog manifest pin")
        if args.bootstrap_model_manifest.name != "manifest.json":
            raise ValueError("bootstrap manifest must be named manifest.json")
        # Metadata supplies only API identity. Worker readiness remains false until
        # the worker downloads and independently verifies the real complete bundle.
        bundle = load_manifest(args.bootstrap_model_manifest.parent)
    else:
        bundle = verify_bundle(
            args.model_bundle,
            expected_manifest_sha256=expected_manifest_sha256 or None,
            hf_cache_dir=hf_cache_dir,
        )
    component_lock_hash = bundle["identities"]["component_lock_hash"]
    policy = load_runtime_policy(args.policy) if args.policy is not None else load_runtime_policy()
    jobs = JobStore(
        args.job_store,
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    jobs.initialize()
    base_statuses = dict.fromkeys(COMPONENT_CODES, False)
    base_statuses.update(
        {
            "runtime_policy": True,
            "principal_store": True,
            "job_store": jobs.journal_mode() == "wal",
            "artifact_store": True,
        }
    )

    def statuses() -> dict[str, bool]:
        return {
            **base_statuses,
            **job_store_component_statuses(
                jobs,
                now=time.time(),
                stale_after_seconds=2 * policy["heartbeat_interval_seconds"],
            ),
        }

    web_settings = WebSettings(
        allowed_hosts=tuple(
            host.strip() for host in args.web_allowed_hosts.split(",") if host.strip()
        ),
        session_idle_seconds=args.web_session_idle_seconds,
        session_max_seconds=args.web_session_max_seconds,
        web_auth_path=args.web_auth_file or (args.job_store.parent / "web-auth.json"),
        setup_tokenless=args.web_setup_tokenless,
        git_sha=args.git_sha,
    )
    dependencies = ApiDependencies(
        principal_store=load_principal_store(args.principal_file),
        job_store=jobs,
        artifact_store=ArtifactStore(args.artifact_root),
        work_root=args.work_root or args.artifact_root.parent / "work",
        runtime_policy=policy,
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash=component_lock_hash,
        component_statuses=statuses,
        diagnostic_docx_status=lambda: diagnostic_docx_status(jobs),
        clock=time.time,
        models_config=(
            load_models_config(args.models_config) if args.models_config is not None else None
        ),
        models_cache_dir=hf_cache_dir,
        web=web_settings,
        web_ui=args.web_ui,
    )
    uvicorn.run(create_app(dependencies), host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the process supervisor
    raise SystemExit(main())

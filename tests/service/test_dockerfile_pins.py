"""T38: `deploy/docker/Dockerfile`'s `web-builder` stage and its wiring into
`runtime-api`, plus `entrypoint-api.sh`'s env->flag mapping for the web UI
(design §8.1/§8.4). No Docker daemon is used here -- these tests read the
Dockerfile text and run the real entrypoint script with a stub `python3`.

T43 adds `models.example.json` shape checks (no docker daemon) and
`compose.yaml` structure checks via `docker compose config` (needs a docker
daemon and the `docker compose` CLI -- skipped if either is unavailable)."""

from __future__ import annotations

import json
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from unflatten.service.api import _WEB_DIST_DIR
from unflatten.service.models_config import (
    ModelsConfigInvalid,
    load_models_config,
    validate_models_config,
)

DOCKERFILE = Path(__file__).parents[2] / "deploy" / "docker" / "Dockerfile"
ENTRYPOINT_API = Path(__file__).parents[2] / "deploy" / "docker" / "entrypoint-api.sh"
DEPLOY_DOCKER_DIR = Path(__file__).parents[2] / "deploy" / "docker"
MODELS_EXAMPLE_JSON = DEPLOY_DOCKER_DIR / "models.example.json"

_NODE_BASE_PATTERN = re.compile(
    r"^ARG NODE_BASE=docker\.io/library/node:22-[a-z-]+@sha256:([0-9a-f]{64})$",
    re.MULTILINE,
)


def test_dockerfile_pins_node_by_digest() -> None:
    text = DOCKERFILE.read_text()

    match = _NODE_BASE_PATTERN.search(text)

    assert match is not None, "NODE_BASE must be pinned to a node:22-* image by @sha256:<64 hex>"
    assert "REPLACE_WITH" not in text.split("NODE_BASE", 1)[1].splitlines()[0]
    # a real digest is exactly 64 hex chars -- catches an accidentally
    # truncated or non-hex placeholder slipping past the regex above.
    digest = match.group(1)
    assert len(digest) == 64
    int(digest, 16)


def test_dockerfile_web_builder_stage_runs_npm_ci_and_build() -> None:
    text = DOCKERFILE.read_text()

    assert "FROM ${NODE_BASE} AS web-builder" in text
    assert "npm ci --ignore-scripts" in text
    assert "npm run build" in text


def test_runtime_api_copies_web_dist() -> None:
    text = DOCKERFILE.read_text()

    # _WEB_DIST_DIR is `.../src/unflatten/service/ui/dist` at import time
    # (repo checkout); the image root is `/app`, so the destination must be
    # the same `src/unflatten/service/ui/dist` suffix under `/app`.
    relative_dist = _WEB_DIST_DIR.relative_to(
        _WEB_DIST_DIR.parents[4]
    )  # src/unflatten/service/ui/dist
    expected_dest = f"/app/{relative_dist.as_posix()}"

    copy_lines = [line for line in text.splitlines() if line.startswith("COPY --from=web-builder")]

    assert copy_lines, "runtime-api must COPY --from=web-builder the built SPA"
    (copy_line,) = copy_lines
    assert copy_line.endswith(f" {expected_dest}")
    assert "/web/dist " in copy_line

    # must land in the runtime-api stage, before it drops root (COPY --chown
    # needs write access to a path not yet owned by the non-root user).
    runtime_api_stage = text.split("FROM runtime-base AS runtime-api", 1)[1]
    runtime_api_stage = runtime_api_stage.split("FROM runtime-base AS runtime-worker", 1)[0]
    assert copy_line in runtime_api_stage
    assert runtime_api_stage.index(copy_line) < runtime_api_stage.index("USER 65532:65532")


def test_dockerfile_sets_git_sha_arg_and_env_in_runtime_api() -> None:
    text = DOCKERFILE.read_text()
    runtime_api_stage = text.split("FROM runtime-base AS runtime-api", 1)[1]
    runtime_api_stage = runtime_api_stage.split("FROM runtime-base AS runtime-worker", 1)[0]

    assert "ARG UNFLATTEN_GIT_SHA=unknown" in runtime_api_stage
    assert "ENV UNFLATTEN_GIT_SHA=${UNFLATTEN_GIT_SHA}" in runtime_api_stage


def _run_entrypoint_api(tmp_path: Path, env_overrides: dict[str, str]) -> list[str]:
    """Runs the real entrypoint-api.sh with a stub `python3` first on PATH
    that dumps its argv (one per line) to a capture file instead of starting
    uvicorn, then returns the captured argv list."""
    capture_file = tmp_path / "argv.txt"
    stub_dir = tmp_path / "stubbin"
    stub_dir.mkdir()
    stub_python = stub_dir / "python3"
    stub_python.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" > "{capture_file}"\n'
    )
    stub_python.chmod(stub_python.stat().st_mode | stat.S_IEXEC)

    env = {
        "PATH": f"{stub_dir}:/usr/bin:/bin",
        "UNFLATTEN_DATA_DIR": str(tmp_path / "data"),
        "UNFLATTEN_MODEL_BUNDLE": str(tmp_path / "models"),
        **env_overrides,
    }
    subprocess.run(
        ["bash", str(ENTRYPOINT_API)], env=env, capture_output=True, text=True, check=True
    )

    return capture_file.read_text().splitlines()


def test_entrypoint_api_maps_web_env(tmp_path: Path) -> None:
    argv = _run_entrypoint_api(
        tmp_path,
        {
            "UNFLATTEN_WEB_UI": "spa",
            "UNFLATTEN_WEB_ALLOWED_HOSTS": "127.0.0.1,localhost",
            "UNFLATTEN_WEB_SESSION_IDLE_SECONDS": "60",
            "UNFLATTEN_WEB_SESSION_MAX_SECONDS": "120",
            "UNFLATTEN_WEB_AUTH_FILE": str(tmp_path / "web-auth.json"),
            "UNFLATTEN_GIT_SHA": "deadbeef",
        },
    )

    def flag_value(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert flag_value("--web-ui") == "spa"
    assert flag_value("--web-allowed-hosts") == "127.0.0.1,localhost"
    assert flag_value("--web-session-idle-seconds") == "60"
    assert flag_value("--web-session-max-seconds") == "120"
    assert flag_value("--web-auth-file") == str(tmp_path / "web-auth.json")
    assert flag_value("--git-sha") == "deadbeef"


def test_entrypoint_api_omits_web_flags_when_env_unset(tmp_path: Path) -> None:
    argv = _run_entrypoint_api(tmp_path, {})

    for flag in (
        "--web-ui",
        "--web-allowed-hosts",
        "--web-session-idle-seconds",
        "--web-session-max-seconds",
        "--web-auth-file",
        "--git-sha",
    ):
        assert flag not in argv, f"{flag} must not be passed when its env var is unset"


def test_models_example_catalogue_validates_without_placeholders() -> None:
    """Owner request 2026-09-29: the example lists BF16, FP8 and INT8 as
    `unpublished` (DEC-050: the HF repos do not exist yet, publishing is the
    owner's action), Unflatten W MoE as `planned`, plus the local bundle. It
    validates as-is, and carries no fake repo link or REPLACE_* placeholder
    a UI could ever show."""
    config = load_models_config(MODELS_EXAMPLE_JSON)

    assert "REPLACE" not in MODELS_EXAMPLE_JSON.read_text(encoding="utf-8")
    by_id = {variant["variant_id"]: variant for variant in config["variants"]}
    assert {
        variant_id: (variant["availability"], variant["precision"], variant["kv_cache"])
        for variant_id, variant in by_id.items()
    } == {
        "bf16-hf": ("unpublished", "bf16", "bf16"),
        "fp8-hf": ("unpublished", "fp8", "fp8"),
        "int8-hf": ("unpublished", "int8", "bf16"),
        "unflatten-w-moe": ("planned", None, None),
        "local-bundle": ("published", "bf16", "bf16"),
    }
    assert not any("source" in variant for variant in config["variants"])
    assert config["default_variant"] == "auto"


def test_models_example_placeholder_hf_source_still_rejected() -> None:
    """T43's guard stays: a published `hf` source with placeholder values is
    refused (unpinned revision/sha256), never accepted as loadable."""
    payload = json.loads(MODELS_EXAMPLE_JSON.read_text(encoding="utf-8"))
    payload["variants"][0] = {
        "variant_id": "bf16-hf",
        "label": "BF16",
        "precision": "bf16",
        "kv_cache": "bf16",
        "source": {
            "kind": "hf",
            "repo_id": "REPLACE_WITH_HF_REPO_ID",
            "revision": "REPLACE_WITH_PINNED_40_HEX_COMMIT_SHA",
            "bundle_manifest_sha256": "sha256:REPLACE_WITH_64_HEX_BUNDLE_MANIFEST_SHA256",
        },
    }
    with pytest.raises(ModelsConfigInvalid):
        validate_models_config(payload)


def test_models_example_bundle_shape_valid() -> None:
    """The example's local `bundle` variant is a real, valid shape on its
    own -- extracted into a minimal config (the operator's usual
    `dev-bundle.sh`-style models.json) it must pass validation."""
    payload = json.loads(MODELS_EXAMPLE_JSON.read_text(encoding="utf-8"))
    bundle_variants = [v for v in payload["variants"] if "bundle" in v]
    assert bundle_variants, "models.example.json must include a 'bundle' variant"

    normalized = validate_models_config(
        {
            "schema_version": payload["schema_version"],
            "default_variant": bundle_variants[0]["variant_id"],
            "variants": bundle_variants,
        }
    )

    assert normalized["variants"][0]["bundle"] == bundle_variants[0]["bundle"]


def _docker_compose_config(
    *, profile: str | None = None, extra_env: str = ""
) -> subprocess.CompletedProcess:
    """Runs `docker compose config` against a throwaway `deploy/docker/.env`
    (required: `env_file: - .env` in `compose.yaml` needs the file to exist,
    not just interpolation variables in the process environment), backing up
    and restoring any real `.env` a developer already has there."""
    import os

    env_path = DEPLOY_DOCKER_DIR / ".env"
    backup = env_path.read_bytes() if env_path.exists() else None
    env_path.write_text(
        "UNFLATTEN_MODEL_REVISION=" + "a" * 40 + "\n"
        "UNFLATTEN_HOST_DATA_DIR=/tmp/unflatten-t43-compose-config-test\n" + extra_env,
        encoding="utf-8",
    )
    try:
        cmd = ["docker", "compose", "-f", "compose.yaml"]
        if profile is not None:
            cmd += ["--profile", profile]
        cmd.append("config")
        return subprocess.run(
            cmd,
            cwd=DEPLOY_DOCKER_DIR,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
        )
    finally:
        if backup is None:
            env_path.unlink()
        else:
            env_path.write_bytes(backup)


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_compose_api_sets_hf_home() -> None:
    result = _docker_compose_config()
    assert result.returncode == 0, result.stderr

    config = _yaml_load(result.stdout)
    assert config["services"]["api"]["environment"]["HF_HOME"] == "/data/hf-cache"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_compose_api_never_receives_hf_token() -> None:
    """Item 10: `HF_TOKEN` is only ever used by the worker's
    `entrypoint-worker.sh` (HF downloads at model-load time) -- the `api`
    service must not inherit the real secret via its blanket `env_file: .env`
    just because `.env` also configures the worker."""
    result = _docker_compose_config(extra_env="HF_TOKEN=super-secret-value\n")
    assert result.returncode == 0, result.stderr

    config = _yaml_load(result.stdout)
    assert config["services"]["api"]["environment"].get("HF_TOKEN") != "super-secret-value"
    # the worker still gets it -- this is a scoping fix, not a removal.
    assert config["services"]["worker"]["environment"]["HF_TOKEN"] == "super-secret-value"


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_compose_api_defaults_to_loopback_publish_and_token_setup() -> None:
    """Owner 2026-09-29 (DEC-052 addendum): a bare `docker compose up`
    publishes on loopback and keeps the bootstrap token required -- only
    run-local.sh exports `UNFLATTEN_SETUP_TOKENLESS=1`, after checking the
    bind address."""
    result = _docker_compose_config(extra_env="UNFLATTEN_SETUP_TOKENLESS=\n")
    assert result.returncode == 0, result.stderr

    api_service = _yaml_load(result.stdout)["services"]["api"]
    assert api_service["environment"]["UNFLATTEN_SETUP_TOKENLESS"] == "0"
    assert [port["host_ip"] for port in api_service["ports"]] == ["127.0.0.1"]


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_compose_api_bind_address_is_the_published_host_ip() -> None:
    result = _docker_compose_config(extra_env="UNFLATTEN_BIND_ADDRESS=192.168.1.20\n")
    assert result.returncode == 0, result.stderr

    api_service = _yaml_load(result.stdout)["services"]["api"]
    assert [port["host_ip"] for port in api_service["ports"]] == ["192.168.1.20"]


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_compose_worker_depends_on_sandbox_image() -> None:
    result = _docker_compose_config()
    assert result.returncode == 0, result.stderr

    config = _yaml_load(result.stdout)
    assert "sandbox-image" in config["services"]
    worker_depends_on = config["services"]["worker"]["depends_on"]
    assert worker_depends_on["sandbox-image"]["condition"] == "service_completed_successfully"


def _yaml_load(text: str):
    import yaml

    return yaml.safe_load(text)


# --- CP5 regression: .dockerignore must not exclude anything `tsc -b` needs
# for `web/tsconfig.node.json` (which `include`s `playwright.config.ts`,
# whose relative imports since T41/T46 reach into `web/e2e/`). The build
# passed outside Docker (full checkout) and failed only inside Docker
# (`.dockerignore` stripped `web/e2e/` from the build context) --
# TS2307 "Cannot find module" on `./e2e/fixtures/ports` and
# `./e2e/prepare-models-cache.mjs`.

ROOT_DIR = Path(__file__).parents[2]
DOCKERIGNORE = ROOT_DIR / ".dockerignore"
WEB_DIR = ROOT_DIR / "web"


def _dockerignore_excludes(pattern: str, rel_path: str) -> bool:
    """Minimal .dockerignore semantics for the patterns this repo's
    .dockerignore actually uses: directory patterns (trailing `/`) exclude
    the directory and everything under it; bare filename globs (no `/`)
    match by basename; anything else matches by path prefix."""
    import fnmatch

    if pattern.endswith("/"):
        prefix = pattern.rstrip("/")
        return rel_path == prefix or rel_path.startswith(prefix + "/")
    if "/" not in pattern:
        return fnmatch.fnmatch(Path(rel_path).name, pattern)
    return rel_path == pattern or rel_path.startswith(pattern + "/")


def _relative_import_targets(ts_file: Path) -> list[Path]:
    """Relative `from './...'` imports in a TS/JS file, resolved to real
    files on disk (playwright.config.ts imports extensionless `./e2e/fixtures/ports`
    and the exact `./e2e/prepare-models-cache.mjs`)."""
    text = ts_file.read_text(encoding="utf-8")
    targets: list[Path] = []
    for match in re.finditer(r"from\s+['\"](\.[^'\"]+)['\"]", text):
        spec = match.group(1)
        candidate = (ts_file.parent / spec).resolve()
        if candidate.exists():
            targets.append(candidate)
            continue
        for ext in (".ts", ".tsx", ".mts", ".mjs", ".js"):
            with_ext = candidate.with_suffix(ext) if candidate.suffix == "" else None
            if with_ext is None:
                with_ext = Path(str(candidate) + ext)
            if with_ext.exists():
                targets.append(with_ext)
                break
        else:
            raise AssertionError(f"{ts_file}: cannot resolve relative import {spec!r} on disk")
    return targets


def _tsc_required_paths() -> list[str]:
    """Every path `tsc -b` needs for `web/tsconfig.node.json`: its own
    `include` entries, plus whatever `playwright.config.ts` imports
    relatively (not itself listed in `include`, but required transitively)."""
    tsconfig_node = json.loads((WEB_DIR / "tsconfig.node.json").read_text(encoding="utf-8"))
    required: list[Path] = [WEB_DIR / entry for entry in tsconfig_node["include"]]

    playwright_config = WEB_DIR / "playwright.config.ts"
    assert playwright_config in required, (
        "test assumption stale: web/tsconfig.node.json no longer includes playwright.config.ts"
    )
    required.extend(_relative_import_targets(playwright_config))

    return [str(p.resolve().relative_to(ROOT_DIR).as_posix()) for p in required]


def test_dockerignore_does_not_exclude_files_tsc_b_needs_for_web_build() -> None:
    patterns = [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    for rel_path in _tsc_required_paths():
        excluding = [p for p in patterns if _dockerignore_excludes(p, rel_path)]
        assert not excluding, (
            f".dockerignore excludes {rel_path!r} (via {excluding}), which `tsc -b` needs "
            "to build web/tsconfig.node.json -- this reproduces the CP5 web-builder "
            "TS2307 failure"
        )


def test_dockerignore_excludes_matcher_flags_a_web_e2e_pattern() -> None:
    """Mutation-provable unit test of `_dockerignore_excludes` in isolation
    (no repo state mutated): a synthetic `web/e2e/` pattern -- the exact line
    this fix removes -- must be flagged as excluding files under `web/e2e/`,
    proving the matcher used above can actually catch the regression."""
    assert _dockerignore_excludes("web/e2e/", "web/e2e/fixtures/ports.ts")
    assert _dockerignore_excludes("web/e2e/", "web/e2e/prepare-models-cache.mjs")
    assert not _dockerignore_excludes("web/e2e/", "web/e2e-other/file.ts")
    assert not _dockerignore_excludes("web/test-results/", "web/e2e/fixtures/ports.ts")

"""T43: `entrypoint-worker.sh`'s env->flag mapping for `UNFLATTEN_ENGINES` ->
`--engines` and `UNFLATTEN_MODEL_VARIANT` -> `--variant` (design §8.2/§8.4).
No Docker daemon is used here -- runs the real entrypoint script with a stub
`python3` first on PATH."""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path

ENTRYPOINT_WORKER = Path(__file__).parents[2] / "deploy" / "docker" / "entrypoint-worker.sh"


def _run_entrypoint_worker(tmp_path: Path, env_overrides: dict[str, str]) -> list[str]:
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
        "UNFLATTEN_SANDBOX_IMAGE": "unflatten/pdf-sandbox:local",
        "UNFLATTEN_MODEL_ID": "Qwen/Qwen3-VL-4B-Instruct",
        "UNFLATTEN_MODEL_REVISION": "a" * 40,
        **env_overrides,
    }
    subprocess.run(
        ["bash", str(ENTRYPOINT_WORKER)], env=env, capture_output=True, text=True, check=True
    )
    return capture_file.read_text().splitlines()


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_entrypoint_worker_engines_defaults_to_native(tmp_path: Path) -> None:
    argv = _run_entrypoint_worker(tmp_path, {})

    assert _flag_value(argv, "--engines") == "native"


def test_entrypoint_worker_maps_unflatten_engines(tmp_path: Path) -> None:
    argv = _run_entrypoint_worker(tmp_path, {"UNFLATTEN_ENGINES": "native,candidate"})

    assert _flag_value(argv, "--engines") == "native,candidate"


def test_entrypoint_worker_omits_variant_when_unset(tmp_path: Path) -> None:
    argv = _run_entrypoint_worker(tmp_path, {})

    assert "--variant" not in argv


def test_entrypoint_worker_maps_unflatten_model_variant(tmp_path: Path) -> None:
    argv = _run_entrypoint_worker(tmp_path, {"UNFLATTEN_MODEL_VARIANT": "bf16-hf"})

    assert _flag_value(argv, "--variant") == "bf16-hf"

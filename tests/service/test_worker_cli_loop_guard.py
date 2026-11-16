"""DEC-055 service switch: `--loop-guard` / `UNFLATTEN_LOOP_GUARD` (and the v3
arm). Unset must leave the bundle's decode policy untouched; every refusal
fires before the job store or a model is opened."""

from __future__ import annotations

from pathlib import Path

import pytest

from unflatten.model.candidate_runtime import DecodePolicy
from unflatten.service import worker_cli

BASE_ARGS = [
    "--job-db",
    "jobs.sqlite3",
    "--artifact-root",
    "artifacts",
    "--work-root",
    "work",
    "--sandbox-image",
    "sha256:" + "a" * 64,
    "--model-bundle",
    "bundle",
    "--model-id",
    "Qwen/Qwen3-VL-4B-Instruct",
    "--revision",
    "deadbeef",
]

BUNDLE_POLICY = DecodePolicy(
    loop_guard="v2",
    max_steps=4096,
    typed_decode=True,
    region_decode=True,
    selection_policy_hash="sha256:" + "c" * 64,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNFLATTEN_LOOP_GUARD", raising=False)
    monkeypatch.delenv("UNFLATTEN_LOOP_GUARD_V3_ARM", raising=False)


def test_loop_guard_flag_defaults_to_v3_r2_in_the_service() -> None:
    """DEC-055 frozen criteria passed for arm r2 on 42 val_iid pages
    (docs/runs/loop-guard-v3-2026-09-29.md), so the local service decodes
    with loop guard v3 / r2 unless the operator says otherwise."""
    args = worker_cli._parser().parse_args(BASE_ARGS)
    assert (args.loop_guard, args.loop_guard_v3_arm) == ("v3", None)
    policy = worker_cli.apply_loop_guard_override(BUNDLE_POLICY, args.loop_guard, None)
    assert (policy.loop_guard, policy.loop_guard_v3_arm) == ("v3", "r2")


def test_loop_guard_bundle_keeps_the_bundle_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNFLATTEN_LOOP_GUARD", "bundle")
    args = worker_cli._parser().parse_args(BASE_ARGS)
    assert args.loop_guard == "bundle"
    assert worker_cli.apply_loop_guard_override(BUNDLE_POLICY, "bundle", None) is BUNDLE_POLICY
    assert worker_cli.apply_loop_guard_override(BUNDLE_POLICY, None, None) is BUNDLE_POLICY


def test_loop_guard_empty_environment_variable_is_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNFLATTEN_LOOP_GUARD", "")
    assert worker_cli._parser().parse_args(BASE_ARGS).loop_guard == "v3"


def test_loop_guard_from_environment_and_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNFLATTEN_LOOP_GUARD", "v3")
    monkeypatch.setenv("UNFLATTEN_LOOP_GUARD_V3_ARM", "r2+r4")
    args = worker_cli._parser().parse_args(BASE_ARGS)
    assert (args.loop_guard, args.loop_guard_v3_arm) == ("v3", "r2+r4")
    assert worker_cli._parser().parse_args([*BASE_ARGS, "--loop-guard", "v2"]).loop_guard == "v2"


def test_apply_override_replaces_only_the_loop_guard_fields() -> None:
    policy = worker_cli.apply_loop_guard_override(BUNDLE_POLICY, "v3", "r1+r4")
    assert (policy.loop_guard, policy.loop_guard_v3_arm) == ("v3", "r1+r4")
    assert (policy.max_steps, policy.selection_policy_hash) == (
        BUNDLE_POLICY.max_steps,
        BUNDLE_POLICY.selection_policy_hash,
    )


def test_loop_guard_flag_refuses_unknown_value() -> None:
    with pytest.raises(SystemExit):
        worker_cli._parser().parse_args([*BASE_ARGS, "--loop-guard", "v4"])


def _main(tmp_path: Path, *extra: str) -> None:
    worker_cli.main(
        [
            "--job-db",
            str(tmp_path / "jobs.sqlite3"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--work-root",
            str(tmp_path / "work"),
            "--sandbox-image",
            "sha256:" + "a" * 64,
            "--model-id",
            "Qwen/Qwen3-VL-4B-Instruct",
            "--revision",
            "deadbeef",
            *extra,
        ]
    )


def test_main_refuses_unknown_loop_guard_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNFLATTEN_LOOP_GUARD", "v9")
    with pytest.raises(ValueError, match="UNFLATTEN_LOOP_GUARD must be one of"):
        _main(tmp_path)
    assert not (tmp_path / "jobs.sqlite3").exists()


def test_main_refuses_a_v3_arm_without_v3(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs --loop-guard v3"):
        _main(tmp_path, "--loop-guard", "v2", "--loop-guard-v3-arm", "r1")


def test_main_refuses_an_arm_the_v3_policy_does_not_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="arm"):
        _main(tmp_path, "--loop-guard", "v3", "--loop-guard-v3-arm", "r3")
    assert not (tmp_path / "jobs.sqlite3").exists()

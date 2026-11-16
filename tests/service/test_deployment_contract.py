from pathlib import Path

from unflatten.model.candidate_loader import QWEN_CANDIDATE_MODEL_ID

DEPLOY = Path(__file__).parents[2] / "deploy" / "service"


def test_api_unit_and_runbook_use_service_owned_read_only_principal_file() -> None:
    api_unit = (DEPLOY / "unflatten-diagnostic-api.service").read_text(encoding="utf-8")
    readme = (DEPLOY / "README.md").read_text(encoding="utf-8")
    runbook = (DEPLOY.parents[1] / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")

    assert "User=unflatten" in api_unit
    assert "ReadOnlyPaths=/etc/unflatten/principals.json" in api_unit
    assert "unflatten:unflatten" in readme
    assert "mode `0400`" in readme
    assert "Principal and environment files must be root-owned mode `0600`" not in readme
    assert "unflatten:unflatten" in runbook
    assert "mode `0400`" in runbook
    assert "Principal file must be root-owned, mode `0600`" not in runbook


def test_worker_does_not_receive_or_read_api_principal_credentials() -> None:
    worker_unit = (DEPLOY / "unflatten-diagnostic-worker.service").read_text(encoding="utf-8")

    assert "principals.json" not in worker_unit
    assert "--principal-file" not in worker_unit


def test_worker_keeps_cuda_devices_visible_with_closed_explicit_allowlist() -> None:
    worker_unit = (DEPLOY / "unflatten-diagnostic-worker.service").read_text(encoding="utf-8")
    api_unit = (DEPLOY / "unflatten-diagnostic-api.service").read_text(encoding="utf-8")

    assert "PrivateDevices=no" in worker_unit
    assert "DevicePolicy=closed" in worker_unit
    for device in ("nvidia0", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"):
        assert f"DeviceAllow=/dev/{device} rw" in worker_unit
    assert "PrivateDevices=yes" in api_unit


def test_deployment_model_id_matches_pinned_candidate_loader() -> None:
    environment = (DEPLOY / "service.env.example").read_text(encoding="utf-8")
    configured = next(
        line.removeprefix("UNFLATTEN_MODEL_ID=")
        for line in environment.splitlines()
        if line.startswith("UNFLATTEN_MODEL_ID=")
    )

    assert configured == QWEN_CANDIDATE_MODEL_ID

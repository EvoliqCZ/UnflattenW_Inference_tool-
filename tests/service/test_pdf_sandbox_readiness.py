from __future__ import annotations

import pytest

from unflatten.service.pdf_sandbox import LOCAL_SANDBOX_TAG, PdfSandbox
from unflatten.service.policy import load_runtime_policy

IMAGE = "localhost/unflatten-pdf-sandbox@sha256:" + "a" * 64
LOCAL_ID_IMAGE = "sha256:" + "b" * 64
RESOLVED_TAG_ID = "sha256:" + "c" * 64
IDENTITY = "unflatten-pdf-admission/qpdf-11.9.0-1.1ubuntu0.1-poppler-24.02.0-1ubuntu9.9/v13"


def _stub_readiness_command(sandbox: PdfSandbox, *, info: str, identity: str) -> None:
    sandbox._readiness_command = lambda command: (  # type: ignore[method-assign]
        info if command[1] == "info" else identity
    )


def _stub_tag_readiness_command(
    sandbox: PdfSandbox, *, info: str, identity: str, resolved_id: str | None
) -> None:
    def fake(command: list[str]) -> str | None:
        if command[1] == "info":
            return info
        if command[4] == "{{.Id}}":
            return resolved_id
        return identity

    sandbox._readiness_command = fake  # type: ignore[method-assign]


def test_readiness_fails_closed_when_runtime_binary_is_absent(monkeypatch) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE, runtime="podman")
    monkeypatch.setattr("unflatten.service.pdf_sandbox.shutil.which", lambda name: None)

    readiness = sandbox.readiness()

    assert readiness == {
        "ready": False,
        "runtime": "unavailable",
        "image": "unverified",
        "rootless": False,
        "runtime_kind": "podman",
        "rootful_allowed": False,
        "image_id": None,
    }


def test_unknown_runtime_is_rejected_by_constructor() -> None:
    with pytest.raises(ValueError, match="docker or podman"):
        PdfSandbox(policy=load_runtime_policy(), image=IMAGE, runtime="crun")


def test_local_image_id_without_name_is_accepted() -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=LOCAL_ID_IMAGE, runtime="docker")

    assert sandbox.image == LOCAL_ID_IMAGE


@pytest.mark.parametrize(
    "image",
    [
        "localhost/unflatten-pdf-sandbox:latest",
        "localhost/unflatten-pdf-sandbox:v6",
        "sha256:" + "a" * 63,
        "localhost/unflatten-pdf-sandbox@sha256:" + "g" * 64,
    ],
)
def test_unpinned_image_is_rejected_by_constructor(image: str) -> None:
    with pytest.raises(ValueError, match="digest-pinned"):
        PdfSandbox(policy=load_runtime_policy(), image=image, runtime="docker")


def test_rootful_docker_without_allow_rootful_is_not_ready(monkeypatch) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE, runtime="docker")
    monkeypatch.setattr(
        "unflatten.service.pdf_sandbox.shutil.which", lambda name: "/usr/bin/docker"
    )
    _stub_readiness_command(
        sandbox, info='["name=apparmor","name=seccomp,profile=default"]', identity=IDENTITY
    )

    readiness = sandbox.readiness()

    assert readiness["ready"] is False
    assert readiness["rootless"] is False
    assert readiness["runtime_kind"] == "docker"
    assert readiness["rootful_allowed"] is False


def test_rootful_docker_with_allow_rootful_can_be_ready(monkeypatch) -> None:
    sandbox = PdfSandbox(
        policy=load_runtime_policy(), image=IMAGE, runtime="docker", allow_rootful=True
    )
    monkeypatch.setattr(
        "unflatten.service.pdf_sandbox.shutil.which", lambda name: "/usr/bin/docker"
    )
    _stub_readiness_command(
        sandbox, info='["name=apparmor","name=seccomp,profile=default"]', identity=IDENTITY
    )

    readiness = sandbox.readiness()

    assert readiness["ready"] is True
    assert readiness["rootless"] is False
    assert readiness["rootful_allowed"] is True


def test_rootful_podman_is_never_ready_even_with_allow_rootful(monkeypatch) -> None:
    sandbox = PdfSandbox(
        policy=load_runtime_policy(), image=IMAGE, runtime="podman", allow_rootful=True
    )
    monkeypatch.setattr(
        "unflatten.service.pdf_sandbox.shutil.which", lambda name: "/usr/bin/podman"
    )
    _stub_readiness_command(sandbox, info="false", identity=IDENTITY)

    readiness = sandbox.readiness()

    assert readiness["ready"] is False
    assert readiness["rootless"] is False


def test_rootless_docker_identity_mismatch_is_unverified_and_not_ready(monkeypatch) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE, runtime="docker")
    monkeypatch.setattr(
        "unflatten.service.pdf_sandbox.shutil.which", lambda name: "/usr/bin/docker"
    )
    _stub_readiness_command(
        sandbox,
        info='["name=rootless","name=seccomp,profile=default"]',
        identity="some-other-identity",
    )

    readiness = sandbox.readiness()

    assert readiness["image"] == "unverified"
    assert readiness["ready"] is False


def test_rootless_docker_ready_path_works(monkeypatch) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE, runtime="docker")
    monkeypatch.setattr(
        "unflatten.service.pdf_sandbox.shutil.which", lambda name: "/usr/bin/docker"
    )
    _stub_readiness_command(
        sandbox,
        info='["name=rootless","name=seccomp,profile=default"]',
        identity=IDENTITY,
    )

    readiness = sandbox.readiness()

    assert readiness["ready"] is True
    assert readiness["runtime_kind"] == "docker"
    assert readiness["rootless"] is True


def test_rootless_podman_ready_path_still_works(monkeypatch) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE, runtime="podman")
    monkeypatch.setattr(
        "unflatten.service.pdf_sandbox.shutil.which", lambda name: "/usr/bin/podman"
    )
    _stub_readiness_command(sandbox, info="true", identity=IDENTITY)

    readiness = sandbox.readiness()

    assert readiness["ready"] is True
    assert readiness["runtime_kind"] == "podman"


def test_local_sandbox_tag_accepted() -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=LOCAL_SANDBOX_TAG, runtime="docker")

    assert sandbox.image == LOCAL_SANDBOX_TAG


def test_other_tag_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="digest-pinned"):
        PdfSandbox(policy=load_runtime_policy(), image="unflatten/pdf-sandbox:latest")


def test_tag_resolved_to_image_id_on_require_ready(monkeypatch) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=LOCAL_SANDBOX_TAG, runtime="docker")
    monkeypatch.setattr(
        "unflatten.service.pdf_sandbox.shutil.which", lambda name: "/usr/bin/docker"
    )
    _stub_tag_readiness_command(
        sandbox,
        info='["name=rootless","name=seccomp,profile=default"]',
        identity=IDENTITY,
        resolved_id=RESOLVED_TAG_ID,
    )

    sandbox.require_ready()

    assert sandbox._resolved_image == RESOLVED_TAG_ID


def test_label_mismatch_readiness_false(monkeypatch) -> None:
    """Mutation: a locally built image with the wrong identity label must never
    resolve to ready, even though the runtime and rootless checks pass."""
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=LOCAL_SANDBOX_TAG, runtime="docker")
    monkeypatch.setattr(
        "unflatten.service.pdf_sandbox.shutil.which", lambda name: "/usr/bin/docker"
    )
    _stub_tag_readiness_command(
        sandbox,
        info='["name=rootless","name=seccomp,profile=default"]',
        identity="some-other-identity",
        resolved_id=RESOLVED_TAG_ID,
    )

    readiness = sandbox.readiness()

    assert readiness["ready"] is False
    assert readiness["image"] == "unverified"
    assert readiness["image_id"] is None
    assert sandbox._resolved_image == LOCAL_SANDBOX_TAG


def test_readiness_reports_image_id(monkeypatch) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=LOCAL_SANDBOX_TAG, runtime="docker")
    monkeypatch.setattr(
        "unflatten.service.pdf_sandbox.shutil.which", lambda name: "/usr/bin/docker"
    )
    _stub_tag_readiness_command(
        sandbox,
        info='["name=rootless","name=seccomp,profile=default"]',
        identity=IDENTITY,
        resolved_id=RESOLVED_TAG_ID,
    )

    readiness = sandbox.readiness()

    assert readiness["ready"] is True
    assert readiness["image_id"] == RESOLVED_TAG_ID

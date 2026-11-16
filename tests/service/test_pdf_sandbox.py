from __future__ import annotations

import json
import os
import signal
import struct
import zlib
from pathlib import Path

import pytest

from unflatten.service.pdf_sandbox import (
    LOCAL_SANDBOX_TAG,
    PdfAdmissionReport,
    PdfSandbox,
    PdfSandboxFailure,
)
from unflatten.service.policy import load_runtime_policy

IMAGE = "localhost/unflatten-pdf-sandbox@sha256:" + "a" * 64


def _report(**changes: object) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": "1.2",
        "scanner_identity": (
            "unflatten-pdf-admission/qpdf-11.9.0-1.1ubuntu0.1-"
            "poppler-24.02.0-1ubuntu9.9/v13"
        ),
        "status": "accepted",
        "reason": None,
        "page_count": 1,
        "page_width_points": 612.0,
        "page_height_points": 792.0,
        "raster_width_pixels": 2550,
        "raster_height_pixels": 3300,
        "raster_pixels": 8_415_000,
        "encrypted": False,
        "javascript_action_objects": 0,
        "attachment_objects": 0,
        "c2pa_metadata_objects": 0,
        "annotation_objects": 0,
        "text_visibility": "visible_only",
        "hidden_text_objects": 0,
        "ambiguous_text_objects": 0,
        "extreme_image_objects": 0,
        "object_count": 8,
    }
    report.update(changes)
    geometry_keys = ("page_width_points", "page_height_points", "raster_width_pixels",
                     "raster_height_pixels", "raster_pixels")
    report["page_geometries"] = [{key: report[key] for key in geometry_keys}
                                for _ in range(min(int(report["page_count"]), 100))]
    if not 1 <= int(report["page_count"]) <= 100:
        report.update(status="rejected", reason="UNSUPPORTED_PAGE_COUNT")
    return report


def _png(width: int = 2550, height: int = 3300) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = b"IHDR" + ihdr
    return signature + struct.pack(">I", len(ihdr)) + chunk + struct.pack(">I", zlib.crc32(chunk))


def test_admission_report_is_closed_and_content_free() -> None:
    parsed = PdfAdmissionReport.from_json(json.dumps(_report()).encode())
    assert parsed.page_count == 1

    with pytest.raises(PdfSandboxFailure, match="closed schema"):
        PdfAdmissionReport.from_json(json.dumps(_report(metadata="secret")).encode())


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ({"javascript_action_objects": 1}, "PDF_ACTIVE_CONTENT"),
        ({"attachment_objects": 1}, "PDF_EMBEDDED_FILE"),
        ({"encrypted": True}, "PDF_ENCRYPTED"),
        ({"extreme_image_objects": 1}, "PDF_PAGE_DIMENSIONS_EXCEEDED"),
        ({"page_count": 101}, "UNSUPPORTED_PAGE_COUNT"),
        ({"page_count": 0}, "UNSUPPORTED_PAGE_COUNT"),
        ({"page_width_points": 1441.0, "raster_width_pixels": 6005,
          "raster_pixels": 6005 * 3300}, "PDF_PAGE_DIMENSIONS_EXCEEDED"),
        ({"status": "rejected", "reason": "PDF_MALFORMED"}, "PDF_MALFORMED"),
    ],
)
def test_governed_pdf_features_reject_before_raster(
    tmp_path: Path, mutation: dict[str, object], code: str
) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE)
    calls: list[str] = []

    def invoke(mode: str, source: Path, expected_sha256: str | None = None) -> bytes:
        calls.append(mode)
        return json.dumps(_report(**mutation)).encode()

    sandbox._invoke = invoke  # type: ignore[method-assign]
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    with pytest.raises(PdfSandboxFailure) as failure:
        sandbox.admit_and_raster(source, tmp_path / "page.png")

    assert failure.value.code == code
    assert calls == ["inspect"]


@pytest.mark.parametrize("mutation", [
    {"annotation_objects": 1},
    {"text_visibility": "hidden_proven", "hidden_text_objects": 1},
    {"text_visibility": "ambiguous", "ambiguous_text_objects": 1},
])
def test_admit_and_raster_visible_raster_profile_accepts_pdf_features(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE)

    def invoke(mode: str, source: Path, expected_sha256: str | None = None,
               *, page_index: int = 0) -> bytes:
        return json.dumps(_report(**mutation)).encode() if mode == "inspect" else _png()

    sandbox._invoke = invoke  # type: ignore[method-assign]
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    output = tmp_path / "page.png"

    report = sandbox.admit_and_raster(source, output)

    assert report.status == "accepted"
    assert output.read_bytes() == _png()


def test_podman_command_is_rootless_hardened_and_uses_argv(tmp_path: Path) -> None:
    source = tmp_path / "source;touch PWNED.pdf"
    source.write_bytes(b"%PDF")
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE)

    command = sandbox.build_command("inspect", source)

    assert isinstance(command, list)
    assert command[:3] == ["podman", "run", "--rm"]
    for required in (
        "--network=none",
        "--cap-drop=all",
        "--read-only",
        "--security-opt=no-new-privileges",
        "--userns=keep-id",
    ):
        assert required in command
    joined = " ".join(command)
    assert "--pids-limit=32" in command
    assert "--memory=1073741824" in command
    assert "--ulimit=cpu=30:30" in command
    assert "--ulimit=nofile=64:64" in command
    assert "size=268435456" in joined
    tmpfs = next(token for token in command if token.startswith("--tmpfs=/tmp:"))
    configured_user = f"{os.getuid()}:{os.getgid()}"
    assert f"--user={configured_user}" in command
    assert f"uid={os.getuid()}" in tmpfs
    assert f"gid={os.getgid()}" in tmpfs
    assert "mode=0700" in tmpfs
    assert str(source.resolve()) in joined
    assert command[-2:] == ["inspect", "/input/source.pdf"]
    assert all(token not in command for token in ("sh", "bash", "-c"))


def test_docker_command_omits_userns_and_uses_mode_1777_tmpfs(tmp_path: Path) -> None:
    source = tmp_path / "source;touch PWNED.pdf"
    source.write_bytes(b"%PDF")
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE, runtime="docker")

    command = sandbox.build_command("inspect", source)

    assert command[:3] == ["docker", "run", "--rm"]
    for required in (
        "--network=none",
        "--cap-drop=all",
        "--read-only",
        "--security-opt=no-new-privileges",
    ):
        assert required in command
    assert "--userns=keep-id" not in command
    tmpfs = next(token for token in command if token.startswith("--tmpfs=/tmp:"))
    assert "mode=1777" in tmpfs
    assert "uid=" not in tmpfs
    assert "gid=" not in tmpfs
    configured_user = f"{os.getuid()}:{os.getgid()}"
    assert f"--user={configured_user}" in command
    assert command[-2:] == ["inspect", "/input/source.pdf"]
    assert all(token not in command for token in ("sh", "bash", "-c"))


def test_docker_command_never_runs_the_scanner_as_root_when_worker_is_root(
    tmp_path: Path, monkeypatch
) -> None:
    # compose.sandbox-socket.yaml runs the worker as uid 0 to reach docker.sock, while the
    # api writes uploads as uid 65532 (mode 0400). With --cap-drop=all a root sandbox
    # cannot read them (scanner rc 23 -> PDF_MALFORMED for every upload), and an
    # untrusted-PDF scanner has no business being root anyway.
    monkeypatch.setattr(os, "getuid", lambda: 0)
    monkeypatch.setattr(os, "getgid", lambda: 0)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF")
    sandbox = PdfSandbox(
        policy=load_runtime_policy(), image=IMAGE, runtime="docker", allow_rootful=True
    )

    command = sandbox.build_command("inspect", source)

    assert "--user=65532:65532" in command
    assert "--user=0:0" not in command


def test_cancel_kills_process_group_for_both_runtimes(monkeypatch) -> None:
    for runtime in ("docker", "podman"):
        sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE, runtime=runtime)
        killed: list[tuple[int, int]] = []
        monkeypatch.setattr(
            "unflatten.service.pdf_sandbox.os.killpg",
            lambda pid, sig: killed.append((pid, sig)),
        )

        class _FakeProcess:
            pid = 4321

            def poll(self) -> None:
                return None

            def wait(self) -> None:
                return None

        sandbox._active_process = _FakeProcess()  # type: ignore[assignment]
        sandbox.cancel()

        assert killed == [(4321, signal.SIGKILL)]


def test_unknown_runtime_is_rejected() -> None:
    with pytest.raises(ValueError, match="docker or podman"):
        PdfSandbox(policy=load_runtime_policy(), image=IMAGE, runtime="crun")


def test_image_must_be_digest_pinned() -> None:
    with pytest.raises(ValueError, match="digest-pinned"):
        PdfSandbox(policy=load_runtime_policy(), image="localhost/unflatten-pdf-sandbox:latest")


def test_invocations_use_resolved_id(tmp_path: Path, monkeypatch) -> None:
    """After require_ready() resolves LOCAL_SANDBOX_TAG to an image ID, every
    invocation must run that ID, not the tag (a tag repointed at runtime must
    have no effect on an already-verified sandbox)."""
    resolved_id = "sha256:" + "d" * 64
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=LOCAL_SANDBOX_TAG, runtime="docker")
    monkeypatch.setattr(
        "unflatten.service.pdf_sandbox.shutil.which", lambda name: "/usr/bin/docker"
    )

    def fake_readiness_command(command: list[str]) -> str | None:
        if command[1] == "info":
            return '["name=rootless","name=seccomp,profile=default"]'
        if command[4] == "{{.Id}}":
            return resolved_id
        return (
            "unflatten-pdf-admission/qpdf-11.9.0-1.1ubuntu0.1-"
            "poppler-24.02.0-1ubuntu9.9/v13"
        )

    sandbox._readiness_command = fake_readiness_command  # type: ignore[method-assign]
    sandbox.require_ready()

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-")
    command = sandbox.build_command("inspect", source)

    assert command[-3] == resolved_id
    assert LOCAL_SANDBOX_TAG not in command


def test_invalid_or_oversize_png_is_rejected(tmp_path: Path) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE)
    sandbox._invoke = lambda *args, **kwargs: b"not-png"  # type: ignore[method-assign]
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-")

    with pytest.raises(PdfSandboxFailure) as failure:
        sandbox.raster(source, PdfAdmissionReport.from_mapping(_report()), tmp_path / "page.png")

    assert failure.value.code == "RASTER_OUTPUT_INVALID"
    assert not (tmp_path / "page.png").exists()


def test_invoke_refuses_unready_sandbox_before_starting_container(
    tmp_path: Path, monkeypatch
) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE, runtime="docker")
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-")
    monkeypatch.setattr(sandbox, "readiness", lambda: {"ready": False})
    monkeypatch.setattr(
        "unflatten.service.pdf_sandbox.subprocess.Popen",
        lambda *args, **kwargs: pytest.fail("container must not run"),
    )

    with pytest.raises(PdfSandboxFailure) as failure:
        sandbox._invoke("inspect", source)

    assert failure.value.code == "POPPLER_RESOURCE_LIMIT"


def test_png_byte_limit_rejects_without_publication(tmp_path: Path) -> None:
    policy = load_runtime_policy()
    policy["max_png_bytes"] = 32
    sandbox = PdfSandbox(policy=policy, image=IMAGE)
    sandbox._invoke = lambda *args, **kwargs: _png()  # type: ignore[method-assign]
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-")
    output = tmp_path / "page.png"

    with pytest.raises(PdfSandboxFailure) as failure:
        sandbox.raster(source, PdfAdmissionReport.from_mapping(_report()), output)

    assert failure.value.code == "RASTER_OUTPUT_INVALID"
    assert not output.exists()


@pytest.mark.parametrize(
    ("contents", "code"),
    [(b"not-pdf", "PDF_SIGNATURE_INVALID"), (b"%PDF-too-large", "UPLOAD_TOO_LARGE")],
)
def test_source_preflight_rejects_before_container(
    tmp_path: Path, contents: bytes, code: str
) -> None:
    policy = load_runtime_policy()
    if code == "UPLOAD_TOO_LARGE":
        policy["max_upload_bytes"] = 4
    sandbox = PdfSandbox(policy=policy, image=IMAGE)
    sandbox._invoke = lambda *args, **kwargs: pytest.fail("container must not run")  # type: ignore[method-assign]
    source = tmp_path / "source.pdf"
    source.write_bytes(contents)

    with pytest.raises(PdfSandboxFailure) as failure:
        sandbox.inspect(source)

    assert failure.value.code == code


@pytest.mark.parametrize(("width", "height"), [(2481, 3508), (2482, 3507), (2550, 3300)])
def test_raster_accepts_poppler_rounding_within_one_pixel_of_the_admission(
    tmp_path: Path, width: int, height: int
) -> None:
    """26.9. on prod: a 595.44 pt page is admitted as ceil(595.44*300/72) = 2482
    px (float error on an exact product) while pdftoppm renders 2481 px; poppler
    rounds differently from every closed formula, always within 1 px (measured
    over 47 page widths). Exact equality refused every such page."""
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE)
    sandbox._invoke = lambda *args, **kwargs: _png(width, height)  # type: ignore[method-assign]
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-")
    admitted = _report(page_width_points=595.44, page_height_points=841.92,
                       raster_width_pixels=2482, raster_height_pixels=3508,
                       raster_pixels=2482 * 3508)
    if (width, height) == (2550, 3300):  # 68 px off: a different page, still refused
        with pytest.raises(PdfSandboxFailure) as failure:
            sandbox.raster(source, PdfAdmissionReport.from_mapping(admitted), tmp_path / "p.png")
        assert failure.value.code == "RASTER_OUTPUT_INVALID"
    else:
        sandbox.raster(source, PdfAdmissionReport.from_mapping(admitted), tmp_path / "p.png")
        assert (tmp_path / "p.png").read_bytes() == _png(width, height)



def test_inspect_c2pa_only_count_is_admitted_as_pixel_input(tmp_path: Path) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE)
    sandbox._invoke = lambda *args: json.dumps(_report(
        attachment_objects=3, c2pa_metadata_objects=3,
    )).encode()  # type: ignore[method-assign]
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    assert sandbox.inspect(source).status == "accepted"


@pytest.mark.parametrize(("total", "metadata", "reason"), [
    (4, 3, "PDF_EMBEDDED_FILE"), (2, 3, "PDF_MALFORMED"),
    (3, -1, "PDF_MALFORMED"),
])
def test_inspect_c2pa_counts_do_not_exempt_other_attachments(
    tmp_path: Path, total: int, metadata: int, reason: str
) -> None:
    sandbox = PdfSandbox(policy=load_runtime_policy(), image=IMAGE)
    sandbox._invoke = lambda *args: json.dumps(_report(
        attachment_objects=total, c2pa_metadata_objects=metadata,
    )).encode()  # type: ignore[method-assign]
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    with pytest.raises(PdfSandboxFailure) as failure:
        sandbox.inspect(source)

    assert failure.value.code == reason

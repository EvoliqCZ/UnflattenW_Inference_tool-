from __future__ import annotations

import hashlib
import json
import math
import os
import re
import selectors
import shutil
import signal
import stat
import struct
import subprocess
import tempfile
import time
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

SCANNER_IDENTITY = (
    "unflatten-pdf-admission/qpdf-11.9.0-1.1ubuntu0.1-"
    "poppler-24.02.0-1ubuntu9.9/v13"
)
_IMAGE_PATTERN = re.compile(r"^(?:\S+@)?sha256:[0-9a-f]{64}$")
LOCAL_SANDBOX_TAG = "unflatten/pdf-sandbox:local"
_RUNTIMES = frozenset({"docker", "podman"})
#: The image's unprivileged `unflatten` user (Dockerfile `USER 65532:65532`).
_ROOTFUL_SANDBOX_UID_GID = "65532:65532"
_TEXT_VISIBILITY = frozenset({"absent", "visible_only", "hidden_proven", "ambiguous"})
_REPORT_STATUS = frozenset({"accepted", "rejected"})
_REASON_CODES = frozenset(
    {
        "PDF_MALFORMED",
        "PDF_ENCRYPTED",
        "PDF_ACTIVE_CONTENT",
        "PDF_EMBEDDED_FILE",
        "PDF_ANNOTATION_UNSUPPORTED",
        "PDF_HIDDEN_TEXT",
        "PDF_TEXT_VISIBILITY_AMBIGUOUS",
        "UNSUPPORTED_PAGE_COUNT",
        "PDF_PAGE_DIMENSIONS_EXCEEDED",
    }
)


RASTER_ROUNDING_PX = 1


class PdfSandboxFailure(RuntimeError):
    """A typed, content-free PDF admission or raster failure."""

    def __init__(self, code: str, message: str = "PDF sandbox operation failed") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PdfPageGeometry:
    page_width_points: float
    page_height_points: float
    raster_width_pixels: int
    raster_height_pixels: int
    raster_pixels: int

    @classmethod
    def from_mapping(cls, payload: object) -> PdfPageGeometry:
        if not isinstance(payload, Mapping) or set(payload) != {f.name for f in fields(cls)}:
            raise PdfSandboxFailure("PDF_MALFORMED", "page geometry violates its closed schema")
        for name in ("page_width_points", "page_height_points"):
            value = payload[name]
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise PdfSandboxFailure("PDF_MALFORMED", "invalid page geometry")
        for name in ("raster_width_pixels", "raster_height_pixels", "raster_pixels"):
            if type(payload[name]) is not int or payload[name] <= 0:
                raise PdfSandboxFailure("PDF_MALFORMED", "invalid page raster geometry")
        if (payload["raster_width_pixels"] != math.ceil(payload["page_width_points"] * 300 / 72)
                or payload["raster_height_pixels"]
                != math.ceil(payload["page_height_points"] * 300 / 72)
                or payload["raster_pixels"]
                != payload["raster_width_pixels"] * payload["raster_height_pixels"]):
            raise PdfSandboxFailure("PDF_MALFORMED", "page geometry is inconsistent")
        return cls(**dict(payload))


@dataclass(frozen=True)
class PdfAdmissionReport:
    """Closed scanner report; deliberately contains no PDF strings or metadata."""

    schema_version: str
    scanner_identity: str
    status: str
    reason: str | None
    page_count: int
    page_geometries: tuple[PdfPageGeometry, ...]
    page_width_points: float
    page_height_points: float
    raster_width_pixels: int
    raster_height_pixels: int
    raster_pixels: int
    encrypted: bool
    javascript_action_objects: int
    attachment_objects: int
    c2pa_metadata_objects: int
    annotation_objects: int
    text_visibility: str
    hidden_text_objects: int
    ambiguous_text_objects: int
    extreme_image_objects: int
    object_count: int

    @classmethod
    def from_json(cls, payload: bytes) -> PdfAdmissionReport:
        if len(payload) > 65_536:
            raise PdfSandboxFailure("PDF_MALFORMED", "scanner report exceeds its closed bound")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PdfSandboxFailure("PDF_MALFORMED", "scanner report is not valid JSON") from exc
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, payload: object) -> PdfAdmissionReport:
        expected = {field.name for field in fields(cls)}
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise PdfSandboxFailure("PDF_MALFORMED", "scanner report violates its closed schema")
        if payload["schema_version"] != "1.2" or payload["scanner_identity"] != SCANNER_IDENTITY:
            raise PdfSandboxFailure("PDF_MALFORMED", "scanner report identity is not pinned")
        if not isinstance(payload["status"], str) or payload["status"] not in _REPORT_STATUS:
            raise PdfSandboxFailure("PDF_MALFORMED", "scanner report status is unknown")
        reason = payload["reason"]
        if reason is not None and (not isinstance(reason, str) or reason not in _REASON_CODES):
            raise PdfSandboxFailure("PDF_MALFORMED", "scanner report reason is unknown")
        if (payload["status"] == "accepted") != (reason is None):
            raise PdfSandboxFailure("PDF_MALFORMED", "scanner report status and reason disagree")
        if type(payload["encrypted"]) is not bool:
            raise PdfSandboxFailure("PDF_MALFORMED", "scanner encryption status is not boolean")
        integer_fields = (
            "page_count",
            "raster_width_pixels",
            "raster_height_pixels",
            "raster_pixels",
            "javascript_action_objects",
            "attachment_objects",
            "c2pa_metadata_objects",
            "annotation_objects",
            "hidden_text_objects",
            "ambiguous_text_objects",
            "extreme_image_objects",
            "object_count",
        )
        if any(type(payload[name]) is not int or payload[name] < 0 for name in integer_fields):
            raise PdfSandboxFailure("PDF_MALFORMED", "scanner report contains invalid counts")
        if payload["c2pa_metadata_objects"] > payload["attachment_objects"]:
            raise PdfSandboxFailure("PDF_MALFORMED", "scanner metadata counts exceed attachments")
        for name in ("page_width_points", "page_height_points"):
            value = payload[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise PdfSandboxFailure("PDF_MALFORMED", "scanner report contains invalid geometry")
        visibility = payload["text_visibility"]
        if not isinstance(visibility, str) or visibility not in _TEXT_VISIBILITY:
            raise PdfSandboxFailure("PDF_MALFORMED", "scanner text visibility is unknown")
        hidden_count = payload["hidden_text_objects"]
        ambiguous_count = payload["ambiguous_text_objects"]
        visibility_counts_valid = {
            "absent": hidden_count == 0 and ambiguous_count == 0,
            "visible_only": hidden_count == 0 and ambiguous_count == 0,
            "hidden_proven": hidden_count > 0 and ambiguous_count == 0,
            "ambiguous": ambiguous_count > 0,
        }[visibility]
        if not visibility_counts_valid:
            raise PdfSandboxFailure("PDF_MALFORMED", "scanner visibility counts disagree")
        pages = payload["page_geometries"]
        if not isinstance(pages, list) or len(pages) > 100:
            raise PdfSandboxFailure("PDF_MALFORMED", "page geometry list exceeds its bound")
        geometries = tuple(PdfPageGeometry.from_mapping(page) for page in pages)
        if payload["status"] == "accepted":
            if len(geometries) != payload["page_count"] or not geometries:
                raise PdfSandboxFailure("PDF_MALFORMED", "page geometries are incomplete")
            first = geometries[0]
            if any(getattr(first, f.name) != payload[f.name] for f in fields(first)):
                raise PdfSandboxFailure("PDF_MALFORMED", "first page geometry disagrees")
        return cls(**{**dict(payload), "page_geometries": geometries})


class PdfSandbox:
    """OCI adapter whose authorization and scanner identity are fail-closed."""

    def __init__(
        self,
        *,
        policy: Mapping[str, Any],
        image: str,
        runtime: str = "podman",
        allow_rootful: bool = False,
    ) -> None:
        if runtime not in _RUNTIMES:
            raise ValueError("PDF sandbox runtime must be docker or podman")
        if not (_IMAGE_PATTERN.fullmatch(image) or image == LOCAL_SANDBOX_TAG):
            raise ValueError("PDF sandbox image must be digest-pinned")
        self.policy = dict(policy)
        self.image = image
        self.runtime = runtime
        self.allow_rootful = allow_rootful
        self._active_process: subprocess.Popen[bytes] | None = None
        # Digest-pinned images are already an immutable reference; the one
        # exception (DEC-052 point 6) is LOCAL_SANDBOX_TAG, which readiness()
        # resolves to an image ID below so a tag repointed at runtime cannot
        # silently change which image future invocations run.
        self._resolved_image = image

    def build_command(
        self,
        mode: str,
        source: Path,
        expected_sha256: str | None = None,
        *,
        page_index: int = 0,
    ) -> list[str]:
        if mode not in {"inspect", "raster"}:
            raise ValueError("PDF sandbox mode must be inspect or raster")
        if isinstance(page_index, bool) or not isinstance(page_index, int) or page_index < 0:
            raise ValueError("PDF sandbox page_index must be a non-negative integer")
        resolved = source.resolve(strict=True)
        command = [
            self.runtime,
            "run",
            "--rm",
            "--network=none",
            "--cap-drop=all",
            "--security-opt=no-new-privileges",
            "--read-only",
            *self._runtime_flags(),
            f"--pids-limit={self.policy['max_sandbox_processes']}",
            f"--memory={self.policy['max_sandbox_memory_bytes']}",
            f"--ulimit=cpu={self.policy['scanner_cpu_time_seconds']}:{self.policy['scanner_cpu_time_seconds']}",
            f"--ulimit=nofile={self.policy['max_sandbox_open_files']}:{self.policy['max_sandbox_open_files']}",
            self._tmpfs_flag(),
            f"--env=RASTER_DPI={self.policy['raster_dpi']}",
            f"--env=MAX_PAGE_WIDTH_POINTS={self.policy['max_page_width_points']}",
            f"--env=MAX_PAGE_HEIGHT_POINTS={self.policy['max_page_height_points']}",
            f"--env=MAX_RASTER_WIDTH_PIXELS={self.policy['max_raster_width_pixels']}",
            f"--env=MAX_RASTER_HEIGHT_PIXELS={self.policy['max_raster_height_pixels']}",
            f"--env=MAX_RASTER_PIXELS={self.policy['max_raster_pixels']}",
            f"--env=MAX_PDF_PAGES={self.policy['max_pdf_pages']}",
            f"--mount=type=bind,src={resolved},dst=/input/source.pdf,ro=true",
            self._resolved_image,
            mode,
            "/input/source.pdf",
        ]
        if mode == "raster":
            if not expected_sha256 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
                raise ValueError("raster invocation requires the admitted source SHA-256")
            command.extend(["--expected-sha256", expected_sha256, "--page-index", str(page_index)])
        return command

    def _runtime_flags(self) -> list[str]:
        """Per-runtime security flags (D8): keep-id userns exists only under podman."""
        flags: list[str] = []
        if self.runtime == "podman":
            flags.append("--userns=keep-id")
        flags.append(self._user_flag())
        return flags

    def _user_flag(self) -> str:
        uid_gid = f"{os.getuid()}:{os.getgid()}"
        if self.runtime == "docker":
            # Rootful docker: the worker is uid 0 only to reach docker.sock
            # (compose.sandbox-socket.yaml), while uploads are written by the
            # api as the image's unprivileged uid. A root scanner with
            # --cap-drop=all cannot read those files (rc 23) and should not be
            # root anyway, so it runs as that unprivileged uid.
            if os.getuid() == 0:
                return f"--user={_ROOTFUL_SANDBOX_UID_GID}"
            return f"--user={uid_gid}"
        return f"--user={uid_gid}"

    def _tmpfs_flag(self) -> str:
        size = self.policy["max_sandbox_tmpfs_bytes"]
        if self.runtime == "docker":
            # OPEN (T7): docker does not reliably accept tmpfs uid=/gid=, so the
            # docker branch uses mode=1777 instead; verified against a real run.
            return f"--tmpfs=/tmp:rw,noexec,nosuid,nodev,size={size},mode=1777"
        return (
            f"--tmpfs=/tmp:rw,noexec,nosuid,nodev,size={size},mode=0700,"
            f"uid={os.getuid()},gid={os.getgid()}"
        )

    def readiness(self) -> dict[str, object]:
        result: dict[str, object] = {
            "ready": False,
            "runtime": "unavailable",
            "image": "unverified",
            "rootless": False,
            "runtime_kind": self.runtime,
            "rootful_allowed": self.allow_rootful,
            "image_id": None,
        }
        if shutil.which(self.runtime) is None:
            return result
        result["runtime"] = "available"
        rootless = self._probe_rootless()
        if rootless is None:
            return result
        result["rootless"] = rootless
        if not rootless and not (self.runtime == "docker" and self.allow_rootful):
            return result
        identity = self._readiness_command(
            [
                self.runtime,
                "image",
                "inspect",
                "--format",
                '{{index .Config.Labels "org.unflatten.pdf-scanner.identity"}}',
                self.image,
            ]
        )
        if identity != SCANNER_IDENTITY:
            return result
        result["image"] = "verified"
        if self.image == LOCAL_SANDBOX_TAG:
            resolved_id = self._readiness_command(
                [self.runtime, "image", "inspect", "--format", "{{.Id}}", self.image]
            )
            if not resolved_id:
                return result
            self._resolved_image = resolved_id
        result["image_id"] = self._resolved_image
        result["ready"] = True
        return result

    def _probe_rootless(self) -> bool | None:
        if self.runtime == "docker":
            output = self._readiness_command(
                [self.runtime, "info", "--format", "{{json .SecurityOptions}}"]
            )
            if output is None:
                return None
            return "name=rootless" in output
        output = self._readiness_command(
            [self.runtime, "info", "--format", "{{.Host.Security.Rootless}}"]
        )
        if output is None:
            return None
        return output == "true"

    def inspect(self, source: Path) -> PdfAdmissionReport:
        self._validate_source(source)
        report = PdfAdmissionReport.from_json(self._invoke("inspect", source))
        self._enforce_admission(report)
        return report

    def admit_and_raster(
        self, source: Path, output: Path, *, page_index: int = 0
    ) -> PdfAdmissionReport:
        report = self.inspect(source)
        self.raster(source, report, output, page_index=page_index)
        return report

    def raster(
        self,
        source: Path,
        report: PdfAdmissionReport,
        output: Path,
        *,
        page_index: int = 0,
    ) -> None:
        self._validate_source(source)
        self._enforce_admission(report)
        if isinstance(page_index, bool) or not isinstance(page_index, int) or page_index < 0:
            raise ValueError("PDF sandbox page_index must be a non-negative integer")
        if page_index >= report.page_count:
            raise PdfSandboxFailure("UNSUPPORTED_PAGE_COUNT", "page_index is out of range")
        source_sha256 = _file_sha256(source)
        geometry = report.page_geometries[page_index]
        png = self._invoke("raster", source, source_sha256, page_index=page_index)
        width, height = _validate_png(png)
        if len(png) > self.policy["max_png_bytes"]:
            raise PdfSandboxFailure("RASTER_OUTPUT_INVALID", "PNG exceeds its output bound")
        # pdftoppm's pixel rounding differs from the admission's ceil(pt*dpi/72)
        # by at most one pixel (measured 26.9. over 47 page widths, both signs);
        # anything further is a different page. The output bounds below are
        # checked on the real PNG size.
        if (
            abs(width - geometry.raster_width_pixels) > RASTER_ROUNDING_PX
            or abs(height - geometry.raster_height_pixels) > RASTER_ROUNDING_PX
        ):
            raise PdfSandboxFailure("RASTER_OUTPUT_INVALID", "PNG geometry differs from admission")
        if (
            width > self.policy["max_raster_width_pixels"]
            or height > self.policy["max_raster_height_pixels"]
            or width * height > self.policy["max_raster_pixels"]
        ):
            raise PdfSandboxFailure("RASTER_PIXEL_LIMIT_EXCEEDED", "PNG exceeds raster limits")
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".raster-", dir=output.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(png)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def cancel(self) -> None:
        """Kill the active Podman process group; safe when no invocation is active."""
        if self._active_process is not None and self._active_process.poll() is None:
            _kill_process_group(self._active_process)

    def _validate_source(self, source: Path) -> None:
        try:
            source_stat = source.lstat()
            if not stat.S_ISREG(source_stat.st_mode):
                raise PdfSandboxFailure("PDF_MALFORMED", "PDF source is not a regular file")
            if source_stat.st_size > self.policy["max_upload_bytes"]:
                raise PdfSandboxFailure("UPLOAD_TOO_LARGE", "PDF source exceeds upload limit")
            with source.open("rb") as stream:
                signature = stream.read(5)
        except OSError as exc:
            raise PdfSandboxFailure("PDF_MALFORMED", "PDF source is unavailable") from exc
        if signature != b"%PDF-":
            raise PdfSandboxFailure("PDF_SIGNATURE_INVALID", "PDF signature is invalid")

    def _enforce_admission(self, report: PdfAdmissionReport) -> None:
        if report.status == "rejected":
            raise PdfSandboxFailure(report.reason or "PDF_MALFORMED")
        for page in report.page_geometries:
            if (page.page_width_points > self.policy["max_page_width_points"]
                    or page.page_height_points > self.policy["max_page_height_points"]
                    or page.raster_width_pixels > self.policy["max_raster_width_pixels"]
                    or page.raster_height_pixels > self.policy["max_raster_height_pixels"]
                    or page.raster_pixels > self.policy["max_raster_pixels"]):
                raise PdfSandboxFailure("PDF_PAGE_DIMENSIONS_EXCEEDED")
        checks = (
            (report.encrypted, "PDF_ENCRYPTED"),
            (report.javascript_action_objects > 0, "PDF_ACTIVE_CONTENT"),
            (report.attachment_objects > report.c2pa_metadata_objects, "PDF_EMBEDDED_FILE"),
            (report.extreme_image_objects > 0, "PDF_PAGE_DIMENSIONS_EXCEEDED"),
            (report.page_count > self.policy["max_pdf_pages"], "UNSUPPORTED_PAGE_COUNT"),
            (report.page_count < 1, "UNSUPPORTED_PAGE_COUNT"),
            (
                report.page_width_points > self.policy["max_page_width_points"]
                or report.page_height_points > self.policy["max_page_height_points"],
                "PDF_PAGE_DIMENSIONS_EXCEEDED",
            ),
            (
                report.raster_width_pixels > self.policy["max_raster_width_pixels"]
                or report.raster_height_pixels > self.policy["max_raster_height_pixels"]
                or report.raster_pixels > self.policy["max_raster_pixels"]
                or report.raster_pixels
                != report.raster_width_pixels * report.raster_height_pixels,
                "PDF_PAGE_DIMENSIONS_EXCEEDED",
            ),
            (
                report.raster_width_pixels
                != math.ceil(report.page_width_points * self.policy["raster_dpi"] / 72)
                or report.raster_height_pixels
                != math.ceil(report.page_height_points * self.policy["raster_dpi"] / 72),
                "PDF_MALFORMED",
            ),
        )
        for rejected, code in checks:
            if rejected:
                raise PdfSandboxFailure(code)

    def _invoke(
        self,
        mode: str,
        source: Path,
        expected_sha256: str | None = None,
        *,
        page_index: int = 0,
    ) -> bytes:
        # This is deliberately checked for every untrusted-PDF invocation, not
        # merely reported by the worker heartbeat. Runtime configuration can
        # change after worker startup; never run a container unless its current
        # rootless/rootful authorization and scanner identity are verified.
        self.require_ready()
        command = self.build_command(mode, source, expected_sha256, page_index=page_index)
        maximum = 65_536 if mode == "inspect" else self.policy["max_sandbox_output_bytes"]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise PdfSandboxFailure(
                "POPPLER_RESOURCE_LIMIT", "sandbox runtime unavailable"
            ) from exc
        self._active_process = process
        try:
            stdout = _bounded_process_output(
                process,
                maximum=maximum,
                timeout=float(self.policy["scanner_wall_time_seconds"]),
            )
        except TimeoutError as exc:
            raise PdfSandboxFailure("POPPLER_TIMEOUT", "sandbox wall-time limit exceeded") from exc
        except OverflowError as exc:
            raise PdfSandboxFailure(
                "POPPLER_RESOURCE_LIMIT", "sandbox output limit exceeded"
            ) from exc
        finally:
            self._active_process = None
        if process.returncode != 0:
            code = "PDF_MALFORMED" if mode == "inspect" else "RASTER_OUTPUT_INVALID"
            raise PdfSandboxFailure(code, "sandbox returned a typed failure")
        return stdout

    def require_ready(self) -> None:
        """Refuse container execution unless the complete sandbox gate passes."""
        if self.readiness()["ready"] is not True:
            # This existing public code is retryable and keeps a runtime or
            # image-configuration failure from being exposed as PDF content.
            raise PdfSandboxFailure(
                "POPPLER_RESOURCE_LIMIT", "sandbox readiness gate is not satisfied"
            )

    @staticmethod
    def _readiness_command(command: list[str]) -> str | None:
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
                text=True,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0 or len(result.stdout) > 1024:
            return None
        return result.stdout.strip()


def _bounded_process_output(
    process: subprocess.Popen[bytes], *, maximum: int, timeout: float
) -> bytes:
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("sandbox pipes are unavailable")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    output = bytearray()
    stderr_bytes = 0
    deadline = time.monotonic() + timeout
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                raise TimeoutError
            events = selector.select(remaining)
            if not events:
                _kill_process_group(process)
                raise TimeoutError
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 65_536)
                if not chunk:
                    selector.unregister(key.fileobj)
                elif key.data == "stdout":
                    output.extend(chunk)
                    if len(output) > maximum:
                        _kill_process_group(process)
                        raise OverflowError
                else:
                    stderr_bytes += len(chunk)
                    if stderr_bytes > 65_536:
                        _kill_process_group(process)
                        raise OverflowError
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_process_group(process)
            raise TimeoutError
        process.wait(timeout=remaining)
        return bytes(output)
    except subprocess.TimeoutExpired as exc:
        _kill_process_group(process)
        raise TimeoutError from exc
    finally:
        selector.close()


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_png(payload: bytes) -> tuple[int, int]:
    if len(payload) < 33 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise PdfSandboxFailure("RASTER_OUTPUT_INVALID", "raster output is not PNG")
    length = struct.unpack(">I", payload[8:12])[0]
    if length != 13 or payload[12:16] != b"IHDR":
        raise PdfSandboxFailure("RASTER_OUTPUT_INVALID", "PNG does not begin with IHDR")
    chunk = payload[12:29]
    expected_crc = struct.unpack(">I", payload[29:33])[0]
    if zlib.crc32(chunk) != expected_crc:
        raise PdfSandboxFailure("RASTER_OUTPUT_INVALID", "PNG IHDR checksum is invalid")
    width, height = struct.unpack(">II", payload[16:24])
    if width == 0 or height == 0:
        raise PdfSandboxFailure("RASTER_OUTPUT_INVALID", "PNG geometry is empty")
    return width, height

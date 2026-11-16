"""Private, principal-scoped service artifact storage."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from unflatten.dataset.artifact_ref import ArtifactRef
from unflatten.service.artifact_inventory import (
    PublishedArtifactRecord,
    StagedArtifactRecord,
    cleanup_orphans,
    published_artifacts,
    staged_artifacts,
)


class ArtifactStoreError(Exception):
    """Artifact storage failed with a stable machine-readable code."""

    def __init__(self, code: str, message: str, *, role: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.role = role


class PublicationFence(Protocol):
    """Run publication only while the job-store ownership CAS is valid."""

    def __call__(self, artifact: ArtifactRef, publish: Callable[[], None]) -> bool: ...


class PublicationFenceMany(Protocol):
    """Run publication of N artifacts only while the job-store ownership CAS is valid."""

    def __call__(self, artifacts: list[ArtifactRef], publish: Callable[[], None]) -> bool: ...


class ArtifactStore:
    """Store immutable service artifacts under generated private paths."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        if self.root.is_symlink():
            raise ArtifactStoreError("ARTIFACT_PATH_UNSAFE", "artifact root must not be a symlink")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def stage(
        self,
        *,
        principal_id: str,
        job_id: str,
        execution_id: str,
        role: str,
        mime_type: str,
        chunks: Iterable[bytes],
        max_bytes: int | None = None,
    ) -> ArtifactRef:
        relative = self.prepare_staging_path(
            principal_id, job_id, execution_id, role, mime_type, max_bytes
        )
        path = self.root / relative
        digest, size = self._write_staged(path, chunks, max_bytes)
        return ArtifactRef(
            ref=relative.as_posix(),
            sha256=f"sha256:{digest}",
            size_bytes=size,
            mime_type=mime_type,
        )

    def prepare_staging_path(
        self,
        principal_id: str,
        job_id: str,
        execution_id: str,
        role: str,
        mime_type: str,
        max_bytes: int | None,
    ) -> Path:
        principal = _identifier(principal_id, "principal_id")
        job = _identifier(job_id, "job_id", prefix="job_")
        execution = _identifier(execution_id, "execution_id")
        _validate_role_mime(role, mime_type)
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ArtifactStoreError("ARTIFACT_LIMIT_INVALID", "max_bytes must be non-negative")
        relative = Path(
            "principals", principal, "jobs", job, "attempts", execution, role,
            f"staged_{uuid4().hex}.staged",
        )
        self._make_private_directories((self.root / relative).parent)
        return relative

    def _write_staged(
        self, path: Path, chunks: Iterable[bytes], max_bytes: int | None
    ) -> tuple[str, int]:
        digest, size = hashlib.sha256(), 0
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                for chunk in chunks:
                    if not isinstance(chunk, bytes):
                        raise ArtifactStoreError(
                            "ARTIFACT_CHUNK_INVALID", "artifact chunks must be bytes"
                        )
                    size += len(chunk)
                    if max_bytes is not None and size > max_bytes:
                        raise ArtifactStoreError(
                            "ARTIFACT_TOO_LARGE", "artifact exceeds its byte ceiling"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        except ArtifactStoreError:
            path.unlink(missing_ok=True)
            raise
        except (OSError, TypeError, ValueError) as exc:
            path.unlink(missing_ok=True)
            raise ArtifactStoreError("ARTIFACT_WRITE_FAILED", "artifact staging failed") from exc
        self._fsync_directory(path.parent)
        return digest.hexdigest(), size

    def publish(self, staged: ArtifactRef, fence: PublicationFence) -> ArtifactRef:
        def many_fence(published: list[ArtifactRef], publish: Callable[[], None]) -> bool:
            [only] = published
            return fence(only, publish)

        [published] = self.publish_many([staged], many_fence)
        return published

    def publish_many(
        self,
        staged: Sequence[ArtifactRef],
        fence: PublicationFenceMany,
        *,
        allow_repeated_roles: bool = False,
    ) -> list[ArtifactRef]:
        """Publish N staged artifacts as a single all-or-nothing filesystem operation.

        `allow_repeated_roles` (DEC-044 Dodatok D §1, B1): a multi-page job
        legitimately publishes several `prediction_ir` artifacts (one per
        page) in the SAME atomic call that also publishes the merged
        `diagnostic_docx` -- the storage layer already gives each published
        artifact a random, collision-free filename regardless of role
        (`_publication_target`), so relaxing the check changes only which
        callers may repeat a role, never how a repeated-role publish is
        stored. Default `False` keeps every existing (single-page) caller's
        exact behaviour, including `test_publish_many_rejects_duplicate_role`.
        """
        if not isinstance(staged, Sequence) or isinstance(staged, (str, bytes)) or not staged:
            raise ArtifactStoreError(
                "ARTIFACT_REF_INVALID", "publish_many requires at least one staged artifact"
            )
        prepared: list[tuple[Path, Path, ArtifactRef, str]] = []
        seen_roles: set[str] = set()
        for ref in staged:
            validated = self._validated_ref(ref)
            parts = Path(validated.ref).parts
            if (
                len(parts) != 8
                or parts[0] != "principals"
                or parts[2] != "jobs"
                or parts[4] != "attempts"
            ):
                raise ArtifactStoreError("ARTIFACT_REF_INVALID", "artifact is not attempt-local")
            principal, job, execution, role = parts[1], parts[3], parts[5], parts[6]
            _identifier(principal, "principal_id")
            _identifier(job, "job_id", prefix="job_")
            _validate_role_mime(role, validated.mime_type)
            if role in seen_roles and not allow_repeated_roles:
                raise ArtifactStoreError(
                    "ARTIFACT_ROLE_DUPLICATE", "publish_many roles must be distinct", role=role
                )
            seen_roles.add(role)
            source = self._safe_existing_path(validated.ref)
            try:
                self._verify_file(source, validated)
            except ArtifactStoreError:
                source.unlink(missing_ok=True)
                raise
            published, target = self._publication_target(
                principal, job, execution, role, validated
            )
            prepared.append((source, target, published, role))
        self._run_publication_fence_many(prepared, fence)
        return [item[2] for item in prepared]

    def _publication_target(
        self, principal: str, job: str, execution: str, role: str, staged: ArtifactRef
    ) -> tuple[ArtifactRef, Path]:
        target_relative = Path(
            "principals",
            principal,
            "jobs",
            job,
            "artifacts",
            role,
            execution,
            f"artifact_{uuid4().hex}.bin",
        )
        target = self.root / target_relative
        self._make_private_directories(target.parent)
        published = ArtifactRef(
            ref=target_relative.as_posix(),
            sha256=staged.sha256,
            size_bytes=staged.size_bytes,
            mime_type=staged.mime_type,
        )
        return published, target

    def _run_publication_fence_many(
        self,
        prepared: list[tuple[Path, Path, ArtifactRef, str]],
        fence: PublicationFenceMany,
    ) -> None:
        publication_called = False

        def atomic_publication() -> None:
            nonlocal publication_called
            if publication_called:
                raise ArtifactStoreError(
                    "ARTIFACT_FENCE_PROTOCOL", "publication callback may run only once"
                )
            publication_called = True
            completed: list[tuple[Path, Path]] = []
            failed_role = prepared[0][3]
            try:
                for source, target, _, role in prepared:
                    failed_role = role
                    os.replace(source, target)
                    os.chmod(target, 0o600)
                    completed.append((source, target))
                for _, target, _, _ in prepared:
                    self._fsync_directory(target.parent)
            except OSError as exc:
                for completed_source, completed_target in reversed(completed):
                    try:
                        os.replace(completed_target, completed_source)
                    except OSError:
                        pass
                for source, target, _, _ in prepared:
                    source.unlink(missing_ok=True)
                    target.unlink(missing_ok=True)
                raise ArtifactStoreError(
                    "ARTIFACT_PUBLICATION_FAILED",
                    f"publication failed for role {failed_role!r}: {exc.__class__.__name__}",
                    role=failed_role,
                ) from exc

        try:
            accepted = fence([item[2] for item in prepared], atomic_publication)
        except ArtifactStoreError:
            for source, target, _, _ in prepared:
                source.unlink(missing_ok=True)
                if publication_called:
                    target.unlink(missing_ok=True)
            raise
        except Exception as exc:
            for source, target, _, _ in prepared:
                source.unlink(missing_ok=True)
                if publication_called:
                    target.unlink(missing_ok=True)
            raise ArtifactStoreError(
                "ARTIFACT_FENCE_FAILED", "publication fence callback failed"
            ) from exc
        if accepted is not True or not publication_called:
            for source, target, _, _ in prepared:
                source.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
            code = "ARTIFACT_FENCE_REJECTED" if accepted is not True else "ARTIFACT_FENCE_PROTOCOL"
            raise ArtifactStoreError(code, "publication fence did not authorize publication")

    def read_verified(
        self,
        *,
        principal_id: str,
        job_id: str,
        role: str,
        artifact: ArtifactRef,
    ) -> bytes:
        principal = _identifier(principal_id, "principal_id")
        job = _identifier(job_id, "job_id", prefix="job_")
        parsed = self._validated_ref(artifact)
        _validate_role_mime(role, parsed.mime_type)
        expected_prefix = Path("principals", principal, "jobs", job, "artifacts", role)
        parts = Path(parsed.ref).parts
        if tuple(parts[: len(expected_prefix.parts)]) != expected_prefix.parts:
            raise ArtifactStoreError("ARTIFACT_NOT_FOUND", "artifact is not in the requested scope")
        path = self._safe_existing_path(parsed.ref)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ArtifactStoreError(
                "ARTIFACT_PATH_UNSAFE", "artifact cannot be opened safely"
            ) from exc
        digest, size, content = hashlib.sha256(), 0, bytearray()
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ArtifactStoreError("ARTIFACT_PATH_UNSAFE", "artifact is not a regular file")
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                content.extend(chunk)
        if size != parsed.size_bytes:
            raise ArtifactStoreError("ARTIFACT_SIZE_MISMATCH", "artifact size verification failed")
        if f"sha256:{digest.hexdigest()}" != parsed.sha256:
            raise ArtifactStoreError("ARTIFACT_HASH_MISMATCH", "artifact hash verification failed")
        return bytes(content)

    def read_downloadable(
        self,
        *,
        principal_id: str,
        job_id: str,
        role: str,
        artifact: ArtifactRef,
        expires_at: float,
        now: float,
    ) -> bytes:
        """Read an authorized download role; source PDF and PNG are never eligible."""
        if role not in DOWNLOADABLE_ROLES:
            raise ArtifactStoreError(
                "ARTIFACT_NOT_DOWNLOADABLE", "source and raster artifacts are private"
            )
        if not _finite_timestamp(expires_at) or not _finite_timestamp(now):
            raise ArtifactStoreError("ARTIFACT_EXPIRY_INVALID", "artifact expiry is invalid")
        if now >= expires_at:
            raise ArtifactStoreError("ARTIFACT_EXPIRED", "artifact download has expired")
        return self.read_verified(
            principal_id=principal_id,
            job_id=job_id,
            role=role,
            artifact=artifact,
        )

    def delete(self, *, principal_id: str, job_id: str, artifact: ArtifactRef) -> None:
        principal = _identifier(principal_id, "principal_id")
        job = _identifier(job_id, "job_id", prefix="job_")
        parsed = self._validated_ref(artifact)
        prefix = Path("principals", principal, "jobs", job, "artifacts").parts
        if Path(parsed.ref).parts[: len(prefix)] != prefix:
            raise ArtifactStoreError("ARTIFACT_NOT_FOUND", "artifact is not in the requested scope")
        try:
            path = self._safe_existing_path(parsed.ref)
        except ArtifactStoreError as exc:
            if exc.code == "ARTIFACT_NOT_FOUND":
                return
            raise
        path.unlink()
        self._fsync_directory(path.parent)

    def discard_staged(self, artifact: ArtifactRef) -> None:
        """Remove an unpublished attempt-local artifact after admission fails or replays."""
        parsed = self._validated_ref(artifact)
        parts = Path(parsed.ref).parts
        if len(parts) != 8 or parts[0] != "principals" or parts[4] != "attempts":
            raise ArtifactStoreError("ARTIFACT_REF_INVALID", "artifact is not attempt-local")
        try:
            path = self._safe_existing_path(parsed.ref)
        except ArtifactStoreError as exc:
            if exc.code == "ARTIFACT_NOT_FOUND":
                return
            raise
        path.unlink()
        self._fsync_directory(path.parent)

    def find_staged_source(self, *, principal_id: str, job_id: str) -> tuple[ArtifactRef, Path]:
        """Resolve exactly one API-staged private source for a claimed job."""
        principal = _identifier(principal_id, "principal_id")
        job = _identifier(job_id, "job_id", prefix="job_")
        attempts = self._safe_directory(
            Path("principals", principal, "jobs", job, "attempts")
        )
        candidates: list[Path] = []
        try:
            executions = list(attempts.iterdir())
        except OSError as exc:
            raise ArtifactStoreError(
                "ARTIFACT_NOT_FOUND", "source artifact is unavailable"
            ) from exc
        for execution in executions:
            role_directory = execution / "source_pdf"
            if execution.is_symlink() or not execution.is_dir() or not role_directory.is_dir():
                continue
            if role_directory.is_symlink():
                raise ArtifactStoreError("ARTIFACT_PATH_UNSAFE", "source path is unsafe")
            candidates.extend(
                path for path in role_directory.iterdir() if path.name.endswith(".staged")
            )
        if len(candidates) != 1:
            raise ArtifactStoreError(
                "ARTIFACT_NOT_FOUND", "job must have exactly one staged source artifact"
            )
        path = candidates[0]
        relative = path.relative_to(self.root)
        safe_path = self._safe_existing_path(relative.as_posix())
        digest, size = self._measure_file(safe_path)
        return (
            ArtifactRef(
                ref=relative.as_posix(),
                sha256=f"sha256:{digest}",
                size_bytes=size,
                mime_type="application/pdf",
            ),
            safe_path,
        )

    def cleanup_orphans(self, *, older_than: float) -> int:
        return cleanup_orphans(self, older_than=older_than)

    def staged_artifacts_for_retention(self) -> list[StagedArtifactRecord]:
        """Enumerate valid staged source/raster files without following links."""
        return staged_artifacts(self)

    def published_artifacts_for_retention(self) -> list[PublishedArtifactRecord]:
        """Enumerate published bytes so unregistered crash orphans can be reconciled."""
        return published_artifacts(self)

    def _validated_ref(self, artifact: ArtifactRef) -> ArtifactRef:
        try:
            return ArtifactRef.from_dict(artifact.to_dict())
        except (AttributeError, TypeError, ValueError) as exc:
            raise ArtifactStoreError("ARTIFACT_REF_INVALID", "invalid ArtifactRef") from exc

    @classmethod
    def _verify_file(cls, path: Path, artifact: ArtifactRef) -> None:
        digest_value, size = cls._measure_file(path)
        if size != artifact.size_bytes:
            raise ArtifactStoreError("ARTIFACT_SIZE_MISMATCH", "artifact size verification failed")
        if f"sha256:{digest_value}" != artifact.sha256:
            raise ArtifactStoreError("ARTIFACT_HASH_MISMATCH", "artifact hash verification failed")

    @staticmethod
    def _measure_file(path: Path) -> tuple[str, int]:
        digest, size = hashlib.sha256(), 0
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ArtifactStoreError(
                "ARTIFACT_PATH_UNSAFE", "artifact cannot be opened safely"
            ) from exc
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ArtifactStoreError("ARTIFACT_PATH_UNSAFE", "artifact is not a regular file")
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size

    def _safe_directory(self, relative: Path) -> Path:
        current = self.root
        for part in relative.parts:
            current /= part
            try:
                info = current.lstat()
            except FileNotFoundError as exc:
                raise ArtifactStoreError(
                    "ARTIFACT_NOT_FOUND", "artifact directory is absent"
                ) from exc
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ArtifactStoreError("ARTIFACT_PATH_UNSAFE", "artifact directory is unsafe")
        return current

    def _make_private_directories(self, directory: Path) -> None:
        relative = directory.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current /= part
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            try:
                info = current.lstat()
            except OSError as exc:
                raise ArtifactStoreError(
                    "ARTIFACT_PATH_UNSAFE", "artifact directory is unsafe"
                ) from exc
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ArtifactStoreError("ARTIFACT_PATH_UNSAFE", "artifact directory is unsafe")
            os.chmod(current, 0o700)

    def _safe_existing_path(self, ref: str) -> Path:
        path = self.root
        for part in Path(ref).parts:
            path /= part
            try:
                info = path.lstat()
            except FileNotFoundError as exc:
                raise ArtifactStoreError("ARTIFACT_NOT_FOUND", "artifact does not exist") from exc
            if stat.S_ISLNK(info.st_mode):
                raise ArtifactStoreError("ARTIFACT_PATH_UNSAFE", "artifact path contains a symlink")
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ArtifactStoreError("ARTIFACT_PATH_UNSAFE", "artifact is not a regular file")
        return path

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{3,128}$")
_ROLE_MIME = {
    "source_pdf": "application/pdf",
    "page_png": "image/png",
    "prediction_ir": "application/json",
    "diagnostic_docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    # DEC-044 Dodatok D (C): side-by-side page previews.
    "source_preview": "image/png",
    "docx_preview": "image/png",
}

DOWNLOADABLE_ROLES = frozenset(
    {"prediction_ir", "diagnostic_docx", "source_preview", "docx_preview"}
)


def _identifier(value: str, name: str, prefix: str | None = None) -> str:
    if (
        not isinstance(value, str)
        or (prefix is not None and not value.startswith(prefix))
        or not _IDENTIFIER.fullmatch(value)
    ):
        raise ArtifactStoreError("ARTIFACT_SCOPE_INVALID", f"{name} is not a generated identifier")
    return value


def _validate_role_mime(role: str, mime_type: str) -> None:
    if role not in _ROLE_MIME or _ROLE_MIME[role] != mime_type:
        raise ArtifactStoreError("ARTIFACT_MIME_INVALID", "artifact role and MIME do not match")


def _finite_timestamp(value: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

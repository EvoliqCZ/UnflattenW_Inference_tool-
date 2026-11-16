"""Safe filesystem inventory for retention and crash reconciliation."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from unflatten.dataset.artifact_ref import ArtifactRef


class InventoryStore(Protocol):
    root: Path

    def _safe_existing_path(self, ref: str) -> Path: ...

    def _measure_file(self, path: Path) -> tuple[str, int]: ...


@dataclass(frozen=True)
class StagedArtifactRecord:
    principal_id: str
    job_id: str
    role: str
    artifact: ArtifactRef
    created_at: float


@dataclass(frozen=True)
class PublishedArtifactRecord:
    principal_id: str
    job_id: str
    execution_id: str | None
    role: str
    artifact: ArtifactRef
    created_at: float


def cleanup_orphans(store: InventoryStore, *, older_than: float) -> int:
    removed = 0
    for directory, child_directories, filenames in os.walk(
        store.root, topdown=False, followlinks=False
    ):
        current = Path(directory)
        if "attempts" not in current.parts:
            continue
        child_directories[:] = [
            name for name in child_directories if not (current / name).is_symlink()
        ]
        for filename in filenames:
            path = current / filename
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if filename.endswith(".staged") and stat.S_ISREG(info.st_mode) and (
                info.st_mtime < older_than
            ):
                path.unlink(missing_ok=True)
                removed += 1
        try:
            current.rmdir()
        except OSError:
            pass
    return removed


def staged_artifacts(store: InventoryStore) -> list[StagedArtifactRecord]:
    records: list[StagedArtifactRecord] = []
    for directory, child_directories, filenames in os.walk(
        store.root, topdown=True, followlinks=False
    ):
        current = Path(directory)
        child_directories[:] = [
            name for name in child_directories if not (current / name).is_symlink()
        ]
        for filename in filenames:
            record = _staged_record(store, current / filename)
            if record is not None:
                records.append(record)
    return records


def _staged_record(store: InventoryStore, path: Path) -> StagedArtifactRecord | None:
    from unflatten.service.artifact_store import _ROLE_MIME, ArtifactStoreError, _identifier

    if not path.name.endswith(".staged"):
        return None
    relative = path.relative_to(store.root)
    parts = relative.parts
    if (
        len(parts) != 8 or parts[0] != "principals" or parts[2] != "jobs"
        or parts[4] != "attempts" or parts[6] not in _ROLE_MIME
    ):
        return None
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return None
        principal = _identifier(parts[1], "principal_id")
        job = _identifier(parts[3], "job_id", prefix="job_")
        role = parts[6]
        mime_type = _ROLE_MIME[role]
        digest, size = store._measure_file(store._safe_existing_path(relative.as_posix()))
    except (ArtifactStoreError, OSError):
        return None
    return StagedArtifactRecord(
        principal, job, role,
        ArtifactRef(relative.as_posix(), f"sha256:{digest}", size, mime_type),
        info.st_mtime,
    )


def published_artifacts(store: InventoryStore) -> list[PublishedArtifactRecord]:
    records: list[PublishedArtifactRecord] = []
    for directory, child_directories, filenames in os.walk(
        store.root, topdown=True, followlinks=False
    ):
        current = Path(directory)
        child_directories[:] = [
            name for name in child_directories if not (current / name).is_symlink()
        ]
        for filename in filenames:
            record = _published_record(store, current / filename)
            if record is not None:
                records.append(record)
    return records


def _published_record(store: InventoryStore, path: Path) -> PublishedArtifactRecord | None:
    from unflatten.service.artifact_store import (
        _ROLE_MIME,
        DOWNLOADABLE_ROLES,
        ArtifactStoreError,
        _identifier,
    )

    relative = path.relative_to(store.root)
    parts = relative.parts
    if not (
        len(parts) in {7, 8} and parts[0] == "principals" and parts[2] == "jobs"
        and parts[4] == "artifacts" and parts[5] in DOWNLOADABLE_ROLES
        and parts[-1].startswith("artifact_") and parts[-1].endswith(".bin")
    ):
        return None
    role = parts[5]
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return None
        principal = _identifier(parts[1], "principal_id")
        job = _identifier(parts[3], "job_id", prefix="job_")
        execution = None if len(parts) == 7 else _identifier(parts[6], "execution_id")
        digest, size = store._measure_file(store._safe_existing_path(relative.as_posix()))
    except (ArtifactStoreError, OSError):
        return None
    return PublishedArtifactRecord(
        principal, job, execution, role,
        ArtifactRef(relative.as_posix(), f"sha256:{digest}", size, _ROLE_MIME[role]),
        info.st_mtime,
    )

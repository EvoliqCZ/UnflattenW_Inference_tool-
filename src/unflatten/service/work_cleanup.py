"""Cleanup of private attempt work, coordinated by the owning job-store fence."""

from __future__ import annotations

import re
import shutil
import stat
from pathlib import Path
from typing import TYPE_CHECKING

from unflatten.service.artifact_store import ArtifactStoreError

if TYPE_CHECKING:
    from unflatten.service.job_store import JobStore

_EXECUTION = re.compile(r"exec_[A-Za-z0-9_-]{1,128}\Z")


def _safe_root(root: Path) -> None:
    if root.is_symlink():
        raise ArtifactStoreError("ARTIFACT_PATH_UNSAFE", "work root must not be a symlink")


def delete_work_execution(root: Path, execution_id: str) -> None:
    if _EXECUTION.fullmatch(execution_id) is None:
        raise ArtifactStoreError("ARTIFACT_PATH_UNSAFE", "unsafe work execution identifier")
    _safe_root(root)
    path = root / execution_id
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        # Remove the link itself, never its target.
        path.unlink()
    elif stat.S_ISDIR(info.st_mode):
        # Linux shutil.rmtree uses descriptor-relative, symlink-safe traversal.
        shutil.rmtree(path)
    else:
        raise ArtifactStoreError("ARTIFACT_PATH_UNSAFE", "work attempt is not a directory")


def cleanup_work_executions(
    root: Path, *, jobs: JobStore, now: float, orphan_before: float
) -> None:
    _safe_root(root)
    if not root.exists():
        return
    for path in root.iterdir():
        if _EXECUTION.fullmatch(path.name) is None:
            continue
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            continue
        jobs.delete_inactive_work_execution(
            path.name,
            now=now,
            orphan_is_old=info.st_mtime <= orphan_before,
            delete_bytes=lambda name=path.name: delete_work_execution(root, name),
        )

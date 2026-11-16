"""Cancellation-safe direct request upload staging on the private artifact volume."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import AsyncIterable

from unflatten.dataset.artifact_ref import ArtifactRef
from unflatten.service.artifact_store import ArtifactStore, ArtifactStoreError


async def stage_upload(
    store: ArtifactStore,
    *,
    principal_id: str,
    job_id: str,
    execution_id: str,
    chunks: AsyncIterable[bytes],
    max_bytes: int,
) -> ArtifactRef:
    """Hash and count an upload while writing only beneath ``store.root``."""
    relative = store.prepare_staging_path(
        principal_id,
        job_id,
        execution_id,
        "source_pdf",
        "application/pdf",
        max_bytes,
    )
    path = store.root / relative
    digest, size = hashlib.sha256(), 0
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            async for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise ArtifactStoreError(
                        "ARTIFACT_CHUNK_INVALID", "artifact chunks must be bytes"
                    )
                size += len(chunk)
                if size > max_bytes:
                    raise ArtifactStoreError(
                        "ARTIFACT_TOO_LARGE", "artifact exceeds its byte ceiling"
                    )
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        store._fsync_directory(path.parent)
    except asyncio.CancelledError:
        path.unlink(missing_ok=True)
        raise
    except ArtifactStoreError:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise ArtifactStoreError("ARTIFACT_WRITE_FAILED", "upload staging failed") from exc
    return ArtifactRef(
        ref=relative.as_posix(),
        sha256=f"sha256:{digest.hexdigest()}",
        size_bytes=size,
        mime_type="application/pdf",
    )

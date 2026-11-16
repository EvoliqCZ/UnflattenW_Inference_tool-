"""Content-addressed object reads for gate evaluation (RFC003 §12/§34.1-1).

`read_object` resolves an ArtifactRef dict against a factory `out_dir`
and verifies the raw sha256 (and declared size) on every read — a stored
object is never trusted by path alone. All failures raise `StoreError`
with a stable machine code so the integrity gate can report them as
typed issues.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from unflatten.dataset.artifact_ref import ArtifactRef


class StoreError(Exception):
    """A content-addressed object could not be read and verified.

    `code` is one of: OBJECT_REF_INVALID, OBJECT_MISSING,
    OBJECT_HASH_MISMATCH, OBJECT_SIZE_MISMATCH.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def read_object(out_dir: Path, ref: dict[str, Any]) -> bytes:
    """Read the object `ref` points at, verifying sha256 + size on read."""
    try:
        parsed = ArtifactRef.from_dict(ref)
    except (ValueError, TypeError) as exc:
        raise StoreError("OBJECT_REF_INVALID", f"invalid ArtifactRef: {exc}") from exc
    path = Path(out_dir) / parsed.ref
    if not path.is_file():
        raise StoreError("OBJECT_MISSING", f"object {parsed.ref!r} does not exist")
    data = path.read_bytes()
    actual = f"sha256:{hashlib.sha256(data).hexdigest()}"
    if actual != parsed.sha256:
        raise StoreError(
            "OBJECT_HASH_MISMATCH",
            f"object {parsed.ref!r} content hash {actual} does not match "
            f"declared {parsed.sha256}",
        )
    if len(data) != parsed.size_bytes:
        raise StoreError(
            "OBJECT_SIZE_MISMATCH",
            f"object {parsed.ref!r} has {len(data)} bytes, declared "
            f"{parsed.size_bytes}",
        )
    return data

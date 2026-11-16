"""ArtifactRef — content-addressed artifact reference (RFC003 §12–§13).

`ref` is a POSIX path relative to the release root only: no traversal
(`..`), no `.` segments, no empty segments, no leading `/`, no URI scheme,
no backslashes. `sha256` is the raw content hash `sha256:<64 lowercase
hex>`. Paths are transport metadata — identity projections hash only the
content fields (`sha256`, `size_bytes`, `mime_type`), never `ref`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

_FIELDS = ("ref", "sha256", "size_bytes", "mime_type")


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable reference to one artifact by relative path and content hash."""

    ref: str
    sha256: str
    size_bytes: int
    mime_type: str

    def __post_init__(self) -> None:
        _validate_ref(self.ref)
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError(
                f"sha256 must be 'sha256:<64 lowercase hex>', got {self.sha256!r}"
            )
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError(f"size_bytes must be a non-negative int, got {self.size_bytes!r}")
        if not isinstance(self.mime_type, str) or not self.mime_type:
            raise ValueError(f"mime_type must be a non-empty string, got {self.mime_type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactRef:
        if set(data) != set(_FIELDS):
            unexpected = sorted(set(data) - set(_FIELDS))
            missing = sorted(set(_FIELDS) - set(data))
            raise ValueError(
                f"ArtifactRef dict keys mismatch: unexpected={unexpected}, missing={missing}"
            )
        return cls(**{name: data[name] for name in _FIELDS})


def _validate_ref(ref: Any) -> None:
    if not isinstance(ref, str) or not ref:
        raise ValueError(f"ref must be a non-empty string, got {ref!r}")
    if ref.startswith("/"):
        raise ValueError(f"ref must be relative, got absolute path {ref!r}")
    if "\\" in ref:
        raise ValueError(f"ref must be a POSIX path (no backslashes): {ref!r}")
    if _SCHEME_RE.match(ref):
        raise ValueError(f"ref must not carry a URI scheme: {ref!r}")
    for segment in ref.split("/"):
        if segment in ("", ".", ".."):
            raise ValueError(f"ref must not contain empty, '.' or '..' segments: {ref!r}")

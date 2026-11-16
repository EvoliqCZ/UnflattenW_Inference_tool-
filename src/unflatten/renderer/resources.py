"""Resource resolution (RFC002 stage S3 "Resource resolution").

Resolves each AST `resource` entry actually referenced by an
`inline_image` node against a `ResourceProvider`: reads the bytes,
verifies the declared SHA-256, enforces the MIME whitelist and a size
limit, and rejects path traversal. Never fetches over the network —
`ResourceProvider` is caller-supplied and local-only by convention
(`DirResourceProvider` reads from a fixed root directory).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from unflatten.ast.issues import Issue

DEFAULT_MAX_RESOURCE_BYTES = 20 * 1024 * 1024

# RFC002 S3: V0 MIME whitelist -> canonical media-part file extension.
MIME_EXTENSION: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
}


class ResourceProvider(Protocol):
    """Reads resource bytes given the AST resource's relative `path`."""

    def open(self, path: str) -> bytes: ...


class ResourcePathTraversalError(OSError):
    """Raised by `DirResourceProvider.open` when the resolved path
    escapes `root` — including via a symlink inside `root` that points
    outside it, which `resolve_resource`'s textual `..`/absolute-path
    check cannot see (the *declared* relative path can look perfectly
    clean while a symlink on disk still resolves outside the root).
    """


@dataclass(frozen=True)
class DirResourceProvider:
    """`ResourceProvider` backed by a fixed local root directory."""

    root: Path

    def open(self, path: str) -> bytes:
        root_resolved = self.root.resolve()
        target = (self.root / path).resolve()
        if not target.is_relative_to(root_resolved):
            raise ResourcePathTraversalError(
                f"resolved path {target} escapes resource root {root_resolved}"
            )
        return target.read_bytes()


@dataclass(frozen=True)
class ResolvedResource:
    resource_id: str
    sha256: str
    mime_type: str
    extension: str
    data: bytes

    @property
    def media_part_name(self) -> str:
        """`/word/media/<sha256-first16>.<canonical-extension>`.

        RFC002 "Lists"/"Package structure".
        """
        return f"word/media/{self.sha256[:16]}.{self.extension}"


def _is_path_traversal(path: str) -> bool:
    if path.startswith("/") or path.startswith("\\"):
        return True
    return ".." in PurePosixPath(path).parts


def resolve_resource(
    entry: dict,
    provider: ResourceProvider,
    path: str,
    *,
    max_bytes: int = DEFAULT_MAX_RESOURCE_BYTES,
) -> tuple[ResolvedResource | None, Issue | None]:
    """Resolve one AST `resource` entry. Returns `(resolved, issue)`.

    Exactly one of the two return values is non-`None`: a resolved
    resource on success, or a blocking `error` Issue otherwise.
    """
    resource_id = entry.get("id")
    resource_path = entry.get("path", "")
    mime_type = entry.get("mime_type")

    if _is_path_traversal(resource_path):
        return None, Issue(
            code="RESOURCE_PATH_TRAVERSAL",
            severity="error",
            node_id=resource_id,
            path=path,
            message=f"resource path '{resource_path}' is absolute or escapes the resource root.",
            details={"path": resource_path},
        )

    if mime_type not in MIME_EXTENSION:
        return None, Issue(
            code="RESOURCE_MIME_UNSUPPORTED",
            severity="error",
            node_id=resource_id,
            path=path,
            message=f"resource mime type '{mime_type}' is not in the word-core-v0 whitelist.",
            details={"mime_type": mime_type},
        )

    try:
        data = provider.open(resource_path)
    except ResourcePathTraversalError as exc:
        return None, Issue(
            code="RESOURCE_PATH_TRAVERSAL",
            severity="error",
            node_id=resource_id,
            path=path,
            message=f"resource path '{resource_path}' resolves outside the resource root: {exc}",
            details={"path": resource_path},
        )
    except OSError as exc:
        return None, Issue(
            code="RESOURCE_NOT_FOUND",
            severity="error",
            node_id=resource_id,
            path=path,
            message=f"resource '{resource_path}' could not be opened: {exc}",
            details={"path": resource_path},
        )

    if len(data) > max_bytes:
        return None, Issue(
            code="RESOURCE_TOO_LARGE",
            severity="error",
            node_id=resource_id,
            path=path,
            message=f"resource '{resource_path}' is {len(data)} bytes, exceeding the "
            f"{max_bytes}-byte limit.",
            details={"size": len(data), "limit": max_bytes},
        )

    actual_sha256 = hashlib.sha256(data).hexdigest()
    declared_sha256 = entry.get("sha256")
    if actual_sha256 != declared_sha256:
        return None, Issue(
            code="RESOURCE_HASH_MISMATCH",
            severity="error",
            node_id=resource_id,
            path=path,
            message=f"resource '{resource_path}' sha256 does not match its declared hash.",
            details={"declared": declared_sha256, "actual": actual_sha256},
        )

    return (
        ResolvedResource(
            resource_id=resource_id,
            sha256=actual_sha256,
            mime_type=mime_type,
            extension=MIME_EXTENSION[mime_type],
            data=data,
        ),
        None,
    )

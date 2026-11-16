"""ZIP + `MANIFEST.json` builder for the `artifacts.zip` download (design §3.6).

Pure module over `ArtifactStore.read_downloadable` (`artifact_store.py:322`): the
caller supplies already-fetched artifact records, the principal's capabilities, and
a `read_downloadable` callable, and gets back in-memory ZIP bytes plus the manifest.
Member names are derived only from `role`/`page_index` -- never from user text --
so there is no path-traversal surface. The API route (T33) is responsible for
authorization, expiry, and the HTTP response.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

#: DEC-044 Dodatok D §1 (B1/C): sentinel `page_index` meaning "not page-scoped"
#: (mirrors `job_store_types.NOT_PAGE_SCOPED`).
NOT_PAGE_SCOPED = -1

MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_FILENAME = "MANIFEST.json"

#: role -> capability required to include it (mirrors `api.py::_DOWNLOAD_ROLES`).
ROLE_CAPABILITIES: dict[str, str] = {
    "prediction_ir": "download_prediction_ir",
    "diagnostic_docx": "download_docx",
    "source_preview": "download_docx",
    "docx_preview": "download_docx",
}


class ArtifactZipRecord(Protocol):
    """The subset of `job_store_types.ArtifactRecord` this module needs."""

    role: str
    page_index: int
    sha256: str
    size_bytes: int
    mime_type: str


ReadDownloadable = Callable[[ArtifactZipRecord], bytes]


@dataclass(frozen=True)
class ArtifactsZipResult:
    content: bytes
    manifest: dict


def member_name(role: str, page_index: int) -> str:
    """Server-generated ZIP member name for `role`/`page_index` (design §3.6)."""
    if role == "prediction_ir":
        if page_index == NOT_PAGE_SCOPED:
            return "prediction_ir.json"
        return f"prediction_ir/page-{page_index:03d}.json"
    if role == "diagnostic_docx":
        return "diagnostic.docx"
    if role == "source_preview":
        return f"source_preview/page-{page_index:03d}.png"
    if role == "docx_preview":
        return f"docx_preview/page-{page_index:03d}.png"
    raise ValueError(f"unknown artifacts.zip role: {role!r}")


def build_artifacts_zip(
    *,
    job_id: str,
    generated_at: str,
    capabilities: frozenset[str],
    records: Sequence[ArtifactZipRecord],
    read_downloadable: ReadDownloadable,
) -> ArtifactsZipResult:
    """Build the `artifacts.zip` payload and its `MANIFEST.json`.

    Roles whose capability is missing from `capabilities` are omitted and
    reported under `omitted_roles` with `reason: "capability_denied"`. Errors
    raised by `read_downloadable` (e.g. a SHA mismatch) propagate unchanged.
    """
    allowed_roles = {role for role, cap in ROLE_CAPABILITIES.items() if cap in capabilities}
    omitted_roles = sorted(role for role in ROLE_CAPABILITIES if role not in allowed_roles)

    files: list[dict] = []
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for record in records:
            if record.role not in allowed_roles:
                continue
            content = read_downloadable(record)
            name = member_name(record.role, record.page_index)
            archive.writestr(name, content)
            files.append(
                {
                    "path": name,
                    "role": record.role,
                    "page_index": record.page_index,
                    "sha256": record.sha256,
                    "size_bytes": record.size_bytes,
                    "mime_type": record.mime_type,
                }
            )
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "job_id": job_id,
            "generated_at": generated_at,
            "files": files,
            "omitted_roles": [
                {"role": role, "reason": "capability_denied"} for role in omitted_roles
            ],
        }
        archive.writestr(MANIFEST_FILENAME, json.dumps(manifest, sort_keys=True))
    return ArtifactsZipResult(content=buffer.getvalue(), manifest=manifest)

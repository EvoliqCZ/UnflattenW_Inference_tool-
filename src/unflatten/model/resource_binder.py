"""Resource Binder (RFC004 §28.4, AST Builder T4).

Turns recovered image bytes into RFC001 `resources` records: the MIME and
pixel size claimed by the recovery are checked against the bytes, the
bytes are addressed by SHA-256 (`assets/<sha256>.<ext>`, id
`res_<16 hex>` grown by 2 hex on a prefix collision), and equal bytes
share one resource. Step 2 (re-encoding under a policy) is the identity
until a resource-materialization policy is pinned: the recovered bytes
are kept as they are.

An image recovery producer may return a named unresolved outcome; the binder
preserves that code while continuing to validate every resolved byte payload.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from unflatten.dataset.import_.importer import _image_pixel_size, _sniff_mime
from unflatten.renderer.resources import MIME_EXTENSION

RESOURCE_SLOT_UNRESOLVED = "RESOURCE_SLOT_UNRESOLVED"
RESOURCE_SLOT_DUPLICATE = "RESOURCE_SLOT_DUPLICATE"
RESOURCE_BYTES_INVALID = "RESOURCE_BYTES_INVALID"
RESOURCE_CROP_LOW_CONFIDENCE = "RESOURCE_CROP_LOW_CONFIDENCE"
RESOURCE_CROP_TOO_SMALL = "RESOURCE_CROP_TOO_SMALL"
RESOURCE_CROP_BLANK = "RESOURCE_CROP_BLANK"

_ID_PREFIX_HEX = 16


@dataclass(frozen=True, slots=True)
class RecoveredResource:
    """One image slot's recovery outcome. `slot` is the builder node id of
    the image it belongs to."""

    slot: str
    status: Literal["resolved", "unresolved"]
    mime_type: str | None = None
    data: bytes | None = None
    pixel_width: int | None = None
    pixel_height: int | None = None
    error_code: str | None = None
    fidelity: str | None = None
    coordinate_transform_hash: str | None = None
    encoding_policy_hash: str | None = None
    #: RFC004 §28.2/§28.3 recovery mode that produced this resource --
    #: additive (default `None`, `exact_crop` leaves it unset and stays
    #: byte-identical); `resource_recovery.masked_crop`/`rectified_crop`
    #: (DEC-047, opt-in) set it to their own mode name.
    mode: str | None = None

    @classmethod
    def resolved(
        cls, slot: str, *, mime_type: str, data: bytes, pixel_width: int, pixel_height: int
    ) -> RecoveredResource:
        return cls(slot, "resolved", mime_type, data, pixel_width, pixel_height)

    @classmethod
    def abstained(cls, slot: str) -> RecoveredResource:
        return cls(slot, "unresolved", error_code=RESOURCE_SLOT_UNRESOLVED)

    @classmethod
    def unresolved(cls, slot: str, error_code: str) -> RecoveredResource:
        """Return a named recovery refusal without fabricating image bytes."""
        return cls(slot, "unresolved", error_code=error_code)


@dataclass(frozen=True, slots=True)
class BinderIssue:
    code: str
    slot: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Binding:
    resources: tuple[dict[str, Any], ...]  # RFC001 records, sorted by id
    refs: Mapping[str, str]  # slot -> resource id
    assets: Mapping[str, bytes]  # path -> bytes (the Resource Manifest mapping)
    issues: tuple[BinderIssue, ...]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bind(recovered: Sequence[RecoveredResource]) -> Binding:
    issues: list[BinderIssue] = []
    slots = [r.slot for r in recovered]
    duplicated = {s for s in slots if slots.count(s) > 1}
    for slot in sorted(duplicated):
        issues.append(BinderIssue(RESOURCE_SLOT_DUPLICATE, slot))
    sha_by_slot: dict[str, str] = {}
    record_by_sha: dict[str, RecoveredResource] = {}
    for record in recovered:
        if record.slot in duplicated:
            continue
        if record.status != "resolved":
            issues.append(BinderIssue(record.error_code or RESOURCE_SLOT_UNRESOLVED, record.slot))
            continue
        problem = _invalid(record)
        if problem is not None:
            issues.append(BinderIssue(RESOURCE_BYTES_INVALID, record.slot, problem))
            continue
        sha = _sha256(record.data)
        sha_by_slot[record.slot] = sha
        record_by_sha.setdefault(sha, record)
    ids = _resource_ids(record_by_sha)
    resources = []
    assets: dict[str, bytes] = {}
    for sha, record in record_by_sha.items():
        path = f"assets/{sha}.{MIME_EXTENSION[record.mime_type]}"
        assets[path] = record.data
        resources.append(
            {
                "id": ids[sha],
                "kind": "image",
                "mime_type": record.mime_type,
                "sha256": sha,
                "path": path,
                "pixel_width": record.pixel_width,
                "pixel_height": record.pixel_height,
            }
        )
    return Binding(
        resources=tuple(sorted(resources, key=lambda r: r["id"])),
        refs={slot: ids[sha] for slot, sha in sha_by_slot.items()},
        assets=assets,
        issues=tuple(issues),
    )


def _invalid(record: RecoveredResource) -> str | None:
    if record.mime_type not in MIME_EXTENSION:
        return f"MIME {record.mime_type!r} is not in the resource allowlist"
    sniffed = _sniff_mime(record.data or b"")
    if sniffed != record.mime_type:
        return f"bytes are {sniffed!r}, the recovery claims {record.mime_type!r}"
    size = _image_pixel_size(record.data, record.mime_type)
    if size != (record.pixel_width, record.pixel_height):
        return (
            f"bytes are {size!r} px, the recovery claims "
            f"{(record.pixel_width, record.pixel_height)!r}"
        )
    return None


def _resource_ids(shas: Mapping[str, Any]) -> dict[str, str]:
    """`res_<prefix>`: 16 hex, grown by 2 until no other hash shares it."""
    ids: dict[str, str] = {}
    for sha in shas:
        length = _ID_PREFIX_HEX
        while length < len(sha) and any(
            other != sha and other[:length] == sha[:length] for other in shas
        ):
            length += 2
        ids[sha] = f"res_{sha[:length]}"
    return ids

"""Pre-parse security gate on a DOCX zip (RFC003 §17.2 stage I0).

Runs before any OOXML part is parsed: macros, OLE/embedded objects,
external template attachment, path traversal in entry names, duplicate
entry names, decompression bombs and oversized part counts are all
rejected with typed `IMPORT_SECURITY_*` Issues. Works purely on the
in-memory `ZipFile` central directory plus (for the template check) one
small rels part — nothing is ever extracted to disk.
"""

from __future__ import annotations

import posixpath
import zipfile
from dataclasses import dataclass
from io import BytesIO

from lxml import etree

from unflatten.ast.issues import Issue

_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_ATTACHED_TEMPLATE_SUFFIX = "/attachedTemplate"
_VBA_PROJECT_SUFFIX = "/vbaProject"
_SETTINGS_RELS = "word/_rels/settings.xml.rels"
_CONTENT_TYPES = "[Content_Types].xml"
_DOCUMENT_RELS = "word/_rels/document.xml.rels"
_SUPPORTED_COMPRESSION = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)


@dataclass(frozen=True)
class ScanLimits:
    """Caps for the decompression-bomb and part-count checks."""

    max_entry_uncompressed: int = 64 * 1024 * 1024
    max_total_uncompressed: int = 256 * 1024 * 1024
    max_compression_ratio: int = 200
    ratio_exempt_bytes: int = 64 * 1024
    max_parts: int = 4096


def _issue(
    code: str, message: str, path: str, severity: str = "error", **details: object
) -> Issue:
    return Issue(
        code=code,
        severity=severity,  # type: ignore[arg-type]
        node_id=None,
        path=path,
        message=message,
        details=dict(details),
    )


def _is_traversal_name(name: str) -> bool:
    if name.startswith(("/", "\\")) or "\\" in name:
        return True
    if len(name) >= 2 and name[1] == ":":  # windows drive letter
        return True
    return ".." in posixpath.normpath(name).split("/")


def _macro_issues(names: list[str]) -> list[Issue]:
    return [
        _issue(
            "IMPORT_SECURITY_MACRO",
            f"zip entry '{name}' is a VBA macro project; macro-enabled documents are rejected.",
            f"/{name}",
        )
        for name in names
        if "vbaproject" in posixpath.basename(name).lower()
    ]


def _ole_issues(names: list[str]) -> list[Issue]:
    issues = []
    for name in names:
        lowered = name.lower()
        if (
            lowered.startswith(("word/embeddings/", "word/activex"))
            or "oleobject" in posixpath.basename(lowered)
        ):
            issues.append(
                _issue(
                    "IMPORT_SECURITY_OLE_OBJECT",
                    f"zip entry '{name}' is an OLE/ActiveX embedded object; embeddings are "
                    "rejected.",
                    f"/{name}",
                )
            )
    return issues


def _macro_content_issues(zf: zipfile.ZipFile, names: set[str]) -> list[Issue]:
    """Catch renamed macro parts by declaration, not filename: macroEnabled
    / vbaProject content types in `[Content_Types].xml` and the vbaProject
    relationship type in `word/_rels/document.xml.rels`.
    """
    issues: list[Issue] = []
    if _CONTENT_TYPES in names:
        try:
            root = etree.fromstring(zf.read(_CONTENT_TYPES))
        except (etree.XMLSyntaxError, zipfile.BadZipFile):
            return [
                _issue(
                    "IMPORT_SECURITY_MALFORMED_PART",
                    f"'{_CONTENT_TYPES}' could not be parsed during the security scan.",
                    f"/{_CONTENT_TYPES}",
                )
            ]
        for el in root.iter(f"{{{_CT_NS}}}Default", f"{{{_CT_NS}}}Override"):
            content_type = (el.get("ContentType") or "").lower()
            if "macroenabled" in content_type or "vbaproject" in content_type:
                issues.append(
                    _issue(
                        "IMPORT_SECURITY_MACRO",
                        f"[Content_Types].xml declares macro content type '{content_type}'; "
                        "macro-enabled documents are rejected.",
                        f"/{_CONTENT_TYPES}",
                        content_type=content_type,
                    )
                )
    if _DOCUMENT_RELS in names:
        try:
            root = etree.fromstring(zf.read(_DOCUMENT_RELS))
        except (etree.XMLSyntaxError, zipfile.BadZipFile):
            return issues + [
                _issue(
                    "IMPORT_SECURITY_MALFORMED_PART",
                    f"'{_DOCUMENT_RELS}' could not be parsed during the security scan.",
                    f"/{_DOCUMENT_RELS}",
                )
            ]
        for rel in root.findall(f"{{{_REL_NS}}}Relationship"):
            if (rel.get("Type") or "").endswith(_VBA_PROJECT_SUFFIX):
                issues.append(
                    _issue(
                        "IMPORT_SECURITY_MACRO",
                        "document.xml.rels declares a vbaProject relationship "
                        f"(target '{rel.get('Target')}'); macro-enabled documents are rejected.",
                        f"/{_DOCUMENT_RELS}",
                        target=rel.get("Target"),
                    )
                )
    return issues


def _template_issues(zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo]) -> list[Issue]:
    """Report `w:attachedTemplate` relationships (template auto-load vector).

    Only reads the small settings rels part, and only when it passed
    the per-entry size cap (checked by the caller before this runs).

    Severity reasoning (inertness argument): this importer never opens
    the document in Word and never fetches ANY relationship target — no
    http(s) URL, `file:` path, UNC path, or drive path is ever resolved,
    read, or applied, regardless of scheme or TargetMode. Every such
    reference is therefore inert here and is reported as a typed
    `IMPORT_EXTERNAL_TEMPLATE_IGNORED` warning (details: target,
    target_mode); the import proceeds and the template is simply not
    applied — which the importer never did anyway. The one shape that
    stays a hard `IMPORT_SECURITY_EXTERNAL_TEMPLATE` error is a rel
    whose Target points INSIDE the package to an existing part that the
    import pipeline would then parse (internal TargetMode with a real
    part — template-part smuggling).
    """
    names = {info.filename for info in infos}
    if _SETTINGS_RELS not in names:
        return []
    try:
        root = etree.fromstring(zf.read(_SETTINGS_RELS))
    except (etree.XMLSyntaxError, zipfile.BadZipFile):
        return [
            _issue(
                "IMPORT_SECURITY_MALFORMED_PART",
                f"'{_SETTINGS_RELS}' could not be parsed during the security scan.",
                f"/{_SETTINGS_RELS}",
            )
        ]
    issues = []
    for rel in root.findall(f"{{{_REL_NS}}}Relationship"):
        if (rel.get("Type") or "").endswith(_ATTACHED_TEMPLATE_SUFFIX):
            target = rel.get("Target") or ""
            target_mode = rel.get("TargetMode") or "Internal"
            if target_mode != "External":
                # Internal rel targets: an OPC absolute target (leading
                # '/') resolves from the package root; a relative one
                # resolves against the source part's directory
                # (word/_rels/settings.xml.rels -> word/).
                if target.startswith("/"):
                    part = posixpath.normpath(target).lstrip("/")
                else:
                    part = posixpath.normpath(
                        posixpath.join("word", target)
                    ).lstrip("/")
                if part in names:
                    issues.append(
                        _issue(
                            "IMPORT_SECURITY_EXTERNAL_TEMPLATE",
                            f"document attaches a template whose target '{target}' "
                            f"resolves to the in-package part '{part}', which the "
                            "import pipeline would parse (template-part smuggling); "
                            "rejected.",
                            f"/{_SETTINGS_RELS}",
                            target=target,
                            target_mode=target_mode,
                            part=part,
                        )
                    )
                    continue
            issues.append(
                _issue(
                    "IMPORT_EXTERNAL_TEMPLATE_IGNORED",
                    f"document attaches a template (target '{target}'); the "
                    "importer never fetches any relationship target, so the "
                    "reference is inert regardless of scheme and ignored "
                    "(typed warning).",
                    f"/{_SETTINGS_RELS}",
                    severity="warning",
                    target=target,
                    target_mode=target_mode,
                )
            )
    return issues


def scan_docx(docx_bytes: bytes, limits: ScanLimits | None = None) -> list[Issue]:
    """Scan `docx_bytes` and return the security-gate Issues.

    An empty list means the package passed the gate cleanly. Error-level
    `IMPORT_SECURITY_*` issues fail the import; warning-level issues
    (`IMPORT_EXTERNAL_TEMPLATE_IGNORED`) are carried into the import
    report without blocking it. Never extracts to disk — in-memory
    `ZipFile` only.
    """
    limits = limits or ScanLimits()
    try:
        zf = zipfile.ZipFile(BytesIO(docx_bytes))
    except zipfile.BadZipFile as exc:
        return [
            _issue(
                "IMPORT_SECURITY_NOT_A_ZIP",
                f"input is not a valid zip archive: {exc}",
                "/",
            )
        ]

    with zf:
        infos = zf.infolist()
        names = [info.filename for info in infos]
        issues: list[Issue] = []

        if len(infos) > limits.max_parts:
            issues.append(
                _issue(
                    "IMPORT_SECURITY_TOO_MANY_PARTS",
                    f"zip contains {len(infos)} entries, exceeding the {limits.max_parts} cap.",
                    "/",
                    entry_count=len(infos),
                    limit=limits.max_parts,
                )
            )

        seen: set[str] = set()
        for name in names:
            if name in seen:
                issues.append(
                    _issue(
                        "IMPORT_SECURITY_DUPLICATE_ENTRY",
                        f"zip entry name '{name}' occurs more than once.",
                        f"/{name}",
                    )
                )
            seen.add(name)
            if _is_traversal_name(name):
                issues.append(
                    _issue(
                        "IMPORT_SECURITY_PATH_TRAVERSAL",
                        f"zip entry name '{name}' is absolute or escapes the package root.",
                        f"/{name}",
                    )
                )

        unreadable = False
        for info in infos:
            if info.flag_bits & 0x1:
                unreadable = True
                issues.append(
                    _issue(
                        "IMPORT_SECURITY_ENCRYPTED_ENTRY",
                        f"zip entry '{info.filename}' is encrypted; encrypted packages are "
                        "rejected.",
                        f"/{info.filename}",
                    )
                )
            if info.compress_type not in _SUPPORTED_COMPRESSION:
                unreadable = True
                issues.append(
                    _issue(
                        "IMPORT_SECURITY_UNSUPPORTED_COMPRESSION",
                        f"zip entry '{info.filename}' uses compression method "
                        f"{info.compress_type}; only stored (0) and deflated (8) entries "
                        "are accepted.",
                        f"/{info.filename}",
                        compress_type=info.compress_type,
                    )
                )

        total_uncompressed = 0
        bomb = False
        for info in infos:
            total_uncompressed += info.file_size
            if info.file_size > limits.max_entry_uncompressed:
                bomb = True
                issues.append(
                    _issue(
                        "IMPORT_SECURITY_BOMB_ENTRY",
                        f"zip entry '{info.filename}' declares {info.file_size} uncompressed "
                        f"bytes, exceeding the {limits.max_entry_uncompressed}-byte cap.",
                        f"/{info.filename}",
                        size=info.file_size,
                        limit=limits.max_entry_uncompressed,
                    )
                )
            if (
                info.file_size > limits.ratio_exempt_bytes
                and info.compress_size > 0
                and info.file_size // info.compress_size > limits.max_compression_ratio
            ):
                bomb = True
                issues.append(
                    _issue(
                        "IMPORT_SECURITY_BOMB_RATIO",
                        f"zip entry '{info.filename}' has a compression ratio of "
                        f"{info.file_size // info.compress_size}:1, exceeding the "
                        f"{limits.max_compression_ratio}:1 cap.",
                        f"/{info.filename}",
                        ratio=info.file_size // info.compress_size,
                        limit=limits.max_compression_ratio,
                    )
                )
        if total_uncompressed > limits.max_total_uncompressed:
            bomb = True
            issues.append(
                _issue(
                    "IMPORT_SECURITY_BOMB_TOTAL",
                    f"zip declares {total_uncompressed} total uncompressed bytes, exceeding "
                    f"the {limits.max_total_uncompressed}-byte cap.",
                    "/",
                    total=total_uncompressed,
                    limit=limits.max_total_uncompressed,
                )
            )

        # These checks read parts, so they only run on a package that
        # passed every size/bomb check and is actually readable
        # (no encrypted entries, no unsupported compression).
        if not bomb and not unreadable:
            issues.extend(_template_issues(zf, infos))
            issues.extend(_macro_content_issues(zf, set(names)))

        issues.extend(_macro_issues(names))
        issues.extend(_ole_issues(names))
        return issues

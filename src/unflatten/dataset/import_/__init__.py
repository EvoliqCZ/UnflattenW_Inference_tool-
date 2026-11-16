"""DOCX import (RFC003 D1 — importer foundation, profile word-core-v0)."""

from unflatten.dataset.import_.importer import ImportOptions, ImportResult, import_docx
from unflatten.dataset.import_.secure_scan import ScanLimits, scan_docx

__all__ = ["ImportOptions", "ImportResult", "ScanLimits", "import_docx", "scan_docx"]

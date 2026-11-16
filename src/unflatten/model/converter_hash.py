"""Deterministic source fingerprinting for the feature-cache converter."""

import hashlib
from pathlib import Path

from unflatten.import_closure import closure

CONVERTER_NAME = "unflatten.feature-cache"
CONVERTER_VERSION = "0.4.0"
_SEED_PATH = "src/unflatten/model/build_cache.py"
_PREFIX = "unflatten."


def _normalize_content(content: str) -> str:
    return "\n".join(line.rstrip() for line in content.splitlines()).rstrip() + "\n"


def collect_converter_sources(repo_root: Path) -> dict[str, str]:
    """Return the normalized AST import closure of the cache builder source."""
    repo_root = Path(repo_root).resolve()
    raw_sources = closure([repo_root / _SEED_PATH], prefix=_PREFIX)
    return {path: _normalize_content(content) for path, content in raw_sources.items()}


def hash_converter_sources(mapping: dict[str, str]) -> str:
    """Hash normalized converter sources with unambiguous path/content framing."""
    payload = "\x00".join(
        f"{relative_path}\x00{_normalize_content(content)}"
        for relative_path, content in sorted(mapping.items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def converter_source_sha256(repo_root: Path) -> str:
    return hash_converter_sources(collect_converter_sources(repo_root))


def converter_block(repo_root: Path) -> dict[str, str]:
    return {
        "name": CONVERTER_NAME,
        "version": CONVERTER_VERSION,
        "source_sha256": converter_source_sha256(repo_root),
    }

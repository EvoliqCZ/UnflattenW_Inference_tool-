"""Neutral AST import-closure walker.

Shared by the feature-cache converter fingerprint
(:mod:`unflatten.model.converter_hash`), the M2b derivation-freshness test
and the M2c gate-code digest (RFC003-ERRATA-001/DEC-023). Deliberately has
no dependency on :mod:`unflatten.model` or :mod:`unflatten.dataset` so that
callers on either side of that layering can share one walker (DEC-022:
moving this out of ``converter_hash`` changes the converter fingerprint by
construction, because ``converter_hash.py`` is itself inside its own
closure).

``prefix`` is required everywhere, with no default: it is the module-name
prefix a caller wants followed, and there is no meaningful "no filter" a
caller should silently inherit. The feature-cache converter follows every
``unflatten.*`` module (prefix ``"unflatten."``); the M2b derivation-closure
test and the M2c gate-code digest use narrower prefixes.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from pathlib import Path


def _infer_repo_root(seed: Path) -> Path:
    parts = seed.resolve().parts
    if "src" not in parts:
        raise ValueError(f"seed path has no 'src' ancestor: {seed}")
    last_src = len(parts) - 1 - parts[::-1].index("src")
    return Path(*parts[:last_src])


def module_name(repo_root: Path, path: Path) -> str:
    parts = path.relative_to(repo_root / "src").with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _package_name(repo_root: Path, path: Path) -> str:
    module = module_name(repo_root, path)
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def resolve_module(repo_root: Path, module: str, *, prefix: str) -> Path | None:
    if not module.startswith(prefix):
        return None
    base = repo_root / "src" / Path(*module.split("."))
    module_file = base.with_suffix(".py")
    if module_file.is_file():
        return module_file
    init_file = base / "__init__.py"
    return init_file if init_file.is_file() else None


def _relative_module(repo_root: Path, path: Path, level: int, module: str | None) -> str:
    package_parts = _package_name(repo_root, path).split(".")
    base_parts = package_parts[: len(package_parts) - level + 1]
    suffix = [] if module is None else module.split(".")
    return ".".join((*base_parts, *suffix))


def imported_modules(
    repo_root: Path, path: Path, tree: ast.AST, *, prefix: str
) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names if alias.name.startswith(prefix))
        elif isinstance(node, ast.ImportFrom):
            base = (
                node.module
                if node.level == 0
                else _relative_module(repo_root, path, node.level, node.module)
            )
            if base and base.startswith(prefix):
                yield base
                yield from (f"{base}.{alias.name}" for alias in node.names)


def closure(seeds: Sequence[Path], *, prefix: str) -> dict[str, str]:
    """Return the raw-source AST import closure reachable from ``seeds``.

    Follows only imports whose resolved module name starts with ``prefix``.
    Keys are paths relative to the repository root (inferred from the first
    seed's ``src`` ancestor); values are unmodified file contents — callers
    that need normalization (e.g. trailing-whitespace-insensitive hashing)
    apply it themselves.
    """
    pending = [Path(seed).resolve() for seed in seeds]
    if not pending:
        return {}
    repo_root = _infer_repo_root(pending[0])
    sources: dict[str, str] = {}
    while pending:
        path = pending.pop()
        relative_path = path.relative_to(repo_root).as_posix()
        if relative_path in sources:
            continue
        content = path.read_text()
        sources[relative_path] = content
        tree = ast.parse(content, filename=str(path))
        for module in imported_modules(repo_root, path, tree, prefix=prefix):
            resolved = resolve_module(repo_root, module, prefix=prefix)
            if resolved is not None and resolved.relative_to(repo_root).as_posix() not in sources:
                pending.append(resolved)
    return sources

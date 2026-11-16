"""Deterministic export of `web/openapi.json`, `web/codes.json` and
`web/routes.json` -- the frontend's only view of the API's routes, closed
code registries and SPA route list (design §5.2/§5.3). These three files are
committed; `--check` is the drift test that keeps them from silently
diverging from the code that defines them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from unflatten.service.api import API_TITLE, API_VERSION, register_routes
from unflatten.service.api_openapi import build_openapi
from unflatten.service.auth import CAPABILITIES
from unflatten.service.contracts import (
    ARTIFACT_ROLES,
    DIAGNOSTIC_REASON_CODES,
    DOWNLOAD_STATES,
    ENGINE_IDS,
    ENGINE_UNAVAILABLE_REASONS,
    FITS_REASONS,
    INPUT_KINDS,
    JOB_STAGES,
    JOB_STATES,
    MODEL_COMMAND_KINDS,
    MODEL_COMMAND_PHASES,
    MODEL_COMMAND_STATES,
)
from unflatten.service.docx_stage import FONT_SUBSTITUTION_CODES
from unflatten.service.errors import PROBLEM_CODES
from unflatten.service.readiness import COMPONENT_CODES, DOCX_BLOCKERS
from unflatten.service.web_static import SPA_ROUTES

#: Repository root: `src/unflatten/service/openapi_export.py` -> `service` ->
#: `unflatten` -> `src` -> root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPORT_PATHS: dict[str, Path] = {
    "openapi": _REPO_ROOT / "web" / "openapi.json",
    "codes": _REPO_ROOT / "web" / "codes.json",
    "routes": _REPO_ROOT / "web" / "routes.json",
}


def build_document() -> dict[str, Any]:
    """The exported OpenAPI document, built from a bare app (no
    `ApiDependencies`) with the SPA UI mode so UI/asset routes -- once
    wired -- are excluded from the schema the same way in every mode."""
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    register_routes(app, web_ui="spa")
    return build_openapi(app)


def build_codes() -> dict[str, Any]:
    """`web/codes.json`: every closed code/enum registry the frontend needs
    to translate, sorted so the export is deterministic (design §5.3)."""
    return {
        "schema_version": "1.0",
        "problem_codes": sorted(PROBLEM_CODES),
        "diagnostic_reason_codes": sorted(DIAGNOSTIC_REASON_CODES),
        "job_states": sorted(JOB_STATES),
        "job_stages": sorted(JOB_STAGES),
        "docx_blockers": sorted(DOCX_BLOCKERS),
        "component_codes": sorted(COMPONENT_CODES),
        "renderer_approximations": sorted(FONT_SUBSTITUTION_CODES),
        "artifact_roles": sorted(ARTIFACT_ROLES),
        "capabilities": sorted(CAPABILITIES),
        "model_command_kinds": sorted(MODEL_COMMAND_KINDS),
        "model_command_states": sorted(MODEL_COMMAND_STATES),
        "model_command_phases": sorted(MODEL_COMMAND_PHASES),
        "download_states": sorted(DOWNLOAD_STATES),
        "fits_reasons": sorted(FITS_REASONS),
        "input_kinds": sorted(INPUT_KINDS),
        "engine_ids": sorted(ENGINE_IDS),
        "engine_unavailable_reasons": sorted(ENGINE_UNAVAILABLE_REASONS),
    }


def build_routes() -> dict[str, Any]:
    """`web/routes.json`: the SPA route list, `web_static.SPA_ROUTES` is the
    single source of truth (design §5.2)."""
    return {"schema_version": "1.0", "spa_routes": list(SPA_ROUTES)}


_BUILDERS: dict[str, Callable[[], Mapping[str, Any]]] = {
    "openapi": build_document,
    "codes": build_codes,
    "routes": build_routes,
}


def _render(document: Mapping[str, Any]) -> str:
    return json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def run(*, check: bool) -> bool:
    """Write (`check=False`) or compare (`check=True`) all three exports.
    Returns whether every export already matched its committed file."""
    matches = True
    for name, builder in _BUILDERS.items():
        rendered = _render(builder())
        path = _EXPORT_PATHS[name]
        if check:
            existing = path.read_text(encoding="utf-8") if path.exists() else None
            matches = matches and existing == rendered
        else:
            path.write_text(rendered, encoding="utf-8")
    return matches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write web/{openapi,codes,routes}.json")
    mode.add_argument("--check", action="store_true", help="exit 1 if the committed files drifted")
    args = parser.parse_args(argv)
    if args.write:
        run(check=False)
        return 0
    return 0 if run(check=True) else 1


if __name__ == "__main__":
    sys.exit(main())

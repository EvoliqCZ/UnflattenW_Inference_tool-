from __future__ import annotations

import ast
from pathlib import Path

from tests.service.test_api_endpoints import _client, _submit
from tests.service.test_worker import Candidate, Sandbox

from unflatten.renderer import compiler as renderer_compiler
from unflatten.service.model_runtime import DiagnosticModelRuntime
from unflatten.service.worker import DiagnosticWorker

SERVICE_ROOT = Path(__file__).parents[2] / "src/unflatten/service"
DEPLOY_ROOT = Path(__file__).parents[2] / "deploy/service"

#: D18: `docx_stage.py`'s own import surface is a closed allowlist -- the
#: only module this file may reach `unflatten.renderer`/`compile_docx`
#: through is `unflatten.model.promotion` (imported by `run.py`, never
#: directly here). Each entry is `module -> {imported names}`.
_DOCX_STAGE_IMPORT_ALLOWLIST = {
    # DEC-044 Dodatok B/D (26.9.) + DEC-059 (5.10.) + DEC-077 (16.10., RFC004-ERRATA-011):
    # the ratified chain policies; none renders. `leader_fill_v1` replaced `leader_fill`.
    "unflatten.eval.builder_chain": {
        "headless_defaults", "table_geometry", "story_dedup", "leader_fill_v1",
        "recovered_image_materialization", "image_region_snap", "run_typography_snap",
        "tab_stops_from_regions",
    },
    "unflatten.eval.builder_chain.run": {"process_model_page", "process_model_pages"},
    "unflatten.eval.s2_gate": {"sealed"},
    "unflatten.eval.s2_gate.decode": {"seal_page"},
}


def test_every_service_module_statically_excludes_renderer_and_docx_calls() -> None:
    violations: list[str] = []
    for path in sorted(SERVICE_ROOT.glob("*.py")):
        if path.name == "docx_stage.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported = ast.unparse(node)
                if "renderer" in imported or "compile_docx" in imported:
                    violations.append(f"{path.name}: {imported}")
            if isinstance(node, ast.Call):
                target = ast.unparse(node.func)
                if target.endswith("compile_docx"):
                    violations.append(f"{path.name}: call {target}")
    assert violations == []


def test_docx_stage_import_surface_is_the_closed_allowlist_and_never_renders_directly() -> None:
    path = SERVICE_ROOT / "docx_stage.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if not node.module.startswith("unflatten"):
                continue
            allowed = _DOCX_STAGE_IMPORT_ALLOWLIST.get(node.module)
            names = {alias.name for alias in node.names}
            if allowed is None or not names <= allowed:
                violations.append(f"docx_stage.py: {ast.unparse(node)}")
        elif isinstance(node, ast.Import):
            if any(alias.name.startswith("unflatten") for alias in node.names):
                violations.append(f"docx_stage.py: {ast.unparse(node)}")
        elif isinstance(node, ast.Call):
            target = ast.unparse(node.func)
            if target.endswith("compile_docx"):
                violations.append(f"docx_stage.py: call {target}")
    assert violations == []


def test_deployment_starts_separate_private_api_and_worker_processes() -> None:
    api = (DEPLOY_ROOT / "unflatten-diagnostic-api.service").read_text(encoding="utf-8")
    worker = (DEPLOY_ROOT / "unflatten-diagnostic-worker.service").read_text(encoding="utf-8")

    assert "unflatten.service.api_cli" in api
    assert "--host 127.0.0.1" in api
    assert "unflatten.service.worker_cli" in worker
    assert "unflatten.service.worker_cli" not in api
    assert "unflatten.service.api_cli" not in worker
    assert "renderer" not in (api + worker).lower()
    assert "docx" not in (api + worker).lower()


def test_poisoned_compile_docx_cannot_intercept_diagnostic_pipeline(
    tmp_path: Path, monkeypatch
) -> None:
    def forbidden_renderer(*_args, **_kwargs):
        raise AssertionError("diagnostic Prediction IR reached compile_docx")

    monkeypatch.setattr(renderer_compiler, "compile_docx", forbidden_renderer)
    client = _client(tmp_path)
    submitted = _submit(client)
    dependencies = client.app.state.dependencies
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=dependencies.job_store,
        artifacts=dependencies.artifact_store,
        sandbox=Sandbox([]),
        model=DiagnosticModelRuntime(Candidate([])),
        policy=dependencies.runtime_policy,
        component_lock_hash=dependencies.component_lock_hash,
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    assert submitted.status_code == 202
    assert worker.run_once() is True

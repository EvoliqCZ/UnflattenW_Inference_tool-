from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.service.fake_engine import FakeEngine

from unflatten.model.quantization.spec import QuantizationSpec, bf16_spec
from unflatten.service import worker_cli
from unflatten.service.inference_engine import EngineCapabilities
from unflatten.service.pdf_sandbox import PdfSandboxFailure


def test_worker_cli_refuses_unready_sandbox_before_opening_job_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnreadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            raise PdfSandboxFailure("POPPLER_RESOURCE_LIMIT", "sandbox is unready")

    monkeypatch.setattr(worker_cli, "PdfSandbox", UnreadySandbox)
    monkeypatch.setattr(
        worker_cli,
        "JobStore",
        lambda *args, **kwargs: pytest.fail("job store must not open before sandbox readiness"),
    )

    with pytest.raises(PdfSandboxFailure, match="sandbox is unready"):
        worker_cli.main(
            [
                "--job-db",
                str(tmp_path / "jobs.sqlite3"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--work-root",
                str(tmp_path / "work"),
                "--sandbox-image",
                "sha256:" + "a" * 64,
                "--model-bundle",
                str(tmp_path / "bundle"),
                "--model-id",
                "Qwen/Qwen3-VL-4B-Instruct",
                "--revision",
                "deadbeef",
            ]
        )


def test_worker_cli_default_precision_is_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner requirement 2026-09-27: `--precision`/`--kv-cache` default to
    `bf16`, keeping every deployment that never sets `UNFLATTEN_PRECISION`/
    `UNFLATTEN_KV_CACHE` on today's byte-identical CLI surface."""
    monkeypatch.delenv("UNFLATTEN_PRECISION", raising=False)
    monkeypatch.delenv("UNFLATTEN_KV_CACHE", raising=False)
    args = worker_cli._parser().parse_args(
        [
            "--job-db",
            "jobs.sqlite3",
            "--artifact-root",
            "artifacts",
            "--work-root",
            "work",
            "--sandbox-image",
            "sha256:" + "a" * 64,
            "--model-bundle",
            "bundle",
            "--model-id",
            "Qwen/Qwen3-VL-4B-Instruct",
            "--revision",
            "deadbeef",
        ]
    )
    assert args.precision == "bf16"
    assert args.kv_cache == "bf16"
    resolved = QuantizationSpec(decoder_precision=args.precision, kv_cache_dtype=args.kv_cache)
    assert resolved.spec_hash == bf16_spec().spec_hash


def test_worker_cli_precision_defaults_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNFLATTEN_PRECISION", "int8")
    monkeypatch.setenv("UNFLATTEN_KV_CACHE", "fp8")
    args = worker_cli._parser().parse_args(
        [
            "--job-db",
            "jobs.sqlite3",
            "--artifact-root",
            "artifacts",
            "--work-root",
            "work",
            "--sandbox-image",
            "sha256:" + "a" * 64,
            "--model-bundle",
            "bundle",
            "--model-id",
            "Qwen/Qwen3-VL-4B-Instruct",
            "--revision",
            "deadbeef",
        ]
    )
    assert args.precision == "int8"
    assert args.kv_cache == "fp8"


def test_worker_cli_explicit_flag_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNFLATTEN_PRECISION", "int8")
    args = worker_cli._parser().parse_args(
        [
            "--job-db",
            "jobs.sqlite3",
            "--artifact-root",
            "artifacts",
            "--work-root",
            "work",
            "--sandbox-image",
            "sha256:" + "a" * 64,
            "--model-bundle",
            "bundle",
            "--model-id",
            "Qwen/Qwen3-VL-4B-Instruct",
            "--revision",
            "deadbeef",
            "--precision",
            "fp8",
        ]
    )
    assert args.precision == "fp8"


def test_worker_cli_rejects_unknown_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        worker_cli._parser().parse_args(
            [
                "--job-db",
                "jobs.sqlite3",
                "--artifact-root",
                "artifacts",
                "--work-root",
                "work",
                "--sandbox-image",
                "sha256:" + "a" * 64,
                "--model-bundle",
                "bundle",
                "--model-id",
                "Qwen/Qwen3-VL-4B-Instruct",
                "--revision",
                "deadbeef",
                "--precision",
                "fp16",
            ]
        )


def test_worker_cli_passes_expected_precision_spec_hash_to_verify_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The requested precision MUST be checked against the bundle's own
    precision identity via the existing `verify_bundle` mechanism."""

    class ReadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            return None

    seen = {}

    def fake_verify_bundle(bundle_dir, **kwargs):
        seen.update(kwargs)
        raise SystemExit("stop test here -- verify_bundle inputs already captured")

    class FakeJobStore:
        def __init__(self, *a, **k) -> None:
            pass

        def initialize(self) -> None:
            return None

        def requeue_stale_model_commands(self, **kwargs) -> int:
            return 0

    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "JobStore", FakeJobStore)
    monkeypatch.setattr(worker_cli, "verify_bundle", fake_verify_bundle)

    with pytest.raises(SystemExit):
        worker_cli.main(
            [
                "--job-db",
                str(tmp_path / "jobs.sqlite3"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--work-root",
                str(tmp_path / "work"),
                "--sandbox-image",
                "sha256:" + "a" * 64,
                "--model-bundle",
                str(tmp_path / "bundle"),
                "--model-id",
                "Qwen/Qwen3-VL-4B-Instruct",
                "--revision",
                "deadbeef",
                "--precision",
                "int4",
            ]
        )

    expected = QuantizationSpec(decoder_precision="int4").spec_hash
    assert seen["expected_precision_spec_hash"] == expected


def test_worker_cli_rejects_both_model_bundle_and_models_config(tmp_path: Path) -> None:
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "default_variant": "fp8",
                "variants": [
                    {
                        "variant_id": "fp8",
                        "label": "fp8",
                        "precision": "fp8",
                        "kv_cache": "fp8",
                        "bundle": str(tmp_path / "bundle"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        worker_cli.main(
            [
                "--job-db",
                str(tmp_path / "jobs.sqlite3"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--work-root",
                str(tmp_path / "work"),
                "--sandbox-image",
                "sha256:" + "a" * 64,
                "--model-bundle",
                str(tmp_path / "bundle"),
                "--models-config",
                str(config_path),
                "--model-id",
                "Qwen/Qwen3-VL-4B-Instruct",
                "--revision",
                "deadbeef",
            ]
        )


def test_worker_cli_requires_model_bundle_or_models_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        worker_cli.main(
            [
                "--job-db",
                str(tmp_path / "jobs.sqlite3"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--work-root",
                str(tmp_path / "work"),
                "--sandbox-image",
                "sha256:" + "a" * 64,
                "--model-id",
                "Qwen/Qwen3-VL-4B-Instruct",
                "--revision",
                "deadbeef",
            ]
        )


def _fake_loaded_variant(variant_id: str) -> worker_cli.LoadedVariant:
    return worker_cli.LoadedVariant(
        model=object(),
        docx_capable=True,
        identities={"component_lock_hash": "sha256:" + "a" * 64},
        selected_variant=worker_cli.SelectedVariant(
            variant_id=variant_id, precision="fp8", kv_cache="fp8", manifest_sha256=None
        ),
    )


def test_switching_model_provider_returns_resident_model_without_reloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEC-044 Dodatok D §1 fix (2026-09-27) item 3: a job naming the
    already-resident variant (or no variant at all) must not pay a reload."""
    calls: list[str] = []
    monkeypatch.setattr(
        worker_cli,
        "_load_variant",
        lambda *a, **k: calls.append("loaded") or _fake_loaded_variant("fp8"),
    )
    published: list[object] = []
    monkeypatch.setattr(
        worker_cli, "_publish_model_state", lambda *a, **k: published.append(a)
    )
    provider = worker_cli._SwitchingModelProvider(
        args=worker_cli._parser().parse_args(
            [
                "--job-db",
                str(tmp_path / "jobs.sqlite3"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--work-root",
                str(tmp_path / "work"),
                "--sandbox-image",
                "sha256:" + "a" * 64,
                "--models-config",
                str(tmp_path / "models.json"),
                "--model-id",
                "Qwen/Qwen3-VL-4B-Instruct",
                "--revision",
                "deadbeef",
            ]
        ),
        policy={},
        jobs=object(),
        readiness_writer=object(),
        initial=_fake_loaded_variant("fp8"),
    )

    resident = provider.ensure_loaded(None)
    same = provider.ensure_loaded("fp8")

    assert resident is provider._loaded.model
    assert same is provider._loaded.model
    assert calls == []
    assert published == []


def test_switching_model_provider_switches_and_publishes_the_new_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 3: a job naming a different variant frees the old model, loads
    the requested one, and republishes device facts under the new variant."""
    loaded_ids: list[str] = []

    def fake_load_variant(args, policy, *, selected_variant):
        loaded_ids.append(selected_variant.variant_id)
        return _fake_loaded_variant(selected_variant.variant_id)

    monkeypatch.setattr(worker_cli, "_load_variant", fake_load_variant)
    published: list[str | None] = []
    monkeypatch.setattr(
        worker_cli,
        "_publish_model_state",
        lambda jobs, writer, device, selected_variant: published.append(
            selected_variant.variant_id if selected_variant is not None else None
        ),
    )
    freed: list[bool] = []
    monkeypatch.setattr(
        worker_cli._SwitchingModelProvider,
        "_free_resident_model",
        lambda self: freed.append(True),
    )
    provider = worker_cli._SwitchingModelProvider(
        args=worker_cli._parser().parse_args(
            [
                "--job-db",
                str(tmp_path / "jobs.sqlite3"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--work-root",
                str(tmp_path / "work"),
                "--sandbox-image",
                "sha256:" + "a" * 64,
                "--models-config",
                str(tmp_path / "models.json"),
                "--model-id",
                "Qwen/Qwen3-VL-4B-Instruct",
                "--revision",
                "deadbeef",
            ]
        ),
        policy={},
        jobs=object(),
        readiness_writer=object(),
        initial=_fake_loaded_variant("fp8"),
    )

    provider.ensure_loaded("int4")

    assert loaded_ids == ["int4"]
    assert freed == [True]
    assert published == ["int4"]
    assert provider._loaded.selected_variant.variant_id == "int4"


def test_switching_model_provider_refuses_a_variant_with_a_different_component_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A models config may only ever name variants of ONE architecture --
    `DiagnosticWorker` pins `component_lock_hash` once at construction."""

    def fake_load_variant(args, policy, *, selected_variant):
        return worker_cli.LoadedVariant(
            model=object(),
            docx_capable=True,
            identities={"component_lock_hash": "sha256:" + "b" * 64},
            selected_variant=selected_variant,
        )

    monkeypatch.setattr(worker_cli, "_load_variant", fake_load_variant)
    monkeypatch.setattr(worker_cli, "_publish_model_state", lambda *a, **k: None)
    monkeypatch.setattr(
        worker_cli._SwitchingModelProvider, "_free_resident_model", lambda self: None
    )
    provider = worker_cli._SwitchingModelProvider(
        args=worker_cli._parser().parse_args(
            [
                "--job-db",
                str(tmp_path / "jobs.sqlite3"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--work-root",
                str(tmp_path / "work"),
                "--sandbox-image",
                "sha256:" + "a" * 64,
                "--models-config",
                str(tmp_path / "models.json"),
                "--model-id",
                "Qwen/Qwen3-VL-4B-Instruct",
                "--revision",
                "deadbeef",
            ]
        ),
        policy={},
        jobs=object(),
        readiness_writer=object(),
        initial=_fake_loaded_variant("fp8"),
    )

    with pytest.raises(ValueError, match="component_lock_hash"):
        provider.ensure_loaded("int4")


def test_worker_cli_models_config_wires_a_switching_model_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 3 end-to-end wiring: `main()` under `--models-config` must build
    `DiagnosticWorker(model_provider=..., model=None)`, not a fixed model."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "default_variant": "fp8",
                "variants": [
                    {
                        "variant_id": "fp8",
                        "label": "fp8",
                        "precision": "fp8",
                        "kv_cache": "fp8",
                        "bundle": str(bundle_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class ReadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            return None

        def readiness(self) -> dict:
            return {"ready": True, "rootless": True, "runtime_kind": "podman"}

    class FakeJobStore:
        def __init__(self, *a, **k) -> None:
            pass

        def initialize(self) -> None:
            return None

        def requeue_stale_model_commands(self, **kwargs) -> int:
            return 0

        def claim_readiness_writer(self, **kwargs):
            return object()

        def publish_readiness(self, *a, **k) -> None:
            return None

        def publish_model_state(self, *a, **k) -> None:
            return None

        def claim_next_model_command(self, **kwargs):
            return None

    captured = {}

    class FakeWorker:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def run_once(self) -> bool:
            raise KeyboardInterrupt

    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "JobStore", FakeJobStore)
    monkeypatch.setattr(worker_cli, "DiagnosticWorker", FakeWorker)
    monkeypatch.setattr(
        worker_cli,
        "_load_variant",
        lambda *a, **k: _fake_loaded_variant("fp8"),
    )
    monkeypatch.setattr(worker_cli, "_publish_model_state", lambda *a, **k: None)
    monkeypatch.setattr(worker_cli, "RetentionSweeper", lambda **k: type(
        "S", (), {"sweep": lambda self, now: None}
    )())

    assert worker_cli.main(
        [
            "--job-db",
            str(tmp_path / "jobs.sqlite3"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--work-root",
            str(tmp_path / "work"),
            "--sandbox-image",
            "sha256:" + "a" * 64,
            "--models-config",
            str(config_path),
            "--model-id",
            "Qwen/Qwen3-VL-4B-Instruct",
            "--revision",
            "deadbeef",
        ]
    ) == 0

    assert captured["model"] is None
    assert isinstance(captured["model_provider"], worker_cli._SwitchingModelProvider)


def test_worker_cli_models_config_resolves_variant_precision_and_manifest_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--models-config` fixes precision/kv_cache from the selected variant
    and forwards its bundle_manifest_sha256 pin to `verify_bundle`, ignoring
    `--model-bundle-sha256` and `--precision`/`--kv-cache`. (T32) the hf
    source variant must already be downloaded -- `_resolve_model_bundle_dir`
    resolves it from the local HF cache, never via `snapshot_download`."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    config_path = tmp_path / "models.json"
    manifest_sha = "sha256:" + "c" * 64
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "default_variant": "remote",
                "variants": [
                    {
                        "variant_id": "remote",
                        "label": "remote",
                        "precision": "int4",
                        "kv_cache": "fp8",
                        "source": {
                            "kind": "hf",
                            "repo_id": "org/repo",
                            "revision": "a" * 40,
                            "bundle_manifest_sha256": manifest_sha,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class ReadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            return None

    class FakeJobStore:
        def __init__(self, *a, **k) -> None:
            pass

        def initialize(self) -> None:
            return None

        def requeue_stale_model_commands(self, **kwargs) -> int:
            return 0

    seen = {}

    def fake_verify_bundle(passed_bundle_dir, **kwargs):
        seen["bundle_dir"] = passed_bundle_dir
        seen.update(kwargs)
        raise SystemExit("stop test here -- verify_bundle inputs already captured")

    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "JobStore", FakeJobStore)
    monkeypatch.setattr(worker_cli, "verify_bundle", fake_verify_bundle)
    monkeypatch.setattr(
        worker_cli, "is_variant_downloaded", lambda variant, *, cache_dir: (True, 123)
    )
    monkeypatch.setattr(
        worker_cli, "_hf_cached_snapshot_dir", lambda source, *, cache_dir: bundle_dir
    )

    with pytest.raises(SystemExit):
        worker_cli.main(
            [
                "--job-db",
                str(tmp_path / "jobs.sqlite3"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--work-root",
                str(tmp_path / "work"),
                "--sandbox-image",
                "sha256:" + "a" * 64,
                "--models-config",
                str(config_path),
                "--model-id",
                "Qwen/Qwen3-VL-4B-Instruct",
                "--revision",
                "deadbeef",
            ]
        )

    assert seen["bundle_dir"] == bundle_dir
    assert seen["expected_manifest_sha256"] == manifest_sha
    assert seen["expected_precision_spec_hash"] == QuantizationSpec(
        decoder_precision="int4", kv_cache_dtype="fp8"
    ).spec_hash


def test_worker_cli_models_config_local_variant_ignores_the_single_bundle_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live 2026-09-27: run-local passes `--model-bundle-sha256` (the default
    fp8 bundle's pin) and a models config of local variants; switching to the
    int4 variant verified the int4 bundle against the fp8 pin and failed
    ModelBundleInvalid. A local variant carries no pin of its own, so none is
    applied -- the single-bundle pin belongs to `--model-bundle` only."""
    bundle_dir = tmp_path / "int4"
    bundle_dir.mkdir()
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "default_variant": "int4-fp8",
                "variants": [
                    {
                        "variant_id": "int4-fp8",
                        "label": "INT4 / KV FP8",
                        "precision": "int4",
                        "kv_cache": "fp8",
                        "bundle": str(bundle_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class ReadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            return None

    class FakeJobStore:
        def __init__(self, *a, **k) -> None:
            pass

        def initialize(self) -> None:
            return None

        def requeue_stale_model_commands(self, **kwargs) -> int:
            return 0

    seen = {}

    def fake_verify_bundle(passed_bundle_dir, **kwargs):
        seen["bundle_dir"] = passed_bundle_dir
        seen.update(kwargs)
        raise SystemExit("stop test here -- verify_bundle inputs already captured")

    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "JobStore", FakeJobStore)
    monkeypatch.setattr(worker_cli, "verify_bundle", fake_verify_bundle)

    with pytest.raises(SystemExit):
        worker_cli.main(
            [
                "--job-db", str(tmp_path / "jobs.sqlite3"),
                "--artifact-root", str(tmp_path / "artifacts"),
                "--work-root", str(tmp_path / "work"),
                "--sandbox-image", "sha256:" + "a" * 64,
                "--models-config", str(config_path),
                "--model-bundle-sha256", "sha256:" + "f" * 64,
                "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
                "--revision", "deadbeef",
            ]
        )

    assert seen["bundle_dir"] == bundle_dir
    assert seen["expected_manifest_sha256"] is None


# --- T22: engine availability, --engines, EngineRegistry fold -------------


def test_worker_cli_engines_flag_defaults_to_native() -> None:
    args = worker_cli._parser().parse_args(
        [
            "--job-db", "jobs.sqlite3",
            "--artifact-root", "artifacts",
            "--work-root", "work",
            "--sandbox-image", "sha256:" + "a" * 64,
            "--model-bundle", "bundle",
            "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
            "--revision", "deadbeef",
        ]
    )
    assert args.engines == "native"


def test_parse_engines_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError, match="unknown"):
        worker_cli._parse_engines("native,imaginary")


def test_parse_engines_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="at least one"):
        worker_cli._parse_engines(" , ")


def test_engines_flag_builds_registry_from_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(worker_cli._ENGINE_FACTORIES, "candidate", lambda: FakeEngine())
    registry = worker_cli._build_engine_registry("native, candidate")
    assert set(registry.factories) == {"native", "candidate"}


def test_worker_publishes_available_engines_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(§17.5) the worker publishes `EngineRegistry.available()` through
    `publish_readiness`'s `available_engines`."""

    class UnreadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            return None

        def readiness(self) -> dict:
            return {"ready": False, "rootless": True, "runtime_kind": "podman"}

    published: list[dict] = []

    class FakeJobStore:
        def __init__(self, *a, **k) -> None:
            pass

        def initialize(self) -> None:
            return None

        def requeue_stale_model_commands(self, **kwargs) -> int:
            return 0

        def claim_readiness_writer(self, **kwargs):
            return object()

        def publish_readiness(self, *a, **kwargs) -> None:
            published.append(kwargs)
            raise KeyboardInterrupt

        def publish_model_state(self, *a, **k) -> None:
            return None

    monkeypatch.setattr(worker_cli, "PdfSandbox", UnreadySandbox)
    monkeypatch.setattr(worker_cli, "JobStore", FakeJobStore)
    monkeypatch.setattr(
        worker_cli, "_load_variant", lambda *a, **k: _fake_loaded_variant("fp8")
    )
    monkeypatch.setattr(worker_cli, "_publish_model_state", lambda *a, **k: None)
    monkeypatch.setattr(
        worker_cli,
        "RetentionSweeper",
        lambda **k: type("S", (), {"sweep": lambda self, now: None})(),
    )
    monkeypatch.setattr(worker_cli.time, "sleep", lambda seconds: None)

    assert (
        worker_cli.main(
            [
                "--job-db", str(tmp_path / "jobs.sqlite3"),
                "--artifact-root", str(tmp_path / "artifacts"),
                "--work-root", str(tmp_path / "work"),
                "--sandbox-image", "sha256:" + "a" * 64,
                "--model-bundle", str(tmp_path / "bundle"),
                "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
                "--revision", "deadbeef",
            ]
        )
        == 0
    )

    assert published
    assert published[0]["available_engines"] == ("native",)


def test_switching_model_provider_refuses_when_initial_variant_engine_unavailable(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not available"):
        worker_cli._SwitchingModelProvider(
            args=worker_cli._parser().parse_args(
                [
                    "--job-db", str(tmp_path / "jobs.sqlite3"),
                    "--artifact-root", str(tmp_path / "artifacts"),
                    "--work-root", str(tmp_path / "work"),
                    "--sandbox-image", "sha256:" + "a" * 64,
                    "--models-config", str(tmp_path / "models.json"),
                    "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
                    "--revision", "deadbeef",
                ]
            ),
            policy={},
            jobs=object(),
            readiness_writer=object(),
            initial=_fake_loaded_variant("fp8"),
            engine_registry=worker_cli.EngineRegistry(factories={}),
        )


def test_switching_model_provider_refuses_switch_to_non_native_engine_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(§17.5) a variant naming a non-native engine is refused even when that
    engine IS available -- `_load_variant` only loads native variants."""

    def fake_load_variant(args, policy, *, selected_variant):
        return worker_cli.LoadedVariant(
            model=object(),
            docx_capable=True,
            identities={"component_lock_hash": "sha256:" + "a" * 64},
            selected_variant=worker_cli.SelectedVariant(
                variant_id="candidate-variant",
                precision="fp8",
                kv_cache="fp8",
                manifest_sha256=None,
                engine="candidate",
            ),
        )

    monkeypatch.setattr(worker_cli, "_load_variant", fake_load_variant)
    monkeypatch.setattr(
        worker_cli._SwitchingModelProvider, "_free_resident_model", lambda self: None
    )
    monkeypatch.setattr(worker_cli, "_publish_model_state", lambda *a, **k: None)

    registry = worker_cli.EngineRegistry(
        factories={
            "native": worker_cli.NativeEngine,
            "candidate": lambda: FakeEngine(
                capabilities=EngineCapabilities(
                    engine_id="candidate",
                    batches_pages=True,
                    max_batch_pages=8,
                    parity_status="proven",
                )
            ),
        }
    )
    provider = worker_cli._SwitchingModelProvider(
        args=worker_cli._parser().parse_args(
            [
                "--job-db", str(tmp_path / "jobs.sqlite3"),
                "--artifact-root", str(tmp_path / "artifacts"),
                "--work-root", str(tmp_path / "work"),
                "--sandbox-image", "sha256:" + "a" * 64,
                "--models-config", str(tmp_path / "models.json"),
                "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
                "--revision", "deadbeef",
            ]
        ),
        policy={},
        jobs=object(),
        readiness_writer=object(),
        initial=_fake_loaded_variant("fp8"),
        engine_registry=registry,
    )

    with pytest.raises(ValueError, match="only 'native'"):
        provider.ensure_loaded("candidate-variant")


# --- T32: --variant auto, executor in the loop, sandbox_rootful -----------


def test_variant_auto_enqueues_worker_load_before_worker_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(design §4.2 "Štartovací auto-load") `--variant auto` must enqueue and
    run the startup `load` model command BEFORE `DiagnosticWorker` is ever
    constructed -- so the model_commands row (and its byte progress) exists
    from the first second, not only once the worker loop starts."""
    from unflatten.service.job_store import JobStore as RealJobStore

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "default_variant": "auto",
                "variants": [
                    {
                        "variant_id": "fp8",
                        "label": "fp8",
                        "precision": "fp8",
                        "kv_cache": "fp8",
                        "bundle": str(bundle_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class ReadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            return None

        def readiness(self) -> dict:
            return {"ready": True, "rootless": True, "runtime_kind": "podman"}

    order: list[str] = []
    original_enqueue = RealJobStore.enqueue_model_command

    def wrapped_enqueue(self, **kwargs):
        order.append("enqueued")
        return original_enqueue(self, **kwargs)

    class FakeWorker:
        def __init__(self, **kwargs) -> None:
            order.append("worker_constructed")

        def run_once(self) -> bool:
            raise KeyboardInterrupt

    monkeypatch.setattr(RealJobStore, "enqueue_model_command", wrapped_enqueue)
    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "DiagnosticWorker", FakeWorker)
    monkeypatch.setattr(worker_cli, "_load_variant", lambda *a, **k: _fake_loaded_variant("fp8"))
    monkeypatch.setattr(worker_cli, "_publish_model_state", lambda *a, **k: None)
    monkeypatch.setattr(
        worker_cli, "RetentionSweeper",
        lambda **k: type("S", (), {"sweep": lambda self, now: None})(),
    )
    monkeypatch.setattr(
        worker_cli, "_measure_own_device_facts",
        lambda device: {"free_vram_mib": 1_000_000.0, "total_vram_mib": 1_000_000.0,
                        "compute_capability": (9, 0)},
    )
    monkeypatch.setattr(worker_cli, "resolve_default_variant", lambda *a, **k: ("fp8", "auto"))

    assert worker_cli.main(
        [
            "--job-db", str(tmp_path / "jobs.sqlite3"),
            "--artifact-root", str(tmp_path / "artifacts"),
            "--work-root", str(tmp_path / "work"),
            "--sandbox-image", "sha256:" + "a" * 64,
            "--models-config", str(config_path),
            "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
            "--revision", "deadbeef",
        ]
    ) == 0

    assert order == ["enqueued", "worker_constructed"]


def test_explicit_variant_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit `--variant` must win over `default_variant: "auto"` --
    `resolve_default_variant` (the "auto" machinery) must never even run."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "default_variant": "auto",
                "variants": [
                    {
                        "variant_id": "fp8",
                        "label": "fp8",
                        "precision": "fp8",
                        "kv_cache": "fp8",
                        "bundle": str(bundle_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class ReadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            return None

        def readiness(self) -> dict:
            return {"ready": True, "rootless": True, "runtime_kind": "podman"}

    class FakeJobStore:
        def __init__(self, *a, **k) -> None:
            pass

        def initialize(self) -> None:
            return None

        def requeue_stale_model_commands(self, **kwargs) -> int:
            return 0

        def claim_readiness_writer(self, **kwargs):
            return object()

        def publish_readiness(self, *a, **k) -> None:
            return None

        def publish_model_state(self, *a, **k) -> None:
            return None

        def claim_next_model_command(self, **kwargs):
            return None

    class FakeWorker:
        def __init__(self, **kwargs) -> None:
            pass

        def run_once(self) -> bool:
            raise KeyboardInterrupt

    def forbidden_resolve(*a, **k):
        pytest.fail("resolve_default_variant must never run when --variant is explicit")

    published_variant = {}

    def fake_publish_model_state(jobs, writer, device, selected_variant):
        published_variant["id"] = (
            selected_variant.variant_id if selected_variant is not None else None
        )

    def fake_load_variant(a, policy, *, selected_variant):
        return _fake_loaded_variant("fp8")

    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "JobStore", FakeJobStore)
    monkeypatch.setattr(worker_cli, "DiagnosticWorker", FakeWorker)
    monkeypatch.setattr(worker_cli, "resolve_default_variant", forbidden_resolve)
    monkeypatch.setattr(worker_cli, "_load_variant", fake_load_variant)
    monkeypatch.setattr(worker_cli, "_publish_model_state", fake_publish_model_state)
    monkeypatch.setattr(
        worker_cli, "RetentionSweeper",
        lambda **k: type("S", (), {"sweep": lambda self, now: None})(),
    )

    assert worker_cli.main(
        [
            "--job-db", str(tmp_path / "jobs.sqlite3"),
            "--artifact-root", str(tmp_path / "artifacts"),
            "--work-root", str(tmp_path / "work"),
            "--sandbox-image", "sha256:" + "a" * 64,
            "--models-config", str(config_path),
            "--variant", "fp8",
            "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
            "--revision", "deadbeef",
        ]
    ) == 0

    assert published_variant["id"] == "fp8"


def test_nothing_fits_systemexit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--variant auto` with no fitting variant must `SystemExit` fail-closed
    -- the same posture as booting without a model today."""
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "default_variant": "auto",
                "variants": [
                    {
                        "variant_id": "fp8",
                        "label": "fp8",
                        "precision": "fp8",
                        "kv_cache": "fp8",
                        "bundle": str(tmp_path / "bundle"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class ReadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            return None

        def readiness(self):
            return {"ready": True, "rootless": True, "runtime_kind": "podman"}

    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "_measure_own_device_facts", lambda *a: None)
    monkeypatch.setattr(worker_cli, "_resolve_auto_variant", lambda *a, **k: None)

    with pytest.raises(SystemExit, match="no configured variant fits"):
        worker_cli.main(
            [
                "--job-db", str(tmp_path / "jobs.sqlite3"),
                "--artifact-root", str(tmp_path / "artifacts"),
                "--work-root", str(tmp_path / "work"),
                "--sandbox-image", "sha256:" + "a" * 64,
                "--models-config", str(config_path),
                "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
                "--revision", "deadbeef",
            ]
        )


def test_snapshot_download_never_called_outside_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(design §4.2 "Sieť") only `ModelCommandExecutor`
    (`model_commands._snapshot_download`) may ever call `snapshot_download`;
    `model_variants._snapshot_download` -- the function the pre-T32
    `resolve_variant_bundle` path used -- must never be touched by
    `worker_cli`."""
    from unflatten.service import model_commands as mc
    from unflatten.service import model_variants as mv

    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "default_variant": "auto",
                "variants": [
                    {
                        "variant_id": "remote",
                        "label": "remote",
                        "precision": "fp8",
                        "kv_cache": "fp8",
                        "source": {
                            "kind": "hf",
                            "repo_id": "org/repo",
                            "revision": "a" * 40,
                            "bundle_manifest_sha256": "sha256:" + "c" * 64,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class ReadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            return None

        def readiness(self) -> dict:
            return {"ready": True, "rootless": True, "runtime_kind": "podman"}

    class FakeWorker:
        def __init__(self, **kwargs) -> None:
            pass

        def run_once(self) -> bool:
            raise KeyboardInterrupt

    calls = {"executor_download": 0}

    def forbidden(**kwargs):
        pytest.fail("model_variants._snapshot_download must never be called by worker_cli")

    def allowed(**kwargs):
        calls["executor_download"] += 1
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(mv, "_snapshot_download", forbidden)
    monkeypatch.setattr(mc, "_snapshot_download", allowed)
    monkeypatch.setattr(mc, "verify_bundle", lambda *a, **k: None)
    monkeypatch.setattr(mc, "is_variant_downloaded", lambda variant, *, cache_dir: (False, None))
    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "DiagnosticWorker", FakeWorker)
    monkeypatch.setattr(
        worker_cli, "_load_variant", lambda *a, **k: _fake_loaded_variant("remote")
    )
    monkeypatch.setattr(worker_cli, "_publish_model_state", lambda *a, **k: None)
    monkeypatch.setattr(
        worker_cli, "RetentionSweeper",
        lambda **k: type("S", (), {"sweep": lambda self, now: None})(),
    )
    monkeypatch.setattr(
        worker_cli, "_measure_own_device_facts",
        lambda device: {"free_vram_mib": 1_000_000.0, "total_vram_mib": 1_000_000.0,
                        "compute_capability": (9, 0)},
    )
    monkeypatch.setattr(worker_cli, "resolve_default_variant", lambda *a, **k: ("remote", "auto"))

    assert worker_cli.main(
        [
            "--job-db", str(tmp_path / "jobs.sqlite3"),
            "--artifact-root", str(tmp_path / "artifacts"),
            "--work-root", str(tmp_path / "work"),
            "--sandbox-image", "sha256:" + "a" * 64,
            "--models-config", str(config_path),
            "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
            "--revision", "deadbeef",
        ]
    ) == 0

    assert calls["executor_download"] == 1


def test_hf_variant_without_snapshot_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_resolve_model_bundle_dir` must refuse fail-closed, never download,
    when an `hf` source variant is not already present locally."""
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "default_variant": "remote",
                "variants": [
                    {
                        "variant_id": "remote",
                        "label": "remote",
                        "precision": "fp8",
                        "kv_cache": "fp8",
                        "source": {
                            "kind": "hf",
                            "repo_id": "org/repo",
                            "revision": "a" * 40,
                            "bundle_manifest_sha256": "sha256:" + "c" * 64,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = worker_cli._parser().parse_args(
        [
            "--job-db", str(tmp_path / "jobs.sqlite3"),
            "--artifact-root", str(tmp_path / "artifacts"),
            "--work-root", str(tmp_path / "work"),
            "--sandbox-image", "sha256:" + "a" * 64,
            "--models-config", str(config_path),
            "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
            "--revision", "deadbeef",
        ]
    )
    monkeypatch.setattr(
        worker_cli, "is_variant_downloaded", lambda variant, *, cache_dir: (False, None)
    )

    with pytest.raises(worker_cli.ModelsConfigInvalid, match="not downloaded locally"):
        worker_cli._resolve_model_bundle_dir(args)


@pytest.mark.parametrize(
    "variant",
    [
        {"variant_id": "not-yet", "label": "FP8", "precision": "fp8", "kv_cache": "fp8",
         "availability": "unpublished"},
        {"variant_id": "not-yet", "label": "Unflatten W MoE", "availability": "planned"},
    ],
)
def test_resolve_model_bundle_dir_non_published_variant_fails_closed(
    tmp_path: Path, variant: dict[str, str]
) -> None:
    """An explicit `--variant` naming an unpublished/planned catalogue entry
    is refused with a named reason -- it has no bundle or source to load."""
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "default_variant": "local",
                "variants": [
                    {"variant_id": "local", "label": "local", "precision": "bf16",
                     "kv_cache": "bf16", "bundle": str(tmp_path / "bundle")},
                    variant,
                ],
            }
        ),
        encoding="utf-8",
    )
    args = worker_cli._parser().parse_args(
        [
            "--job-db", str(tmp_path / "jobs.sqlite3"),
            "--artifact-root", str(tmp_path / "artifacts"),
            "--work-root", str(tmp_path / "work"),
            "--sandbox-image", "sha256:" + "a" * 64,
            "--models-config", str(config_path),
            "--variant", "not-yet",
            "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
            "--revision", "deadbeef",
        ]
    )

    with pytest.raises(worker_cli.ModelsConfigInvalid, match="not published"):
        worker_cli._resolve_model_bundle_dir(args)


def test_requeue_stale_on_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`main()` must requeue stale `model_commands` rows before claiming
    anything, on every start (design §4.2 "Reštart workera uprostred")."""

    class ReadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            return None

    calls = {}

    class FakeJobStore:
        def __init__(self, *a, **k) -> None:
            pass

        def initialize(self) -> None:
            return None

        def requeue_stale_model_commands(self, **kwargs) -> int:
            calls.update(kwargs)
            return 0

    def fake_verify_bundle(*a, **k):
        raise SystemExit("stop test here -- requeue_stale_model_commands already captured")

    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "JobStore", FakeJobStore)
    monkeypatch.setattr(worker_cli, "verify_bundle", fake_verify_bundle)

    with pytest.raises(SystemExit):
        worker_cli.main(
            [
                "--job-db", str(tmp_path / "jobs.sqlite3"),
                "--artifact-root", str(tmp_path / "artifacts"),
                "--work-root", str(tmp_path / "work"),
                "--sandbox-image", "sha256:" + "a" * 64,
                "--model-bundle", str(tmp_path / "bundle"),
                "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
                "--revision", "deadbeef",
            ]
        )

    policy = worker_cli.load_runtime_policy(worker_cli.DEFAULT_RUNTIME_POLICY_PATH)
    assert calls["stale_after"] == 2 * policy["heartbeat_interval_seconds"]


def test_executor_runs_before_worker_run_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In the main loop, `ModelCommandExecutor.run_once()` must run before
    `worker.run_once()` (design §4.2) so a queued switch takes effect for the
    *next* job claim, not the one already picked up."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "default_variant": "fp8",
                "variants": [
                    {
                        "variant_id": "fp8",
                        "label": "fp8",
                        "precision": "fp8",
                        "kv_cache": "fp8",
                        "bundle": str(bundle_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class ReadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            return None

        def readiness(self) -> dict:
            return {"ready": True, "rootless": True, "runtime_kind": "podman"}

    class FakeJobStore:
        def __init__(self, *a, **k) -> None:
            pass

        def initialize(self) -> None:
            return None

        def requeue_stale_model_commands(self, **kwargs) -> int:
            return 0

        def claim_readiness_writer(self, **kwargs):
            return object()

        def publish_readiness(self, *a, **k) -> None:
            return None

        def publish_model_state(self, *a, **k) -> None:
            return None

    order: list[str] = []

    class FakeExecutor:
        def __init__(self, **kwargs) -> None:
            pass

        def run_once(self) -> bool:
            order.append("executor")
            return False

    class FakeWorker:
        def __init__(self, **kwargs) -> None:
            pass

        def run_once(self) -> bool:
            order.append("worker")
            raise KeyboardInterrupt

    monkeypatch.setattr(worker_cli, "PdfSandbox", ReadySandbox)
    monkeypatch.setattr(worker_cli, "JobStore", FakeJobStore)
    monkeypatch.setattr(worker_cli, "ModelCommandExecutor", FakeExecutor)
    monkeypatch.setattr(worker_cli, "DiagnosticWorker", FakeWorker)
    monkeypatch.setattr(worker_cli, "_load_variant", lambda *a, **k: _fake_loaded_variant("fp8"))
    monkeypatch.setattr(worker_cli, "_publish_model_state", lambda *a, **k: None)
    monkeypatch.setattr(
        worker_cli, "RetentionSweeper",
        lambda **k: type("S", (), {"sweep": lambda self, now: None})(),
    )

    assert worker_cli.main(
        [
            "--job-db", str(tmp_path / "jobs.sqlite3"),
            "--artifact-root", str(tmp_path / "artifacts"),
            "--work-root", str(tmp_path / "work"),
            "--sandbox-image", "sha256:" + "a" * 64,
            "--models-config", str(config_path),
            "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
            "--revision", "deadbeef",
        ]
    ) == 0

    assert order == ["executor", "worker"]


def test_publish_readiness_sandbox_rootful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`publish_readiness` must derive `sandbox_rootful` from
    `PdfSandbox.readiness()` (`rootless is False and runtime_kind ==
    'docker'`, design §4.1/§4.2)."""

    class UnreadySandbox:
        def __init__(self, **kwargs: object) -> None:
            pass

        def require_ready(self) -> None:
            return None

        def readiness(self) -> dict:
            return {"ready": False, "rootless": False, "runtime_kind": "docker"}

    published: list[dict] = []

    class FakeJobStore:
        def __init__(self, *a, **k) -> None:
            pass

        def initialize(self) -> None:
            return None

        def requeue_stale_model_commands(self, **kwargs) -> int:
            return 0

        def claim_readiness_writer(self, **kwargs):
            return object()

        def publish_readiness(self, *a, **kwargs) -> None:
            published.append(kwargs)
            raise KeyboardInterrupt

        def publish_model_state(self, *a, **k) -> None:
            return None

    monkeypatch.setattr(worker_cli, "PdfSandbox", UnreadySandbox)
    monkeypatch.setattr(worker_cli, "JobStore", FakeJobStore)
    monkeypatch.setattr(worker_cli, "_load_variant", lambda *a, **k: _fake_loaded_variant("fp8"))
    monkeypatch.setattr(worker_cli, "_publish_model_state", lambda *a, **k: None)
    monkeypatch.setattr(
        worker_cli, "RetentionSweeper",
        lambda **k: type("S", (), {"sweep": lambda self, now: None})(),
    )
    monkeypatch.setattr(worker_cli.time, "sleep", lambda seconds: None)

    assert worker_cli.main(
        [
            "--job-db", str(tmp_path / "jobs.sqlite3"),
            "--artifact-root", str(tmp_path / "artifacts"),
            "--work-root", str(tmp_path / "work"),
            "--sandbox-image", "sha256:" + "a" * 64,
            "--model-bundle", str(tmp_path / "bundle"),
            "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
            "--revision", "deadbeef",
        ]
    ) == 0

    assert published
    assert published[0]["sandbox_rootful"] is True


def test_auto_considers_only_available_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """(item A.6) `--variant auto` must resolve only among variants whose
    engine is available -- a variant naming an unavailable engine (here
    `candidate`, `parity_status="unproven"`) must never be a candidate even if it
    would otherwise be preferred."""
    config = {
        "schema_version": "1.0",
        "default_variant": "auto",
        "variants": [
            {
                "variant_id": "candidate-variant", "label": "v", "precision": "fp8",
                "kv_cache": "fp8", "engine": "candidate", "bundle": "/x",
            },
            {
                "variant_id": "native-variant", "label": "n", "precision": "int4",
                "kv_cache": "bf16", "engine": "native", "bundle": "/y",
            },
        ],
    }
    registry = worker_cli.EngineRegistry(
        factories={
            "native": FakeEngine,
            "candidate": lambda: FakeEngine(
                capabilities=EngineCapabilities(
                    engine_id="candidate", batches_pages=False, max_batch_pages=1,
                    parity_status="unproven",
                )
            ),
        }
    )

    class FakeJobs:
        def get_setting(self, key: str) -> str | None:
            return None

    monkeypatch.setattr(
        worker_cli, "_measure_own_device_facts",
        lambda device: {"free_vram_mib": 1_000_000.0, "total_vram_mib": 1_000_000.0,
                        "compute_capability": (9, 0)},
    )

    args = worker_cli._parser().parse_args(
        [
            "--job-db", "jobs.sqlite3",
            "--artifact-root", "artifacts",
            "--work-root", "work",
            "--sandbox-image", "sha256:" + "a" * 64,
            "--models-config", "models.json",
            "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
            "--revision", "deadbeef",
        ]
    )

    resolved = worker_cli._resolve_auto_variant(args, config, registry, FakeJobs())

    assert resolved == "native-variant"


_MINIMAL_WORKER_ARGV = [
    "--job-db", "jobs.sqlite3",
    "--artifact-root", "artifacts",
    "--work-root", "work",
    "--sandbox-image", "sha256:" + "a" * 64,
    "--model-bundle", "bundle",
    "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
    "--revision", "deadbeef",
]


def test_worker_cli_kv_cache_mode_defaults_to_static_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner decision 2026-09-29 (`docs/runs/vc-eviction-prep-2026-09-28.md`
    `bypass_static_kv`): the service decodes on a preallocated static KV
    cache unless the operator says otherwise; since the owner decision later
    that day (CUDA graphs, `docs/runs/cuda-graphs-decode-2026-09-29.md`)
    the q=1 step also replays piecewise CUDA graphs."""
    monkeypatch.delenv("UNFLATTEN_KV_CACHE_MODE", raising=False)
    args = worker_cli._parser().parse_args(_MINIMAL_WORKER_ARGV)
    assert args.kv_cache_mode == "static_graph"


def test_worker_cli_kv_cache_mode_empty_environment_is_static_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # docker compose's env_file exports a present-but-empty variable.
    monkeypatch.setenv("UNFLATTEN_KV_CACHE_MODE", "")
    args = worker_cli._parser().parse_args(_MINIMAL_WORKER_ARGV)
    assert args.kv_cache_mode == "static_graph"


def test_worker_cli_kv_cache_mode_environment_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNFLATTEN_KV_CACHE_MODE", "bf16_dynamic")
    args = worker_cli._parser().parse_args(_MINIMAL_WORKER_ARGV)
    assert args.kv_cache_mode == "bf16_dynamic"


def test_worker_cli_kv_cache_mode_flag_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNFLATTEN_KV_CACHE_MODE", "static")
    args = worker_cli._parser().parse_args(
        [*_MINIMAL_WORKER_ARGV, "--kv-cache-mode", "bf16_dynamic"]
    )
    assert args.kv_cache_mode == "bf16_dynamic"


def test_worker_cli_rejects_unknown_kv_cache_mode_flag() -> None:
    with pytest.raises(SystemExit):
        worker_cli._parser().parse_args([*_MINIMAL_WORKER_ARGV, "--kv-cache-mode", "paged"])


def test_worker_cli_invalid_kv_cache_mode_environment_refused_before_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """argparse never checks a default against `choices`, so a typo in the
    environment variable must be refused explicitly -- before the sandbox
    probe or any model load, and naming the variable and the valid values."""
    monkeypatch.setenv("UNFLATTEN_KV_CACHE_MODE", "statik")
    monkeypatch.setattr(
        worker_cli,
        "PdfSandbox",
        lambda **kwargs: pytest.fail("sandbox must not be built with an invalid kv cache mode"),
    )
    with pytest.raises(ValueError, match="UNFLATTEN_KV_CACHE_MODE") as excinfo:
        worker_cli.main(
            [
                "--job-db", str(tmp_path / "jobs.sqlite3"),
                "--artifact-root", str(tmp_path / "artifacts"),
                "--work-root", str(tmp_path / "work"),
                "--sandbox-image", "sha256:" + "a" * 64,
                "--model-bundle", str(tmp_path / "bundle"),
                "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
                "--revision", "deadbeef",
            ]
        )
    message = str(excinfo.value)
    assert "'statik'" in message
    assert "bf16_dynamic" in message and "static" in message


def _capture_load_candidate_runtime_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_argv: tuple[str, ...] = ()
) -> dict:
    seen: dict = {}

    class _FakeCandidateRuntime:
        def __init__(self, kv_cache_mode: str) -> None:
            self.kv_cache_mode = kv_cache_mode

        def kv_cache_effective(self) -> str:
            return self.kv_cache_mode

    def fake_load_candidate_runtime(**kwargs):
        seen.update(kwargs)
        return _FakeCandidateRuntime(kwargs["kv_cache_mode"])

    monkeypatch.setattr(
        worker_cli,
        "verify_bundle",
        lambda bundle_dir, **kwargs: {
            "identities": {
                "tokenizer_artifact_set_hash": "sha256:" + "1" * 64,
                "checkpoint_sha256": "sha256:" + "2" * 64,
                "component_lock_hash": "sha256:" + "3" * 64,
                "loop_guard_v2_caps_sha256": "sha256:" + "4" * 64,
            }
        },
    )
    monkeypatch.setattr(worker_cli, "bind_bundle_loop_guard_caps", lambda bundle_dir: None)
    monkeypatch.setattr(
        worker_cli,
        "_load_decode_policy_block",
        lambda bundle_dir: {
            "decode": {
                "loop_guard": "v2",
                "max_steps": 4608,
                "typed_decode": True,
                "region_decode": True,
            },
            "selection_policy_hash": "sha256:" + "5" * 64,
        },
    )
    monkeypatch.setattr(worker_cli, "load_candidate_runtime", fake_load_candidate_runtime)
    monkeypatch.setattr(worker_cli, "DiagnosticModelRuntime", lambda candidate: candidate)
    monkeypatch.setattr(worker_cli, "diagnostic_docx_capability", lambda model: True)
    monkeypatch.setattr(worker_cli, "manifest_sha256", lambda bundle_dir: "sha256:" + "6" * 64)
    args = worker_cli._parser().parse_args(
        [
            "--job-db", str(tmp_path / "jobs.sqlite3"),
            "--artifact-root", str(tmp_path / "artifacts"),
            "--work-root", str(tmp_path / "work"),
            "--sandbox-image", "sha256:" + "a" * 64,
            "--model-bundle", str(tmp_path / "bundle"),
            "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
            "--revision", "deadbeef",
            *extra_argv,
        ]
    )
    loaded = worker_cli._load_variant(
        args, {"decode_steps_hard_ceiling": 8192}, selected_variant=None
    )
    assert loaded.model.kv_cache_mode == seen["kv_cache_mode"]
    return seen


def test_load_variant_passes_static_graph_kv_cache_mode_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("UNFLATTEN_KV_CACHE_MODE", raising=False)
    seen = _capture_load_candidate_runtime_kwargs(tmp_path, monkeypatch)
    assert seen["kv_cache_mode"] == "static_graph"


def test_hf_snapshot_checkpoint_link_resolves_to_verified_blob_before_safe_load(
    tmp_path,
    monkeypatch,
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    blob = tmp_path / "verified-blob"
    blob.write_bytes(b"weights")
    (snapshot / "checkpoint.weights.pt").symlink_to(blob)
    variant = worker_cli.SelectedVariant("hf-model", "bf16", "bf16", "sha256:" + "6" * 64)
    monkeypatch.setattr(
        worker_cli, "_resolve_model_bundle_dir", lambda *a, **k: (snapshot, variant)
    )
    seen = _capture_load_candidate_runtime_kwargs(tmp_path, monkeypatch)
    assert seen["checkpoint_path"] == blob.resolve()
    assert seen["pin"].checkpoint_sha256 == "sha256:" + "2" * 64


def test_load_variant_passes_overridden_kv_cache_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNFLATTEN_KV_CACHE_MODE", "bf16_dynamic")
    seen = _capture_load_candidate_runtime_kwargs(tmp_path, monkeypatch)
    assert seen["kv_cache_mode"] == "bf16_dynamic"


def test_load_variant_logs_kv_cache_effective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("UNFLATTEN_KV_CACHE_MODE", raising=False)
    with caplog.at_level("INFO", logger=worker_cli.LOGGER.name):
        _capture_load_candidate_runtime_kwargs(tmp_path, monkeypatch)
    assert "kv_cache_effective=static" in caplog.text


def test_resolve_quantization_auto_reserves_static_kv_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--precision auto` must cost the static KV reserve the same way
    `GET /v1/models`'s `fits` does, or the two disagree on which tier fits."""
    import torch

    seen: list[float] = []

    def fake_resolve(**kwargs):
        seen.append(kwargs["extra_reserve_mib"])
        return QuantizationSpec(decoder_precision="int4", kv_cache_dtype="fp8")

    monkeypatch.setattr(worker_cli, "resolve_auto_quantization", fake_resolve)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8 << 30, 12 << 30))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(is_integrated=False))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    for mode in ("static", "bf16_dynamic"):
        args = worker_cli._parser().parse_args(
            [*_MINIMAL_WORKER_ARGV, "--precision", "auto", "--kv-cache-mode", mode]
        )
        worker_cli._resolve_quantization(args, None)
    assert seen == [648.0, 0.0]


# --- fp8 shared input quantization (owner speed item 5, 2026-09-29) -------


def test_worker_cli_fp8_shared_input_quant_defaults_to_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner decisions 2026-09-29/30: sharing is on by default for an fp8
    decoder (bit-identical, docs/runs/fp8-qkv-fusion-2026-09-29.md); "unset"
    (None) means that default, distinct from an explicit request."""
    monkeypatch.delenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", raising=False)
    args = worker_cli._parser().parse_args(_MINIMAL_WORKER_ARGV)
    assert args.fp8_shared_input_quant is None


def test_worker_cli_fp8_shared_input_quant_empty_environment_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # docker compose's env_file exports a present-but-empty variable.
    monkeypatch.setenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", "")
    args = worker_cli._parser().parse_args(_MINIMAL_WORKER_ARGV)
    assert args.fp8_shared_input_quant is None


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE"])
def test_worker_cli_fp8_shared_input_quant_false_environment_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", value)
    args = worker_cli._parser().parse_args(_MINIMAL_WORKER_ARGV)
    assert args.fp8_shared_input_quant is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "True"])
def test_worker_cli_fp8_shared_input_quant_true_environment_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", value)
    args = worker_cli._parser().parse_args(_MINIMAL_WORKER_ARGV)
    assert args.fp8_shared_input_quant is True


def test_worker_cli_fp8_shared_input_quant_flags_override_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", "1")
    args = worker_cli._parser().parse_args(
        [*_MINIMAL_WORKER_ARGV, "--no-fp8-shared-input-quant"]
    )
    assert args.fp8_shared_input_quant is False
    monkeypatch.setenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", "0")
    args = worker_cli._parser().parse_args([*_MINIMAL_WORKER_ARGV, "--fp8-shared-input-quant"])
    assert args.fp8_shared_input_quant is True


def _main_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--job-db", str(tmp_path / "jobs.sqlite3"),
        "--artifact-root", str(tmp_path / "artifacts"),
        "--work-root", str(tmp_path / "work"),
        "--sandbox-image", "sha256:" + "a" * 64,
        "--model-bundle", str(tmp_path / "bundle"),
        "--model-id", "Qwen/Qwen3-VL-4B-Instruct",
        "--revision", "deadbeef",
        *extra,
    ]


def test_worker_cli_invalid_fp8_shared_input_quant_environment_refused_before_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", "maybe")
    monkeypatch.setattr(
        worker_cli, "PdfSandbox", lambda **kwargs: pytest.fail("sandbox must not be built")
    )
    with pytest.raises(ValueError, match="UNFLATTEN_FP8_SHARED_INPUT_QUANT") as excinfo:
        worker_cli.main(_main_argv(tmp_path))
    assert "'maybe'" in str(excinfo.value)


@pytest.mark.parametrize("precision", ["bf16", "int8", "int4"])
def test_worker_cli_fp8_shared_input_quant_refused_for_explicit_non_fp8_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, precision: str
) -> None:
    """An explicit non-fp8 `--precision` plus the flag is an operator error:
    refused at startup, before the sandbox or any load, not silently
    ignored."""
    monkeypatch.delenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", raising=False)
    monkeypatch.setattr(
        worker_cli, "PdfSandbox", lambda **kwargs: pytest.fail("sandbox must not be built")
    )
    with pytest.raises(ValueError, match="fp8") as excinfo:
        worker_cli.main(
            _main_argv(tmp_path, "--precision", precision, "--fp8-shared-input-quant")
        )
    assert precision in str(excinfo.value)


def test_load_variant_shares_fp8_input_quant_by_default_for_an_fp8_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", raising=False)
    with caplog.at_level("INFO", logger=worker_cli.LOGGER.name):
        seen = _capture_load_candidate_runtime_kwargs(
            tmp_path, monkeypatch, ("--precision", "fp8")
        )
    assert seen["fp8_shared_input_quant"] is True
    assert "fp8_shared_input_quant=True" in caplog.text
    assert "fp8_shared_input_quant_requested=False" in caplog.text


def test_load_variant_explicit_off_disables_sharing_for_an_fp8_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", raising=False)
    with caplog.at_level("INFO", logger=worker_cli.LOGGER.name):
        seen = _capture_load_candidate_runtime_kwargs(
            tmp_path, monkeypatch, ("--precision", "fp8", "--no-fp8-shared-input-quant")
        )
    assert seen["fp8_shared_input_quant"] is False
    assert "fp8_shared_input_quant=False" in caplog.text


def test_load_variant_default_with_a_bf16_decoder_loads_unshared_without_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The on-by-default must never refuse a non-fp8 decoder -- only an
    explicit --fp8-shared-input-quant does."""
    monkeypatch.delenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", raising=False)
    with caplog.at_level("INFO", logger=worker_cli.LOGGER.name):
        seen = _capture_load_candidate_runtime_kwargs(
            tmp_path, monkeypatch, ("--precision", "bf16")
        )
    assert seen["fp8_shared_input_quant"] is False
    assert "fp8_shared_input_quant=False" in caplog.text


def test_load_variant_passes_fp8_shared_input_quant_for_an_fp8_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", raising=False)
    with caplog.at_level("INFO", logger=worker_cli.LOGGER.name):
        seen = _capture_load_candidate_runtime_kwargs(
            tmp_path, monkeypatch, ("--precision", "fp8", "--fp8-shared-input-quant")
        )
    assert seen["fp8_shared_input_quant"] is True
    assert "fp8_shared_input_quant=True" in caplog.text
    assert "fp8_shared_input_quant_requested=True" in caplog.text


def test_load_variant_resolved_non_fp8_variant_loads_unshared_and_logs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A variant resolved at load time (models-config switch, `auto`) that is
    not fp8 must still load -- a switch has already freed the resident
    model, so refusing here would leave the worker with none -- but without
    sharing, and the log line says requested vs effective."""
    monkeypatch.delenv("UNFLATTEN_FP8_SHARED_INPUT_QUANT", raising=False)
    monkeypatch.setattr(
        worker_cli,
        "_resolve_quantization",
        lambda args, selected_variant: QuantizationSpec(decoder_precision="int8"),
    )
    with caplog.at_level("INFO", logger=worker_cli.LOGGER.name):
        seen = _capture_load_candidate_runtime_kwargs(
            tmp_path, monkeypatch, ("--fp8-shared-input-quant",)
        )
    assert seen["fp8_shared_input_quant"] is False
    assert "fp8_shared_input_quant=False" in caplog.text
    assert "fp8_shared_input_quant_requested=True" in caplog.text

"""RED tests for T8: `unflatten.service.model_bundle` (design D7/D9).

`scripts/build_model_bundle.py` builds the fixture bundles here re-uses
(the same "mutate the real artefact" discipline as `tests/scripts/
test_build_model_bundle.py`) rather than hand-assembling bundle files, so
these tests exercise `load_manifest`/`component_lock_hash_from_bundle`/
`verify_bundle` against a genuinely produced bundle.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tests.model.test_checkpoint_ancestry import _base_model
from tests.model.test_checkpoint_ancestry import _build as _build_base_ancestry
from tests.model.test_manifests import valid_exact_text_s1_lock

from unflatten.model import checkpoint_ancestry as ca
from unflatten.model.candidate_loader import QWEN_CANDIDATE_MODEL_ID, QWEN_CANDIDATE_REVISION
from unflatten.model.capability_profile import PRIMARY_S1_CAPABILITY_PROFILE
from unflatten.model.loop_guard_v2_caps import (
    CAPS_BINDING_KEYS,
    CAPS_SCHEMA,
    CAPS_SEMANTICS,
    PERIOD_MAX,
)
from unflatten.model.qwen_tokenizer_artifacts import tokenizer_artifact_set_hash
from unflatten.model.typed_head_config import typed_head_config_hash
from unflatten.service import api_cli
from unflatten.service.model_bundle import (
    BUNDLE_SCHEMA_VERSION,
    COMPONENT_LOCK_FILENAME,
    MANIFEST_FILENAME,
    ModelBundleInvalid,
    component_lock_hash_from_bundle,
    load_manifest,
    manifest_sha256,
    verify_bundle,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_model_bundle.py"
_spec = importlib.util.spec_from_file_location("build_model_bundle", _SCRIPT)
assert _spec is not None and _spec.loader is not None
bmb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bmb)

RUN_ID = "s1run_" + "5" * 64
S1_RUN_MANIFEST_HASH = "sha256:" + "8" * 64


def _cache_identity() -> dict:
    return {key: f"value-{key}" for key in CAPS_BINDING_KEYS}


def _component_lock() -> dict:
    lock = valid_exact_text_s1_lock()
    for entry in lock["components"]:
        if entry["kind"] == "model_capability_profile":
            entry["semantic_hash"] = PRIMARY_S1_CAPABILITY_PROFILE.profile_hash
            entry["version"] = PRIMARY_S1_CAPABILITY_PROFILE.schema_version
    return lock


def _allowlist() -> dict:
    return {
        "qwen_text_allowlist_schema_version": "1.0.0",
        "tokenizer_artifact_set_hash": "sha256:" + "a" * 64,
        "native_vocab_size": 8,
        "allowed_native_ids": list(range(8)),
        "denied": [],
        "classification_policy_version": "1.0.0",
        "control_text_check": "applied",
    }


def _caps_payload(cache_identity: dict) -> dict:
    return {
        "schema": CAPS_SCHEMA,
        "guard": "v2",
        "semantics": CAPS_SEMANTICS,
        "period_max": PERIOD_MAX,
        "close_recovery_variant": "A",
        "capability_profile_id": PRIMARY_S1_CAPABILITY_PROFILE.profile_id,
        "text_run_cap": 5,
        "periodic_cap": {str(p): 5 for p in range(1, PERIOD_MAX + 1)},
        "source": {"target_identity": dict(cache_identity)},
    }


def _write_pinned_hashes(tmp_path: Path) -> Path:
    path = tmp_path / "qwen-pinned-hashes.json"
    path.write_text(
        json.dumps(
            {
                "revision": QWEN_CANDIDATE_REVISION,
                "weights": {"model.safetensors": "sha256:" + "1" * 64},
                "processor": {"processor_config.json": "sha256:" + "2" * 64},
                "weights_set_sha256": "sha256:" + "1" * 64,
                "processor_set_sha256": "sha256:" + "2" * 64,
            }
        )
    )
    return path


def _prepare_hf_cache(tmp_path: Path, out_dir: Path) -> Path:
    snapshot = (
        tmp_path / "hf-cache" / ("models--" + QWEN_CANDIDATE_MODEL_ID.replace("/", "--"))
        / "snapshots" / QWEN_CANDIDATE_REVISION
    )
    snapshot.mkdir(parents=True)
    for name, content in {
        "model.safetensors": b"weights", "processor_config.json": b"processor",
        "tokenizer.json": b"{}", "tokenizer_config.json": b"{}",
    }.items():
        (snapshot / name).write_bytes(content)
    pins_path = out_dir / "qwen-pinned-hashes.json"
    pins = json.loads(pins_path.read_text())
    pins["weights"] = {"model.safetensors": "sha256:" + hashlib.sha256(b"weights").hexdigest()}
    pins["processor"] = {
        "processor_config.json": "sha256:" + hashlib.sha256(b"processor").hexdigest()
    }
    pins_path.write_text(json.dumps(pins))
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][pins_path.name] = {
        "sha256": "sha256:" + hashlib.sha256(pins_path.read_bytes()).hexdigest(),
        "size_bytes": pins_path.stat().st_size,
    }
    manifest["identities"]["tokenizer_artifact_set_hash"] = tokenizer_artifact_set_hash(
        snapshot, model_id=QWEN_CANDIDATE_MODEL_ID, revision=QWEN_CANDIDATE_REVISION
    )
    manifest_path.write_text(json.dumps(manifest))
    return tmp_path / "hf-cache"


def _build_bundle(tmp_path: Path, *, precision=None) -> Path:
    base = _build_base_ancestry(tmp_path, base_model=_base_model())
    ca.verify_base(base)
    extended = ca.extend_for_checkpoint(
        base, step=1, run_id=RUN_ID, s1_run_manifest_hash=S1_RUN_MANIFEST_HASH
    )
    cache_identity = _cache_identity()
    state = {
        "model_state_dict": {"w": torch.zeros(2)},
        "cache_identity": cache_identity,
        "typed_head_config_hash": typed_head_config_hash(),
        "step": 1,
        "ancestry_hash": extended["ancestry_hash"],
        "optimizer_state_dict": {"m": torch.zeros(2)},
    }
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    checkpoint_path = source_dir / "checkpoint.pt"
    torch.save(state, checkpoint_path)
    finalized = ca.finalize_checkpoint_document(extended, checkpoint_path)
    ca.write_sidecar(finalized, checkpoint_path)

    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / COMPONENT_LOCK_FILENAME).write_text(json.dumps(_component_lock()))
    (run_dir / "qwen_text_allowlist.json").write_text(json.dumps(_allowlist()))
    caps_path = tmp_path / "caps.json"
    caps_path.write_text(json.dumps(_caps_payload(cache_identity)))

    out_dir = tmp_path / "bundle"
    bmb.build(
        checkpoint_path=checkpoint_path,
        component_lock_path=run_dir / COMPONENT_LOCK_FILENAME,
        allowlist_path=run_dir / "qwen_text_allowlist.json",
        caps_path=caps_path,
        selection_policy_path=bmb._DEFAULT_SELECTION_POLICY_PATH,
        pinned_hashes_path=_write_pinned_hashes(tmp_path),
        out_dir=out_dir,
        precision=precision,
    )
    return out_dir


class TestNoModelSurface:
    def test_module_import_leaves_torch_out_of_sys_modules(self):
        script = (
            "import sys; import unflatten.service.model_bundle; "
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


class TestApiCliStaysTorchFree:
    def test_api_cli_argument_parsing_and_bundle_read_never_import_torch(self):
        # Subprocess (design D9 §6.1's own guard, `test_checkpoint_ancestry.
        # TestNoModelSurface`'s pattern): another test module in this
        # session may already have imported torch, which would make an
        # in-process `sys.modules` check depend on test order.
        script = (
            "import sys\n"
            "import unflatten.service.api_cli as api_cli\n"
            "from unflatten.service.model_bundle import component_lock_hash_from_bundle\n"
            "api_cli.build_parser().parse_args([\n"
            "    '--principal-file', '/nonexistent/principals.json',\n"
            "    '--job-store', '/nonexistent/jobs.sqlite3',\n"
            "    '--artifact-root', '/nonexistent/artifacts',\n"
            "    '--model-bundle', '/nonexistent/bundle',\n"
            "])\n"
            "assert 'torch' not in sys.modules, sorted("
            "m for m in sys.modules if 'torch' in m)\n"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize(
        ("environment_pin", "cli_pin", "expected_pin", "rejects_pin"),
        [
            (None, None, None, False),
            ("", None, None, False),
            ("sha256:" + "b" * 64, None, "sha256:" + "b" * 64, False),
            ("sha256:" + "b" * 64, "sha256:" + "c" * 64, "sha256:" + "c" * 64, True),
        ],
        ids=("absent", "empty", "matching", "cli-mismatch"),
    )
    def test_startup_verifies_bundle_with_optional_sha_pin(
        self, tmp_path, monkeypatch, environment_pin, cli_pin, expected_pin, rejects_pin
    ):
        seen: dict[str, object] = {}

        def verify(bundle_dir, *, expected_manifest_sha256, hf_cache_dir=None):
            seen.update(bundle_dir=bundle_dir, pin=expected_manifest_sha256, cache=hf_cache_dir)
            if expected_manifest_sha256 not in (None, "sha256:" + "b" * 64):
                raise ModelBundleInvalid("model bundle manifest sha256 does not match pin")
            return {"identities": {"component_lock_hash": "sha256:" + "a" * 64}}

        principal_file = tmp_path / "principals.json"
        principal_file.write_text(
            json.dumps(
                {
                    "principal_file_schema_version": "1.0.0",
                    "principals": [
                        {
                            "principal_id": "principal_alpha",
                            "token": "test-token-with-at-least-thirty-two-bytes",
                            "capabilities": {
                                "cancel_own_job": False,
                                "download_prediction_ir": True,
                                "read_status": True,
                                "submit_diagnostic": True,
                                "download_docx": True,
                            },
                        }
                    ],
                }
            )
        )
        principal_file.chmod(0o400)
        if environment_pin is None:
            monkeypatch.delenv("UNFLATTEN_MODEL_BUNDLE_SHA256", raising=False)
        else:
            monkeypatch.setenv("UNFLATTEN_MODEL_BUNDLE_SHA256", environment_pin)
        monkeypatch.setattr(api_cli, "verify_bundle", verify)
        monkeypatch.setattr(api_cli.uvicorn, "run", lambda *args, **kwargs: None)

        argv = [
            "--principal-file", str(principal_file),
            "--job-store", str(tmp_path / "jobs.sqlite3"),
            "--artifact-root", str(tmp_path / "artifacts"),
            "--model-bundle", str(tmp_path / "bundle"),
        ]
        if cli_pin is not None:
            argv.extend(("--model-bundle-sha256", cli_pin))
        if rejects_pin:
            with pytest.raises(ModelBundleInvalid, match="does not match pin"):
                api_cli.main(argv)
        else:
            assert api_cli.main(argv) == 0
        assert seen == {
            "bundle_dir": tmp_path / "bundle",
            "pin": expected_pin,
            "cache": None,
        }


class TestLoadManifest:
    def test_load_manifest_of_a_built_bundle(self, tmp_path):
        out_dir = _build_bundle(tmp_path)
        manifest = load_manifest(out_dir)
        assert manifest["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION

    def test_load_manifest_refuses_missing_file(self, tmp_path):
        with pytest.raises(ModelBundleInvalid):
            load_manifest(tmp_path)

    def test_load_manifest_refuses_wrong_schema_version(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "bundle_schema_version": "9.9.9",
                    "files": {},
                    "identities": {},
                    "base_model": {},
                }
            )
        )
        with pytest.raises(ModelBundleInvalid, match="bundle_schema_version"):
            load_manifest(tmp_path)


class TestComponentLockHashFromBundle:
    def test_reads_only_the_manifest(self, tmp_path):
        out_dir = _build_bundle(tmp_path)
        manifest = load_manifest(out_dir)
        # Delete every OTHER bundle file -- proves this function never
        # opens them (design D9 §6.1).
        for name in list(out_dir.iterdir()):
            if name.name != MANIFEST_FILENAME:
                name.unlink()
        assert component_lock_hash_from_bundle(out_dir) == manifest["identities"][
            "component_lock_hash"
        ]

    def test_refuses_missing_component_lock_hash(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
                    "files": {},
                    "identities": {},
                    "base_model": {},
                }
            )
        )
        with pytest.raises(ModelBundleInvalid, match="component_lock_hash"):
            component_lock_hash_from_bundle(tmp_path)


class TestManifestSha256:
    def test_manifest_sha256_is_stable_and_changes_on_edit(self, tmp_path):
        out_dir = _build_bundle(tmp_path)
        first = manifest_sha256(out_dir)
        assert manifest_sha256(out_dir) == first
        manifest_path = out_dir / MANIFEST_FILENAME
        manifest_path.write_text(manifest_path.read_text() + "\n")
        assert manifest_sha256(out_dir) != first


class TestVerifyBundle:
    def test_verify_accepts_a_freshly_built_bundle_and_returns_the_manifest(self, tmp_path):
        out_dir = _build_bundle(tmp_path)
        manifest = verify_bundle(out_dir, hf_cache_dir=_prepare_hf_cache(tmp_path, out_dir))
        assert manifest["identities"]["component_lock_hash"]

    def test_verify_accepts_a_matching_expected_sha256_pin(self, tmp_path):
        out_dir = _build_bundle(tmp_path)
        cache_dir = _prepare_hf_cache(tmp_path, out_dir)
        verify_bundle(
            out_dir,
            expected_manifest_sha256=manifest_sha256(out_dir),
            hf_cache_dir=cache_dir,
        )

    def test_verify_refuses_a_mismatched_expected_sha256_pin(self, tmp_path):
        out_dir = _build_bundle(tmp_path)
        with pytest.raises(ModelBundleInvalid, match="UNFLATTEN_MODEL_BUNDLE_SHA256"):
            verify_bundle(
                out_dir,
                expected_manifest_sha256="sha256:" + "0" * 64,
                hf_cache_dir=_prepare_hf_cache(tmp_path, out_dir),
            )


def test_verify_accepts_bare_hex_pinned_weight_hashes_as_the_real_pins_file_stores_them(tmp_path):
    """`data/qwen-pinned-hashes.json` on prod stores bare hex digests; the
    manifest/file digests carry a `sha256:` prefix. The first real verify
    (26.9.) refused a correct weight file on the prefix alone."""
    from unflatten.service import model_bundle as mb

    assert mb._sha_eq("sha256:" + "a" * 64, "a" * 64)
    assert mb._sha_eq("a" * 64, "sha256:" + "a" * 64)
    assert not mb._sha_eq("sha256:" + "a" * 64, "sha256:" + "b" * 64)
    assert not mb._sha_eq("", "")
    assert not mb._sha_eq("sha256:", "")


# --- processor asset absent from the tokenizer snapshot (S5 on S4 v16, 17. 10.) --------------


def _pin_extra_processor_asset(out_dir: Path, name: str, content: bytes) -> None:
    pins_path = out_dir / "qwen-pinned-hashes.json"
    pins = json.loads(pins_path.read_text())
    pins["processor"][name] = "sha256:" + hashlib.sha256(content).hexdigest()
    pins_path.write_text(json.dumps(pins))
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][pins_path.name] = {
        "sha256": "sha256:" + hashlib.sha256(pins_path.read_bytes()).hexdigest(),
        "size_bytes": pins_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest))


def _fallback_cache(tmp_path: Path, name: str, content: bytes) -> Path:
    snapshot = (
        tmp_path / "hf-fallback" / ("models--" + QWEN_CANDIDATE_MODEL_ID.replace("/", "--"))
        / "snapshots" / QWEN_CANDIDATE_REVISION
    )
    snapshot.mkdir(parents=True)
    (snapshot / name).write_bytes(content)
    return tmp_path / "hf-fallback"


def test_verify_takes_a_pinned_asset_missing_from_the_tokenizer_snapshot_from_the_fallback_cache(
    tmp_path, monkeypatch
):
    """The v16 runs hashed the tokenizer over a snapshot WITHOUT
    video_preprocessor_config.json (the file's name enters the hash), while the
    pinned processor set requires it: no single snapshot satisfies both."""
    out_dir = _build_bundle(tmp_path)
    cache = _prepare_hf_cache(tmp_path, out_dir)
    _pin_extra_processor_asset(out_dir, "video_preprocessor_config.json", b"video")
    monkeypatch.delenv("UNFLATTEN_HF_ASSET_FALLBACK_CACHE", raising=False)
    with pytest.raises(ModelBundleInvalid, match="missing pinned processor asset"):
        verify_bundle(out_dir, hf_cache_dir=cache)
    fallback = _fallback_cache(tmp_path, "video_preprocessor_config.json", b"video")
    monkeypatch.setenv("UNFLATTEN_HF_ASSET_FALLBACK_CACHE", str(fallback))
    verify_bundle(out_dir, hf_cache_dir=cache)


def test_verify_fallback_asset_with_other_bytes_is_refused(tmp_path, monkeypatch):
    out_dir = _build_bundle(tmp_path)
    cache = _prepare_hf_cache(tmp_path, out_dir)
    _pin_extra_processor_asset(out_dir, "video_preprocessor_config.json", b"video")
    fallback = _fallback_cache(tmp_path, "video_preprocessor_config.json", b"tampered")
    monkeypatch.setenv("UNFLATTEN_HF_ASSET_FALLBACK_CACHE", str(fallback))
    with pytest.raises(ModelBundleInvalid, match="sha256 does not match"):
        verify_bundle(out_dir, hf_cache_dir=cache)


def test_verify_fallback_never_replaces_an_asset_present_in_the_primary_snapshot(
    tmp_path, monkeypatch
):
    out_dir = _build_bundle(tmp_path)
    cache = _prepare_hf_cache(tmp_path, out_dir)
    snapshot = next((cache).glob("models--*/snapshots/*"))
    (snapshot / "processor_config.json").write_bytes(b"changed")
    fallback = _fallback_cache(tmp_path, "processor_config.json", b"processor")
    monkeypatch.setenv("UNFLATTEN_HF_ASSET_FALLBACK_CACHE", str(fallback))
    with pytest.raises(ModelBundleInvalid, match="sha256 does not match"):
        verify_bundle(out_dir, hf_cache_dir=cache)

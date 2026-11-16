"""Owner requirement item 3: a model bundle's `precision` identity, and
`verify_bundle`'s refusal to load a bundle at the wrong precision.

Reuses `test_model_bundle._build_bundle` (a real bundle built by
`scripts/build_model_bundle.py`, not a hand-assembled fixture).
"""

from __future__ import annotations

import pytest
from tests.service.test_model_bundle import _build_bundle

from unflatten.model.quantization.spec import QuantizationSpec, bf16_spec
from unflatten.service.model_bundle import (
    ModelBundleInvalid,
    load_manifest,
    precision_spec_hash_from_bundle,
    verify_bundle,
)


def test_default_build_records_the_bf16_sentinel_precision(tmp_path):
    out_dir = _build_bundle(tmp_path)
    manifest = load_manifest(out_dir)
    assert manifest["identities"]["precision"]["spec_hash"] == bf16_spec().spec_hash


def test_quantized_build_records_the_given_spec(tmp_path):
    spec = QuantizationSpec(decoder_precision="int8", kv_cache_dtype="fp8")
    out_dir = _build_bundle(tmp_path, precision=spec)
    manifest = load_manifest(out_dir)
    assert manifest["identities"]["precision"]["spec_hash"] == spec.spec_hash
    assert manifest["identities"]["precision"]["spec_hash"] != bf16_spec().spec_hash


def test_precision_spec_hash_from_bundle_defaults_to_bf16_when_absent(tmp_path):
    """A bundle built before quantization existed carries no `precision`
    block at all -- it must still be read as bf16, not refused."""
    out_dir = _build_bundle(tmp_path)
    manifest_path = out_dir / "manifest.json"
    import json

    manifest = json.loads(manifest_path.read_text())
    del manifest["identities"]["precision"]
    manifest_path.write_text(json.dumps(manifest))
    assert precision_spec_hash_from_bundle(out_dir) == bf16_spec().spec_hash


def test_verify_bundle_accepts_matching_expected_precision(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "unflatten.service.model_bundle._verify_base_model_pin", lambda *a, **k: None
    )
    out_dir = _build_bundle(tmp_path)
    verify_bundle(out_dir, expected_precision_spec_hash=bf16_spec().spec_hash)


def test_verify_bundle_refuses_bf16_bundle_when_quantized_expected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "unflatten.service.model_bundle._verify_base_model_pin", lambda *a, **k: None
    )
    out_dir = _build_bundle(tmp_path)
    spec = QuantizationSpec(decoder_precision="int4")
    with pytest.raises(ModelBundleInvalid, match="precision"):
        verify_bundle(out_dir, expected_precision_spec_hash=spec.spec_hash)


def test_verify_bundle_refuses_quantized_bundle_when_bf16_expected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "unflatten.service.model_bundle._verify_base_model_pin", lambda *a, **k: None
    )
    spec = QuantizationSpec(decoder_precision="fp8", kv_cache_dtype="fp8")
    out_dir = _build_bundle(tmp_path, precision=spec)
    with pytest.raises(ModelBundleInvalid, match="precision"):
        verify_bundle(out_dir, expected_precision_spec_hash=bf16_spec().spec_hash)

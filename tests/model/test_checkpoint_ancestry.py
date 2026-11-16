"""Tests for `unflatten.model.checkpoint_ancestry` (plan
`spec-checkpoint-ancestry-c6`, task 1).

Fixtures are synthetic dicts: a fake pinned-hashes-derived `base_model`, a
component lock built from `tests.model.test_manifests.valid_exact_text_s1_lock`
with its `base_model_weights` artifact_sha256 aligned to the fake
`weights_set_sha256`, a minimal S0 report dict, and a minimal release
manifest dict. Checkpoint "files" are any bytes written under `tmp_path` --
no torch needed anywhere in this module.
"""

from __future__ import annotations

import copy
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.model.test_m1_schemas import valid_checkpoint_ancestry
from tests.model.test_manifests import valid_exact_text_s1_lock
from unflatten.model import checkpoint_ancestry as ca
from unflatten.model.checkpoint_ancestry import CheckpointAncestryInvalid

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64

_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def _decision(decision_id: str = "DEC-010") -> dict[str, Any]:
    return {"decision_id": decision_id, "path": f"docs/{decision_id}.md", "sha256": SHA_C}


def _release_manifest(*, release_root_hash: str = SHA_B) -> dict[str, Any]:
    return {"release_root_hash": release_root_hash}


def _s0_report(
    *,
    legacy_value: bool | None = False,
    release_root_hash: str = SHA_B,
    data_roots: tuple[str, ...] = ("/data/roots/a",),
) -> dict[str, Any]:
    return {
        "inputs": {"release_root_hash": release_root_hash},
        "legacy_ancestry": {
            "value": legacy_value,
            "evidence": {"data_roots": list(data_roots)},
        },
    }


def _lock(*, weights_sha256: str = SHA_A) -> dict[str, Any]:
    lock = valid_exact_text_s1_lock()
    for entry in lock["components"]:
        if entry["kind"] == "base_model_weights":
            entry["artifact_sha256"] = weights_sha256
    return lock


def _base_model(*, revision: str = _REVISION, weights_set_sha256: str = SHA_A) -> dict[str, Any]:
    return {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": revision,
        "weights": [{"path": "model-00001.safetensors", "sha256": SHA_A}],
        "weights_set_sha256": weights_set_sha256,
        "processor": [{"path": "preprocessor_config.json", "sha256": SHA_B}],
        "processor_set_sha256": SHA_B,
        "pinned_hashes": {"path": "pinned_hashes.json", "sha256": SHA_D},
    }


def _build(
    tmp_path: Path,
    *,
    base_model: dict[str, Any] | None = None,
    component_lock: dict[str, Any] | None = None,
    s0_report: dict[str, Any] | None = None,
    release_manifest: dict[str, Any] | None = None,
    base_model_decision: dict[str, Any] | None = ...,
    dataset_decision: dict[str, Any] | None = ...,
    distribution_decision: dict[str, Any] | None = None,
    expected_revision: str = _REVISION,
    parent: dict[str, Any] | None = None,
    also_exposed: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    s0_report_path = tmp_path / "s0_report.json"
    s0_report_path.write_text(json.dumps(s0_report if s0_report is not None else _s0_report()))
    return ca.build_base_ancestry(
        base_model=base_model if base_model is not None else _base_model(),
        component_lock=component_lock if component_lock is not None else _lock(),
        s0_report=s0_report if s0_report is not None else _s0_report(),
        s0_report_path=s0_report_path,
        release_manifest=release_manifest if release_manifest is not None else _release_manifest(),
        release_name="release-v11",
        store_name="m2a-v6",
        base_model_decision=(
            _decision("DEC-010") if base_model_decision is ... else base_model_decision
        ),
        dataset_decision=_decision("DEC-006") if dataset_decision is ... else dataset_decision,
        expected_revision=expected_revision,
        distribution_decision=distribution_decision,
        parent=parent,
        also_exposed=also_exposed,
    )


# ---------------------------------------------------------------------------
# Torch-free surface
# ---------------------------------------------------------------------------


class TestNoModelSurface:
    def test_module_import_leaves_torch_out_of_sys_modules(self):
        # Run in a fresh subprocess: other test modules in this session may
        # already have imported torch, which would make an in-process
        # sys.modules check depend on test order rather than on what this
        # module itself imports.
        script = (
            "import sys; import unflatten.model.checkpoint_ancestry; "
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_module_defines_no_torch_source_reference(self):
        source = inspect.getsource(ca)
        assert "import torch" not in source
        assert "nn.Module" not in source


# ---------------------------------------------------------------------------
# derive_rights_status / derive_privacy_status
# ---------------------------------------------------------------------------


class TestDeriveStatuses:
    def test_derive_rights_status_all_decisions_present_returns_approved(self, tmp_path):
        doc = _build(tmp_path)
        assert ca.derive_rights_status(doc) == "approved"
        assert doc["checkpoint_rights_status"] == "approved"

    def test_derive_rights_status_missing_base_model_decision_returns_unresolved(self, tmp_path):
        doc = _build(tmp_path, base_model_decision=None)
        assert ca.derive_rights_status(doc) == "unresolved"
        assert doc["checkpoint_rights_status"] == "unresolved"

    def test_derive_privacy_status_missing_exposure_decision_returns_unresolved(self, tmp_path):
        doc = _build(tmp_path, dataset_decision=None)
        assert ca.derive_privacy_status(doc) == "unresolved"
        assert doc["checkpoint_privacy_status"] == "unresolved"

    def test_derive_privacy_status_all_decisions_present_returns_approved(self, tmp_path):
        doc = _build(tmp_path)
        assert ca.derive_privacy_status(doc) == "approved"

    def test_derive_privacy_status_no_exposures_returns_unresolved(self, tmp_path):
        doc = _build(tmp_path)
        doc["dataset_exposures"] = []
        assert ca.derive_privacy_status(doc) == "unresolved"

    def test_derive_rights_status_no_exposures_returns_unresolved(self, tmp_path):
        doc = _build(tmp_path)
        doc["dataset_exposures"] = []
        assert ca.derive_rights_status(doc) == "unresolved"


# ---------------------------------------------------------------------------
# build_base_ancestry refusals
# ---------------------------------------------------------------------------


class TestBuildBaseAncestryRefusals:
    def test_build_base_ancestry_s0_legacy_ancestry_true_refuses(self, tmp_path):
        with pytest.raises(CheckpointAncestryInvalid):
            _build(tmp_path, s0_report=_s0_report(legacy_value=True))

    def test_build_base_ancestry_s0_legacy_ancestry_null_refuses(self, tmp_path):
        with pytest.raises(CheckpointAncestryInvalid):
            _build(tmp_path, s0_report=_s0_report(legacy_value=None))

    def test_build_base_ancestry_release_root_hash_mismatch_refuses(self, tmp_path):
        with pytest.raises(CheckpointAncestryInvalid):
            _build(
                tmp_path,
                s0_report=_s0_report(release_root_hash=SHA_C),
                release_manifest=_release_manifest(release_root_hash=SHA_B),
            )

    def test_build_base_ancestry_lock_weights_hash_mismatch_refuses(self, tmp_path):
        with pytest.raises(CheckpointAncestryInvalid):
            _build(tmp_path, component_lock=_lock(weights_sha256=SHA_C))

    def test_build_base_ancestry_revision_mismatch_refuses(self, tmp_path):
        with pytest.raises(CheckpointAncestryInvalid):
            _build(tmp_path, expected_revision="0" * 40)


# ---------------------------------------------------------------------------
# build_base_ancestry / distribution_allowed (DEC-011)
# ---------------------------------------------------------------------------


class TestDistributionAllowed:
    def test_build_base_ancestry_no_distribution_decision_derives_false(self, tmp_path):
        doc = _build(tmp_path)
        assert doc["distribution_decision"] is None
        assert doc["distribution_allowed"] is False

    def test_build_base_ancestry_distribution_decision_present_derives_true(self, tmp_path):
        doc = _build(tmp_path, distribution_decision=_decision("DEC-011"))
        assert doc["checkpoint_rights_status"] == "approved"
        assert doc["checkpoint_privacy_status"] == "approved"
        assert doc["distribution_allowed"] is True

    def test_build_base_ancestry_distribution_decision_present_but_rights_unresolved_derives_false(
        self, tmp_path
    ):
        doc = _build(tmp_path, base_model_decision=None, distribution_decision=_decision("DEC-011"))
        assert doc["distribution_allowed"] is False


# ---------------------------------------------------------------------------
# ancestry_projection / compute_ancestry_hash
# ---------------------------------------------------------------------------


class TestAncestryProjection:
    def test_ancestry_projection_removes_ancestry_hash_key(self, tmp_path):
        doc = _build(tmp_path)
        projection = ca.ancestry_projection(doc)
        assert "ancestry_hash" not in projection

    def test_ancestry_projection_missing_ancestry_hash_key_refuses(self):
        with pytest.raises(CheckpointAncestryInvalid):
            ca.ancestry_projection({"document_kind": "base"})

    def test_ancestry_projection_removes_checkpoint_sha_and_id_when_checkpoint_present(
        self, tmp_path
    ):
        base = _build(tmp_path)
        checkpoint_doc = ca.extend_for_checkpoint(
            base, step=0, run_id=None, s1_run_manifest_hash=None
        )
        checkpoint_doc["checkpoint"]["checkpoint_sha256"] = SHA_A
        checkpoint_doc["checkpoint"]["checkpoint_id"] = "ckpt_" + "a" * 64
        projection = ca.ancestry_projection(checkpoint_doc)
        assert "checkpoint_sha256" not in projection["checkpoint"]
        assert "checkpoint_id" not in projection["checkpoint"]

    def test_compute_ancestry_hash_stable_across_checkpoint_sha256_value(self, tmp_path):
        base = _build(tmp_path)
        checkpoint_doc = ca.extend_for_checkpoint(
            base, step=0, run_id=None, s1_run_manifest_hash=None
        )
        with_one_sha = copy.deepcopy(checkpoint_doc)
        with_one_sha["checkpoint"]["checkpoint_sha256"] = SHA_A
        with_one_sha["checkpoint"]["checkpoint_id"] = "ckpt_" + "a" * 64
        with_other_sha = copy.deepcopy(checkpoint_doc)
        with_other_sha["checkpoint"]["checkpoint_sha256"] = SHA_C
        with_other_sha["checkpoint"]["checkpoint_id"] = "ckpt_" + "c" * 64
        assert ca.compute_ancestry_hash(with_one_sha) == ca.compute_ancestry_hash(with_other_sha)


# ---------------------------------------------------------------------------
# compute_checkpoint_id
# ---------------------------------------------------------------------------


class TestComputeCheckpointId:
    def test_compute_checkpoint_id_matches_make_id_domain(self):
        from unflatten.model.hashing import make_id

        expected = make_id(
            "ckpt",
            "rfc005.checkpoint-id/v1",
            {"ancestry_hash": SHA_A, "checkpoint_sha256": SHA_B},
        )
        assert ca.compute_checkpoint_id(SHA_A, SHA_B) == expected

    def test_compute_checkpoint_id_differs_for_different_checkpoint_sha256(self):
        one = ca.compute_checkpoint_id(SHA_A, SHA_B)
        other = ca.compute_checkpoint_id(SHA_A, SHA_C)
        assert one != other


# ---------------------------------------------------------------------------
# extend_for_checkpoint
# ---------------------------------------------------------------------------


class TestExtendForCheckpoint:
    def test_extend_for_checkpoint_mixed_nullness_of_run_id_and_manifest_hash_refuses(
        self, tmp_path
    ):
        base = _build(tmp_path)
        with pytest.raises(CheckpointAncestryInvalid):
            ca.extend_for_checkpoint(
                base, step=0, run_id="s1run_" + "a" * 64, s1_run_manifest_hash=None
            )
        with pytest.raises(CheckpointAncestryInvalid):
            ca.extend_for_checkpoint(base, step=0, run_id=None, s1_run_manifest_hash=SHA_A)

    def test_extend_for_checkpoint_both_null_accepted(self, tmp_path):
        base = _build(tmp_path)
        doc = ca.extend_for_checkpoint(base, step=0, run_id=None, s1_run_manifest_hash=None)
        assert doc["document_kind"] == "checkpoint"
        assert doc["checkpoint"]["base_ancestry_hash"] == base["ancestry_hash"]

    def test_extend_for_checkpoint_base_document_mutated_refuses(self, tmp_path):
        base = _build(tmp_path)
        base["checkpoint_rights_status"] = "prohibited"
        with pytest.raises(CheckpointAncestryInvalid):
            ca.extend_for_checkpoint(base, step=0, run_id=None, s1_run_manifest_hash=None)


# ---------------------------------------------------------------------------
# finalize_checkpoint_document / write_sidecar / verify_sidecar
# ---------------------------------------------------------------------------


def _finalized_and_written(tmp_path: Path, **build_kwargs: Any) -> tuple[dict, Path, dict]:
    base = _build(tmp_path, **build_kwargs)
    extended = ca.extend_for_checkpoint(
        base, step=100, run_id="s1run_" + "e" * 64, s1_run_manifest_hash=SHA_D
    )
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_bytes(
        b"fake checkpoint bytes, embeds " + extended["ancestry_hash"].encode()
    )
    finalized = ca.finalize_checkpoint_document(extended, checkpoint_path)
    ca.write_sidecar(finalized, checkpoint_path)
    return base, checkpoint_path, finalized


class TestVerifySidecar:
    def test_verify_sidecar_full_roundtrip_succeeds(self, tmp_path):
        base, checkpoint_path, finalized = _finalized_and_written(tmp_path)
        verified = ca.verify_sidecar(
            checkpoint_path,
            expect_base_ancestry_hash=base["ancestry_hash"],
            expect_s1_run_manifest_hash=SHA_D,
            embedded_ancestry_hash=finalized["ancestry_hash"],
        )
        assert verified["checkpoint"]["checkpoint_id"] == finalized["checkpoint"]["checkpoint_id"]

    def test_verify_sidecar_tampered_checkpoint_bytes_refuses(self, tmp_path):
        base, checkpoint_path, finalized = _finalized_and_written(tmp_path)
        checkpoint_path.write_bytes(b"tampered bytes")
        with pytest.raises(CheckpointAncestryInvalid):
            ca.verify_sidecar(
                checkpoint_path,
                expect_base_ancestry_hash=base["ancestry_hash"],
                expect_s1_run_manifest_hash=SHA_D,
                embedded_ancestry_hash=finalized["ancestry_hash"],
            )

    def test_verify_sidecar_corrupt_sidecar_json_refuses(self, tmp_path):
        base, checkpoint_path, finalized = _finalized_and_written(tmp_path)
        sidecar = ca.sidecar_path(checkpoint_path)
        sidecar.write_bytes(b"not valid json {{{")
        with pytest.raises(CheckpointAncestryInvalid):
            ca.verify_sidecar(
                checkpoint_path,
                expect_base_ancestry_hash=base["ancestry_hash"],
                expect_s1_run_manifest_hash=SHA_D,
                embedded_ancestry_hash=finalized["ancestry_hash"],
            )

    def test_verify_sidecar_missing_sidecar_file_refuses(self, tmp_path):
        base, checkpoint_path, finalized = _finalized_and_written(tmp_path)
        ca.sidecar_path(checkpoint_path).unlink()
        with pytest.raises(CheckpointAncestryInvalid):
            ca.verify_sidecar(
                checkpoint_path,
                expect_base_ancestry_hash=base["ancestry_hash"],
                expect_s1_run_manifest_hash=SHA_D,
                embedded_ancestry_hash=finalized["ancestry_hash"],
            )

    def test_verify_sidecar_tampered_sidecar_field_refuses(self, tmp_path):
        base, checkpoint_path, finalized = _finalized_and_written(tmp_path)
        sidecar = ca.sidecar_path(checkpoint_path)
        doc = json.loads(sidecar.read_text())
        doc["checkpoint"]["step"] = 999
        sidecar.write_text(json.dumps(doc))
        with pytest.raises(CheckpointAncestryInvalid):
            ca.verify_sidecar(
                checkpoint_path,
                expect_base_ancestry_hash=base["ancestry_hash"],
                expect_s1_run_manifest_hash=SHA_D,
                embedded_ancestry_hash=finalized["ancestry_hash"],
            )

    def test_verify_sidecar_embedded_hash_mismatch_refuses(self, tmp_path):
        base, checkpoint_path, finalized = _finalized_and_written(tmp_path)
        with pytest.raises(CheckpointAncestryInvalid):
            ca.verify_sidecar(
                checkpoint_path,
                expect_base_ancestry_hash=base["ancestry_hash"],
                expect_s1_run_manifest_hash=SHA_D,
                embedded_ancestry_hash=SHA_A,
            )

    def test_verify_sidecar_base_ancestry_hash_mismatch_refuses(self, tmp_path):
        base, checkpoint_path, finalized = _finalized_and_written(tmp_path)
        with pytest.raises(CheckpointAncestryInvalid):
            ca.verify_sidecar(
                checkpoint_path,
                expect_base_ancestry_hash=SHA_A,
                expect_s1_run_manifest_hash=SHA_D,
                embedded_ancestry_hash=finalized["ancestry_hash"],
            )

    def test_verify_sidecar_manifest_hash_mismatch_refuses(self, tmp_path):
        base, checkpoint_path, finalized = _finalized_and_written(tmp_path)
        with pytest.raises(CheckpointAncestryInvalid):
            ca.verify_sidecar(
                checkpoint_path,
                expect_base_ancestry_hash=base["ancestry_hash"],
                expect_s1_run_manifest_hash=SHA_A,
                embedded_ancestry_hash=finalized["ancestry_hash"],
            )

    def test_verify_sidecar_distribution_true_without_decision_refuses(self, tmp_path):
        base, checkpoint_path, finalized = _finalized_and_written(tmp_path)
        sidecar = ca.sidecar_path(checkpoint_path)
        doc = json.loads(sidecar.read_text())
        assert doc["distribution_decision"] is None
        doc["distribution_allowed"] = True
        doc["ancestry_hash"] = ca.compute_ancestry_hash(doc)
        sidecar.write_text(json.dumps(doc))
        checkpoint_path.write_bytes(
            b"fake checkpoint bytes, embeds " + doc["ancestry_hash"].encode()
        )
        doc["checkpoint"]["checkpoint_sha256"] = ca.file_sha256(checkpoint_path)
        doc["checkpoint"]["checkpoint_id"] = ca.compute_checkpoint_id(
            doc["ancestry_hash"], doc["checkpoint"]["checkpoint_sha256"]
        )
        sidecar.write_text(json.dumps(doc))
        with pytest.raises(CheckpointAncestryInvalid):
            ca.verify_sidecar(
                checkpoint_path,
                expect_base_ancestry_hash=base["ancestry_hash"],
                expect_s1_run_manifest_hash=SHA_D,
                embedded_ancestry_hash=doc["ancestry_hash"],
            )


# ---------------------------------------------------------------------------
# Warm start / parent lineage (schema 1.1.0, DEC-028 D1, task T5a)
# ---------------------------------------------------------------------------


def _parent_sidecar(tmp_path: Path, *, release_root_hash: str = SHA_B) -> dict[str, Any]:
    """A finalized `document_kind: "checkpoint"` document to warm-start
    from -- same shape `_finalized_and_written` produces, with its
    `dataset_exposures` release_root_hash controllable for the DEC-006
    subset test."""
    _, _, finalized = _finalized_and_written(
        tmp_path / "parent",
        s0_report=_s0_report(release_root_hash=release_root_hash),
        release_manifest=_release_manifest(release_root_hash=release_root_hash),
    )
    return finalized


def _companion(
    *, run_class: str = "discovery", s1_run_manifest_hash: str = SHA_D
) -> dict[str, Any]:
    """A minimal parent companion S1-format run manifest -- only the two
    fields `_parent_entry` reads."""
    return {"run_class": run_class, "s1_run_manifest_hash": s1_run_manifest_hash}


class TestWarmStartParent:
    def test_build_base_ancestry_warm_start_records_parent(self, tmp_path):
        (tmp_path / "parent").mkdir()
        parent_sidecar = _parent_sidecar(tmp_path)
        doc = _build(
            tmp_path,
            parent={"sidecar": parent_sidecar, "companion": _companion()},
        )
        assert doc["initialization_kind"] == "warm_start"
        assert doc["parents"] == [
            {
                "checkpoint_id": parent_sidecar["checkpoint"]["checkpoint_id"],
                "ancestry_hash": parent_sidecar["ancestry_hash"],
                "checkpoint_sha256": parent_sidecar["checkpoint"]["checkpoint_sha256"],
                "run_id": parent_sidecar["checkpoint"]["run_id"],
                "s1_run_manifest_hash": parent_sidecar["checkpoint"]["s1_run_manifest_hash"],
                "step": parent_sidecar["checkpoint"]["step"],
                "run_class": "discovery",
            }
        ]
        ca.verify_base(doc)

    def test_build_base_ancestry_parent_exposure_not_subset_refused(self, tmp_path):
        (tmp_path / "parent").mkdir()
        parent_sidecar = _parent_sidecar(tmp_path, release_root_hash=SHA_C)
        with pytest.raises(CheckpointAncestryInvalid):
            _build(
                tmp_path,
                release_manifest=_release_manifest(release_root_hash=SHA_B),
                s0_report=_s0_report(release_root_hash=SHA_B),
                parent={"sidecar": parent_sidecar, "companion": _companion()},
            )

    def test_build_base_ancestry_parent_exposure_declared_via_also_exposed_accepted(
        self, tmp_path
    ):
        """S3 on release-v14-r4 warm-starts from S2 v13: the parent's v13
        exposure is declared explicitly, so DEC-006 holds without inheriting."""
        (tmp_path / "parent").mkdir()
        parent_sidecar = _parent_sidecar(tmp_path, release_root_hash=SHA_C)
        doc = _build(
            tmp_path,
            release_manifest=_release_manifest(release_root_hash=SHA_B),
            s0_report=_s0_report(release_root_hash=SHA_B),
            parent={"sidecar": parent_sidecar, "companion": _companion()},
            also_exposed=(
                {"release_name": "release-v13", "release_root_hash": SHA_C,
                 "store_name": "m2a-v8"},
            ),
        )
        assert [e["release_root_hash"] for e in doc["dataset_exposures"]] == [SHA_B, SHA_C]
        assert doc["dataset_exposures"][1]["decision"] == doc["dataset_exposures"][0]["decision"]
        ca.verify_base(doc)

    def test_build_base_ancestry_parent_companion_hash_mismatch_refused(self, tmp_path):
        (tmp_path / "parent").mkdir()
        parent_sidecar = _parent_sidecar(tmp_path)
        with pytest.raises(CheckpointAncestryInvalid):
            _build(
                tmp_path,
                parent={
                    "sidecar": parent_sidecar,
                    "companion": _companion(s1_run_manifest_hash=SHA_A),
                },
            )

    def test_clean_pretrained_with_parents_refused_by_schema(self, tmp_path):
        (tmp_path / "parent").mkdir()
        parent_sidecar = _parent_sidecar(tmp_path)
        doc = _build(tmp_path)
        assert doc["initialization_kind"] == "clean_pretrained"
        doc["parents"] = [
            {
                "checkpoint_id": parent_sidecar["checkpoint"]["checkpoint_id"],
                "ancestry_hash": parent_sidecar["ancestry_hash"],
                "checkpoint_sha256": parent_sidecar["checkpoint"]["checkpoint_sha256"],
                "run_id": parent_sidecar["checkpoint"]["run_id"],
                "s1_run_manifest_hash": parent_sidecar["checkpoint"]["s1_run_manifest_hash"],
                "step": parent_sidecar["checkpoint"]["step"],
                "run_class": "discovery",
            }
        ]
        doc["ancestry_hash"] = ca.compute_ancestry_hash(doc)
        with pytest.raises(CheckpointAncestryInvalid):
            ca.verify_base(doc)

    def test_warm_start_without_parents_refused_by_schema(self, tmp_path):
        (tmp_path / "parent").mkdir()
        parent_sidecar = _parent_sidecar(tmp_path)
        doc = _build(
            tmp_path,
            parent={"sidecar": parent_sidecar, "companion": _companion()},
        )
        assert doc["parents"]
        doc["parents"] = []
        doc["ancestry_hash"] = ca.compute_ancestry_hash(doc)
        with pytest.raises(CheckpointAncestryInvalid):
            ca.verify_base(doc)

    def test_existing_1_0_0_sidecar_still_validates(self):
        instance = valid_checkpoint_ancestry()
        assert instance["checkpoint_ancestry_schema_version"] == "1.0.0"
        assert instance["initialization_kind"] == "clean_pretrained"
        assert instance["parents"] == []
        ca._validate_schema(instance, what="pinned 1.0.0 fixture")

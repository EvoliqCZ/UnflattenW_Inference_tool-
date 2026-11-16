"""RFC004 §38–§39 manifest schema + validator + release-identity tests."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

import unflatten.model.manifests as manifests_module
from unflatten.model.lock_kinds import NEW_LOCK_KINDS
from unflatten.model.manifests import (
    COMPONENT_KIND,
    EXACT_TEXT_S1_COMPONENT_KIND,
    EXACT_TEXT_S1_REQUIRED_COMPONENT_KIND,
    PROVISIONAL_COMPONENT_KIND,
    model_release_id,
    validate_architecture_manifest,
    validate_component_lock,
    validate_exact_text_s1_architecture_manifest,
    validate_exact_text_s1_component_lock,
)
from unflatten.model.reasoning.lock import REASONING_COMPONENT_KINDS

_SCHEMA_DIR = Path(manifests_module.__file__).parent / "schemas"


def _component_lock_schema() -> dict:
    return json.loads((_SCHEMA_DIR / "component_lock.schema.json").read_text())


def valid_manifest() -> dict:
    return {
        "architecture_manifest_schema_version": "0.1.0",
        "architecture_id": "unflatten-word-moe-v0",
        "architecture_version": "0.1.0",
        "compatible_ast_versions": ["0.1.0"],
        "profiles": ["word-core-v0"],
        "d_model": 1024,
        "components": {
            "backbone_adapter_contract": "0.1.0",
            "input_contract": "0.1.0",
            "visual_memory_contract": "0.1.0",
            "structural_grammar": "0.1.0",
            "prediction_ir": "0.1.0",
            "result_envelope": "0.1.0",
        },
        "expert_adapter": {
            "required": True,
            "supported_modes": ["dense", "sparse_moe"],
        },
        "optional_slots": {
            "visual_compressor": True,
            "dflash_speculator": True,
            "mtp_speculator": True,
        },
    }


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def valid_lock() -> dict:
    return {
        "model_component_lock_schema_version": "0.1.0",
        "model_name": "unflatten-word-model",
        "model_version": "0.1.0",
        "architecture_id": "unflatten-word-moe-v0",
        "ast_version": "0.1.0",
        "profiles": ["word-core-v0"],
        "components": [
            {"kind": "architecture_manifest", "version": "0.1.0", "artifact_sha256": SHA_A},
            {"kind": "cbor_profile", "version": "0.1.0", "semantic_hash": SHA_B},
            {"kind": "grammar", "version": "0.1.0", "artifact_sha256": SHA_A},
        ],
    }


class TestComponentKinds:
    def test_closed_kind_set(self):
        assert COMPONENT_KIND == frozenset(
            {
                "rfc001_schema_binding",
                "rfc001_hash_projection",
                "cbor_profile",
                "io_schemas",
                "architecture_manifest",
                "backbone_adapter",
                "base_model_weights",
                "image_preprocessor",
                "document_context_encoder",
                "structural_tokenizer",
                "text_tokenizer",
                "grammar",
                "expert_adapter",
                "typed_head_registry",
                "default_policy",
                "decode_config",
                "error_code_registry",
                "runtime_lock",
                "renderer_binding",
            }
        )


class TestArchitectureManifest:
    def test_valid(self):
        assert validate_architecture_manifest(valid_manifest()) == []

    def test_missing_field(self):
        doc = valid_manifest()
        del doc["d_model"]
        assert validate_architecture_manifest(doc) != []

    def test_unknown_field_rejected_closed_schema(self):
        doc = valid_manifest()
        doc["extra"] = 1
        assert validate_architecture_manifest(doc) != []

    def test_d_model_must_be_positive_int(self):
        doc = valid_manifest()
        doc["d_model"] = 0
        assert validate_architecture_manifest(doc) != []
        doc["d_model"] = 1024.0
        assert validate_architecture_manifest(doc) != []

    def test_supported_modes_closed(self):
        doc = valid_manifest()
        doc["expert_adapter"]["supported_modes"] = ["dense", "quantum"]
        assert validate_architecture_manifest(doc) != []

    def test_supported_modes_nonempty_unique(self):
        doc = valid_manifest()
        doc["expert_adapter"]["supported_modes"] = []
        assert validate_architecture_manifest(doc) != []
        doc["expert_adapter"]["supported_modes"] = ["dense", "dense"]
        assert validate_architecture_manifest(doc) != []

    def test_optional_slots_closed(self):
        doc = valid_manifest()
        doc["optional_slots"]["warp_drive"] = True
        assert validate_architecture_manifest(doc) != []

    def test_optional_slots_subset_allowed(self):
        doc = valid_manifest()
        doc["optional_slots"] = {"visual_compressor": False}
        assert validate_architecture_manifest(doc) == []

    def test_components_contract_map_closed(self):
        doc = valid_manifest()
        doc["components"]["mystery_contract"] = "0.1.0"
        assert validate_architecture_manifest(doc) != []

    def test_bad_semver(self):
        doc = valid_manifest()
        doc["architecture_version"] = "1.0"
        assert validate_architecture_manifest(doc) != []

    def test_non_dict_input(self):
        assert validate_architecture_manifest([]) != []


class TestComponentLock:
    def test_valid(self):
        assert validate_component_lock(valid_lock()) == []

    def test_unknown_kind_rejected(self):
        doc = valid_lock()
        doc["components"][0]["kind"] = "dflash"
        assert validate_component_lock(doc) != []

    def test_entry_requires_exactly_one_hash(self):
        doc = valid_lock()
        entry = doc["components"][0]
        entry["semantic_hash"] = SHA_B  # both present
        assert validate_component_lock(doc) != []
        doc = valid_lock()
        del doc["components"][0]["artifact_sha256"]  # neither present
        assert validate_component_lock(doc) != []

    def test_duplicate_kind_rejected(self):
        doc = valid_lock()
        doc["components"].append(copy.deepcopy(doc["components"][0]))
        issues = validate_component_lock(doc)
        assert any(i.code == "MODEL_LOCK_DUPLICATE_COMPONENT_KIND" for i in issues)

    def test_components_must_be_bytewise_sorted_by_kind(self):
        doc = valid_lock()
        doc["components"].reverse()
        issues = validate_component_lock(doc)
        assert any(i.code == "MODEL_LOCK_COMPONENTS_UNSORTED" for i in issues)

    def test_cbor_profile_component_mandatory(self):
        # §3.1: the profile implementation/test-vector hash is a mandatory
        # Component Lock field.
        doc = valid_lock()
        doc["components"] = [c for c in doc["components"] if c["kind"] != "cbor_profile"]
        issues = validate_component_lock(doc)
        assert any(i.code == "MODEL_LOCK_CBOR_PROFILE_MISSING" for i in issues)

    def test_bad_hash_shape(self):
        doc = valid_lock()
        doc["components"][0]["artifact_sha256"] = "sha256:XYZ"
        assert validate_component_lock(doc) != []

    def test_non_dict_input(self):
        assert validate_component_lock(None) != []


class TestModelReleaseId:
    def test_shape_and_determinism(self):
        rid = model_release_id(valid_lock())
        assert re.fullmatch(r"model_[0-9a-f]{64}", rid)
        assert rid == model_release_id(valid_lock())

    def test_golden_vector(self):
        assert model_release_id(valid_lock()) == (
            "model_db64d9f2316c0f51cb468b5735cd1f2e4debe343ba58780764cac794826a0be0"
        )

    def test_release_channel_changes_identity(self):
        assert model_release_id(valid_lock()) != model_release_id(
            valid_lock(), release_channel="prod"
        )

    def test_lock_content_changes_identity(self):
        doc = valid_lock()
        doc["components"][0]["artifact_sha256"] = SHA_B
        assert model_release_id(doc) != model_release_id(valid_lock())

    def test_invalid_lock_rejected(self):
        doc = valid_lock()
        del doc["profiles"]
        with pytest.raises(ValueError):
            model_release_id(doc)

    def test_release_channel_closed_enum(self):
        for channel in ("dev", "rc", "prod"):
            rid = model_release_id(valid_lock(), release_channel=channel)
            assert re.fullmatch(r"model_[0-9a-f]{64}", rid)
        for bad in ("production", "", "DEV", "dev ", "canary"):
            with pytest.raises(ValueError):
                model_release_id(valid_lock(), release_channel=bad)

    def test_supported_profiles_order_independent_projection(self):
        # The release-id projection sorts supported_profiles so identity
        # never depends on declaration order. (The component-lock hash is a
        # raw byte hash of the lock and still covers its literal order, so
        # the full id cannot be pinned order-independent end to end.)
        from unflatten.model.manifests import _release_projection

        a = valid_lock()
        a["profiles"] = ["word-core-v0", "a-profile"]
        b = valid_lock()
        b["profiles"] = ["a-profile", "word-core-v0"]
        lock_hash = SHA_A
        assert _release_projection(a, lock_hash, "dev") == _release_projection(b, lock_hash, "dev")
        assert _release_projection(a, lock_hash, "dev")["supported_profiles"] == [
            "a-profile",
            "word-core-v0",
        ]


def valid_exact_text_s1_lock() -> dict:
    """A0 :func:`valid_lock` extended with the exact-text S1 bindings
    (design §4.8 table, widened 2026-09-05 per RFC005 §14.4's exact seven:
    "base weights, processor, tokenizer, vocabulary, grammar, decoder
    additions, decode config"). ``grammar`` and ``architecture_manifest``
    are already present in the A0 base and are not duplicated here."""
    doc = valid_lock()
    doc["components"] = sorted(
        doc["components"]
        + [
            {"kind": "base_model_weights", "version": "0.1.0", "artifact_sha256": SHA_A},
            {"kind": "image_preprocessor", "version": "0.1.0", "artifact_sha256": SHA_A},
            {"kind": "structural_tokenizer", "version": "0.1.0", "artifact_sha256": SHA_A},
            {"kind": "text_tokenizer", "version": "0.1.0", "artifact_sha256": SHA_A},
            {"kind": "composite_vocabulary", "version": "0.1.0", "semantic_hash": SHA_B},
            {"kind": "document_context_encoder", "version": "0.1.0", "semantic_hash": SHA_B},
            {"kind": "expert_adapter", "version": "0.1.0", "semantic_hash": SHA_B},
            {"kind": "model_capability_profile", "version": "0.1.0", "semantic_hash": SHA_B},
            {"kind": "decode_config", "version": "0.1.0", "semantic_hash": SHA_B},
            {"kind": "runtime_lock", "version": "0.1.0", "artifact_sha256": SHA_A},
        ],
        key=lambda c: c["kind"].encode("utf-8"),
    )
    return doc


def valid_exact_text_s1_manifest() -> dict:
    doc = valid_manifest()
    doc["components"]["composite_decoder_contract"] = "0.1.0"
    doc["optional_slots"] = {
        "visual_compressor": False,
        "dflash_speculator": False,
        "mtp_speculator": False,
    }
    return doc


class TestExactTextS1ComponentKind:
    def test_additive_to_a0_disjoint(self):
        # MINOR extension: the two new kinds are additive, not a rename/removal
        # of any A0 kind.
        assert EXACT_TEXT_S1_COMPONENT_KIND == frozenset(
            {"composite_vocabulary", "model_capability_profile"}
        )
        assert EXACT_TEXT_S1_COMPONENT_KIND.isdisjoint(COMPONENT_KIND)

    def test_schema_enum_is_a0_union_exact_text_s1(self):
        schema = _component_lock_schema()
        enum = set(schema["$defs"]["component"]["properties"]["kind"]["enum"])
        # RFC004-ERRATA-005 E3/W1-7 adds reasoning_module/reasoning_gate/
        # pre_encoder_compressor/visual_compressor as a further MINOR union
        # member; no producer emits them yet (unflatten.model.reasoning.lock).
        # W2-3 adds six more §39.2 kinds the same way (unflatten.model.lock_kinds).
        assert (
            enum
            == COMPONENT_KIND
            | EXACT_TEXT_S1_COMPONENT_KIND
            | REASONING_COMPONENT_KINDS
            | NEW_LOCK_KINDS
        )

    def test_model_capability_profile_provisional_composite_vocabulary_is_not(self):
        # Pins the provisional split (design §4.8, RR-4): a later edit that
        # folds composite_vocabulary into "provisional" or drops
        # model_capability_profile's provisional status must fail this.
        assert PROVISIONAL_COMPONENT_KIND == frozenset({"model_capability_profile"})
        assert "composite_vocabulary" not in PROVISIONAL_COMPONENT_KIND
        assert PROVISIONAL_COMPONENT_KIND <= EXACT_TEXT_S1_COMPONENT_KIND

    def test_schema_comment_pins_provisional_status(self):
        comment = _component_lock_schema()["$comment"]
        assert "'model_capability_profile'" in comment
        assert "NOT named in §39.2" in comment
        assert "'composite_vocabulary' is NOT provisional" in comment

    def test_error_code_registry_kind_untouched_by_m1(self):
        assert "error_code_registry" in COMPONENT_KIND
        assert "error_code_registry" not in EXACT_TEXT_S1_COMPONENT_KIND
        assert "error_code_registry" not in EXACT_TEXT_S1_REQUIRED_COMPONENT_KIND


class TestExactTextS1ComponentLock:
    def test_valid(self):
        assert validate_exact_text_s1_component_lock(valid_exact_text_s1_lock()) == []

    def test_old_a0_only_lock_still_passes_basic_but_fails_exact_text_s1(self):
        doc = valid_lock()
        assert validate_component_lock(doc) == []  # backward compatibility
        issues = validate_exact_text_s1_component_lock(doc)
        assert issues != []
        assert all(i.code == "MODEL_LOCK_EXACT_TEXT_S1_BINDING_MISSING" for i in issues)

    @pytest.mark.parametrize("kind", sorted(EXACT_TEXT_S1_REQUIRED_COMPONENT_KIND))
    def test_omitting_a_required_binding_is_rejected(self, kind):
        doc = valid_exact_text_s1_lock()
        doc["components"] = [c for c in doc["components"] if c["kind"] != kind]
        issues = validate_exact_text_s1_component_lock(doc)
        assert any(
            i.code == "MODEL_LOCK_EXACT_TEXT_S1_BINDING_MISSING" and kind in i.message
            for i in issues
        )

    def test_mutating_a_required_kind_breaks_the_binding(self):
        doc = valid_exact_text_s1_lock()
        for entry in doc["components"]:
            if entry["kind"] == "composite_vocabulary":
                entry["kind"] = "renderer_binding"
        doc["components"].sort(key=lambda c: c["kind"].encode("utf-8"))
        assert validate_component_lock(doc) == []  # still a structurally valid A0 lock
        issues = validate_exact_text_s1_component_lock(doc)
        assert any(
            i.code == "MODEL_LOCK_EXACT_TEXT_S1_BINDING_MISSING"
            and "composite_vocabulary" in i.message
            for i in issues
        )

    def test_schema_invalid_lock_short_circuits(self):
        doc = valid_exact_text_s1_lock()
        del doc["profiles"]
        issues = validate_exact_text_s1_component_lock(doc)
        assert any(i.code == "MODEL_SCHEMA_INVALID" for i in issues)


class TestExactTextS1ArchitectureManifest:
    def test_valid(self):
        assert validate_exact_text_s1_architecture_manifest(valid_exact_text_s1_manifest()) == []

    def test_old_manifest_without_contract_still_valid_basic_but_fails_exact_text_s1(self):
        doc = valid_manifest()
        assert validate_architecture_manifest(doc) == []  # backward compatibility
        issues = validate_exact_text_s1_architecture_manifest(doc)
        assert any(i.code == "MODEL_MANIFEST_EXACT_TEXT_CONTRACT_MISSING" for i in issues)

    def test_dense_not_declared_rejected(self):
        doc = valid_exact_text_s1_manifest()
        doc["expert_adapter"]["supported_modes"] = ["sparse_moe"]
        issues = validate_exact_text_s1_architecture_manifest(doc)
        assert any(i.code == "MODEL_MANIFEST_EXACT_TEXT_DENSE_NOT_DECLARED" for i in issues)

    def test_enabled_optional_slot_rejected_as_not_applicable_violation(self):
        doc = valid_exact_text_s1_manifest()
        doc["optional_slots"]["visual_compressor"] = True
        issues = validate_exact_text_s1_architecture_manifest(doc)
        assert any(
            i.code == "MODEL_MANIFEST_EXACT_TEXT_OPTIONAL_SLOT_NOT_APPLICABLE" for i in issues
        )

    def test_schema_invalid_manifest_short_circuits(self):
        doc = valid_exact_text_s1_manifest()
        del doc["d_model"]
        issues = validate_exact_text_s1_architecture_manifest(doc)
        assert any(i.code == "MODEL_SCHEMA_INVALID" for i in issues)

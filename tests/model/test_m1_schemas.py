"""M1 exact-text closed artifact schemas (design §4.1-4.8, task 0).

Mirrors the validator pattern in ``tests/model/test_manifests.py``: load each
schema, confirm it is itself a valid JSON Schema, confirm a minimal valid
instance validates, and confirm the ``additionalProperties: false`` / missing
``required`` field negative controls actually reject.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema.protocols
import jsonschema.validators
import pytest

_SCHEMA_DIR = Path(__file__).parent.parent.parent / "src" / "unflatten" / "model" / "schemas"

SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def load_validator(name: str) -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{name}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def is_valid(name: str, instance: dict) -> bool:
    return load_validator(name).is_valid(instance)


# ---------------------------------------------------------------------------
# Minimal valid instances, one per artifact (design §4.1-4.8)
# ---------------------------------------------------------------------------


def valid_qwen_text_allowlist() -> dict:
    return {
        "qwen_text_allowlist_schema_version": "1.0.0",
        "tokenizer_artifact_set_hash": SHA_A,
        "native_vocab_size": 2,
        "allowed_native_ids": [0],
        "denied": [{"native_id": 1, "reasons": ["special_token"]}],
        "classification_policy_version": "1.0.0",
        "control_text_check": "applied",
    }


def valid_composite_vocabulary() -> dict:
    return {
        "composite_vocabulary_schema_version": "1.0.0",
        "structural": {
            "token_map_version": "0.3.0",
            "token_map_hash": SHA_A,
            "count": 71,
            "packed_start": 0,
        },
        "qwen": {
            "tokenizer_artifact_set_hash": SHA_A,
            "text_allowlist_hash": SHA_B,
            "native_vocab_size": 2,
            "packed_start": 71,
        },
        "byte_fallback": {
            "policy_version": "1.0.0",
            "encoding": "UTF-8",
            "scalar_normalization": "NFC",
            "frame": ["BYTE_BEGIN", "BYTE_VALUE*", "BYTE_END"],
            "byte_controls": [["BYTE_BEGIN", 0], ["BYTE_END", 1]],
            "byte_value_count": 256,
            "packed_start": 75,
        },
        "packing_policy_version": "1.0.0",
        "text_encoding_policy_version": "1.0.0",
        "greedy_tie_break": "lowest_packed_id",
    }


def valid_model_capability_profile() -> dict:
    return {
        "model_capability_profile_schema_version": "1.0.0",
        "profile_id": "dense-v0-s1-exact-text-v1",
        "ast_profile": "word-visual-v1.1",
        "scope": "single_page_primary_s1",
        "allowed_node_types": [
            "cell",
            "document",
            "list",
            "list_item",
            "paragraph",
            "row",
            "run",
            "section",
            "table",
        ],
        "allowed_leaf_controls": ["line_break", "tab"],
        "capability_excluded": [
            "column_break",
            "endnote",
            "field",
            "floating_image",
            "footer",
            "footnote",
            "header",
            "hyperlink",
            "inline_image",
            "note_ref",
            "page_break",
            "text_frame",
        ],
        "exact_text": True,
        "typed_properties": "not_applicable",
        "expert_adapter_mode": "dense",
        "sparse_moe": "not_applicable",
        "visual_compressor": "not_applicable",
        "speculators": "not_applicable",
    }


def valid_composite_grammar() -> dict:
    return {
        "composite_grammar_schema_version": "1.2.0",
        "base_structural_controller_hash": SHA_A,
        "structural_token_map_hash": SHA_B,
        "composite_vocabulary_hash": SHA_C,
        "capability_profile_hash": SHA_A,
        "mode_transition_policy_version": "1.0.0",
        "byte_frame_policy_version": "1.0.0",
        "empty_paragraph_policy_version": "2.0.0",
        "max_consecutive_empty_paragraphs": 4,
        "max_consecutive_tabs": 3,
        "budget_policy_hash": SHA_B,
    }


def valid_training_target_materializer() -> dict:
    return {
        "training_target_materializer_schema_version": "1.0.0",
        "materializer_version": "1.0.0",
        "implementation_hash": SHA_A,
        "loss_unit_schema_hash": SHA_B,
        "boundary_projection_policy_version": "1.0.0",
        "composite_vocabulary_hash": SHA_C,
        "composite_grammar_hash": SHA_A,
        "capability_profile_hash": SHA_B,
        "supported_loss_unit_kinds": ["rfc003_supervised"],
        "supported_loss_families": ["L_structure", "L_text"],
    }


def valid_runtime_limits_policy() -> dict:
    return {
        "runtime_limits_policy_schema_version": "1.0.0",
        "policy_version": "1.0.0",
        "text_encoding_budget_ops": 1,
        "max_scalars_per_sample": 1,
        "max_materialized_tokens_per_sample": 1,
        "max_tokenizer_probes_per_range": 1,
        "max_structural_events_per_sample": 1,
        "max_controller_steps_per_sample": 1,
        "max_byte_frames_per_sample": 1,
        "max_cache_row_bytes": 1,
        "max_cache_shard_bytes": 1,
        "max_cache_rows_per_shard": 1,
        "max_json_nesting_depth": 1,
        "max_cbor_nesting_depth": 1,
        "max_tokenizer_file_bytes": 1,
    }


def valid_exact_text_roundtrip_corpus() -> dict:
    return {
        "exact_text_roundtrip_corpus_schema_version": "1.0.0",
        "corpus_version": "1.0.0",
        "case_count": 15,
        "cases_hash": SHA_A,
        "category_counts": {
            "ascii": 1,
            "spaces": 1,
            "nfc_accents": 1,
            "non_nfc_reject": 1,
            "bidi": 1,
            "cjk": 1,
            "emoji_zwj": 1,
            "variation_selector": 1,
            "supplementary": 1,
            "allowed_controls": 1,
            "invalid_surrogate": 1,
            "byte_len_1": 1,
            "byte_len_2": 1,
            "byte_len_3": 1,
            "byte_len_4": 1,
        },
    }


def valid_text_range_locator() -> dict:
    return {"text_span_id": "span-0", "scalar_start": 0, "scalar_end": 3}


def valid_event_range_locator() -> dict:
    return {"event_start": 0, "event_end": 3}


def valid_ownership_proof(*, owner_scope: str = "fragment_local") -> dict:
    fragment_local = owner_scope == "fragment_local"
    return {
        "document_bundle_id": "bundle-0",
        "target_bundle_hash": SHA_A,
        "fragment_id": "fragment-0" if fragment_local else None,
        "fragment_schema_version": "1.0.0" if fragment_local else None,
        "fragment_sequence_index": 0 if fragment_local else None,
        "node_id": "node-0",
        "structure_owner": True,
        "typed_target_owner": None,
        "text_range_owner": True,
        "page_evidence": None,
        "owned_char_ranges": [[0, 3]],
        "ownership_projection_hash": SHA_B,
    }


def valid_effective_mask() -> dict:
    return {
        "mask_projection_hash": SHA_A,
        "known_mask": True,
        "observable_mask": True,
        "supervision_mask": True,
        "ownership_enabled": True,
        "stage_enabled": True,
        "effective": True,
        "reason_codes": [],
    }


def valid_loss_unit(*, owner_scope: str = "fragment_local", locator: dict | None = None) -> dict:
    return {
        "loss_unit_schema_version": "1.0.0",
        "loss_unit_id": "loss-unit-0",
        "loss_unit_kind": "rfc003_supervised",
        "output_locator": "decoder_step_0",
        "attachment": "decoder_step",
        "loss_family": "L_text",
        "stage_enabled": True,
        "owner_scope": owner_scope,
        "owner_fragment_id": "fragment-0" if owner_scope == "fragment_local" else None,
        "locator": locator if locator is not None else valid_text_range_locator(),
        "authority_target_hash": SHA_A,
        "projection_hash": SHA_B,
        "mask_projection_hash": SHA_C,
        "ownership_proof": valid_ownership_proof(owner_scope=owner_scope),
        "effective_mask": valid_effective_mask(),
        "composite_vocabulary_hash": SHA_A,
        "composite_grammar_hash": SHA_B,
        "capability_profile_hash": SHA_C,
        "head_registry_hash": None,
    }


def valid_checkpoint_ancestry(*, document_kind: str = "base") -> dict:
    doc = {
        "checkpoint_ancestry_schema_version": "1.0.0",
        "document_kind": document_kind,
        "base_model": {
            "model_id": "Qwen/Qwen3-VL-4B-Instruct",
            "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "weights": [{"path": "model-00001.safetensors", "sha256": SHA_A}],
            "weights_set_sha256": SHA_A,
            "processor": [{"path": "preprocessor_config.json", "sha256": SHA_B}],
            "processor_set_sha256": SHA_B,
            "pinned_hashes": {"path": "pinned_hashes.json", "sha256": SHA_C},
        },
        "component_lock_hash": SHA_A,
        "initialization_kind": "clean_pretrained",
        "parents": [],
        "adapters": [],
        "merge_history": [],
        "legacy_ancestry": {
            "value": False,
            "evidence": {
                "s0_report": {"path": "s0_report.json", "sha256": SHA_B},
                "s0_report_value": False,
                "release_root_hash": SHA_A,
                "audited_roots": ["/data/roots/a"],
            },
        },
        "dataset_exposures": [
            {
                "release_name": "release-v11",
                "release_root_hash": SHA_B,
                "store_name": "m2a-v6",
                "decision": {"decision_id": "DEC-006", "path": "docs/DEC-006.md", "sha256": SHA_C},
            }
        ],
        "exposure_ledger": {
            "kind": "not_produced",
            "reason": "no release checkpoint distribution root exists in v1",
        },
        "transitive_exposure_manifest": {
            "kind": "not_produced",
            "reason": "no transitive exposure manifest exists in v1",
        },
        "unknown_project_exposure": {
            "value": False,
            "reason": {
                "code": "clean_pretrained_audited",
                "detail": "S0 legacy_ancestry audit found no checkpoints",
            },
        },
        "base_model_decision": {
            "decision_id": "DEC-010",
            "path": "docs/DEC-010.md",
            "sha256": SHA_A,
        },
        "checkpoint_rights_status": "approved",
        "checkpoint_privacy_status": "approved",
        "distribution_allowed": False,
        "distribution_decision": None,
        "checkpoint": None,
        "ancestry_hash": SHA_C,
    }
    if document_kind == "checkpoint":
        doc["checkpoint"] = {
            "checkpoint_id": "ckpt_" + "d" * 64,
            "checkpoint_sha256": SHA_A,
            "step": 100,
            "run_id": "s1run_" + "e" * 64,
            "s1_run_manifest_hash": SHA_B,
            "base_ancestry_hash": SHA_C,
        }
    return doc


def valid_candidate_track() -> dict:
    return {
        "candidate_track_schema_version": "1.0.0",
        "candidate_track_id": "track_" + "a" * 64,
        "admission_track_name": "dense_contract_canary",
        "backbone_profile": "qwen3-vl-word-adapter-v0",
        "ast_profile": "word-visual-v1.1",
        "capability_profile_id": "dense-v0-s1-exact-text-v1",
        "capability_profile_hash": SHA_A,
        "architecture_manifest_sha256": SHA_B,
        "architecture_id": "unflatten-word-moe-v0",
        "base_checkpoint": {"kind": "base_weights_only", "weights_set_sha256": SHA_A},
        "initialization_kind": "clean_pretrained",
        "adaptation_kind": "adapter_plus_new_heads_no_lora",
        "legacy_ancestry": False,
        "ancestry_hash": SHA_C,
        "checkpoint_rights_status": "approved",
        "checkpoint_privacy_status": "approved",
        "transitive_exposure_manifest": {
            "kind": "not_produced",
            "reason": "no transitive exposure manifest exists in v1",
        },
        "adapter_policy": {
            "kind": "registry",
            "trainable_group_registry_hash": SHA_B,
            "lora": "disabled",
            "rfc_ref": "RFC005 §14.3",
        },
        "model_component_lock_hash": SHA_A,
    }


#: Gate names, in gate-number order 1..15 -- must equal the s1_runner.py
#: GateResult name literals (cross-checked, without importing torch, by
#: tests/model/test_stage_gate_report.py).
STAGE_GATE_NAMES = [
    "forward_backward_finite",
    "frozen_groups_zero_grad",
    "trainable_groups_finite_nonzero_grad",
    "masks_zero_loss_contribution",
    "missing_target_no_default",
    "checkpoint_roundtrip",
    "grammar_constrained_valid_tree",
    "forbidden_metadata_no_leak",
    "loader_lineage_disjoint",
    "text_roundtrip",
    "namespace_no_collision",
    "padding_equivalence",
    "free_running_no_silent_truncation",
    "component_lock",
    "optimizer_membership_complete",
]


def valid_stage_gate_policy() -> dict:
    return {
        "stage_gate_policy_schema_version": "1.0.0",
        "policy_version": "1.0.0",
        "stage": "S1",
        "gates": [
            {
                "gate": number,
                "name": name,
                "kind": "contract",
                "missing_evidence_action": "blocked",
            }
            for number, name in enumerate(STAGE_GATE_NAMES, start=1)
        ],
        "quality": {
            "status": "not_applicable",
            "reason": (
                "S1/S2 gates are contract-only (owner R5 2026-09-07, D4); gate 12's "
                "atol/rtol tolerances are runner-declared per run and reported in "
                "that run's own gate evidence, never a value this frozen policy pins"
            ),
        },
        "minimum_independent_lineages": {
            "status": "not_applicable",
            "reason": "no minimum lineage count is set at S1/S2 (D4)",
        },
        "missing_evidence_action_policy_wide": "blocked",
    }


def _valid_gate_result(number: int, name: str, *, evidence: dict | None = None) -> dict:
    return {
        "number": number,
        "name": name,
        "status": "pass",
        "detail": "ok",
        "evidence": evidence if evidence is not None else {},
    }


def valid_stage_gate_report() -> dict:
    gates = [
        _valid_gate_result(number, name)
        for number, name in enumerate(STAGE_GATE_NAMES, start=1)
    ]
    gates[8]["evidence"] = {
        "overlap_count": 0,
        "cohort_lineage_root_count": 3,
        "requested_document_count": 3,
        "resolved_document_count": 3,
    }
    gates[11]["evidence"] = {"negative_control_status": "fail"}
    return {
        "stage_gate_report_schema_version": "1.0.0",
        "stage": "S1",
        "run_stage": "S2",
        "notes": [
            "DEC-009: this S2 run is evaluated under the frozen S1 loss policy "
            "(stage_policy.json stage_id=S1)"
        ],
        "run_class": "discovery",
        "runner_report_sha256": SHA_A,
        "bindings": {
            "policy_hash": SHA_A,
            "run_id": "s1run_" + "a" * 64,
            "s1_run_manifest_hash": SHA_B,
            "component_lock_hash": SHA_A,
            "candidate_track_id": "track_" + "a" * 64,
            "admission_track_name": "dense_contract_canary",
            "ancestry_hash": SHA_C,
            "exact_text_cache_identity_hash": SHA_B,
            "dataset_release_root_hash": SHA_A,
            "loss_policy_stage_id": "S1",
            "stage_policy_hash": SHA_B,
            "ancestry_rights_status": "approved",
            "ancestry_privacy_status": "approved",
            "exposure_ledger": {
                "kind": "not_produced",
                "reason": "no transitive exposure manifest exists in v1",
            },
            "s0_report": {
                "path": "runs/s0/s0_report.json",
                "sha256": SHA_A,
                "report_schema_version": "0.1.0",
                "admitted_tracks": ["dense_contract_canary"],
            },
            "roundtrip_corpus": {
                "descriptor": {
                    "exact_text_roundtrip_corpus_schema_version": "1.0.0",
                    "corpus_version": "1.0.0",
                    "case_count": 32,
                    "cases_hash": SHA_A,
                    "category_counts": {},
                },
                "check": {
                    "passed": True,
                    "checked_case_count": 32,
                    "codec_mode": "qwen",
                    "byte_len_route": "byte_fallback",
                    "budget_ops": 10000,
                    "tokenizer_artifact_set_hash": SHA_A,
                    "composite_vocabulary_hash": SHA_B,
                },
            },
        },
        "evaluated_candidate": {"kind": "base_weights_only", "weights_set_sha256": SHA_A},
        "gates": gates,
        "gate_12_negative_control": {"ran": True, "status": "fail"},
        "lineage_evidence": {
            "status": "evidenced",
            "overlap_count": 0,
            "cohort_lineage_root_count": 3,
            "requested_document_count": 3,
            "resolved_document_count": 3,
        },
        "gates_all_pass": True,
        "promotion_blockers": ["evaluated_candidate_not_run_output"],
        "missing_evidence_outcomes": [],
        "promoted": False,
        "stage_gate_report_hash": SHA_C,
    }


ARTIFACTS = {
    "qwen_text_allowlist": valid_qwen_text_allowlist,
    "composite_vocabulary": valid_composite_vocabulary,
    "model_capability_profile": valid_model_capability_profile,
    "composite_grammar": valid_composite_grammar,
    "training_target_materializer": valid_training_target_materializer,
    "runtime_limits_policy": valid_runtime_limits_policy,
    "exact_text_roundtrip_corpus": valid_exact_text_roundtrip_corpus,
    "loss_unit": valid_loss_unit,
    "checkpoint_ancestry": valid_checkpoint_ancestry,
    "candidate_track": valid_candidate_track,
    "stage_gate_policy": valid_stage_gate_policy,
    "stage_gate_report": valid_stage_gate_report,
}


# ---------------------------------------------------------------------------
# Generic coverage: schema validity, minimal instance, additionalProperties,
# required-field absence -- for all eight artifacts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ARTIFACTS))
def test_schema_file_exists_and_is_valid_json_schema(name: str):
    validator = load_validator(name)
    assert validator.schema["additionalProperties"] is False


@pytest.mark.parametrize("name", sorted(ARTIFACTS))
def test_minimal_valid_instance_validates(name: str):
    assert is_valid(name, ARTIFACTS[name]())


@pytest.mark.parametrize("name", sorted(ARTIFACTS))
def test_unknown_extra_property_rejected(name: str):
    instance = ARTIFACTS[name]()
    instance["unexpected_extra_field"] = "nope"
    assert not is_valid(name, instance)


@pytest.mark.parametrize("name", sorted(ARTIFACTS))
def test_each_required_field_absence_rejected(name: str):
    validator = load_validator(name)
    for field in validator.schema["required"]:
        instance = ARTIFACTS[name]()
        del instance[field]
        assert not validator.is_valid(instance), f"{name}: missing {field!r} should be rejected"


# ---------------------------------------------------------------------------
# loss_unit.schema.json -- the delicate one (task 0 explicit asks)
# ---------------------------------------------------------------------------


class TestLossUnitOwnerScope:
    @pytest.mark.parametrize(
        "owner_scope", ["global_target", "fragment_local", "example_local", "document_scope"]
    )
    def test_all_four_owner_scope_values_accepted(self, owner_scope: str):
        instance = valid_loss_unit(owner_scope=owner_scope)
        assert is_valid("loss_unit", instance)

    def test_owner_scope_outside_closed_enum_rejected(self):
        instance = valid_loss_unit()
        instance["owner_scope"] = "not_a_real_scope"
        assert not is_valid("loss_unit", instance)


class TestLossUnitHeadRegistryHash:
    def test_head_registry_hash_null_accepted(self):
        instance = valid_loss_unit()
        instance["head_registry_hash"] = None
        assert is_valid("loss_unit", instance)

    def test_head_registry_hash_valid_sha256_accepted(self):
        instance = valid_loss_unit()
        instance["head_registry_hash"] = SHA_A
        assert is_valid("loss_unit", instance)

    def test_head_registry_hash_malformed_rejected(self):
        instance = valid_loss_unit()
        instance["head_registry_hash"] = "not-a-sha256"
        assert not is_valid("loss_unit", instance)


class TestLossUnitOwnershipProof:
    def test_missing_ownership_proof_rejected(self):
        instance = valid_loss_unit()
        del instance["ownership_proof"]
        assert not is_valid("loss_unit", instance)

    def test_ownership_proof_with_extra_field_rejected(self):
        instance = valid_loss_unit()
        instance["ownership_proof"] = {**instance["ownership_proof"], "extra": 1}
        assert not is_valid("loss_unit", instance)

    def test_ownership_proof_missing_required_field_rejected(self):
        instance = valid_loss_unit()
        proof = copy.deepcopy(instance["ownership_proof"])
        del proof["ownership_projection_hash"]
        instance["ownership_proof"] = proof
        assert not is_valid("loss_unit", instance)

    def test_global_target_owner_scope_allows_null_fragment_id(self):
        instance = valid_loss_unit(owner_scope="global_target")
        assert instance["ownership_proof"]["fragment_id"] is None
        assert is_valid("loss_unit", instance)


class TestLossUnitLocator:
    def test_text_range_locator_accepted(self):
        instance = valid_loss_unit(locator=valid_text_range_locator())
        assert is_valid("loss_unit", instance)

    def test_event_range_locator_accepted(self):
        instance = valid_loss_unit(locator=valid_event_range_locator())
        assert is_valid("loss_unit", instance)

    def test_locator_mixing_both_shapes_rejected(self):
        instance = valid_loss_unit(
            locator={**valid_text_range_locator(), **valid_event_range_locator()}
        )
        assert not is_valid("loss_unit", instance)

    def test_locator_of_unknown_shape_rejected(self):
        instance = valid_loss_unit(locator={"region_id": "r0"})
        assert not is_valid("loss_unit", instance)


class TestLossUnitKindAndAttachment:
    def test_loss_unit_kind_outside_closed_set_rejected(self):
        instance = valid_loss_unit()
        instance["loss_unit_kind"] = "sealed_derived"
        assert not is_valid("loss_unit", instance)

    def test_attachment_outside_closed_set_rejected(self):
        instance = valid_loss_unit()
        instance["attachment"] = "something_else"
        assert not is_valid("loss_unit", instance)


# ---------------------------------------------------------------------------
# checkpoint_ancestry.schema.json -- 6 negative controls (plan task 0)
# ---------------------------------------------------------------------------


class TestCheckpointAncestryNegatives:
    def test_malformed_distribution_decision_rejected(self):
        instance = valid_checkpoint_ancestry()
        instance["distribution_decision"] = {"decision_id": "not-a-decision-id"}
        assert not is_valid("checkpoint_ancestry", instance)

    def test_base_document_with_non_null_checkpoint_rejected(self):
        instance = valid_checkpoint_ancestry(document_kind="base")
        instance["checkpoint"] = {
            "checkpoint_id": "ckpt_" + "d" * 64,
            "checkpoint_sha256": SHA_A,
            "step": 0,
            "run_id": None,
            "s1_run_manifest_hash": None,
            "base_ancestry_hash": SHA_C,
        }
        assert not is_valid("checkpoint_ancestry", instance)

    def test_checkpoint_document_with_null_checkpoint_rejected(self):
        instance = valid_checkpoint_ancestry(document_kind="checkpoint")
        instance["checkpoint"] = None
        assert not is_valid("checkpoint_ancestry", instance)

    def test_exposure_ledger_bare_array_rejected(self):
        instance = valid_checkpoint_ancestry()
        instance["exposure_ledger"] = []
        assert not is_valid("checkpoint_ancestry", instance)

    def test_bare_hex_without_sha256_prefix_rejected(self):
        instance = valid_checkpoint_ancestry()
        instance["component_lock_hash"] = "a" * 64
        assert not is_valid("checkpoint_ancestry", instance)

    def test_non_empty_parents_rejected(self):
        instance = valid_checkpoint_ancestry()
        instance["parents"] = ["some-parent-id"]
        assert not is_valid("checkpoint_ancestry", instance)

    def test_checkpoint_ancestry_schema_empty_dataset_exposures_rejected(self):
        instance = valid_checkpoint_ancestry()
        instance["dataset_exposures"] = []
        assert not is_valid("checkpoint_ancestry", instance)


# ---------------------------------------------------------------------------
# candidate_track.schema.json -- 4 negative controls (plan task 0)
# ---------------------------------------------------------------------------


class TestCandidateTrackNegatives:
    def test_transitive_exposure_manifest_bare_array_rejected(self):
        instance = valid_candidate_track()
        instance["transitive_exposure_manifest"] = []
        assert not is_valid("candidate_track", instance)

    def test_adaptation_kind_outside_closed_enum_rejected(self):
        instance = valid_candidate_track()
        instance["adaptation_kind"] = "lora_plus_new_heads"
        assert not is_valid("candidate_track", instance)

    def test_candidate_track_id_bare_hex_without_prefix_rejected(self):
        instance = valid_candidate_track()
        instance["candidate_track_id"] = "a" * 64
        assert not is_valid("candidate_track", instance)

    def test_base_checkpoint_mixed_branch_fields_rejected(self):
        instance = valid_candidate_track()
        instance["base_checkpoint"] = {
            "kind": "base_weights_only",
            "weights_set_sha256": SHA_A,
            "checkpoint_id": "ckpt_" + "d" * 64,
        }
        assert not is_valid("candidate_track", instance)


# ---------------------------------------------------------------------------
# stage_gate_policy.schema.json / stage_gate_report.schema.json negatives
# (plan spec-stage-gate-report-c5, task 0)
# ---------------------------------------------------------------------------


class TestStageGatePolicyNegatives:
    def test_gates_not_exactly_fifteen_rejected(self):
        instance = valid_stage_gate_policy()
        instance["gates"] = instance["gates"][:14]
        assert not is_valid("stage_gate_policy", instance)

    def test_quality_applies_branch_requires_metrics_not_reason(self):
        instance = valid_stage_gate_policy()
        instance["quality"] = {"status": "applies", "reason": "not allowed here"}
        assert not is_valid("stage_gate_policy", instance)

    def test_missing_evidence_action_policy_wide_outside_blocked_rejected(self):
        instance = valid_stage_gate_policy()
        instance["missing_evidence_action_policy_wide"] = "allowed"
        assert not is_valid("stage_gate_policy", instance)


class TestStageGateReportNegatives:
    def test_gates_not_exactly_fifteen_rejected(self):
        instance = valid_stage_gate_report()
        instance["gates"] = instance["gates"][:14]
        assert not is_valid("stage_gate_report", instance)

    def test_promotion_blocker_outside_closed_enum_rejected(self):
        instance = valid_stage_gate_report()
        instance["promotion_blockers"] = ["some_new_reason"]
        assert not is_valid("stage_gate_report", instance)

    def test_evaluated_candidate_mixed_branch_fields_rejected(self):
        instance = valid_stage_gate_report()
        instance["evaluated_candidate"] = {
            "kind": "base_weights_only",
            "weights_set_sha256": SHA_A,
            "checkpoint_id": "ckpt_" + "d" * 64,
        }
        assert not is_valid("stage_gate_report", instance)

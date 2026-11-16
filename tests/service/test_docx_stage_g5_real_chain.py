"""G5 (spec-pdf-to-word-service follow-up, T5 defect fix): the diagnostic
Word chain over a REAL `section > paragraph > run` page carrying a null
typed value (`RFC004-ERRATA-003` E1: `paragraph:/style/borders` decoded
`None` by the `borders_is_null` head) -- nothing in this module is mocked
past the composite vocabulary/tokenizer construction, which is itself the
real `composite_vocab` machinery over a byte-only fixture tokenizer
(structural-only page: no `TEXT_BEGIN`/`TEXT_END` span, so the tokenizer is
never called -- `tests.assembly.test_hypothesis`'s own pattern).

Proves the IR-schema-null defect fix end to end: `seal_diagnostic_prediction`
(real IR schema validation + real hash) -> `docx_stage.build_diagnostic_docx`
with the REAL `run.process_model_page` (real `ast_builder`/`promotion`/
renderer, CPU-only) -> a real DOCX zip with `word/document.xml`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from tests.assembly.test_hypothesis import TOKENIZER, VOCABULARY

from unflatten.dataset.targets.properties import pointer_registry
from unflatten.model import composite_vocab as cv
from unflatten.model import events_vocab as ev
from unflatten.model import typed_projection as tp
from unflatten.model.node_lifecycle import NodeState
from unflatten.model.prediction_ir import RawDocumentHypothesis
from unflatten.model.prediction_ir import prediction_ir_hash as compute_prediction_ir_hash
from unflatten.model.typed_decode import TypedPrediction, _inactive_arm_leaves
from unflatten.model.typed_head_config import (
    BORDER_OBJECT_POINTERS,
    CARDINALITY_HEADS,
    COLOR_POINTERS,
    NULLABLE_COLOR_POINTERS,
)
from unflatten.model.typed_property_pointer_map import enum_domain
from unflatten.service import docx_stage as ds
from unflatten.service.model_runtime import seal_diagnostic_prediction

_BORDER_CONTAINERS: dict[str, list[str]] = {"section": [], "paragraph": [], "run": []}
for _nt, _ptr in BORDER_OBJECT_POINTERS:
    if _nt in _BORDER_CONTAINERS:
        _BORDER_CONTAINERS[_nt].append(_ptr)

_COLOR_POINTER_SET = dict.fromkeys(COLOR_POINTERS)

#: The only two overrides needed beyond a generic `0`/`False`/domain-first
#: value: the semantic validator refuses a paragraph with both first-line
#: indent and hanging indent non-zero at once (RFC004-ERRATA-003 podnet #3).
_INTEGER_OVERRIDES: dict[tuple[str, str], int] = {
    ("paragraph", "/layout/indent/first_line_mpt"): 0,
    ("paragraph", "/layout/indent/hanging_mpt"): 0,
    ("section", "/columns/count"): 1,
}
_BOOLEAN_OVERRIDES: dict[tuple[str, str], bool] = {
    ("section", "/columns/equal_width"): True,
}


def _typed_values_for(node_type: str) -> dict[str, object]:
    """Every registered pointer for `node_type`, given a schema-legal value --
    the same rules `typed_decode.decode_typed` applies at decode time
    (palette null class, border-object null suppressing its 12 leaves, one
    net first-line/hanging offset, inactive union-arm leaves dropped), just
    computed directly instead of through a live head/anchor."""
    values: dict[str, object] = {}
    for pointer, value_type in sorted(pointer_registry().get(node_type, {}).items()):
        if "*" in pointer:
            continue  # star-templated slot: covered by the cardinality head below.
        if node_type == "section" and pointer.startswith(("/line_numbering", "/page_borders")):
            continue  # E1: dropped, no presence head (RFC004-ERRATA-003).
        containers = _BORDER_CONTAINERS.get(node_type, [])
        if any(pointer == c or pointer.startswith(c + "/") for c in containers):
            if pointer in containers:
                values[pointer] = None  # the defect under test: JSON null.
            continue
        if (node_type, pointer) in _COLOR_POINTER_SET:
            nullable = (node_type, pointer) in NULLABLE_COLOR_POINTERS
            values[pointer] = None if nullable else "auto"
            continue
        if value_type == "string":
            continue  # deferred, no head (E2 default fills it, e.g. font_family).
        if value_type == "boolean":
            values[pointer] = _BOOLEAN_OVERRIDES.get((node_type, pointer), False)
        elif value_type == "enum":
            values[pointer] = enum_domain(node_type, pointer)[0]
        elif value_type == "integer":
            base = _INTEGER_OVERRIDES.get((node_type, pointer), 1000)
            values[pointer] = tp.clamp_integer(node_type, pointer, base)
        else:  # pragma: no cover - pointer_registry() has no other value type
            raise AssertionError((node_type, pointer, value_type))
    for cardinality_node_type, container, _cap_attribute in CARDINALITY_HEADS:
        if cardinality_node_type == node_type:
            values[container] = 0
    for pointer in _inactive_arm_leaves(node_type, values):
        values.pop(pointer, None)
    return values


def _section_paragraph_run_hypothesis() -> RawDocumentHypothesis:
    """A real, structural-only token stream (`section > paragraph > run`,
    the empty run) packed through the real composite vocabulary -- and its
    typed predictions, including `paragraph:/style/borders = None`."""
    items = (
        "FRAGMENT_BEGIN", "SECTION_OPEN", "PARAGRAPH_OPEN", "RUN_OPEN",
        "RUN_CLOSE", "PARAGRAPH_CLOSE", "SECTION_CLOSE", "END_FRAGMENT",
    )
    token_ids = tuple(
        VOCABULARY.pack(cv.StructuralRef(local_id=ev.TOKEN_IDS[name])) for name in items
    )
    node_states = (
        NodeState("n0", "opened_node", "section", None, open_event_index=1),
        NodeState("n1", "opened_node", "paragraph", "n0", open_event_index=2),
        NodeState("n2", "opened_node", "run", "n1", open_event_index=3),
    )
    typed_predictions: list[TypedPrediction] = []
    for node_type, local_node_id in (("section", "n0"), ("paragraph", "n1"), ("run", "n2")):
        for pointer, value in _typed_values_for(node_type).items():
            typed_predictions.append(
                TypedPrediction(local_node_id, node_type, pointer, "expert_adapter.dense", value)
            )
    return RawDocumentHypothesis(
        accepted_tokens=token_ids,
        node_states=node_states,
        typed_predictions=tuple(typed_predictions),
        terminated=True,
    )


def test_typed_values_include_a_null_paragraph_borders() -> None:
    """Sanity on the fixture itself: it exercises the exact null this defect
    is about, not merely a page that happens to build."""
    values = _typed_values_for("paragraph")
    assert values["/style/borders"] is None
    assert not any(p.startswith("/style/borders/") for p in values)


def test_g5_real_diagnostic_docx_chain_publishes_the_font_substitution(tmp_path: Path) -> None:
    """G5: seal through the real service sealing path (real IR schema
    validation + real hash over a null typed value), then run the REAL
    `run.process_model_page` (real ast_builder/promotion/renderer)."""
    raw = _section_paragraph_run_hypothesis()

    # Real IR schema validation + real hash over a payload carrying JSON null.
    payload = seal_diagnostic_prediction(raw)
    border_prediction = next(
        p
        for p in payload["document_hypotheses"][0]["fragments"][0]["typed_predictions"]
        if p["pointer"] == "/style/borders"
    )
    assert border_prediction["value"] is None
    ir_hash = compute_prediction_ir_hash(payload)

    runtime = SimpleNamespace(
        eval_cache_identity={"dataset_release_root_hash": "sha256:release"},
        component_lock_hash="sha256:" + "1" * 64,
        selection_policy_hash="sha256:" + "2" * 64,
        decode_policy=SimpleNamespace(max_steps=4608, prefill_chunk_tokens=None),
        vocabulary=VOCABULARY,
        tokenizer=TOKENIZER,
    )

    result = ds.build_diagnostic_docx(
        raw,
        SimpleNamespace(visual_features=torch.zeros(1)),
        b"png-bytes",
        runtime=runtime,
        job_id="bundle_g5",
        work_dir=tmp_path,
        max_docx_bytes=10_000_000,
        expected_prediction_ir_hash=ir_hash,
    )

    assert result.produced is True
    assert result.docx_chain["ast_status"] == "strict_valid"
    # DEC-044 Dodatok C (owner 2026-09-26): the E2 Calibri default renders as
    # Carlito, a metric-compatible substitution; the Word is published and the
    # substitution is named in the chain and in the reason codes.
    assert result.docx_chain["renderer_status"] == "approximation_required"
    assert result.docx_chain["renderer_approximations"] == ["FONT_FALLBACK"]
    assert result.docx_chain["policy_counts"]["default:run:/style/font_family"] == 1
    # DEC-044 Dodatok B/D / RFC004-ERRATA-004 / DEC-077 (RFC004-ERRATA-011): every
    # service policy ran on the real page (a dict, `{}` when nothing to edit -- never
    # `None`, which is `run.py`'s "policy off"), and the profile is the E4-derived one.
    policies = {p["policy_id"]: p for p in result.docx_chain["policies"]}
    assert list(policies) == [
        "headless-defaults.v0",
        "run-typography-snap.v0",
        "image-region-snap.v0",
        "tab-stops-from-regions.v0",
        "table-geometry.v0",
        "story-kind-dedup.v0",
        "leader-fill.v1",
        "recovered-image-materialization.v0",
    ]
    assert all(p["policy_status"] == "RATIFIED" for p in policies.values())
    assert policies["headless-defaults.v0"]["counts"] == result.docx_chain["policy_counts"]
    assert policies["table-geometry.v0"]["counts"] == {}  # no table on this page
    assert policies["story-kind-dedup.v0"]["counts"] == {}  # no duplicate stories
    assert policies["leader-fill.v1"]["counts"] == {}  # no fill leader
    assert policies["recovered-image-materialization.v0"]["counts"] == {}  # no image
    assert result.docx_chain["profile"] == "word-core-v0"
    assert result.docx_bytes is not None and result.docx_bytes[:2] == b"PK"
    assert result.reason_codes == ("HEADLESS_DEFAULTS_V0_APPLIED", "DOCX_FONT_SUBSTITUTED")
    from unflatten.service.contracts import validate_contract

    validate_contract(
        {"prediction_ir_status": "sealed", "claim_scope": "diagnostic",
         "canonical_ast_produced": True, "docx_produced": True,
         "reason_codes": list(result.reason_codes), "docx_chain": result.docx_chain},
        "diagnostic_outcome",
    )


def test_g5_poisoned_renderer_is_a_chain_error_but_ir_still_published(
    tmp_path: Path, monkeypatch
) -> None:
    """The poisoned-`compile_docx` variant (G1's IR-safety rule): a real
    renderer failure makes `build_diagnostic_docx` report `DOCX_CHAIN_ERROR`,
    but the Prediction IR sealing/hashing step this test also exercises
    above never depends on the renderer at all -- `seal_diagnostic_prediction`
    succeeds (and would already have been published by the worker, design
    D4/G1) regardless of what happens to the docx chain afterwards."""
    raw = _section_paragraph_run_hypothesis()
    payload = seal_diagnostic_prediction(raw)  # IR sealing never touches the renderer.
    ir_hash = compute_prediction_ir_hash(payload)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("poisoned compile_docx")

    monkeypatch.setattr("unflatten.model.promotion.compile_docx", _boom)

    runtime = SimpleNamespace(
        eval_cache_identity={"dataset_release_root_hash": "sha256:release"},
        component_lock_hash="sha256:" + "1" * 64,
        selection_policy_hash="sha256:" + "2" * 64,
        decode_policy=SimpleNamespace(max_steps=4608, prefill_chunk_tokens=None),
        vocabulary=VOCABULARY,
        tokenizer=TOKENIZER,
    )

    result = ds.build_diagnostic_docx(
        raw,
        SimpleNamespace(visual_features=torch.zeros(1)),
        b"png-bytes",
        runtime=runtime,
        job_id="bundle_g5b",
        work_dir=tmp_path,
        max_docx_bytes=10_000_000,
        expected_prediction_ir_hash=ir_hash,
    )

    assert result.produced is False
    assert result.docx_bytes is None
    assert result.reason_codes == ("DOCX_CHAIN_ERROR",)
    # The already-sealed IR payload (what the worker would have published
    # before ever reaching the docx stage, design D4) still carries the null
    # value untouched by the poisoned renderer.
    border = next(
        p
        for p in payload["document_hypotheses"][0]["fragments"][0]["typed_predictions"]
        if p["pointer"] == "/style/borders"
    )
    assert border["value"] is None

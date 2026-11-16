"""Sealed candidate producer (RFC003-ERRATA-003 E3, DEC-047 dormant code).

The "výrobný pás" step (a) missing per `docs/plans/reasoning-data-readiness-
2026-09-28.md` §2 row b: given a loaded runtime/checkpoint, one page and its
decode policy, run the decode and seal the result into a Prediction IR
artefact plus provenance -- reusing the existing sealing machinery
(:mod:`unflatten.model.prediction_ir`, :mod:`unflatten.service.
prediction_ir_artifact`) rather than inventing a new artefact format, the
way `scripts/dspark_rehearsal.py:494-527` did when it kept only
`accepted_tokens`.

Dormant by construction (DEC-047): `run_class="promotable"` is refused
unless the caller supplies a ratification record, mirroring
`s6r_train.require_s6r_preconditions`'s fail-closed pattern. Nothing here
writes a release, a cache, or a training loop -- see
`unflatten.dataset.reasoning.__init__`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from unflatten.dataset.identity import _ANCESTRY_PRODUCER_EXPOSURES, _require_sha256
from unflatten.model.prediction_ir import (
    InferenceScope,
    RawDocumentHypothesis,
    assemble,
    prediction_ir_projection,
    seal_fragment,
)
from unflatten.service.prediction_ir_artifact import (
    PredictionIrArtifact,
    build_prediction_ir_artifact,
)

#: The three decode precisions the readiness doc requires be recorded
#: (`docs/plans/reasoning-data-readiness-2026-09-28.md` §4: fp8 locally on
#: the 4070, bf16 on prod -- comparing across them confounds precision with
#: split difference).
DECODE_PRECISIONS: frozenset[str] = frozenset({"fp8", "bf16", "int8"})

#: A candidate is only ever `discovery`/`rehearsal` (never promotable)
#: unless an explicit ratification record is supplied -- see
#: `RunClassNotPromotableError`.
RUN_CLASSES: frozenset[str] = frozenset({"discovery", "rehearsal", "promotable"})

#: The train-only role a sealed candidate carries when handed to
#: `train_val_gap` / `failure_taxonomy.report` tooling (`report.check_role`
#: refuses any role containing "test"/"locked"/"p6" -- this one contains
#: none of them, so it always passes; no closed role registry exists to
#: extend, see the readiness doc item 5 follow-up in
#: `docs/plans/reasoning-data-readiness-2026-09-28.md`).
PRODUCER_TRAIN_ROLE = "reasoning_producer_train_sample"


class CandidateProducerError(ValueError):
    """Base class for sealed-candidate-producer refusals."""


class RunClassNotPromotableError(CandidateProducerError):
    """`run_class="promotable"` without a ratification record (DEC-047:
    every trainer/producer refuses until a ratification record exists)."""


@dataclass(frozen=True, slots=True)
class CandidateProvenance:
    """Provenance a `model_candidate` Input Ancestry binding needs
    (RFC003-ERRATA-003 E3:114-124), plus the decode precision the readiness
    doc's §4 flags as load-bearing (fp8 vs bf16 confounds any train-vs-
    val_iid comparison unless both sides use the same one)."""

    checkpoint_sha256: str
    checkpoint_component_lock_hash: str
    decode_policy_hash: str
    decode_precision: Literal["fp8", "bf16", "int8"]
    producer_exposure: Literal["in_sample", "out_of_fold", "earlier_checkpoint"]
    run_class: Literal["discovery", "rehearsal", "promotable"]


@dataclass(frozen=True, slots=True)
class SealedCandidateArtefact:
    """One page's sealed candidate: a Prediction IR artefact plus the
    provenance needed to bind it as `source_kind: "model_candidate"`."""

    document_bundle_id: str
    source_page_index: int
    prediction_ir: PredictionIrArtifact
    provenance: CandidateProvenance


def produce_sealed_candidate(
    *,
    document_bundle_id: str,
    source_page_index: int,
    decode_fn: Callable[..., RawDocumentHypothesis],
    decode_kwargs: Mapping[str, Any],
    checkpoint_sha256: str,
    checkpoint_component_lock_hash: str,
    decode_policy_hash: str,
    decode_precision: str,
    producer_exposure: str,
    run_class: str,
    ratification_record: Mapping[str, Any] | None = None,
) -> SealedCandidateArtefact:
    """Run `decode_fn(**decode_kwargs)` for one page and seal its output.

    `decode_fn` is `free_running_decode.decode_free_running` in production,
    or a stub in every test -- exactly `s2_gate.decode.decode_cohort`'s
    existing `decode_fn` injection pattern, reused rather than
    reinvented. Refuses BEFORE decoding when the requested shape cannot be
    sealed at all: an unknown precision/exposure/run_class, or
    `run_class="promotable"` without `ratification_record` (DEC-047).
    """
    if decode_precision not in DECODE_PRECISIONS:
        raise CandidateProducerError(
            f"decode_precision must be one of {sorted(DECODE_PRECISIONS)}, "
            f"got {decode_precision!r}"
        )
    if producer_exposure not in _ANCESTRY_PRODUCER_EXPOSURES:
        raise CandidateProducerError(
            f"producer_exposure must be one of {sorted(_ANCESTRY_PRODUCER_EXPOSURES)}, "
            f"got {producer_exposure!r}"
        )
    if run_class not in RUN_CLASSES:
        raise CandidateProducerError(f"run_class must be one of {sorted(RUN_CLASSES)}, got {run_class!r}")
    if run_class == "promotable" and not ratification_record:
        raise RunClassNotPromotableError(
            "run_class='promotable' requires an explicit ratification record "
            "(DEC-047: every producer refuses until one exists for its stage); "
            "got none"
        )
    _require_sha256("checkpoint_sha256", checkpoint_sha256)
    _require_sha256("checkpoint_component_lock_hash", checkpoint_component_lock_hash)
    _require_sha256("decode_policy_hash", decode_policy_hash)

    raw = decode_fn(**decode_kwargs)
    fragment = seal_fragment(raw, InferenceScope())
    ir = assemble([fragment])
    payload = prediction_ir_projection(ir)
    artifact = build_prediction_ir_artifact(payload)

    provenance = CandidateProvenance(
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_component_lock_hash=checkpoint_component_lock_hash,
        decode_policy_hash=decode_policy_hash,
        decode_precision=decode_precision,  # type: ignore[arg-type]
        producer_exposure=producer_exposure,  # type: ignore[arg-type]
        run_class=run_class,  # type: ignore[arg-type]
    )
    return SealedCandidateArtefact(
        document_bundle_id=document_bundle_id,
        source_page_index=source_page_index,
        prediction_ir=artifact,
        provenance=provenance,
    )

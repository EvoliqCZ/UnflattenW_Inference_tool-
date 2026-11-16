"""S8 correctness-label extractor (RFC005 §22, RFC004 §36).

This module does not read a sealed decode artefact or a `cohort_gold`
split directly -- both are cohort-specific and out of scope here (task
instruction: no new split partitions, no dataset access from this
package, matching :mod:`unflatten.model.calibration.metrics`). Instead
the caller supplies the decode/gold records already keyed by `page_id`
plus two small accessor functions:

- `correct_fn(decode_record, gold_record) -> bool` -- the caller's
  existing correctness definition for the chosen target (e.g. page
  `ast_status == "strict_valid"`, or a typed-property exact match),
- `score_fn(decode_record) -> float | None` -- pulls the confidence the
  prediction IR already carries for that target. `None` means the score
  is genuinely absent and is recorded as such (`CorrectnessRecord.score
  is None`) -- it is never imputed to 0.0, 1.0 or any other value.

`default_confidence_score` is the one built-in `score_fn`: it reads
`confidence_report.overall_score` (or `per_axis[<axis>]`) per
`confidence-report.schema.json` (`0..1000` integer) and rescales to
`[0, 1]`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CorrectnessRecord:
    """One (score, correct) observation for `target`, keyed by `page_id`.

    `score is None` means "score_missing": the prediction IR carried no
    confidence field for this target on this page.
    """

    page_id: str
    target: str
    correct: bool
    score: float | None


class MissingGoldError(KeyError):
    """A decode record's `page_id` has no matching gold record."""


def extract_correctness_labels(
    decode_records: Mapping[str, Mapping[str, Any]],
    gold_records: Mapping[str, Mapping[str, Any]],
    target: str,
    correct_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
    score_fn: Callable[[Mapping[str, Any]], float | None],
) -> list[CorrectnessRecord]:
    """Join `decode_records` with `gold_records` by `page_id` and emit one
    :class:`CorrectnessRecord` per decode record for `target`.

    Raises :class:`MissingGoldError` if a decode record's `page_id` has no
    gold counterpart -- a silent skip would understate the denominator.
    """
    records: list[CorrectnessRecord] = []
    for page_id, decode_record in decode_records.items():
        if page_id not in gold_records:
            raise MissingGoldError(page_id)
        gold_record = gold_records[page_id]
        correct = correct_fn(decode_record, gold_record)
        score = score_fn(decode_record)
        records.append(
            CorrectnessRecord(page_id=page_id, target=target, correct=correct, score=score)
        )
    return records


def default_confidence_score(
    decode_record: Mapping[str, Any], axis: str = "overall"
) -> float | None:
    """Default `score_fn`: `confidence_report.overall_score` (or
    `per_axis[axis]`) rescaled from the schema's `0..1000` integer range
    to `[0, 1]`. Returns `None` (score_missing) if the report or the
    requested field is absent -- never imputes."""
    report = decode_record.get("confidence_report")
    if report is None:
        return None
    value = report.get("overall_score") if axis == "overall" else report.get("per_axis", {}).get(
        axis
    )
    if value is None:
        return None
    return value / 1000.0


def split_missing(
    records: Sequence[CorrectnessRecord],
) -> tuple[list[CorrectnessRecord], list[CorrectnessRecord]]:
    """`(present, missing)` -- records partitioned by `score is not None`."""
    present = [r for r in records if r.score is not None]
    missing = [r for r in records if r.score is None]
    return present, missing


def as_score_label_arrays(records: Sequence[CorrectnessRecord]) -> tuple[list[float], list[float]]:
    """`(scores, labels)` over records with a present score, in order.

    Records with `score is None` are excluded -- callers wanting to
    account for them separately should inspect :func:`split_missing`
    first.
    """
    present = [r for r in records if r.score is not None]
    scores = [float(r.score) for r in present]
    labels = [1.0 if r.correct else 0.0 for r in present]
    return scores, labels

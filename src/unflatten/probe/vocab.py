"""Classification scoring of the two ordinal probe labels.

`modal_font_size_mpt` and `modal_space_after_mpt` are scored as
regression in `unflatten.probe.run`, and squared error is the wrong cost
for them: they live on the renderer's per-property quantisation grid
(`renderer/units.py:9-11` — 500 mpt for font size, 50 mpt for lengths),
so a page's modal value is one of a handful of lattice points and
9 pt -> 11 pt is a different CLASS, not four times the error of
9 pt -> 10 pt. This module scores the same labels, the same features and
the same head as CLASSIFICATION over the values the train split shows.

It does not replace the regression arm. Both numbers are reported side
by side, because the published R²/rho figures are the comparison
baseline and a metric change that deletes its predecessor cannot be
audited.

Like `unflatten.probe.bands`, this is a differently-scored VIEW of an
existing label: nothing here enters `ProbeCacheIdentity`, no label
derivation changes, and an existing feature cache re-scores on the CPU
in seconds.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from unflatten.probe.heads import (
    ALPHAS,
    NULL_PERMUTATIONS,
    _predict_from_factor,
    fold_plan,
    ridge_factors,
    standardize,
)
from unflatten.probe.run import MIN_EXAMPLES

#: Labels this module re-scores: the two on Word's quantisation grid.
VOCAB_LABELS: tuple[str, ...] = ("modal_font_size_mpt", "modal_space_after_mpt")

#: Class index for an eval value the train split never showed. It is a
#: real class and never a drop: 0.5-0.6 % of nodes are out of vocabulary
#: on this corpus, and dropping them would score the head on a
#: population it was not asked about — the same substitution D2 rejects.
#: The head cannot predict it (no train rows carry it), so those rows
#: count as errors, which is exactly what a vocabulary head owes them.
OTHER_CLASS = -1


@dataclass(frozen=True)
class VocabOutcome:
    """One label scored as classification over its train vocabulary."""

    label: str
    kind: str
    n_train: int
    n_eval: int
    #: Distinct values observed on the TRAIN split. Fitting the
    #: vocabulary on train+eval would leak the eval support.
    n_classes: int | None
    #: Eval rows whose value is outside that vocabulary.
    oov_eval: int | None
    accuracy: float | None
    macro_f1: float | None
    #: Rate of the most frequent eval class: what a constant scores.
    majority: float | None
    #: Best accuracy the SAME procedure reaches on shuffled train labels.
    null_accuracy: float | None
    null_macro_f1: float | None
    #: `max(majority, null_accuracy)` — the bar accuracy has to clear.
    threshold: float | None
    beats_baseline: bool | None
    skipped_reason: str | None
    alpha: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_vocabulary(labels: Sequence[Any]) -> tuple[float, ...]:
    """Sorted distinct labelled values — from ONE split, never both."""
    return tuple(sorted({float(v) for v in labels if v is not None}))


def encode(
    labels: Sequence[Any], vocabulary: Sequence[float]
) -> tuple[list[int], torch.Tensor]:
    """Row indices with a label, and their class ids (`OTHER_CLASS` if OOV)."""
    positions = {value: index for index, value in enumerate(vocabulary)}
    rows = [i for i, value in enumerate(labels) if value is not None]
    classes = torch.tensor(
        [positions.get(float(labels[i]), OTHER_CLASS) for i in rows],
        dtype=torch.long,
    )
    return rows, classes


def accuracy_from_classes(
    targets: torch.Tensor, predicted: torch.Tensor
) -> float:
    """Top-1 accuracy over class ids."""
    if targets.numel() == 0:
        return float("nan")
    return float((targets == predicted).float().mean())


def macro_f1(targets: torch.Tensor, predicted: torch.Tensor) -> float:
    """Unweighted mean F1 over the classes PRESENT IN THE TARGETS.

    Present-in-targets and not present-in-predictions: a class the head
    never predicts must score 0 and drag the mean, which is the whole
    reason to report macro-F1 next to accuracy on a corpus where one
    class holds 48.7 % of the mass.
    """
    scores = []
    for value in targets.unique().tolist():
        hit = float(((targets == value) & (predicted == value)).sum())
        predicted_count = float((predicted == value).sum())
        target_count = float((targets == value).sum())
        if predicted_count == 0 or hit == 0:
            scores.append(0.0)
            continue
        precision = hit / predicted_count
        recall = hit / target_count
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else float("nan")


def _one_hot(classes: torch.Tensor, width: int) -> torch.Tensor:
    """Indicator columns for one-vs-rest ridge.

    Only TRAIN classes are ever encoded, and the vocabulary is fitted
    from those same train labels (`score_vocabulary_label`), so
    `OTHER_CLASS` cannot occur here. It is refused rather than tolerated
    because a negative index would wrap onto the last column and encode
    the row as the wrong class in silence.
    """
    if bool((classes < 0).any()):
        raise ValueError(
            "_one_hot got OTHER_CLASS: the vocabulary must be fitted from "
            "the labels being encoded"
        )
    columns = torch.zeros(classes.numel(), width)
    columns[torch.arange(classes.numel()), classes] = 1.0
    return columns


def _fit_and_score(
    plan: list,
    factors: dict[float, torch.Tensor],
    train_classes: torch.Tensor,
    width: int,
    cross: torch.Tensor,
    eval_classes: torch.Tensor,
    alphas: Sequence[float],
) -> tuple[float, float, float]:
    """Select a penalty on train folds, then score the eval split.

    One-vs-rest over the vocabulary, argmax across the columns: the
    estimator stays the closed-form ridge the regression arm uses, so a
    difference between the two arms is attributable to the SCORING and
    not to a second model. The penalty is selected on held-out train
    folds by the reported metric (accuracy), never on eval.
    """
    onehot = _one_hot(train_classes, width)
    best_alpha, best_cv = float(alphas[0]), float("-inf")
    for alpha in alphas:
        total, count = 0.0, 0
        for hold, fold_cross, fold_factors in plan:
            predicted = _predict_from_factor(
                fold_factors[alpha], onehot[~hold], fold_cross
            )
            score = accuracy_from_classes(
                train_classes[hold], predicted.argmax(dim=1)
            )
            if score == score:
                total += score
                count += 1
        if count and total / count > best_cv:
            best_alpha, best_cv = float(alpha), total / count
    predicted = _predict_from_factor(factors[best_alpha], onehot, cross).argmax(
        dim=1
    )
    return (
        best_alpha,
        accuracy_from_classes(eval_classes, predicted),
        macro_f1(eval_classes, predicted),
    )


def score_vocabulary_label(
    label: str,
    train_features: torch.Tensor,
    train_labels: Sequence[Any],
    eval_features: torch.Tensor,
    eval_labels: Sequence[Any],
    *,
    alphas: Sequence[float] = ALPHAS,
    min_examples: int = MIN_EXAMPLES,
    permutations: int = NULL_PERMUTATIONS,
    seed: int = 0,
) -> VocabOutcome:
    """Score `label` as classification under the permutation protocol.

    The pass criterion is the permutation floor — the largest accuracy
    the same selection reaches over `permutations` shuffled train
    targets, `p <= 1/(permutations+1)` — raised to the eval majority
    rate. A parametric `mean + 3 sd` bar is deliberately not used: on a
    bounded, imbalanced statistic it has already returned a threshold
    above 1.0 on this codebase (`heads.permutation_null`).
    """
    vocabulary = fit_vocabulary(train_labels)
    train_rows, train_classes = encode(train_labels, vocabulary)
    eval_rows, eval_classes = encode(eval_labels, vocabulary)
    n_train, n_eval = len(train_rows), len(eval_rows)
    skipped = _skip_reason(n_train, n_eval, min_examples, vocabulary, eval_classes)
    if skipped is not None:
        return VocabOutcome(
            label, "vocabulary", n_train, n_eval, None, None, None, None,
            None, None, None, None, None, skipped,
        )

    x_train = train_features[torch.tensor(train_rows, dtype=torch.long)]
    x_eval = eval_features[torch.tensor(eval_rows, dtype=torch.long)]
    z_train, mean, std = standardize(x_train)
    z_eval, _, _ = standardize(x_eval, mean=mean, std=std)
    gram = z_train @ z_train.T
    cross = z_eval @ z_train.T
    plan = fold_plan(gram, alphas=alphas)
    factors = ridge_factors(gram, alphas)
    width = len(vocabulary)
    alpha, accuracy, f1 = _fit_and_score(
        plan, factors, train_classes, width, cross, eval_classes, alphas
    )

    generator = torch.Generator().manual_seed(seed)
    draws = [
        _fit_and_score(
            plan,
            factors,
            train_classes[torch.randperm(n_train, generator=generator)],
            width,
            cross,
            eval_classes,
            alphas,
        )[1:]
        for _ in range(permutations)
    ]
    null_accuracy = max(value for value, _ in draws)
    null_f1 = max(value for _, value in draws)
    counts = torch.bincount(eval_classes[eval_classes >= 0], minlength=width)
    oov = int((eval_classes == OTHER_CLASS).sum())
    majority = float(max(int(counts.max()), oov)) / n_eval
    threshold = max(majority, null_accuracy)
    return VocabOutcome(
        label, "vocabulary", n_train, n_eval, width, oov, accuracy, f1,
        majority, null_accuracy, null_f1, threshold, accuracy > threshold,
        None, alpha,
    )


def _skip_reason(
    n_train: int,
    n_eval: int,
    min_examples: int,
    vocabulary: Sequence[float],
    eval_classes: torch.Tensor,
) -> str | None:
    """Why this label cannot be scored, or None."""
    if min(n_train, n_eval) == 0:
        return "no_labelled_examples"
    if min(n_train, n_eval) < min_examples:
        return "too_few_labelled_examples"
    if len(vocabulary) < 2 or int(eval_classes.unique().numel()) < 2:
        # a single class scores 1.0 for a head that learned nothing
        return "single_class_target"
    return None


def report(outcomes: Sequence[VocabOutcome]) -> str:
    """One line per label: score, threshold and what it had to clear."""
    lines = ["label                     n_eval  K  oov   acc  thresh  macro_f1  null_f1  alpha"]
    for outcome in outcomes:
        if outcome.skipped_reason is not None:
            lines.append(f"{outcome.label:<24}  SKIPPED ({outcome.skipped_reason})")
            continue
        lines.append(
            f"{outcome.label:<24} {outcome.n_eval:>6} {outcome.n_classes:>3} "
            f"{outcome.oov_eval:>4} {outcome.accuracy:5.3f} {outcome.threshold:6.3f} "
            f"{outcome.macro_f1:9.3f} {outcome.null_macro_f1:8.3f} "
            f"{outcome.alpha:.0e} "
            f"{'PASS' if outcome.beats_baseline else 'no-signal'}"
        )
    return "\n".join(lines)


def score_cache(
    train_cache: Path, eval_cache: Path, labels: Sequence[str] = VOCAB_LABELS
) -> list[VocabOutcome]:
    """Re-score an existing probe feature cache. CPU only, no backbone."""
    train = torch.load(train_cache, weights_only=True)
    evaluation = torch.load(eval_cache, weights_only=True)
    return [
        score_vocabulary_label(
            label,
            train["features"],
            [row.get(label) for row in train["labels"]],
            evaluation["features"],
            [row.get(label) for row in evaluation["labels"]],
        )
        for label in labels
    ]


def main(argv: list[str] | None = None) -> int:
    """`python -m unflatten.probe.vocab TRAIN.pt VAL.pt [--label NAME ...]`."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_cache", type=Path)
    parser.add_argument("eval_cache", type=Path)
    parser.add_argument("--label", action="append", dest="labels")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    outcomes = score_cache(
        args.train_cache, args.eval_cache, args.labels or VOCAB_LABELS
    )
    print(report(outcomes))
    if args.json is not None:
        args.json.write_text(
            json.dumps([o.as_dict() for o in outcomes], indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

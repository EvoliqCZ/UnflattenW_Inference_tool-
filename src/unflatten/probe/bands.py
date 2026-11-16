"""Representations and scoring for the pooling comparison.

The runner (`unflatten.probe.run_bands`) owns the GPU pass and the
cache; this module owns what is compared and how. Splitting them is what
makes the comparison re-scorable from a cache without a backbone —
refitting is seconds, extraction is not.

The head is deliberately the production probe's: closed-form ridge on
standardised columns, missing labels dropped rather than zeroed, counts
on log1p. Only the representation changes, which is the whole point of
the exercise. It is fitted through `kernel_ridge_predict` because a
banded arm is wider than the primal Gram matrix can be allocated for;
the estimator is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from unflatten.probe.heads import (
    ALPHAS,
    cross_validate_alpha,
    kernel_ridge_predict,
    r2_score,
    roc_auc,
    spearman,
    standardize,
    take_labelled,
)
from unflatten.probe.run import (
    BINARY_LABELS,
    COUNT_LABELS,
    MIN_EXAMPLES,
    REGRESSION_LABELS,
)

#: Band counts stored per page. Two, not a sweep: 4 asks whether the
#: coarsest possible vertical split (page quarters) is enough, 16 asks
#: whether a finer one buys anything. Both come out of the one
#: extraction pass, because the GPU pass is the only expensive step.
BAND_COUNTS: tuple[int, ...] = (4, 16)

#: Spacing as a PRESENCE question, derived from `modal_space_after_mpt`
#: rather than extracted: nearly half the pages are exactly 0 mpt and the
#: rest run to 82000 with a long tail, so the regression asks a linear
#: head to fit a zero-inflated target where one outlier dominates the
#: squared error. "Is this page's spacing non-zero" is a different
#: question from "how much", and only the second one is what R² was
#: measuring. Derived here rather than added to `probe.labels`, which
#: would bump `LABELS_VERSION` for a question the production probe does
#: not ask.
#:
#: NOTE — the predicate CHANGED with `LABELS_VERSION` 0.3.0. Over the
#: mean it read "any paragraph on this page is spaced"; over the mode it
#: reads "the page's TYPICAL paragraph is spaced", which is false for a
#: page that is mostly unspaced with a few spaced paragraphs. Any
#: `has_spacing` number measured before 0.3.0 answers the first question
#: and is not comparable with one measured after.
SPACING_PRESENCE = "has_spacing"

#: Labels scored as classification in this comparison.
BINARY_ARM_LABELS: tuple[str, ...] = (*BINARY_LABELS, SPACING_PRESENCE)


def arm_features(payload: dict[str, Any], arm: str) -> torch.Tensor:
    """The (pages, width) matrix for one arm of the comparison.

    `pooled` is the production vector verbatim; `bands{k}` is the same
    page's k strips flattened, so the head sees every strip as its own
    block of columns and can form differences between them.
    """
    if arm == "pooled":
        return payload["pooled"]
    if arm.startswith("bands"):
        count = int(arm[len("bands") :])
        stored = payload["bands"].get(count)
        if stored is None:
            raise KeyError(f"cache holds no {count}-band features")
        return stored.reshape(stored.shape[0], -1)
    raise ValueError(f"unknown arm {arm!r}")


# ------------------------------------------------------------------- fitting --


def with_spacing_presence(
    labels: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """`labels` plus `has_spacing`: is the page's MODAL spacing non-zero?

    Modal, not "any" — the source label is the page's most frequent
    value since `LABELS_VERSION` 0.3.0, so a page whose majority is 0 mpt
    is negative here even when several of its paragraphs are spaced. That
    is a narrower question than the pre-0.3.0 predicate asked; see the
    note on `SPACING_PRESENCE`.

    A page with NO spacing value stays unlabelled here too. "We could not
    read this page's spacing" is not "this page has no spacing", and
    turning the first into a negative would teach the head to call
    unmeasured pages unspaced — the same rule `probe.labels` applies to
    every other label.
    """
    return [
        {
            **row,
            SPACING_PRESENCE: (
                None
                if row.get("modal_space_after_mpt") is None
                else float(row["modal_space_after_mpt"]) > 0.0
            ),
        }
        for row in labels
    ]


def bootstrap_spearman_sigma(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    groups: Sequence[Any],
    *,
    resamples: int = 200,
    seed: int = 0,
) -> float | None:
    """How many noise sd a rank correlation stands above zero, over
    DOCUMENTS rather than pages.

    Eval pages are not independent draws: one document contributes many
    pages and every one of them carries the same document-level spacing
    label, so a threshold built from the page count is too small by
    roughly the square root of the pages per document. Resampling whole
    documents with replacement measures the sd that actually applies —
    the same correction `evaluate_spatial` makes for tokens within a
    page.

    None when the sd is undefined (one document, or a bootstrap that
    never varies); 0.0 would read as "measured, no signal".
    """
    index_of: dict[Any, list[int]] = {}
    for position, key in enumerate(groups):
        index_of.setdefault(key, []).append(position)
    documents = list(index_of.values())
    if len(documents) < 2:
        return None
    generator = torch.Generator().manual_seed(seed)
    draws: list[float] = []
    for _ in range(resamples):
        drawn = torch.randint(len(documents), (len(documents),), generator=generator)
        rows = [position for i in drawn.tolist() for position in documents[i]]
        picked = torch.tensor(rows, dtype=torch.long)
        value = spearman(targets[picked], predictions[picked])
        if value == value:
            draws.append(value)
    if len(draws) < 2:
        return None
    noise = float(torch.tensor(draws, dtype=torch.float64).std())
    if noise <= 0.0:
        return None
    rho = spearman(targets, predictions)
    return None if rho != rho else rho / noise


def score_arm(
    label: str,
    train_features: torch.Tensor,
    train_labels: Sequence[Any],
    eval_features: torch.Tensor,
    eval_labels: Sequence[Any],
    *,
    alphas: Sequence[float] = ALPHAS,
    eval_groups: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Fit one arm for one label and score it on the eval split.

    Label handling follows the production probe exactly — missing labels
    dropped rather than zeroed, counts fitted on log1p — so the numbers
    are comparable to the ones it reported.
    """
    kind = "binary" if label in BINARY_ARM_LABELS else "regression"
    x_train, y_train = take_labelled(train_features, train_labels)
    x_eval, y_eval = take_labelled(eval_features, eval_labels)
    # the same rows `take_labelled` kept, so a document key never drifts
    # off the page it belongs to
    kept_groups = (
        None
        if eval_groups is None
        else [key for key, value in zip(eval_groups, eval_labels) if value is not None]
    )
    if label in COUNT_LABELS:
        y_train, y_eval = torch.log1p(y_train), torch.log1p(y_eval)
    if min(x_train.shape[0], x_eval.shape[0]) < MIN_EXAMPLES:
        # the production probe's threshold, so an arm is skipped here
        # exactly where the pooled probe skipped it and the comparison
        # never rests on a label one side scored and the other did not
        return {
            "label": label,
            "kind": kind,
            "skipped_reason": (
                "no_labelled_examples"
                if min(x_train.shape[0], x_eval.shape[0]) == 0
                else "too_few_labelled_examples"
            ),
        }

    z_train, mean, std = standardize(x_train)
    z_eval, _, _ = standardize(x_eval, mean=mean, std=std)
    # centred columns: the kernel form is then the same estimator the
    # production probe's `fit_ridge` is, and one Gram serves every alpha
    gram = z_train @ z_train.T
    cross = z_eval @ z_train.T
    alpha, cv_curve = cross_validate_alpha(gram, y_train, kind, alphas=alphas)

    curve = []
    for candidate in alphas:
        predicted = kernel_ridge_predict(gram, y_train, cross, alpha=candidate)
        rho = spearman(y_eval, predicted)
        curve.append(
            {
                "alpha": candidate,
                "r2": r2_score(y_eval, predicted),
                "spearman": None if rho != rho else rho,
                "auroc": (
                    None if kind != "binary" else roc_auc(y_eval, predicted)
                ),
            }
        )
    chosen = next(row for row in curve if row["alpha"] == alpha)
    sigma = (
        None
        if kept_groups is None
        else bootstrap_spearman_sigma(
            y_eval, kernel_ridge_predict(gram, y_train, cross, alpha=alpha), kept_groups
        )
    )
    return {
        "label": label,
        "kind": kind,
        "n_train": int(x_train.shape[0]),
        "n_eval": int(x_eval.shape[0]),
        "width": int(x_train.shape[1]),
        "alpha": alpha,
        "r2": chosen["r2"],
        "spearman": chosen["spearman"],
        "spearman_sigma": sigma,
        "auroc": chosen["auroc"],
        "eval_documents": None if kept_groups is None else len(set(kept_groups)),
        "cv_curve": cv_curve,
        "eval_curve": curve,
        "skipped_reason": None,
    }


def compare(
    train: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    arms: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Every arm against every label, on one page set and one label set."""
    arms = arms or ("pooled", *(f"bands{k}" for k in BAND_COUNTS))
    labels = labels or (*BINARY_LABELS, *REGRESSION_LABELS, SPACING_PRESENCE)
    train = {**train, "labels": with_spacing_presence(train["labels"])}
    evaluation = {
        **evaluation, "labels": with_spacing_presence(evaluation["labels"])
    }
    results: dict[str, dict[str, Any]] = {}
    for arm in arms:
        train_x = arm_features(train, arm)
        eval_x = arm_features(evaluation, arm)
        results[arm] = {
            name: score_arm(
                name,
                train_x,
                [row.get(name) for row in train["labels"]],
                eval_x,
                [row.get(name) for row in evaluation["labels"]],
                eval_groups=evaluation.get("documents"),
            )
            for name in labels
        }
        print(f"[bands] arm {arm} hotové", flush=True)
    return {
        "pages_train": int(train["pooled"].shape[0]),
        "pages_eval": int(evaluation["pooled"].shape[0]),
        "arms": results,
    }


def report(summary: dict[str, Any]) -> str:
    """Side-by-side table: one row per label, one block per arm."""
    lines = [
        f"pages: train {summary['pages_train']}, eval {summary['pages_eval']}",
        f"{'signal':22}{'arm':10}{'width':>8}{'alpha':>9}"
        f"{'R2':>9}{'rho':>8}{'sigma':>8}{'auroc':>8}",
        "-" * 82,
    ]
    arms = list(summary["arms"])
    labels = list(summary["arms"][arms[0]])
    for name in labels:
        for arm in arms:
            row = summary["arms"][arm][name]
            if row.get("skipped_reason"):
                lines.append(
                    f"{name:22}{arm:10}{'-':>8}{'-':>9}{'-':>9}{'-':>8}{'-':>8}"
                    f"{'-':>8}  skipped ({row['skipped_reason']})"
                )
                continue
            rho = "-" if row["spearman"] is None else f"{row['spearman']:+.3f}"
            auroc = "-" if row["auroc"] is None else f"{row['auroc']:.3f}"
            sigma = (
                "-" if row["spearman_sigma"] is None else f"{row['spearman_sigma']:.1f}"
            )
            lines.append(
                f"{name:22}{arm:10}{row['width']:>8}{row['alpha']:>9.0f}"
                f"{row['r2']:>+9.3f}{rho:>8}{sigma:>8}{auroc:>8}"
            )
        lines.append("-" * 82)
    return "\n".join(lines)

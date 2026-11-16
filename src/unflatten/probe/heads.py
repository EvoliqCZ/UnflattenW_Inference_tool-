"""Linear heads and scoring for the Stage-0 backbone probe.

A probe answers "is this information present in the frozen features?",
so the head must be as weak as possible: anything with capacity of its
own would answer "is this information learnable by my head?" instead.
These are therefore plain linear maps fitted in closed form — no
optimiser, no learning rate, no seed, no early stopping. The same
features give the same numbers every time, which is what makes a probe
result something you can act on.

Ridge (rather than plain least squares) only because the feature matrix
is wide and correlated; `alpha` regularises, it does not add capacity.

Scoring is deliberately against a *constant* baseline:

- regression heads report R², where 0 means "no better than predicting
  the training mean" and negative means worse;
- binary heads report accuracy against the majority-class rate, plus
  AUROC, which is insensitive to class imbalance.

`take_labelled` enforces the RFC005 rule that a missing label is dropped,
never encoded as a numeric zero — a page with no runs has no font size,
and training a head to predict 0 for it would teach a lie.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch


def take_labelled(
    features: torch.Tensor, labels: Sequence[Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of `features` whose label is present, and those labels.

    A `None` label means the signal is absent on that page, not zero.
    """
    keep = [i for i, value in enumerate(labels) if value is not None]
    if not keep:
        return features[:0], torch.empty(0, dtype=torch.float32)
    index = torch.tensor(keep, dtype=torch.long)
    values = torch.tensor(
        [float(labels[i]) for i in keep], dtype=torch.float32
    )
    return features[index], values


def standardize(
    features: torch.Tensor,
    *,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-mean, unit-variance columns; reuses train statistics if given.

    A constant column has zero variance; its scale is left at 1 so the
    column becomes all-zero rather than NaN.
    """
    if mean is None:
        mean = features.mean(dim=0)
    if std is None:
        std = features.std(dim=0)
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
    return (features - mean) / std, mean, std


def fit_ridge(
    features: torch.Tensor, targets: torch.Tensor, *, alpha: float = 1.0
) -> torch.Tensor:
    """Closed-form ridge weights, with an unpenalised bias appended last."""
    n, d = features.shape
    ones = torch.ones(n, 1, dtype=features.dtype, device=features.device)
    design = torch.cat([features, ones], dim=1)
    penalty = torch.eye(d + 1, dtype=features.dtype, device=features.device) * alpha
    penalty[d, d] = 0.0  # never shrink the bias toward zero
    gram = design.T @ design + penalty
    return torch.linalg.solve(gram, design.T @ targets)


def kernel_ridge_predict(
    gram: torch.Tensor,
    targets: torch.Tensor,
    cross: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    """`predict(fit_ridge(...))` computed from inner products alone.

    Same estimator, different arithmetic. `fit_ridge` builds a
    (d+1)x(d+1) Gram matrix; a banded page representation is
    d = bands x feature_dim wide — 139 k columns at 16 bands — where that
    matrix is 77 GB, so the primal is unallocatable rather than merely
    slow. The dual identity `Xᵀ(XXᵀ + aI)⁻¹ = (XᵀX + aI)⁻¹Xᵀ` gives the
    same weights from an n x n solve over `gram = X Xᵀ`, and n here is a
    few hundred pages. Predictions then need only
    `cross = X_eval Xᵀ`, never the weights themselves.

    The equality holds on COLUMN-CENTRED features (what `standardize`
    returns): `Xᵀ1 = 0` decouples the intercept, so it is the target
    mean and the slopes match `fit_ridge` to floating-point tolerance.
    On uncentred columns this is a different estimator — one with a
    penalised intercept — so centre first.

    Comparing two representations means holding the head fixed, and this
    is what lets a 139 k-wide arm be fitted by the head the pooled arm
    was fitted by. It also makes a sweep cheap: `gram` does not depend on
    the target or on `alpha`, so one matrix serves every label, every
    penalty and every cross-validation fold.
    """
    n = gram.shape[0]
    center = targets.double().mean()
    ridge = gram.double() + torch.eye(n, dtype=torch.float64) * alpha
    dual = torch.linalg.solve(ridge, targets.double() - center)
    return (cross.double() @ dual + center).float()


def predict(weights: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Apply `fit_ridge` weights to `features`."""
    # device matters: a bias column built without it lands on the CPU and
    # the concat fails only once someone runs the head on an accelerator
    ones = torch.ones(
        features.shape[0], 1, dtype=features.dtype, device=features.device
    )
    return torch.cat([features, ones], dim=1) @ weights


def r2_score(targets: torch.Tensor, predictions: torch.Tensor) -> float:
    """Coefficient of determination; NaN when the target is constant."""
    total = ((targets - targets.mean()) ** 2).sum()
    if total <= 0:
        return float("nan")
    residual = ((targets - predictions) ** 2).sum()
    return float(1.0 - residual / total)


def accuracy(
    targets: torch.Tensor, scores: torch.Tensor, *, threshold: float = 0.5
) -> float:
    """Fraction correct after thresholding `scores`."""
    return float(((scores >= threshold).float() == targets).float().mean())


def majority_accuracy(targets: torch.Tensor) -> float:
    """Accuracy of always predicting the more frequent class."""
    positive = float(targets.mean())
    return max(positive, 1.0 - positive)


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Ranks of `values`, tied entries sharing their average rank."""
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(1, values.numel() + 1, dtype=values.dtype)
    unique, inverse = torch.unique(values, return_inverse=True)
    sums = torch.zeros(unique.numel(), dtype=values.dtype).index_add_(0, inverse, ranks)
    counts = torch.zeros(unique.numel(), dtype=values.dtype).index_add_(
        0, inverse, torch.ones_like(ranks)
    )
    return (sums / counts)[inverse]


def spearman(targets: torch.Tensor, predictions: torch.Tensor) -> float:
    """Rank correlation; NaN when either side is constant.

    R² asks "does the head predict the right magnitude?"; a probe asks
    "is the information present?". On heavy-tailed targets like run
    counts a linear head extrapolates badly and R² goes sharply negative
    while the ordering is still informative, so both are reported.
    """
    x = _average_ranks(targets.float())
    y = _average_ranks(predictions.float())
    xc, yc = x - x.mean(), y - y.mean()
    denom = torch.sqrt((xc**2).sum() * (yc**2).sum())
    if denom <= 0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def roc_auc(targets: torch.Tensor, scores: torch.Tensor) -> float:
    """Area under the ROC curve; NaN when only one class is present.

    Computed from the rank-sum identity, so ties are handled by using
    average ranks.
    """
    positives = int(targets.sum())
    negatives = int(targets.numel() - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    rank_sum = float(ranks[targets > 0].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


#: Contiguous cross-validation folds over the train split. Contiguous
#: rather than random: pages arrive grouped by document, and a random
#: split would put pages of one document on both sides of a fold — the
#: penalty would then be chosen against a leak.
CV_FOLDS = 5

#: Penalty grid every head is selected from. A fixed penalty is the wrong
#: instrument here: the features are wider than the sample, so how hard a
#: head is shrunk decides what it can report, and the right amount
#: differs per label — page counts wanted 1e3 where paragraph spacing
#: wanted 1e6. The grid runs two decades past where any label's optimum
#: has fallen, because an optimum ON the last grid point is a truncated
#: measurement rather than a measured one.
ALPHAS: tuple[float, ...] = (1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8)


def _metric(kind: str, targets: torch.Tensor, scores: torch.Tensor) -> float:
    """The verdict metric for a label kind: AUROC binary, R² regression."""
    return roc_auc(targets, scores) if kind == "binary" else r2_score(targets, scores)


def ridge_factors(
    gram: torch.Tensor, alphas: Sequence[float] = ALPHAS
) -> dict[float, torch.Tensor]:
    """Cholesky factor of `gram + alpha I` for every penalty on the grid.

    Cached and reused rather than re-solved. `gram` does not depend on
    the target, so one factor per (fold, penalty) answers the real fit
    and every label permutation; only the right-hand side changes, and
    `torch.cholesky_solve` on a factor is O(n²) against the O(n³) of a
    fresh solve.

    Cholesky and not an eigendecomposition, which would reuse across
    penalties as well: `torch.linalg.eigh` measures 12 s on an 880x880
    matrix on this machine against 0.03 s for `cholesky`, so the
    factorisation that reuses more is four hundred times the wrong
    choice. The matrix is positive definite for any positive `alpha`, so
    Cholesky is always available.
    """
    matrix = gram.double()
    eye = torch.eye(matrix.shape[0], dtype=torch.float64)
    return {
        alpha: torch.linalg.cholesky(matrix + eye * alpha) for alpha in alphas
    }


def _predict_from_factor(
    factor: torch.Tensor, targets: torch.Tensor, cross: torch.Tensor
) -> torch.Tensor:
    """`kernel_ridge_predict` against a precomputed Cholesky factor.

    `targets` may be (n,) or (n, k). The second form is what makes a
    permutation test affordable: every shuffle shares this factor and
    differs only in its right-hand side, so k of them cost one solve
    rather than k. That matters more than the arithmetic suggests —
    a torch op on this machine carries ~15 ms of fixed overhead
    whatever its size, so 945 small solves per label were 14 s of pure
    dispatch.
    """
    flat = targets.dim() == 1
    columns = targets.double().unsqueeze(1) if flat else targets.double()
    center = columns.mean(dim=0, keepdim=True)
    dual = torch.cholesky_solve(columns - center, factor)
    predicted = (cross.double() @ dual + center).float()
    return predicted.squeeze(1) if flat else predicted


def fold_plan(
    gram: torch.Tensor,
    *,
    alphas: Sequence[float] = ALPHAS,
    folds: int = CV_FOLDS,
) -> list[tuple[torch.Tensor, torch.Tensor, dict[float, torch.Tensor]]]:
    """Per-fold `(held-out mask, cross block, factor per penalty)`.

    Contiguous folds, not random: pages arrive grouped by document, and a
    random split would put pages of one document on both sides of a fold,
    so the penalty would be chosen against a leak.

    Built once and handed back because it depends only on the features —
    a permutation test refits these same folds twenty times over.
    """
    n = gram.shape[0]
    bounds = [index * n // folds for index in range(folds + 1)]
    plan = []
    for start, stop in zip(bounds, bounds[1:]):
        hold = torch.zeros(n, dtype=torch.bool)
        hold[start:stop] = True
        keep = ~hold
        if int(keep.sum()) < 2 or int(hold.sum()) < 2:
            continue
        plan.append(
            (hold, gram[hold][:, keep], ridge_factors(gram[keep][:, keep], alphas))
        )
    return plan


def cross_validate_curve(
    plan: list,
    targets: torch.Tensor,
    kind: str,
    *,
    alphas: Sequence[float] = ALPHAS,
) -> torch.Tensor:
    """Mean fold score of every penalty, for each column of `targets`.

    Returns (len(alphas), k). Columns are scored together because they
    share the folds' factorisations; an unscorable fold is dropped per
    column, as a fold nobody could score rather than a zero.
    """
    columns = targets.unsqueeze(1) if targets.dim() == 1 else targets
    width = columns.shape[1]
    curve = torch.full((len(alphas), width), float("nan"))
    for index, alpha in enumerate(alphas):
        totals = torch.zeros(width)
        counts = torch.zeros(width)
        for hold, cross, factors in plan:
            predicted = _predict_from_factor(factors[alpha], columns[~hold], cross)
            for column in range(width):
                value = _metric(kind, columns[hold, column], predicted[:, column])
                if value == value:
                    totals[column] += value
                    counts[column] += 1
        scored = counts > 0
        curve[index] = torch.where(
            scored, totals / counts.clamp(min=1), torch.full_like(totals, float("nan"))
        )
    return curve


def cross_validate_alpha(
    gram: torch.Tensor,
    targets: torch.Tensor,
    kind: str,
    *,
    alphas: Sequence[float] = ALPHAS,
    folds: int = CV_FOLDS,
    plan: list | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """Pick the penalty on TRAIN alone, and say what every one scored.

    Selection has to happen somewhere, and doing it on the eval split
    would report the best of nine draws as if it were one measurement.
    A fold whose target is single-valued scores NaN and is dropped rather
    than counted as zero — it is a fold nobody could score.

    `plan` lets a caller that refits these folds many times build them
    once (`fold_plan`).
    """
    plan = fold_plan(gram, alphas=alphas, folds=folds) if plan is None else plan
    scores = cross_validate_curve(plan, targets, kind, alphas=alphas)[:, 0]
    curve = [
        {"alpha": alpha, "cv": float(value)}
        for alpha, value in zip(alphas, scores.tolist())
    ]
    scored = [row for row in curve if row["cv"] == row["cv"]]
    if not scored:
        raise ValueError("no cross-validation fold could be scored")
    best = max(scored, key=lambda row: row["cv"])
    return float(best["alpha"]), curve


#: Label shuffles the null baseline is measured over. Twenty, because
#: the floor is the largest draw — an exact permutation criterion, so a
#: score that clears it has `p <= 1/(n+1)` — and twenty is where that
#: buys a usable p without the fit cost mattering. Five was too few: the
#: null here is wide (on `has_image`, whose 25 positive eval pages sit in
#: six documents, shuffled labels score AUROC with sd 0.20), so a floor
#: from a handful of draws moves with the seed.
NULL_PERMUTATIONS = 20


def permutation_null(
    gram: torch.Tensor,
    targets: torch.Tensor,
    cross: torch.Tensor,
    eval_targets: torch.Tensor,
    kind: str,
    *,
    alphas: Sequence[float] = ALPHAS,
    folds: int = CV_FOLDS,
    permutations: int = NULL_PERMUTATIONS,
    plan: list | None = None,
    seed: int = 0,
) -> dict[str, float | None]:
    """What the same procedure scores when the train labels are SHUFFLED.

    Choosing a penalty per label is a search, and a search improves
    whatever it is pointed at — including features that carry nothing.
    The constant baselines (R² 0, AUROC 0.5) stay correct nulls for a
    FIXED penalty, but they say nothing about what the selection itself
    can extract, so the baseline is re-measured under the same procedure
    rather than assumed unmoved. The whole pipeline runs again,
    cross-validated selection included, on permuted training targets
    against the true eval targets: same features, same width, same
    search, association destroyed.

    The floor returned is the LARGEST draw. That is the exact
    permutation criterion — a score above every shuffle has
    `p <= 1/(n+1)` — and it needs no assumption about the null's shape.
    A normal `mean + 3 sd` was the obvious alternative and is wrong here:
    it extrapolates a tail from twenty samples of a BOUNDED statistic,
    and on an imbalanced label it returns an AUROC floor above 1.0, which
    no result can clear.

    It also measures something the parametric thresholds cannot see.
    `roc_auc_noise_sd` assumes independent pages; these pages come in
    documents that share a label, so the real null is several times
    wider than that formula and the permutation floor is the honest
    bound.

    Cheap by construction: `gram` does not depend on the target, so all
    permutations share one set of fold decompositions.
    """
    plan = fold_plan(gram, alphas=alphas, folds=folds) if plan is None else plan
    full = ridge_factors(gram, alphas)
    generator = torch.Generator().manual_seed(seed)
    shuffles = torch.stack(
        [
            targets[torch.randperm(targets.numel(), generator=generator)]
            for _ in range(permutations)
        ],
        dim=1,
    )
    curve = cross_validate_curve(plan, shuffles, kind, alphas=alphas)
    draws: dict[str, list[float]] = {"r2": [], "spearman": [], "auroc": []}
    for column in range(permutations):
        scores = curve[:, column]
        if bool(torch.isnan(scores).all()):
            # a shuffle can leave every fold single-class; it scores nothing
            continue
        alpha = alphas[
            max(
                (i for i in range(len(alphas)) if scores[i] == scores[i]),
                key=lambda i: float(scores[i]),
            )
        ]
        predicted = _predict_from_factor(
            full[alpha], shuffles[:, column], cross
        )
        for name, value in (
            ("r2", r2_score(eval_targets, predicted)),
            ("spearman", spearman(eval_targets, predicted)),
            ("auroc", roc_auc(eval_targets, predicted) if kind == "binary"
             else float("nan")),
        ):
            if value == value:
                draws[name].append(value)
    return {
        name: (max(seen) if seen else None) for name, seen in draws.items()
    }


def spearman_noise_sd(n: int) -> float:
    """Standard deviation of Spearman's rho under the null hypothesis."""
    return float("inf") if n < 3 else 1.0 / math.sqrt(n - 1)


def roc_auc_noise_sd(positives: int, negatives: int) -> float:
    """Standard deviation of AUROC under the null (Hanley-McNeil)."""
    if positives < 1 or negatives < 1:
        return float("inf")
    return math.sqrt(
        (positives + negatives + 1) / (12.0 * positives * negatives)
    )

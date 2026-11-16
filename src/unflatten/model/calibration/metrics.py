"""RFC005 §22.3/§41.2/§41.3 calibration diagnostics.

Every function here takes plain `scores`/`labels` arrays supplied by the
caller — no split resolution, no dataset access. `labels` is always
binary correctness (1 = correct/accept-worthy, 0 = incorrect), and
`scores` is the model's confidence in `[0, 1]`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _as_arrays(scores: Sequence[float], labels: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    if scores_arr.shape != labels_arr.shape:
        raise ValueError(
            f"scores and labels must have the same shape, got "
            f"{scores_arr.shape} and {labels_arr.shape}"
        )
    if scores_arr.ndim != 1:
        raise ValueError(f"scores/labels must be 1-D, got ndim={scores_arr.ndim}")
    if scores_arr.size == 0:
        raise ValueError("scores/labels must not be empty")
    return scores_arr, labels_arr


def expected_calibration_error(
    scores: Sequence[float],
    labels: Sequence[float],
    bin_edges: Sequence[float] | int = 10,
) -> float:
    """Binned ECE (RFC005 §22.3 "ECE s pinned bins").

    `bin_edges` is either an explicit, caller-pinned, strictly increasing
    sequence covering `[0, 1]` (the RFC005 "pinned bins" requirement), or
    an int number of equal-width bins over `[0, 1]` for convenience.
    Empty bins contribute zero (they carry no evidence, not a penalty).
    """
    scores_arr, labels_arr = _as_arrays(scores, labels)
    edges = _resolve_bin_edges(bin_edges)
    n = scores_arr.size
    total = 0.0
    bin_indices = np.digitize(scores_arr, edges[1:-1], right=False)
    for bin_id in range(len(edges) - 1):
        mask = bin_indices == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        avg_confidence = float(scores_arr[mask].mean())
        avg_accuracy = float(labels_arr[mask].mean())
        total += (count / n) * abs(avg_confidence - avg_accuracy)
    return total


def _resolve_bin_edges(bin_edges: Sequence[float] | int) -> np.ndarray:
    if isinstance(bin_edges, int):
        if bin_edges <= 0:
            raise ValueError(f"n_bins must be positive, got {bin_edges}")
        return np.linspace(0.0, 1.0, bin_edges + 1)
    edges = np.asarray(list(bin_edges), dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must have at least 2 values")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing")
    return edges


def brier_score(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Mean squared error between confidence and correctness (§22.3)."""
    scores_arr, labels_arr = _as_arrays(scores, labels)
    return float(np.mean((scores_arr - labels_arr) ** 2))


def risk_coverage_curve(
    scores: Sequence[float], labels: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve (§41.2): sort by descending confidence, sweep the
    accept threshold from most- to least-confident.

    Returns `(coverage, risk)`, both length `n`, sorted by non-decreasing
    coverage. `coverage[i] = (i+1)/n`; `risk[i]` is the error rate among
    the `i+1` most-confident predictions.
    """
    scores_arr, labels_arr = _as_arrays(scores, labels)
    n = scores_arr.size
    order = np.argsort(-scores_arr, kind="stable")
    sorted_labels = labels_arr[order]
    cumulative_errors = np.cumsum(1.0 - sorted_labels)
    counts = np.arange(1, n + 1, dtype=np.float64)
    coverage = counts / n
    risk = cumulative_errors / counts
    return coverage, risk


def aurc(coverage: Sequence[float], risk: Sequence[float]) -> float:
    """Area under the risk-coverage curve (§41.2), trapezoidal in coverage."""
    coverage_arr = np.asarray(coverage, dtype=np.float64)
    risk_arr = np.asarray(risk, dtype=np.float64)
    if coverage_arr.shape != risk_arr.shape:
        raise ValueError("coverage and risk must have the same shape")
    if coverage_arr.size < 2:
        raise ValueError("need at least 2 points to integrate")
    order = np.argsort(coverage_arr)
    return float(np.trapezoid(risk_arr[order], coverage_arr[order]))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (§41.3 "conservative
    confidence bound"; "zero observed failure does not mean zero risk").

    `n=0` returns the maximally uninformative `(0.0, 1.0)` rather than
    raising, so an empty-evidence caller cannot silently treat it as
    zero risk.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not (0 <= successes <= n):
        raise ValueError(f"successes must be in [0, n], got successes={successes}, n={n}")
    if n == 0:
        return 0.0, 1.0
    phat = successes / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return max(0.0, lower), min(1.0, upper)

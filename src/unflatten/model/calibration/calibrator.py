"""RFC005 §22.2/§41.1 calibrator interface, fit on caller-supplied arrays.

Neither implementation resolves a split, reads a dataset or picks
`auto_accept` thresholds — the caller (a future S8 program, not this
package) supplies `scores`/`labels` already restricted to
`val_calibration_fit`, per :mod:`unflatten.model.calibration.oof_manifest`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import torch


class Calibrator(Protocol):
    """A post-hoc calibrator: `fit` once, `predict` many times."""

    def fit(self, scores: Sequence[float], labels: Sequence[float]) -> None: ...

    def predict(self, scores: Sequence[float]) -> np.ndarray: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, object]) -> None: ...


class TemperatureCalibrator:
    """Single-parameter temperature scaling in logit space.

    `temperature` starts at `1.0` (identity — step-0-equivalent to the
    uncalibrated score) and is fit by minimizing binary cross-entropy
    with LBFGS, matching the standard temperature-scaling procedure.
    """

    def __init__(self, max_iter: int = 50, eps: float = 1e-6) -> None:
        self.temperature: float = 1.0
        self._max_iter = max_iter
        self._eps = eps

    def _logits(self, scores: Sequence[float]) -> torch.Tensor:
        clipped = np.clip(np.asarray(scores, dtype=np.float64), self._eps, 1 - self._eps)
        return torch.logit(torch.tensor(clipped, dtype=torch.float64))

    def fit(self, scores: Sequence[float], labels: Sequence[float]) -> None:
        logits = self._logits(scores)
        labels_t = torch.tensor(np.asarray(labels, dtype=np.float64))
        log_temperature = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=self._max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature)
            probs = torch.sigmoid(logits / temperature)
            loss = torch.nn.functional.binary_cross_entropy(probs, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temperature).detach().item())

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        logits = self._logits(scores)
        probs = torch.sigmoid(logits / self.temperature)
        return probs.numpy()

    def state_dict(self) -> dict[str, object]:
        return {"temperature": self.temperature}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.temperature = float(state["temperature"])  # type: ignore[arg-type]


class IsotonicCalibrator:
    """Isotonic (PAVA) calibrator: a non-decreasing step function of score."""

    def __init__(self) -> None:
        self._x_thresholds: np.ndarray | None = None
        self._y_fit: np.ndarray | None = None

    def fit(self, scores: Sequence[float], labels: Sequence[float]) -> None:
        scores_arr = np.asarray(scores, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.float64)
        order = np.argsort(scores_arr, kind="stable")
        self._x_thresholds = scores_arr[order]
        self._y_fit = _pool_adjacent_violators(labels_arr[order])

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        if self._x_thresholds is None or self._y_fit is None:
            raise RuntimeError("IsotonicCalibrator.predict called before fit")
        return np.interp(
            np.asarray(scores, dtype=np.float64), self._x_thresholds, self._y_fit
        )

    def state_dict(self) -> dict[str, object]:
        if self._x_thresholds is None or self._y_fit is None:
            raise RuntimeError("IsotonicCalibrator.state_dict called before fit")
        return {
            "x_thresholds": self._x_thresholds.tolist(),
            "y_fit": self._y_fit.tolist(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._x_thresholds = np.asarray(state["x_thresholds"], dtype=np.float64)  # type: ignore[arg-type]
        self._y_fit = np.asarray(state["y_fit"], dtype=np.float64)  # type: ignore[arg-type]


def _pool_adjacent_violators(y: np.ndarray) -> np.ndarray:
    """Isotonic (non-decreasing) regression of `y` under L2 loss, uniform
    weights, via the pool-adjacent-violators algorithm."""
    values = list(y.astype(np.float64))
    weights = [1.0] * len(values)
    i = 0
    while i < len(values) - 1:
        if values[i] > values[i + 1]:
            merged_value = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / (
                weights[i] + weights[i + 1]
            )
            merged_weight = weights[i] + weights[i + 1]
            values[i : i + 2] = [merged_value]
            weights[i : i + 2] = [merged_weight]
            i = max(i - 1, 0)
        else:
            i += 1
    out = np.empty(len(y), dtype=np.float64)
    pos = 0
    for value, weight in zip(values, weights, strict=True):
        count = int(round(weight))
        out[pos : pos + count] = value
        pos += count
    return out

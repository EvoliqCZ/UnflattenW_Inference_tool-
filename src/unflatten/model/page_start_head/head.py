"""Heads over 6D band moments; outputs are logits for (has_header, has_footer)."""

from __future__ import annotations

import torch
from torch import nn


class _Standardised(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))

    def standardise(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mu) / self.sd

    def set_standardisation(self, mu: torch.Tensor, sd: torch.Tensor) -> None:
        if (sd <= 0).any() or mu.shape != self.mu.shape or sd.shape != self.sd.shape:
            raise ValueError("page start head: standardisation needs positive sd of shape [6D]")
        self.mu.copy_(mu)
        self.sd.copy_(sd)


class LogisticPageStartHead(_Standardised):
    kind = "logistic"

    def __init__(self, in_dim: int) -> None:
        super().__init__(in_dim)
        self.linear = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.standardise(x))


class MlpPageStartHead(_Standardised):
    kind = "mlp"

    def __init__(self, in_dim: int, hidden: int = 256) -> None:
        super().__init__(in_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.standardise(x))


HEAD_KINDS = {"logistic": LogisticPageStartHead, "mlp": MlpPageStartHead}

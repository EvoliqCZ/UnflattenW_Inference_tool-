"""`LoRALinear` — RFC005 §25 LoRA wrapper for one frozen `nn.Linear`.

Step 0 == base (DEC-047 condition 2): `B` is zero-initialized, so the
`A @ B` delta is exactly zero at construction and `forward` returns the
wrapped linear's own output bit-for-bit (no dropout, no scaling artifact,
because the whole delta term is the zero tensor).

Merge/unmerge (§25.5): `merge()` folds `scaling * B^T @ A` into the base
weight in place and marks the module merged; `unmerge()` subtracts the
same delta back out. Both are exact inverses in the same dtype the base
weight is stored in; the equivalence test allows only floating-point
round-off tolerance, never a semantic difference.
"""

from __future__ import annotations

import torch
from torch import nn


class LoRAMergeStateError(RuntimeError):
    """`merge()`/`unmerge()` called in the wrong state."""


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` with a trainable low-rank delta.

    The base linear's parameters are frozen (`requires_grad=False`) —
    only `lora_A` and `lora_B` are trainable, matching a `delta_update`
    group per RFC005 §24 (this module is not itself registered in any
    such group; see the package docstring).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        in_features = base.in_features
        out_features = base.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._merged = False

    @property
    def merged(self) -> bool:
        return self._merged

    def _delta_weight(self) -> torch.Tensor:
        """`scaling * B @ A`, same shape as `base.weight` (out, in)."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self._merged:
            return base_out
        delta = self.dropout(x) @ self._delta_weight().T
        return base_out + delta

    def merge(self) -> None:
        """Fold the LoRA delta into `base.weight` in place (§25.5)."""
        if self._merged:
            raise LoRAMergeStateError("already merged")
        with torch.no_grad():
            self.base.weight.add_(self._delta_weight())
        self._merged = True

    def unmerge(self) -> None:
        """Subtract the LoRA delta back out of `base.weight` (§25.5)."""
        if not self._merged:
            raise LoRAMergeStateError("not merged")
        with torch.no_grad():
            self.base.weight.sub_(self._delta_weight())
        self._merged = False

    def trainable_parameter_count(self) -> int:
        """§25.1's evolving output field, computed from this instance."""
        return self.lora_A.numel() + self.lora_B.numel()

"""Hard reasoning gate -- RFC004-ERRATA-005 E1 variant 1 (the chosen gate).

`HardReasoningGate` computes only the gate decision `g ∈ {0,1}` (E1 §22.3's
`h' = h + ... + g(h) · R_spec(h, V)`); it never computes a residual itself,
so it has no opinion about any specialization's internals.

Three rules land here:

- **Rule 1 (architectural bypass).** `enabled=False` (`reasoning=off`) forces
  `g ≡ 0` unconditionally -- the gate never even reads `document_state`
  through its projection on that path, so no specialization output can leak
  through regardless of what it computes.
- **Rule 6 (grammar eligibility mask).** `eligibility_mask` (an analogue of
  the router's §22.5 grammar mask) zeroes the gate wherever the grammar
  context of the selected specialization does not apply; a masked position
  can never activate, on or off.
- **Variant 1's estimator.** The gate is a hard threshold at inference
  (`sigmoid(logits) > 0.5`) with a straight-through estimator so gradient
  still reaches `gate_logit_proj` through the softened value.

`ZeroInitOutputProjection` is the reusable building block E1 rule 4 asks
every specialization to end with ("Výstupná projekcia `R_spec` sa
inicializuje nulou"): a `Linear` whose weight and bias start at exactly zero,
so a freshly registered specialization computes an all-zero residual at
step 0 regardless of what precedes this projection.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HardReasoningGate(nn.Module):
    """The gate decision only -- see module docstring for the three rules.

    `attach_point` is a required constructor argument (OPEN-004-014 is open
    on which point the framework will use once a specialization is selected;
    Wave 1 makes it a parameter rather than picking one).
    """

    def __init__(self, d_model: int, attach_point: str, *, enabled: bool = True) -> None:
        super().__init__()
        if attach_point not in ("token", "node"):
            raise ValueError(
                f"attach_point must be 'token' or 'node' (OPEN-004-014), got {attach_point!r}"
            )
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model!r}")
        self.attach_point = attach_point
        self.enabled = enabled
        self.gate_logit_proj = nn.Linear(d_model, 1)

    def forward(self, document_state: Tensor, eligibility_mask: Tensor) -> Tensor:
        """`document_state`: `[..., d_model]`. `eligibility_mask`: `[...]`,
        one fewer trailing dim than `document_state`, `{0,1}`-valued.

        Returns a hard `{0,1}` tensor shaped like `eligibility_mask`.
        """
        if document_state.shape[:-1] != eligibility_mask.shape:
            raise ValueError(
                f"document_state slot shape {tuple(document_state.shape[:-1])!r} must "
                f"match eligibility_mask shape {tuple(eligibility_mask.shape)!r}"
            )
        if not self.enabled:
            # Rule 1: the bypass does not read the projection's output at
            # all -- `g` is a constant zero, independent of any weight.
            return torch.zeros(eligibility_mask.shape, dtype=torch.float32)

        logits = self.gate_logit_proj(document_state).squeeze(-1)
        soft = torch.sigmoid(logits)
        hard = (soft > 0.5).to(soft.dtype)
        straight_through = hard + (soft - soft.detach())
        return straight_through * eligibility_mask.to(straight_through.dtype)


class ZeroInitOutputProjection(nn.Module):
    """A `Linear(d_model, d_model)` initialized to exact zero (E1 rule 4).

    Every reasoning specialization should route its final residual through
    an instance of this (or an equivalently zero-initialized projection) so
    that, at step 0, `reasoning=on` and `reasoning=off` agree bit-for-bit
    (RFC005-ERRATA-007 R-A). This module is generic infrastructure, not a
    specialization: it has no forward path outside `Linear.forward`.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)

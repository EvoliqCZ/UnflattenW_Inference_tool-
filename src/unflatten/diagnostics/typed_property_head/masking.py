"""Design §6 — the masked-mean reduction, shared verbatim across head families.

`known_mask=False` means `no_supervision` (RFC003 §29.1/§29.5) -- it must
NEVER be treated as a fabricated `0`, `abstained`, `not_applicable` or
`conflict`. Every head family's `loss()` reduces its per-example loss
through this one function so the reduction is defined exactly once,
instead of re-derived (and possibly re-broken) per subclass.

A batch with zero known examples for a pointer must return a defined
`0`-valued loss -- via `known_mask.float().sum().clamp(min=1)` -- rather
than a `NaN` from a `0/0` divide (design §6): "silently NaN" is worse
than "silently contributes nothing."

The load-bearing property this module exists to prove is stronger than
"the forward value is correct at `known_mask=False`": a masked example
must contribute EXACTLY ZERO gradient, not merely a zero-valued forward
term that still happens to backprop through some other tensor alias. See
`tests/diagnostics/typed_property_head/test_masking.py` for the
adversarial-mutation proof (design §6, five steps) -- this reduction was
watched to fail against a plain, unmasked `.mean()` before being trusted.
"""

from __future__ import annotations

import torch


def masked_mean(per_example_loss: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor:
    """Design §6's masked reduction, verbatim.

    `known_mask` is required, not optional and not defaulted to `True` --
    callers must decide supervision explicitly for every example.
    """
    mask = known_mask.to(per_example_loss.dtype)
    return (per_example_loss * mask).sum() / mask.sum().clamp(min=1)

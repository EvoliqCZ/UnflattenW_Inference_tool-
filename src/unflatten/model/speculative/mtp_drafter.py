"""RFC004 §37.3: `MtpDrafter`, an `MtpHeads`-backed `SpeculativeDrafter`.

`SpeculativeDrafter.draft(snapshot)` (`drafter.py`) only carries the
committed token tuple -- no hidden state -- because the harness's own
`PrimaryDecoder` is opaque. `MtpHeads.propose` needs the primary's hidden
state at the snapshot's position. `hidden_state_at` closes that gap by
composition: it is a plain callable a caller binds to whatever object
already tracks hidden states keyed by prefix (a `PrimaryDecoder` adapter
over a real `IncrementalDecodeState`, or a test double) -- `MtpDrafter`
itself holds no model, no cache, and makes no side-effecting calls; it
reads one hidden state and returns tokens, exactly `SpeculativeDrafter`'s
"without side effects" contract (§37.1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from unflatten.model.speculative.drafter import DecoderSnapshot
from unflatten.model.speculative.mtp import MtpHeads


@dataclass(frozen=True, slots=True)
class MtpDrafter:
    """Proposes up to `mtp_heads.k` tokens from the hidden state at
    `snapshot.committed_tokens`, via `hidden_state_at`."""

    mtp_heads: MtpHeads
    hidden_state_at: Callable[[tuple[int, ...]], torch.Tensor]

    def draft(self, snapshot: DecoderSnapshot) -> tuple[int, ...]:
        hidden = self.hidden_state_at(snapshot.committed_tokens)
        with torch.inference_mode():
            return self.mtp_heads.propose(hidden)

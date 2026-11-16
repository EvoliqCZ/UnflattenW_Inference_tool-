"""RFC004 §37.3 MTP heads: `k` auxiliary heads predicting tokens `t+1..t+k`
from the primary decoder's own hidden state at `t`.

Sharing the token domain (design's own composite-vocabulary decision,
`composite_vocab.PackedRanges.total_count`) means an `MtpHeads` output is
directly comparable to `composite_head`'s own logits -- an MTP head's
argmax is a candidate for the SAME step `composite_head` would decide, not
a token from a separate vocabulary. `MtpHeads` never reads or wraps
`composite_head` itself (RFC004 §37.3: MTP heads are their own trainable
identity, not a repackaging of the primary's output projection) -- it is a
plain `k`-way linear readout over whatever hidden state the caller hands it.

**Init: zero, not independent random (documented choice).** A freshly
`installed_speculators`-listed but not yet trained `MtpHeads` must never
change what a caller *observes* merely by existing (RFC004 §37.5: "an
installed component must not be automatically active"). Zero-initialized
`nn.Linear` weight+bias make every head's logits row uniform (all zeros)
regardless of the hidden state, so `MtpHeads.propose` is deterministic and
carries no accidental signal from an untrained random projection -- the
lowest-packed-id tie-break (mirroring `free_running_decode.PINNED_TIE_BREAK`)
is the only thing that decides its output. This does *not* mean quality:
against a real vocabulary this predicts index 0 every time, i.e. near-zero
acceptance -- exactly the "random MTP weights, low acceptance, still
lossless" test case (`mtp_aux_training_used=False`'s natural starting
point). Training with :func:`mtp_aux_loss` is what gives it real signal;
nothing here trains anything.

This module authorizes no gradient update on its own -- it declares
`nn.Module`s and a standalone loss function only.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MtpHeads(nn.Module):
    """`k` independent linear readouts `d_model -> vocab_size`, zero-init.

    `forward(hidden)` takes `hidden` shaped `(batch, d_model)` (the same
    single-position hidden state `QwenDenseCandidateModel.encode_step`/
    `encode_prefix` already return) and returns `(batch, k, vocab_size)`
    stacked logits -- head `i` predicts the token `i+1` steps ahead of
    `hidden`'s own position.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        k: int,
        tie_output_head: bool = False,
        frozen_output_head_weight: torch.Tensor | None = None,
        frozen_output_head_bias: torch.Tensor | None = None,
    ) -> None:
        """`tie_output_head` (2026-09-28, opt-in): instead of `k` INDEPENDENT
        `d_model -> vocab_size` projections (`151936 x 2560` each -- most of
        an untied `MtpHeads`' 1.17G parameters, per
        `docs/runs/dspark-rehearsal-2026-09-27.md` §5), share ONE frozen
        `d_model -> vocab_size` projection across every head -- the target's
        own output head, exactly what the target itself uses -- and give
        each head only a small trainable `d_model -> d_model` residual
        adapter (`hidden + head(hidden)`, THEN the shared frozen
        projection). Requires `frozen_output_head_weight` (the target's
        post-bridge output projection, e.g. the DSpark/MTP export's
        `frozen_tables.pt`'s `head_weight`/`head_bias`, same tables
        `s7_dflash_train.py`/`s7_mtp_train.py` already load).
        Zero-initialized adapters (like the untied path's zero-initialized
        heads) make an installed-but-untrained tied `MtpHeads` reduce to
        `frozen_output_head(hidden)` -- the target's OWN readout of the
        unmodified hidden state, a defined (not random) starting point
        rather than the untied path's uniform-logits starting point.
        Retraining note: a checkpoint trained with `tie_output_head=False`
        cannot be loaded into a `tie_output_head=True` instance or vice
        versa (`state_dict` shapes differ) -- switching this flag for a
        production run requires the trainer (`s7_mtp_train.py`'s
        `--tie-output-head`) to retrain from scratch, not resume."""
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if tie_output_head and frozen_output_head_weight is None:
            raise ValueError("tie_output_head=True requires frozen_output_head_weight")
        self.k = k
        self.vocab_size = vocab_size
        self.tie_output_head = tie_output_head
        if tie_output_head:
            self.output_head = nn.Linear(
                d_model, vocab_size, bias=frozen_output_head_bias is not None
            )
            with torch.no_grad():
                self.output_head.weight.copy_(frozen_output_head_weight)
                if frozen_output_head_bias is not None:
                    self.output_head.bias.copy_(frozen_output_head_bias)
            self.output_head.weight.requires_grad_(False)
            if frozen_output_head_bias is not None:
                self.output_head.bias.requires_grad_(False)
            self.heads = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(k))
        else:
            self.output_head = None
            self.heads = nn.ModuleList(nn.Linear(d_model, vocab_size) for _ in range(k))
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 2:
            raise ValueError(f"hidden must be (batch, d_model), got shape {tuple(hidden.shape)}")
        if self.tie_output_head:
            return torch.stack(
                [self.output_head(hidden + head(hidden)) for head in self.heads], dim=1
            )
        return torch.stack([head(hidden) for head in self.heads], dim=1)

    def head_logits(self, index: int, hidden: torch.Tensor) -> torch.Tensor:
        """Head `index`'s own vocab-sized logits, either mode. Under
        `tie_output_head=True`, indexing `self.heads[index]` directly gives
        only the small `d_model -> d_model` adapter's raw output (measured
        2026-09-28: feeding that into `cross_entropy` against a real token
        id crashes CUDA's `nll_loss` with an out-of-range class index,
        since its width is `d_model`, not `vocab_size`) -- this is the one
        correct way to train or evaluate a single head in isolation
        (`s7_mtp_train.py`'s per-head loop) without materializing all `k`
        heads via `forward`."""
        head = self.heads[index]
        if self.tie_output_head:
            return self.output_head(hidden + head(hidden))
        return head(hidden)

    def propose(self, hidden: torch.Tensor) -> tuple[int, ...]:
        """Greedy argmax per head over a single (`batch == 1`) hidden state,
        lowest-id tie-break -- `run_speculative_decode`'s draft contract
        (`SpeculativeDrafter.draft`) wants a plain token tuple, not logits."""
        if hidden.shape[0] != 1:
            raise ValueError(f"propose is batch=1 only, got batch {hidden.shape[0]}")
        logits = self.forward(hidden)[0]  # (k, vocab_size)
        proposals: list[int] = []
        for row in logits:
            best_score = row.max()
            tied = (row == best_score).nonzero(as_tuple=True)[0]
            proposals.append(int(tied.min().item()))
        return tuple(proposals)


def mtp_aux_loss(logits: torch.Tensor, targets: tuple[int, ...]) -> torch.Tensor:
    """Cross-entropy over `MtpHeads.forward`'s `(1, k, vocab_size)` output
    against up to `k` future target token ids.

    `targets` may carry fewer than `k` entries (e.g. near the end of a
    decode) -- only the heads with a real target contribute; a caller with
    zero targets gets a `0.0` loss rather than an empty-tensor `nan` (a
    missing future position is never encoded as a numeric zero target,
    matching CLAUDE.md's masking rule -- it is dropped from the sum
    entirely, not scored against target id `0`).

    NOT registered in `training_materializer.LOSS_FAMILY_IDS` -- RFC004
    §37.3 auxiliary MTP training is deliberately outside the primary's
    multitask loss; see :func:`unflatten.model.speculative.s7_mtp_train
    .run_mtp_stub_step` for the (dormant, primary-frozen) trainer that
    calls this."""
    if logits.dim() != 3 or logits.shape[0] != 1:
        raise ValueError(f"logits must be (1, k, vocab_size), got shape {tuple(logits.shape)}")
    k = logits.shape[1]
    if len(targets) > k:
        raise ValueError(f"got {len(targets)} targets for only {k} heads")
    if not targets:
        return logits.new_zeros(())
    target_ids = torch.tensor(targets, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits[0, : len(targets), :], target_ids)

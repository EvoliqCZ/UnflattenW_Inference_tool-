"""Document Context Encoder -- closes the M3 carried-forward blocker: the
`document_context_encoder` trainable-group-registry entry
(`trainable_group_registry.py`, kind `trainable_primary_s1`, one owned
tensor identity `document_context_encoder/base_weights`) named a group with
no module (`trainability_smoke.py`'s "acknowledged scope gap" paragraph).
RFC005 §14.4 gate 3 requires every trainable *relevant* group to show a
finite non-zero gradient; a registry entry with no module can never pass
that gate, and M3's task brief is explicit that leaving it unbuilt "fails
gate 3 on the day the canary runs" -- silently, since nothing else would
notice a group that simply never trained.

**Design is an ASSUMPTION, not a DEC-007 pin.** Design §6's table gives this
group exactly one line of contract: "global page/document state", trainable.
Neither the design doc nor DEC-007 specifies its input/output shape, so this
module makes the smallest defensible choice that (a) gives it real,
trainable parameters distinct from `adapter.projection`'s (VisualMemory) and
`expert_adapter.dense`'s, (b) produces something usable as an additional
decoder-input prefix segment (mirroring how `VisualMemory` already injects
frozen-backbone context as a prefix), and (c) does not invent any
Qwen-specific shape (no hardcoded hidden dimension, per CLAUDE.md).

Learned-query cross-attention pooling over the (already-projected)
`VisualMemory` output: `context_tokens` trainable query vectors attend over
the visual memory sequence and its padding mask, producing a small, fixed
number of "global document state" summary positions -- capacity for
document-level facts (e.g. overall layout shape) that a purely per-token
decoder stream has no dedicated place to accumulate. If this shape turns out
wrong once M4's typed heads need to read something specific from it, this
is the module to revisit -- nothing downstream depends on its internal
pooling mechanism, only on `forward`'s shape contract.

**RFC004 §16's actual purpose is cross-page: this module's main purpose is
UNEXERCISED at S1, and that is a fact to record, not to hide.** §16 exists
for semantics that cross a page boundary -- paragraph/table/list
continuation, section boundaries, repeated typography -- built from
"ordered global page tokens -> bounded document-context transformer -> page
context vectors -> injected into every local reconstruction window". V0's
single-page canary has exactly one page's global tokens and exactly one
reconstruction window, so there is no cross-page information for this
module to carry: it summarises the one page it is given and trains on
that, which is genuinely useful (§17.2's shared node-anchor construction
reads document context alongside decoder hidden state and node type,
RFC004 ~line 1372), but the entire reason §16 exists has never run a single
time by the end of S1. The first real test of continuation semantics is
whatever stage first admits multi-page input -- do not read a passing S1
canary as evidence this module's cross-page behaviour works.

**§16's four prohibitions, and which one a decode loop can violate by
accident.** This module must not create `Page` nodes (it does not -- it
returns bare tensors, no node type), must not change pixel ownership (it
only reads `VisualMemory`'s already-projected output, never touches
resource/pixel bytes), must not carry ground-truth fragment metadata (its
`forward` signature has no fragment/label parameter to carry one through),
and must not use future target structure as input. The last one is the one
that is easy to get wrong not in this module but in how a CALLER wires it:
if `document_context_encoder` were ever computed from something the decoder
has not yet causally seen (e.g. the full target token sequence rather than
`VisualMemory`'s frozen-backbone output), the causal decoder's later
composite-range positions would end up conditioned on a summary of tokens
they have not generated yet. `forward` structurally cannot see targets (its
only inputs are `memory`/`key_padding_mask`), and
`test_document_context_causal_isolation.py` proves this holds through the
full `candidate_model.encode_hidden` pipeline too -- see that test's
docstring for the mutation this guard is checked against.

This module authorizes no gradient update: it declares an `nn.Module` with
trainable parameters, but nothing here runs `.backward()` or builds an
optimizer.
"""

from __future__ import annotations

import torch
from torch import nn

#: The one `trainable_group_registry` group id this module's parameters
#: belong to (owns `document_context_encoder/base_weights`).
DOCUMENT_CONTEXT_ENCODER_GROUP_ID = "document_context_encoder"


class DocumentContextEncoder(nn.Module):
    """`(projected visual memory, padding mask) -> context_tokens summary positions`.

    Output shape `(batch, context_tokens, d_model)`, the same last-axis
    width as `VisualMemory`'s output, so a caller can concatenate the two
    into one decoder-input prefix without any further projection.
    """

    def __init__(
        self, *, d_model: int, context_tokens: int = 4, output_scale: float | None = None
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if context_tokens <= 0:
            raise ValueError(f"context_tokens must be positive, got {context_tokens}")
        if output_scale is not None and not output_scale > 0:
            raise ValueError(f"output_scale must be positive, got {output_scale}")
        self.d_model = d_model
        self.context_tokens = context_tokens
        # A single trainable query bank, not per-batch-item state: every
        # sample in a batch starts pooling from the same learned queries.
        self.query = nn.Parameter(torch.randn(context_tokens, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        # `output_scale` (2026-10-06): the initial LayerNorm gain, i.e. the RMS of
        # the context tokens at step 0. They are fed to the decoder next to native
        # token embeddings, so a caller sets it to that embedding table's RMS; the
        # unit default put them ~46x above Qwen's embeddings and the untrained model
        # read text worse WITH the image than without it (fresh S4 preflight).
        # Filling an existing tensor draws no RNG, so init order is unchanged.
        if output_scale is not None:
            with torch.no_grad():
                self.norm.weight.fill_(output_scale)

    def forward(
        self,
        memory: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`memory`: `(batch, tokens, d_model)`, e.g. `VisualMemory`'s output.
        `key_padding_mask`: `(batch, tokens)`, True == padding, matching
        `VisualMemory.forward`'s own convention -- passed straight through to
        `nn.MultiheadAttention`, which uses the identical polarity.
        """
        if memory.dim() != 3 or memory.shape[-1] != self.d_model:
            raise ValueError(
                "memory must be (batch, tokens, d_model) with last dim "
                f"{self.d_model}, got shape {tuple(memory.shape)}"
            )
        batch = memory.shape[0]
        queries = self.query.unsqueeze(0).expand(batch, -1, -1)
        pooled, _weights = self.attn(
            queries, memory, memory, key_padding_mask=key_padding_mask, need_weights=False
        )
        return self.norm(queries + pooled)

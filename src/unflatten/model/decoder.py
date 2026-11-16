"""Structural decoder: visual features → RFC004 event tokens.

The first learned component of the document branch. It cross-attends to a
frozen backbone's visual features and autoregressively emits the §18.2
event alphabet, which `model.encode` has already proven the dataset's
structural stream projects onto exactly.

`visual_dim` is a constructor argument and appears nowhere else: RFC004
requires the backbone to be swappable, and the measured candidates differ
(1024, 1152, 1536, 8704). A hidden size baked into the decoder would make
the interface a lie.

Frozen features are *conditioned* before they are used, in two
places, because two different things are wrong with them. Measured on
the `ab-multiscale` cache (8704 dims): the raw values have std 11.4 and
reach 2464, and one 1024-wide block carries per-channel std up to 457
while the other seven sit near 0.25. So:

- **per-channel standardisation before the projection**, from statistics
  of the cache itself, held in buffers. A per-token LayerNorm would fix
  the overall scale and leave the 3500x per-channel disparity, which
  means the loud block dominates `visual_projection` at init and the
  quiet scales — the entire point of a multiscale cache — arrive
  attenuated into an input covariance no optimiser fixes quickly. The
  statistics are FIXED, not learned: the source is frozen, so its scale
  is a constant of the dataset, and a learned gain could re-introduce
  the ill-conditioning it was there to remove.
- **a LayerNorm on the projected memory**, so what reaches cross-
  attention is unit-scale whatever the source. In a `norm_first` stack
  the memory would normally be an encoder's final norm output; here it
  comes from outside, and `norm_first` normalises the queries, not the
  memory. Without it, unnormalised memory enters the key/value
  projections directly and the random-features null model — tidy
  N(0, 1) — is not the same model with the information removed but a
  better-conditioned one, which makes it useless as a baseline.

Measured at the memory boundary on 8 `ab-multiscale` pages, d_model 512:
real memory std 8.08 (absmax 34.9) against the null model's 0.578 before,
1.000 and 1.000 after. Block contributions to that memory, loud 1024 dims
against the remaining 7680, go from a variance ratio of 3784 to 0.13 —
which is just their width ratio, i.e. the scales now contribute in
proportion to how much of the feature vector they are.

The loss is where the grammar earns its place. `masked_token_loss`
normalises only over grammar-legal tokens, so an illegal token cannot
absorb probability mass and the model never spends capacity learning not
to emit what the decoder will refuse anyway. Two failure modes are turned
into errors rather than silent damage:

- a mask that forbids the true target makes the step unlearnable — the
  correct answer has no probability to gain, so the gradient pushes
  nowhere useful. That is a contract break between grammar and data.
- a batch with no scored positions returns zero, not NaN; a document
  whose targets are entirely padding must not poison an epoch.
"""

from __future__ import annotations

import torch
from torch import nn

from unflatten.model.events_vocab import TOKEN_IDS
from unflatten.model.targets import IGNORE_INDEX

VOCAB_SIZE = len(TOKEN_IDS)


class StructureDecoder(nn.Module):
    """Autoregressive token decoder conditioned on visual features."""

    def __init__(
        self,
        *,
        visual_dim: int,
        d_model: int = 512,
        layers: int = 6,
        heads: int = 8,
        ff_multiplier: int = 4,
        max_steps: int = 8192,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.d_model = d_model
        self.max_steps = max_steps
        # The only place the backbone's width is mentioned.
        self.visual_projection = nn.Linear(visual_dim, d_model)
        # Identity until `set_feature_stats`; persistent, so a checkpoint
        # carries the conditioning it was trained under.
        self.register_buffer("feature_mean", torch.zeros(visual_dim))
        self.register_buffer("feature_std", torch.ones(visual_dim))
        self.memory_norm = nn.LayerNorm(d_model)
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.position_embedding = nn.Embedding(max_steps, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE)

    def set_feature_stats(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        """Install the frozen source's per-channel mean and sd.

        A constant channel keeps scale 1, matching `probe.heads.standardize`:
        it becomes all-zero rather than NaN.
        """
        for name, value in (("mean", mean), ("std", std)):
            if value.shape != (self.visual_dim,):
                raise ValueError(
                    f"feature {name} has shape {tuple(value.shape)}, expected "
                    f"({self.visual_dim},)"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"feature {name} contains non-finite values")
        if bool((std < 0).any()):
            raise ValueError("feature std contains negative values")
        safe = torch.where(std > 1e-8, std, torch.ones_like(std))
        self.feature_mean.copy_(mean.to(self.feature_mean.dtype))
        self.feature_std.copy_(safe.to(self.feature_std.dtype))

    def project_memory(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Cross-attention memory: standardise, project, normalise.

        Padding rows do not survive this as zeros — they never did, the
        projection has a bias — which is why the memory padding mask is
        not optional.
        """
        standardized = (visual_features - self.feature_mean) / self.feature_std
        return self.memory_norm(self.visual_projection(standardized))

    def forward(
        self,
        visual_features: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`(batch, steps, vocab)` logits.

        `visual_features` is `(batch, tokens, visual_dim)`; `input_ids` is
        the teacher-forcing input from `targets.decoder_targets`.

        `memory_key_padding_mask` is `(batch, tokens)`, True where a
        visual token is padding. It is not optional in practice: pages
        differ in token count, and `visual_projection` has a bias, so a
        zero pad row does not project to zero — it projects to the
        learned bias and cross-attends as real content. Without the mask
        a page's logits depend on what it was batched with.
        """
        steps = input_ids.shape[1]
        if steps > self.max_steps:
            raise ValueError(
                f"{steps} steps exceeds max_steps {self.max_steps}"
            )
        positions = torch.arange(steps, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        memory = self.project_memory(visual_features)
        causal = nn.Transformer.generate_square_subsequent_mask(
            steps, device=input_ids.device
        )
        hidden = self.decoder(
            hidden,
            memory,
            tgt_mask=causal,
            tgt_is_causal=True,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.head(self.norm(hidden))


def masked_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    allowed: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Cross-entropy over grammar-legal tokens only.

    `allowed` is a `(batch, steps, vocab)` boolean mask from
    `targets.allowed_mask_sequence`. Illegal tokens are driven to
    -infinity before the softmax, so they carry no probability and the
    normalisation runs over the legal set.
    """
    scored = targets != ignore_index
    if not bool(scored.any()):
        return logits.sum() * 0.0

    flat_logits = logits[scored]
    flat_targets = targets[scored]
    flat_allowed = allowed[scored]

    target_allowed = flat_allowed.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
    if not bool(target_allowed.all()):
        bad = int((~target_allowed).sum())
        raise ValueError(
            f"the grammar mask forbids the true target at {bad} scored "
            "position(s); grammar and data have drifted"
        )

    masked = flat_logits.masked_fill(~flat_allowed, float("-inf"))
    return nn.functional.cross_entropy(masked, flat_targets)

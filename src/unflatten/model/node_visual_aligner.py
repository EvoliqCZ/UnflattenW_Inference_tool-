"""``NodeVisualAligner`` head — anchor → visual-token cross-attention.

Design: ``docs/plans/spec-node-visual-aligner-design.md`` §5, variant (d) —
the design's chosen output: one layer of multi-head attention from the
node's decode anchor onto the page's visual hidden states, with a 2-D
(row/col) position bias on the keys, then a ppm box regression plus a
3-class alignment status (RFC004 §21.1/§21.2/§21.5).

Ownership contract (pinned by ``test_node_visual_aligner.py`` and the
``node_visual_aligner`` trainable-group entry):

- exactly four top-level children — ``query``, ``attention``,
  ``bbox_head``, ``status_head`` — 5 265 927 parameters at d_model 2560;
- ``attention`` holds the key/value/out projections, the zero-initialized
  ``(row, col)`` position embeddings and the affine-free LayerNorm;
- ``decode_bbox`` maps pre-sigmoid logits to a half-open integer ppm box
  that ``region_codec._validate_bbox`` accepts for **every** input —
  abstention is a status class, never a degenerate box.

Inputs come from the candidate model's forward (the same visual hidden
states ``gather_anchor_hidden`` indexes ``visual_steps`` before the
anchor, ``s4_typed_loss.py``); gold geometry never enters this module
(design §21.4, mutant R5).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from unflatten.model.region_codec import PPM_MAX

#: Default alignment width (design §5.1(d): d_a = 512).
DEFAULT_D_ALIGN = 512

#: Default number of attention heads (design §5.1(d)).
DEFAULT_NUM_HEADS = 8

#: The trained status classes, in this exact order (design §5.2/Q6).
#: ``abstain_degenerate`` is deliberately NOT a class: it is a decode-time
#: diagnosis for a decoded box that fails the codec, never a prediction.
STATUS_VOCABULARY: tuple[str, ...] = (
    "region",
    "abstain_not_visible",
    "abstain_container",
)
STATUS_COUNT = len(STATUS_VOCABULARY)

#: Versioned coordinate/output contract consumed by the aligner loss and
#: decoder.  Keep this separate from crop-coordinate policy: this names the
#: model's logical half-open ppm box, not a raster conversion.
BBOX_PPM_VERSION = "bbox_ppm.v0"


@dataclass(frozen=True)
class NodeVisualAlignerOutput:
    """One node's alignment hypothesis, pre-decode.

    ``bbox_logits`` are pre-sigmoid ppm logits for ``(cx, cy, w, h)``;
    ``status_logits`` are one logit per ``STATUS_VOCABULARY`` class.
    """

    bbox_logits: torch.Tensor  # (batch, 4)
    status_logits: torch.Tensor  # (batch, STATUS_COUNT)


class _AlignerAttention(nn.Module):
    """Single-layer multi-head cross-attention, anchor → visual tokens.

    The 2-D position bias (row/col, normalized 0..1 from ``grid_thw``) is
    added to the keys: without it the attention cannot tell *where* a
    token sits (design §5.1(d)). The final ``LayerNorm`` is affine-free —
    the bias belongs to the attention itself, not a post-hoc re-centering.
    """

    def __init__(self, d_model: int, d_align: int) -> None:
        super().__init__()
        self.key = nn.Linear(d_model, d_align)
        self.value = nn.Linear(d_model, d_align)
        self.out = nn.Linear(d_align, d_model)
        self.pos_row = nn.Parameter(torch.zeros(d_align))
        self.pos_col = nn.Parameter(torch.zeros(d_align))
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)


class NodeVisualAlignerHead(nn.Module):
    """Cross-attention head from a node anchor to the page's visual tokens.

    ``forward(anchor, visual_hidden, grid_thw)`` where

    - ``anchor`` is ``(batch, d_model)`` — the node's decode anchor
      (``float``);
    - ``visual_hidden`` is ``(batch, seq, d_model)`` — the page's visual
      hidden states, at least as long as the packed visual step count
      (``float``);
    - ``grid_thw`` is ``(1, 3)`` or ``(batch, 3)`` integer ``[t, h, w]``
      patch grids, ``t == 1``, ``h``/``w`` even — the packed step count of
      a row is ``(h // 2) * (w // 2)``.

    Returns a :class:`NodeVisualAlignerOutput`. Tokens at or past a row's
    packed step count are masked out of the keys and cannot influence the
    output. Every invalid shape is a fail-closed ``ValueError``.

    With ``visual_row`` (``(batch,)`` integer), ``visual_hidden`` and
    ``grid_thw`` carry ``R`` rows instead of ``batch`` and anchor ``i``
    attends to visual row ``visual_row[i]`` -- keys/values are computed once
    per visual row, not once per anchor. The result equals the forward over
    ``visual_hidden[visual_row]``; without it a page with ~600 regions
    copied its 8 580-step visual span per region (38.8 GiB, S4 v15b 24.9.).
    """

    def __init__(
        self,
        d_model: int,
        d_align: int = DEFAULT_D_ALIGN,
        num_heads: int = DEFAULT_NUM_HEADS,
    ) -> None:
        super().__init__()
        if d_align <= 0 or d_align % num_heads != 0:
            raise ValueError(
                f"d_align {d_align} must be a positive multiple of "
                f"num_heads {num_heads}"
            )
        self.d_model = d_model
        self.d_align = d_align
        self.num_heads = num_heads
        self.head_dim = d_align // num_heads
        self.query = nn.Linear(d_model, d_align)
        self.attention = _AlignerAttention(d_model, d_align)
        self.bbox_head = nn.Linear(d_model, 4)
        self.status_head = nn.Linear(d_model, STATUS_COUNT)

    def _validate_inputs(
        self,
        anchor: torch.Tensor,
        visual_hidden: torch.Tensor,
        grid_thw: torch.Tensor,
        visual_row: torch.Tensor | None = None,
    ) -> int:
        """Fail-closed input validation; returns the max packed step count."""
        if anchor.dim() != 2 or anchor.shape[1] != self.d_model:
            raise ValueError(
                f"anchor must be (batch, {self.d_model}), "
                f"got shape {tuple(anchor.shape)}"
            )
        if not anchor.is_floating_point():
            raise ValueError(
                f"anchor must be a floating-point tensor, got {anchor.dtype}"
            )
        if visual_hidden.dim() != 3 or visual_hidden.shape[2] != self.d_model:
            raise ValueError(
                f"visual_hidden must be (batch, seq, {self.d_model}), "
                f"got shape {tuple(visual_hidden.shape)}"
            )
        if not visual_hidden.is_floating_point():
            raise ValueError(
                "visual_hidden must be a floating-point tensor, "
                f"got {visual_hidden.dtype}"
            )
        rows = visual_hidden.shape[0]
        if visual_row is None:
            if rows != anchor.shape[0]:
                raise ValueError(
                    f"anchor batch {anchor.shape[0]} and visual_hidden batch "
                    f"{rows} must match"
                )
        elif (
            visual_row.dim() != 1
            or visual_row.shape[0] != anchor.shape[0]
            or visual_row.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                f"visual_row must be a ({anchor.shape[0]},) integer tensor, got "
                f"shape {tuple(visual_row.shape)} dtype {visual_row.dtype}"
            )
        elif bool((visual_row < 0).any()) or bool((visual_row >= rows).any()):
            raise ValueError(
                f"visual_row values must lie in [0, {rows}), got {visual_row.tolist()}"
            )
        seq = visual_hidden.shape[1]
        if seq == 0:
            raise ValueError("visual_hidden must contain at least one token")
        if grid_thw.dim() != 2 or grid_thw.shape[1] != 3:
            raise ValueError(
                "grid_thw must be (1, 3) or (batch, 3) integer [t, h, w], "
                f"got shape {tuple(grid_thw.shape)}"
            )
        if grid_thw.shape[0] not in (1, rows):
            raise ValueError(
                f"grid_thw row count {grid_thw.shape[0]} must be 1 or "
                f"batch {rows}"
            )
        if grid_thw.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                f"grid_thw must be an integer tensor, got {grid_thw.dtype}"
            )
        grid = grid_thw if grid_thw.shape[0] == rows else grid_thw.expand(rows, 3)
        t, h, w = grid.unbind(-1)
        if not torch.all(t == 1):
            raise ValueError(
                "grid_thw t (temporal patches) must be 1 for every row, "
                f"got {t.tolist()}"
            )
        if not torch.all(h > 0) or not torch.all(w > 0):
            raise ValueError(
                "grid_thw h and w must be positive, "
                f"got h={h.tolist()} w={w.tolist()}"
            )
        if not torch.all(h % 2 == 0) or not torch.all(w % 2 == 0):
            raise ValueError(
                "grid_thw h and w must be even (2x2 token merge), "
                f"got h={h.tolist()} w={w.tolist()}"
            )
        steps = (h // 2) * (w // 2)
        max_steps = int(steps.max().item())
        if max_steps > seq:
            raise ValueError(
                f"grid_thw requires {max_steps} packed visual steps but "
                f"visual_hidden has {seq}"
            )
        return max_steps

    def forward(
        self,
        anchor: torch.Tensor,
        visual_hidden: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        visual_row: torch.Tensor | None = None,
    ) -> NodeVisualAlignerOutput:
        max_steps = self._validate_inputs(anchor, visual_hidden, grid_thw, visual_row)
        batch = anchor.shape[0]
        rows = visual_hidden.shape[0]
        heads, head_dim = self.num_heads, self.head_dim
        visual = visual_hidden[:, :max_steps]

        q = self.query(anchor).view(batch, heads, 1, head_dim)
        k = self.attention.key(visual)
        k = k.view(rows, max_steps, heads, head_dim).transpose(1, 2)
        v = self.attention.value(visual)
        v = v.view(rows, max_steps, heads, head_dim).transpose(1, 2)

        # Per-row 2-D position bias: tokens are laid out row-major over the
        # merged (h/2, w/2) grid, each axis normalized to 0..1 (a
        # degenerate axis of size 1 normalizes to 0).
        grid = grid_thw if grid_thw.shape[0] == rows else grid_thw.expand(rows, 3)
        h, w = grid[:, 1], grid[:, 2]
        hm, wm = h // 2, w // 2
        idx = torch.arange(max_steps, device=k.device, dtype=torch.long)
        row = (idx[None, :] // wm[:, None]).float()  # (batch, steps)
        col = (idx[None, :] % wm[:, None]).float()  # (batch, steps)
        row_n = row / torch.clamp(hm.float() - 1, min=1.0)[:, None]
        col_n = col / torch.clamp(wm.float() - 1, min=1.0)[:, None]
        # Indices/normalization are computed in float32 for exactness, then
        # cast to the computation dtype (bf16/fp16 passes must not promote).
        pos_bias = (
            row_n[..., None] * self.attention.pos_row[None, None, :]
            + col_n[..., None] * self.attention.pos_col[None, None, :]
        ).to(k.dtype).view(rows, max_steps, heads, head_dim)
        k = k + pos_bias.transpose(1, 2)

        # Key mask: a row's tokens at or past its packed step count are
        # absent, never "zero values" — softmax over -inf is exactly 0.
        steps = (h // 2) * (w // 2)
        valid = idx[None, :] < steps[:, None]
        if visual_row is not None:
            ctx = self._shared_row_context(q, k, v, valid, visual_row)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)
            if not bool(valid.all()):
                scores = scores.masked_fill(
                    ~valid[:, None, None, :], float("-inf")
                )
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v)
        z = self.attention.norm(
            self.attention.out(ctx.view(batch, self.d_align)) + anchor
        )
        return NodeVisualAlignerOutput(
            bbox_logits=self.bbox_head(z),
            status_logits=self.status_head(z),
        )

    def _shared_row_context(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid: torch.Tensor,
        visual_row: torch.Tensor,
    ) -> torch.Tensor:
        """Attention context for anchors that share visual rows: for each
        row, all of its anchors' queries attend to that row's keys/values in
        one matmul. Memory is anchors x steps per head, never anchors x
        steps x d_align. Returns ``(batch, heads, 1, head_dim)`` like the
        per-anchor path."""
        ctx = q.new_zeros(q.shape)
        for r in range(k.shape[0]):
            sel = (visual_row == r).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            q_r = q.index_select(0, sel).squeeze(2).transpose(0, 1)  # (heads, n_r, hd)
            scores = torch.matmul(q_r, k[r].transpose(-2, -1)) * (self.head_dim**-0.5)
            if not bool(valid[r].all()):
                scores = scores.masked_fill(~valid[r][None, None, :], float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            ctx_r = torch.matmul(attn, v[r]).transpose(0, 1).unsqueeze(2)
            ctx = ctx.index_copy(0, sel, ctx_r)
        return ctx


def decode_bbox(bbox_logits: torch.Tensor) -> torch.Tensor:
    """Decode pre-sigmoid ppm logits to half-open integer boxes.

    ``bbox_logits`` is ``(batch, 4)`` with columns ``(cx, cy, w, h)``:
    ``v = round(sigmoid(logits) * 10⁶)``, ``w``/``h`` clamped to at least
    1 ppm, and the center clamped so the half-open box
    ``[x1, y1, x2, y2)`` stays within ``[0, 10⁶]``. The center/size
    parametrization guarantees ``x1 < x2`` for ``w > 0`` (design
    §5.1(d)); the result is always codec-legal.
    """
    if bbox_logits.dim() != 2 or bbox_logits.shape[1] != 4:
        raise ValueError(
            "bbox_logits must be (batch, 4) [cx, cy, w, h], "
            f"got shape {tuple(bbox_logits.shape)}"
        )
    v = (torch.sigmoid(bbox_logits.float()) * PPM_MAX).round().to(torch.long)
    cx, cy, w, h = v.unbind(-1)
    w = w.clamp(min=1)
    h = h.clamp(min=1)
    x1 = torch.minimum((cx - w // 2).clamp(min=0), PPM_MAX - w)
    y1 = torch.minimum((cy - h // 2).clamp(min=0), PPM_MAX - h)
    return torch.stack([x1, y1, x1 + w, y1 + h], dim=-1)

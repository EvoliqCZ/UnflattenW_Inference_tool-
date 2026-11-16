"""Per-level taps of a frozen Qwen3-VL vision tower (ViT-level probe).

HF `Qwen3VLVisionModel.forward` (transformers 5.x) runs its 24 blocks and,
AFTER block `i` for every `i` in `config.deepstack_visual_indexes`, hands
that block's output to `deepstack_merger_list[k]` (post-shuffle LayerNorm
over the 4 patches of a 2x2 merge, then fc1 -> GELU -> fc2 to
`out_hidden_size`). Those are the features DeepStack adds at the visual
positions of decoder layers 0, 1, 2. `pooler_output` is the last block's
output through the ordinary `merger` and is what our runtime has consumed
since 2026-09-04.

Representations per merged token:

- `ds{i}`: the DeepStack feature from block `i` (exactly HF's
  `deepstack_features[k]`);
- `raw{i}`: block `i`'s own output, the 4 patches of the merge
  concatenated (the merger's input before its norm), for every `i` in
  `raw_layers`;
- `pooler`: `pooler_output`.

The processor orders patches so that every 4 consecutive rows form one
2x2 merge (the mergers' `view(-1, 4 * hidden)` depends on that), so row
`t` of every representation is the same merged token.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def level_names(deepstack_indexes: Sequence[int], raw_layers: Sequence[int]) -> list[str]:
    return [f"ds{i}" for i in deepstack_indexes] + ["pooler"] + [f"raw{i}" for i in raw_layers]


@torch.inference_mode()
def tap_levels(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    raw_layers: Sequence[int],
) -> dict[str, torch.Tensor]:
    """One tower forward; every representation as (tokens, dim)."""
    depth = len(visual.blocks)
    for layer in raw_layers:
        if not 0 <= layer < depth:
            raise ValueError(f"raw layer {layer} outside the tower's {depth} blocks")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in raw_layers:

        def hook(_module, _inputs, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden

        handles.append(visual.blocks[layer].register_forward_hook(hook))
    try:
        out = visual(pixel_values, grid_thw=grid_thw)
    finally:
        for handle in handles:
            handle.remove()
    merge_unit = visual.spatial_merge_size**2
    indexes = list(visual.deepstack_visual_indexes)
    deepstack = list(out.deepstack_features or [])
    if len(deepstack) != len(indexes):
        raise RuntimeError(
            f"tower returned {len(deepstack)} deepstack levels for indexes {indexes}"
        )
    taps: dict[str, torch.Tensor] = {
        f"ds{i}": level for i, level in zip(indexes, deepstack, strict=True)
    }
    taps["pooler"] = out.pooler_output
    for layer in raw_layers:
        hidden = captured[layer]
        taps[f"raw{layer}"] = hidden.reshape(-1, hidden.shape[-1] * merge_unit)
    tokens = taps["pooler"].shape[0]
    for name, value in taps.items():
        if value.shape[0] != tokens:
            raise RuntimeError(f"{name} has {value.shape[0]} rows, pooler has {tokens}")
    return taps

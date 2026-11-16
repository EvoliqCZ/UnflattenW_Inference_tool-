"""Granite Vision 4.1: an AnyRes tower behind the one-grid probe contract.

Every other backbone in the registry hands the probe **one** patch grid per
page, and the whole geometry layer is built on that: `infer_merge_size`
derives the token pitch from it, `token_boxes` turns it into page
rectangles, `band_pooled` splits its rows. Granite does not work that way.
`LlavaNextImageProcessor` selects a resolution from `image_grid_pinpoints`,
slices the page into 384x384 tiles, and prepends a whole-page thumbnail —
for our 2550x3300 page that is a 3x3 tile grid plus the thumbnail, ten
independent 24x24 patch grids where the harness expects one.

The reduction taken here is the model's own. `pack_image_features` already
knows how to lay the tiles back into a single raster and crop the padding
AnyRes introduced; this module performs exactly that assembly (minus the
`image_newline` column, which is a token with no page rectangle) and
reports the resulting rectangle as `grid_thw`. So AnyRes is *reduced to*
the single-grid contract rather than the contract being widened to admit
it, and `MergeSizeError` keeps meaning what it meant: a page whose tokens
do not account for its reported geometry is refused, not guessed.

The thumbnail tile is dropped. It covers the same page at a sixth of the
resolution, so it has no rectangle of its own in the assembled grid; a
tenth tile pooled into a nine-tile raster would be a token whose position
is a fiction.

**Two taps, both reachable, neither the default.** Granite fuses vision
into the LLM through eight Window-Q-Former projectors — four `deepstack`
layers `(-19, -13, -7, -1)` and four spatial-offset groups off layer `-1`
— and each projector emits into the *language* width (2560), halving the
grid on the way (`downsample_rate = 4/8`). Measuring only that compares
Granite's projector stack against Qwen's bare DeepStack taps, so both are
offered:

- ``siglip4`` — the four raw SigLIP2 layers the projectors read, at
  1152 wide on the full patch grid. Capacity-normalised against Qwen.
- ``projected8`` — all eight projected streams at 2560 wide on the halved
  grid. What the model actually feeds its decoder.

`vision_feature_layer = -2` in the config is inherited from
`LlavaNextConfig` and is **dead**: `Granite4VisionModel.get_image_features`
reads `deepstack_layer_map` and `spatial_vision_layer` and never consults
it. Neither tap is a single layer.

Measured, on a 2550x3300 page from `m2a-v2` (300 dpi), not derived:

    processor      10 tiles of 384x384  (a 3x3 AnyRes grid at [1152, 1152]
                                         plus the whole-page thumbnail)
    siglip4        grid 72x56 = 4032 tokens, 4x1152 wide, 46 px per token
    projected8     grid 36x28 = 1008 tokens, 8x2560 wide, 91 px per token

For scale: Qwen3-VL at `--max-pixels 1003520` puts the same page on a
70x54 patch grid, 945 merged tokens at 94 px each. So the *comparable*
tap — the one whose tokens the decoder would attend over — is within a
tenth of Qwen's token count and its spatial pitch, not 5x either. The
extra resolution is real but it lives one level down, in `siglip4`.

Both grids are the model's own, so both inherit its rounding: `unpad_image`
keeps a whole number of tokens either side of centre, leaving the assembled
grid 1.8 % (siglip4) to 3.7 % (projected8) wider than the page content it
is mapped onto. What that costs is not uniform, and the difference matters
more than the number:

- **`pooled` and `band_pooled` are exactly unbiased.** For a portrait page
  `unpad_image` crops the WIDTH only, and `band_pooled` averages across
  columns before splitting ROWS — the biased axis is averaged away and the
  split axis is untouched. There is nothing to caveat for those two.
- **`token_boxes`, and so the spatial probe, is displaced.** The crop is
  centre-symmetric, so the error is zero at the page centre and grows
  linearly with distance from it, bounded by about half a cell at the
  margin.
- **It can only understate Granite.** Labels come from box overlap, so a
  sub-cell displacement flips only tokens already sitting on a content
  boundary; a head fitted on slightly misaligned edge tokens scores lower,
  never higher. The bias cannot manufacture a win — but a narrow spatial
  loss against Qwen, which carries no equivalent, is not decisive either.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from unflatten.probe.backbones import VisionRuntime

#: The checkpoint this module is written against.
GRANITE_MODEL_ID = "ibm-granite/granite-vision-4.1-4b"

#: Feature taps, in the order the plan asks them to be reported. Both, or
#: neither: quoting the better one is the defect this comparison exists
#: to remove.
GRANITE_TAPS: tuple[str, ...] = ("siglip4", "projected8")

#: Used when `--tap` is not given. The capacity-normalised tap, because a
#: default that silently picked the model's strongest representation would
#: make the fair comparison the one you have to ask for.
DEFAULT_GRANITE_TAP = "siglip4"


def assemble_anyres_tiles(
    tiles: torch.Tensor,
    image_size: tuple[int, int],
    pinpoints: list,
    tile_pixels: int,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Lay AnyRes tiles back into one page grid; drop the thumbnail.

    `tiles` is (num_tiles, tokens_per_tile, dim) with the whole-page
    thumbnail first, exactly as the vision tower returns it. `image_size`
    is the processor's `(height, width)` of the ORIGINAL page, which is
    what decides both the tile grid and how much of the last tile row and
    column is padding.

    Returns `(rows_of_tokens, (1, height, width))` — token rows in the
    row-major order `token_boxes` assumes, and the grid they fill.

    Raises `AnyResGeometryError` when the tile count, the token count and
    the reported page size do not agree. That is the same refusal
    `MergeSizeError` makes for a single grid, which is why it is a
    subclass of it: a caller that skips ungeometric pages skips these too,
    and a caller that treats a bad tap as fatal still does.
    """
    from transformers.models.llava_next.modeling_llava_next import (
        get_anyres_image_grid_shape,
        unpad_image,
    )

    from unflatten.probe.spatial import AnyResGeometryError

    if tiles.dim() != 3:
        raise AnyResGeometryError(
            f"AnyRes tiles must be (tiles, tokens, dim); got {tuple(tiles.shape)}"
        )
    count, per_tile, _ = tiles.shape
    side = int(math.isqrt(per_tile))
    if side * side != per_tile:
        raise AnyResGeometryError(
            f"a tile of {per_tile} tokens is not a square patch grid; "
            "the tile geometry cannot be reconstructed"
        )
    tile_rows, tile_cols = get_anyres_image_grid_shape(
        list(image_size), pinpoints, tile_pixels
    )
    if count != tile_rows * tile_cols + 1:
        raise AnyResGeometryError(
            f"a {tile_rows}x{tile_cols} AnyRes grid plus one thumbnail is "
            f"{tile_rows * tile_cols + 1} tiles, but the tower returned {count}"
        )

    grid = tiles[1:].reshape(tile_rows, tile_cols, side, side, -1)
    # (dim, tile_rows*side, tile_cols*side), the model's own ordering
    grid = grid.permute(4, 0, 2, 1, 3).flatten(1, 2).flatten(2, 3)
    grid = unpad_image(grid, image_size)
    height, width = int(grid.shape[1]), int(grid.shape[2])
    if height < 1 or width < 1:
        raise AnyResGeometryError(
            f"unpadding a {tile_rows}x{tile_cols} tile grid for a "
            f"{image_size[1]}x{image_size[0]} page left a {height}x{width} "
            "token grid; the page has no reconstructable geometry"
        )
    rows = grid.flatten(1, 2).transpose(0, 1).contiguous()
    if rows.shape[0] != height * width:
        raise AnyResGeometryError(
            f"assembled {rows.shape[0]} token rows for a {height}x{width} grid"
        )
    return rows, (1, height, width)


def anyres_token_grid(
    image_size: tuple[int, int], pinpoints: list, tile_pixels: int, side: int
) -> tuple[int, int, int]:
    """The grid `assemble_anyres_tiles` will produce, computed without it.

    Derived from the page size and the tile side alone — no tokens are
    consulted — so it can be checked AGAINST the token count rather than
    inferred from it. That is the direction `MergeSizeError` insists on: a
    grid the tower's output has to justify, not one read off it.
    """
    from transformers.models.llava_next.modeling_llava_next import (
        get_anyres_image_grid_shape,
        unpad_image,
    )

    tile_rows, tile_cols = get_anyres_image_grid_shape(
        list(image_size), pinpoints, tile_pixels
    )
    probe = torch.zeros(1, tile_rows * side, tile_cols * side)
    cropped = unpad_image(probe, image_size)
    return (1, int(cropped.shape[1]), int(cropped.shape[2]))


def single_page_size(image_sizes: Any) -> tuple[int, int]:
    """The one page's `(height, width)`, or a refusal.

    `assemble_anyres_tiles` reconstructs ONE page from ONE tile stack, and
    by the time the tower has run there is no batch axis left to tell the
    pages apart. Taking `[0]` out of a longer batch would drop every page
    after the first and return a grid describing only that one — a page
    count silently reduced, inside the module whose whole argument is that
    geometry it cannot reconstruct is refused rather than approximated.
    """
    from unflatten.probe.spatial import AnyResGeometryError

    if len(image_sizes) != 1:
        raise AnyResGeometryError(
            f"AnyRes assembly reconstructs one page at a time, but the batch "
            f"holds {len(image_sizes)} page sizes; batching here would drop "
            "all but the first"
        )
    return (int(image_sizes[0][0]), int(image_sizes[0][1]))


def single_page_tiles(pixel_values: torch.Tensor) -> torch.Tensor:
    """Drop the processor's batch axis, refusing a batch of real pages.

    The processor emits `(1, tiles, 3, H, W)`; the tower wants
    `(tiles, 3, H, W)`. A leading axis of one is the wrapper and comes
    off; anything longer is pages, and see `single_page_size`.
    """
    from unflatten.probe.spatial import AnyResGeometryError

    if pixel_values.dim() != 5:
        return pixel_values
    if pixel_values.shape[0] != 1:
        raise AnyResGeometryError(
            f"AnyRes assembly reconstructs one page at a time, but the batch "
            f"holds {pixel_values.shape[0]} pages of tiles; batching here "
            "would drop all but the first"
        )
    return pixel_values[0]


def _tile_features(
    hidden_states: tuple[torch.Tensor, ...], layer: int
) -> torch.Tensor:
    """One vision layer's per-tile tokens, as `get_image_features` reads it.

    Deliberately the raw hidden state and not `last_hidden_state`: the
    model's projectors read `hidden_states[layer]` before the tower's
    `post_layernorm`, so a tap that normalised would measure something the
    model never sees.
    """
    return hidden_states[layer]


def _siglip4_streams(
    vision_outputs: Any, config: Any
) -> tuple[torch.Tensor, ...]:
    """The four SigLIP layers the projectors read, deepest last."""
    layers = [int(vision) for vision, _ in config.deepstack_layer_map]
    return tuple(_tile_features(vision_outputs.hidden_states, layer) for layer in layers)


def _projected8_streams(
    vision_outputs: Any, model: Any, config: Any
) -> tuple[torch.Tensor, ...]:
    """All eight projected streams, on the halved grid, at LLM width.

    Runs the projectors on the tile features directly rather than reading
    `get_image_features`'s output: that packs, unpads AND interleaves
    `image_newline`, and a newline token has no page rectangle. The
    projections themselves are the same modules on the same inputs.
    """
    streams = []
    for index, (vision_layer, _) in enumerate(config.deepstack_layer_map):
        streams.append(
            model.layerwise_projectors[index](
                _tile_features(vision_outputs.hidden_states, int(vision_layer))
            )
        )
    spatial = _tile_features(
        vision_outputs.hidden_states, int(config.spatial_vision_layer)
    )
    for group in range(len(config.spatial_target_layers)):
        streams.append(model.spatial_projectors[group](spatial))
    return tuple(streams)


def assemble_streams(
    streams: tuple[torch.Tensor, ...],
    image_size: tuple[int, int],
    pinpoints: list,
    tile_pixels: int,
    tap: str,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Assemble every stream of a tap onto one shared page grid.

    Returns `(visual_features, multiscale_features)` in the order the
    probe interface expects: the deepest stream first, mirroring Qwen3-VL
    where `last_hidden_state` leads and the DeepStack levels follow.

    The streams must land on the SAME grid. They do by construction — one
    tap, one tile side — so a disagreement means a stream was taken from
    somewhere it does not belong, and one set of token boxes could no
    longer describe all of them.
    """
    from unflatten.probe.spatial import AnyResGeometryError

    assembled = [
        assemble_anyres_tiles(stream, image_size, pinpoints, tile_pixels)
        for stream in streams
    ]
    grids = {grid for _, grid in assembled}
    if len(grids) != 1:
        raise AnyResGeometryError(
            f"the {len(streams)} streams of tap {tap!r} assembled onto "
            f"{len(grids)} different grids ({sorted(grids)}); they cannot "
            "share one set of token boxes"
        )
    rows = [row for row, _ in assembled]
    return rows[-1], tuple(rows[:-1])


def checked_anyres_grid(
    image_size: tuple[int, int],
    pinpoints: list,
    tile_pixels: int,
    side: int,
    token_count: int,
    tap: str,
) -> tuple[int, int, int]:
    """`anyres_token_grid`, refused unless the tokens account for it.

    The grid is computed from the page alone and then has to be JUSTIFIED
    by the row count, which is the direction `MergeSizeError` exists to
    protect: a page whose geometry cannot be reconstructed is skipped, not
    approximated.
    """
    from unflatten.probe.spatial import AnyResGeometryError

    grid = anyres_token_grid(image_size, pinpoints, tile_pixels, side)
    if grid[1] * grid[2] != token_count:
        raise AnyResGeometryError(
            f"a {image_size[1]}x{image_size[0]} page assembles to a "
            f"{grid[1]}x{grid[2]} token grid, but tap {tap!r} produced "
            f"{token_count} rows"
        )
    return grid


def _tap_side(config: Any, tap: str) -> int:
    """Tokens per tile edge at `tap`.

    ``siglip4`` sits on the tower's own patch grid; ``projected8`` sits on
    the grid after the Window Q-Former's `downsample_rate`, which is the
    same fraction the model applies in `pack_image_features`.
    """
    from fractions import Fraction

    side = int(config.vision_config.image_size) // int(config.vision_config.patch_size)
    if tap == "projected8":
        side = int(side * Fraction(config.downsample_rate))
    return side


def load_granite(model_id: str, dtype: torch.dtype, tap: str | None) -> VisionRuntime:
    """Granite Vision 4.1: SigLIP2 tower plus its eight projectors.

    Natively supported by transformers, so no `trust_remote_code` and no
    `HF_MODULES_CACHE` despite the checkpoint shipping its own modelling
    files. The language half is dropped as everywhere else; under the
    ``siglip4`` tap the projectors go with it, since that tap never calls
    them and they are ~1.2 GB of Q-Former.
    """
    from transformers import AutoImageProcessor, AutoModelForImageTextToText

    from unflatten.probe.backbones import VisionRuntime, _drop_language_half

    tap = tap or DEFAULT_GRANITE_TAP
    if tap not in GRANITE_TAPS:
        raise ValueError(
            f"unknown Granite feature tap {tap!r}; known: {', '.join(GRANITE_TAPS)}"
        )

    wrapper = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
    wrapper.eval()
    model = wrapper.model
    config = wrapper.config
    _drop_language_half(wrapper)
    if tap == "siglip4":
        del model.layerwise_projectors
        del model.spatial_projectors

    pinpoints = config.image_grid_pinpoints
    tile_pixels = int(config.vision_config.image_size)

    def forward(pixel_values, image_sizes):
        size = single_page_size(image_sizes)
        outputs = model.vision_tower(
            single_page_tiles(pixel_values), output_hidden_states=True
        )
        if tap == "siglip4":
            streams = _siglip4_streams(outputs, config)
        else:
            streams = _projected8_streams(outputs, model, config)
        return assemble_streams(streams, size, pinpoints, tile_pixels, tap)

    def prepare(batch):
        return batch["pixel_values"], batch["image_sizes"]

    side = _tap_side(config, tap)

    def grid_of(batch, token_count):
        size = single_page_size(batch["image_sizes"])
        return checked_anyres_grid(
            size, pinpoints, tile_pixels, side, token_count, tap
        )

    # Counted from the config rather than written out: `pooled()`
    # concatenates one mean per stream, so a checkpoint with a different
    # `deepstack_layer_map` must change this width, not silently mislabel it.
    deepstack = len(config.deepstack_layer_map)
    if tap == "siglip4":
        feature_dim = deepstack * int(config.vision_config.hidden_size)
    else:
        spatial = len(config.spatial_target_layers)
        feature_dim = (deepstack + spatial) * int(config.text_config.hidden_size)
    modules = [model.vision_tower]
    if tap == "projected8":
        modules += [model.layerwise_projectors, model.spatial_projectors]
    return VisionRuntime(
        model_id=model_id,
        module=torch.nn.ModuleList(modules) if len(modules) > 1 else modules[0],
        processor=AutoImageProcessor.from_pretrained(model_id),
        forward=forward,
        feature_dim=feature_dim,
        param_count=sum(
            p.numel() for module in modules for p in module.parameters()
        ),
        tap=tap,
        prepare=prepare,
        grid_of=grid_of,
        # LlavaNext hands the tower whole 384x384 rasters, not flattened
        # patches: rewriting its patch conv as a matmul would reinterpret
        # image memory as patch rows and silently return garbage.
        flattened_patch_input=False,
    )

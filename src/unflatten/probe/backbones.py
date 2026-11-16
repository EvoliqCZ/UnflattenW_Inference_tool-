"""Backbone registry: loading a frozen document vision tower.

RFC004 requires the backbone to be swappable behind one interface, and
OPEN-004-001 leaves the product choice to a measured comparison. That
comparison needs every candidate loadable in the *same* environment,
which is most of what this module is: four model families that agree on
almost nothing about how their vision tower is reached or called.

What differs, concretely:

- **Where the tower lives.** Qwen3-VL puts it at `model.model.visual`;
  dots.mocr and MonkeyOCRv2 at `model.vision_tower`; PaddleOCR-VL has no
  usable wrapper at all in this transformers version, so its
  `PaddleOCRVisionModel` is loaded straight from the dynamic module and
  its weights taken from the `visual.` prefix of the checkpoint.
- **What the tower returns.** Qwen3-VL returns a structured output with
  DeepStack levels; the Qwen2-VL-derived towers return a bare tensor;
  PaddleOCR-VL returns hidden states only when asked.
- **What it needs to be importable.** dots.mocr's vision file imports
  `flash_attn` at module level even though it also ships an SDPA path.
  A stub satisfies the import without pretending the kernels exist — it
  raises if anything actually calls it.

Loading only the vision tower is deliberate: the language half is most of
each checkpoint and contributes nothing to a visual-feature probe.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

#: Candidates named by RFC004 OPEN-004-001 plus PaddleOCR-VL, in the
#: order the Stage-0 probe runs them. Order is not a prediction.
CANDIDATES: tuple[str, ...] = (
    "ibm-granite/granite-vision-4.1-4b",
    "PaddlePaddle/PaddleOCR-VL-1.6",
    "rednote-hilab/dots.mocr",
    "Qwen/Qwen3-VL-4B-Instruct",
    "zenosai/MonkeyOCRv2-B-Parsing",
)


def install_flash_attn_stub() -> bool:
    """Make `import flash_attn` succeed without claiming the kernels work.

    Returns True if a stub was installed. `__version__` is deliberately
    below 2.1 so transformers' own availability checks keep selecting
    SDPA; the imported function raises if it is ever actually called.
    """
    if "flash_attn" in sys.modules:
        return False
    try:
        import flash_attn  # noqa: F401

        return False
    except ImportError:
        pass

    stub = types.ModuleType("flash_attn")
    stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    stub.__version__ = "0.0.0"

    def _unavailable(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "flash_attn is not installed; this model must run with "
            "attn_implementation='sdpa'"
        )

    stub.flash_attn_varlen_func = _unavailable
    stub.flash_attn_func = _unavailable
    sys.modules["flash_attn"] = stub
    return True


def as_token_rows(features: torch.Tensor) -> torch.Tensor:
    """Normalise a tower's output to (tokens, dim).

    Towers disagree: Qwen3-VL returns (tokens, dim) while PaddleOCR-VL
    keeps a leading batch axis. Pooling a (1, tokens, dim) tensor over
    dim 0 silently yields a matrix instead of a vector, which only shows
    up later as a stack-shape error.
    """
    if features.dim() == 3 and features.shape[0] == 1:
        return features.squeeze(0)
    return features


@dataclass(frozen=True)
class VisionRuntime:
    """A loaded frozen vision tower plus how to call it."""

    model_id: str
    module: torch.nn.Module
    processor: Any
    #: (pixel_values, grid_thw|None) -> (last_hidden, multiscale levels)
    forward: Callable[..., tuple[torch.Tensor, tuple[torch.Tensor, ...]]]
    feature_dim: int
    param_count: int
    #: Which feature tap was loaded, for backbones that expose more than
    #: one. Part of the probe's cache identity: two taps of one checkpoint
    #: are two different feature sets under the same model id.
    tap: str | None = None
    #: Optional processor-batch adapter, `batch -> (pixel_values, aux)`.
    #: `None` keeps the `pixel_values` / `image_grid_thw` path every
    #: single-grid backbone uses, byte for byte.
    prepare: Callable[[Any], tuple[torch.Tensor, Any]] | None = None
    #: Optional grid derivation, `(batch, token_count) -> (t, h, w)`, for
    #: backbones whose grid is not in the batch. It is CHECKED against the
    #: token count rather than inferred from it — see `MergeSizeError`.
    grid_of: Callable[[Any, int], tuple[int, int, int]] | None = None
    #: False when the tower is handed whole rasters rather than flattened
    #: patches, which makes `optimize_patch_embed`'s matmul rewrite wrong
    #: rather than merely unnecessary.
    flattened_patch_input: bool = True


def _drop_language_half(model: torch.nn.Module) -> None:
    """Free the language model before anything reaches the GPU."""
    inner = getattr(model, "model", model)
    for attr in ("language_model", "embed_tokens", "layers", "norm", "rotary_emb"):
        if hasattr(inner, attr):
            try:
                delattr(inner, attr)
            except (AttributeError, TypeError):
                pass
    for attr in ("lm_head",):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except (AttributeError, TypeError):
                pass


def optimize_patch_embed(tower: torch.nn.Module) -> bool:
    """Rewrite the tower's patch convolution, wherever it lives.

    Qwen3-VL exposes it as `patch_embed.proj`; the Qwen2-VL-derived towers
    nest it one deeper as `patch_embed.patchifier.proj`. Must run AFTER
    the module is on its device: the rewrite closes over the weight
    tensor, so doing it first leaves a CPU tensor behind.
    """
    for name, module in tower.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if "patch" not in leaf:
            continue
        if isinstance(module, torch.nn.modules.conv._ConvNd):
            if _conv_as_matmul(module):
                return True
        elif patch_embed_as_matmul(module):
            return True
    return False


def patch_embed_as_matmul(patch_embed: Any) -> bool:
    """Replace a stride==kernel patch convolution with its matmul form.

    The convolution is a non-overlapping patch projection and the
    processors hand over already-flattened patches, so it is exactly
    `x @ W.reshape(out, -1).T + b`. Identical output; on builds with no
    fast kernel for it the convolution costs ~23 s per page and the
    matmul 0.2 ms. Returns False when the module has no such conv.
    """
    conv = getattr(patch_embed, "proj", None)
    if not isinstance(conv, torch.nn.modules.conv._ConvNd):
        return False
    return _conv_as_matmul(conv, host=patch_embed)


def _conv_as_matmul(conv: Any, host: Any = None) -> bool:
    """Install the matmul forward on `host` (default: the conv itself)."""
    patch_embed = host if host is not None else conv
    if conv.stride != conv.kernel_size:
        return False
    weight = conv.weight.reshape(conv.weight.shape[0], -1)
    bias = conv.bias

    features = weight.shape[1]

    def forward(hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        flat = hidden_states.reshape(-1, features).to(weight.dtype)
        out = flat @ weight.T
        if bias is not None:
            out = out + bias
        if hidden_states.dim() > 2:
            # Called in place of the convolution itself: the caller expects
            # conv output rank, with the (1, 1) spatial slot a stride==kernel
            # patch always produces. Returning the flat form silently
            # corrupts a downstream `.flatten(-2).squeeze(-1)`.
            out = out.reshape(*out.shape, *([1] * (hidden_states.dim() - 2)))
        return out

    patch_embed.forward = forward
    return True


def _one_tap_only(model_id: str, tap: str | None) -> None:
    """Refuse a tap for a backbone that exposes exactly one feature set."""
    if tap is not None:
        raise ValueError(
            f"{model_id} exposes a single feature tap, so {tap!r} cannot be "
            "selected; only Granite Vision takes a tap"
        )


def _load_qwen3vl(
    model_id: str, dtype: torch.dtype, tap: str | None = None
) -> VisionRuntime:
    """Qwen3-VL: tower at `model.model.visual`, DeepStack levels native."""
    _one_tap_only(model_id, tap)
    from transformers import AutoImageProcessor, AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
    model.eval()
    visual = model.model.visual
    _drop_language_half(model)

    def forward(pixel_values, grid_thw):
        out = visual(pixel_values, grid_thw=grid_thw)
        levels = tuple(
            as_token_rows(level) for level in getattr(out, "deepstack_features", ()) or ()
        )
        return as_token_rows(out.last_hidden_state), levels

    config = visual.config
    levels = len(getattr(config, "deepstack_visual_indexes", ()) or ())
    return VisionRuntime(
        model_id=model_id,
        module=visual,
        processor=AutoImageProcessor.from_pretrained(model_id),
        forward=forward,
        feature_dim=config.hidden_size + levels * config.out_hidden_size,
        param_count=sum(p.numel() for p in visual.parameters()),
    )


def _load_qwen2vl_style(
    model_id: str, dtype: torch.dtype, tap: str | None = None
) -> VisionRuntime:
    """dots.mocr and MonkeyOCRv2: `model.vision_tower`, bare tensor out."""
    _one_tap_only(model_id, tap)
    install_flash_attn_stub()
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForCausalLM

    # These towers pick their attention class from their OWN vision config
    # field (`DotsVisionBlock(config, config.attn_implementation)`), which
    # defaults to flash_attention_2. Setting that field is what matters;
    # passing `attn_implementation` to `from_pretrained` instead makes
    # transformers validate it against the custom class, which does not
    # declare support and refuses.
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, "vision_config"):
        config.vision_config.attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, dtype=dtype, trust_remote_code=True
    )
    model.eval()
    tower = model.vision_tower
    _drop_language_half(model)

    def forward(pixel_values, grid_thw):
        # MonkeyOCRv2 returns `(final_hidden_states, hidden_states)` where
        # dots.mocr returns a bare tensor; the first element is the output
        # in both cases. There is no multi-scale output to expose without
        # tapping blocks ourselves.
        out = tower(pixel_values, grid_thw)
        if isinstance(out, tuple):
            out = out[0]
        return as_token_rows(out), ()

    hidden = getattr(tower.config, "hidden_size", None) if hasattr(tower, "config") else None
    if hidden is None:
        hidden = model.config.vision_config.hidden_size
    return VisionRuntime(
        model_id=model_id,
        module=tower,
        processor=AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True),
        forward=forward,
        feature_dim=hidden,
        param_count=sum(p.numel() for p in tower.parameters()),
    )


def _load_paddleocr(
    model_id: str, dtype: torch.dtype, tap: str | None = None
) -> VisionRuntime:
    """PaddleOCR-VL: no loadable wrapper here, so take the vision model.

    `PaddleOCRVLForConditionalGeneration` does not construct under this
    transformers version (its config exposes no `text_config` and its rope
    block is rejected). The vision half is a self-contained
    `PreTrainedModel`, so it is built from `config.vision_config` and its
    weights lifted from the checkpoint's `visual.` prefix.
    """
    _one_tap_only(model_id, tap)
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    cls = get_class_from_dynamic_module(
        "modeling_paddleocr_vl.PaddleOCRVisionModel", model_id
    )
    module = cls._from_config(config.vision_config).to(dtype)

    weights: dict[str, torch.Tensor] = {}
    for shard in Path(snapshot_download(model_id, allow_patterns=["*.safetensors"])).glob(
        "*.safetensors"
    ):
        weights.update(load_file(str(shard)))
    prefix = "visual."
    state = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
    if not state:
        raise RuntimeError(f"{model_id}: no '{prefix}' weights in the checkpoint")
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"{model_id}: {len(missing)} vision weights missing")
    module.eval()

    def forward(pixel_values, grid_thw):
        # This tower takes patches as (batch, seq, C, kh, kw) with explicit
        # position ids; the processor emits them one dimension flatter.
        patches = pixel_values.unsqueeze(0)
        position_ids = torch.arange(
            patches.shape[1], device=patches.device
        ).unsqueeze(0)
        grid = [tuple(int(v) for v in row) for row in grid_thw]
        out = module(
            patches,
            position_ids=position_ids,
            image_grid_thw=grid,
            interpolate_pos_encoding=True,
        )
        return as_token_rows(out.last_hidden_state), ()

    return VisionRuntime(
        model_id=model_id,
        module=module,
        processor=AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True),
        forward=forward,
        feature_dim=config.vision_config.hidden_size,
        param_count=sum(p.numel() for p in module.parameters()),
    )


def _load_granite_vision(
    model_id: str, dtype: torch.dtype, tap: str | None = None
) -> VisionRuntime:
    """Granite Vision 4.1 — AnyRes tiles and two taps, see `probe.granite`."""
    from unflatten.probe.granite import load_granite

    return load_granite(model_id, dtype, tap)


#: model id -> loader. A family, not a single checkpoint, per entry.
LOADERS: dict[str, Callable[[str, torch.dtype, str | None], VisionRuntime]] = {
    "Qwen/Qwen3-VL-4B-Instruct": _load_qwen3vl,
    "rednote-hilab/dots.mocr": _load_qwen2vl_style,
    "zenosai/MonkeyOCRv2-B-Parsing": _load_qwen2vl_style,
    "PaddlePaddle/PaddleOCR-VL-1.6": _load_paddleocr,
    "ibm-granite/granite-vision-4.1-4b": _load_granite_vision,
}


def load_vision_runtime(
    model_id: str, *, dtype: torch.dtype = torch.bfloat16, tap: str | None = None
) -> VisionRuntime:
    """Load `model_id`'s frozen vision tower, optionally at a named tap."""
    loader = LOADERS.get(model_id)
    if loader is None:
        raise KeyError(
            f"no backbone loader for {model_id!r}; known: {', '.join(sorted(LOADERS))}"
        )
    runtime = loader(model_id, dtype, tap)
    for parameter in runtime.module.parameters():
        parameter.requires_grad_(False)
    return runtime

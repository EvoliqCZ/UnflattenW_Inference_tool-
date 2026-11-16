"""`UnflattenWForDocumentReconstruction` -- the `trust_remote_code` HF model
class for one complete Unflatten W package (DEC-050).

This file is copied verbatim into every exported repository
(`scripts/export_complete_model.py`) and is what `AutoModel.from_pretrained
(repo_dir, trust_remote_code=True)` resolves via `config.json`'s
`auto_map`. It requires the `unflatten` package to be importable (pip
install from this repository) -- exactly the relationship dots.ocr's own
remote code has to its supporting library; this file is a thin assembly of
already-tested `unflatten.model`/`unflatten.eval`/`unflatten.service`
primitives, not a reimplementation of the model.

`reconstruct(image)` returns a Canonical Document AST (RFC001) -- never
OOXML or `.docx`; only `unflatten.renderer` (a separate package, imported
by the caller, not by this class) turns that AST into an editable Word
document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import PreTrainedModel

from .configuration_unflatten import UnflattenWConfig


class UnflattenWExportInvalid(ValueError):
    """The exported repository directory is missing a file this loader
    requires, or one of its declared identities does not resolve."""


class UnflattenWForDocumentReconstruction(PreTrainedModel):
    """One complete, self-contained Unflatten W diagnostic model.

    `from_pretrained` is fully overridden (not `PreTrainedModel`'s tensor-
    sharded loading machinery, which assumes every weight is a named
    `nn.Parameter` registered at `__init__` time): this class only builds
    the same `unflatten.model.candidate_runtime.CandidateRuntime` the
    resident diagnostic worker builds, from local files this repository
    itself carries -- so `reconstruct` on one page gives the same output
    the worker gives for the same checkpoint and image.
    """

    config_class = UnflattenWConfig
    base_model_prefix = "unflatten_w"

    def __init__(self, config: UnflattenWConfig) -> None:
        super().__init__(config)
        self._runtime: Any = None
        # `PreTrainedModel` expects at least one registered tensor for its
        # device/dtype bookkeeping; the real weights live inside
        # `self._runtime.model`, assembled only once `from_pretrained` has
        # resolved the backbone and heads (never at plain `__init__` time,
        # which HF's `AutoConfig`-only paths may call without a repo dir).
        self.register_buffer("_unflatten_marker", torch.zeros(1), persistent=False)

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any
    ) -> UnflattenWForDocumentReconstruction:
        from safetensors.torch import load_file as load_safetensors

        from unflatten.model.candidate_backbone import CandidateVisionBackbone
        from unflatten.model.candidate_loader import load_qwen_candidate
        from unflatten.model.candidate_runtime import (
            CandidateRuntime,
            DecodePolicy,
            RuntimeIdentityPin,
            _build_locked_vocabulary,
            build_candidate_model,
        )
        from unflatten.model.capability_profile import (
            resolve_capability_profile_from_component_lock,
        )
        from unflatten.model.manifests import component_lock_hash

        repo_dir = Path(pretrained_model_name_or_path)
        if not repo_dir.is_dir():
            raise UnflattenWExportInvalid(
                "UnflattenWForDocumentReconstruction.from_pretrained requires a local "
                f"directory (a repository already downloaded/cloned); got "
                f"{pretrained_model_name_or_path!r}"
            )

        config = kwargs.get("config")
        if config is None:
            config = UnflattenWConfig.from_pretrained(repo_dir)
        model = cls(config)

        heads_path = repo_dir / "unflatten_heads.safetensors"
        if not heads_path.is_file():
            raise UnflattenWExportInvalid(f"{heads_path} is missing")
        component_lock = _read_json(repo_dir / "component-lock.json")
        allowlist = _read_json(repo_dir / "qwen_text_allowlist.json")
        decode_policy_doc = _read_json(repo_dir / "decode_policy.json")

        backbone_dir = repo_dir / "backbone"
        offline_backbone = backbone_dir.is_dir()
        backbone_source = str(backbone_dir) if offline_backbone else config.backbone_model_id
        revision = None if offline_backbone else (config.backbone_revision or None)

        candidate = load_qwen_candidate(
            model_id=backbone_source,
            revision=revision or "main",
            attn_implementation="sdpa",
        )
        vocabulary = _build_locked_vocabulary(
            component_lock=component_lock,
            allowlist=allowlist,
            tokenizer_artifact_set_hash=config.tokenizer_artifact_set_hash,
            native_vocab_size=candidate.vocab_size,
        )
        net = build_candidate_model(candidate, torch.device("cpu"))
        state_dict = load_safetensors(str(heads_path))
        net.load_state_dict(state_dict, strict=True)

        from transformers import AutoImageProcessor, AutoTokenizer

        processor = AutoImageProcessor.from_pretrained(backbone_source)
        tokenizer = AutoTokenizer.from_pretrained(backbone_source)
        vision = CandidateVisionBackbone(candidate=candidate, processor=processor, device="cpu")
        capability_profile = resolve_capability_profile_from_component_lock(component_lock)

        decode_block = decode_policy_doc["decode"]
        decode_policy = DecodePolicy(
            loop_guard=decode_block.get("loop_guard"),
            max_steps=decode_block["max_steps"],
            typed_decode=decode_block["typed_decode"],
            region_decode=decode_block["region_decode"],
            selection_policy_hash=decode_policy_doc["selection_policy_hash"],
        )
        pin = RuntimeIdentityPin(
            model_id=candidate.model_id,
            revision=revision or config.backbone_revision,
            attention_implementation=candidate.attn_implementation,
            checkpoint_sha256=config.checkpoint_sha256,
            component_lock_hash=config.component_lock_hash,
        )

        model._runtime = CandidateRuntime(
            candidate=candidate,
            vision=vision,
            model=net,
            vocabulary=vocabulary,
            decode_policy=decode_policy,
            identity=pin,
            capability_profile=capability_profile,
            tokenizer=tokenizer,
            eval_cache_identity={},
            component_lock_hash=component_lock_hash(component_lock),
            selection_policy_hash=decode_policy_doc["selection_policy_hash"],
        )
        return model

    def reconstruct(self, image: Image.Image) -> dict[str, Any]:
        """Reconstruct one page's Canonical Document AST (RFC001) from a PIL image.

        Returns a dict with `canonical_ast` (the RFC001 tree, or `None` when
        the page did not reach `ast_status == "strict_valid"`),
        `ast_status`, `page_outcome`, and `prediction_ir` (the sealed
        Prediction IR artifact -- the same object the diagnostic service
        publishes). Word output is never produced here: pass `canonical_ast`
        to a separate `unflatten.renderer` install to get a `.docx`.
        """
        if self._runtime is None:
            raise UnflattenWExportInvalid(
                "model has no loaded runtime -- construct it via from_pretrained"
            )
        from unflatten.eval.builder_chain.run import process_model_page
        from unflatten.eval.s2_gate.decode import seal_page
        from unflatten.service.docx_stage import _loop_guard_detail
        from unflatten.service.model_runtime import DiagnosticModelRuntime, build_single_page_input

        rgb = image.convert("RGB")
        width, height = rgb.size
        model_input = build_single_page_input(
            width_px=width, height_px=height, rgb_pixels=rgb.tobytes()
        )
        diag = DiagnosticModelRuntime(candidate_runtime=self._runtime)
        prepared = diag.extract(model_input)
        raw = diag.decode_features(prepared)
        prediction_ir = diag.seal(raw)

        page = seal_page(
            document_bundle_id="hf-export-reconstruct",
            hypothesis=raw,
            eval_cache_identity=self._runtime.eval_cache_identity,
            visual_features=prepared.visual_features,
            component_lock_hash=self._runtime.component_lock_hash,
            selection_policy_hash=self._runtime.selection_policy_hash,
            decode_fn_identity="decode_free_running",
            max_steps=self._runtime.decode_policy.max_steps,
            detail=_loop_guard_detail(
                raw.issues, prefill_chunk_tokens=self._runtime.decode_policy.prefill_chunk_tokens
            ),
        )
        row = process_model_page(
            page,
            vocabulary=self._runtime.vocabulary,
            tokenizer=self._runtime.tokenizer,
            headless_defaults=True,
            table_geometry=True,
            story_dedup=True,
            recovered_image_materialization=True,
        )
        canonical_ast = None
        ast_status = None
        if row.envelope is not None:
            ast_status = row.envelope.axes.ast_status
            if ast_status == "strict_valid":
                canonical_ast = row.envelope.ast.canonical_ast
        return {
            "canonical_ast": canonical_ast,
            "ast_status": ast_status,
            "page_outcome": row.page_outcome,
            "prediction_ir": prediction_ir,
        }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "UnflattenWForDocumentReconstruction has no tensor-in/tensor-out "
            "forward(); call reconstruct(image) instead"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UnflattenWExportInvalid(f"cannot read {path}: {exc}") from exc

"""`UnflattenWConfig` -- the `trust_remote_code` HF config for one complete
Unflatten W model package (DEC-050).

This file is copied verbatim into every exported repository
(`scripts/export_complete_model.py`) and loaded there by
`AutoConfig.from_pretrained(repo_dir, trust_remote_code=True)` via
`config.json`'s own `auto_map`. It carries no executable model logic --
only the identity fields `modeling_unflatten.py` needs to locate and verify
the rest of the package (backbone, heads, component lock, decode policy).
"""

from __future__ import annotations

from transformers import PretrainedConfig

#: DEC-050 §1: one complete repository, one product-quality claim -- always
#: false for this diagnostic export (RFC005 discipline; see the model card).
PRODUCT_QUALITY_CLAIM = False


class UnflattenWConfig(PretrainedConfig):
    """Identity fields for one exported Unflatten W package.

    `backbone_model_id`/`backbone_revision` name the pinned Qwen3-VL
    candidate this checkpoint was trained against (DEC-007 D3); when the
    repository also carries a `backbone/` directory (DEC-050 §3), that
    local copy is loaded OFFLINE instead of these being resolved against
    the HF hub -- see `modeling_unflatten.UnflattenWForDocumentReconstruction.
    from_pretrained`. `component_lock_hash`/`tokenizer_artifact_set_hash`/
    `checkpoint_sha256` are the same identities `unflatten.service.
    model_bundle` already pins in `manifest.json`; they are duplicated here
    only so `trust_remote_code` loading never has to read the bundle
    manifest to know what it is loading.
    """

    model_type = "unflatten_w"

    def __init__(
        self,
        backbone_model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        backbone_revision: str = "",
        component_lock_hash: str = "",
        tokenizer_artifact_set_hash: str = "",
        checkpoint_sha256: str = "",
        selection_policy_hash: str = "",
        product_quality_claim: bool = PRODUCT_QUALITY_CLAIM,
        **kwargs,
    ) -> None:
        self.backbone_model_id = backbone_model_id
        self.backbone_revision = backbone_revision
        self.component_lock_hash = component_lock_hash
        self.tokenizer_artifact_set_hash = tokenizer_artifact_set_hash
        self.checkpoint_sha256 = checkpoint_sha256
        self.selection_policy_hash = selection_policy_hash
        self.product_quality_claim = bool(product_quality_claim)
        super().__init__(**kwargs)

"""Export one complete Unflatten W model package (DEC-050).

Builds a D7 service model bundle (`scripts/build_model_bundle.py`'s own
`build()`), optionally carries the pinned Qwen3-VL backbone inside it
(`unflatten.service.model_bundle.add_backbone_to_bundle`, DEC-050 §3), then
adds the HF `trust_remote_code` surface on top: `unflatten_heads.safetensors`
(our params, weights only), `configuration_unflatten.py`/`modeling_unflatten.py`
(copied from `src/unflatten/hf_export/`), a root `config.json` with `auto_map`,
and a model card `README.md`. The result is BOTH a valid HF repository
(`AutoModel.from_pretrained(out_dir, trust_remote_code=True)`) and a valid
service model bundle (`unflatten.service.model_bundle.verify_bundle`).

    python scripts/export_complete_model.py \\
        --checkpoint ~/unflatten-runs/ckpt-v15b-140000/checkpoint.pt \\
        --run-dir <run_dir> \\
        --out /path/to/export \\
        [--backbone-snapshot-dir <local HF snapshot dir>] \\
        [--precision bf16|int8|fp8|int4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import build_model_bundle as bmb  # noqa: E402
from unflatten.hf_export.apache2_license import APACHE_2_0_LICENSE_TEXT  # noqa: E402
from unflatten.model.checkpoint_ancestry import file_sha256  # noqa: E402
from unflatten.model.quantization.spec import QuantizationSpec  # noqa: E402
from unflatten.service.model_bundle import (  # noqa: E402
    CHECKPOINT_FILENAME,
    MANIFEST_FILENAME,
    ModelBundleInvalid,
    add_backbone_to_bundle,
    load_manifest,
)

_HF_EXPORT_SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "unflatten" / "hf_export"
HEADS_FILENAME = "unflatten_heads.safetensors"
CONFIG_JSON_FILENAME = "config.json"
README_FILENAME = "README.md"
LICENSE_FILENAME = "LICENSE"
#: Icon-only brand mark (`scripts/build_brand_assets.py`), shown in the model card.
BRAND_MARK_FILENAME = "unflatten-w-mark.png"
_REMOTE_CODE_FILES = ("configuration_unflatten.py", "modeling_unflatten.py")


class ModelExportError(RuntimeError):
    """The source checkpoint or a companion file could not be exported."""


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_heads_safetensors(bundle_dir: Path) -> None:
    """Convert the bundle's weights-only checkpoint into a plain
    safetensors file -- HF-native loading, no pickle deserialization."""
    import torch
    from safetensors.torch import save_file

    state = torch.load(
        bundle_dir / CHECKPOINT_FILENAME, map_location="cpu", weights_only=True
    )
    model_state = state["model_state_dict"]
    tensors = {name: tensor.contiguous() for name, tensor in model_state.items()}
    save_file(tensors, str(bundle_dir / HEADS_FILENAME))


def _copy_remote_code(out_dir: Path) -> None:
    """Copy the two `trust_remote_code` files flat into `out_dir`, byte for
    byte. `modeling_unflatten.py`'s relative `from .configuration_unflatten
    import ...` is intentional: transformers' dynamic-module loader
    (`get_cached_module_file`) resolves sibling relative imports within a
    `trust_remote_code` repository by copying every file it relatively
    imports alongside it and synthesizing the package `__init__.py` itself
    -- a flat top-level `import configuration_unflatten` is NOT recognized
    as a local sibling by `check_imports` and is (incorrectly) treated as
    an external pip dependency, refusing the load outright."""
    for name in _REMOTE_CODE_FILES:
        source = _HF_EXPORT_SRC_DIR / name
        _write_text(out_dir / name, source.read_text(encoding="utf-8"))


def _write_config_json(out_dir: Path, manifest: dict[str, Any]) -> None:
    identities = manifest["identities"]
    base_model = manifest["base_model"]
    config = {
        "model_type": "unflatten_w",
        "architectures": ["UnflattenWForDocumentReconstruction"],
        "auto_map": {
            "AutoConfig": "configuration_unflatten.UnflattenWConfig",
            "AutoModel": "modeling_unflatten.UnflattenWForDocumentReconstruction",
        },
        "backbone_model_id": base_model["model_id"],
        "backbone_revision": base_model["revision"],
        "component_lock_hash": identities["component_lock_hash"],
        "tokenizer_artifact_set_hash": identities["tokenizer_artifact_set_hash"],
        "checkpoint_sha256": identities["checkpoint_sha256"],
        "selection_policy_hash": identities["selection_policy_hash"],
        "product_quality_claim": False,
    }
    _write_text(out_dir / CONFIG_JSON_FILENAME, json.dumps(config, indent=2, sort_keys=True) + "\n")


def _write_model_card(out_dir: Path, manifest: dict[str, Any], *, carries_backbone: bool) -> None:
    base_model = manifest["base_model"]
    backbone_note = (
        "This repository carries its own offline copy of the backbone under `backbone/`; "
        "loading it does not contact the Hugging Face Hub."
        if carries_backbone
        else "This repository does NOT carry the backbone; `from_pretrained` resolves "
        f"`{base_model['model_id']}@{base_model['revision']}` from the Hugging Face Hub "
        "or your local cache."
    )
    card = f"""\
<img src="{BRAND_MARK_FILENAME}" alt="Unflatten W" width="96">

# Unflatten W (diagnostic export)

Reconstructs an editable-Word **Canonical Document AST** (RFC001) from a single 300 DPI
page image: `image -> AI model -> Canonical Document AST`. It never emits OOXML or
`.docx` itself -- rendering to an editable `.docx` is a separate step, done by the
`unflatten` package's deterministic renderer (RFC002), not by this model.

`product_quality_claim: false` -- this is a diagnostic-scope research export
(RFC005 discipline), not a product quality claim.

## What this repository is

One complete package (DEC-050): the pinned `{base_model["model_id"]}` backbone
(vision + language, frozen) plus ~161M parameters of Unflatten-trained adapters,
typed property heads and a document/tree decoder (`unflatten_heads.safetensors`,
weights only), the composite decode vocabulary, capability profile and frozen
decode policy this checkpoint was selected under.

{backbone_note}

## License

The backbone (`{base_model["model_id"]}@{base_model["revision"]}`) is distributed by
Qwen (Alibaba Cloud) under the **Apache License, Version 2.0** (confirmed via the
model's own Hugging Face Hub metadata, `license: apache-2.0`); the full license text is
reproduced in `LICENSE` in this repository, with attribution. Unflatten's own adapter/head
weights and code follow this repository's license.

## Usage

```bash
pip install -e /path/to/Unflatten-1.0-MoE  # the `unflatten` package (this remote code imports it)
```

```python
from PIL import Image
from transformers import AutoModel

model = AutoModel.from_pretrained("<this repo>", trust_remote_code=True)
result = model.reconstruct(Image.open("page.png"))
canonical_ast = result["canonical_ast"]  # RFC001 Canonical Document AST, or None
```

## Scope

Single 300 DPI page input only (`word-core-v0`/typed-decode capability profile,
RFC004 §11). Multi-page assembly, MoE, and product claims are out of scope for this
export.
"""
    _write_text(out_dir / README_FILENAME, card)
    _write_text(out_dir / LICENSE_FILENAME, APACHE_2_0_LICENSE_TEXT)
    (out_dir / BRAND_MARK_FILENAME).write_bytes(
        (_HF_EXPORT_SRC_DIR / BRAND_MARK_FILENAME).read_bytes()
    )


def _extend_manifest_with_hf_files(out_dir: Path) -> None:
    """`verify_bundle`'s `_verify_file_set` refuses any top-level file not
    named in `manifest["files"]` -- add the HF-surface files it wrote so the
    export stays a valid bundle (DEC-050 §3: "the exported dir MUST also be
    a valid service model bundle")."""
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extra_names = (
        HEADS_FILENAME,
        CONFIG_JSON_FILENAME,
        README_FILENAME,
        LICENSE_FILENAME,
        BRAND_MARK_FILENAME,
        *_REMOTE_CODE_FILES,
    )
    for name in extra_names:
        path = out_dir / name
        manifest["files"][name] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def export(
    *,
    checkpoint_path: Path,
    component_lock_path: Path,
    allowlist_path: Path,
    caps_path: Path,
    selection_policy_path: Path,
    pinned_hashes_path: Path,
    out_dir: Path,
    precision: QuantizationSpec | None = None,
    backbone_snapshot_dir: Path | None = None,
    backbone_model_id: str | None = None,
    backbone_revision: str | None = None,
) -> Path:
    try:
        bmb.build(
            checkpoint_path=checkpoint_path,
            component_lock_path=component_lock_path,
            allowlist_path=allowlist_path,
            caps_path=caps_path,
            selection_policy_path=selection_policy_path,
            pinned_hashes_path=pinned_hashes_path,
            out_dir=out_dir,
            precision=precision,
        )
    except (bmb.ModelBundleBuildError, ModelBundleInvalid) as exc:
        raise ModelExportError(f"model bundle build refused: {exc}") from exc

    manifest = load_manifest(out_dir)
    carries_backbone = backbone_snapshot_dir is not None
    if carries_backbone:
        add_backbone_to_bundle(
            out_dir,
            backbone_snapshot_dir,
            model_id=backbone_model_id or manifest["base_model"]["model_id"],
            revision=backbone_revision or manifest["base_model"]["revision"],
        )

    _write_heads_safetensors(out_dir)
    _copy_remote_code(out_dir)
    _write_config_json(out_dir, manifest)
    _write_model_card(out_dir, manifest, carries_backbone=carries_backbone)
    _extend_manifest_with_hf_files(out_dir)
    return out_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--component-lock", type=Path, default=None)
    parser.add_argument("--allowlist", type=Path, default=None)
    parser.add_argument("--pinned-hashes", type=Path, default=None)
    parser.add_argument("--caps", type=Path, default=bmb.CAPS_PATH)
    parser.add_argument(
        "--selection-policy", type=Path, default=bmb._DEFAULT_SELECTION_POLICY_PATH
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--precision", choices=("bf16", "int8", "fp8", "int4"), default=None
    )
    parser.add_argument("--backbone-snapshot-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dir: Path | None = args.run_dir
    component_lock_path = args.component_lock or (
        run_dir / "component-lock.json" if run_dir else None
    )
    allowlist_path = args.allowlist or (run_dir / "qwen_text_allowlist.json" if run_dir else None)
    pinned_hashes_path = args.pinned_hashes or (
        run_dir / "qwen-pinned-hashes.json" if run_dir else None
    )
    if component_lock_path is None or allowlist_path is None or pinned_hashes_path is None:
        raise SystemExit(
            "export requires --component-lock/--allowlist/--pinned-hashes "
            "(directly, or via --run-dir)"
        )
    precision = QuantizationSpec(decoder_precision=args.precision) if args.precision else None
    try:
        out_dir = export(
            checkpoint_path=args.checkpoint,
            component_lock_path=component_lock_path,
            allowlist_path=allowlist_path,
            caps_path=args.caps,
            selection_policy_path=args.selection_policy,
            pinned_hashes_path=pinned_hashes_path,
            out_dir=args.out,
            precision=precision,
            backbone_snapshot_dir=args.backbone_snapshot_dir,
        )
    except (ModelExportError, ModelBundleInvalid) as exc:
        print(f"export refused: {exc}", file=sys.stderr)
        return 1
    print(f"complete model package written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Head bundle: contract + safetensors weights; sha256 = canonical contract + head bytes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

from unflatten.model.page_start_policy import HEAD_SOURCE_PREFIX, StoryPattern

from .head import HEAD_KINDS
from .moments import MomentsContract, page_start_moments

_CONTRACT = "contract.json"
_WEIGHTS = "head.safetensors"
_SHA = "sha256.txt"


def _contract_bytes(contract: MomentsContract) -> bytes:
    return json.dumps(
        contract.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _bundle_sha256(contract: MomentsContract, head_bytes: bytes) -> str:
    return hashlib.sha256(_contract_bytes(contract) + head_bytes).hexdigest()


@dataclass
class HeadBundle:
    contract: MomentsContract
    head: torch.nn.Module
    sha256: str

    def _features(self, features: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        c = self.contract
        if features.ndim != 2 or features.shape[1] != c.visual_dim:
            raise ValueError(
                f"head bundle: features must be [V, {c.visual_dim}], got {tuple(features.shape)}"
            )
        return page_start_moments(
            features,
            grid_thw,
            merge_size=c.merge_size,
            band_top=c.band_fraction_top,
            band_bottom=c.band_fraction_bottom,
        )

    @torch.no_grad()
    def predict(self, features: torch.Tensor, grid_thw: torch.Tensor) -> StoryPattern:
        x = self._features(features, grid_thw)
        device = next(self.head.parameters()).device
        p = torch.sigmoid(self.head(x.to(device)[None]).float())[0].tolist()
        confidence = min(max(q, 1.0 - q) for q in p)
        return StoryPattern(
            has_header=p[0] >= 0.5,
            has_footer=p[1] >= 0.5,
            source=f"{HEAD_SOURCE_PREFIX}{self.sha256}",
            confidence=confidence,
        )


def save_head_bundle(directory: Path, *, contract: MomentsContract, head: torch.nn.Module) -> str:
    """Write a new bundle directory; returns its sha256. Never overwrites."""
    directory = Path(directory)
    kind = getattr(head, "kind", None)
    if kind not in HEAD_KINDS:
        raise ValueError(f"head bundle: unsupported head kind {kind!r}")
    state = {k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("head bundle: head state contains nonfinite values")
    in_dim = 6 * contract.visual_dim
    if state["mu"].shape != (in_dim,):
        raise ValueError(f"head bundle: head input {tuple(state['mu'].shape)} != 6*visual_dim")
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / _WEIGHTS
    save_file(state, weights, metadata={"kind": kind})
    sha = _bundle_sha256(contract, weights.read_bytes())
    (directory / _CONTRACT).write_bytes(_contract_bytes(contract))
    (directory / _SHA).write_text(sha + "\n")
    return sha


def load_head_bundle(
    directory: Path,
    *,
    device: str,
    expect_feature_identity_sha256: str,
    expect_visual_policy_hash: str,
) -> HeadBundle:
    """Load a bundle bound to a feature identity; refuses any mismatch with ValueError."""
    directory = Path(directory)
    try:
        contract = MomentsContract.from_dict(json.loads((directory / _CONTRACT).read_text()))
        head_bytes = (directory / _WEIGHTS).read_bytes()
        recorded = (directory / _SHA).read_text().strip()
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"head bundle: cannot read bundle at {directory}: {exc}") from exc
    if contract.feature_identity_sha256 != expect_feature_identity_sha256:
        raise ValueError(
            f"head bundle {directory}: feature_identity_sha256 "
            f"{contract.feature_identity_sha256} != expected {expect_feature_identity_sha256}"
        )
    if contract.visual_input_policy_hash != expect_visual_policy_hash:
        raise ValueError(
            f"head bundle {directory}: visual_input_policy_hash "
            f"{contract.visual_input_policy_hash} != expected {expect_visual_policy_hash}"
        )
    sha = _bundle_sha256(contract, head_bytes)
    if sha != recorded:
        raise ValueError(f"head bundle {directory}: sha256 {sha} != recorded {recorded}")
    try:
        state = load_file(directory / _WEIGHTS, device="cpu")
        with (directory / _WEIGHTS).open("rb") as fh:
            header_len = int.from_bytes(fh.read(8), "little")
            kind = json.loads(fh.read(header_len)).get("__metadata__", {}).get("kind")
        if kind not in HEAD_KINDS:
            raise ValueError(f"unsupported head kind {kind!r}")
        head = HEAD_KINDS[kind](6 * contract.visual_dim)
        head.load_state_dict(state, strict=True)
    except (ValueError, RuntimeError, SafetensorError, OSError) as exc:
        raise ValueError(f"head bundle {directory}: invalid weights: {exc}") from exc
    return HeadBundle(contract=contract, head=head.to(device).eval(), sha256=sha)

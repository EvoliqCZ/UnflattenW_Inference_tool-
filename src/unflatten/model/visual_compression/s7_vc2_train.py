"""RFC005-ERRATA-007 E5 VC-2 / VC-3: pre-encoder scout training.

VC-2 (signal a) trains ONLY :class:`~unflatten.model.visual_compression.
scout.ScoutModel` on the AST/aligner-derived
:mod:`region_importance_targets` risk classes, over a CPU-scale low-
resolution page raster -- **never** over the feature cache (rule 3: "the
scout consumes low-res page rasters", RFC004-ERRATA-005 E4's own flow puts
the scout *before* the real vision encoder ever runs, so there is no cache
row for it to read). :func:`run_vc2_signal_a_stub_step`'s signature has no
cache-shaped parameter at all; :func:`refuse_feature_cache_kwargs` is the
explicit, tested guard a caller cannot route around by adding an extra
keyword.

VC-3 (signal b) never reaches a gradient step here: it calls
:func:`unflatten.model.visual_compression.losses.l_pre_encoder_compressor_signal_b`,
which unconditionally raises `PreEncoderSignalBUnavailable` -- this module
does not add a second refusal path, it reuses the one that already exists.

Every report this module writes carries its own ``architecture_contract_passed``
verdict (:func:`check_vc2_architecture_gate`) -- narrower than VC-1's full
RFC005 §21.2 `architecture_contract` gate (there is no `VisualCompressor`-
shaped protocol for the pre-encoder scout yet), but real and
provably-failable: it fails if the scout stops emitting a deterministic
``(batch, grid, grid, 5)`` risk map at its configured grid. This is the
"VC-2 report" :mod:`s7_vc4_combined` reads before it will let the combined
PRE+POST arm run at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from unflatten.model.hashing import domain_hash
from unflatten.model.visual_compression.adaptive_v0 import RISK_CLASSES
from unflatten.model.visual_compression.losses import (
    l_pre_encoder_compressor_signal_a,
    l_pre_encoder_compressor_signal_b,
)
from unflatten.model.visual_compression.s7_policy import load_s7_stage_policy, s7_stage_policy_hash
from unflatten.model.visual_compression.s7_ratification import (
    S7RatificationRecord,
    require_ratification,
)
from unflatten.model.visual_compression.scout import ScoutModel

_S7_VC2_MANIFEST_DOMAIN = "rfc005.s7-vc2-run-manifest/v1"

_RISK_CLASS_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(RISK_CLASSES)}


class FeatureCacheInputRefused(RuntimeError):
    """A VC-2 caller tried to hand the scout a feature-cache-shaped input.

    Rule 3 (RFC005-ERRATA-007 E5): the scout consumes low-res page rasters,
    never the frozen backbone's cached features.
    """


def refuse_feature_cache_kwargs(kwargs: dict[str, Any]) -> None:
    """Raise :class:`FeatureCacheInputRefused` if any keyword name mentions
    a feature cache -- called before anything else in the training step, so
    the refusal happens before a cache-shaped tensor could ever be forwarded."""
    offending = [name for name in kwargs if "cache" in name.lower()]
    if offending:
        raise FeatureCacheInputRefused(
            f"VC-2 training refuses cache-shaped input keyword(s) {offending}; "
            "the scout must be called with a low-res page raster, never a "
            "feature cache read"
        )


def risk_class_names_to_indices(risk_classes: list[str]) -> Tensor:
    """`RegionImportanceTarget.risk_class` strings -> `RISK_CLASSES` indices,
    the shape :func:`l_pre_encoder_compressor_signal_a` compares logits against."""
    return torch.tensor([_RISK_CLASS_INDEX[name] for name in risk_classes], dtype=torch.long)


def run_vc2_signal_a_stub_step(
    *,
    scout: ScoutModel,
    low_res_view: Tensor,
    target_risk_class: Tensor,
    mask: Tensor,
    optimizer: torch.optim.Optimizer,
    policy: dict[str, Any],
) -> dict[str, float]:
    """One optimizer step over `scout` only, signal a
    (`l_pre_encoder_compressor_signal_a`), weighted by the S7 policy's
    `vc2.signal_a_loss_weight_micro` -- never a code-literal weight."""
    logits = scout(low_res_view)  # (batch, grid, grid, 5)
    per_cell_loss = l_pre_encoder_compressor_signal_a(logits, target_risk_class, mask)
    weight = policy["vc2"]["signal_a_loss_weight_micro"] / 1_000_000
    loss = weight * per_cell_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"signal_a_loss": float(per_cell_loss.detach()), "weighted_loss": float(loss.detach())}


def check_vc2_architecture_gate(scout: ScoutModel, low_res_view: Tensor) -> bool:
    """VC-2's own architecture gate (RFC005 §21.2 in miniature): the scout
    must keep emitting a deterministic ``(batch, grid, grid, 5)`` risk map
    at its configured grid for ``low_res_view``'s shape. Provably-failable:
    a scout whose output shape drifted from ``scout.grid``, or whose two
    calls over the same input disagreed, would fail this.
    """
    with torch.no_grad():
        first = scout(low_res_view)
        second = scout(low_res_view)
    expected_shape = (low_res_view.shape[0], scout.grid, scout.grid, len(RISK_CLASSES))
    return (
        tuple(first.shape) == expected_shape
        and tuple(second.shape) == expected_shape
        and torch.equal(first, second)
    )


def run_vc3_signal_b_refusal(*args: object, **kwargs: object) -> None:
    """VC-3 always refuses -- reuses `l_pre_encoder_compressor_signal_b`'s
    own `PreEncoderSignalBUnavailable`, propagated unchanged."""
    l_pre_encoder_compressor_signal_b(*args, **kwargs)


def build_vc2_run_manifest(
    *, policy_hash: str, seed: int, ratification_record: S7RatificationRecord | None
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "s7_vc2_run_manifest_schema_version": "0.1.0",
        "stage": "S7",
        "child": "VC-2",
        "policy_hash": policy_hash,
        "seed": seed,
        "ratification_decision_id": (
            ratification_record.decision_id if ratification_record is not None else None
        ),
    }
    manifest["s7_vc2_run_manifest_hash"] = domain_hash(_S7_VC2_MANIFEST_DOMAIN, manifest)
    return manifest


def train_vc2_step(
    *,
    scout: ScoutModel,
    low_res_view: Tensor,
    target_risk_class: Tensor,
    mask: Tensor,
    optimizer: torch.optim.Optimizer,
    dry_run: bool,
    ratification_record: Path | None = None,
    seed: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    """The gated VC-2 entry point. `**extra` exists only so
    :func:`refuse_feature_cache_kwargs` has something to inspect for a
    caller that adds an unexpected cache-shaped keyword -- it is never
    forwarded to the scout."""
    refuse_feature_cache_kwargs(extra)
    record = require_ratification(
        child="VC-2", dry_run=dry_run, ratification_record=ratification_record
    )
    policy = load_s7_stage_policy()
    result = run_vc2_signal_a_stub_step(
        scout=scout,
        low_res_view=low_res_view,
        target_risk_class=target_risk_class,
        mask=mask,
        optimizer=optimizer,
        policy=policy,
    )
    manifest = build_vc2_run_manifest(
        policy_hash=s7_stage_policy_hash(policy), seed=seed, ratification_record=record
    )
    gate_passed = check_vc2_architecture_gate(scout, low_res_view)
    return {
        **result,
        "architecture_contract_passed": gate_passed,
        "manifest": manifest,
        "dry_run": dry_run,
    }


def _build_stub_scenario(
    *, grid: int = 4, in_channels: int = 3
) -> tuple[ScoutModel, Tensor, Tensor, Tensor]:
    scout = ScoutModel(in_channels=in_channels, hidden=8, grid=grid)
    low_res_view = torch.randn(1, in_channels, 32, 32)
    target_risk_class = torch.zeros(1, grid, grid, dtype=torch.long)
    mask = torch.ones(1, grid, grid, dtype=torch.bool)
    return scout, low_res_view, target_risk_class, mask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RFC005-ERRATA-007 VC-2 dormant trainer")
    parser.add_argument("--out", type=Path, required=True, help="output JSON report path")
    parser.add_argument(
        "--dry-run", action="store_true", help="run one stub step (the only mode implemented)"
    )
    parser.add_argument("--ratification-record", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    scout, low_res_view, target_risk_class, mask = _build_stub_scenario()
    optimizer = torch.optim.SGD(scout.parameters(), lr=1e-3)
    result = train_vc2_step(
        scout=scout,
        low_res_view=low_res_view,
        target_risk_class=target_risk_class,
        mask=mask,
        optimizer=optimizer,
        dry_run=args.dry_run,
        ratification_record=args.ratification_record,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

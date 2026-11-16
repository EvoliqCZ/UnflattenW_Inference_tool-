"""Dormant S4.5 trainer (RFC005-ERRATA-010 E2/E7, DEC-047 addendum A v2, PROPOSED).

Two entries:

* :func:`run_s4_5` is the REAL-run entry. Its first act is the ratification
  gate (:func:`unflatten.model.rl.ratification.require_ratification`, bound to
  the composite policy hash); it touches no optimizer, dataset, checkpoint or
  renderer before that. Only a valid record, RATIFIED policies and
  ``S4_5Config(enabled=True)`` reach ``_real_update_loop``, which delegates to
  :func:`unflatten.model.rl.real_run.run_real` with the injected
  :class:`~unflatten.model.rl.real_run_ports.RealRunPorts`. Nothing real is
  opened by default: a missing port is a typed ``RealRunPortMissing``.
* :func:`run_s4_5_dry_run` exercises the pipeline seams once on injected stubs
  (sample -> score -> collate -> ``l_rl``) and reports ``gradient_steps = 0``.
  It opens nothing real, builds no optimizer and never calls ``backward``;
  parameters are hashed before and after and must be byte-identical.

The CLI has exactly one legal branch, ``--dry-run``, and no checkpoint or
dataset flag; the real run is a library call only.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch
from torch import nn

from unflatten.model import composite_grammar as cg
from unflatten.model import composite_vocab as cv
from unflatten.model.rl.collate import collate_rollout_batch
from unflatten.model.rl.gate_policy import verify_rl_stage_gate_policy
from unflatten.model.rl.ids import STAGE_ID
from unflatten.model.rl.loss import (
    OLD_LOGPROBS_SOURCE,
    RlLossPolicy,
    RlLossProvenance,
    l_rl,
)
from unflatten.model.rl.policies import (
    RlPolicyInvalid,
    build_execution_profile,
    build_rollout_policy,
    verify_dataset_manifest,
    verify_reward_policy,
    verify_rollout_policy,
)
from unflatten.model.rl.ratification import (
    S4_5_CHILDREN,
    load_ratification_record,
    require_ratification,
)
from unflatten.model.rl.real_run_ports import RealRunPortMissing, RealRunPorts
from unflatten.model.rl.reference import ReferencePolicyIdentity
from unflatten.model.rl.rollout import RolloutTrajectory, sample_rollouts
from unflatten.model.rl.run_manifest import (
    RunArtefacts,
    build_rl_run_manifest,
    composite_policy_hash,
    policy_status,
)
from unflatten.model.rl.training_policy import verify_rl_training_policy

if TYPE_CHECKING:
    from unflatten.model.rl.real_run import RealRunResult

__all__ = [
    "S4_5_CHILDREN",
    "DryRunMovedParameters",
    "DryRunWorld",
    "RealRunPortMissing",
    "RealRunPorts",
    "S4_5Config",
    "S4_5Disabled",
    "build_stub_world",
    "main",
    "run_s4_5",
    "run_s4_5_dry_run",
]

_REFUSAL_POINTER: Final = (
    "S4.5 has no real run: DEC-047 addendum A (docs/decisions/"
    "DEC-047-addendum-S4_5-RL-PROPOSED.md) allows only --dry-run on stubs"
)
_STUB_PROFILE: Final = "dense-v0-s1-exact-text-v1"
_STUB_STRUCTURAL_PROFILE: Final = "word-visual-v1.1"
_STUB_TOKENIZER_HASH: Final = "sha256:" + "bb" * 32
_STUB_NATIVE_VOCAB: Final = 8
_STUB_VISUAL_TOKENS: Final = 3
_STUB_MAX_STEPS: Final = 24


class S4_5Disabled(RuntimeError):
    """S4.5 is off: the config that would run it was not explicitly enabled."""


class DryRunMovedParameters(RuntimeError):
    """A dry run changed a parameter; the dry run takes no step by contract."""


@dataclass(frozen=True, slots=True)
class S4_5Config:
    """Off by default; nothing in this package turns it on by itself."""

    enabled: bool = False


# --- real-run entry ------------------------------------------------------------


def _real_update_loop(
    manifest: Mapping[str, Any],
    *,
    artefacts: RunArtefacts,
    run_class: str,
    selected_typed_heads: list[str],
    ports: RealRunPorts | None,
    allow_unbound_registry: bool = False,
) -> RealRunResult:
    """The single seam to the update loop; imported lazily so the dry run and the
    package import stay light and touch no chain/model machinery."""
    if ports is None:
        raise RealRunPortMissing(
            f"run manifest {manifest['run_id']} passed the gate but no RealRunPorts were injected"
        )
    from unflatten.model.rl.real_run import run_real

    return run_real(
        manifest=manifest,
        artefacts=artefacts,
        run_class=run_class,
        selected_typed_heads=selected_typed_heads,
        ports=ports,
        allow_unbound_registry=allow_unbound_registry,
    )


def run_s4_5(
    *,
    ratification_record: Path | None,
    artefacts: RunArtefacts,
    run_class: str,
    selected_typed_heads: Iterable[str] = (),
    config: S4_5Config = S4_5Config(),  # noqa: B008 - frozen, immutable default
    ports: RealRunPorts | None = None,
    run_seed: int = 0,
    allow_unbound_registry: bool = False,
) -> RealRunResult:
    """Gate first; then verify the artefacts; then (enabled only) the update loop.

    A missing record, or one that cannot be loaded structurally, is reported
    (``RlRatificationRequired``) before any artefact is looked at; the artefacts
    are then verified (typed ``RlPolicyInvalid``) before their hashes are
    composed, so a malformed one is never a ``KeyError`` or ``AttributeError``.
    Binding of the record to *these* artefacts is checked afterwards.
    """
    heads = sorted(set(selected_typed_heads))
    if ratification_record is None:
        require_ratification(None, policy_hash="", policy_status="")  # raises: no record
    else:
        load_ratification_record(ratification_record)  # structural refusal first
    if not isinstance(artefacts.policy_parent, ReferencePolicyIdentity):
        raise RlPolicyInvalid(
            "RunArtefacts.policy_parent must be a ReferencePolicyIdentity, "
            f"got {type(artefacts.policy_parent).__name__}"
        )
    verify_reward_policy(artefacts.reward_policy)
    verify_rollout_policy(artefacts.rollout_policy)
    verify_rl_training_policy(artefacts.training_policy)
    verify_dataset_manifest(artefacts.dataset_manifest)
    verify_rl_stage_gate_policy(artefacts.stage_policy)
    record = require_ratification(
        ratification_record,
        policy_hash=composite_policy_hash(artefacts, run_class, heads),
        policy_status=policy_status(artefacts),
        parent_checkpoint_id=artefacts.policy_parent.checkpoint_id,
    )
    bound = dataclasses.replace(artefacts, ratification=record)
    manifest = build_rl_run_manifest(
        run_class=run_class, run_seed=run_seed, artefacts=bound, selected_typed_heads=heads
    )
    if not config.enabled:
        raise S4_5Disabled("S4.5 is off (S4_5Config.enabled is False); nothing was run")
    return _real_update_loop(
        manifest,
        artefacts=bound,
        run_class=run_class,
        selected_typed_heads=heads,
        ports=ports,
        allow_unbound_registry=allow_unbound_registry,
    )


# --- dry run -------------------------------------------------------------------

type _Score = Callable[[RolloutTrajectory], tuple[str, int | None]]
type _LogProbs = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class DryRunWorld:
    """Everything a dry run touches, all injected. ``module`` is the thing whose
    parameters must not move; ``logprobs`` / ``reference_logprobs`` map packed
    token ids ``(N, T)`` to the sampled tokens' log-probabilities ``(N, T)``."""

    module: nn.Module
    model: Any
    controller: Any
    vocabulary: cv.CompositeVocabulary
    capability_profile_id: str
    visual_features: torch.Tensor
    image_grid_thw: torch.Tensor
    image_token_id: int
    policy: Mapping[str, Any]
    score: _Score
    logprobs: _LogProbs
    reference_logprobs: _LogProbs
    loss_policy: RlLossPolicy
    provenance: RlLossProvenance
    max_steps: int = _STUB_MAX_STEPS
    pad_id: int = 0


def _parameter_digest(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in module.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _trajectory_digest(trajectories: list[RolloutTrajectory]) -> str:
    payload = json.dumps([list(t.token_ids) for t in trajectories], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_s4_5_dry_run(world: DryRunWorld, *, run_seed: int, doc_id: str) -> dict[str, Any]:
    """One pass over the pipeline seams on ``world``; no optimizer, no backward."""
    before = _parameter_digest(world.module)
    trajectories = sample_rollouts(
        model=world.model,
        controller=world.controller,
        vocabulary=world.vocabulary,
        visual_features=world.visual_features,
        image_grid_thw=world.image_grid_thw,
        image_token_id=world.image_token_id,
        policy=world.policy,
        doc_id=doc_id,
        run_seed=run_seed,
        capability_profile_id=world.capability_profile_id,
        max_steps=world.max_steps,
    )
    scores = [world.score(t) for t in trajectories]
    batch = collate_rollout_batch(
        trajectories, group_ids=[0] * len(trajectories), pad_id=world.pad_id
    )
    new = world.logprobs(batch.token_ids)
    with torch.no_grad():
        old = world.logprobs(batch.token_ids).detach()
        ref = world.reference_logprobs(batch.token_ids).detach()
    result = l_rl(
        new_logprobs=new,
        old_logprobs=old,
        ref_logprobs=ref,
        padding_mask=batch.padding_mask,
        legal_mask=batch.legal_mask,
        group_ids=batch.group_ids,
        reward_status=[s for s, _ in scores],
        scalar_micro=[m for _, m in scores],
        policy=world.loss_policy,
        provenance=world.provenance,
        old_logprobs_source=OLD_LOGPROBS_SOURCE,
    )
    after = _parameter_digest(world.module)
    if after != before:
        raise DryRunMovedParameters("a dry run changed model parameters; it takes no step")
    return {
        "dry_run": True,
        "stage_id": STAGE_ID,
        "gradient_steps": 0,
        "optimizer_steps": 0,
        "backward_called": False,
        "rollouts": len(trajectories),
        "statuses": [t.status for t in trajectories],
        "scored": sum(1 for s, _ in scores if s == "scored"),
        "excluded_insufficient_evidence": result.excluded_insufficient_evidence,
        "gate_failed_count": result.gate_failed_count,
        "groups_total": result.groups_total,
        "zero_advantage_rate_micro": result.zero_advantage_rate_micro,
        "loss": float(result.loss.detach()),
        "kl_mean": result.kl_mean,
        "loss_requires_grad": bool(result.loss.requires_grad),
        "trajectory_digest": _trajectory_digest(trajectories),
        "parameter_digest": after,
        "parameters_changed": False,
    }


# --- stub world (no checkpoint, no dataset, no renderer) -----------------------


class _StubPolicy(nn.Module):
    """Uniform-ish composite head over the whole vocabulary: one bias vector."""

    def __init__(self, total: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-0.05, 0.05, total))

    def encode_prefix(self, batch: Any) -> tuple[None, torch.Tensor]:
        del batch
        return None, torch.zeros(1, 1)

    def encode_step(self, state: None, token_id: Any) -> tuple[None, torch.Tensor]:
        del token_id
        return state, torch.zeros(1, 1)

    def composite_head(self, hidden: torch.Tensor) -> torch.Tensor:
        del hidden
        return self.bias[None, :]

    def token_logprobs(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.bias, dim=-1)[token_ids]


def _stub_vocabulary() -> cv.CompositeVocabulary:
    payload = cv.build_qwen_text_allowlist(
        tokenizer_artifact_set_hash=_STUB_TOKENIZER_HASH,
        native_vocab_size=_STUB_NATIVE_VOCAB,
        classify=lambda native_id: None,
    )
    return cv.make_composite_vocabulary(
        tokenizer_artifact_set_hash=_STUB_TOKENIZER_HASH, allowlist_payload=payload
    )


def _stub_score(trajectory: RolloutTrajectory) -> tuple[str, int]:
    """Illustrative, never a reward claim: spreads the group so advantages exist."""
    return "scored", (len(trajectory.token_ids) % 7) * 100_000


def build_stub_world() -> DryRunWorld:
    vocabulary = _stub_vocabulary()
    policy_module = _StubPolicy(vocabulary.ranges.total_count)
    reference = _StubPolicy(vocabulary.ranges.total_count)
    with torch.no_grad():
        reference.bias.zero_()
    return DryRunWorld(
        module=policy_module,
        model=policy_module,
        controller=cg.CompositeGrammarController(
            vocabulary=vocabulary,
            capability_profile_id=_STUB_PROFILE,
            structural_profile=_STUB_STRUCTURAL_PROFILE,
        ),
        vocabulary=vocabulary,
        capability_profile_id=_STUB_PROFILE,
        visual_features=torch.zeros(1, _STUB_VISUAL_TOKENS, 1),
        image_grid_thw=torch.tensor([[1, 1, _STUB_VISUAL_TOKENS]]),
        image_token_id=0,
        policy=build_rollout_policy(
            temperature_micro=1_000_000,
            top_p_micro=1_000_000,
            group_size=3,
            rollout_budget=3,
            max_steps=_STUB_MAX_STEPS,
            loop_guard_arm="none",
            execution_profile=build_execution_profile(),
            loop_guards_in_policy=False,
        ),
        score=_stub_score,
        logprobs=policy_module.token_logprobs,
        reference_logprobs=reference.token_logprobs,
        loss_policy=RlLossPolicy(
            clip_epsilon_micro=200_000,
            kl_coefficient_micro=100_000,
            length_normalisation="token_mean",
            gate_failed_reward_micro=-1_000_000,
        ),
        provenance=RlLossProvenance(
            reward_policy_hash="dry-run-stub",
            rollout_policy_hash="dry-run-stub",
            reference_checkpoint_id="dry-run-stub",
            reference_ancestry_hash="dry-run-stub",
            typed_head_config_hash="dry-run-stub",
        ),
    )


# --- CLI -----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.rl.s4_5_train", description=__doc__
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.dry_run:
        print(f"refused: {_REFUSAL_POINTER}", file=sys.stderr)
        return 2
    report = run_s4_5_dry_run(build_stub_world(), run_seed=args.seed, doc_id="dry-run-doc")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Identity constants for the dormant S4.5 RL package.

Every hash domain here is owned by this package alone (prefix
``unflatten.rl.``); none may equal a domain already owned in
``unflatten.model``. The stage id is deliberately NOT a member of
``stage_state_policy.STAGES``.
"""

from __future__ import annotations

from typing import Final

STAGE_ID: Final = "S4_5_reward_guided_reconstruction_refinement"
MODULE_ID: Final = "RFC005-S4.5-reward-rl"
RUN_ID_PREFIX: Final = "s45run"

REWARD_POLICY_DOMAIN: Final = "unflatten.rl.reward-policy/v0"
ROLLOUT_POLICY_DOMAIN: Final = "unflatten.rl.rollout-policy/v0"
DATASET_MANIFEST_DOMAIN: Final = "unflatten.rl.dataset-manifest/v0"
CANDIDATE_EVALUATION_DOMAIN: Final = "unflatten.rl.candidate-evaluation/v0"
EXPOSURE_LEDGER_DOMAIN: Final = "unflatten.rl.exposure-ledger/v0"
STAGE_GATE_REPORT_DOMAIN: Final = "unflatten.rl.stage-gate-report/v0"
RUN_MANIFEST_DOMAIN: Final = "unflatten.rl.run-manifest/v0"
TRAINING_POLICY_DOMAIN: Final = "unflatten.rl.training-policy/v0"
STAGE_POLICY_DOMAIN: Final = "unflatten.rl.stage-policy/v0"
ROLLOUT_SEED_DOMAIN: Final = "unflatten.rl.rollout-seed/v0"
PAIRED_ARM_DOMAIN: Final = "unflatten.rl.paired-arm/v0"

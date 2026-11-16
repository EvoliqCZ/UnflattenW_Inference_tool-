"""The ONE door through which a production path (`s4_train`,
`candidate_runtime`) may select the RFC007 level router.

`test_candidate_model_architecture_variant.py::TestNoProductionReferencesTheVariant`
keeps every production module away from the architecture-variant config;
RFC007 needs `s4_train` (the v16 dense fine-tune) and the runtime (v16
bundles) to build the router when -- and only when -- a frozen manifest or
a checkpoint says so. Routing both through this helper keeps that exception
to exactly one shape: `variant=level_router_variant(policy)`, which is
`None` (today's dense model, byte-identical) whenever `policy` is `None`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from unflatten.model.candidate_architecture_variant import (
    CandidateArchitectureVariant,
    LevelRouterVariant,
)
from unflatten.model.level_router.policy import LevelRouterPolicy

#: Checkpoint key holding the policy payload of a router-bearing checkpoint
#: (absent == no router, every v15b checkpoint).
CHECKPOINT_POLICY_KEY = "level_router_policy"


def level_router_variant(policy: LevelRouterPolicy | None) -> CandidateArchitectureVariant | None:
    if policy is None:
        return None
    return CandidateArchitectureVariant(
        level_router=LevelRouterVariant(enabled=True, policy=policy)
    )


def level_router_checkpoint_payload(model: nn.Module) -> Mapping[str, Any] | None:
    """The policy payload to stamp into a checkpoint, `None` without a router."""
    router = getattr(model, "level_router", None)
    return None if router is None else router.policy.to_payload()

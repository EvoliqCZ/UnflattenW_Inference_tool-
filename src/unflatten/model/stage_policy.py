"""The `stage_policy` producer (RFC005 §26.4, §26.6): the loss policy that
carries every `λ_f`, the closed TaskView→loss allowlist, and the
`stage_policy_hash` `ExactTextCacheIdentity` and `s1_run_manifest` bind.

**This module builds and hashes a policy; it does not choose one.** RFC005
§26.4 requires every `λ_f` to be explicit, versioned and stored in a loss
policy -- explicit meaning *chosen by someone*, which is exactly what a
default would erase. Weights are carried as `loss_weights_milli`, integer
thousandths (λ=1.0 is 1000), because the deterministic CBOR profile these
identities hash under (`rfc8949-core-det-nfc-int-v0`) has no float in its
data model: a float λ could be validated but never hashed. The unit is in
the name so the scaling cannot be forgotten at a call site.

:func:`build_stage_policy` therefore has **no defaults**: a weight
missing for any family is refused by name, a weight for a family
that does not exist is refused by name, and the same holds for the stage
id, the dynamic-weighting declaration and the allowlist. Until the owner
freezes the values, this producer can compute an identity for a policy but
cannot supply the policy -- that gap is real and is not closed here.

The family set is read from
:data:`unflatten.model.training_materializer.LOSS_FAMILY_IDS` rather than
restated, for the reason `runtime_limits.LIMIT_FIELDS` gives for reading
the schema's own `required` list: a family added there becomes required of
every policy with nobody having to remember this file.

Three §26.6 rules the schema cannot express are enforced here:

* the `task_kind` must be an exact RFC003 task kind
  (:data:`unflatten.dataset.enums.TASK_KIND`) -- `renderer_negative_v0`
  and `full_ast_exact_v0` are real kinds, `full_ast` and `*` are not;
* one output locator may not enter two loss families for the same task
  kind ("Jeden output logit nesmie vstúpiť do dvoch loss families");
* `renderer_negative_v0` may carry no positive reconstruction loss
  family.

Hashed under `rfc005.stage-policy/v1` (design §4.9), the same
`domain_hash` convention as every other identity in this package.

Ships no `nn.Module`, no optimizer, and authorizes no gradient update.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.dataset.enums import TASK_KIND
from unflatten.model import training_materializer
from unflatten.model.hashing import domain_hash
from unflatten.model.training_materializer import LOSS_FAMILY_IDS

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_SCHEMA_NAME = "stage_policy"

#: Design §4.9 registry row `rfc005.stage-policy/v1`. Owned here: no other
#: module computes a `stage_policy_hash`.
_STAGE_POLICY_DOMAIN = "rfc005.stage-policy/v1"

STAGE_POLICY_SCHEMA_VERSION = "1.0.0"

#: The loss families every policy must weight, read from the materializer
#: (module docstring) so the two can never disagree.
REQUIRED_LOSS_FAMILIES: tuple[str, ...] = tuple(sorted(LOSS_FAMILY_IDS))

#: Kept as an alias of :data:`REQUIRED_LOSS_FAMILIES` for existing callers.
LOSS_FAMILIES: tuple[str, ...] = REQUIRED_LOSS_FAMILIES

#: Loss families the materializer declares supported but does not pack into
#: the token stream (design §3, DEC-027-era "declared, not packed" class):
#: `training_materializer.SUPPORTED_LOSS_FAMILIES` minus `LOSS_FAMILY_IDS`.
#: Derived, never restated -- a declared family may carry a weight and enter
#: the allowlist without moving `loss_unit_schema_hash`.
DECLARED_LOSS_FAMILIES: tuple[str, ...] = tuple(
    sorted(set(training_materializer.SUPPORTED_LOSS_FAMILIES) - set(LOSS_FAMILY_IDS))
)

#: Loss families that supervise positive reconstruction of the document,
#: which RFC005 §26.6 forbids for `renderer_negative_v0`. Stated
#: explicitly, and cross-checked against required ∪ declared families at
#: import, so that adding a family forces someone to classify it here
#: rather than letting it slip past the negative-task rule unclassified.
#: `L_alignment` (RFC005 §27.3) supervises node/text/leaf region position --
#: document reconstruction, exactly like `L_typed` -- so it belongs here too
#: (DEC-043 §0, design Q12).
POSITIVE_RECONSTRUCTION_LOSS_FAMILIES: frozenset[str] = frozenset(
    {"L_structure", "L_text", "L_typed", "L_alignment"}
)

#: §26.6: the task kind that may carry no positive reconstruction loss.
NEGATIVE_TASK_KIND = "renderer_negative_v0"

_unclassified = sorted(
    (set(REQUIRED_LOSS_FAMILIES) | set(DECLARED_LOSS_FAMILIES))
    - POSITIVE_RECONSTRUCTION_LOSS_FAMILIES
)
if _unclassified:
    raise ImportError(
        "stage_policy.POSITIVE_RECONSTRUCTION_LOSS_FAMILIES does not classify "
        f"loss famil(ies) {', '.join(_unclassified)} -- RFC005 §26.6 needs to "
        f"know whether {NEGATIVE_TASK_KIND} may carry them, and this module "
        "will not guess"
    )
del _unclassified


class StagePolicyInvalid(ValueError):
    """A stage policy is incomplete, weights a family that does not exist,
    breaks an RFC005 §26.4/§26.6 rule, or fails its closed schema.

    M1-internal (design §9.3); not an RFC004 registry code.
    """


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _check_weights(policy: Mapping[str, Any]) -> None:
    weights = policy["loss_weights_milli"]
    allowed = set(REQUIRED_LOSS_FAMILIES) | set(DECLARED_LOSS_FAMILIES)
    missing = sorted(set(REQUIRED_LOSS_FAMILIES) - set(weights))
    if missing:
        raise StagePolicyInvalid(
            f"missing λ for loss famil(ies): {', '.join(missing)} -- RFC005 §26.4 "
            "requires every λ_f to be explicit and this producer refuses to "
            "default them"
        )
    unknown = sorted(set(weights) - allowed)
    if unknown:
        raise StagePolicyInvalid(
            f"λ given for unknown loss famil(ies): {', '.join(unknown)} -- "
            f"expected one of {', '.join(sorted(allowed))}"
        )
    for family in sorted(allowed & set(weights)):
        value = weights[family]
        if isinstance(value, bool) or not isinstance(value, int):
            raise StagePolicyInvalid(
                f"λ for {family} must be a true integer in milli-units "
                f"(1000 = λ 1.0), got {value!r}"
            )


def _check_dynamic_weighting(policy: Mapping[str, Any]) -> None:
    dynamic = policy["dynamic_weighting"]
    if dynamic is None:
        return
    if dynamic["lambda_min_milli"] > dynamic["lambda_max_milli"]:
        raise StagePolicyInvalid(
            f"dynamic_weighting bounds are inverted: lambda_min_milli "
            f"{dynamic['lambda_min_milli']} > lambda_max_milli "
            f"{dynamic['lambda_max_milli']} -- "
            "RFC005 §26.4 requires bounded behavior, and an empty interval "
            "bounds nothing"
        )


def _check_allowlist(policy: Mapping[str, Any]) -> None:
    allowed = set(REQUIRED_LOSS_FAMILIES) | set(DECLARED_LOSS_FAMILIES)
    seen_pairs: set[tuple[str, str]] = set()
    locator_family: dict[tuple[str, str], str] = {}
    for row in policy["task_view_allowlist"]:
        task_kind, family = row["task_kind"], row["loss_family"]
        if task_kind not in TASK_KIND:
            raise StagePolicyInvalid(
                f"{task_kind!r} is not an exact RFC003 task_kind -- RFC005 §26.6 "
                "forbids wildcards and loose names in a machine-readable policy; "
                f"expected one of {', '.join(sorted(TASK_KIND))}"
            )
        if family not in allowed:
            raise StagePolicyInvalid(
                f"task_view_allowlist row ({task_kind}, {family}) names an unknown "
                f"loss family -- expected one of {', '.join(sorted(allowed))}"
            )
        if (task_kind, family) in seen_pairs:
            raise StagePolicyInvalid(
                f"task_view_allowlist has two rows for ({task_kind}, {family}) -- "
                "a pair is enabled once or not at all"
            )
        seen_pairs.add((task_kind, family))
        if (
            task_kind == NEGATIVE_TASK_KIND
            and family in POSITIVE_RECONSTRUCTION_LOSS_FAMILIES
        ):
            raise StagePolicyInvalid(
                f"{NEGATIVE_TASK_KIND} may not be allowlisted with the positive "
                f"reconstruction loss family {family} (RFC005 §26.6)"
            )
        if not row["head_ids"] and not row["output_locators"]:
            raise StagePolicyInvalid(
                f"task_view_allowlist row ({task_kind}, {family}) names neither a "
                "head id nor an output locator, so it enables nothing -- RFC005 "
                "§26.6 disables an unspecified pair by omitting it, not by listing "
                "it empty"
            )
        for locator in row["output_locators"]:
            owner = locator_family.setdefault((task_kind, locator), family)
            if owner != family:
                raise StagePolicyInvalid(
                    f"output locator {locator!r} enters both {owner} and {family} "
                    f"for task_kind {task_kind} -- RFC005 §26.6: one output logit "
                    "must not enter two loss families"
                )


def validate_stage_policy(policy: Mapping[str, Any]) -> None:
    """Validate `policy` against `stage_policy.schema.json`, plus the
    §26.4/§26.6 rules the schema cannot express (module docstring)."""
    errors = list(_validator().iter_errors(policy))
    if errors:
        raise StagePolicyInvalid(
            f"schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )
    _check_weights(policy)
    _check_dynamic_weighting(policy)
    _check_allowlist(policy)


def build_stage_policy(
    *,
    policy_version: str,
    stage_id: str,
    loss_weights_milli: Mapping[str, Any],
    dynamic_weighting: Mapping[str, Any] | None,
    reads_validation_or_test_metrics: bool,
    task_view_allowlist: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble one stage policy, validated on the way out.

    Every argument is keyword-only and required; there are no defaults
    (module docstring). `dynamic_weighting=None` is the declaration that
    weighting is static -- it must be passed, not omitted.
    """
    policy = {
        "stage_policy_schema_version": STAGE_POLICY_SCHEMA_VERSION,
        "policy_version": policy_version,
        "stage_id": stage_id,
        "loss_weights_milli": dict(loss_weights_milli),
        "dynamic_weighting": (
            None if dynamic_weighting is None else dict(dynamic_weighting)
        ),
        "reads_validation_or_test_metrics": reads_validation_or_test_metrics,
        "task_view_allowlist": [dict(row) for row in task_view_allowlist],
    }
    validate_stage_policy(policy)
    return policy


def load_stage_policy(path: Path | str) -> dict[str, Any]:
    """Read a frozen stage policy document from disk, validated on the way in."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise StagePolicyInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise StagePolicyInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_stage_policy(payload)
    return payload


def stage_policy_hash(policy: Mapping[str, Any]) -> str:
    """`stage_policy_hash` over the whole validated policy."""
    validate_stage_policy(policy)
    return domain_hash(_STAGE_POLICY_DOMAIN, dict(policy))

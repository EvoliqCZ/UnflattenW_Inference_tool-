"""Frozen KL reference for S4.5 (RFC005-ERRATA-010 E4.1/E7.2; dormant).

The reference is the S4 checkpoint the policy was initialised from. Its
identity is compared against the POLICY's independently derived parent
identity (``init_from`` block / checkpoint sidecar), never against a hash the
reference computed about itself: a self-hash passes by construction, so
``check_reference_compatible`` refuses a reference compared with the very same
object.

The four identity fields are checked in a fixed order and each mismatch
raises :class:`ReferenceIncompatible` carrying ``.field``, so a test (or a
reader of a failure) can tell which cause fired. Nothing here loads a
checkpoint or touches the model; building the identities from real sidecars
is the run manifest's job (T7).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from torch import nn

RUN_CLASSES: Final = ("discovery", "promotion")
#: Identity fields compared reference-vs-policy, in check order.
IDENTITY_FIELDS: Final = (
    "ancestry_hash",
    "checkpoint_id",
    "typed_head_config_hash",
    "checkpoint_sha256",
)
#: Attribute of ``CandidateModel`` holding the frozen pretrained decoder.
BACKBONE_ATTR: Final = "_decode"


class ReferenceIncompatible(ValueError):
    """The KL reference does not match the policy's parent (``.field`` says why)."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"reference incompatible at {field}: {message}")
        self.field = field


class ReferenceNotFrozen(RuntimeError):
    """A reference module with trainable parameters or in train mode."""


class ReferenceBackboneNotShared(RuntimeError):
    """The reference does not reuse the policy's frozen backbone parameters."""


@dataclass(frozen=True, slots=True)
class ReferencePolicyIdentity:
    ancestry_hash: str
    checkpoint_id: str
    checkpoint_sha256: str
    typed_head_config_hash: str
    run_class: str

    def __post_init__(self) -> None:
        for name in self.__slots__:  # type: ignore[attr-defined]
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")


def check_reference_compatible(
    reference: ReferencePolicyIdentity,
    policy_identity: ReferencePolicyIdentity,
    *,
    run_class: str,
) -> None:
    """Refuse a reference that is not the policy's own parent.

    ``policy_identity`` is the identity the policy was initialised from, derived
    independently of ``reference``.

    Limitation: "independently derived" is enforced by object identity only
    (``reference is policy_identity`` is refused); a copy of the reference with
    equal fields is compared field-by-field like any other value. Nothing here
    can prove the caller did not build ``policy_identity`` from the reference's
    own values, so the caller must take ``policy_identity`` from the checkpoint
    sidecar / T7 run manifest (``policy_parent``), never from the reference.

    ``run_class`` is the S4.5 run's class: a
    ``promotion`` run may not sit on a ``discovery`` parent (RFC005 s12.1,
    ERRATA-005), on either the reference or the policy side.
    """
    if run_class not in RUN_CLASSES:
        raise ValueError(f"run_class {run_class!r} not in {RUN_CLASSES}")
    if reference is policy_identity:
        raise ReferenceIncompatible(
            "self_comparison",
            "the reference was compared with itself; derive the policy identity independently",
        )
    for name in IDENTITY_FIELDS:
        ref_value, pol_value = getattr(reference, name), getattr(policy_identity, name)
        if ref_value != pol_value:
            raise ReferenceIncompatible(
                name, f"reference {ref_value!r} != policy parent {pol_value!r}"
            )
    if run_class == "promotion":
        for side, identity in (("reference", reference), ("policy parent", policy_identity)):
            if identity.run_class == "discovery":
                raise ReferenceIncompatible(
                    "run_class",
                    f"{side} is a discovery-class checkpoint but the run is 'promotion'",
                )


def assert_reference_frozen(module: nn.Module) -> int:
    """Number of parameters, after checking all are ``requires_grad=False`` and the
    module (and every submodule) is in eval mode."""
    params = list(module.parameters())
    if not params:
        raise ReferenceNotFrozen("reference module has no parameters: nothing was frozen")
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    if trainable:
        raise ReferenceNotFrozen(f"reference parameters with requires_grad=True: {trainable[:5]}")
    training = [n or "<root>" for n, m in module.named_modules() if m.training]
    if training:
        raise ReferenceNotFrozen(f"reference modules not in eval mode: {training[:5]}")
    return len(params)


def freeze_reference_parameters(module: nn.Module) -> int:
    """Freeze in place (``requires_grad_(False)`` + ``eval()``), verify, return
    the parameter count."""
    if not any(True for _ in module.parameters()):
        raise ReferenceNotFrozen("reference module has no parameters: nothing was frozen")
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return assert_reference_frozen(module)


def _module_path_shared(policy: Any, reference: Any, backbone_attr: str) -> int | None:
    """The ``nn.Module`` backbone path; ``None`` when ``policy`` has no such module
    attribute (the closure backbone of a real ``QwenDenseCandidateModel``)."""
    if not isinstance(getattr(policy, backbone_attr, None), nn.Module):
        return None
    modules = []
    for label, owner in (("policy", policy), ("reference", reference)):
        sub = getattr(owner, backbone_attr, None)
        if not isinstance(sub, nn.Module):
            raise ReferenceBackboneNotShared(f"{label} has no module attribute {backbone_attr!r}")
        modules.append(sub)
    pol, ref = (dict(m.named_parameters()) for m in modules)
    if pol.keys() != ref.keys() or not pol:
        raise ReferenceBackboneNotShared(
            f"backbone parameter names differ or are empty: {sorted(pol.keys() ^ ref.keys())[:5]}"
        )
    not_same = [n for n in pol if pol[n] is not ref[n]]
    if not_same:
        raise ReferenceBackboneNotShared(
            "reference backbone parameters are not the same objects as the policy's: "
            f"{not_same[:5]}"
        )
    trainable = [n for n, p in pol.items() if p.requires_grad]
    if trainable:
        raise ReferenceBackboneNotShared(f"shared backbone has requires_grad=True: {trainable[:5]}")
    return len(pol)


#: Closure-backbone identity checks, in order; the raised message names the one that fired.
CLOSURE_CHECKS: Final = (
    "_decode is",
    "_inject_visual is",
    "qwen_embedding_weight is",
    "qwen_lm_head_weight is",
    "qwen buffers requires_grad",
    "candidate parameters frozen",
)


def _closure_path_shared(
    policy: Any, reference: Any, candidate_parameters: Mapping[str, Any] | None
) -> int:
    """Closure backbone (``candidate_model.py``: ``_decode``/``_inject_visual`` are
    plain functions, the Qwen embedding/lm-head weights are non-persistent buffers
    of ``composite_head``). Identity is checked object by object; value equality
    is never accepted (a copy would double VRAM)."""
    for attr in ("_decode", "_inject_visual"):
        if getattr(policy, attr, None) is None or getattr(policy, attr) is not getattr(
            reference, attr, None
        ):
            raise ReferenceBackboneNotShared(f"check '{attr} is' failed: not the same object")
    pol_head = getattr(policy, "composite_head", None)
    ref_head = getattr(reference, "composite_head", None)
    buffers = []
    for name in ("qwen_embedding_weight", "qwen_lm_head_weight"):
        pol_buf = getattr(pol_head, name, None)
        if pol_buf is None or pol_buf is not getattr(ref_head, name, None):
            raise ReferenceBackboneNotShared(
                f"check '{name} is' failed: composite_head buffer is not the same tensor"
            )
        buffers.append((name, pol_buf))
    live = [name for name, buf in buffers if buf.requires_grad]
    if live:
        raise ReferenceBackboneNotShared(
            f"check 'qwen buffers requires_grad' failed: {live} require grad"
        )
    if not candidate_parameters:
        raise ReferenceBackboneNotShared(
            "check 'candidate parameters frozen' failed: candidate_parameters missing or empty"
        )
    trainable = [n for n, p in candidate_parameters.items() if p.requires_grad]
    if trainable:
        raise ReferenceBackboneNotShared(
            f"check 'candidate parameters frozen' failed: requires_grad=True: {trainable[:5]}"
        )
    return len(candidate_parameters)


def assert_reference_shares_frozen_backbone(
    policy: Any,
    reference: Any,
    *,
    backbone_attr: str = BACKBONE_ATTR,
    candidate_parameters: Mapping[str, Any] | None = None,
) -> int:
    """The reference reuses the policy's frozen backbone by identity (no second
    copy) and it is all frozen. Returns the number of shared parameters.

    Dispatch: an ``nn.Module`` backbone attribute keeps the parameter-by-parameter
    path; otherwise (the real closure backbone) the identity checks of
    ``CLOSURE_CHECKS`` run and the error names the one that fired.
    ``candidate_parameters`` is ``QwenCandidate.named_parameters`` (all must be
    frozen); omitted -> the check refuses rather than assume.
    """
    shared = _module_path_shared(policy, reference, backbone_attr)
    if shared is not None:
        return shared
    return _closure_path_shared(policy, reference, candidate_parameters)

"""Open the S4.5 policy candidate and its frozen KL reference from a TRAINING
checkpoint (design DP1 / invariant 8; dormant, CPU-testable).

Why this does not call ``candidate_runtime.load_candidate_runtime``: that
function serves a *decode* runtime and refuses a state holding
``optimizer_state_dict`` (``verify_decode_bindings``), which every training
checkpoint carries, and it builds its own model. The sequence here follows
``eval/s2_gate/cli.py build_decode_dependencies`` instead: sidecar first, then
state, then candidate, vocabulary, controller, model.

Order inside :meth:`CandidateOpener.__call__` (each step refuses before the
next, costlier one runs):

1. :func:`load_trusted_training_checkpoint` -- ``verify_sidecar`` (file bytes
   hashed, ancestry, expectations) and only then ``torch.load``;
2. typed-head config hash equals the expected one (the parent identity's);
3. checkpoint ``visual_feature_tap`` agrees with the explicit ``deepstack`` flag
   and the level-router policy is resolved, both before any GPU load;
4. vocabulary from the allowlist, hash compared with the expected one;
5. the candidate is loaded (injected for tests), its width compared with the
   allowlist's, the controller and the policy are built.

The reference (:func:`make_open_reference`) is built from the SAME candidate
object after the policy placed it (``reference_loader`` documents why that
order is mandatory).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from unflatten.model import checkpoint_ancestry as ca

if TYPE_CHECKING:  # pragma: no cover
    from torch import nn

    from unflatten.model.candidate_model import QwenDenseCandidateModel
    from unflatten.model.rl.real_run_ports import OpenedCandidate
    from unflatten.model.rl.reference import ReferencePolicyIdentity


class CandidateOpenRefused(ValueError):
    """The candidate/reference cannot be opened from the given checkpoint."""


def load_trusted_training_checkpoint(
    checkpoint_path: Path | str,
    *,
    expect_base_ancestry_hash: str,
    expect_s1_run_manifest_hash: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(state, sidecar_doc)`` of a training checkpoint.

    The ONLY place a training checkpoint is unpickled
    (``torch.load(weights_only=False)`` executes arbitrary pickle code), and it
    runs only after ``verify_sidecar`` hashed the file bytes against the sidecar.
    ``verify_sidecar`` also wants the ancestry hash embedded *in* the file, which
    is unreadable before the load; it is therefore checked against the
    sidecar's own value first (so the bytes are verified) and the embedded value
    is compared after the load.
    """
    import json

    import torch

    path = Path(checkpoint_path)
    try:
        sidecar_hash = json.loads(ca.sidecar_path(path).read_text()).get("ancestry_hash", "")
    except (OSError, ValueError, AttributeError):
        sidecar_hash = ""  # verify_sidecar raises with the real reason
    doc = ca.verify_sidecar(
        path,
        expect_base_ancestry_hash=expect_base_ancestry_hash,
        expect_s1_run_manifest_hash=expect_s1_run_manifest_hash,
        embedded_ancestry_hash=sidecar_hash,
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise CandidateOpenRefused(f"checkpoint {path} did not unpickle to a state dict")
    if state.get("ancestry_hash") != doc["ancestry_hash"]:
        raise CandidateOpenRefused(
            f"ancestry_hash embedded in {path.name} {state.get('ancestry_hash')!r} != "
            f"sidecar {doc['ancestry_hash']!r}"
        )
    return state, doc


def reference_identity_from_sidecar(
    sidecar_path: Path | str,
    state: Mapping[str, Any],
    companion: Mapping[str, Any],
    *,
    expect_base_ancestry_hash: str,
    expect_s1_run_manifest_hash: str | None,
) -> ReferencePolicyIdentity:
    """The parent identity from its verified sidecar (never from a model).

    ``verify_sidecar`` refuses when the checkpoint file's sha256 differs from the
    sidecar's. ``state`` is the already-loaded checkpoint (its embedded ancestry
    hash is checked); ``companion`` supplies ``run_class``.
    """
    from unflatten.model.rl.reference import ReferencePolicyIdentity

    sidecar = Path(sidecar_path)
    suffix = ca.SIDECAR_SUFFIX
    if not str(sidecar).endswith(suffix):
        raise CandidateOpenRefused(f"{sidecar} is not a checkpoint ancestry sidecar ({suffix})")
    checkpoint = Path(str(sidecar)[: -len(suffix)])
    doc = ca.verify_sidecar(
        checkpoint,
        expect_base_ancestry_hash=expect_base_ancestry_hash,
        expect_s1_run_manifest_hash=expect_s1_run_manifest_hash,
        embedded_ancestry_hash=state.get("ancestry_hash") or "",
    )
    block = doc["checkpoint"]
    return ReferencePolicyIdentity(
        ancestry_hash=doc["ancestry_hash"],
        checkpoint_id=block["checkpoint_id"],
        checkpoint_sha256=block["checkpoint_sha256"],
        typed_head_config_hash=state.get("typed_head_config_hash") or "",
        run_class=companion.get("run_class") or "",
    )


class CandidateOpener:
    """``Callable[[], OpenedCandidate]`` that remembers the candidate and state
    it opened, so :func:`make_open_reference` can build over the same object."""

    def __init__(
        self,
        *,
        checkpoint_path: Path | str,
        expect_base_ancestry_hash: str,
        expect_s1_run_manifest_hash: str | None,
        expect_typed_head_config_hash: str,
        deepstack: bool,
        device: str,
        tokenizer_artifact_set_hash: str,
        allowlist_payload: Mapping[str, Any],
        expect_vocabulary_hash: str,
        capability_profile_hash: str,
        load_candidate: Callable[[str], Any] | None = None,
        load_tokenizer: Callable[[], Any] | None = None,
    ) -> None:
        self._cfg = dict(locals())
        del self._cfg["self"]
        self.opened: OpenedCandidate | None = None
        self.candidate: Any = None
        self.state: dict[str, Any] | None = None
        self.deepstack = deepstack
        self.device = device
        self.level_router_policy: Any = None

    def __call__(self) -> OpenedCandidate:
        if self.opened is None:
            self.opened = self._open()
        return self.opened

    def _vocabulary_and_profile(self):
        from unflatten.model import composite_vocab as cv
        from unflatten.model.capability_profile import resolve_capability_profile_by_hash
        from unflatten.model.s1_runner import build_controller_for_profile

        cfg = self._cfg
        payload = cfg["allowlist_payload"]
        vocabulary = cv.make_composite_vocabulary(
            tokenizer_artifact_set_hash=cfg["tokenizer_artifact_set_hash"],
            allowlist_payload=payload,
        )
        if vocabulary.artifact_hash != cfg["expect_vocabulary_hash"]:
            raise CandidateOpenRefused(
                f"vocabulary.artifact_hash {vocabulary.artifact_hash!r} != expected "
                f"{cfg['expect_vocabulary_hash']!r}: not the vocabulary the cache was built with"
            )
        profile = resolve_capability_profile_by_hash(cfg["capability_profile_hash"])
        return vocabulary, profile, build_controller_for_profile(vocabulary, profile)

    def _open(self) -> OpenedCandidate:
        from unflatten.model import candidate_runtime as cr
        from unflatten.model.rl.real_run_ports import OpenedCandidate
        from unflatten.model.rl.reference_loader import build_policy_model

        cfg = self._cfg
        state, _doc = load_trusted_training_checkpoint(
            cfg["checkpoint_path"],
            expect_base_ancestry_hash=cfg["expect_base_ancestry_hash"],
            expect_s1_run_manifest_hash=cfg["expect_s1_run_manifest_hash"],
        )
        if state.get("typed_head_config_hash") != cfg["expect_typed_head_config_hash"]:
            raise CandidateOpenRefused(
                f"checkpoint typed_head_config_hash {state.get('typed_head_config_hash')!r} != "
                f"expected {cfg['expect_typed_head_config_hash']!r}"
            )
        cr.require_runtime_deepstack_matches(state, deepstack=self.deepstack)
        router = cr.runtime_level_router_policy(state)
        vocabulary, profile, controller = self._vocabulary_and_profile()
        loader = cfg["load_candidate"]
        if loader is None:
            from unflatten.model.candidate_loader import load_qwen_candidate

            def loader(device: str) -> Any:
                return load_qwen_candidate(device=device)

        candidate = loader(self.device)
        native = cfg["allowlist_payload"]["native_vocab_size"]
        if native != candidate.vocab_size:
            raise CandidateOpenRefused(
                f"allowlist native_vocab_size {native} != candidate vocab_size "
                f"{candidate.vocab_size}"
            )
        if self.deepstack:
            cr.require_runtime_deepstack_matches(
                state,
                deepstack=True,
                deepstack_visual_indexes=tuple(candidate.deepstack_visual_indexes),
            )
        tokenizer_loader = cfg["load_tokenizer"]
        if tokenizer_loader is None:
            from unflatten.model.qwen_tokenizer_artifacts import load_pinned_tokenizer

            def tokenizer_loader() -> Any:
                return load_pinned_tokenizer()[0]

        state = {k: v for k, v in state.items() if k != "optimizer_state_dict"}
        policy = build_policy_model(
            candidate, state, self.device, level_router_policy=router, deepstack=self.deepstack
        )
        self.candidate, self.state, self.level_router_policy = candidate, state, router
        return OpenedCandidate(
            policy=policy,
            vocabulary=vocabulary,
            controller=controller,
            tokenizer=tokenizer_loader(),
            capability_profile_id=profile.profile_id,
            image_token_id=candidate.image_token_id,
            candidate_parameters=candidate.named_parameters,
        )


def make_open_candidate(**kwargs: Any) -> CandidateOpener:
    """The ``open_candidate`` port; see :class:`CandidateOpener` for the arguments."""
    return CandidateOpener(**kwargs)


def make_open_reference(
    opener: CandidateOpener,
    *,
    reference_identity: ReferencePolicyIdentity,
    policy_parent_identity: ReferencePolicyIdentity,
    run_class: str,
) -> Callable[[OpenedCandidate], nn.Module]:
    """The ``open_reference`` port: a frozen second model over the candidate
    ``opener`` opened, built AFTER the policy placed it."""

    def open_reference(opened: OpenedCandidate) -> nn.Module:
        from unflatten.model.rl.reference_loader import build_reference_model

        if opener.opened is None or opener.state is None or opened is not opener.opened:
            raise CandidateOpenRefused(
                "open_reference was handed an OpenedCandidate this opener did not produce"
            )
        return build_reference_model(
            opener.candidate,
            opener.state,
            opener.device,
            policy=cast("QwenDenseCandidateModel", opened.policy),
            reference_identity=reference_identity,
            policy_parent_identity=policy_parent_identity,
            run_class=run_class,
            level_router_policy=opener.level_router_policy,
            deepstack=opener.deepstack,
        )

    return open_reference

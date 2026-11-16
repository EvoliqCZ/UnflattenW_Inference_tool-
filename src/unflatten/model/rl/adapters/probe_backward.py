"""Forward+backward block of the S4.5 probe (design P10 (b), DEC-082 / A v4 point 11).

Measures the peak VRAM and time of one teacher-forced forward and one backward of a
sampled group (``group_size`` candidates of one page) for each prefill chunk -- and
nothing else: there is NO optimizer, the parameter digest is compared before and after,
every ``.grad`` is dropped right after each backward (``zero_grad(set_to_none=True)``)
and a gradient left behind is a defect, not a result. The block runs only when
:func:`authorised_decision` resolves ``--backward-decision`` to a DECIDED/RATIFIED decision
whose text names both ``backward`` and ``A v4``; otherwise the probe reports
``"backward": "not_authorised"``.

This module is checked by ``probe_backward_violations`` (the strict gradient scanner minus
the ``.backward`` item): ``.step``, optim imports and ``*optimizer*`` identifiers are banned.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from unflatten.model.rl.adapters import launch as lp
from unflatten.model.rl.adapters import probe_inference as pi

NOT_AUTHORISED: Final = "not_authorised"
MEASURED: Final = "measured"
_REQUIRED_WORDS: Final = ("backward", "A v4")


class BackwardRefused(ValueError):
    """The decision named by ``--backward-decision`` does not authorise the block."""


class BackwardLeftGradients(RuntimeError):
    """A ``.grad`` survived ``zero_grad(set_to_none=True)`` (a defect of the probe)."""


def authorised_decision(decisions_dir: Path | str, decision_id: str | None) -> Path | None:
    """``None`` without a flag; else the DECIDED/RATIFIED decision file naming the permission.

    ``resolve_decision`` refuses zero/several files and any status but DECIDED/RATIFIED.
    """
    if decision_id is None:
        return None
    path = lp.resolve_decision(decisions_dir, decision_id)
    text = path.read_text(encoding="utf-8")
    for word in _REQUIRED_WORDS:
        if word.lower() not in text.lower():
            raise BackwardRefused(
                f"decision file {path.name} does not name {word!r}: it does not authorise "
                "a forward+backward probe without an optimizer step"
            )
    return path


class Capture:
    """Remembers what ``open_candidate`` / ``open_reference`` returned (one open, two blocks)."""

    def __init__(self, ports: Any) -> None:
        self.opened: Any = None
        self.reference: Any = None
        self.ports = dataclasses.replace(
            ports, open_candidate=self._open(ports.open_candidate),
            open_reference=self._reference(ports.open_reference),
        )  # fmt: skip

    def _open(self, inner: Callable[[], Any]) -> Callable[[], Any]:
        def open_candidate() -> Any:
            self.opened = inner()
            return self.opened

        return open_candidate

    def _reference(self, inner: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def open_reference(opened: Any) -> Any:
            self.reference = inner(opened)
            return self.reference

        return open_reference


def _zero_grad(model: Any) -> None:
    model.zero_grad(set_to_none=True)


def _leftover_grads(model: Any) -> list[str]:
    return [n for n, p in model.named_parameters() if p.grad is not None]


def _forward_backward(micro: Any) -> dict[str, Any]:
    """One teacher-forced forward under grad and one backward of a surrogate loss.

    The loss is the mean negative legal log-probability of the sampled tokens: it only
    has to exercise the same graph as the RL loss; its value is never used.
    """
    import torch

    new_logprobs = micro.new_logprobs_fn()
    legal = micro.rollout.legal_mask
    picked = torch.where(legal, new_logprobs, torch.zeros_like(new_logprobs))
    loss = -picked.sum() / legal.sum().clamp(min=1)
    requires_grad = bool(loss.requires_grad)
    if requires_grad:
        loss.backward()
    return {"loss_requires_grad": requires_grad}


def _cuda() -> Any | None:
    import torch

    return torch.cuda if torch.cuda.is_available() else None


def _context(
    *, ports: Any, artefacts: Any, opened: Any, reference: Any, rollout_policy: Mapping[str, Any],
    run_seed: int, heads: Sequence[str], render_dir: Path,
) -> Any:  # fmt: skip
    from unflatten.model.rl.loss import RlLossProvenance
    from unflatten.model.rl.real_run_ports import LoopContext

    return LoopContext(
        opened=opened,
        reference=reference,
        rollout_policy=rollout_policy,
        reward_policy=artefacts.reward_policy,
        heads=tuple(sorted(set(heads))),
        run_seed=run_seed,
        provenance=RlLossProvenance(
            reward_policy_hash=artefacts.reward_policy["reward_policy_hash"],
            rollout_policy_hash=rollout_policy["rollout_policy_hash"],
            reference_checkpoint_id=artefacts.reference.checkpoint_id,
            reference_ancestry_hash=artefacts.reference.ancestry_hash,
            typed_head_config_hash=artefacts.policy_parent.typed_head_config_hash,
        ),
        supervised_weight_micro=0,
        ports=pi._probe_ports(ports, render_dir),
    )


def _chunk_policy(base: Mapping[str, Any], chunk: int | None) -> dict[str, Any]:
    from unflatten.model.rl import policies

    profile = policies.build_execution_profile(
        prefill_chunk_tokens=chunk, deepstack=base["execution_profile"]["deepstack"]
    )
    return pi.reseal_rollout_policy(
        {**base, "execution_profile": profile}, int(base["temperature_micro"])
    )


def _measure_chunk(ctx: Any, page: Any, *, chunk: int | None, timer: Callable[[], float],
                   epoch: int) -> dict[str, Any]:  # fmt: skip
    import torch

    from unflatten.model.rl.real_run_loop import microbatch_from_group, rollout_group

    with torch.inference_mode():
        group = rollout_group(ctx, page, epoch=epoch)
    micro, _record = microbatch_from_group(ctx, group)
    cuda = _cuda()
    if cuda is not None:
        cuda.reset_peak_memory_stats()
    block: dict[str, Any] = {
        "prefill_chunk_tokens": chunk,
        "phase": "backward",
        "oom": False,
        "group_size": len(group.trajectories),
        "candidate_tokens": int(micro.rollout.token_ids.shape[1]),
        "max_steps": int(ctx.rollout_policy["max_steps"]),
    }
    started = timer()
    try:
        block.update(_forward_backward(micro))
    except torch.cuda.OutOfMemoryError:
        block["oom"] = True
    finally:
        _zero_grad(ctx.opened.policy)
        if cuda is not None:
            cuda.synchronize()
    block["timings_ms"] = {"forward_backward": (timer() - started) * 1000.0}
    block["vram_peak_bytes"] = None if cuda is None else int(cuda.max_memory_allocated())
    if block["oom"] and cuda is not None:
        cuda.empty_cache()
    left = _leftover_grads(ctx.opened.policy)
    if left:
        raise BackwardLeftGradients(f"gradients left after zero_grad(set_to_none=True): {left[:3]}")
    return block


def run_backward_block(
    *,
    ports: Any,
    artefacts: Any,
    opened: Any,
    reference: Any,
    pages: Sequence[Any],
    chunks: Sequence[int | None],
    render_dir: Path,
    decision_id: str,
    run_seed: int = 0,
    selected_typed_heads: Sequence[str] = (),
    timer: Callable[[], float] = time.perf_counter,
    epoch: int = 0,
) -> dict[str, Any]:
    """The ``backward`` block of ``probe.json`` (the caller has checked the authorisation).

    One page (the first of ``pages``), one sampled group, for each chunk of ``chunks``.
    """
    if not pages:
        raise BackwardRefused("the backward block needs at least one page")
    digest_before = pi.parameter_digest(opened.policy)
    results = []
    for chunk in chunks:
        ctx = _context(
            ports=ports, artefacts=artefacts, opened=opened, reference=reference,
            rollout_policy=_chunk_policy(artefacts.rollout_policy, chunk), run_seed=run_seed,
            heads=selected_typed_heads, render_dir=render_dir,
        )  # fmt: skip
        results.append(_measure_chunk(ctx, pages[0], chunk=chunk, timer=timer, epoch=epoch))
    digest_after = pi.parameter_digest(opened.policy)
    if digest_before != digest_after:
        raise pi.ProbeParametersMoved("parameters changed during the backward probe")
    return {
        "status": MEASURED,
        "decision": decision_id,
        "optimizer": None,
        "parameter_digest": {
            "before": digest_before,
            "after": digest_after,
            "unchanged": digest_before == digest_after,
        },
        "chunks": results,
    }


__all__ = [
    "NOT_AUTHORISED",
    "BackwardLeftGradients",
    "BackwardRefused",
    "Capture",
    "authorised_decision",
    "run_backward_block",
]

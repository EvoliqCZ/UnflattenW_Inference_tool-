"""Inference-only S4.5 probe library (design P10 (a), DEC-047 addendum A v3 point 8).

Measures what a real run would cost and how its reward signal looks -- rollout
time and peak VRAM, reward-component percentiles, zero-advantage rate, loop-guard
interventions, ``gold_chain_failed`` -- WITHOUT any gradient construct: the
candidate is opened for inference only (``torch.inference_mode``), there is no
ledger, no checkpoint write and nothing is written outside ``out_dir``. The
forward+backward block is a different, separately authorised module.

Gate order, every step a named refusal and all BEFORE ``open_candidate``:
window record loaded -> checkpoint id resolved -> window names that checkpoint
and ``now`` lies inside it -> ``out_dir`` created (must be new).

For each temperature the rollout policy is re-sealed through
:func:`~unflatten.model.rl.policies.build_rollout_policy` (its hash changes; the
probe binds nothing, its output is not a run artefact). All numbers are integers
(micro units); floats exist only under ``timings_ms``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from unflatten.model.rl.adapters import taxonomy_producer as tp

MICRO: Final = 1_000_000

PROBE_SCHEMA: Final = "unflatten.rl.probe-inference/1"
PROBE_JSON: Final = "probe.json"
PROBE_MD: Final = "probe.md"
PERCENTILES_MICRO: Final = (100_000, 500_000, 900_000)
#: the ports the probe needs; every other port (ledger, checkpoint dir, ...) is dropped
_PROBE_PORTS: Final = (
    "open_candidate",
    "open_reference",
    "build_visual",
    "build_gold",
    "loop_guard_factory",
    "render_probe",
)
_ROLLOUT_FIELDS: Final = (
    "top_p_micro",
    "group_size",
    "rollout_budget",
    "max_steps",
    "loop_guard_arm",
    "execution_profile",
    "typed_head_sampling",
    "seed_scheme",
    "renormalise",
    "loop_guards_in_policy",
    "render_probe",
    "status",
)


class ProbeRefused(ValueError):
    """The probe inputs or outputs break its contract."""


class ProbeParametersMoved(RuntimeError):
    """The parameter digest changed during an inference-only probe (a defect)."""


def parameter_digest(module: Any) -> str:
    """sha256 over ``(name, raw bytes)`` of every parameter -- the same value as
    ``s4_5_train._parameter_digest`` (re-implemented: that helper is private and
    its module stays byte-unchanged; a parity test pins the equality)."""
    h = hashlib.sha256()
    for name, p in module.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def reseal_rollout_policy(base: Mapping[str, Any], temperature_micro: int) -> dict[str, Any]:
    """``base`` with ``temperature_micro`` replaced, sealed and verified by ``policies``."""
    from unflatten.model.rl import policies

    kwargs = {k: base[k] for k in _ROLLOUT_FIELDS}
    sealed = policies.build_rollout_policy(temperature_micro=temperature_micro, **kwargs)
    policies.verify_rollout_policy(sealed)
    return sealed


def percentile_micro(values: Sequence[int], at_micro: int) -> int | None:
    """Nearest-rank percentile of integers (``None`` for an empty sample)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-at_micro * len(ordered) // MICRO))
    return ordered[rank - 1]


def _percentiles(values: Sequence[int]) -> dict[str, int | None]:
    return {f"p{at // 10_000}": percentile_micro(values, at) for at in PERCENTILES_MICRO}


def _is_zero_advantage(rewards: Sequence[int]) -> bool:
    """The loss's rule: fewer than two included candidates or all rewards equal."""
    return len(rewards) < 2 or max(rewards) == min(rewards)


def _group_reward(
    status: str, scalar: int | None, gate_failed_reward_micro: int | None
) -> int | None:
    if status == "scored" and scalar is not None:
        return scalar
    if status.startswith("gate_failed:"):
        return gate_failed_reward_micro
    return None  # insufficient_evidence: excluded, never a number


class _Tally:
    """Integer accumulators of one temperature."""

    def __init__(self, gate_failed_reward_micro: int | None) -> None:
        self.gate_failed_reward_micro = gate_failed_reward_micro
        self.statuses: dict[str, int] = {}
        self.outcomes: dict[str, int] = {}
        self.gate_statuses: dict[str, dict[str, int]] = {}
        self.failed_gates: list[dict[str, Any]] = []
        self.components: dict[str, list[int]] = {}
        self.scalars: list[int] = []
        self.groups = 0
        self.zero_groups = 0
        self.rollouts = 0
        self.guarded = 0
        self.intervened = 0
        self.unknown_reward = False

    def add(self, group: Any) -> None:
        rewards: list[int] = []
        for t, s in zip(group.trajectories, group.scored, strict=True):
            self.rollouts += 1
            key = "gate_failed" if s.status.startswith("gate_failed:") else s.status
            self.statuses[key] = self.statuses.get(key, 0) + 1
            self.outcomes[s.outcome] = self.outcomes.get(s.outcome, 0) + 1
            self._tally_gates(s)
            if s.scalar_micro is not None:
                self.scalars.append(s.scalar_micro)
            for name, comp in (s.components or {}).items():
                if comp["state"] == "scored":
                    self.components.setdefault(name, []).append(comp["micro"])
            reward = _group_reward(s.status, s.scalar_micro, self.gate_failed_reward_micro)
            if reward is not None:
                rewards.append(reward)
            elif key == "gate_failed":
                self.unknown_reward = True  # the penalty is not frozen: rate not computable
            if t.guard_record is not None:
                self.guarded += 1
                self.intervened += 1 if len(t.substituted_positions) > 0 else 0
        self.groups += 1
        self.zero_groups += 1 if _is_zero_advantage(rewards) else 0

    def _tally_gates(self, scored: Any) -> None:
        """Every gate's status is counted; a failing gate keeps its evidence per rollout."""
        for g in scored.gates or ():
            by_status = self.gate_statuses.setdefault(g["gate"], {})
            by_status[g["status"]] = by_status.get(g["status"], 0) + 1
            if g["status"] == "fail":
                self.failed_gates.append(
                    {"rollout": self.rollouts, "gate": g["gate"], "evidence": dict(g["evidence"])}
                )

    def block(self) -> dict[str, Any]:
        known = self.groups and not self.unknown_reward
        rate = self.zero_groups * MICRO // self.groups if known else None
        share = self.intervened * MICRO // self.guarded if self.guarded else None
        return {
            "rollouts": self.rollouts,
            "status_counts": dict(sorted(self.statuses.items())),
            "outcome_counts": dict(sorted(self.outcomes.items())),
            "gate_status_counts": {
                g: dict(sorted(v.items())) for g, v in sorted(self.gate_statuses.items())
            },
            "failed_gates": self.failed_gates,
            "scalar_micro_percentiles": _percentiles(self.scalars),
            "component_micro_percentiles": {
                n: {"observed": len(v), **_percentiles(v)}
                for n, v in sorted(self.components.items())
            },
            "zero_advantage": {
                "groups": self.groups,
                "zero_groups": self.zero_groups if known else None,
                "rate_micro": rate,
            },
            "guard": {
                "rollouts_with_guard": self.guarded,
                "rollouts_with_intervention": self.intervened,
                "intervention_share_micro": share,
            },
        }


def _cuda() -> Any | None:
    import torch

    return torch.cuda if torch.cuda.is_available() else None


def _probe_ports(ports: Any, render_dir: Path) -> Any:
    over = {f.name: None for f in dataclasses.fields(ports) if f.name not in _PROBE_PORTS}
    return dataclasses.replace(ports, **{**over, "render_dir": render_dir})


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise ProbeRefused(f"{path} exists: a probe output is never overwritten")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _summary_md(doc: Mapping[str, Any]) -> str:
    lines = [
        "# S4.5 inference probe (backward: not_authorised)",
        f"checkpoint {doc['window']['checkpoint_id']} precision {doc['window']['precision']}",
        f"pages {doc['pages']} group_size {doc['group_size']} "
        f"parameters_unchanged {doc['parameter_digest']['unchanged']}",
    ]
    for block in doc["temperatures"]:
        zero = block["zero_advantage"]
        lines.append(
            f"T={block['temperature_micro']}: rollouts {block['rollouts']} "
            f"status {block['status_counts']} zero_adv_rate_micro {zero['rate_micro']} "
            f"guard_share_micro {block['guard']['intervention_share_micro']} "
            f"vram_peak {block['vram_peak_bytes']} "
            f"gold_chain_failed {block['gold_chain_failed']}"
        )
    return "\n".join(lines[:20]) + "\n"


def _measure_temperature(
    ctx: Any,
    pages: Sequence[Any],
    *,
    temperature_micro: int,
    gate_failed_reward_micro: int | None,
    gold_counters: Callable[[], Mapping[str, int]] | None,
    timer: Callable[[], float],
    epoch: int,
) -> dict[str, Any]:
    import torch

    from unflatten.model.rl.real_run_loop import rollout_group

    cuda = _cuda()
    if cuda is not None:
        cuda.reset_peak_memory_stats()
    tally = _Tally(gate_failed_reward_micro)
    page_ms: list[float] = []
    with torch.inference_mode():
        for page in pages:
            started = timer()
            group = rollout_group(ctx, page, epoch=epoch)
            page_ms.append((timer() - started) * 1000.0)
            tally.add(group)
    counters = dict(gold_counters()) if gold_counters is not None else None
    return {
        "temperature_micro": temperature_micro,
        "rollout_policy_hash": ctx.rollout_policy["rollout_policy_hash"],
        **tally.block(),
        "gold_chain_failed": None if counters is None else counters.get("gold_chain_failed", 0),
        "vram_peak_bytes": None if cuda is None else int(cuda.max_memory_allocated()),
        "timings_ms": {
            "rollout_group_total": sum(page_ms),
            "rollout_group_per_page_mean": sum(page_ms) / len(page_ms) if page_ms else None,
        },
    }


def run_inference_probe(
    *,
    ports: Any,
    artefacts: Any,
    pages: Iterable[Any],
    temperatures_micro: Sequence[int],
    out_dir: Path | str,
    clock: Callable[[], datetime],
    window_record_path: Path | str,
    resolve_checkpoint_id: Callable[[], str],
    run_seed: int = 0,
    selected_typed_heads: Iterable[str] = (),
    gold_counters: Callable[[], Mapping[str, int]] | None = None,
    timer: Callable[[], float] = time.perf_counter,
    epoch: int = 0,
) -> dict[str, Any]:
    """Open the candidate for inference only and write ``probe.json`` / ``probe.md``.

    ``resolve_checkpoint_id`` is the ancestry-derived id of the checkpoint the
    ports will open (``taxonomy_producer.resolve_checkpoint_id`` bound to its
    paths). Returns the ``probe.json`` document.
    """
    from unflatten.model.rl.loss import RlLossProvenance
    from unflatten.model.rl.real_run_ports import LoopContext

    temperatures = tuple(temperatures_micro)
    if not temperatures or len(set(temperatures)) != len(temperatures):
        raise ProbeRefused(f"temperatures_micro must be non-empty and distinct: {temperatures}")
    page_list = list(pages)
    window = tp.load_window_record(window_record_path)
    tp.require_window(window, checkpoint_id=resolve_checkpoint_id(), now=clock())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)  # new directory, before any weight is read

    probe_ports = _probe_ports(ports, out / "render")
    probe_ports.render_dir.mkdir()
    opened = probe_ports.open_candidate()
    reference = probe_ports.open_reference(opened)
    digest_before = parameter_digest(opened.policy)
    base_policy = artefacts.rollout_policy
    gate_failed = artefacts.training_policy["loss_policy"]["gate_failed_reward_micro"]
    blocks = []
    for temperature in temperatures:
        rp = reseal_rollout_policy(base_policy, temperature)
        ctx = LoopContext(
            opened=opened,
            reference=reference,
            rollout_policy=rp,
            reward_policy=artefacts.reward_policy,
            heads=tuple(sorted(set(selected_typed_heads))),
            run_seed=run_seed,
            provenance=RlLossProvenance(
                reward_policy_hash=artefacts.reward_policy["reward_policy_hash"],
                rollout_policy_hash=rp["rollout_policy_hash"],
                reference_checkpoint_id=artefacts.reference.checkpoint_id,
                reference_ancestry_hash=artefacts.reference.ancestry_hash,
                typed_head_config_hash=artefacts.policy_parent.typed_head_config_hash,
            ),
            supervised_weight_micro=0,
            ports=probe_ports,
        )
        blocks.append(
            _measure_temperature(
                ctx,
                page_list,
                temperature_micro=temperature,
                gate_failed_reward_micro=gate_failed,
                gold_counters=gold_counters,
                timer=timer,
                epoch=epoch,
            )
        )
    digest_after = parameter_digest(opened.policy)
    doc = {
        "schema": PROBE_SCHEMA,
        "backward": "not_authorised",
        "window": {
            "checkpoint_id": window.checkpoint_id,
            "approved_by": window.approved_by,
            "precision": window.precision,
        },
        "pages": len(page_list),
        "group_size": int(base_policy["group_size"]),
        "parameter_digest": {
            "before": digest_before,
            "after": digest_after,
            "unchanged": digest_before == digest_after,
        },
        "temperatures": blocks,
    }
    _write_new(out / PROBE_JSON, json.dumps(doc, sort_keys=True, indent=2) + "\n")
    _write_new(out / PROBE_MD, _summary_md(doc))
    if digest_before != digest_after:
        raise ProbeParametersMoved("parameters changed during an inference-only probe")
    return doc


__all__ = [
    "PROBE_JSON",
    "PROBE_MD",
    "ProbeParametersMoved",
    "ProbeRefused",
    "parameter_digest",
    "percentile_micro",
    "reseal_rollout_policy",
    "run_inference_probe",
]

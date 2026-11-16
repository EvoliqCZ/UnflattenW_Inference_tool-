"""``RealRunPorts.render_probe`` bound to ``rl.render_probe.probe_render``.

Policy and tag are validated when the port is built (a bad tag must not surface
hours into a run). The returned callable has the loop's signature
``(docx, evaluation_hash, slot_k)``; LibreOffice is opened only by
``probe_render`` itself and never when the policy is ``off``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from unflatten.model.rl import render_probe as rp
from unflatten.model.rl.policies import RENDER_PROBE_MODES

_PLACEHOLDER_HASH = "sha256:" + "0" * 64


def make_render_probe(
    policy_render_probe: str, tag: str, pid_max: int | None
) -> Callable[[Path, str, int], rp.RenderProbeRecord]:
    # `off` mode returns before any LibreOffice/disk access but runs the
    # policy-mode and tag validation of probe_render.
    rp.probe_render(
        b"",
        candidate_evaluation_hash=_PLACEHOLDER_HASH,
        policy_render_probe="off",
        slot_k=0,
        tag=tag,
    )
    if policy_render_probe not in RENDER_PROBE_MODES:
        raise ValueError(f"render_probe {policy_render_probe!r} not in {RENDER_PROBE_MODES}")
    rp.rl_render_slot(0, pid_max=pid_max)

    def render_probe(docx: Path, evaluation_hash: str, slot_k: int) -> rp.RenderProbeRecord:
        return rp.probe_render(
            docx,
            candidate_evaluation_hash=evaluation_hash,
            policy_render_probe=policy_render_probe,
            slot_k=slot_k,
            tag=tag,
            pid_max=pid_max,
        )

    return render_probe

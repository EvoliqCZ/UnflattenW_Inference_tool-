"""S4.5 probe CLI (DEC-047 addendum A v3 point 8 + A v4 point 11, DEC-082): measure what a
real run would cost BEFORE any ratification record exists.

``(a)`` inference only (window record, ``probe_inference``); ``(b)`` the forward+backward block
(``probe_backward``) only when ``--backward-decision DEC-NNN`` resolves to a DECIDED/RATIFIED
decision naming ``backward`` and ``A v4``. Without the flag ``probe.json`` says
``"backward": "not_authorised"`` (not an error); a flag naming an unratified decision is a
named refusal, before anything is opened. There is no ratification-record flag, no
checkpoint/ledger/stop-file path and no optimizer: everything is written below ``--out``.

The inputs are those of ``s4_5_launch freeze-run`` read WITHOUT the ratification record and
without the RATIFIED-policies requirement (the probe measures the numbers a PROPOSED policy
would freeze). Exit codes: 0 OK, 2 named refusal (``REFUSED <Type>: <msg>`` on stderr).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from unflatten.model.checkpoint_ancestry import CheckpointAncestryInvalid
from unflatten.model.rl import data_prep_support as sup
from unflatten.model.rl.adapters import launch as lp
from unflatten.model.rl.adapters import launch_run as lr
from unflatten.model.rl.adapters import probe_backward as pb
from unflatten.model.rl.adapters import probe_inference as pi
from unflatten.model.rl.adapters import taxonomy_producer as tp
from unflatten.model.rl.adapters.exclusions import DEFAULT_DECISIONS_DIR
from unflatten.model.rl.ratification import RlRatificationRequired

_REFUSALS: Final = (ValueError, OSError, CheckpointAncestryInvalid, RlRatificationRequired)
#: the freeze-run inputs minus the ratification record and the execution paths
_PROBE_BUILD_STATUSES: Final = ("PROPOSED", "RATIFIED")
_INPUT_FLAGS: Final = (
    "policies", "dataset-manifest", "selection", "registry", "ancestry-base", "parent-checkpoint",
    "parent-run-dir", "cache-dir", "release-dir", "store-dir", "stage-data-matrix", "allowlist",
    "visual-input-policy", "loop-guard-caps",
)  # fmt: skip


@dataclasses.dataclass(frozen=True)
class Wiring:
    """What the probe needs from the inputs; ``pages`` is lazy (opened after the window gate)."""

    ports: Any
    artefacts: Any
    pages: Callable[[], Sequence[Any]]
    checkpoint_id: str
    selected_typed_heads: tuple[str, ...]
    run_seed: int


def _now() -> datetime:
    return datetime.now(UTC)


def _policies(paths: lp.InputPaths) -> dict[str, dict[str, Any]]:
    from unflatten.model.rl.gate_policy import verify_rl_stage_gate_policy
    from unflatten.model.rl.policies import verify_reward_policy, verify_rollout_policy
    from unflatten.model.rl.training_policy import verify_rl_training_policy

    files = {
        "reward": (paths.reward_policy, verify_reward_policy),
        "rollout": (paths.rollout_policy, verify_rollout_policy),
        "training": (paths.training_policy, verify_rl_training_policy),
        "stage_gate": (paths.stage_policy, verify_rl_stage_gate_policy),
    }
    docs = {}
    for name, (path, verify) in files.items():
        docs[name] = sup.read_json(path, f"{name} policy")
        verify(docs[name])  # PROPOSED is allowed: the probe measures the numbers
    return docs


def _input_paths(args: argparse.Namespace, build: dict[str, Any]) -> lp.InputPaths:
    pol = Path(args.policies)
    return lp.InputPaths(
        reward_policy=pol / lp.POLICY_FILES["reward"],
        rollout_policy=pol / lp.POLICY_FILES["rollout"],
        training_policy=pol / lp.POLICY_FILES["training"],
        stage_policy=pol / lp.POLICY_FILES["stage_gate"],
        dataset_manifest=Path(args.dataset_manifest), selection=Path(args.selection),
        registry=Path(args.registry), ancestry_base=Path(args.ancestry_base),
        parent_checkpoint=Path(args.parent_checkpoint), parent_run_dir=Path(args.parent_run_dir),
        cache_dir=Path(args.cache_dir), release_dir=Path(args.release_dir),
        loop_guard_caps=Path(args.loop_guard_caps),
        visual_input_policy=Path(args.visual_input_policy),
        ratification_record=Path(os.devnull), decisions_dir=Path(args.decisions_dir),
        artefact_root=None if args.artefact_root is None else Path(args.artefact_root),
        run_class=build["run_class"], run_seed=build["run_seed"],
        selected_typed_heads=tuple(build["selected_typed_heads"]),
    )  # fmt: skip


def _assemble(paths: lp.InputPaths) -> lp.Assembled:
    """``launch.assemble_inputs`` minus the ratified-policies and ratification-record steps."""
    from unflatten.dataset.samples import release_root_hash
    from unflatten.model.capability_profile import resolve_capability_profile_by_hash
    from unflatten.model.rl.adapters import loop_guard as lg
    from unflatten.model.rl.adapters.admission import cache_identity_hash
    from unflatten.model.rl.run_manifest import RunArtefacts

    policies = _policies(paths)
    root = release_root_hash(paths.release_dir)
    manifest, selection, registry, base = lp._check_data(paths, root)
    identity, vip = lp._cache(paths, root)
    sidecar, reference, parent = lp._parent(paths, base, vip)
    if selection.checkpoint_id != parent.checkpoint_id:
        raise lp.ParentIdentityRefused(
            f"frozen selection was derived for {selection.checkpoint_id!r}, "
            f"the parent is {parent.checkpoint_id!r}"
        )
    caps_sha = lp.file_sha256(paths.loop_guard_caps)
    profile = resolve_capability_profile_by_hash(identity.get("capability_profile_hash"))
    caps = lg.verify_caps_binding(
        caps_path=paths.loop_guard_caps, caps_sha256=caps_sha, cache_identity=identity,
        capability_profile_id=profile.profile_id,
    )  # fmt: skip
    arts = RunArtefacts(
        reward_policy=policies["reward"], rollout_policy=policies["rollout"],
        dataset_manifest=manifest, stage_policy=policies["stage_gate"],
        training_policy=policies["training"], reference=reference, policy_parent=parent,
        cache_identity=cache_identity_hash(identity), visual_feature_tap=vip.visual_feature_tap(),
    )  # fmt: skip
    return lp.Assembled(
        paths, policies, manifest, selection, registry, base, identity, arts.cache_identity,
        vip, sidecar, reference, parent, caps, caps_sha, None, None, None, "", arts,
    )  # type: ignore[arg-type]  # fmt: skip


def _probe_ports(ra: lr.RunAssembly, a: lp.Assembled) -> Any:
    """The probe's ports over the inputs; opens nothing (the loaders are lazy)."""
    from unflatten.model.rl.adapters import pages
    from unflatten.model.rl.real_run_ports import RealRunPorts

    opener, open_reference, open_candidate = lr._candidate_ports(ra, False)
    expectation = lr.cache_expectation(ra.expectation)
    reader = lr.open_cache_reader(ra.launch["cache_dir"], expectation)
    rp = a.policies["rollout"]
    render_probe = None
    if rp["render_probe"] == "status_only":
        from unflatten.model.rl.adapters.render_probe_port import make_render_probe
        from unflatten.model.rl.render_probe import read_pid_max

        render_probe = make_render_probe(rp["render_probe"], ra.run_id, read_pid_max())
    loop_guard = lr._build_loop_guard(ra) if rp["loop_guards_in_policy"] is True else None
    return RealRunPorts(
        open_candidate=open_candidate, open_reference=open_reference,
        open_pages=pages.make_open_pages(
            ra.launch["release_dir"], ra.launch["store_dir"], reader, expectation, a.registry,
            a.selection, Path(ra.launch["stage_data_matrix"]),
            artefact_root=ra.launch["artefact_root"], decisions_dir=ra.launch["decisions_dir"],
        ),
        build_visual=lr._after_open(opener, lr._build_visual(ra, reader, expectation, opener)),
        build_gold=lr._after_open(opener, lr._build_gold(ra, reader, expectation)),
        loop_guard_factory=None if loop_guard is None else lr._after_open(opener, loop_guard),
        render_probe=render_probe,
    )  # fmt: skip


def _open_wiring(args: argparse.Namespace) -> Wiring:
    missing = [f"--{f}" for f in _INPUT_FLAGS if getattr(args, f.replace("-", "_")) is None]
    if missing:
        raise lp.LaunchRefused(f"missing input flags: {missing}")
    build = lp.read_policy_build(
        args.policies, allowed_statuses=_PROBE_BUILD_STATUSES
    )
    a = _assemble(_input_paths(args, build))
    launch = {
        k: (None if getattr(args, k) is None else str(Path(getattr(args, k)).resolve()))
        for k in (
            "allowlist",
            "store_dir",
            "release_dir",
            "stage_data_matrix",
            "artefact_root",
            "decisions_dir",
            "cache_dir",
        )  # fmt: skip
    } | {"device": args.device}
    expectation = {"expected_page_keys": lp.read_page_keys(a.paths.cache_dir),
                   "approved_exclusions": []}  # fmt: skip
    ra = lr.RunAssembly(
        Path(args.policies), "probe", launch, expectation, build["run_class"],
        build["run_seed"], tuple(build["selected_typed_heads"]), a,
        args.expect_s1_run_manifest_hash,
    )  # fmt: skip
    ports = _probe_ports(ra, a)

    def pages() -> list[Any]:
        return list(ports.open_pages(a.dataset_manifest).train_pages.pages)

    return Wiring(
        ports, a.artefacts, pages, a.policy_parent.checkpoint_id,
        tuple(build["selected_typed_heads"]), int(build["run_seed"]),
    )  # fmt: skip


def _rewrite(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _publish_backward(out: Path, doc: dict[str, Any], block: dict[str, Any]) -> None:
    doc["backward"] = block
    _rewrite(out / pi.PROBE_JSON, json.dumps(doc, sort_keys=True, indent=2) + "\n")
    lines = (out / pi.PROBE_MD).read_text(encoding="utf-8").splitlines()
    lines[0] = "# S4.5 probe (backward: measured)"
    for c in block["chunks"]:
        lines.append(
            f"backward chunk={c['prefill_chunk_tokens']} oom={c['oom']} "
            f"vram_peak {c['vram_peak_bytes']} tokens {c['candidate_tokens']}"
        )
    _rewrite(out / pi.PROBE_MD, "\n".join(lines[:20]) + "\n")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    decision = pb.authorised_decision(args.decisions_dir, args.backward_decision)
    out = Path(args.out)
    wiring = _open_wiring(args)
    # the window gate runs before any page or weight is read (probe_inference gates again)
    tp.require_window(
        tp.load_window_record(args.window_record), checkpoint_id=wiring.checkpoint_id, now=_now()
    )
    pages = list(wiring.pages())[: args.pages]
    capture = pb.Capture(wiring.ports)
    doc = pi.run_inference_probe(
        ports=capture.ports, artefacts=wiring.artefacts, pages=pages,
        temperatures_micro=args.temperature_micro, out_dir=out, clock=_now,
        window_record_path=args.window_record,
        resolve_checkpoint_id=lambda: wiring.checkpoint_id,
        run_seed=wiring.run_seed, selected_typed_heads=wiring.selected_typed_heads,
    )  # fmt: skip
    if decision is not None:
        block = pb.run_backward_block(
            ports=capture.ports, artefacts=wiring.artefacts, opened=capture.opened,
            reference=capture.reference, pages=pages, chunks=args.prefill_chunk_tokens,
            render_dir=out / "render", decision_id=args.backward_decision,
            run_seed=wiring.run_seed, selected_typed_heads=wiring.selected_typed_heads,
        )  # fmt: skip
        _publish_backward(out, doc, block)
    return {
        "command": "probe", "out": str(out), "pages": doc["pages"],
        "backward": doc["backward"] if decision is None else pb.MEASURED,
        "parameters_unchanged": doc["parameter_digest"]["unchanged"],
    }  # fmt: skip


def _chunk(text: str) -> int | None:
    if text == "null":
        return None
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"prefill chunk must be 'null' or an int >= 1: {text!r}")
    return value


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="s4_5_probe", description=__doc__)
    p.add_argument("--window-record", required=True)
    p.add_argument("--pages", type=int, required=True)
    p.add_argument("--temperature-micro", type=int, nargs="+", required=True)
    p.add_argument("--prefill-chunk-tokens", type=_chunk, nargs="+", default=[None, 2048])
    p.add_argument("--out", required=True)
    p.add_argument("--backward-decision", default=None)
    for flag in _INPUT_FLAGS:
        p.add_argument(f"--{flag}", default=None)
    p.add_argument("--decisions-dir", default=str(DEFAULT_DECISIONS_DIR))
    p.add_argument("--artefact-root", default=None)
    p.add_argument("--expect-s1-run-manifest-hash", default=None)
    p.add_argument("--device", default="cuda")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    try:
        summary = _run(args)
    except _REFUSALS as exc:
        text = " ".join(str(exc).split()) or repr(exc)
        print(f"REFUSED {type(exc).__name__}: {text}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

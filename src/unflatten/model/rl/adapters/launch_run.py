"""The launch side of the sealed RUN directory (design P6; torch-free import).

``assemble(run_dir)`` re-reads everything ``freeze-run`` sealed, through the same
``assemble_inputs`` the freeze used, and refuses by name when the RUN is not what was
sealed. ``build_ports`` turns the assembly into ``RealRunPorts``: it opens nothing (the
checkpoint, the reference, the visual encoder and the gold builder are lazy closures that
exist only after ``open_candidate``). The launcher takes no checkpoint, dataset or record
path from the command line; every path comes from the sealed ``rl-launch.json``.

Order inside ``assemble``: the seal (a ``sha256sum -c`` equivalent over absolute paths),
the closed ``rl-launch.json``, execution paths apart from the sealed run and the parent,
no stale stop file, ``assemble_inputs``, no SFT auxiliary, preview manifest == rebuilt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from unflatten.model.rl import data_prep_support as sup
from unflatten.model.rl.adapters import launch as lp
from unflatten.model.rl.ratification import RlRatificationRequired

if TYPE_CHECKING:  # pragma: no cover
    from unflatten.model.rl.real_run_ports import OpenedPages, RealRunPorts


class SealBroken(lp.LaunchRefused):
    """``inputs.sha256`` does not describe the RUN directory (missing, changed or extra file)."""


class StaleStopFile(lp.LaunchRefused):
    """A stop file already exists at launch: the run would stop at its first poll."""


class SftNotWired(lp.LaunchRefused):
    """``supervised_weight_micro > 0`` but the launcher wires no SFT auxiliary batch."""


class ManifestDiffers(lp.LaunchRefused):
    """The preview ``rl-run-manifest.json`` is not what the artefacts derive now."""


class BuildBeforeOpen(lp.LaunchRefused):
    """A port that needs the opened candidate was called before ``open_candidate``."""


@dataclass(frozen=True, slots=True)
class RunAssembly:
    run_dir: Path
    run_id: str
    launch: Mapping[str, Any]
    expectation: Mapping[str, Any]
    run_class: str
    run_seed: int
    heads: tuple[str, ...]
    assembled: lp.Assembled
    expect_s1_run_manifest_hash: str | None = None


# --- the seal ------------------------------------------------------------------------------


def verify_seal(run_dir: Path) -> dict[str, str]:
    """``{path: sha256}`` of ``inputs.sha256``: every line re-hashed, every file covered."""
    try:
        text = (run_dir / lp.SEAL_FILE).read_text(encoding="utf-8")
    except OSError as exc:
        raise SealBroken(f"{run_dir / lp.SEAL_FILE} cannot be read: {exc}") from exc
    sealed: dict[str, str] = {}
    for line in text.splitlines():
        digest, sep, path = line.partition("  ")
        if not sep or len(digest) != 64 or not path:
            raise SealBroken(f"{lp.SEAL_FILE}: malformed line {line!r}")
        sealed[path] = digest
    for path, digest in sealed.items():
        try:
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError as exc:
            raise SealBroken(f"sealed file {path} cannot be read: {exc}") from exc
        if actual != digest:
            raise SealBroken(f"sealed file {path} has sha256 {actual}, sealed {digest}")
    loose = sorted(
        p.name for p in run_dir.iterdir() if p.name != lp.SEAL_FILE and str(p) not in sealed
    )
    if loose:
        raise SealBroken(f"RUN files not covered by {lp.SEAL_FILE}: {loose}")
    return sealed


# --- assemble ------------------------------------------------------------------------------


def read_launch(run_dir: Path) -> dict[str, Any]:
    doc = sup.read_json(run_dir / lp.LAUNCH_FILE, "launch document")
    if not isinstance(doc, dict) or set(doc) != set(lp.LAUNCH_KEYS):
        keys = sorted(doc) if isinstance(doc, dict) else type(doc).__name__
        raise lp.LaunchRefused(f"{lp.LAUNCH_FILE} keys {keys} != {sorted(lp.LAUNCH_KEYS)}")
    if doc["schema"] != lp.LAUNCH_SCHEMA:
        raise lp.LaunchRefused(f"{lp.LAUNCH_FILE} schema {doc['schema']!r} != {lp.LAUNCH_SCHEMA!r}")
    return doc


def _identity_of(preview: Any) -> tuple[str, int, tuple[str, ...]]:
    try:
        heads = tuple(preview["selected_typed_heads"])
        return str(preview["run_class"]), int(preview["run_seed"]), heads
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestDiffers(
            f"{lp.MANIFEST_FILE} lacks run_class/run_seed/heads: {exc!r}"
        ) from exc


def _input_paths(run: Path, doc: Mapping[str, Any], ident: tuple[str, int, tuple[str, ...]]):
    def at(key: str) -> Path:
        return Path(doc[key])

    root = doc["artefact_root"]
    return lp.InputPaths(
        reward_policy=run / lp.POLICY_FILES["reward"],
        rollout_policy=run / lp.POLICY_FILES["rollout"],
        training_policy=run / lp.POLICY_FILES["training"],
        stage_policy=run / lp.POLICY_FILES["stage_gate"],
        dataset_manifest=run / lp.DATASET_FILE,
        selection=run / lp.SELECTION_FILE,
        registry=run / lp.REGISTRY_FILE,
        ancestry_base=run / lp.ANCESTRY_FILE,
        parent_checkpoint=at("parent_checkpoint"),
        parent_run_dir=at("parent_run_dir"),
        cache_dir=at("cache_dir"),
        release_dir=at("release_dir"),
        loop_guard_caps=at("loop_guard_caps"),
        visual_input_policy=at("visual_input_policy"),
        ratification_record=at("ratification_record"),
        decisions_dir=at("decisions_dir"),
        artefact_root=None if root is None else Path(root),
        run_class=ident[0],
        run_seed=ident[1],
        selected_typed_heads=ident[2],
    )


def _require_sealed(
    sealed: Mapping[str, str], paths: lp.InputPaths, doc: Mapping[str, Any]
) -> None:
    sidecar = paths.parent_checkpoint.with_name(paths.parent_checkpoint.name + ".ancestry.json")
    wanted = [
        sidecar, paths.parent_run_dir / lp.S1_MANIFEST, paths.parent_run_dir / lp.S4_MANIFEST,
        paths.loop_guard_caps, paths.visual_input_policy, paths.ratification_record,
        Path(doc["allowlist"]), Path(doc["stage_data_matrix"]),
    ]  # fmt: skip
    missing = [str(p) for p in wanted if str(p.resolve()) not in sealed]
    if missing:
        raise SealBroken(f"files read at launch but not covered by {lp.SEAL_FILE}: {missing}")


def _outputs(doc: Mapping[str, Any]) -> dict[str, Path]:
    return {k: Path(doc[k]) for k in ("checkpoint_dir", "render_dir", "ledger_path", "stop_file")}


def assemble(run_dir: Path | str) -> RunAssembly:
    run = Path(run_dir).resolve()
    sealed = verify_seal(run)
    doc = read_launch(run)
    record = Path(doc["ratification_record"])
    if not record.is_file():
        raise RlRatificationRequired(f"no ratification record at {record}: a real run needs one")
    preview = sup.read_json(run / lp.MANIFEST_FILE, "preview run manifest")
    paths = _input_paths(run, doc, _identity_of(preview))
    _require_sealed(sealed, paths, doc)
    lp.require_outputs_apart(run, paths.parent_run_dir, paths.parent_checkpoint, _outputs(doc))
    if Path(doc["stop_file"]).exists():
        raise StaleStopFile(f"stop file {doc['stop_file']} exists: remove it to launch")
    assembled = lp.assemble_inputs(paths)
    weight = assembled.policies["training"]["supervised_weight_micro"]
    if type(weight) is not int or weight > 0:
        raise SftNotWired(f"supervised_weight_micro is {weight!r}: the SFT auxiliary is not wired")
    if lp.build_preview_manifest(assembled) != preview:
        raise ManifestDiffers(
            f"{lp.MANIFEST_FILE} differs from the manifest the artefacts derive: it is not "
            "what the owner saw at freeze"
        )
    expectation = sup.read_json(run / lp.EXPECTATION_FILE, "expectation input")
    run_class, run_seed, heads = _identity_of(preview)
    return RunAssembly(
        run, str(preview["run_id"]), doc, expectation, run_class, run_seed, heads, assembled,
        doc["expect_s1_run_manifest_hash"],
    )


# --- ports ---------------------------------------------------------------------------------


def cache_expectation(payload: Mapping[str, Any]) -> Any:
    from unflatten.model import exact_text_cache as etc

    return etc.CacheExpectation.from_payload(payload)


def open_cache_reader(cache_dir: Path | str, expectation: Any) -> Any:
    """The gradient-bearing reader pinned to the frozen expectation (refuses a dev-only cache)."""
    from unflatten.model import exact_text_cache as etc

    return etc.ExactTextCacheReader(cache_dir, gradient_bearing=True, expectation=expectation)


def is_fresh_run(checkpoint_dir: Path | str) -> bool:
    from unflatten.model.rl.real_run import CHECKPOINT_GLOB

    return not any(Path(checkpoint_dir).glob(CHECKPOINT_GLOB))


def _after_open(opener: Any, build: Callable[[Any], Callable[..., Any]]) -> Callable[..., Any]:
    """A port that exists only once ``opener`` has opened the candidate; built on first call."""
    built: list[Callable[..., Any]] = []

    def call(*args: Any) -> Any:
        if not built:
            if opener.opened is None:
                raise BuildBeforeOpen("this port was called before open_candidate")
            built.append(build(opener.opened))
        return built[0](*args)

    return call


def _build_visual(
    ra: RunAssembly, reader: Any, expectation: Any, opener: Any
) -> Callable[[Any], Callable[..., Any]]:
    def build(opened: Any) -> Callable[..., Any]:
        from transformers import AutoImageProcessor

        from unflatten.eval.s2_gate.cli import decode_visual_policy_for_checkpoint
        from unflatten.model import onfly_visual as ov
        from unflatten.model.candidate_loader import (
            QWEN_CANDIDATE_MODEL_ID,
            QWEN_CANDIDATE_REVISION,
        )
        from unflatten.model.rl.adapters.visual import make_build_visual

        policy = decode_visual_policy_for_checkpoint(ra.assembled.visual_input_policy, opener.state)
        encoder = ov.OnTheFlyVisualEncoder(
            opener.candidate, policy=policy, device=ra.launch["device"]
        )
        return make_build_visual(
            store_dir=ra.launch["store_dir"],
            policy_payload=policy.payload(),
            processor=AutoImageProcessor.from_pretrained(
                QWEN_CANDIDATE_MODEL_ID, revision=QWEN_CANDIDATE_REVISION
            ),
            encoder=encoder,
            reader=reader,
            expectation=expectation,
            model=opened.policy,
            checkpoint_state=opener.state,
            feature_source=ra.assembled.cache_identity["feature_tap"],
        )

    return build


def _build_gold(
    ra: RunAssembly, reader: Any, expectation: Any
) -> Callable[[Any], Callable[..., Any]]:
    def build(opened: Any) -> Callable[..., Any]:
        from unflatten.model.rl.adapters import gold
        from unflatten.model.rl.adapters.release import load_bundle_source

        return gold.make_build_gold(
            store_dir=ra.launch["store_dir"],
            reader=reader,
            expectation=expectation,
            vocabulary=opened.vocabulary,
            tokenizer=opened.tokenizer,
            chain_flags=ra.assembled.policies["reward"]["chain_policy_flags"],
            bundle_source=load_bundle_source(ra.launch["store_dir"]),
            cells=gold.s4_cells_by_source(),
        )

    return build


def _build_loop_guard(ra: RunAssembly) -> Callable[[Any], Callable[..., Any]]:
    def build(opened: Any) -> Callable[..., Any]:
        from unflatten.model.rl.adapters import loop_guard

        return loop_guard.make_loop_guard_factory(
            caps_path=ra.assembled.paths.loop_guard_caps,
            caps_sha256=ra.assembled.caps_sha256,
            cache_identity=ra.assembled.cache_identity,
            capability_profile_id=opened.capability_profile_id,
            rollout_policy=ra.assembled.policies["rollout"],
            ranges=opened.vocabulary.ranges,
        )

    return build


def _candidate_ports(ra: RunAssembly, dry_run: bool) -> tuple[Any, Callable[..., Any], Any]:
    from unflatten.model.rl.adapters import candidate as cand

    a = ra.assembled
    opener = cand.make_open_candidate(
        checkpoint_path=a.paths.parent_checkpoint,
        expect_base_ancestry_hash=a.sidecar["checkpoint"]["base_ancestry_hash"],
        expect_s1_run_manifest_hash=ra.expect_s1_run_manifest_hash,
        expect_typed_head_config_hash=a.policy_parent.typed_head_config_hash,
        deepstack=a.visual_input_policy.deepstack,
        device=ra.launch["device"],
        tokenizer_artifact_set_hash=a.cache_identity["tokenizer_artifact_set_hash"],
        allowlist_payload=json.loads(Path(ra.launch["allowlist"]).read_text(encoding="utf-8")),
        expect_vocabulary_hash=a.cache_identity["composite_vocabulary_hash"],
        capability_profile_hash=a.cache_identity["capability_profile_hash"],
    )
    reference = cand.make_open_reference(
        opener,
        reference_identity=a.artefacts.reference,
        policy_parent_identity=a.artefacts.policy_parent,
        run_class=ra.run_class,
    )

    def refuse_in_dry_run() -> Any:
        raise lp.LaunchRefused("--dry-run opens no checkpoint")

    return opener, reference, (refuse_in_dry_run if dry_run else opener)


def build_ports(ra: RunAssembly, dry_run: bool) -> RealRunPorts:
    """``RealRunPorts`` over the sealed run; opens nothing (dry run: ``open_candidate`` refuses)."""
    from unflatten.model.rl.adapters import ancestry, ledger_io, pages, run_control
    from unflatten.model.rl.real_run_ports import RealRunPorts

    a, doc = ra.assembled, ra.launch
    opener, open_reference, open_candidate = _candidate_ports(ra, dry_run)
    expectation = cache_expectation(ra.expectation)
    reader = open_cache_reader(doc["cache_dir"], expectation)
    open_ledger, write_ledger = ledger_io.make_ledger_ports(
        Path(doc["ledger_path"]), fresh_run=is_fresh_run(doc["checkpoint_dir"])
    )
    rp = a.policies["rollout"]
    probe = None
    if rp["render_probe"] == "status_only":
        from unflatten.model.rl.adapters.render_probe_port import make_render_probe
        from unflatten.model.rl.render_probe import read_pid_max

        probe = make_render_probe(rp["render_probe"], ra.run_id, read_pid_max())
    return RealRunPorts(
        open_candidate=open_candidate,
        open_reference=open_reference,
        open_pages=pages.make_open_pages(
            doc["release_dir"],
            doc["store_dir"],
            reader,
            expectation,
            a.registry,
            a.selection,
            Path(doc["stage_data_matrix"]),
            artefact_root=doc["artefact_root"],
            decisions_dir=doc["decisions_dir"],
        ),  # fmt: skip
        read_selection=pages.make_read_selection(ra.run_dir / lp.SELECTION_FILE),
        build_visual=_after_open(opener, _build_visual(ra, reader, expectation, opener)),
        build_gold=_after_open(opener, _build_gold(ra, reader, expectation)),
        build_sft_batch=None,
        open_ledger=open_ledger,
        write_ledger=write_ledger,
        ancestry_base=ancestry.make_ancestry_base_port(ra.run_dir / lp.ANCESTRY_FILE),
        checkpoint_dir=Path(doc["checkpoint_dir"]),
        render_dir=Path(doc["render_dir"]),
        feature_source=a.cache_identity["feature_tap"],
        stop_requested=run_control.make_stop_requested(Path(doc["stop_file"])),
        loop_guard_factory=(
            _after_open(opener, _build_loop_guard(ra))
            if rp["loop_guards_in_policy"] is True
            else None
        ),
        render_probe=probe,
    )


# --- admission without a model -------------------------------------------------------------


def admit(ra: RunAssembly, ports: RealRunPorts) -> dict[str, Any]:
    """Everything the run needs but the model: ports complete, pages open, registry bound."""
    from unflatten.model.rl.selection import require_selection_matches_manifest

    a = ra.assembled
    rp = a.policies["rollout"]
    ports.require_complete(supervised_weight_micro=0, render_probe_mode=rp["render_probe"])
    assert ports.open_pages is not None
    opened: OpenedPages = ports.open_pages(a.dataset_manifest)
    if opened.exclusion_registry_hash is None:
        raise lp.LaunchRefused("the opened pages carry no exclusion registry hash")
    require_selection_matches_manifest(opened.selection, a.dataset_manifest)
    if opened.selection.checkpoint_id != a.policy_parent.checkpoint_id:
        raise lp.LaunchRefused(
            f"selection was derived for {opened.selection.checkpoint_id!r}, the parent is "
            f"{a.policy_parent.checkpoint_id!r}"
        )
    return {
        "pages_included": len(opened.train_pages.pages),
        "exclusion_registry_hash": opened.exclusion_registry_hash,
        "fresh_run": is_fresh_run(ra.launch["checkpoint_dir"]),
    }

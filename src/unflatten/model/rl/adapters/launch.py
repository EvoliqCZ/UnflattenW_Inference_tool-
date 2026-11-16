"""Inputs of the S4.5 launcher: one path to the artefacts for ``freeze-run`` and ``launch``
(design P4, D-8; torch-free import, heavy modules are imported inside the functions).

``assemble_inputs`` reads every artefact a run is frozen over, from the paths it is given,
and verifies each one by name before a ``RunArtefacts`` is built. ``freeze-run`` calls it
with the owner's source files, ``launch`` later with the files inside the sealed RUN
directory -- the same function, so a frozen run cannot be read differently than it was
frozen. The 16 GB parent checkpoint is never opened here: its identity is read from its
sidecar JSON (the bytes are hashed by ``verify_sidecar`` when the candidate is opened).
"""

from __future__ import annotations

import glob
import hashlib
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from unflatten.model import checkpoint_ancestry as ca
from unflatten.model.rl import data_prep_support as sup
from unflatten.model.rl.adapters import exclusions as ex
from unflatten.model.rl.adapters import loop_guard as lg
from unflatten.model.rl.adapters.admission import cache_identity_hash, require_strict_admission
from unflatten.model.rl.adapters.exclusions_io import RATIFIED_STATUSES, decision_status
from unflatten.model.rl.ratification import RlRatificationRecord, require_ratification

if TYPE_CHECKING:  # pragma: no cover
    from unflatten.model.loop_guard_v2_caps import LoopGuardV2Caps
    from unflatten.model.rl.reference import ReferencePolicyIdentity
    from unflatten.model.rl.run_manifest import RunArtefacts
    from unflatten.model.rl.selection import FrozenSelection

POLICY_FILES: Final = {
    "reward": "rl-reward-policy.json",
    "rollout": "rl-rollout-policy.json",
    "training": "rl-training-policy.json",
    "stage_gate": "rl-stage-gate-policy.json",
}
BUILD_FILE: Final = "policy-build.json"
BUILD_SCHEMA: Final = "unflatten.rl.s4_5-policy-build/1"
DATASET_FILE: Final = "rl-dataset-manifest.json"
SELECTION_FILE: Final = "selection.json"
REGISTRY_FILE: Final = "rl-exclusion-registry.json"
ANCESTRY_FILE: Final = "checkpoint_ancestry_base.json"
EXPECTATION_FILE: Final = "rl-expectation-input.json"
LAUNCH_FILE: Final = "rl-launch.json"
MANIFEST_FILE: Final = "rl-run-manifest.json"
FREEZE_FILE: Final = "rl-freeze.json"
SEAL_FILE: Final = "inputs.sha256"
LAUNCH_SCHEMA: Final = "unflatten.rl.s4_5-launch/1"
LAUNCH_KEYS: Final = (
    "schema", "device", "checkpoint_dir", "render_dir", "stop_file", "ledger_path",
    "ratification_record", "decisions_dir", "artefact_root", "cache_dir", "release_dir",
    "store_dir", "stage_data_matrix", "allowlist", "visual_input_policy", "parent_checkpoint",
    "parent_run_dir", "loop_guard_caps", "expect_s1_run_manifest_hash",
)  # fmt: skip
S1_MANIFEST: Final = "s1_run_manifest.json"
S4_MANIFEST: Final = "s4_run_manifest.json"
RATIFICATIONS_DIR: Final = "ratifications"
_HASH_FIELDS: Final = {
    "reward": "reward_policy_hash",
    "rollout": "rollout_policy_hash",
    "training": "training_policy_hash",
    "stage_gate": "stage_policy_hash",
}


class LaunchRefused(ValueError):
    """A named refusal of ``freeze-run`` / ``launch`` (exit code 2)."""


class PolicyNotRatified(LaunchRefused):
    """A policy (or the build record) is not RATIFIED."""


class DecisionRefused(LaunchRefused):
    """The decision a ratification record names is missing, ambiguous or not ratified."""


class RatificationRecordMisplaced(LaunchRefused):
    """The ratification record does not live under ``<decisions_dir>/ratifications/``."""


class ParentIdentityRefused(LaunchRefused):
    """The parent checkpoint sidecar / run manifests do not give a consistent identity."""


class CacheRefused(LaunchRefused):
    """The train cache is not admitted for this run."""


class RunDirRefused(LaunchRefused):
    """The RUN directory or an execution path is not free / not apart from the sealed run."""


# --- decision + record location (D-8) ---------------------------------------------------


def resolve_decision(decisions_dir: Path | str, decision_id: str) -> Path:
    """The one ``<decisions_dir>/<decision_id>*.md`` whose declared status is DECIDED/RATIFIED.

    Zero matches and several matches (a ``DEC-081`` TEMPLATE next to the final file) are both
    refusals: the file the status is read from must be unambiguous.
    """
    root = Path(decisions_dir)
    matches = sorted(glob.glob(str(root / f"{glob.escape(decision_id)}*.md")))
    if len(matches) != 1:
        raise DecisionRefused(
            f"decision {decision_id!r}: expected exactly one {decision_id}*.md in {root}, "
            f"found {len(matches)}: {[Path(m).name for m in matches]}"
        )
    path = Path(matches[0])
    status = decision_status(path.read_text(encoding="utf-8"))
    if status not in RATIFIED_STATUSES:
        raise DecisionRefused(
            f"decision file {path.name} declares status {status!r}; "
            f"only {sorted(RATIFIED_STATUSES)} may ratify an S4.5 run"
        )
    return path


def record_under_ratifications(record_path: Path | str, decisions_dir: Path | str) -> Path:
    """The resolved record path; a refusal when it is outside ``<decisions_dir>/ratifications``.

    Resolved, so a symlink under that directory that points elsewhere is refused too.
    """
    resolved = Path(record_path).resolve()
    root = (Path(decisions_dir) / RATIFICATIONS_DIR).resolve()
    if not resolved.is_relative_to(root):
        raise RatificationRecordMisplaced(
            f"ratification record {resolved} is not under {root}: a record the owner signed "
            "lives in the decisions tree, never beside the run"
        )
    return resolved


# --- small readers ------------------------------------------------------------------------


def file_sha256(path: Path | str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise LaunchRefused(f"cannot read {path}: {exc}") from exc


def _get(doc: Any, keys: Sequence[str], what: str) -> Any:
    cur = doc
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            raise ParentIdentityRefused(f"{what}: missing {'.'.join(keys)}")
        cur = cur[key]
    if not isinstance(cur, str) or not cur:
        raise ParentIdentityRefused(f"{what}: {'.'.join(keys)} must be a non-empty string")
    return cur


def read_page_keys(cache_dir: Path | str) -> list[list[Any]]:
    """Page keys of the train cache, as the expectation input wants them."""
    from unflatten.model import exact_text_cache as etc

    try:
        reader = etc.ExactTextCacheReader(str(cache_dir), gradient_bearing=False)
    except etc.ExactTextCacheIncompatible as exc:
        raise CacheRefused(f"cache {cache_dir} is not readable: {exc}") from exc
    return [list(k) for k in reader.page_keys]


def require_outputs_apart(
    run_dir: Path, parent_run_dir: Path, parent_checkpoint: Path, outputs: Mapping[str, Path]
) -> None:
    """No output path under the sealed RUN, the parent's run dir or the parent's checkpoint dir."""
    guarded = {
        "RUN": run_dir.resolve(),
        "parent_run_dir": parent_run_dir.resolve(),
        "parent checkpoint dir": parent_checkpoint.resolve().parent,
    }
    for name, path in outputs.items():
        for label, root in guarded.items():
            if Path(path).resolve().is_relative_to(root):
                raise RunDirRefused(f"{name} {path} lies under {label} {root}")


# --- assemble_inputs ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InputPaths:
    """Every artefact ``assemble_inputs`` reads, plus the run identity the policies are for."""

    reward_policy: Path
    rollout_policy: Path
    training_policy: Path
    stage_policy: Path
    dataset_manifest: Path
    selection: Path
    registry: Path
    ancestry_base: Path
    parent_checkpoint: Path
    parent_run_dir: Path
    cache_dir: Path
    release_dir: Path
    loop_guard_caps: Path
    visual_input_policy: Path
    ratification_record: Path
    decisions_dir: Path
    artefact_root: Path | None
    run_class: str
    run_seed: int
    selected_typed_heads: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Assembled:
    paths: InputPaths
    policies: Mapping[str, Mapping[str, Any]]
    dataset_manifest: Mapping[str, Any]
    selection: FrozenSelection
    registry: Mapping[str, Any]
    ancestry_base: Mapping[str, Any]
    cache_identity: Mapping[str, Any]
    cache_identity_hash: str
    visual_input_policy: Any
    sidecar: Mapping[str, Any]
    reference: ReferencePolicyIdentity
    policy_parent: ReferencePolicyIdentity
    caps: LoopGuardV2Caps
    caps_sha256: str
    record: RlRatificationRecord
    record_path: Path
    decision_path: Path
    composite: str
    artefacts: RunArtefacts


def _policies(paths: InputPaths) -> dict[str, dict[str, Any]]:
    from unflatten.model.rl.gate_policy import verify_rl_stage_gate_policy
    from unflatten.model.rl.policies import verify_reward_policy, verify_rollout_policy
    from unflatten.model.rl.training_policy import verify_rl_training_policy

    files = {
        "reward": (paths.reward_policy, verify_reward_policy),
        "rollout": (paths.rollout_policy, verify_rollout_policy),
        "training": (paths.training_policy, verify_rl_training_policy),
        "stage_gate": (paths.stage_policy, verify_rl_stage_gate_policy),
    }
    docs: dict[str, dict[str, Any]] = {}
    for name, (path, verify) in files.items():
        doc = sup.read_json(path, f"{name} policy")
        verify(doc)
        if doc.get("status") != "RATIFIED":
            raise PolicyNotRatified(
                f"{name} policy {path} has status {doc.get('status')!r}; a run is frozen over "
                "RATIFIED policies only"
            )
        docs[name] = doc
    return docs


def _check_data(paths: InputPaths, release_root: str) -> tuple[dict, FrozenSelection, dict, dict]:
    from unflatten.model.rl.policies import verify_dataset_manifest
    from unflatten.model.rl.selection import (
        require_selection_matches_manifest,
        selection_from_document,
    )

    manifest = sup.read_json(paths.dataset_manifest, "dataset manifest")
    verify_dataset_manifest(manifest)
    selection = selection_from_document(sup.read_json(paths.selection, "selection"))
    require_selection_matches_manifest(selection, manifest)
    registry = sup.read_json(paths.registry, "exclusion registry")
    ex.verify_exclusion_registry(
        registry, release_root_hash=release_root, release_dir=paths.release_dir,
        artefact_root=paths.artefact_root, decisions_dir=paths.decisions_dir,
    )  # fmt: skip
    base = sup.read_json(paths.ancestry_base, "checkpoint ancestry base")
    ca.verify_base(base)
    return manifest, selection, registry, base


def _cache(paths: InputPaths, release_root: str) -> tuple[dict[str, Any], Any]:
    from unflatten.model.onfly_visual import VisualInputPolicy

    identity = dict(sup.cache_identity(str(paths.cache_dir)))  # split == "train"
    try:
        require_strict_admission(identity)
    except ValueError as exc:
        raise CacheRefused(str(exc)) from exc
    if identity.get("dataset_release_root_hash") != release_root:
        raise CacheRefused(
            f"cache was built over release {identity.get('dataset_release_root_hash')!r}, "
            f"not {release_root!r}"
        )
    vip = VisualInputPolicy.from_payload(
        sup.read_json(paths.visual_input_policy, "visual input policy")
    )
    if vip.feature_tap() != identity.get("feature_tap"):
        raise CacheRefused(
            f"cache feature_tap {identity.get('feature_tap')!r} != the visual input policy's "
            f"{vip.feature_tap()!r}"
        )
    return identity, vip


def _parent(
    paths: InputPaths, base: Mapping[str, Any], vip: Any
) -> tuple[dict[str, Any], ReferencePolicyIdentity, ReferencePolicyIdentity]:
    from unflatten.model.rl.reference import ReferencePolicyIdentity

    if not paths.parent_checkpoint.is_file():
        raise ParentIdentityRefused(f"parent checkpoint {paths.parent_checkpoint} is not a file")
    sidecar = sup.read_json(ca.sidecar_path(paths.parent_checkpoint), "parent sidecar")
    what = "parent sidecar"
    try:
        if sidecar["document_kind"] != "checkpoint":
            raise ParentIdentityRefused(f"{what}: document_kind is {sidecar['document_kind']!r}")
        if sidecar["ancestry_hash"] != ca.compute_ancestry_hash(sidecar):
            raise ParentIdentityRefused(f"{what}: ancestry_hash does not match its payload")
        block = sidecar["checkpoint"]
        if block["checkpoint_id"] != ca.compute_checkpoint_id(
            sidecar["ancestry_hash"], block["checkpoint_sha256"]
        ):
            raise ParentIdentityRefused(f"{what}: checkpoint_id is not derived from its hashes")
        if block["base_ancestry_hash"] != base["ancestry_hash"]:
            raise ParentIdentityRefused(
                f"{what}: base_ancestry_hash {block['base_ancestry_hash']!r} != the frozen "
                f"ancestry base {base['ancestry_hash']!r}"
            )
    except (KeyError, TypeError) as exc:
        raise ParentIdentityRefused(f"{what} is malformed: {exc!r}") from exc
    s4 = sup.read_json(paths.parent_run_dir / S4_MANIFEST, "parent s4 run manifest")
    s1 = sup.read_json(paths.parent_run_dir / S1_MANIFEST, "parent s1 run manifest")
    tap = _get(s4, ("visual_input", "visual_feature_tap"), S4_MANIFEST)
    if tap != vip.visual_feature_tap():
        raise ParentIdentityRefused(
            f"parent trained on visual tap {tap!r}, the visual input policy gives "
            f"{vip.visual_feature_tap()!r}"
        )

    def identity() -> ReferencePolicyIdentity:  # two objects: the check refuses `is`
        return ReferencePolicyIdentity(
            ancestry_hash=sidecar["ancestry_hash"],
            checkpoint_id=block["checkpoint_id"],
            checkpoint_sha256=block["checkpoint_sha256"],
            typed_head_config_hash=_get(s4, ("typed_head_config_hash",), S4_MANIFEST),
            run_class=_get(s1, ("run_class",), S1_MANIFEST),
        )

    return sidecar, identity(), identity()


def assemble_inputs(paths: InputPaths) -> Assembled:
    """Verify every artefact by name and build the bound ``RunArtefacts`` (no model, no GPU)."""
    from unflatten.dataset.samples import release_root_hash
    from unflatten.model.capability_profile import resolve_capability_profile_by_hash
    from unflatten.model.rl.run_manifest import RunArtefacts, composite_policy_hash, policy_status
    from unflatten.model.rl.training_policy import check_execution_profile_matches, check_lr_groups

    policies = _policies(paths)
    heads = paths.selected_typed_heads
    check_execution_profile_matches(policies["training"], policies["rollout"])
    check_lr_groups(policies["training"], paths.run_class, heads)
    release_root = release_root_hash(paths.release_dir)
    manifest, selection, registry, base = _check_data(paths, release_root)
    identity, vip = _cache(paths, release_root)
    sidecar, reference, policy_parent = _parent(paths, base, vip)
    if selection.checkpoint_id != policy_parent.checkpoint_id:
        raise ParentIdentityRefused(
            f"frozen selection was derived for checkpoint {selection.checkpoint_id!r}, "
            f"the parent is {policy_parent.checkpoint_id!r}"
        )
    caps_sha = file_sha256(paths.loop_guard_caps)
    profile = resolve_capability_profile_by_hash(identity.get("capability_profile_hash"))
    caps = lg.verify_caps_binding(
        caps_path=paths.loop_guard_caps, caps_sha256=caps_sha, cache_identity=identity,
        capability_profile_id=profile.profile_id,
    )  # fmt: skip
    arts = RunArtefacts(
        reward_policy=policies["reward"], rollout_policy=policies["rollout"],
        dataset_manifest=manifest, stage_policy=policies["stage_gate"],
        training_policy=policies["training"], reference=reference, policy_parent=policy_parent,
        cache_identity=cache_identity_hash(identity), visual_feature_tap=vip.visual_feature_tap(),
    )  # fmt: skip
    composite = composite_policy_hash(arts, paths.run_class, heads)
    record_path = record_under_ratifications(paths.ratification_record, paths.decisions_dir)
    record = require_ratification(
        record_path, policy_hash=composite, policy_status=policy_status(arts),
        parent_checkpoint_id=policy_parent.checkpoint_id,
    )  # fmt: skip
    decision = resolve_decision(paths.decisions_dir, record.decision_id)
    return Assembled(
        paths, policies, manifest, selection, registry, base, identity, arts.cache_identity,
        vip, sidecar, reference, policy_parent, caps, caps_sha, record, record_path, decision,
        composite, replace(arts, ratification=record),
    )  # fmt: skip


def build_preview_manifest(assembled: Assembled) -> dict[str, Any]:
    """The ``rl-run-manifest.json`` the owner sees: exactly what ``run_s4_5`` derives."""
    from unflatten.model.rl.run_manifest import build_rl_run_manifest

    return build_rl_run_manifest(
        run_class=assembled.paths.run_class,
        run_seed=assembled.paths.run_seed,
        artefacts=assembled.artefacts,
        selected_typed_heads=assembled.paths.selected_typed_heads,
    )


# --- policy build record ------------------------------------------------------------------


def read_policy_build(
    policies_dir: Path | str, *, allowed_statuses: Collection[str] = ("RATIFIED",)
) -> dict[str, Any]:
    """``policy-build.json`` of ``build-policies``: the run class, seed and heads come from it.

    Only RATIFIED by default; the probe alone widens ``allowed_statuses`` (it runs before the
    owner freezes the policy numbers)."""
    build = sup.read_json(Path(policies_dir) / BUILD_FILE, "policy build record")
    if not isinstance(build, dict) or build.get("schema") != BUILD_SCHEMA:
        raise LaunchRefused(f"{BUILD_FILE} in {policies_dir}: schema is not {BUILD_SCHEMA!r}")
    if build.get("status") not in allowed_statuses:
        raise PolicyNotRatified(
            f"{BUILD_FILE} in {policies_dir} has status {build.get('status')!r}"
        )
    return build


def check_policy_build(build: Mapping[str, Any], assembled: Assembled) -> None:
    """The four policy files are the ones ``build-policies`` wrote; its composite is this one."""
    recorded = build.get("policy_hashes")
    if not isinstance(recorded, Mapping):
        raise LaunchRefused(f"{BUILD_FILE}: no policy_hashes object")
    for name, field in _HASH_FIELDS.items():
        actual = assembled.policies[name][field]
        if recorded.get(field) != actual:
            raise LaunchRefused(
                f"{BUILD_FILE} records {field} {recorded.get(field)!r}, "
                f"the policy file has {actual!r}"
            )
    composite = build.get("composite_policy_hash")
    if composite is not None and composite != assembled.composite:
        raise LaunchRefused(
            f"{BUILD_FILE} records composite {composite!r}, "
            f"the artefacts give {assembled.composite!r}"
        )


# --- the seal -------------------------------------------------------------------------------


def seal_text(run_dir: Path, final_dir: Path, extras: Iterable[Path]) -> bytes:
    """``sha256sum``-format lines (absolute paths) for every RUN file but the seal, plus ``extras``.

    ``run_dir`` is where the files are now (a staging directory), ``final_dir`` where they will
    live; the paths written are the final ones.
    """
    members = sorted(p for p in run_dir.iterdir() if p.name != SEAL_FILE)
    entries = [(final_dir / p.name, p) for p in members]
    entries += [(Path(e).resolve(), Path(e)) for e in sorted(set(extras))]
    lines = []
    for shown, actual in entries:
        text = str(shown)
        if "\n" in text or "\\" in text:
            raise RunDirRefused(f"{text!r}: a path with a newline or backslash cannot be sealed")
        lines.append(f"{file_sha256(actual)}  {text}\n")
    return "".join(lines).encode("utf-8")


def launch_document(values: Mapping[str, Any]) -> dict[str, Any]:
    """``rl-launch.json`` with the closed key set (execution values and paths only)."""
    doc = {"schema": LAUNCH_SCHEMA, **values}
    if set(doc) != set(LAUNCH_KEYS):
        raise LaunchRefused(f"{LAUNCH_FILE} keys {sorted(doc)} != {sorted(LAUNCH_KEYS)}")
    return doc


# --- the launch side (implemented in ``launch_run``; this module stays under its size cap) ---


def assemble(run_dir: Path | str) -> Any:
    """``RunAssembly`` of a sealed RUN directory (see :func:`launch_run.assemble`)."""
    from unflatten.model.rl.adapters import launch_run

    return launch_run.assemble(run_dir)


def build_ports(assembly: Any, dry_run: bool) -> Any:
    """``RealRunPorts`` over a ``RunAssembly``; opens nothing (:func:`launch_run.build_ports`)."""
    from unflatten.model.rl.adapters import launch_run

    return launch_run.build_ports(assembly, dry_run)

"""Path -> library wiring of the train taxonomy producer (heavy imports lazy).

``run_producer(config)`` composes the verified pieces: sidecar-derived checkpoint
id and the approved window first (window record loaded before the checkpoint is
hashed; nothing heavy read before they pass), then the
hash-verified release train records, the train exact-text cache, the candidate
opener and the free-running page decoder. ``seams`` replace the two parts that
cannot run on CPU (candidate load, page decoder); everything else is the real
code. Inference only: no ``torch.optim``, no optimiser, no backward.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from unflatten.model import checkpoint_ancestry as ca
from unflatten.model.hashing import domain_hash
from unflatten.model.rl.adapters import run_control
from unflatten.model.rl.adapters import taxonomy_assembly as ta
from unflatten.model.rl.adapters import taxonomy_producer as tp

SELECTION_RULE_DOMAIN = "unflatten.rl.train-taxonomy-selection-rule/v0"


@dataclass(frozen=True)
class ProducerConfig:
    checkpoint: Path
    sidecar: Path
    release_dir: Path
    store_dir: Path
    cache_dir: Path
    window_record: Path
    out: Path
    allowlist: Path
    expect_base_ancestry_hash: str
    expect_s1_run_manifest_hash: str | None
    expect_typed_head_config_hash: str
    page_indices: tuple[int, ...] = (0,)
    limit: int | None = None
    offset: int | None = None  # per page index k: keys[offset:offset+limit]; None = from the start
    stop_file: Path | None = None
    visual_input_policy: Path | None = None
    deepstack: bool = False
    device: str = "cuda"
    max_steps: int = 4608
    budget: int = 1000
    typed_decode: bool = False
    region_decode: bool = False
    loop_guard: str = "off"  # "off" | "v3"
    loop_guard_v3_arm: str = "r1"
    loop_guard_caps: Path | None = None  # None = the code tree's caps file


def decode_overrides(config: ProducerConfig) -> dict[str, Any]:
    """The ``DecodeConfig`` fields the producer flags switch on (empty = bare decode)."""
    if config.loop_guard not in ("off", "v3"):
        raise tp.ProducerRefused(f"--loop-guard must be 'off' or 'v3', got {config.loop_guard!r}")
    if config.loop_guard == "off" and config.loop_guard_caps is not None:
        raise tp.ProducerRefused("--loop-guard-caps needs --loop-guard v3")
    out: dict[str, Any] = {}
    if config.typed_decode:
        out["typed_decode"] = True
    if config.region_decode:
        out["region_decode"] = True
    if config.loop_guard == "v3":
        from unflatten.model.loop_guard_v3 import load_loop_guard_v3_policy, resolve_arm

        try:
            resolve_arm(load_loop_guard_v3_policy(), config.loop_guard_v3_arm)
        except ValueError as exc:
            raise tp.ProducerRefused(f"--loop-guard-v3-arm refused: {exc}") from exc
        out["loop_guard"] = "v3"
        out["loop_guard_v3_arm"] = config.loop_guard_v3_arm
    return out


def _caps_file(config: ProducerConfig) -> Path:
    from unflatten.model import loop_guard_v2_caps as caps_mod

    return Path(config.loop_guard_caps) if config.loop_guard_caps else caps_mod.CAPS_PATH


def decode_inputs(config: ProducerConfig) -> dict[str, Any]:
    """Report ``inputs`` naming what ran; keys are absent for flags that are off."""
    out = decode_overrides(config)
    if config.loop_guard == "v3":
        from unflatten.model.loop_guard_v2_caps import load_loop_guard_v2_caps

        # the canonical caps hash the service and the docs pin, not the file-bytes sha
        out["loop_guard_v2_caps_sha256"] = load_loop_guard_v2_caps(_caps_file(config)).sha256
    return out


def bind_loop_guard_caps(
    config: ProducerConfig, identity: Mapping[str, Any], capability_profile_id: str
) -> None:
    """Loop guard v3 reads the module's caps file at decode time. Verify the file binds the
    cache identity and the candidate's profile (named refusal), and point the module at
    it when ``--loop-guard-caps`` is given (as ``service.worker_cli`` does for a bundle)."""
    if config.loop_guard != "v3":
        return
    from unflatten.model import loop_guard_v2_caps as caps_mod
    from unflatten.model.rl.adapters.loop_guard import LoopGuardCapsRefused, verify_caps_binding

    path = _caps_file(config)
    try:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        verify_caps_binding(
            caps_path=path, caps_sha256=sha, cache_identity=identity,
            capability_profile_id=capability_profile_id,
        )  # fmt: skip
    except (OSError, LoopGuardCapsRefused) as exc:
        raise tp.ProducerRefused(f"loop guard caps refused: {exc}") from exc
    caps_mod.CAPS_PATH = path


@dataclass
class ProducerSeams:
    """Test/CPU seams; every field defaults to the real implementation."""

    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    load_candidate: Callable[[str], Any] | None = None
    load_tokenizer: Callable[[], Any] | None = None
    #: ``(opened, opener, identity) -> decode(page)``; ``None`` = the real free-running one.
    make_decoder: Callable[[Any, Any, Mapping[str, Any]], Callable[[Any], Any]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def select_pages(
    page_keys: Collection[tuple[str, int]],
    *,
    page_indices: tuple[int, ...],
    lineage: Mapping[str, str],
    limit: int | None,
    offset: int | None = None,
) -> list[tp.KeyPage]:
    """Cache keys with ``k`` in ``page_indices``, sorted, at most ``limit`` per ``k``,
    after skipping the first ``offset`` keys of each ``k``.
    A cache bundle that is not a verified train record is refused, never dropped."""
    alien = sorted({b for b, _ in page_keys if b not in lineage})
    if alien:
        raise tp.ProducerRefused(
            f"{len(alien)} cache bundle(s) are not verified train records, e.g. {alien[:3]}"
        )
    if limit is not None and limit < 1:
        raise tp.ProducerRefused(f"--limit must be >= 1, got {limit}")
    if offset is not None and offset < 0:
        raise tp.ProducerRefused(f"--offset must be >= 0, got {offset}")
    start = offset or 0
    chosen: list[tp.KeyPage] = []
    for k in sorted(set(page_indices)):
        keys = sorted(key for key in page_keys if key[1] == k)
        window = keys[start:] if limit is None else keys[start : start + limit]
        chosen.extend(tp.KeyPage((b, p)) for b, p in window)
    if not chosen:
        raise tp.ProducerRefused(f"no page selected (offset={offset}, limit={limit})")
    return chosen


def selection_payload(config: ProducerConfig) -> dict[str, Any]:
    """The hashed selection rule; ``offset`` appears only when given (older hashes unchanged)."""
    payload: dict[str, Any] = {
        "role": ta.TRAIN_TAXONOMY_ROLE,
        "page_indices": sorted(set(config.page_indices)),
        "limit": config.limit,
    }
    if config.offset is not None:
        payload["offset"] = config.offset
    return payload


def _real_decoder(
    config: ProducerConfig, reader: Any, identity: Mapping[str, Any], selection_hash: str
) -> Callable[[Any, Any, Mapping[str, Any]], Callable[[Any], Any]]:
    def make(opened: Any, opener: Any, _identity: Mapping[str, Any]) -> Callable[[Any], Any]:
        from transformers import AutoImageProcessor

        from unflatten.eval.s2_gate.cli import decode_visual_policy_for_checkpoint
        from unflatten.model import onfly_visual as ov
        from unflatten.model.candidate_loader import (
            QWEN_CANDIDATE_MODEL_ID,
            QWEN_CANDIDATE_REVISION,
        )
        from unflatten.model.rl.adapters.visual import make_build_visual

        overrides = decode_overrides(config)
        bind_loop_guard_caps(config, identity, opened.capability_profile_id)
        if config.visual_input_policy is None:
            raise tp.ProducerRefused("--visual-input-policy is required for the real decoder")
        cache_policy = ov.VisualInputPolicy.from_payload(
            json.loads(config.visual_input_policy.read_text())
        )
        policy = decode_visual_policy_for_checkpoint(cache_policy, opener.state)
        encoder = ov.OnTheFlyVisualEncoder(opener.candidate, policy=policy, device=config.device)

        class _Keys:
            def target_keys(self) -> tuple[tuple[str, int], ...]:
                return tuple((b, int(p)) for b, p in reader.page_keys)

        build_visual = make_build_visual(
            store_dir=config.store_dir,
            policy_payload=policy.payload(),
            processor=AutoImageProcessor.from_pretrained(
                QWEN_CANDIDATE_MODEL_ID, revision=QWEN_CANDIDATE_REVISION
            ),
            encoder=encoder,
            reader=reader,
            expectation=_Keys(),
            model=opened.policy,
            checkpoint_state=opener.state,
            feature_source=reader.identity.feature_tap,
        )
        return tp.make_free_running_page_decoder(
            opened,
            build_visual=build_visual,
            eval_cache_identity=identity,
            component_lock_hash=identity["component_lock_hash"],
            selection_policy_hash=selection_hash,
            max_steps=config.max_steps,
            device=config.device,
            decode_overrides=overrides,
        )

    return make


def run_producer(
    config: ProducerConfig, *, seams: ProducerSeams | None = None
) -> tp.ProducerResult:
    from unflatten.eval import cohort_gold
    from unflatten.model.exact_text_cache import ExactTextCacheReader
    from unflatten.model.rl.adapters import candidate as cand
    from unflatten.model.rl.adapters.release import read_split_records_verified

    seams = seams or ProducerSeams()
    inputs_extra = decode_inputs(config)  # refuses a bad flag combination before any I/O
    if Path(config.sidecar) != ca.sidecar_path(Path(config.checkpoint)):
        raise tp.ProducerRefused(
            f"--sidecar {config.sidecar} is not the ancestry sidecar of {config.checkpoint}"
        )
    window = tp.load_window_record(config.window_record)  # a malformed window: no checkpoint I/O
    checkpoint_id = tp.resolve_checkpoint_id(
        config.checkpoint,
        expect_base_ancestry_hash=config.expect_base_ancestry_hash,
        expect_s1_run_manifest_hash=config.expect_s1_run_manifest_hash,
    )
    tp.require_window(window, checkpoint_id=checkpoint_id, now=seams.clock())

    lineage = ta.bundle_lineage_from_split_records(read_split_records_verified(config.release_dir))
    reader = ExactTextCacheReader(config.cache_dir, gradient_bearing=False)
    identity = dataclasses.asdict(reader.identity)
    pages = select_pages(
        reader.page_keys,
        page_indices=config.page_indices,
        lineage=lineage,
        limit=config.limit,
        offset=config.offset,
    )
    selection_hash = domain_hash(SELECTION_RULE_DOMAIN, selection_payload(config))
    if config.offset is not None:
        inputs_extra["offset"] = config.offset
    opener = cand.make_open_candidate(
        checkpoint_path=config.checkpoint,
        expect_base_ancestry_hash=config.expect_base_ancestry_hash,
        expect_s1_run_manifest_hash=config.expect_s1_run_manifest_hash,
        expect_typed_head_config_hash=config.expect_typed_head_config_hash,
        deepstack=config.deepstack,
        device=config.device,
        tokenizer_artifact_set_hash=identity["tokenizer_artifact_set_hash"],
        allowlist_payload=json.loads(config.allowlist.read_text()),
        expect_vocabulary_hash=identity["composite_vocabulary_hash"],
        capability_profile_hash=identity["capability_profile_hash"],
        load_candidate=seams.load_candidate,
        load_tokenizer=seams.load_tokenizer,
    )
    maker = seams.make_decoder or _real_decoder(config, reader, identity, selection_hash)
    return tp.produce_train_failure_report(
        window=window,
        checkpoint_id=checkpoint_id,
        open_candidate=opener,
        make_decoder=lambda opened: maker(opened, opener, identity),
        pages=pages,
        page_indices=config.page_indices,
        cache_identity=identity,
        load_gold=lambda bundles, k: cohort_gold.load_gold_rows(
            config.cache_dir, bundles, page_index=k
        ),
        bundle_lineage=lineage,
        allowed_lineage_ids=set(lineage.values()),
        budget=config.budget,
        out_dir=config.out,
        stop_requested=(
            run_control.make_stop_requested(config.stop_file)
            if config.stop_file is not None
            else (lambda: False)
        ),
        clock=seams.clock,
        decode_inputs=inputs_extra,
    )

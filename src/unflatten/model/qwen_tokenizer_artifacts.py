"""`tokenizer_artifact_set_hash` — what binds a run to the exact tokenizer
bytes it used (design §4.2).

`ExactTextCacheIdentity.tokenizer_artifact_set_hash` and the Component
Lock's `text_tokenizer` component have both been *required* fields since
M1, but nothing in this repository ever computed one: `grep` found only
parameter declarations. This module is the producer.

Two deliberate properties:

- **A missing required file raises.** There is no "hash whatever happened
  to be there" path -- a snapshot without `tokenizer.json` is an
  incomplete download or a foreign directory, and hashing it would mint a
  confident identity for something that cannot tokenize.
- **`ignored_file_names` carries names, never contents.** Every other file
  in the snapshot directory contributes its name to the hashed payload, so
  a new tokenizer-relevant file appearing upstream moves the hash and
  forces a look -- without this function ever reading the ~8 GB of
  safetensors shards that live in the same directory.

This module authorizes no gradient update: pure functions over files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from unflatten.model.candidate_loader import QWEN_CANDIDATE_MODEL_ID, QWEN_CANDIDATE_REVISION
from unflatten.model.hashing import domain_hash

__all__ = [
    "OPTIONAL_TOKENIZER_FILES",
    "QWEN_CANDIDATE_MODEL_ID",
    "QWEN_CANDIDATE_REVISION",
    "REQUIRED_TOKENIZER_FILES",
    "TOKENIZER_ARTIFACT_SET_DOMAIN",
    "TokenizerArtifactSetInvalid",
    "load_pinned_tokenizer",
    "tokenizer_artifact_set_hash",
]

TOKENIZER_ARTIFACT_SET_DOMAIN = "rfc004.qwen-tokenizer-artifact-set/v1"

#: Without these the directory cannot define a tokenizer at all.
REQUIRED_TOKENIZER_FILES: frozenset[str] = frozenset({"tokenizer.json", "tokenizer_config.json"})

#: Hashed by content when present, absent without complaint: which of these
#: a snapshot ships is a property of the upstream repository (the pinned
#: Qwen3-VL snapshot has `vocab.json` and `merges.txt` but no
#: `special_tokens_map.json`), not of whether the tokenizer is complete.
OPTIONAL_TOKENIZER_FILES: frozenset[str] = frozenset(
    {
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    }
)

_HASHED_FILES: frozenset[str] = REQUIRED_TOKENIZER_FILES | OPTIONAL_TOKENIZER_FILES


class TokenizerArtifactSetInvalid(ValueError):
    """A snapshot directory cannot produce a tokenizer artifact-set hash.

    M1-internal (design §9.3); not an RFC004 registry code.
    """


def _file_entry(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
        "size": len(data),
    }


def tokenizer_artifact_set_hash(
    snapshot_dir: Path | str, *, model_id: str, revision: str
) -> str:
    """Hash the tokenizer-defining files of one snapshot directory.

    `model_id` and `revision` ride inside the hashed payload so a
    tokenizer loaded at a different revision than the model it is paired
    with is visible as a hash difference -- the failure the DEC-007 D3 pin
    exists to prevent.
    """
    directory = Path(snapshot_dir)
    if not directory.is_dir():
        raise TokenizerArtifactSetInvalid(f"{directory} is not a directory")

    present = {p.name for p in directory.iterdir() if p.is_file()}
    missing = sorted(REQUIRED_TOKENIZER_FILES - present)
    if missing:
        raise TokenizerArtifactSetInvalid(
            f"{directory} is missing required tokenizer file(s): {', '.join(missing)} "
            "-- an incomplete snapshot is refused, never hashed as if it were whole"
        )

    files = [_file_entry(directory / name) for name in sorted(present & _HASHED_FILES)]
    ignored = sorted(present - _HASHED_FILES)
    return domain_hash(
        TOKENIZER_ARTIFACT_SET_DOMAIN,
        {
            "model_id": model_id,
            "revision": revision,
            "files": files,
            "ignored_file_names": ignored,
        },
    )


def load_pinned_tokenizer(
    *, model_id: str = QWEN_CANDIDATE_MODEL_ID, revision: str = QWEN_CANDIDATE_REVISION
) -> tuple[Any, Path]:
    """Load the DEC-007 D3-pinned tokenizer and return it with the snapshot
    directory it came from.

    `candidate_loader.QwenCandidate` carries no tokenizer, so the tokenizer
    is loaded independently. The snapshot directory is then resolved for
    the SAME `(model_id, revision)` pair out of the local hub cache, and
    cross-checked against any local file path the loaded tokenizer itself
    reports: hashing one directory while having tokenized from another is
    precisely the drift this pin exists to prevent.
    """
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    cached = try_to_load_from_cache(model_id, "tokenizer.json", revision=revision)
    if not isinstance(cached, str):
        raise TokenizerArtifactSetInvalid(
            f"no local snapshot for {model_id}@{revision} in the hub cache "
            f"(try_to_load_from_cache returned {cached!r})"
        )
    snapshot = Path(cached).parent
    for key in ("vocab_file", "merges_file"):
        reported = tokenizer.init_kwargs.get(key)
        if isinstance(reported, str) and Path(reported).parent != snapshot:
            raise TokenizerArtifactSetInvalid(
                f"tokenizer loaded its {key} from {Path(reported).parent}, but the "
                f"{model_id}@{revision} snapshot resolved to {snapshot}"
            )
    return tokenizer, snapshot

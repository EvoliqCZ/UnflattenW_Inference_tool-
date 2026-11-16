#!/usr/bin/env bash
# Builds local model bundle variants for the `docker compose` deployment
# (design §8.3) and writes `deploy/docker/models.json` -- the WORKER always
# loads its resident variant from that file now (T47, compose.yaml hardcodes
# UNFLATTEN_MODELS_CONFIG=/config/models.json); `deploy/local/run-local.sh up`
# refuses to start if it is missing or fails validation.
#
# Unlike `deploy/local/dev-native.sh` (bundles under
# $UNFLATTEN_LOCAL_DATA_DIR/model-bundles for the native `service.sh` path),
# bundles here live under deploy/docker/models/<precision>-<kv_cache> -- the
# same host directory `UNFLATTEN_HOST_MODEL_BUNDLE_DIR` (default ./models)
# bind-mounts into the api/worker containers at /models, so models.json's
# `bundle` paths are the CONTAINER path /models/<precision>-<kv_cache>, not
# a host path.
#
# Usage: deploy/local/dev-bundle.sh [--checkpoint PATH] [--run-dir PATH]

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKER_DIR="$REPO_ROOT/deploy/docker"

: "${UNFLATTEN_CHECKPOINT:=$HOME/unflatten-runs/ckpt-v15b-140000/checkpoint.pt}"
: "${UNFLATTEN_RUN_DIR:=$(dirname "$UNFLATTEN_CHECKPOINT")}"
: "${UNFLATTEN_PYTHON:=uv run --group ml --group dev --group server --no-sync python}"

MODEL_BUNDLE_ROOT="$DOCKER_DIR/models"
MODELS_CONFIG_PATH="$DOCKER_DIR/models.json"

log() { echo "dev-bundle.sh: $*"; }
die() {
  echo "dev-bundle.sh: $*" >&2
  exit 1
}

usage() {
  cat >&2 <<'EOF'
usage: deploy/local/dev-bundle.sh [--checkpoint PATH] [--run-dir PATH]

Builds every precision/kv_cache combo this checkpoint/GPU can produce under
deploy/docker/models/<precision>-<kv_cache> and writes
deploy/docker/models.json (bundle-source variants, container paths
/models/<precision>-<kv_cache>, plus the unpublished/planned catalogue
entries of deploy/docker/models.example.json). At least one combo must
succeed.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint)
      UNFLATTEN_CHECKPOINT="$2"
      shift 2
      ;;
    --run-dir)
      UNFLATTEN_RUN_DIR="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage
      die "unknown argument: $1"
      ;;
  esac
done

_build_one_bundle() {
  local precision="$1" kv_cache="$2" out_dir="$3"
  if [ -d "$out_dir" ]; then
    log "reusing existing model bundle at $out_dir"
    return 0
  fi
  log "building model bundle at $out_dir (precision=$precision kv_cache=$kv_cache)"
  (
    cd "$REPO_ROOT" && $UNFLATTEN_PYTHON scripts/build_model_bundle.py build \
      --checkpoint "$UNFLATTEN_CHECKPOINT" \
      --run-dir "$UNFLATTEN_RUN_DIR" \
      --precision "$precision" \
      --kv-cache "$kv_cache" \
      --out "$out_dir"
  )
}

# DEC-044 Dodatok D §1: build every precision variant this checkpoint can
# produce and write deploy/docker/models.json so the compose worker's
# --models-config path can offer all of them. Each variant build is
# independent and best-effort -- a variant this GPU/checkpoint cannot
# produce is skipped with a warning, never aborts the whole script (only no
# variant at all is fatal, since models.json needs a default_variant).
write_models_config() {
  local combos=("bf16 bf16" "int8 bf16" "fp8 fp8" "int4 fp8")
  local variants_jsonl="" combo precision kv_cache dir default_variant=""
  mkdir -p "$MODEL_BUNDLE_ROOT"
  for combo in "${combos[@]}"; do
    read -r precision kv_cache <<<"$combo"
    dir="$MODEL_BUNDLE_ROOT/$precision-$kv_cache"
    if _build_one_bundle "$precision" "$kv_cache" "$dir"; then
      [ -z "$default_variant" ] && default_variant="$precision-$kv_cache"
      variants_jsonl="$variants_jsonl$precision $kv_cache /models/$precision-$kv_cache
"
    else
      log "warning: skipping models config variant $precision/$kv_cache (build failed)"
    fi
  done
  [ -n "$default_variant" ] \
    || die "no model bundle variant could be built -- see the warnings above (deploy/local/README.md documents the --run-dir component-lock blocker)"
  # Owner 2026-09-29: the file also lists the not-yet-published catalogue
  # entries of deploy/docker/models.example.json (write_models_config.py).
  printf '%s' "$variants_jsonl" | (
    cd "$REPO_ROOT" && $UNFLATTEN_PYTHON deploy/local/write_models_config.py \
      --default "$default_variant" --out "$MODELS_CONFIG_PATH"
  ) && log "wrote models config to $MODELS_CONFIG_PATH (default variant: $default_variant)"
}

write_models_config

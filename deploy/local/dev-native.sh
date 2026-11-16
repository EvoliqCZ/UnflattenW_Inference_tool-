#!/usr/bin/env bash
# One-command NATIVE (non-docker-compose) local launcher for the private
# diagnostic service on a single-user workstation with an NVIDIA GPU (owner
# requirement 2026-09-27, generalised beyond one card: "sprav to tak, že to
# nebude len pre RTX 4070, ale všeobecne"). Wraps deploy/service/service.sh;
# adds nothing service.sh itself doesn't already support.
#
# T47: this file WAS `deploy/local/run-local.sh` -- that name is now the
# docker-compose wrapper (design §8.5, plan T47 note 7). Behaviour here is
# otherwise unchanged; the only addition is step 0 below (building the SPA
# once, so a native `start` serves the real web UI instead of failing
# closed on a missing `dist/` -- plan note 2, chunk 4 drops `legacy`).
#
# What this does, in order:
#   0. builds the web SPA once, if `src/unflatten/service/ui/dist` does not
#      exist yet (`npm --prefix web ci && npm --prefix web run build`, then
#      copies `web/dist` there);
#   1. creates $UNFLATTEN_LOCAL_DATA_DIR ($HOME/unflatten-service-local by
#      default);
#   2. generates (or reuses) a principal file with one bearer token holding
#      every capability, and PRINTS the token;
#   3. locates or builds the PDF sandbox image (RUNBOOK §10);
#   4. resolves --precision (default "auto": measures this GPU's free VRAM
#      and compute capability via quantization.auto_policy -- general
#      across cards, not keyed on the RTX 4070) and builds/reuses a LOCAL
#      model bundle at that resolved precision, from a LOCAL checkpoint --
#      never a prod bundle (its component lock/tokenizer identity is
#      pinned to prod's own HF snapshot, which hashes differently locally,
#      memory `tokenizer-hash-counts-ignored-names`);
#   5. starts API + worker via deploy/service/service.sh.
#
# Usage: deploy/local/dev-native.sh {start|stop|status|verify}

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${UNFLATTEN_LOCAL_DATA_DIR:=$HOME/unflatten-service-local}"
# UNFLATTEN_CHECKPOINT (owner-facing name) picks which local training
# checkpoint to build the bundle from; UNFLATTEN_LOCAL_CHECKPOINT is kept
# as a back-compat alias for anyone who already set it.
: "${UNFLATTEN_CHECKPOINT:=${UNFLATTEN_LOCAL_CHECKPOINT:-$HOME/unflatten-runs/ckpt-v15b-140000/checkpoint.pt}}"
UNFLATTEN_LOCAL_CHECKPOINT="$UNFLATTEN_CHECKPOINT"
: "${UNFLATTEN_LOCAL_RUN_DIR:=$(dirname "$UNFLATTEN_LOCAL_CHECKPOINT")}"
: "${UNFLATTEN_LOCAL_SANDBOX_TAG:=unflatten/pdf-sandbox:local}"
: "${UNFLATTEN_PRECISION:=auto}"
: "${UNFLATTEN_KV_CACHE:=}"
: "${UNFLATTEN_LOCAL_PYTHON:=uv run --group ml --group dev --group server --no-sync python}"

PRINCIPAL_FILE="$UNFLATTEN_LOCAL_DATA_DIR/principal.json"
MODEL_BUNDLE_ROOT="$UNFLATTEN_LOCAL_DATA_DIR/model-bundles"

log() { echo "dev-native.sh: $*"; }
die() {
  echo "dev-native.sh: $*" >&2
  exit 1
}

ensure_data_dir() {
  mkdir -p "$UNFLATTEN_LOCAL_DATA_DIR"
  chmod 0700 "$UNFLATTEN_LOCAL_DATA_DIR"
  mkdir -p "$MODEL_BUNDLE_ROOT"
}

# T47/plan note 2: `api_cli --web-ui spa` (and chunk 4's default) serves
# `src/unflatten/service/ui/dist` -- Docker builds it via the `web-builder`
# stage (deploy/docker/Dockerfile), but a native start has no such build
# step. Build it once here and reuse it on every later start, exactly like
# the model bundle build above.
WEB_DIST_DIR="$REPO_ROOT/src/unflatten/service/ui/dist"

ensure_web_dist() {
  if [ -d "$WEB_DIST_DIR" ]; then
    log "reusing existing web build at $WEB_DIST_DIR"
    return 0
  fi
  command -v npm >/dev/null 2>&1 || die "npm not found on PATH -- required to build web/ (npm --prefix web ci && npm --prefix web run build)"
  log "building web SPA (npm --prefix web ci && npm --prefix web run build)"
  (cd "$REPO_ROOT" && npm --prefix web ci && npm --prefix web run build) \
    || die "web build failed -- see the npm output above"
  rm -rf "$WEB_DIST_DIR"
  mkdir -p "$(dirname "$WEB_DIST_DIR")"
  cp -R "$REPO_ROOT/web/dist" "$WEB_DIST_DIR" \
    || die "cannot copy web/dist to $WEB_DIST_DIR"
}

ensure_principal() {
  if [ -f "$PRINCIPAL_FILE" ]; then
    log "principal file already exists at $PRINCIPAL_FILE"
  else
    log "generating a new local principal (all capabilities)"
    local token
    token="$(openssl rand -hex 32)"
    (
      umask 0377
      python3 -c "
import json, sys
token = sys.argv[1]
doc = {
    'principal_file_schema_version': '1.0.0',
    'principals': [
        {
            'principal_id': 'local-owner',
            'token': token,
            'capabilities': {
                'submit_diagnostic': True,
                'read_status': True,
                'download_prediction_ir': True,
                'cancel_own_job': True,
                'download_docx': True,
            },
        }
    ],
}
with open(sys.argv[2], 'w') as f:
    json.dump(doc, f)
" "$token" "$PRINCIPAL_FILE"
    )
    chmod 0400 "$PRINCIPAL_FILE"
  fi
  UNFLATTEN_TOKEN="$(python3 -c "import json; print(json.load(open('$PRINCIPAL_FILE'))['principals'][0]['token'])")"
  echo "dev-native.sh: bearer token for API clients (curl -H \"Authorization: Bearer ...\"; the web UI's first-run setup asks only for a password):"
  echo "$UNFLATTEN_TOKEN"
}

ensure_sandbox_image() {
  # The worker refuses an image whose scanner-identity label differs from
  # pdf_sandbox.SCANNER_IDENTITY, so a stale local tag (e.g. an old v7 build)
  # is rebuilt here instead of failing later at worker readiness.
  local expected actual
  expected="$(cd "$REPO_ROOT" && $UNFLATTEN_LOCAL_PYTHON -c 'from unflatten.service.pdf_sandbox import SCANNER_IDENTITY; print(SCANNER_IDENTITY)')" \
    || die "cannot read the expected sandbox scanner identity"
  actual="$(docker image inspect --format '{{index .Config.Labels "org.unflatten.pdf-scanner.identity"}}' "$UNFLATTEN_LOCAL_SANDBOX_TAG" 2>/dev/null || true)"
  if [ "$actual" = "$expected" ]; then
    log "sandbox image $UNFLATTEN_LOCAL_SANDBOX_TAG matches scanner identity"
  else
    log "building sandbox image $UNFLATTEN_LOCAL_SANDBOX_TAG (found identity '${actual:-none}', need '$expected'; RUNBOOK §10)"
    docker build -f "$REPO_ROOT/deploy/pdf-sandbox/Containerfile" \
      -t "$UNFLATTEN_LOCAL_SANDBOX_TAG" "$REPO_ROOT/deploy/pdf-sandbox" || die "sandbox image build failed"
  fi
  SANDBOX_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$UNFLATTEN_LOCAL_SANDBOX_TAG")" || die "cannot resolve sandbox image id"
}

resolve_precision() {
  if [ "$UNFLATTEN_PRECISION" != "auto" ]; then
    RESOLVED_PRECISION="$UNFLATTEN_PRECISION"
    RESOLVED_KV_CACHE="${UNFLATTEN_KV_CACHE:-bf16}"
    log "using explicit precision=$RESOLVED_PRECISION kv_cache=$RESOLVED_KV_CACHE"
    return 0
  fi
  local resolved
  resolved="$(
    cd "$REPO_ROOT" && $UNFLATTEN_LOCAL_PYTHON -c "
import torch
from unflatten.model.quantization.auto_policy import resolve_auto_quantization
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
capability = torch.cuda.get_device_capability(0)
spec = resolve_auto_quantization(
    free_vram_bytes=free_bytes, total_vram_bytes=total_bytes, compute_capability=capability
)
print(spec.decoder_precision)
print(spec.kv_cache_dtype)
"
  )" || die "auto precision resolution failed (no CUDA GPU, or nothing fits -- see the printed refusal above)"
  RESOLVED_PRECISION="$(echo "$resolved" | sed -n '1p')"
  RESOLVED_KV_CACHE="$(echo "$resolved" | sed -n '2p')"
  log "auto-resolved precision=$RESOLVED_PRECISION kv_cache=$RESOLVED_KV_CACHE from measured free VRAM/compute capability"
}

_build_one_bundle() {
  local precision="$1" kv_cache="$2" out_dir="$3"
  if [ -d "$out_dir" ]; then
    log "reusing existing model bundle at $out_dir"
    return 0
  fi
  log "building local model bundle at $out_dir (precision=$precision kv_cache=$kv_cache)"
  (
    cd "$REPO_ROOT" && $UNFLATTEN_LOCAL_PYTHON scripts/build_model_bundle.py build \
      --checkpoint "$UNFLATTEN_LOCAL_CHECKPOINT" \
      --run-dir "$UNFLATTEN_LOCAL_RUN_DIR" \
      --precision "$precision" \
      --kv-cache "$kv_cache" \
      --out "$out_dir"
  )
}

ensure_model_bundle() {
  MODEL_BUNDLE_DIR="$MODEL_BUNDLE_ROOT/v15b-$RESOLVED_PRECISION-$RESOLVED_KV_CACHE"
  _build_one_bundle "$RESOLVED_PRECISION" "$RESOLVED_KV_CACHE" "$MODEL_BUNDLE_DIR" \
    || die "model bundle build failed -- --run-dir=$UNFLATTEN_LOCAL_RUN_DIR must carry component-lock.json, qwen_text_allowlist.json, loop_guard_v2_caps.json and qwen-pinned-hashes.json alongside the checkpoint (see deploy/local/README.md)"
  MODEL_BUNDLE_SHA256="$(
    cd "$REPO_ROOT" && $UNFLATTEN_LOCAL_PYTHON -c "
from unflatten.service.model_bundle import manifest_sha256
print(manifest_sha256('$MODEL_BUNDLE_DIR'))
"
  )" || die "cannot compute model bundle manifest sha256"
}

# DEC-044 Dodatok D §1: build every precision variant this checkpoint can
# produce and write a models config JSON so the API/worker's --models-config
# path (docker compose deployment) can offer all of them, not only the one
# `resolve_precision` picked for THIS start. Each variant build is
# independent and best-effort -- a variant this GPU/checkpoint cannot
# produce is skipped with a warning rather than failing the whole script;
# the single resolved bundle from `ensure_model_bundle` above is always
# built and always included. NOTE: this path has not been exercised against
# a real checkpoint (see deploy/local/README.md's tokenizer-hash blocker);
# treat it as unverified until that blocker is resolved.
write_models_config() {
  local combos=("bf16 bf16" "int8 bf16" "fp8 fp8" "int4 fp8")
  local variants_jsonl="" combo precision kv_cache dir
  for combo in "${combos[@]}"; do
    read -r precision kv_cache <<<"$combo"
    dir="$MODEL_BUNDLE_ROOT/v15b-$precision-$kv_cache"
    if { [ "$precision" = "$RESOLVED_PRECISION" ] && [ "$kv_cache" = "$RESOLVED_KV_CACHE" ]; } \
      || _build_one_bundle "$precision" "$kv_cache" "$dir"; then
      variants_jsonl="$variants_jsonl$precision $kv_cache $dir
"
    else
      log "warning: skipping models config variant $precision/$kv_cache (build failed)"
    fi
  done
  MODELS_CONFIG_PATH="$UNFLATTEN_LOCAL_DATA_DIR/models.json"
  # Owner 2026-09-29: the file also lists the not-yet-published catalogue
  # entries of deploy/docker/models.example.json (write_models_config.py).
  printf '%s' "$variants_jsonl" | (
    cd "$REPO_ROOT" && $UNFLATTEN_LOCAL_PYTHON deploy/local/write_models_config.py \
      --default "$RESOLVED_PRECISION-$RESOLVED_KV_CACHE" --out "$MODELS_CONFIG_PATH"
  ) && log "wrote models config to $MODELS_CONFIG_PATH (default variant: $RESOLVED_PRECISION-$RESOLVED_KV_CACHE)"
}

do_start() {
  ensure_web_dist
  ensure_data_dir
  ensure_principal
  ensure_sandbox_image
  resolve_precision
  ensure_model_bundle
  write_models_config
  # The models config is handed to the service by default now that the
  # worker measures `fits` from total VRAM (the resident variant always
  # fits), publishes its resident variant/device facts after every load, and
  # switches variants live on request (fixed 2026-09-27, was refused with
  # MODEL_VARIANT_DOES_NOT_FIT). Set UNFLATTEN_LOCAL_MODEL_PICKER=0 to opt
  # back into the single fixed --model-bundle path.
  if [ "${UNFLATTEN_LOCAL_MODEL_PICKER:-1}" = "1" ] && [ -f "${MODELS_CONFIG_PATH:-}" ]; then
    export UNFLATTEN_MODELS_CONFIG="$MODELS_CONFIG_PATH"
  fi
  export UNFLATTEN_DATA_DIR="$UNFLATTEN_LOCAL_DATA_DIR"
  export UNFLATTEN_MODEL_BUNDLE="$MODEL_BUNDLE_DIR"
  export UNFLATTEN_MODEL_BUNDLE_SHA256="$MODEL_BUNDLE_SHA256"
  export UNFLATTEN_PRINCIPAL_FILE="$PRINCIPAL_FILE"
  export UNFLATTEN_SANDBOX_IMAGE="$SANDBOX_IMAGE_ID"
  export UNFLATTEN_SANDBOX_RUNTIME=docker
  # Owner-local choice for a single-user workstation only (RUNBOOK §10):
  # this host's docker is rootful, so the sandbox needs the explicit
  # opt-in. Never set this default in deploy/service/service.sh or
  # deploy/service/service.env.example -- rootful docker is root-equivalent
  # (RUNBOOK §10 checklist item 5) and must stay an explicit choice for
  # every OTHER deployment.
  export UNFLATTEN_SANDBOX_ALLOW_ROOTFUL=1
  export UNFLATTEN_HOST=127.0.0.1
  # Owner 2026-09-29 (DEC-052 addendum): loopback-only bind above, so the
  # first-run setup wizard asks only for a password (no token to paste).
  export UNFLATTEN_SETUP_TOKENLESS=1
  export UNFLATTEN_PORT="${UNFLATTEN_PORT:-8420}"
  export UNFLATTEN_DEVICE="${UNFLATTEN_DEVICE:-cuda:0}"
  export UNFLATTEN_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
  export UNFLATTEN_MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17
  export UNFLATTEN_PRECISION="$RESOLVED_PRECISION"
  export UNFLATTEN_KV_CACHE="$RESOLVED_KV_CACHE"
  export UNFLATTEN_PYTHON="$UNFLATTEN_LOCAL_PYTHON"
  "$REPO_ROOT/deploy/service/service.sh" start
}

do_verify() {
  ensure_data_dir
  ensure_sandbox_image
  resolve_precision
  ensure_model_bundle
  export UNFLATTEN_DATA_DIR="$UNFLATTEN_LOCAL_DATA_DIR"
  export UNFLATTEN_MODEL_BUNDLE="$MODEL_BUNDLE_DIR"
  export UNFLATTEN_MODEL_BUNDLE_SHA256="$MODEL_BUNDLE_SHA256"
  export UNFLATTEN_PRINCIPAL_FILE="$PRINCIPAL_FILE"
  export UNFLATTEN_SANDBOX_IMAGE="$SANDBOX_IMAGE_ID"
  export UNFLATTEN_SANDBOX_RUNTIME=docker
  export UNFLATTEN_SANDBOX_ALLOW_ROOTFUL=1
  export UNFLATTEN_HOST=127.0.0.1
  export UNFLATTEN_PORT="${UNFLATTEN_PORT:-8420}"
  export UNFLATTEN_DEVICE="${UNFLATTEN_DEVICE:-cuda:0}"
  export UNFLATTEN_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
  export UNFLATTEN_MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17
  export UNFLATTEN_PYTHON="$UNFLATTEN_LOCAL_PYTHON"
  "$REPO_ROOT/deploy/service/service.sh" verify
}

do_stop() {
  export UNFLATTEN_DATA_DIR="$UNFLATTEN_LOCAL_DATA_DIR"
  export UNFLATTEN_MODEL_BUNDLE="${UNFLATTEN_MODEL_BUNDLE:-$MODEL_BUNDLE_ROOT/placeholder}"
  export UNFLATTEN_PRINCIPAL_FILE="$PRINCIPAL_FILE"
  export UNFLATTEN_SANDBOX_IMAGE="${UNFLATTEN_SANDBOX_IMAGE:-sha256:0}"
  export UNFLATTEN_SANDBOX_RUNTIME=docker
  export UNFLATTEN_PORT="${UNFLATTEN_PORT:-8420}"
  export UNFLATTEN_DEVICE="${UNFLATTEN_DEVICE:-cuda:0}"
  export UNFLATTEN_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
  export UNFLATTEN_MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17
  "$REPO_ROOT/deploy/service/service.sh" stop
}

do_status() {
  export UNFLATTEN_DATA_DIR="$UNFLATTEN_LOCAL_DATA_DIR"
  export UNFLATTEN_PORT="${UNFLATTEN_PORT:-8420}"
  export UNFLATTEN_HOST=127.0.0.1
  export UNFLATTEN_MODEL_BUNDLE="${UNFLATTEN_MODEL_BUNDLE:-$MODEL_BUNDLE_ROOT/placeholder}"
  export UNFLATTEN_PRINCIPAL_FILE="$PRINCIPAL_FILE"
  export UNFLATTEN_SANDBOX_IMAGE="${UNFLATTEN_SANDBOX_IMAGE:-sha256:0}"
  export UNFLATTEN_SANDBOX_RUNTIME=docker
  export UNFLATTEN_DEVICE="${UNFLATTEN_DEVICE:-cuda:0}"
  export UNFLATTEN_MODEL_ID=Qwen/Qwen3-VL-4B-Instruct
  export UNFLATTEN_MODEL_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17
  "$REPO_ROOT/deploy/service/service.sh" status
}

main() {
  local action="${1:-}"
  case "$action" in
    start) do_start ;;
    stop) do_stop ;;
    status) do_status ;;
    verify) do_verify ;;
    *)
      echo "usage: $0 {start|stop|status|verify}" >&2
      return 2
      ;;
  esac
}

main "$@"

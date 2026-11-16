#!/usr/bin/env bash
# Runs `unflatten.service.worker_cli` in the `worker` compose container.
#
# THE ONE PLACE that maps env vars to worker_cli flags.
#
# `PdfSandbox.require_ready()` (src/unflatten/service/pdf_sandbox.py) runs
# BEFORE any model is loaded (worker_cli.main), so without the docker-socket
# override (compose.sandbox-socket.yaml) this fails fast on a missing/unready
# sandbox runtime -- confirmed by design, exercised in
# deploy/docker/README.md's no-GPU verification steps.
set -euo pipefail

: "${UNFLATTEN_DATA_DIR:=/data}"
# T47: exactly one of UNFLATTEN_MODEL_BUNDLE / UNFLATTEN_MODELS_CONFIG must be
# set -- worker_cli itself refuses both-or-neither, so this stays a plain
# default here rather than a duplicate check; compose.yaml now sets
# UNFLATTEN_MODELS_CONFIG by default (design §8.2), UNFLATTEN_MODEL_BUNDLE
# is the older single-bundle path some other deployment may still use.
: "${UNFLATTEN_MODEL_BUNDLE:=}"
: "${UNFLATTEN_SANDBOX_IMAGE:?UNFLATTEN_SANDBOX_IMAGE is required even to fail closed cleanly (image ref checked at readiness time)}"
: "${UNFLATTEN_SANDBOX_RUNTIME:=docker}"
: "${UNFLATTEN_DEVICE:=cuda:0}"
: "${UNFLATTEN_MODEL_ID:?UNFLATTEN_MODEL_ID is required}"
: "${UNFLATTEN_MODEL_REVISION:?UNFLATTEN_MODEL_REVISION is required}"
: "${UNFLATTEN_ATTENTION_IMPLEMENTATION:=sdpa}"
: "${UNFLATTEN_WORKER_ID:=worker_gpu_0}"
: "${UNFLATTEN_IDLE_SECONDS:=1.0}"

mkdir -p "$UNFLATTEN_DATA_DIR/artifacts" "$UNFLATTEN_DATA_DIR/work"

# UNFLATTEN_PRECISION / UNFLATTEN_KV_CACHE map to worker_cli --precision /
# --kv-cache (empty = worker_cli default bf16). UNFLATTEN_MODELS_CONFIG maps
# to --models-config -- mutually exclusive with UNFLATTEN_MODEL_BUNDLE at the
# worker_cli level (do not set both). UNFLATTEN_ENGINES maps to --engines
# (default "native"); UNFLATTEN_MODEL_VARIANT maps to --variant, passed
# only when set -- absent, worker_cli resolves the config's own
# default_variant (T43).
: "${UNFLATTEN_MODELS_CONFIG:=}"
: "${UNFLATTEN_PRECISION:=}"
: "${UNFLATTEN_KV_CACHE:=}"
: "${UNFLATTEN_ENGINES:=native}"
: "${UNFLATTEN_MODEL_VARIANT:=}"
export UNFLATTEN_MODELS_CONFIG UNFLATTEN_PRECISION UNFLATTEN_KV_CACHE UNFLATTEN_ENGINES

# DEC-050: a complete model bundle embeds the Qwen backbone under
# backbone/ and needs no HF cache or network at all. This stays set (and
# the dir created) unconditionally for backward compatibility with an
# older bundle that still downloads the backbone from Hugging Face; a
# DEC-050 bundle simply never touches it.
: "${HF_HOME:=$UNFLATTEN_DATA_DIR/hf-cache}"
export HF_HOME
mkdir -p "$HF_HOME"
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
[ "${UNFLATTEN_HF_HUB_OFFLINE:-0}" = "1" ] && export HF_HUB_OFFLINE=1
# torch itself reads this at import time and crashes on int("") -- an env_file
# with the variable present but empty (the normal, unset-by-default case in
# .env.example) must produce an UNSET variable, not an exported empty string.
if [ "${TORCH_DISABLE_NATIVE_JIT:-}" = "1" ]; then
  export TORCH_DISABLE_NATIVE_JIT
else
  unset TORCH_DISABLE_NATIVE_JIT
fi

CMD=(
  python3 -m unflatten.service.worker_cli
  --job-db "$UNFLATTEN_DATA_DIR/jobs.sqlite3"
  --artifact-root "$UNFLATTEN_DATA_DIR/artifacts"
  --work-root "$UNFLATTEN_DATA_DIR/work"
  --sandbox-image "$UNFLATTEN_SANDBOX_IMAGE"
  --sandbox-runtime "$UNFLATTEN_SANDBOX_RUNTIME"
  --model-id "$UNFLATTEN_MODEL_ID"
  --revision "$UNFLATTEN_MODEL_REVISION"
  --attention-implementation "$UNFLATTEN_ATTENTION_IMPLEMENTATION"
  --device "$UNFLATTEN_DEVICE"
  --worker-id "$UNFLATTEN_WORKER_ID"
  --idle-seconds "$UNFLATTEN_IDLE_SECONDS"
  --engines "$UNFLATTEN_ENGINES"
)
[ -n "${UNFLATTEN_MODEL_BUNDLE:-}" ] && CMD+=(--model-bundle "$UNFLATTEN_MODEL_BUNDLE")
[ -n "${UNFLATTEN_MODEL_BUNDLE_SHA256:-}" ] && CMD+=(--model-bundle-sha256 "$UNFLATTEN_MODEL_BUNDLE_SHA256")
[ -n "${UNFLATTEN_POLICY:-}" ] && CMD+=(--policy "$UNFLATTEN_POLICY")
[ "${UNFLATTEN_SANDBOX_ALLOW_ROOTFUL:-0}" = "1" ] && CMD+=(--sandbox-allow-rootful)
[ -n "${UNFLATTEN_PRECISION:-}" ] && CMD+=(--precision "$UNFLATTEN_PRECISION")
[ -n "${UNFLATTEN_KV_CACHE:-}" ] && CMD+=(--kv-cache "$UNFLATTEN_KV_CACHE")
[ -n "${UNFLATTEN_MODELS_CONFIG:-}" ] && CMD+=(--models-config "$UNFLATTEN_MODELS_CONFIG")
[ -n "${UNFLATTEN_MODEL_VARIANT:-}" ] && CMD+=(--variant "$UNFLATTEN_MODEL_VARIANT")

echo "entrypoint-worker: exec ${CMD[*]}"
exec "${CMD[@]}"

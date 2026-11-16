#!/usr/bin/env bash
# Portable launcher for the private diagnostic API + single-GPU worker
# (spec-pdf-to-word-service T9, design D9). Everything site-specific comes
# from the environment -- no path is hardcoded here. Copy
# deploy/service/service.env.example, fill it in, and either `source` it
# before calling this script or point UNFLATTEN_SERVICE_ENV at your copy.
#
# Usage: deploy/service/service.sh {start|stop|status|verify}
#
# start:  runs api_cli and worker_cli each under `setsid`; pid files and logs
#         live under $UNFLATTEN_DATA_DIR/run and $UNFLATTEN_DATA_DIR/logs.
# stop:   SIGTERM to each process group, waits on `kill -0` against the
#         pid file's own known pid (never a process-name scan, which can
#         match its own watcher loop) up to UNFLATTEN_STOP_TIMEOUT_SECONDS
#         (default 150s), then SIGKILL.
# status: pid liveness for both child processes, plus a readiness curl.
# verify: build_model_bundle.py verify + complete sandbox readiness + env
#         completeness. Never starts a process.
#
# No restart supervision: whoever runs this script decides whether to wrap it
# in systemd (deploy/service/*.service are an optional template, not required).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Optional convenience: source an env file if one is named and exists.
# Required variables may equally be exported by the caller directly.
if [ -n "${UNFLATTEN_SERVICE_ENV:-}" ] && [ -f "${UNFLATTEN_SERVICE_ENV}" ]; then
  # shellcheck disable=SC1090
  . "${UNFLATTEN_SERVICE_ENV}"
fi

_REQUIRED_VARS=(
  UNFLATTEN_DATA_DIR
  UNFLATTEN_MODEL_BUNDLE
  UNFLATTEN_PRINCIPAL_FILE
  UNFLATTEN_SANDBOX_IMAGE
  UNFLATTEN_SANDBOX_RUNTIME
  UNFLATTEN_PORT
  UNFLATTEN_DEVICE
  UNFLATTEN_MODEL_ID
  UNFLATTEN_MODEL_REVISION
)

require_env() {
  local missing=()
  local name value
  for name in "${_REQUIRED_VARS[@]}"; do
    value="${!name:-}"
    [ -n "$value" ] || missing+=("$name")
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    local name
    for name in "${missing[@]}"; do
      echo "service.sh: missing required environment variable: $name" >&2
    done
    return 1
  fi
  case "$UNFLATTEN_SANDBOX_RUNTIME" in
    docker | podman) ;;
    *)
      echo "service.sh: UNFLATTEN_SANDBOX_RUNTIME must be docker or podman, got: $UNFLATTEN_SANDBOX_RUNTIME" >&2
      return 1
      ;;
  esac
  # Owner 2026-09-29 (DEC-052 addendum): password-only first-run setup is
  # safe only while nobody but this machine can reach the API.
  if [ "${UNFLATTEN_SETUP_TOKENLESS:-0}" = "1" ]; then
    case "${UNFLATTEN_HOST:-127.0.0.1}" in
      127.* | ::1) ;;
      *)
        echo "service.sh: UNFLATTEN_SETUP_TOKENLESS=1 requires a loopback UNFLATTEN_HOST, got: $UNFLATTEN_HOST" >&2
        return 1
        ;;
    esac
  fi
  return 0
}

apply_defaults() {
  : "${UNFLATTEN_HOST:=127.0.0.1}"
  : "${UNFLATTEN_MODEL_BUNDLE_SHA256:=}"
  : "${UNFLATTEN_SANDBOX_ALLOW_ROOTFUL:=0}"
  : "${UNFLATTEN_POLICY:=}"
  : "${UNFLATTEN_PYTHON:=uv run --no-sync python}"
  : "${UNFLATTEN_STOP_TIMEOUT_SECONDS:=150}"
  : "${UNFLATTEN_IDLE_SECONDS:=1.0}"
  : "${UNFLATTEN_WORKER_ID:=worker_gpu_0}"
  : "${UNFLATTEN_ATTENTION_IMPLEMENTATION:=sdpa}"
  : "${UNFLATTEN_PRECISION:=bf16}"
  : "${UNFLATTEN_KV_CACHE:=bf16}"
  : "${UNFLATTEN_SETUP_TOKENLESS:=0}"

  RUN_DIR="$UNFLATTEN_DATA_DIR/run"
  LOG_DIR="$UNFLATTEN_DATA_DIR/logs"
  ARTIFACT_ROOT="$UNFLATTEN_DATA_DIR/artifacts"
  WORK_ROOT="$UNFLATTEN_DATA_DIR/work"
  JOB_DB="$UNFLATTEN_DATA_DIR/jobs.sqlite3"

  # Environment files commonly contain ordinary shell assignments rather than
  # `export` statements.  The API, worker, and sandbox readiness subprocess
  # must see this service configuration too.  Keep an intentionally empty
  # manifest pin exported: consumers normalize it to "no pin".
  export "${_REQUIRED_VARS[@]}"
  export UNFLATTEN_MODEL_BUNDLE_SHA256
  export UNFLATTEN_SANDBOX_ALLOW_ROOTFUL
  export UNFLATTEN_POLICY
  export UNFLATTEN_SETUP_TOKENLESS
}

setup_dirs() {
  local dir
  for dir in "$UNFLATTEN_DATA_DIR" "$ARTIFACT_ROOT" "$WORK_ROOT" "$RUN_DIR" "$LOG_DIR"; do
    mkdir -p "$dir"
    chmod 0700 "$dir"
  done
}

# shellcheck disable=SC2206
_python_command() {
  PY_CMD=($UNFLATTEN_PYTHON)
}

_api_command() {
  _python_command
  API_CMD=(
    "${PY_CMD[@]}" -m unflatten.service.api_cli
    --principal-file "$UNFLATTEN_PRINCIPAL_FILE"
    --job-store "$JOB_DB"
    --artifact-root "$ARTIFACT_ROOT"
    --model-bundle "$UNFLATTEN_MODEL_BUNDLE"
    --host "$UNFLATTEN_HOST"
    --port "$UNFLATTEN_PORT"
  )
  [ -n "$UNFLATTEN_POLICY" ] && API_CMD+=(--policy "$UNFLATTEN_POLICY")
  [ -n "$UNFLATTEN_MODEL_BUNDLE_SHA256" ] && API_CMD+=(--model-bundle-sha256 "$UNFLATTEN_MODEL_BUNDLE_SHA256")
  [ -n "${UNFLATTEN_MODELS_CONFIG:-}" ] && API_CMD+=(--models-config "$UNFLATTEN_MODELS_CONFIG")
}

_worker_command() {
  _python_command
  WORKER_CMD=(
    "${PY_CMD[@]}" -m unflatten.service.worker_cli
    --job-db "$JOB_DB"
    --artifact-root "$ARTIFACT_ROOT"
    --work-root "$WORK_ROOT"
    --sandbox-image "$UNFLATTEN_SANDBOX_IMAGE"
    --sandbox-runtime "$UNFLATTEN_SANDBOX_RUNTIME"
    --model-id "$UNFLATTEN_MODEL_ID"
    --revision "$UNFLATTEN_MODEL_REVISION"
    --attention-implementation "$UNFLATTEN_ATTENTION_IMPLEMENTATION"
    --device "$UNFLATTEN_DEVICE"
    --worker-id "$UNFLATTEN_WORKER_ID"
    --idle-seconds "$UNFLATTEN_IDLE_SECONDS"
    --precision "$UNFLATTEN_PRECISION"
    --kv-cache "$UNFLATTEN_KV_CACHE"
  )
  [ -n "$UNFLATTEN_POLICY" ] && WORKER_CMD+=(--policy "$UNFLATTEN_POLICY")
  [ -n "$UNFLATTEN_MODEL_BUNDLE_SHA256" ] && WORKER_CMD+=(--model-bundle-sha256 "$UNFLATTEN_MODEL_BUNDLE_SHA256")
  [ "$UNFLATTEN_SANDBOX_ALLOW_ROOTFUL" = "1" ] && WORKER_CMD+=(--sandbox-allow-rootful)
  # worker_cli takes exactly one of --model-bundle / --models-config; a
  # models config selects the bundle and its precision per variant.
  if [ -n "${UNFLATTEN_MODELS_CONFIG:-}" ]; then
    WORKER_CMD+=(--models-config "$UNFLATTEN_MODELS_CONFIG")
  else
    WORKER_CMD+=(--model-bundle "$UNFLATTEN_MODEL_BUNDLE")
  fi
}

_is_running() {
  local pidfile="$1"
  [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile" 2>/dev/null)" 2>/dev/null
}

start_process() {
  local name="$1"
  shift
  local -a cmd=("$@")
  local pidfile="$RUN_DIR/$name.pid"
  local logfile="$LOG_DIR/$name.log"

  if _is_running "$pidfile"; then
    echo "$name: already running (pid $(cat "$pidfile"))"
    return 0
  fi
  rm -f "$pidfile"

  (
    cd "$REPO_ROOT" || exit 1
    exec setsid "${cmd[@]}"
  ) >>"$logfile" 2>&1 &
  local pid=$!
  echo "$pid" >"$pidfile"
  # Catch an immediate CLI refusal (for example an unready sandbox) rather
  # than recording a dead child as a running service.
  sleep 0.1
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pidfile"
    echo "$name: failed during startup; see $logfile" >&2
    return 1
  fi
  echo "$name: started (pid $pid), log $logfile"
}

stop_process() {
  local name="$1"
  local pidfile="$RUN_DIR/$name.pid"

  if [ ! -f "$pidfile" ]; then
    echo "$name: not running (no pid file)"
    return 0
  fi
  local pid
  pid="$(cat "$pidfile" 2>/dev/null)"
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    echo "$name: not running (stale pid file)"
    rm -f "$pidfile"
    return 0
  fi

  echo "$name: stopping (pid $pid)"
  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null

  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$UNFLATTEN_STOP_TIMEOUT_SECONDS" ]; then
      echo "$name: did not stop within ${UNFLATTEN_STOP_TIMEOUT_SECONDS}s, sending SIGKILL"
      kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done
  rm -f "$pidfile"
  echo "$name: stopped"
}

status_process() {
  local name="$1"
  local pidfile="$RUN_DIR/$name.pid"
  if _is_running "$pidfile"; then
    echo "$name: running (pid $(cat "$pidfile"))"
    return 0
  else
    echo "$name: stopped"
    return 1
  fi
}

do_start() {
  require_env || return 1
  apply_defaults
  setup_dirs

  export HF_HUB_OFFLINE=1
  export PYTHONUNBUFFERED=1
  # Only exported when the operator opted in (H200 host quirk: no C compiler
  # -- memory h200-host-quirks); left unset otherwise.
  if [ -n "${TORCH_DISABLE_NATIVE_JIT:-}" ]; then
    export TORCH_DISABLE_NATIVE_JIT=1
  fi
  [ -n "${HF_HOME:-}" ] && export HF_HOME

  _api_command
  _worker_command
  if ! start_process api "${API_CMD[@]}"; then
    return 1
  fi
  if ! start_process worker "${WORKER_CMD[@]}"; then
    stop_process api
    return 1
  fi
}

do_stop() {
  require_env || return 1
  apply_defaults
  stop_process worker
  stop_process api
}

do_status() {
  require_env || return 1
  apply_defaults
  local failed=0
  status_process api || failed=1
  status_process worker || failed=1
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS -m 3 "http://$UNFLATTEN_HOST:$UNFLATTEN_PORT/health/ready" >/dev/null 2>&1; then
      echo "readiness: ok"
    else
      echo "readiness: unavailable"
      failed=1
    fi
  else
    echo "readiness: curl not on PATH"
    failed=1
  fi
  return "$failed"
}

do_verify() {
  require_env || return 1
  apply_defaults
  [ -n "${HF_HOME:-}" ] && export HF_HOME
  local failed=0

  _python_command
  local -a verify_cmd=("${PY_CMD[@]}" scripts/build_model_bundle.py verify "$UNFLATTEN_MODEL_BUNDLE")
  [ -n "$UNFLATTEN_MODEL_BUNDLE_SHA256" ] && verify_cmd+=(--expected-sha256 "$UNFLATTEN_MODEL_BUNDLE_SHA256")
  if ! (cd "$REPO_ROOT" && "${verify_cmd[@]}"); then
    echo "service.sh verify: model bundle verification failed" >&2
    failed=1
  fi

  local -a sandbox_verify_cmd=(
    "${PY_CMD[@]}" -c
    'from unflatten.service.pdf_sandbox import PdfSandbox; from unflatten.service.policy import load_runtime_policy; import os; path = os.environ.get("UNFLATTEN_POLICY"); policy = load_runtime_policy(path) if path else load_runtime_policy(); PdfSandbox(policy=policy, image=os.environ["UNFLATTEN_SANDBOX_IMAGE"], runtime=os.environ["UNFLATTEN_SANDBOX_RUNTIME"], allow_rootful=os.environ.get("UNFLATTEN_SANDBOX_ALLOW_ROOTFUL") == "1").require_ready()'
  )
  if ! (cd "$REPO_ROOT" && "${sandbox_verify_cmd[@]}"); then
    echo "service.sh verify: sandbox readiness gate failed" >&2
    failed=1
  fi

  if [ "$failed" -eq 0 ]; then
    echo "verify: ok"
    return 0
  fi
  return 1
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

#!/usr/bin/env bash
# Runs `unflatten.service.api_cli` in the `api` compose container.
#
# THE ONE PLACE that maps env vars to api_cli flags.
set -euo pipefail

: "${UNFLATTEN_DATA_DIR:=/data}"
: "${UNFLATTEN_MODEL_BUNDLE:=}"
: "${UNFLATTEN_BOOTSTRAP_MODEL_MANIFEST:=}"
if [[ -z "$UNFLATTEN_MODEL_BUNDLE" && -z "$UNFLATTEN_BOOTSTRAP_MODEL_MANIFEST" ]] || [[ -n "$UNFLATTEN_MODEL_BUNDLE" && -n "$UNFLATTEN_BOOTSTRAP_MODEL_MANIFEST" ]]; then
  echo 'Exactly one of UNFLATTEN_MODEL_BUNDLE or UNFLATTEN_BOOTSTRAP_MODEL_MANIFEST is required' >&2
  exit 2
fi
: "${UNFLATTEN_PRINCIPAL_FILE:=$UNFLATTEN_DATA_DIR/principals.json}"
: "${UNFLATTEN_HOST:=0.0.0.0}"
: "${UNFLATTEN_PORT:=8000}"

mkdir -p "$UNFLATTEN_DATA_DIR/artifacts"

CMD=(
  python3 -m unflatten.service.api_cli
  --principal-file "$UNFLATTEN_PRINCIPAL_FILE"
  --job-store "$UNFLATTEN_DATA_DIR/jobs.sqlite3"
  --artifact-root "$UNFLATTEN_DATA_DIR/artifacts"
  --work-root "${UNFLATTEN_WORK_ROOT:-$UNFLATTEN_DATA_DIR/work}"
  --host "$UNFLATTEN_HOST"
  --port "$UNFLATTEN_PORT"
)
[ -n "$UNFLATTEN_MODEL_BUNDLE" ] && CMD+=(--model-bundle "$UNFLATTEN_MODEL_BUNDLE")
[ -n "$UNFLATTEN_BOOTSTRAP_MODEL_MANIFEST" ] && CMD+=(--bootstrap-model-manifest "$UNFLATTEN_BOOTSTRAP_MODEL_MANIFEST")
[ -n "${UNFLATTEN_MODEL_BUNDLE_SHA256:-}" ] && CMD+=(--model-bundle-sha256 "$UNFLATTEN_MODEL_BUNDLE_SHA256")
[ -n "${UNFLATTEN_POLICY:-}" ] && CMD+=(--policy "$UNFLATTEN_POLICY")
[ -n "${UNFLATTEN_MODELS_CONFIG:-}" ] && CMD+=(--models-config "$UNFLATTEN_MODELS_CONFIG")
# design §6.1/§8.4: web UI mode, host allowlist, session lifetimes, version.
[ -n "${UNFLATTEN_WEB_UI:-}" ] && CMD+=(--web-ui "$UNFLATTEN_WEB_UI")
[ -n "${UNFLATTEN_WEB_ALLOWED_HOSTS:-}" ] && CMD+=(--web-allowed-hosts "$UNFLATTEN_WEB_ALLOWED_HOSTS")
[ -n "${UNFLATTEN_WEB_SESSION_IDLE_SECONDS:-}" ] && CMD+=(--web-session-idle-seconds "$UNFLATTEN_WEB_SESSION_IDLE_SECONDS")
[ -n "${UNFLATTEN_WEB_SESSION_MAX_SECONDS:-}" ] && CMD+=(--web-session-max-seconds "$UNFLATTEN_WEB_SESSION_MAX_SECONDS")
[ -n "${UNFLATTEN_WEB_AUTH_FILE:-}" ] && CMD+=(--web-auth-file "$UNFLATTEN_WEB_AUTH_FILE")
[ -n "${UNFLATTEN_GIT_SHA:-}" ] && CMD+=(--git-sha "$UNFLATTEN_GIT_SHA")
# Owner 2026-09-29 (DEC-052 addendum): set by run-local.sh for a loopback bind only.
[ "${UNFLATTEN_SETUP_TOKENLESS:-0}" = "1" ] && CMD+=(--web-setup-tokenless)

echo "entrypoint-api: exec ${CMD[*]}"
exec "${CMD[@]}"

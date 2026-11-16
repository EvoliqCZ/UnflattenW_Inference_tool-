#!/usr/bin/env bash
# `docker compose` wrapper for the private diagnostic service (design §8.5,
# DEC-052 replaces DEC-044 Dodatok D §6's sentence about this file).
#
# This name used to be the NATIVE (non-docker) launcher -- that script is
# now `deploy/local/dev-native.sh`, unchanged in behaviour. `start` is
# refused here on purpose (note 7 below).
#
# Usage: deploy/local/run-local.sh {up|down|status|logs|reset-password|verify} [--dry-run]
#
# `up`, in order (every step below runs for real, including under
# --dry-run -- only the two `docker compose ... build`/`up -d` calls that
# actually start containers are replaced by a printed command):
#   1. `docker info` -- daemon reachable, else a precise message (RUNBOOK §10);
#   2. deploy/docker/.env missing -> copied from .env.example, filling
#      UNFLATTEN_HOST_DATA_DIR and UNFLATTEN_GIT_SHA;
#   3. UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1 required (the one consent flag --
#      mounting the host docker socket into the worker is root-equivalent,
#      compose.sandbox-socket.yaml); a bare UNFLATTEN_SANDBOX_ALLOW_ROOTFUL=1
#      without it is refused too (two flags for one consent would confuse);
#   (between 3 and 4) decide the first-run setup mode from
#      UNFLATTEN_BIND_ADDRESS (`resolve_setup_mode`, owner 2026-09-29):
#      loopback -> password-only setup, anything else -> token link;
#   4. models.json must exist and validate (see deploy/local/dev-bundle.sh);
#   5. `docker compose ... build` and `up -d`;
#   6. print the URL, or for a non-loopback bind the one-click setup link
#      `http://<host>:<port>/setup#token=<t>` read from `docker compose logs init`.
#
# `start` disappeared in T47 -- the owner used it daily, so this prints a
# hint instead of silently doing nothing (plan note 7).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKER_DIR="$REPO_ROOT/deploy/docker"
COMPOSE_FILE="$DOCKER_DIR/compose.yaml"
SOCKET_OVERRIDE="$DOCKER_DIR/compose.sandbox-socket.yaml"

: "${UNFLATTEN_ENV_FILE:=$DOCKER_DIR/.env}"
: "${UNFLATTEN_MODELS_CONFIG_FILE:=$DOCKER_DIR/models.json}"
# Runs on the HOST, not in a container -- needs `unflatten` importable (e.g.
# `uv run --no-sync python3`, or an activated `uv sync --group server` venv).
# Plain `python3` by default so a stub on PATH is enough for tests.
: "${UNFLATTEN_PYTHON:=python3}"

DRY_RUN=0

log() { echo "run-local.sh: $*"; }
die() {
  echo "run-local.sh: $*" >&2
  exit "${2:-1}"
}

# --- helpers -----------------------------------------------------------

_compose() {
  (cd "$DOCKER_DIR" && docker compose --env-file "$UNFLATTEN_ENV_FILE" \
    -f "$COMPOSE_FILE" -f "$SOCKET_OVERRIDE" "$@")
}

_env_value() {
  local key="$1"
  [ -f "$UNFLATTEN_ENV_FILE" ] || return 1
  grep -E "^${key}=" "$UNFLATTEN_ENV_FILE" | tail -n1 | cut -d= -f2-
}

step_docker_info() {
  log "[1/6] checking docker daemon"
  command -v docker >/dev/null 2>&1 || die "docker CLI not found on PATH (RUNBOOK §10)"
  docker info >/dev/null 2>&1 \
    || die "docker daemon not reachable (is it running? do you have permission? RUNBOOK §10)"
}

step_ensure_env_file() {
  log "[2/6] checking $UNFLATTEN_ENV_FILE"
  if [ -f "$UNFLATTEN_ENV_FILE" ]; then
    log "reusing existing $UNFLATTEN_ENV_FILE"
    return 0
  fi
  [ -f "$DOCKER_DIR/.env.example" ] || die "$DOCKER_DIR/.env.example is missing -- cannot seed a new .env"
  cp "$DOCKER_DIR/.env.example" "$UNFLATTEN_ENV_FILE"
  local host_data_dir git_sha
  host_data_dir="$HOME/unflatten-diagnostic-data"
  git_sha="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  mkdir -p "$host_data_dir"
  {
    echo "UNFLATTEN_HOST_DATA_DIR=$host_data_dir"
    echo "UNFLATTEN_GIT_SHA=$git_sha"
  } >>"$UNFLATTEN_ENV_FILE"
  log "created $UNFLATTEN_ENV_FILE (UNFLATTEN_HOST_DATA_DIR=$host_data_dir, UNFLATTEN_GIT_SHA=$git_sha) -- fill in the REPLACE_WITH_* values before continuing"
}

step_check_rootful_consent() {
  log "[3/6] checking rootful socket consent"
  local accept allow
  accept="$(_env_value UNFLATTEN_ACCEPT_ROOTFUL_SOCKET || true)"
  allow="$(_env_value UNFLATTEN_SANDBOX_ALLOW_ROOTFUL || true)"
  if [ "$allow" = "1" ] && [ "$accept" != "1" ]; then
    die "UNFLATTEN_SANDBOX_ALLOW_ROOTFUL=1 in $UNFLATTEN_ENV_FILE without UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1 -- \
set only UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1 (never both, deploy/docker/.env.example)" 2
  fi
  if [ "$accept" != "1" ]; then
    die "refusing to start: mounting the host docker socket into the worker is root-equivalent \
control of this machine (see the warning in $SOCKET_OVERRIDE). Set UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1 \
in $UNFLATTEN_ENV_FILE to accept this on your own single-user workstation." 2
  fi
}

# Owner 2026-09-29 (DEC-052 addendum): the first-run setup wizard asks for
# the bootstrap token only when a stranger could reach the port. Trust comes
# from THIS deployment config, never from the client IP (inside the
# container every client is the docker bridge gateway):
#   - UNFLATTEN_BIND_ADDRESS (compose's published host IP, default
#     127.0.0.1) loopback -> tokenless unless .env says
#     UNFLATTEN_SETUP_TOKENLESS=0 (SSH tunnel / reverse proxy in front);
#   - anything else -> tokenless off; UNFLATTEN_SETUP_TOKENLESS=1 is refused.
# The decision is exported, so compose interpolates it over whatever .env
# says (shell env beats --env-file).
resolve_setup_mode() {
  local requested
  BIND_ADDRESS="$(_env_value UNFLATTEN_BIND_ADDRESS || true)"
  BIND_ADDRESS="${BIND_ADDRESS:-127.0.0.1}"
  requested="$(_env_value UNFLATTEN_SETUP_TOKENLESS || true)"
  case "$requested" in
    ""|0|1) ;;
    *) die "UNFLATTEN_SETUP_TOKENLESS in $UNFLATTEN_ENV_FILE must be 0, 1 or empty (got '$requested')" 2 ;;
  esac
  case "$BIND_ADDRESS" in
    127.*|::1|"[::1]")
      if [ "$requested" = "0" ]; then SETUP_TOKENLESS=0; else SETUP_TOKENLESS=1; fi
      ;;
    *)
      if [ "$requested" = "1" ]; then
        die "refusing to start: UNFLATTEN_SETUP_TOKENLESS=1 with UNFLATTEN_BIND_ADDRESS=$BIND_ADDRESS -- \
anyone who reaches that address could claim the admin password first. Remove UNFLATTEN_SETUP_TOKENLESS \
(the setup link printed after start carries the token)." 2
      fi
      SETUP_TOKENLESS=0
      ;;
  esac
  export UNFLATTEN_SETUP_TOKENLESS="$SETUP_TOKENLESS"
  if [ "$SETUP_TOKENLESS" = "1" ]; then
    log "first-run setup: password only (port published on $BIND_ADDRESS)"
  else
    log "first-run setup: via the one-click setup link printed after start (port published on $BIND_ADDRESS)"
  fi
}

# The host a browser uses for the published port: the bind address itself,
# or this machine's first address for a wildcard bind.
_browser_host() {
  case "$BIND_ADDRESS" in
    0.0.0.0|::|"[::]")
      hostname -I 2>/dev/null | awk '{print $1}' | grep . || echo "<this-machine-address>"
      ;;
    *) echo "$BIND_ADDRESS" ;;
  esac
}

print_setup_hint() {
  local port host token
  port="$(_env_value UNFLATTEN_PORT || true)"
  port="${port:-8420}"
  if [ "$SETUP_TOKENLESS" = "1" ]; then
    log "service should be reachable at http://127.0.0.1:${port}/ -- first run: just choose a password"
    log "API bearer token for scripts (first run only): docker compose logs init"
    return 0
  fi
  host="$(_browser_host)"
  # The token rides in the URL #fragment: a browser never sends a fragment to
  # a server, so it never reaches any access log; the web UI strips it from
  # the address bar right after reading it.
  token="$(_compose logs --no-log-prefix init 2>/dev/null \
    | sed -n 's/^init: bearer token[^:]*: //p' | tail -n1)"
  if [ -n "$token" ]; then
    log "first-run setup -- open this link once in your browser:"
    echo "  http://${host}:${port}/setup#token=${token}"
  else
    log "service should be reachable at http://${host}:${port}/ (no fresh init token in 'docker compose logs init' --"
    log "setup is probably already done; forgot the password? run-local.sh reset-password)"
  fi
  case ",$(_env_value UNFLATTEN_WEB_ALLOWED_HOSTS || true)," in
    *",${host},"*) ;;
    *) log "note: add ${host} to UNFLATTEN_WEB_ALLOWED_HOSTS in $UNFLATTEN_ENV_FILE, or the API rejects the browser's Host header" ;;
  esac
}

step_check_models_config() {
  log "[4/6] checking $UNFLATTEN_MODELS_CONFIG_FILE"
  if [ ! -f "$UNFLATTEN_MODELS_CONFIG_FILE" ]; then
    die "models config $UNFLATTEN_MODELS_CONFIG_FILE is missing -- run deploy/local/dev-bundle.sh \
first, or see deploy/local/README.md" 2
  fi
  (cd "$REPO_ROOT" && $UNFLATTEN_PYTHON -c "
from unflatten.service.models_config import load_models_config
load_models_config('$UNFLATTEN_MODELS_CONFIG_FILE')
") || die "models config $UNFLATTEN_MODELS_CONFIG_FILE failed validation -- see deploy/local/README.md" 2
}

step_compose_up() {
  log "[5/6] building images"
  if [ "$DRY_RUN" = "1" ]; then
    echo "run-local.sh: [dry-run] would run: docker compose --env-file $UNFLATTEN_ENV_FILE -f $COMPOSE_FILE -f $SOCKET_OVERRIDE build"
  else
    _compose build || die "docker compose build failed"
  fi
  log "[6/6] starting containers"
  if [ "$DRY_RUN" = "1" ]; then
    echo "run-local.sh: [dry-run] would run: UNFLATTEN_SETUP_TOKENLESS=$UNFLATTEN_SETUP_TOKENLESS docker compose --env-file $UNFLATTEN_ENV_FILE -f $COMPOSE_FILE -f $SOCKET_OVERRIDE up -d"
    return 0
  fi
  _compose up -d || die "docker compose up failed"
}

do_up() {
  step_docker_info
  step_ensure_env_file
  step_check_rootful_consent
  resolve_setup_mode
  step_check_models_config
  step_compose_up
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  print_setup_hint
}

do_down() {
  # Never `-v`: a data volume/bind mount is never discarded by this wrapper
  # (memory: never delete ~/unflatten-service-local or run `down -v`).
  if [ "$DRY_RUN" = "1" ]; then
    echo "run-local.sh: [dry-run] would run: docker compose --env-file $UNFLATTEN_ENV_FILE -f $COMPOSE_FILE -f $SOCKET_OVERRIDE down"
    return 0
  fi
  _compose down
}

do_status() {
  if [ "$DRY_RUN" = "1" ]; then
    echo "run-local.sh: [dry-run] would run: docker compose ... ps"
    return 0
  fi
  _compose ps
  local port
  port="$(_env_value UNFLATTEN_PORT || true)"
  curl -sS "http://127.0.0.1:${port:-8420}/health/ready" || true
  echo
}

do_logs() {
  if [ "$DRY_RUN" = "1" ]; then
    echo "run-local.sh: [dry-run] would run: docker compose ... logs -f"
    return 0
  fi
  _compose logs -f
}

do_reset_password() {
  # entrypoint-init.sh/compose write the principal file at
  # /data/principals.json (UNFLATTEN_PRINCIPAL_FILE), not web_auth_cli's own
  # default of <data-dir>/principal.json -- pass it explicitly, or a fresh
  # install exits 2.
  if [ "$DRY_RUN" = "1" ]; then
    echo "run-local.sh: [dry-run] would run: docker compose ... exec api python3 -m unflatten.service.web_auth_cli reset-password --data-dir /data --principal-id local --principal-file /data/principals.json"
    return 0
  fi
  _compose exec api python3 -m unflatten.service.web_auth_cli reset-password \
    --data-dir /data --principal-id local --principal-file /data/principals.json
}

do_verify() {
  log "validating compose config"
  if [ "$DRY_RUN" = "1" ]; then
    echo "run-local.sh: [dry-run] would run: docker compose ... config -q"
    echo "run-local.sh: [dry-run] would run: docker image inspect unflatten/pdf-sandbox:local"
    return 0
  fi
  _compose config -q || die "docker compose config is invalid"
  local expected actual
  expected="$(cd "$REPO_ROOT" && $UNFLATTEN_PYTHON -c 'from unflatten.service.pdf_sandbox import SCANNER_IDENTITY; print(SCANNER_IDENTITY)')" \
    || die "cannot read the expected sandbox scanner identity"
  actual="$(docker image inspect --format '{{index .Config.Labels "org.unflatten.pdf-scanner.identity"}}' unflatten/pdf-sandbox:local 2>/dev/null || true)"
  [ "$actual" = "$expected" ] || die "sandbox image unflatten/pdf-sandbox:local scanner identity mismatch (found '${actual:-none}', need '$expected')"
  log "compose config valid, sandbox image scanner identity matches"
}

do_start_hint() {
  echo "run-local.sh: 'start' no longer exists -- this wrapper now runs docker compose." >&2
  echo "run-local.sh: for the native (non-docker) launcher you used before, run:" >&2
  echo "  deploy/local/dev-native.sh start" >&2
  echo "run-local.sh: for the docker-compose deployment, run:" >&2
  echo "  deploy/local/run-local.sh up" >&2
  return 2
}

usage() {
  echo "usage: $0 {up|down|status|logs|reset-password|verify} [--dry-run]" >&2
}

main() {
  local action="${1:-}"
  shift || true
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) DRY_RUN=1; shift ;;
      *) die "unknown argument: $1" ;;
    esac
  done
  case "$action" in
    up) do_up ;;
    down) do_down ;;
    status) do_status ;;
    logs) do_logs ;;
    reset-password) do_reset_password ;;
    verify) do_verify ;;
    start) do_start_hint ;;
    *)
      usage
      return 2
      ;;
  esac
}

main "$@"

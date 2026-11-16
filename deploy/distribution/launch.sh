#!/usr/bin/env bash
set -euo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/messages.sh"
validate_path() {
    case "$1" in
        *$'\n'*|*$'\r'*|*$'\t'*|*'"'*|*'\'*|*'$'*|*'%'*|*':'*|*','*)
            printf 'Unflatten W: %s\n' "$(message unsupported_path)" >&2
            exit 1 ;;
    esac
}
validate_path "${BASH_SOURCE[0]}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
DATA_INPUT="${UNFLATTEN_APP_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/unflatten-w-app-1.0}"
validate_path "$DATA_INPUT"
DATA="$(realpath -m -- "$DATA_INPUT")"
validate_path "$ROOT"
validate_path "$DATA"
CONTROL="${DATA}.launcher"
UNFLATTEN_REQUESTED_PORT="${UNFLATTEN_APP_PORT:-}"
PORT="${UNFLATTEN_REQUESTED_PORT:-8421}"
COMPOSE="$ROOT/deploy/distribution/compose.yaml"
die() { printf 'Unflatten W: %s\n' "$*" >&2; exit 1; }
[[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT > 1023 && PORT < 65536)) || die "$(message invalid_port)"
# Compose must use this launcher's saved paths, identities and port, even
# when unrelated deployment variables were inherited from the shell.
unset UNFLATTEN_HOST_DATA_DIR UNFLATTEN_APP_UID UNFLATTEN_APP_GID
unset UNFLATTEN_DOCKER_GID UNFLATTEN_APP_PORT UNFLATTEN_APP_HF_TOKEN_FILE
unset UNFLATTEN_APP_SOURCE_ROOT UNFLATTEN_APP_SANDBOX_IMAGE

read_saved_port() {
    local key value
    while IFS='=' read -r key value; do
        if [[ "$key" == UNFLATTEN_APP_PORT ]]; then
            [[ "$value" =~ ^[0-9]+$ ]] && ((value > 1023 && value < 65536)) || die "$(message invalid_saved_port)"
            PORT="$value"
            return
        fi
    done < "$CONTROL/app.env"
    die "$(message missing_port)"
}
[[ ! -f "$CONTROL/app.env" || -n "${UNFLATTEN_REQUESTED_PORT:-}" ]] || read_saved_port

configure() {
    mkdir -p "$CONTROL/secrets" "$DATA/artifacts" "$DATA/work" "$DATA/hf-cache" "$DATA/home"
    chmod 700 "$DATA" "$CONTROL" "$CONTROL/secrets"
    local token="${HF_TOKEN:-}" cached
    if [[ -z "$token" && ! -s "$CONTROL/secrets/hf-token" ]]; then
        cached="${HF_TOKEN_PATH:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/token}"
        if [[ -s "$cached" ]]; then token="$(cat -- "$cached")"; fi
    fi
    if [[ -n "$token" ]]; then
        [[ "$token" != *$'\n'* && "$token" != *$'\r'* && "$token" == hf_* ]] || die "$(message invalid_hf_token)"
        printf '%s' "$token" > "$CONTROL/secrets/hf-token"
    elif [[ ! -e "$CONTROL/secrets/hf-token" ]]; then
        # Public models download anonymously when Hugging Face reads an empty
        # token file. Keep the same worker-only secret mount in both modes.
        : > "$CONTROL/secrets/hf-token"
    fi
    chmod 600 "$CONTROL/secrets/hf-token"
    unset token
    local docker_gid
    docker_gid="$(stat -c %g /var/run/docker.sock 2>/dev/null || printf '0')"
    cat > "$CONTROL/app.env" <<EOF
UNFLATTEN_HOST_DATA_DIR="$DATA"
UNFLATTEN_APP_UID=$(id -u)
UNFLATTEN_APP_GID=$(id -g)
UNFLATTEN_DOCKER_GID=$docker_gid
UNFLATTEN_APP_PORT=$PORT
UNFLATTEN_APP_HF_TOKEN_FILE="$CONTROL/secrets/hf-token"
UNFLATTEN_APP_SOURCE_ROOT="$ROOT"
EOF
    chmod 600 "$CONTROL/app.env"
    printf 'Unflatten W 1.0 · EVOLIQ s.r.o.\n%s%s\n' "$(message local_data)" "$DATA"
}

compose() { docker compose -p unflatten-w-app-1-0 --env-file "$CONTROL/app.env" -f "$COMPOSE" "$@"; }

consent() {
    [[ -f "$CONTROL/docker-consent" ]] && return
    if [[ "${UNFLATTEN_ACCEPT_ROOTFUL_SOCKET:-0}" != 1 ]]; then
        if [[ "${UNFLATTEN_APP_NONINTERACTIVE:-0}" == 1 || ! -t 0 ]]; then
            die "$(message docker_consent_required)"
        fi
        printf '%s' "$(message docker_consent_prompt)" >&2
        local choice
        read -r choice
        [[ "$choice" == ano || "$choice" == áno || "$choice" == yes ]] || die "$(message docker_consent_denied)"
    fi
    mkdir -p "$CONTROL"
    chmod 700 "$CONTROL"
    printf 'docker-sandbox-access-accepted-v1\n' > "$CONTROL/docker-consent"
    chmod 600 "$CONTROL/docker-consent"
}

open_app() {
    printf '\nUnflatten W 1.0: http://127.0.0.1:%s/\n%s\n' "$PORT" "$(message gpu_prepare)"
    if [[ "${UNFLATTEN_APP_OPEN_BROWSER:-1}" == 1 ]] && command -v xdg-open >/dev/null; then
        xdg-open "http://127.0.0.1:$PORT/" >/dev/null 2>&1 || true
    fi
}

start() {
    local mode="${1:-start}"
    command -v docker >/dev/null || die "$(message install_docker)"
    docker info >/dev/null 2>&1 || die "$(message docker_unavailable)"
    docker compose version >/dev/null 2>&1 || die "$(message missing_compose)"
    if [[ "$mode" == start && -f "$CONTROL/app.env" ]] && \
       has_saved_sandbox; then
        read_saved_port
        local running
        running="$(compose ps --services --filter status=running api worker)"
        if [[ $'\n'"$running"$'\n' != *$'\napi\n'* || $'\n'"$running"$'\n' != *$'\nworker\n'* ]]; then
            # Start missing/stopped services without replacing a live worker.
            compose up -d --no-build --no-recreate
        fi
        open_app
        return
    fi
    consent
    command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 || die "$(message gpu_unavailable)"
    configure
    printf '%s\n' "$(message preparing)"
    compose config --quiet
    compose build
    local sandbox_id
    sandbox_id="$(docker image inspect --format '{{.Id}}' unflatten-w-app-pdf-sandbox:1.0)"
    [[ "$sandbox_id" =~ ^sha256:[0-9a-f]{64}$ ]] || die "$(message sandbox_unverified)"
    printf 'UNFLATTEN_APP_SANDBOX_IMAGE=%s\n' "$sandbox_id" >> "$CONTROL/app.env"
    compose run --rm --no-deps --entrypoint python3 api -c \
      'from unflatten.service.models_config import load_models_config; load_models_config("/config/models.json")'
    compose up -d --force-recreate
    open_app
}

has_saved_sandbox() {
    # Do not source a dotenv file as shell code.
    local line
    while IFS= read -r line; do
        [[ "$line" =~ ^UNFLATTEN_APP_SANDBOX_IMAGE=sha256:[0-9a-f]{64}$ ]] && return 0
    done < "$CONTROL/app.env"
    return 1
}

reset_password() {
    [[ -f "$CONTROL/app.env" ]] || die "$(message not_configured)"
    compose run --rm --no-deps --entrypoint python3 api \
      -m unflatten.service.web_auth_cli reset-password --data-dir "$DATA"
}

case "${1:-start}" in
    configure) configure ;;
    start) start ;;
    restart) start restart ;;
    reset-password) reset_password ;;
    stop) [[ -f "$CONTROL/app.env" ]] || die "$(message not_configured)"; compose down ;;
    status) [[ -f "$CONTROL/app.env" ]] || die "$(message not_configured)"; compose ps ;;
    logs) [[ -f "$CONTROL/app.env" ]] || die "$(message not_configured)"; compose logs -f api worker ;;
    *) die "$(message usage)" ;;
esac

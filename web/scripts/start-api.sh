#!/usr/bin/env bash
# T37 e2e harness: boots the real FastAPI service (`api_cli`, `--web-ui spa`,
# port 8431, `UNFLATTEN_E2E=1`) plus the torch-free `fake_worker.py` (design
# Sec 7.6/Sec 2.5) against an isolated temp data dir -- never
# `~/unflatten-service-local` (memory `production-run-requirements`: never
# reuse a real deployment's state for a test run). `playwright.config.ts`
# runs this as its `webServer` command; it exits when the API process exits
# or this script receives SIGTERM/SIGINT (Playwright's own shutdown).
#
# T41: `playwright.config.ts` now starts *several* instances of this script
# concurrently (one backend per browser project, plus a shared "app" backend
# and a short-idle "session-expiry" backend), each on its own
# `UNFLATTEN_E2E_PORT` -- see that file's own comment for the full list and
# why (plan §I: a single shared backend made `POST /v1/setup` a one-shot
# race, so only one project's `setup-login.spec.ts` run ever saw the real
# wizard). Two things below are therefore guarded against concurrent
# invocations of this same script: the bootstrap-token file is named per
# port (never a shared fixed name another instance could overwrite), and the
# `web/dist` -> `DIST_TARGET` copy is `flock`-guarded (every instance would
# otherwise race a co-running server's `DIST_TARGET` reader).
#
# T-fix (2026-09-27): the copy used to run only when `DIST_TARGET` had no
# `index.html` at all, so a `DIST_TARGET` copied before the latest
# `web/dist` rebuild stayed in place forever -- specs then silently ran
# against UI that no longer existed. `needs_copy` (lib-freshness.sh)
# content-hashes both trees and re-copies whenever they differ, and the
# actual swap builds the new copy in a temp dir first, then replaces
# `DIST_TARGET` with two renames (never an `rm -rf` window where the path is
# briefly missing or half-written).
#
# Fixtures this depends on:
#   - `web/dist/` (built once by the caller: `npm run build`), copied to
#     `src/unflatten/service/ui/dist/` -- the fixed path `create_app` reads
#     for `--web-ui spa` (both paths are `.gitignore`d build artefacts).
#   - `tests/data/e2e/model-bundle/` + `tests/data/e2e/hf-cache/`: a small
#     (~90 KB) synthetic model bundle + offline HF snapshot built once with
#     `tests.service.test_model_bundle._build_bundle`/`_prepare_hf_cache`,
#     verified by `unflatten.service.model_bundle.verify_bundle` -- `api_cli`
#     only ever reads the manifest's `component_lock_hash` from it (D9 §6.1),
#     it never loads a model or imports torch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WEB_DIR="$REPO_ROOT/web"
DIST_TARGET="$REPO_ROOT/src/unflatten/service/ui/dist"
FIXTURE_DIR="$REPO_ROOT/tests/data/e2e"

# shellcheck source=./lib-freshness.sh
source "$SCRIPT_DIR/lib-freshness.sh"

PORT="${UNFLATTEN_E2E_PORT:-8431}"
DATA_DIR="$(mktemp -d -t unflatten-e2e-XXXXXX)"
# T41: per-port, not a fixed shared name -- several instances of this script
# run at once, each spec reads the token for the port it talks to
# (`e2e/fixtures/bootstrap-token.ts`).
BOOTSTRAP_TOKEN_FILE="${TMPDIR:-/tmp}/unflatten-e2e-bootstrap-token-${PORT}"
DIST_COPY_LOCK="${TMPDIR:-/tmp}/unflatten-e2e-dist-copy.lock"

log() { echo "start-api.sh: $*" >&2; }

cleanup() {
  local status=$?
  [ -n "${WORKER_PID:-}" ] && kill "$WORKER_PID" 2>/dev/null || true
  [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null || true
  wait "${WORKER_PID:-}" "${API_PID:-}" 2>/dev/null || true
  rm -rf "$DATA_DIR"
  exit "$status"
}
trap cleanup EXIT INT TERM

if [ ! -f "$WEB_DIR/dist/index.html" ]; then
  log "web/dist missing -- run 'npm run build' in web/ first"
  exit 1
fi
mkdir -p "$(dirname "$DIST_TARGET")"
(
  flock 9
  # Idempotent: every concurrent instance computes the same hashes and would
  # perform the same swap, so a co-running server that already sees a
  # current target is left alone rather than raced.
  if needs_copy "$WEB_DIR/dist" "$DIST_TARGET"; then
    TMP_PARENT="$(dirname "$DIST_TARGET")"
    TMP_TARGET="$(mktemp -d "$TMP_PARENT/.dist-tmp-XXXXXX")"
    cp -r "$WEB_DIR/dist" "$TMP_TARGET/new"
    OLD_TARGET=""
    if [ -e "$DIST_TARGET" ]; then
      OLD_TARGET="$TMP_PARENT/.dist-old-$$"
      mv "$DIST_TARGET" "$OLD_TARGET"
    fi
    mv "$TMP_TARGET/new" "$DIST_TARGET"
    rm -rf "$TMP_TARGET"
    if [ -n "$OLD_TARGET" ]; then
      rm -rf "$OLD_TARGET"
    fi
  fi
) 9>"$DIST_COPY_LOCK"

# One bearer token holding every capability (deploy/local/run-local.sh's own
# pattern) -- also the §2.5 "bootstrap token" the setup wizard's first
# `POST /v1/setup` authenticates with.
BOOTSTRAP_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
(
  umask 0377
  python3 -c "
import json, sys
token, out_path = sys.argv[1], sys.argv[2]
doc = {
    'principal_file_schema_version': '1.0.0',
    'principals': [
        {
            'principal_id': 'e2e-owner',
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
with open(out_path, 'w') as f:
    json.dump(doc, f)
" "$BOOTSTRAP_TOKEN" "$DATA_DIR/principal.json"
)
chmod 0400 "$DATA_DIR/principal.json"
printf '%s' "$BOOTSTRAP_TOKEN" > "$DATA_DIR/bootstrap-token.txt"
# e2e specs read the token from this fixed, per-port path outside the repo --
# they have no other way to learn a per-run temp dir, and `.gitignore` is
# T12's file alone to edit (plan §B), so this stays out of the repo tree
# entirely rather than adding a new ignore rule.
printf '%s' "$BOOTSTRAP_TOKEN" > "$BOOTSTRAP_TOKEN_FILE"

export UNFLATTEN_E2E=1
export HF_HUB_CACHE="$FIXTURE_DIR/hf-cache"

# Owner 2026-09-29 (DEC-052 addendum): `playwright.config.ts` sets
# UNFLATTEN_SETUP_TOKENLESS=1 for the wizard backends (password-only setup,
# the loopback-only deployment shape -- this script always binds 127.0.0.1)
# and 0 for the setup-link backends (`/setup#token=` link). Passed as the
# explicit flag, and exported as 0 otherwise, so api_cli's own env fallback
# can never read a stray value from the caller's shell.
TOKENLESS_ARGS=()
if [ "${UNFLATTEN_SETUP_TOKENLESS:-0}" = "1" ]; then
  TOKENLESS_ARGS=(--web-setup-tokenless)
else
  export UNFLATTEN_SETUP_TOKENLESS=0
fi

cd "$REPO_ROOT"
uv run python3 -m unflatten.service.api_cli \
  --principal-file "$DATA_DIR/principal.json" \
  --job-store "$DATA_DIR/jobs.sqlite3" \
  --artifact-root "$DATA_DIR/artifacts" \
  --model-bundle "$FIXTURE_DIR/model-bundle" \
  --host 127.0.0.1 --port "$PORT" --web-ui spa \
  "${TOKENLESS_ARGS[@]}" \
  >"$DATA_DIR/api.log" 2>&1 &
API_PID=$!

# The worker needs `jobs.sqlite3` already `.initialize()`-d by the API
# process above -- poll for the file instead of a fixed sleep.
for _ in $(seq 1 100); do
  [ -f "$DATA_DIR/jobs.sqlite3" ] && break
  sleep 0.1
done

uv run python3 -m tests.service.fake_worker \
  --job-db "$DATA_DIR/jobs.sqlite3" \
  --artifact-root "$DATA_DIR/artifacts" \
  >"$DATA_DIR/fake_worker.log" 2>&1 &
WORKER_PID=$!

log "api pid=$API_PID worker pid=$WORKER_PID data_dir=$DATA_DIR port=$PORT"
wait "$API_PID"

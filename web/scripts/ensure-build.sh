#!/usr/bin/env bash
# T41: `playwright.config.ts` starts several `start-api.sh` instances
# concurrently (see that script's own comment), and every one of them needs
# `web/dist/` to exist first. Without this guard each instance's own `npm
# run build` would race the others over the same output directory. `flock`
# serializes the check, and the build itself only runs once (the 2nd..Nth
# instance sees `dist/index.html` already there and skips it).
#
# T-fix (2026-09-27): "already there" used to mean *any* dist/index.html,
# even one built before the latest source edit -- a stale build then went
# untested for real (a spec asserting on new UI silently ran against old
# UI). `needs_build` (lib-freshness.sh) rebuilds whenever a tracked source
# input is newer than dist/index.html, not just when dist/ is absent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCK_FILE="${TMPDIR:-/tmp}/unflatten-e2e-build.lock"

# shellcheck source=./lib-freshness.sh
source "$SCRIPT_DIR/lib-freshness.sh"

(
  flock 9
  if needs_build "$WEB_DIR"; then
    (cd "$WEB_DIR" && npm run build)
  fi
) 9>"$LOCK_FILE"

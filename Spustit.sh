#!/usr/bin/env bash
# Unflatten W 1.0 by EVOLIQ s.r.o. — Linux launcher; no host Python required.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$ROOT/deploy/distribution/launch.sh" "${@:-start}"

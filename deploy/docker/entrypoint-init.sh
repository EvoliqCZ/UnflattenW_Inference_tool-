#!/usr/bin/env bash
# One-shot `init` compose service: generates the principal file
# (src/unflatten/service/auth.py) with one random bearer token if it does
# not exist yet, mode 0400, owned by the same user the api/worker containers
# run as (the image's default `unflatten`, uid 65532) so
# `auth._validate_file_security` accepts it unmodified. Prints the token
# ONCE for legacy launchers that need a setup link. The local distribution
# sets UNFLATTEN_INIT_PRINT_TOKEN=0: its setup asks only for a password and
# never needs to display the internal credential.
set -euo pipefail

PRINCIPAL_FILE="${UNFLATTEN_PRINCIPAL_FILE:-/data/principals.json}"

if [ -f "$PRINCIPAL_FILE" ]; then
  echo "init: principal file already exists at $PRINCIPAL_FILE -- leaving it untouched"
  exit 0
fi

TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"

python3 - "$PRINCIPAL_FILE" "$TOKEN" <<'PY'
import json
import os
import sys

path, token = sys.argv[1], sys.argv[2]
payload = {
    "principal_file_schema_version": "1.0.0",
    "principals": [
        {
            "principal_id": "local",
            "token": token,
            "capabilities": {
                "submit_diagnostic": True,
                "read_status": True,
                "download_prediction_ir": True,
                "cancel_own_job": True,
                "download_docx": True,
            },
        }
    ],
}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(payload, f)
os.chmod(tmp, 0o400)
os.replace(tmp, path)
PY

echo "init: generated principal file at $PRINCIPAL_FILE"
if [[ "${UNFLATTEN_INIT_PRINT_TOKEN:-1}" == 1 ]]; then
  echo "init: bearer token (shown once, not stored anywhere else): $TOKEN"
fi

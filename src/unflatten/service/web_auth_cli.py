"""``python -m unflatten.service.web_auth_cli`` -- local password reset
(design §2.7). The password is read from stdin, never from argv or a log
line: argv ends up in shell history and process listings, stdin does not.

Callers must ensure `--data-dir` already exists; this CLI does not create it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TextIO

from unflatten.service.auth import PrincipalFileInvalid, load_principal_store
from unflatten.service.job_store import JobStore
from unflatten.service.web_auth import WebAuthFile, WebSessionStore

# Arbitrary but valid job-queue limits: `reset-password` only touches
# `web_sessions`, never `conversion_jobs`, so these values are never read.
_UNUSED_JOB_STORE_LIMITS = {
    "lease_duration_seconds": 3600,
    "max_attempts": 1,
    "global_active_jobs": 1,
    "active_jobs_per_principal": 1,
}


def _read_password(stdin: TextIO) -> str:
    if stdin.isatty():
        import getpass

        return getpass.getpass("New password: ")
    line = stdin.readline()
    if not line:
        raise SystemExit("no password provided on stdin")
    return line.rstrip("\n")


def reset_password(
    data_dir: Path,
    *,
    stdin: TextIO,
    principal_id: str | None = None,
    principal_file: Path | None = None,
) -> int:
    """Overwrite `web-auth.json` with a new password and revoke every web
    session, so a stale cookie can never outlive its password.

    If `web-auth.json` already exists, its `principal_id` is kept as-is
    (`principal_id`/`principal_file` are ignored). Otherwise -- a fresh
    install -- there is no recorded `principal_id` to default to: guessing
    one (e.g. "local") could silently bind sessions to a principal that
    does not exist in this deployment's principal file (`run-local.sh`
    uses `local-owner`, the Docker init uses `local`), which would make
    every job invisible to the UI (`get_job` is principal-scoped). The
    caller must name an existing principal explicitly.
    """
    web_auth = WebAuthFile(data_dir / "web-auth.json")
    if web_auth.exists():
        target_principal_id = web_auth.read()["principal_id"]
    elif principal_id is None:
        print(
            "setup not done yet -- use the setup wizard, or pass --principal-id",
            file=sys.stderr,
        )
        return 2
    else:
        resolved_principal_file = (
            principal_file if principal_file is not None else data_dir / "principal.json"
        )
        try:
            principals = load_principal_store(resolved_principal_file)
        except PrincipalFileInvalid as exc:
            print(f"cannot load principal file {resolved_principal_file}: {exc}", file=sys.stderr)
            return 2
        if not principals.has_principal(principal_id):
            print(
                f"unknown principal_id {principal_id!r} in {resolved_principal_file}",
                file=sys.stderr,
            )
            return 2
        target_principal_id = principal_id

    password = _read_password(stdin)
    web_auth.write(target_principal_id, password)

    store = JobStore(data_dir / "jobs.sqlite3", **_UNUSED_JOB_STORE_LIMITS)
    store.initialize()
    WebSessionStore(store, time_source=time.time).revoke_all(target_principal_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reset_parser = subparsers.add_parser(
        "reset-password",
        help="overwrite the local password (read from stdin) and revoke all sessions",
    )
    reset_parser.add_argument("--data-dir", type=Path, required=True)
    reset_parser.add_argument(
        "--principal-id",
        default=None,
        help="required on a fresh install (no web-auth.json yet); must exist in --principal-file",
    )
    reset_parser.add_argument(
        "--principal-file",
        type=Path,
        default=None,
        help="defaults to <data-dir>/principal.json",
    )
    args = parser.parse_args(argv)
    if args.command == "reset-password":
        return reset_password(
            args.data_dir,
            stdin=sys.stdin,
            principal_id=args.principal_id,
            principal_file=args.principal_file,
        )
    return 1  # pragma: no cover -- unreachable, `subparsers` closes the choice set


if __name__ == "__main__":
    sys.exit(main())

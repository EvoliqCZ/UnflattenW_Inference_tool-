import { readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

/**
 * T41: `start-api.sh` now runs several times concurrently (one backend per
 * browser project, plus the shared `app`/`session-expiry` backends,
 * `playwright.config.ts`), each writing its own bootstrap token to a
 * per-port path -- a single shared filename would have the same instances
 * race and overwrite each other's token that the per-project backend split
 * itself fixes. Read fresh per test so a server restart between runs is
 * picked up (same reasoning as the original `setup-login.spec.ts` comment).
 */
export function readBootstrapToken(port: number): string {
  const path = join(tmpdir(), `unflatten-e2e-bootstrap-token-${String(port)}`)
  return readFileSync(path, 'utf-8').trim()
}

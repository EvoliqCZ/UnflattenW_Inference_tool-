#!/usr/bin/env node
// T46: prepares the one `HF_HOME` directory a `models` e2e backend
// (`models.spec.ts`, `MODELS_PORTS` in `fixtures/ports.ts` -- one dedicated
// backend per browser project) points its worker/API pair at
// (`modelsHfHomeForPort` below, also imported by `playwright.config.ts` for
// the `HF_HOME` env value it passes to that backend's `apiWebServer(...)`
// command). Keyed by port, one directory per project, so three concurrent
// projects never race the same `rm -rf` + copy.
//
// Run as a plain step prepended to each `models` backend's `webServer.command`
// (not a Playwright `globalSetup`: those run as the LAST of the
// `globalSetup` task group, which itself runs AFTER `webServer` entries --
// themselves plugins started earlier in the same task group in this
// Playwright version -- so a `globalSetup` here would still see an empty
// `HF_HOME` by the time the API process boots and calls `verify_bundle`).
//
// `api_cli.py` resolves `--models-cache-dir` (absent from every
// `start-api.sh` invocation, including the `models` ones -- that script is
// owned by another stream, see `playwright.config.ts`'s own comment) to
// `HF_HOME/hub` when `HF_HOME` is set. That SAME directory is also what
// `verify_bundle()` checks the base-model pin against at API startup, so it
// must already contain the Qwen snapshot `tests/data/e2e/hf-cache` carries --
// copied fresh here rather than pointed at directly, so `fake_worker.py`
// (T46) can freely add the synthetic `hf-fp8` variant's own snapshot folder
// under it across repeated runs without ever touching that git-tracked
// fixture.
import { cpSync, existsSync, mkdirSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')

export function modelsHfHomeForPort(port) {
  return path.join(os.tmpdir(), `unflatten-e2e-models-hf-home-${String(port)}`)
}

function prepare(port) {
  const hfHome = modelsHfHomeForPort(port)
  const hubDir = path.join(hfHome, 'hub')
  if (existsSync(hfHome)) {
    rmSync(hfHome, { recursive: true, force: true })
  }
  mkdirSync(hubDir, { recursive: true })
  cpSync(path.join(REPO_ROOT, 'tests', 'data', 'e2e', 'hf-cache'), hubDir, { recursive: true })
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const port = process.argv[2]
  if (!port) {
    throw new Error('prepare-models-cache.mjs: a port argument is required')
  }
  prepare(port)
}

import path from 'node:path'
import { defineConfig, devices } from '@playwright/test'
import { APP_PORT, MODELS_PORTS, SESSION_EXPIRY_PORT, SETUP_LINK_PORTS, WIZARD_PORTS } from './e2e/fixtures/ports'
import { modelsHfHomeForPort } from './e2e/prepare-models-cache.mjs'

// design §7.6: every e2e scenario runs in Chromium, Firefox and WebKit
// (cookie/SameSite/Sec-Fetch-Site behaviour differs between engines, so one
// engine is not proof). WebKit cannot launch on this development host --
// `playwright install-deps` needs root and this host has none (37 missing
// system libraries, see docs/runs/radix-csp-probe-2026-09-27.md). The
// project stays configured (CI / a root-capable host runs it); specs skip it
// locally via `test.skip(browserName === 'webkit', ...)`.
//
// T41 (plan §I): a *single* shared `api` backend used to serve every
// project, which made `POST /v1/setup` a one-shot race between whichever
// project's worker reached `/setup` first -- the other project's
// `setup-login.spec.ts` run silently fell back to plain login and never
// exercised the wizard. Fixed by giving each project its own backend on its
// own port (`use.baseURL` below), so `setup-login.spec.ts` runs the real
// wizard once per project with no fallback. Two more backend groups exist
// beside the three per-project wizard ones (ports in `e2e/fixtures/ports.ts`):
//   - `app` (8440): shared by every non-wizard spec (they authenticate
//     directly against the API, never through `/setup`'s UI -- see
//     `e2e/fixtures/auth.ts`), across all three projects. Sharing it is safe
//     because those specs only assert on their own job(s), identified by a
//     unique label, never on the total job count.
//   - `session-expiry` (8450): its own backend with
//     `UNFLATTEN_WEB_SESSION_IDLE_SECONDS=2` (plan §I), also shared across
//     projects for the same reason.
//   - `models` (8460-8462, T46): one dedicated backend PER PROJECT (like the
//     wizard ones, unlike `app`) -- `models.spec.ts` mutates the same fixed
//     three variant ids (download/load/set-default state), which two
//     projects sharing one backend would race. The only backends configured
//     with `UNFLATTEN_MODELS_CONFIG` (`tests/data/e2e/models.json` -- one
//     local `bundle` variant, one `hf` source variant) and `HF_HOME` (`e2e/prepare-models-cache.mjs`, prepended to
//     each one's `command` below, prepares that directory fresh on every run
//     -- NOT a Playwright `globalSetup`: those run AFTER every `webServer`
//     entry in this Playwright version, which is too late for a backend
//     whose own boot depends on the directory already being there).
function apiWebServer(port: number, env?: Record<string, string>, prependCommand?: string) {
  return {
    command: `${prependCommand ?? ''}bash scripts/ensure-build.sh && bash scripts/start-api.sh`,
    url: `http://127.0.0.1:${String(port)}/health/live`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: { UNFLATTEN_E2E_PORT: String(port), ...env },
  }
}

function modelsWebServer(port: number) {
  return apiWebServer(
    port,
    {
      UNFLATTEN_MODELS_CONFIG: path.resolve(import.meta.dirname, '..', 'tests', 'data', 'e2e', 'models.json'),
      HF_HOME: modelsHfHomeForPort(port),
    },
    `node e2e/prepare-models-cache.mjs ${String(port)} && `,
  )
}

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: 'html',
  use: {
    baseURL: 'http://127.0.0.1:4174',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], baseURL: `http://127.0.0.1:${String(WIZARD_PORTS.chromium)}` },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'], baseURL: `http://127.0.0.1:${String(WIZARD_PORTS.firefox)}` },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'], baseURL: `http://127.0.0.1:${String(WIZARD_PORTS.webkit)}` },
    },
  ],
  webServer: [
    {
      command: 'npm run preview:csp-probe',
      url: 'http://127.0.0.1:4174/index.html',
      reuseExistingServer: !process.env.CI,
    },
    // T37/T41: the real FastAPI service (`--web-ui spa`, `UNFLATTEN_E2E=1`)
    // plus the torch-free `fake_worker.py`, one instance per browser project
    // (wizard-only, `setup-login.spec.ts`), one shared by the rest of the
    // suite (`app`), and one with a 2s idle timeout (`session-expiry`).
    // Owner 2026-09-29 (DEC-052 addendum): the wizard backends are the
    // loopback-only deployment shape (password-only setup); the setup-link
    // backends are the token-protected one (`/setup#token=` link).
    apiWebServer(WIZARD_PORTS.chromium, { UNFLATTEN_SETUP_TOKENLESS: '1' }),
    apiWebServer(WIZARD_PORTS.firefox, { UNFLATTEN_SETUP_TOKENLESS: '1' }),
    apiWebServer(WIZARD_PORTS.webkit, { UNFLATTEN_SETUP_TOKENLESS: '1' }),
    apiWebServer(SETUP_LINK_PORTS.chromium, { UNFLATTEN_SETUP_TOKENLESS: '0' }),
    apiWebServer(SETUP_LINK_PORTS.firefox, { UNFLATTEN_SETUP_TOKENLESS: '0' }),
    apiWebServer(SETUP_LINK_PORTS.webkit, { UNFLATTEN_SETUP_TOKENLESS: '0' }),
    apiWebServer(APP_PORT),
    apiWebServer(SESSION_EXPIRY_PORT, { UNFLATTEN_WEB_SESSION_IDLE_SECONDS: '2' }),
    modelsWebServer(MODELS_PORTS.chromium),
    modelsWebServer(MODELS_PORTS.firefox),
    modelsWebServer(MODELS_PORTS.webkit),
  ],
})

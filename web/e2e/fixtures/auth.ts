import type { Page } from '@playwright/test'
import { readBootstrapToken } from './bootstrap-token'

/**
 * T41: the password every non-wizard spec's backend ends up with. Fixed and
 * shared across those specs -- only `setup-login.spec.ts` needs to pick its
 * own (it drives the real wizard UI, design §2.4's 12-character minimum is
 * the only constraint).
 */
export const PASSWORD = 'correct horse battery staple'

const WEB_REQUEST_HEADERS = {
  'content-type': 'application/json',
  'x-unflatten-client': 'web',
  // design §2.3/§11 bullet 17: `POST /v1/setup` and `POST /v1/session`
  // accept only same-origin-shaped requests (`_web_request_context_ok`,
  // api.py). A real browser sets this header itself and refuses to let
  // script override it; Playwright's `page.request` is not a browser fetch,
  // so it has to be set by hand here.
  'sec-fetch-site': 'none',
}

/**
 * Establishes an authenticated session directly against the API, bypassing
 * the setup wizard and login form UI entirely (T41, plan §I). This is
 * deliberate: `setup-login.spec.ts` is the one spec that must exercise the
 * real wizard end to end, so every *other* spec that merely needs "already
 * logged in" gets there without competing for that backend's one-shot
 * `POST /v1/setup`. `page.request` shares its cookie jar with `page`'s own
 * browser context, so the session cookie `POST /v1/setup`/`POST /v1/session`
 * sets is already attached by the time the caller does `page.goto(...)`.
 *
 * Idempotent: several parallel workers can call this against the same
 * shared backend (`APP_PORT`/`SESSION_EXPIRY_PORT`) -- whichever request
 * reaches `POST /v1/setup` first completes it (201), every other one gets
 * `SETUP_ALREADY_COMPLETE` (409) and just logs in with the same password.
 */
export async function ensureAuthenticated(page: Page, port: number): Promise<void> {
  const token = readBootstrapToken(port)
  const setupResponse = await page.request.post('/v1/setup', {
    headers: { ...WEB_REQUEST_HEADERS, authorization: `Bearer ${token}` },
    data: { password: PASSWORD },
  })
  if (setupResponse.status() === 201) {
    return
  }
  if (setupResponse.status() !== 409) {
    throw new Error(
      `POST /v1/setup failed unexpectedly: ${String(setupResponse.status())} ${await setupResponse.text()}`,
    )
  }
  const loginResponse = await page.request.post('/v1/session', {
    headers: WEB_REQUEST_HEADERS,
    data: { password: PASSWORD },
  })
  if (loginResponse.status() !== 201) {
    throw new Error(
      `POST /v1/session failed unexpectedly: ${String(loginResponse.status())} ${await loginResponse.text()}`,
    )
  }
}

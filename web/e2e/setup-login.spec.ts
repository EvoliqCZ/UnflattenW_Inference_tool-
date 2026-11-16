import type { Page } from '@playwright/test'
import { expect, test } from './fixtures/csp-guard'
import { readBootstrapToken } from './fixtures/bootstrap-token'
import { SETUP_LINK_PORTS } from './fixtures/ports'

// T37/T41 e2e smoke: setup wizard -> login -> lands on the convert page ->
// logout, against the real API. Each project (chromium/firefox/webkit) now
// gets its own backend, on its own port, used by nothing else
// (`playwright.config.ts`'s `WIZARD_PORTS`, plan §I) -- that backend has
// never seen `POST /v1/setup` before this test runs, so the wizard branch
// below is the only branch that can be taken, with no fallback to a
// pre-completed setup masking a defect in it.
//
// Owner 2026-09-29 (DEC-052 addendum): the wizard backends run with
// `UNFLATTEN_SETUP_TOKENLESS=1` (loopback-only deployment) and the wizard
// asks only for a password. The second describe block drives the other
// shape -- a token-protected backend (`SETUP_LINK_PORTS`) reached through
// the `/setup#token=` link `run-local.sh` prints.
const PASSWORD = 'correct horse battery staple'

async function finishWizardAfterPassword(page: Page): Promise<void> {
  // Step "system": either "Continue" (ready) or "Continue anyway".
  await page.getByRole('button', { name: /^Continue/ }).click()

  // Step "model": "Finish later" always advances regardless of the fake
  // worker's download state (T45).
  await page.getByRole('button', { name: 'Finish later' }).click()
  await expect(page.getByRole('heading', { name: 'Done' })).toBeVisible()
}

async function fillPassword(page: Page): Promise<void> {
  await page.getByLabel('Password', { exact: true }).fill(PASSWORD)
  await page.getByLabel('Confirm password').fill(PASSWORD)
  await page.getByRole('button', { name: 'Create password' }).click()
}

async function completeSetup(page: Page): Promise<void> {
  await page.goto('/setup')
  await expect(page.getByLabel('Password', { exact: true })).toBeVisible()
  // The brand mark is a same-origin bundled file the real CSP lets through
  // (`img-src 'self'`): a blocked or missing image decodes to width 0.
  const mark = page.getByRole('img', { name: 'Unflatten W' })
  await expect(mark).toBeVisible()
  expect(await mark.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0)
  // No token field, no token wording anywhere on the page.
  await expect(page.getByText(/token/i)).toHaveCount(0)

  await fillPassword(page)
  await finishWizardAfterPassword(page)

  // Leave via the settings menu so this run ends logged out, matching the
  // state `logout.spec.ts`'s (and every other spec's) direct-API login
  // helper starts from.
  await page.getByRole('link', { name: 'Upload your own file' }).click()
  await expect(page.getByRole('heading', { name: 'Convert a document' })).toBeVisible()
  await page.getByRole('button', { name: 'Nastavenia a téma' }).click()
  await page.getByRole('menuitem', { name: 'Odhlásiť' }).click()
  await expect(page).toHaveURL(/\/login$/)
}

test.describe('setup wizard, login, logout', () => {
  test.beforeAll(({ browserName }) => {
    test.skip(
      browserName === 'webkit',
      'WebKit cannot launch on this host (missing system libs, no sudo); deferred to CP8 on a root-capable host.',
    )
  })

  test('tokenless setup wizard completes, login lands on convert, logout returns to login', async ({ page }) => {
    await completeSetup(page)

    // `AppShell` (with the settings/logout menu) only wraps this and the
    // other authenticated routes -- `/setup` uses the bare `AuthShell`.
    await page.goto('/login')
    await page.getByLabel('Password', { exact: true }).fill(PASSWORD)
    await page.getByRole('button', { name: 'Log in' }).click()

    await expect(page.getByRole('heading', { name: 'Convert a document' })).toBeVisible()

    await page.getByRole('button', { name: 'Nastavenia a téma' }).click()
    await page.getByRole('menuitem', { name: 'Odhlásiť' }).click()
    await expect(page).toHaveURL(/\/login$/)
  })
})

test.describe('setup through the run-local.sh setup link (token-protected backend)', () => {
  test.beforeAll(({ browserName }) => {
    test.skip(
      browserName === 'webkit',
      'WebKit cannot launch on this host (missing system libs, no sudo); deferred to CP8 on a root-capable host.',
    )
  })

  test('without the link the wizard says where to find it; the link sets up and leaves no token in the URL', async ({
    page,
    browserName,
  }) => {
    const port = SETUP_LINK_PORTS[browserName]
    const origin = `http://127.0.0.1:${String(port)}`

    await page.goto(`${origin}/setup`)
    await expect(page.getByText('Open the setup link printed by run-local.sh.')).toBeVisible()
    await expect(page.getByLabel('Password', { exact: true })).toHaveCount(0)

    const token = readBootstrapToken(port)
    // A fresh document load, as when the link is opened from the terminal --
    // `/setup` -> `/setup#token=` in the same document would be a mere hash
    // change the already-mounted page never re-reads.
    await page.goto('about:blank')
    await page.goto(`${origin}/setup#token=${token}`)
    await expect(page.getByLabel('Password', { exact: true })).toBeVisible()
    await expect(page).toHaveURL(`${origin}/setup`)

    await fillPassword(page)
    await finishWizardAfterPassword(page)
    expect(page.url()).not.toContain(token)
  })
})

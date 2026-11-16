import { expect, test } from './fixtures/csp-guard'
import { ensureAuthenticated } from './fixtures/auth'
import { APP_PORT } from './fixtures/ports'

// T41: the settings-menu logout button, against the shared `app` backend.
// `setup-login.spec.ts` already covers logout once as the tail of its own
// wizard flow; this spec starts from an already-authenticated session
// (`fixtures/auth.ts`) so it exercises logout on its own, independent of the
// wizard.
test.use({ baseURL: `http://127.0.0.1:${String(APP_PORT)}` })

test.describe('logout', () => {
  test.beforeAll(({ browserName }) => {
    test.skip(
      browserName === 'webkit',
      'WebKit cannot launch on this host (missing system libs, no sudo); deferred to CP8 on a root-capable host.',
    )
  })

  test.beforeEach(async ({ page }) => {
    await ensureAuthenticated(page, APP_PORT)
  })

  test('logging out from the settings menu returns to the login screen', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'Convert a document' })).toBeVisible()

    await page.getByRole('button', { name: 'Nastavenia a téma' }).click()
    await page.getByRole('menuitem', { name: 'Odhlásiť' }).click()
    await expect(page).toHaveURL(/\/login$/)

    // The session is really gone, not just a client-side route change:
    // reloading a protected route bounces back to /login (design §7.5 guard
    // loader).
    await page.goto('/jobs')
    await expect(page).toHaveURL(/\/login(\?|$)/)
  })
})

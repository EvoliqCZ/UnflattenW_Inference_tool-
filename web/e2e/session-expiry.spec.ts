import { expect, test } from './fixtures/csp-guard'
import { ensureAuthenticated } from './fixtures/auth'
import { SESSION_EXPIRY_PORT } from './fixtures/ports'

// T41: idle session expiry, against its own backend with
// `UNFLATTEN_WEB_SESSION_IDLE_SECONDS=2` (`playwright.config.ts`). Every
// authenticated request that reaches the API touches the session, but
// `WebSessionStore.touch` only persists that at most once every 60s
// (`web_auth.py` `SESSION_TOUCH_THROTTLE_SECONDS`) -- with a 2s idle window
// that throttle never fires within the test, so the session genuinely goes
// idle 2s after `ensureAuthenticated` logs in, background polling or not.
test.use({ baseURL: `http://127.0.0.1:${String(SESSION_EXPIRY_PORT)}` })

test.describe('session idle expiry', () => {
  test.beforeAll(({ browserName }) => {
    test.skip(
      browserName === 'webkit',
      'WebKit cannot launch on this host (missing system libs, no sudo); deferred to CP8 on a root-capable host.',
    )
  })

  test.beforeEach(async ({ page }) => {
    await ensureAuthenticated(page, SESSION_EXPIRY_PORT)
  })

  test('an idle session prompts to log in again without losing the current screen', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'Convert a document' })).toBeVisible()

    // Outlast the 2s idle window while resting on this same screen (no
    // navigation -- a navigation re-runs the router's guard loader, which
    // would redirect to /login on its own before the dialog ever gets a
    // chance to open, design §7.5).
    await page.waitForTimeout(2_500)

    // TanStack Query's default `refetchOnWindowFocus` listens for
    // `visibilitychange` on `window` (`focusManager.ts`, not a `focus` event)
    // and refetches every mounted query -- the same trigger a real user
    // switching back to this tab after the idle window would hit, without
    // waiting out this page's own 30s readiness poll interval.
    await page.evaluate(() => {
      window.dispatchEvent(new Event('visibilitychange'))
    })

    await expect(page.getByRole('dialog', { name: 'Session expired' })).toBeVisible({ timeout: 10_000 })
    // Still on the convert screen underneath -- the dialog does not
    // navigate. The heading is inert (`aria-hidden`) while the modal is
    // open (Radix's own a11y behaviour), so `getByRole` cannot see it here
    // -- `getByText` does not consult the accessibility tree.
    await expect(page.getByText('Convert a document')).toBeVisible()

    await page.getByLabel('Password').fill('correct horse battery staple')
    await page.getByRole('button', { name: 'Log in' }).click()
    await expect(page.getByRole('dialog', { name: 'Session expired' })).toBeHidden()
  })
})

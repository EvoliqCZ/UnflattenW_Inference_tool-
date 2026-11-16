import { expect, test } from './fixtures/csp-guard'
import { ensureAuthenticated } from './fixtures/auth'
import { minimalPdfBytes } from './fixtures/minimal-pdf'
import { APP_PORT } from './fixtures/ports'

// T41: convert flow for a single PDF, against the shared `app` backend
// (`playwright.config.ts`). Authenticates directly against the API
// (`fixtures/auth.ts`) rather than through the setup wizard UI -- that is
// `setup-login.spec.ts`'s job alone (plan §I).
test.use({ baseURL: `http://127.0.0.1:${String(APP_PORT)}` })

test.describe('convert: PDF upload', () => {
  test.beforeAll(({ browserName }) => {
    test.skip(
      browserName === 'webkit',
      'WebKit cannot launch on this host (missing system libs, no sudo); deferred to CP8 on a root-capable host.',
    )
  })

  test.beforeEach(async ({ page }) => {
    await ensureAuthenticated(page, APP_PORT)
  })

  test('a single PDF converts through to a downloadable Word result', async ({ page }, testInfo) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'Convert a document' })).toBeVisible()

    // With no file chosen Submit is `aria-disabled` (never native `disabled`):
    // it must LOOK disabled via the shared Button (muted tokens, not-allowed
    // cursor, no hover change) and stay focusable so its reason is reachable.
    const submit = page.getByRole('button', { name: 'Submit' })
    await expect(submit).toHaveAttribute('aria-disabled', 'true')
    await expect(submit).not.toHaveAttribute('disabled')
    await expect(submit).toHaveCSS('cursor', 'not-allowed')
    const mutedBackground = await submit.evaluate((el) => getComputedStyle(el).backgroundColor)
    await submit.hover()
    await expect(submit).toHaveCSS('background-color', mutedBackground)
    await expect(submit).toHaveCSS('opacity', '1')
    await submit.focus()
    await expect(submit).toBeFocused()
    await page.screenshot({ path: testInfo.outputPath('convert-submit-aria-disabled.png'), fullPage: true })

    const label = `e2e-pdf-${String(Date.now())}`
    await page.getByLabel('Choose files').setInputFiles({
      name: 'report.pdf',
      mimeType: 'application/pdf',
      buffer: minimalPdfBytes(),
    })
    await expect(page.getByText('report.pdf')).toBeVisible()

    await page.getByLabel('Job label').fill(label)
    await expect(submit).toHaveAttribute('aria-disabled', 'false')
    await expect(submit).not.toHaveCSS('cursor', 'not-allowed')
    await expect(submit).not.toHaveCSS('background-color', mutedBackground)
    await page.getByRole('button', { name: 'Submit' }).click()

    await expect(page).toHaveURL(/\/jobs\/job_/)
    // fake_worker (T37) drains a job in ~1s -- no GPU, no torch. `Succeeded`
    // is the eventual `JobStateBadge` label; the terminal "Výsledok"/Result
    // tabs only render once the job reaches a terminal state (job-detail/page.tsx).
    await expect(page.getByRole('link', { name: 'Download Word (diagnostic)' })).toBeVisible({ timeout: 15_000 })
  })
})

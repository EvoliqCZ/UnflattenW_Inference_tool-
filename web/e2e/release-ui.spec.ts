import { expect, test } from '@playwright/test'
import { authenticatedSession, runningJob, unauthenticatedSession } from '../src/test/msw/fixtures'

test('all 48 progress rows are reachable by wheel and keyboard on desktop and mobile', async ({ page }) => {
  await page.route('**/v1/session', (route) => route.fulfill({ json: authenticatedSession }))
  await page.route('**/v1/conversions/job-running', (route) =>
    route.fulfill({
      json: {
        ...runningJob,
        label: 'Synthetic 48-page document',
        pages_total: 48,
        pages_done: 18,
        current_page: 18,
        pages: Array.from({ length: 18 }, (_, index) => ({
          index,
          state: 'succeeded',
          seconds: 30,
          decode_steps: 1000,
          error_code: null,
        })),
      },
    }),
  )
  await page.goto('/jobs/job-running')
  const rows = page.getByRole('list').filter({ has: page.getByText('Page 48', { exact: true }) })
  await expect(rows.getByRole('listitem')).toHaveCount(48)
  const viewport = page.locator('[data-radix-scroll-area-viewport], [role="region"][aria-label="Page progress"]')
  // Reproduces the original bug: a max-height root clipped its unbounded viewport.
  expect(await viewport.evaluate((el) => el.scrollHeight - el.clientHeight)).toBeGreaterThan(100)
  await viewport.hover()
  await page.mouse.wheel(0, 3000)
  await expect.poll(() => viewport.evaluate((el) => el.scrollTop)).toBeGreaterThan(100)
  const lastRow = rows.getByRole('listitem').last()
  await expect(lastRow).toBeInViewport()
  await viewport.focus()
  await page.keyboard.press('Control+Home')
  await expect.poll(() => viewport.evaluate((el) => el.scrollTop)).toBe(0)
  await page.keyboard.press('Control+End')
  await expect(lastRow).toBeInViewport()
  await page.setViewportSize({ width: 390, height: 740 })
  await viewport.evaluate((el) => {
    el.scrollTop = 0
  })
  await viewport.hover()
  await page.mouse.wheel(0, 3000)
  await expect(lastRow).toBeInViewport()
})

test('company logo is visible in both themes before login and links to the selected language', async ({
  page,
}, testInfo) => {
  await page.route('**/v1/session', (route) => route.fulfill({ json: unauthenticatedSession }))
  for (const theme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme: theme })
    await page.goto('/login')
    const company = page.getByRole('contentinfo').getByRole('link', { name: 'EVOLIQ s.r.o.' })
    await expect(company).toHaveAttribute('href', 'https://www.evoliq.cz/en')
    await expect(company).toHaveAttribute('target', '_blank')
    const visibleLogo = company.locator('img:visible')
    await expect(visibleLogo).toHaveCount(1)
    await expect(visibleLogo).toHaveAttribute(
      'src',
      new RegExp(theme === 'dark' ? 'evoliq-logo-white' : 'evoliq-logo-dark'),
    )
    expect(await visibleLogo.evaluate((el) => (el as HTMLImageElement).naturalWidth)).toBeGreaterThan(900)
    await page.screenshot({ path: testInfo.outputPath(`login-${theme}.png`), fullPage: true })
  }
  await page.setViewportSize({ width: 390, height: 740 })
  await page.getByRole('button', { name: 'Slovenčina' }).click()
  const company = page.getByRole('contentinfo').getByRole('link', { name: 'EVOLIQ s.r.o.' })
  await expect(company).toHaveAttribute('href', 'https://www.evoliq.cz/sk')
  await expect(company).toBeInViewport()
  await page.screenshot({ path: testInfo.outputPath('login-mobile-sk.png'), fullPage: true })
})

test('authenticated navigation and theme controls survive denied browser storage', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      get() {
        throw new DOMException('Denied', 'SecurityError')
      },
    })
  })
  await page.route('**/v1/session', (route) => route.fulfill({ json: authenticatedSession }))
  await page.route('**/v1/conversions/job-running', (route) => route.fulfill({ json: runningJob }))
  await page.goto('/jobs/job-running')
  await expect(page.getByRole('navigation', { name: 'Main navigation' })).toBeVisible()
  await page.getByRole('button', { name: 'Settings and theme' }).click()
  await page.getByRole('menuitemradio', { name: 'Dark' }).click()
  await expect(page.locator('html')).toHaveClass(/dark/)
  await page.getByRole('button', { name: 'Slovenčina' }).click()
  await expect(page.getByRole('navigation', { name: 'Hlavná navigácia' })).toBeVisible()
  await expect(page.getByText('Unexpected Application Error!')).toHaveCount(0)
})

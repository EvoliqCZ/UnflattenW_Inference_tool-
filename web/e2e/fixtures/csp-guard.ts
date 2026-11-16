import { expect, test as base } from '@playwright/test'

/**
 * T37 (design §7.6): every e2e spec against the real SPA must produce zero
 * `securitypolicyviolation` events. `addInitScript` runs before any page
 * script, so it also covers the very first navigation; the collected events
 * live on `window.__csp` (readable from the test for a more specific
 * assertion), and an auto-fixture asserts `length === 0` once per test,
 * after the test body ran -- the effect the plan calls "afterEach".
 *
 * `react-hooks/rules-of-hooks` misfires on Playwright's own fixture shape --
 * a fixture named `page` calling `use(...)` looks, to that rule, like a
 * non-hook function calling a hook -- disabled for this block only.
 */
/* eslint-disable react-hooks/rules-of-hooks */
export const test = base.extend<{ cspGuard: undefined }>({
  page: async ({ page }, use) => {
    await page.addInitScript(() => {
      const w = window as unknown as { __csp: string[] }
      w.__csp = []
      document.addEventListener('securitypolicyviolation', (event) => {
        w.__csp.push(`${event.violatedDirective}: ${event.blockedURI}`)
      })
    })
    await use(page)
  },
  cspGuard: [
    async ({ page }, use) => {
      await use(undefined)
      const violations = await page.evaluate(() => (window as unknown as { __csp: string[] }).__csp ?? [])
      expect(violations, `CSP violations recorded during the test: ${JSON.stringify(violations)}`).toEqual([])
    },
    { auto: true },
  ],
})
/* eslint-enable react-hooks/rules-of-hooks */

export { expect }

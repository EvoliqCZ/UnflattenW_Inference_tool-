import { expect, test } from '@playwright/test'
import { installViolationCollector, openEachPrimitive } from './csp-probe-helpers'

/**
 * Permanent port of the T08 CSP-nonce follow-up (docs/runs/radix-csp-probe-2026-09-27.md,
 * design §6.3, DEC-052 bullet 1): asserts the shipped nonce mechanism produces
 * zero `securitypolicyviolation` events for the two Radix primitives that
 * inject a `<style>` tag. Runs against `e2e/csp-probe` (built + served by
 * `npm run preview:csp-probe`, see playwright.config.ts `webServer`), which
 * emulates `web_static.py`'s per-request nonce substitution -- not the real
 * FastAPI server, which is wired up starting T18/T35.
 */
test.describe('CSP style nonce regression', () => {
  // T41: projects now default to a per-project API backend baseURL
  // (`playwright.config.ts`, wizard-only ports) -- this suite targets the
  // CSP-probe server instead, so it needs its own explicit baseURL back.
  test.use({ baseURL: 'http://127.0.0.1:4174' })

  test.beforeAll(({ browserName }) => {
    test.skip(
      browserName === 'webkit',
      'WebKit cannot launch on this host (missing system libs, playwright install-deps needs root); ' +
        'see docs/runs/radix-csp-probe-2026-09-27.md. Run on CI or a root-capable host.',
    )
  })

  test('CSP header carries a nonce and script-src stays plain self', async ({ page }) => {
    const response = await page.goto('/index.html')
    const header = response?.headers()['content-security-policy'] ?? ''
    expect(header).toMatch(/^default-src 'none'; script-src 'self'; style-src 'self' 'nonce-[\w-]+';/)
    expect(header).not.toContain("script-src 'self' 'nonce-")
  })

  test('two requests mint two different nonces', async ({ page }) => {
    const first = await page.goto('/index.html')
    const second = await page.goto('/index.html')
    const nonceOf = (h: string) => /nonce-([\w-]+)/.exec(h)?.[1]
    const a = nonceOf(first?.headers()['content-security-policy'] ?? '')
    const b = nonceOf(second?.headers()['content-security-policy'] ?? '')
    expect(a).toBeTruthy()
    expect(a).not.toBe(b)
  })

  test('opening every affected primitive produces zero securitypolicyviolation events', async ({ page }) => {
    await installViolationCollector(page)
    const response = await page.goto('/index.html')
    const header = response?.headers()['content-security-policy'] ?? ''
    expect(/nonce-([\w-]+)/.exec(header)?.[1], 'server did not mint a style-src nonce').toBeTruthy()

    for (const { name, violations } of await openEachPrimitive(page)) {
      expect(violations, `${name} triggered a CSP violation: ${JSON.stringify(violations)}`).toEqual([])
    }
  })
})

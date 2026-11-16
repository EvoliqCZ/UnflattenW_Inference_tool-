import type { Page } from '@playwright/test'

/** The two Radix primitives that inject a `<style>` tag (T08 finding, docs/runs/radix-csp-probe-2026-09-27.md):
 * Dialog via `react-remove-scroll`'s scrollbar-width `<style>`, Select via its own
 * `Select.Viewport` inline `<style>`. That is enough to prove the nonce mechanism
 * works end to end without re-running T08's full 8-primitive survey here. */
export const PRIMITIVES: { name: string; trigger: string }[] = [
  { name: 'Dialog', trigger: 'trigger-dialog' },
  { name: 'Select', trigger: 'trigger-select' },
]

export interface CapturedViolation {
  violatedDirective: string
  blockedURI: string
}

export async function installViolationCollector(page: Page) {
  await page.addInitScript(() => {
    const violations: CapturedViolation[] = []
    ;(window as unknown as { __cspViolations: CapturedViolation[] }).__cspViolations = violations
    document.addEventListener('securitypolicyviolation', (event) => {
      violations.push({ violatedDirective: event.violatedDirective, blockedURI: event.blockedURI })
    })
  })
}

/**
 * Reads the violations captured so far, then clears the buffer *in place*.
 * The `securitypolicyviolation` listener installed by `installViolationCollector`
 * closes over the original array object, not `window.__cspViolations` -- reassigning
 * that property to a new array (as an earlier version of this helper did) silently
 * detaches the listener from anything read afterwards, making every primitive
 * opened after the first drain look violation-free regardless of what actually
 * happened. Truncating the same array (`.length = 0` via splice) keeps the
 * reference intact.
 */
async function drainViolations(page: Page): Promise<CapturedViolation[]> {
  return page.evaluate(() => {
    const w = window as unknown as { __cspViolations: CapturedViolation[] }
    return w.__cspViolations.splice(0, w.__cspViolations.length)
  })
}

/**
 * Opens each primitive in turn and reports only the violations produced
 * while *that* primitive was open (the collector buffer is drained between
 * steps -- otherwise one real violation would falsely "leak" onto every
 * primitive opened afterwards). Closes with Escape before moving on.
 */
export async function openEachPrimitive(page: Page): Promise<{ name: string; violations: CapturedViolation[] }[]> {
  const results: { name: string; violations: CapturedViolation[] }[] = []
  for (const { name, trigger } of PRIMITIVES) {
    await page.getByTestId(trigger).click()
    // Radix primitives mount their portal content asynchronously.
    await page.waitForTimeout(150)
    const violations = await drainViolations(page)
    await page.keyboard.press('Escape')
    await page.waitForTimeout(300)
    violations.push(...(await drainViolations(page)))
    results.push({ name, violations })
  }
  return results
}

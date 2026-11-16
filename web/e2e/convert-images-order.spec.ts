import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { expect, test } from './fixtures/csp-guard'
import { ensureAuthenticated } from './fixtures/auth'
import { APP_PORT } from './fixtures/ports'

// T41: multi-image selection + reordering (UX §4.4 `UploadQueue`), against
// the shared `app` backend.
test.use({ baseURL: `http://127.0.0.1:${String(APP_PORT)}` })

const TINY_PNG = readFileSync(join(import.meta.dirname, '..', '..', 'tests', 'data', 'assets', 'tiny_2x2.png'))

test.describe('convert: image order', () => {
  test.beforeAll(({ browserName }) => {
    test.skip(
      browserName === 'webkit',
      'WebKit cannot launch on this host (missing system libs, no sudo); deferred to CP8 on a root-capable host.',
    )
  })

  test.beforeEach(async ({ page }) => {
    await ensureAuthenticated(page, APP_PORT)
  })

  test('images queue in selection order and the move buttons reorder them', async ({ page }) => {
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'Convert a document' })).toBeVisible()

    // Selected in this exact order: "page-b" first, "page-a" second.
    await page.getByLabel('Choose files').setInputFiles([
      { name: 'page-b.png', mimeType: 'image/png', buffer: TINY_PNG },
      { name: 'page-a.png', mimeType: 'image/png', buffer: TINY_PNG },
    ])

    const queueItems = page.locator('ol li')
    await expect(queueItems).toHaveCount(2)
    await expect(queueItems.nth(0)).toContainText('page-b.png')
    await expect(queueItems.nth(1)).toContainText('page-a.png')

    // Move the second item ("page-a.png") up one slot.
    await queueItems.nth(1).getByRole('button', { name: 'Move up' }).click()
    await expect(queueItems.nth(0)).toContainText('page-a.png')
    await expect(queueItems.nth(1)).toContainText('page-b.png')

    const label = `e2e-images-${String(Date.now())}`
    await page.getByLabel('Job label').fill(label)
    await page.getByRole('button', { name: 'Submit' }).click()

    await expect(page).toHaveURL(/\/jobs\/job_/)
    await expect(page.getByRole('link', { name: 'Download Word (diagnostic)' })).toBeVisible({ timeout: 15_000 })
  })
})

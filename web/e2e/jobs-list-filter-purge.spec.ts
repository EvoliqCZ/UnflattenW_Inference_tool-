import { expect, test } from './fixtures/csp-guard'
import { ensureAuthenticated } from './fixtures/auth'
import { minimalPdfBytes } from './fixtures/minimal-pdf'
import { APP_PORT } from './fixtures/ports'

// T41: jobs list search/filter and permanent delete, against the shared
// `app` backend. The backend is shared with other specs and other parallel
// workers of this same spec (`fullyParallel`), so every assertion below is
// scoped to this test's own job by its unique label -- never a total job
// count, which other tests/workers would also be affecting concurrently.
test.use({ baseURL: `http://127.0.0.1:${String(APP_PORT)}` })

test.describe('jobs list: filter and purge', () => {
  test.beforeAll(({ browserName }) => {
    test.skip(
      browserName === 'webkit',
      'WebKit cannot launch on this host (missing system libs, no sudo); deferred to CP8 on a root-capable host.',
    )
  })

  test.beforeEach(async ({ page }) => {
    await ensureAuthenticated(page, APP_PORT)
  })

  test('search finds the job by label, the Succeeded filter keeps it, and purge removes it for good', async ({
    page,
  }) => {
    const label = `e2e-purge-${String(Date.now())}`

    await page.goto('/')
    await page.getByLabel('Choose files').setInputFiles({
      name: 'report.pdf',
      mimeType: 'application/pdf',
      buffer: minimalPdfBytes(),
    })
    await page.getByLabel('Job label').fill(label)
    await page.getByRole('button', { name: 'Submit' }).click()
    await expect(page).toHaveURL(/\/jobs\/job_/)
    // Wait for the job to reach a terminal state before leaving its page --
    // fake_worker (T37) finishes it in ~1s.
    await expect(page.getByRole('link', { name: 'Download Word (diagnostic)' })).toBeVisible({ timeout: 15_000 })

    await page.goto('/jobs')
    await page.getByLabel('Search by label').fill(label)

    const row = page.getByRole('row').filter({ hasText: label })
    await expect(row).toHaveCount(1)
    await expect(row).toContainText('Succeeded')

    // `jobs/api.ts` `FILTER_STATES`: "Succeeded" maps to state `succeeded`
    // (the state this job is now in); "Ended" is `cancelled`/`expired`
    // only, so a succeeded job is deliberately absent from it.
    await page.getByRole('tab', { name: 'Succeeded' }).click()
    await expect(row).toHaveCount(1)

    // The "Active" tab excludes it -- the job already succeeded.
    await page.getByRole('tab', { name: 'Active' }).click()
    await expect(page.getByText('No job matches this filter.')).toBeVisible()

    await page.getByRole('tab', { name: 'All' }).click()
    await row.getByRole('button', { name: 'Actions' }).click()
    await page.getByRole('menuitem', { name: 'Delete permanently' }).click()
    await page.getByRole('button', { name: 'Delete', exact: true }).click()

    await expect(page.getByText('The job was permanently deleted.').first()).toBeVisible()
    await expect(page).toHaveURL(/\/jobs$/)
    await page.getByLabel('Search by label').fill(label)
    await expect(page.getByText('No job matches this filter.')).toBeVisible()
  })
})

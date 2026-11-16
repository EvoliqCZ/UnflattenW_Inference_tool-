import { readFile } from 'node:fs/promises'
import { expect, test } from './fixtures/csp-guard'
import { readBootstrapToken } from './fixtures/bootstrap-token'
import { minimalPdfBytes } from './fixtures/minimal-pdf'

test('Swagger authorizes, uploads, polls and downloads Word without the app UI', async ({
  page,
  baseURL,
  browserName,
}, testInfo) => {
  test.skip(browserName === 'webkit', 'WebKit system libraries are unavailable on this host.')
  if (!baseURL) throw new Error('Swagger browser test requires a base URL')
  const origin = new URL(baseURL).origin
  const token = readBootstrapToken(Number(new URL(origin).port))
  const external: string[] = []
  page.on('request', (request) => {
    if (new URL(request.url()).origin !== origin) external.push(request.url())
  })
  // A pre-existing UI cookie must not be mixed with machine credentials.
  await page.context().addCookies([{ name: 'unflatten_session', value: 'old-ui-session', url: origin }])
  await page.goto('/docs')
  await page.getByRole('button', { name: 'Authorize', exact: true }).click()
  await page.locator('#auth-bearer-value').fill(token)
  await page.locator('.dialog-ux').getByRole('button', { name: 'Apply credentials', exact: true }).first().click()
  await page.locator('.dialog-ux').getByRole('button', { name: 'Close', exact: true }).first().click()

  const submit = page.locator('#operations-Conversions-_submit_v1_conversions_post')
  await submit.locator('.opblock-summary').click()
  await submit.getByRole('button', { name: 'Try it out', exact: true }).click()
  await submit
    .locator('input[type="file"]')
    .setInputFiles({ name: 'swagger.pdf', mimeType: 'application/pdf', buffer: minimalPdfBytes() })
  const accepted = page.waitForResponse(
    (response) =>
      response.request().method() === 'POST' && new URL(response.url()).pathname.endsWith('/v1/conversions'),
  )
  await submit.getByRole('button', { name: 'Execute', exact: true }).click()
  const response = await accepted
  expect(response.status(), await response.text()).toBe(202)
  expect(response.request().headers()['idempotency-key']).toMatch(/^[0-9a-f-]{36}$/)
  expect(response.request().headers().cookie).toBeUndefined()
  const job = (await response.json()) as { job_id: string }

  const status = page.locator('#operations-Conversions-_status_v1_conversions__job_id__get')
  await status.locator('.opblock-summary').click()
  await status.getByRole('button', { name: 'Try it out', exact: true }).click()
  await status.getByPlaceholder('job_id', { exact: true }).fill(job.job_id)
  await expect
    .poll(
      async () => {
        const result = page.waitForResponse(
          (r) => r.request().method() === 'GET' && new URL(r.url()).pathname.endsWith(`/v1/conversions/${job.job_id}`),
        )
        await status.getByRole('button', { name: 'Execute', exact: true }).click()
        const polled = await result
        expect(polled.status()).toBe(200)
        return ((await polled.json()) as { state: string }).state
      },
      { timeout: 20_000, intervals: [300, 500, 1000] },
    )
    .toBe('succeeded')

  const word = page.locator('.opblock').filter({
    has: page.locator('.opblock-summary-path[data-path="/v1/conversions/{job_id}/artifacts/diagnostic_docx"]'),
  })
  await word.locator('.opblock-summary').click()
  await word.getByRole('button', { name: 'Try it out', exact: true }).click()
  await word.getByPlaceholder('job_id', { exact: true }).fill(job.job_id)
  await word.getByRole('button', { name: 'Execute', exact: true }).click()
  const downloadEvent = page.waitForEvent('download')
  await word.getByRole('link', { name: 'Download file', exact: true }).click()
  const download = await downloadEvent
  const output = testInfo.outputPath('swagger-result.docx')
  await download.saveAs(output)
  expect((await readFile(output)).subarray(0, 2).toString()).toBe('PK')
  expect(external).toEqual([])
  await page.reload()
  await page.getByRole('button', { name: 'Authorize', exact: true }).click()
  await expect(page.locator('#auth-bearer-value')).toHaveValue('')
})

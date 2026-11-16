import { test as cspTest, expect } from './fixtures/csp-guard'
import { ensureAuthenticated } from './fixtures/auth'
import { MODELS_PORTS } from './fixtures/ports'

// T46: against a dedicated `models` backend PER PROJECT (`MODELS_PORTS`,
// unlike the shared `app`/`session-expiry` ones) -- this spec mutates the
// same fixed three variant ids (download/load/set-default state), which two
// projects racing one shared backend would corrupt for each other. The
// `baseURL` fixture below picks the right port for whichever project is
// currently running, the same way `WIZARD_PORTS` does per-project via
// `projects[].use.baseURL` in `playwright.config.ts` -- done as a fixture
// override here instead, since a project's own `use.baseURL` is already
// spoken for by the wizard port.
function portForProject(projectName: string): number {
  if (projectName in MODELS_PORTS) {
    return MODELS_PORTS[projectName as keyof typeof MODELS_PORTS]
  }
  throw new Error(`models.spec.ts: no MODELS_PORTS entry for project ${projectName}`)
}

// Playwright's own fixture-override shape (a function calling `use(...)`)
// misfires both `no-empty-pattern` and `react-hooks/rules-of-hooks` -- same
// as `fixtures/csp-guard.ts`'s own `cspGuard` fixture.
/* eslint-disable no-empty-pattern, react-hooks/rules-of-hooks */
const test = cspTest.extend<{ modelsPort: number }>({
  modelsPort: async ({}, use, testInfo) => {
    await use(portForProject(testInfo.project.name))
  },
  baseURL: async ({ modelsPort }, use) => {
    await use(`http://127.0.0.1:${String(modelsPort)}`)
  },
})
/* eslint-enable no-empty-pattern, react-hooks/rules-of-hooks */

const HF_LABEL = 'FP8 (Hugging Face)'

test.describe('models: download, load, default, delete refusal, engine badge', () => {
  test.beforeAll(({ browserName }) => {
    test.skip(
      browserName === 'webkit',
      'WebKit cannot launch on this host (missing system libs, no sudo); deferred to CP8 on a root-capable host.',
    )
  })

  test.beforeEach(async ({ page, modelsPort }) => {
    await ensureAuthenticated(page, modelsPort)
  })

  test('hf variant: download -> load -> set default, then a resident delete is refused', async ({ page }) => {
    await page.goto('/models')

    const hfCard = page.locator('main').locator('.rounded-md.border').filter({ hasText: HF_LABEL })
    await expect(hfCard).toBeVisible()

    // 1. Download: fake_worker (T46) reports 10 steps of growing bytes (1 B
    // .. 10 B out of 10 B, ~1s total) before marking the command succeeded
    // and materializing the fake snapshot on disk (`_materialize_hf_snapshot`).
    // `useModelsQuery`'s own poll cadence (`api/queries.ts`: 2s while a
    // download is active, 15s while idle) is coarser than that ~1s window,
    // so the intermediate byte-progress text is not reliably observable here
    // -- this only asserts the state Download leads to, not every step.
    await hfCard.getByRole('button', { name: 'Download' }).click()
    await expect(hfCard.getByRole('button', { name: 'Load onto GPU' })).toBeVisible({ timeout: 20_000 })

    // 2. Load.
    await hfCard.getByRole('button', { name: 'Load onto GPU' }).click()
    const loadDialog = page.getByRole('alertdialog')
    await expect(loadDialog).toBeVisible()
    await loadDialog.getByRole('button', { name: 'Load onto GPU' }).click()
    await expect(loadDialog).toBeHidden()

    // 3. `fake_worker.run_model_command`'s own 10x100ms progress applies to
    // `load` too; once it succeeds the variant is resident.
    await expect(hfCard.getByText('On GPU')).toBeVisible({ timeout: 20_000 })

    // 4. Set default.
    await hfCard.getByRole('button', { name: 'Set as default' }).click()
    await expect(hfCard.getByText('Default')).toBeVisible({ timeout: 5_000 })

    // 5. Delete of the now-resident variant is refused: `ModelCard.tsx`
    // disables the action client-side with the same reason the API's own
    // `DELETE /v1/models/{variant_id}/download` enforces server-side
    // (`409 MODEL_COMMAND_CONFLICT`, `_models_download_delete`) -- checked
    // directly below so the refusal is proven at the API, not only inferred
    // from a disabled button.
    const deleteButton = hfCard.getByRole('button', { name: 'Delete copy' })
    await expect(deleteButton).toHaveAttribute('aria-disabled', 'true')
    await expect(hfCard.getByText('The model is currently loaded on the GPU; load another one first.')).toBeVisible()

    // `_authorize`'s cookie-session path requires the same `x-unflatten-client: web` CSRF header the SPA itself always sends
    // (design §2.4) -- omitting it is a 403 `WEB_REQUEST_REJECTED`, not the 409 this checks for.
    const directDelete = await page.request.delete('/v1/models/hf-fp8/download', {
      headers: { 'x-unflatten-client': 'web', 'sec-fetch-site': 'none' },
    })
    expect(directDelete.status()).toBe(409)
    const body = (await directDelete.json()) as { code: string }
    expect(body.code).toBe('MODEL_COMMAND_CONFLICT')
  })

  test('catalogue (models_response 1.2): unpublished and planned variants are listed, not downloadable', async ({
    page,
  }, testInfo) => {
    await page.goto('/models')

    const cards = page.locator('main').locator('.rounded-md.border')
    const unpublishedCard = cards.filter({ hasText: 'BF16 (bf16 weights, bf16 KV cache)' })
    await expect(unpublishedCard.getByText('Not yet published on Hugging Face', { exact: true })).toBeVisible()
    await expect(unpublishedCard.getByRole('button', { name: 'Download' })).toHaveAttribute('aria-disabled', 'true')
    await expect(unpublishedCard.getByText(/repository is not public yet/)).toBeVisible()

    const plannedCard = cards.filter({ hasText: 'Unflatten W MoE (future model)' })
    await expect(plannedCard.getByText('In preparation', { exact: true })).toBeVisible()
    await expect(plannedCard.getByRole('button', { name: 'Download' })).toHaveAttribute('aria-disabled', 'true')
    await expect(plannedCard.getByText(/does not exist yet/)).toBeVisible()

    // A disabled Download must also LOOK disabled (muted tokens, not-allowed
    // cursor) while staying focusable so its reason is reachable.
    for (const card of [unpublishedCard, plannedCard]) {
      const downloadButton = card.getByRole('button', { name: 'Download' })
      await expect(downloadButton).toHaveCSS('cursor', 'not-allowed')
      // `aria-disabled`, never the native `disabled` (which would drop focus).
      await expect(downloadButton).not.toHaveAttribute('disabled')
      await downloadButton.focus()
      await expect(downloadButton).toBeFocused()
    }

    // The disabled buttons mirror a server-side refusal, proven at the API.
    for (const [variantId, code] of [
      ['bf16-unpublished', 'MODEL_VARIANT_NOT_PUBLISHED'],
      ['unflatten-w-moe', 'MODEL_VARIANT_PLANNED'],
    ] as const) {
      const response = await page.request.post(`/v1/models/${variantId}/download`, {
        headers: { 'x-unflatten-client': 'web', 'sec-fetch-site': 'none' },
      })
      expect(response.status()).toBe(422)
      expect(((await response.json()) as { code: string }).code).toBe(code)
    }

    await page.screenshot({ path: testInfo.outputPath('models-catalogue.png'), fullPage: true })
  })
})

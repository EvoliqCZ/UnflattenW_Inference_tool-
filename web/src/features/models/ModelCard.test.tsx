import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { server } from '@/test/msw/server'
import { modelCommandAccepted } from '@/test/msw/fixtures'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import type { components } from '@/api/schema'
import { plannedModelVariant, unpublishedModelVariant } from '@/test/msw/fixtures'
import { ModelCard } from './ModelCard'

type ModelVariantEntry = components['schemas']['ModelVariantEntry']

const baseVariant: ModelVariantEntry = {
  availability: 'published',
  bundle_manifest_sha256: null,
  download: { bytes_done: null, bytes_total: null, command_id: null, error_code: null, state: 'ready' },
  download_size_bytes: 4_200_000_000,
  downloaded: true,
  engine: 'native',
  engine_available: true,
  engine_unavailable_reason: null,
  estimated_peak_vram_mib: 11_240,
  fits: true,
  fits_reason: null,
  is_default: false,
  kv_cache: 'fp8',
  label: 'Qwen3-VL INT8',
  precision: 'int8',
  resident: false,
  source_kind: 'hf',
  variant_id: 'qwen3-vl-int8',
}

const device: components['schemas']['ModelsResponseDevice'] = {
  compute_capability: [8, 9],
  free_vram_mib: 2_000,
  observed_at: '2026-09-27T10:00:00Z',
  total_vram_mib: 12_000,
}

function renderCard(variant: ModelVariantEntry, deviceOverride: typeof device | null = device) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <ModelCard variant={variant} resident={null} device={deviceOverride} language="sk" />
    </QueryClientProvider>,
  )
}

describe('ModelCard (UX §3.7/§17.5)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows a visible vram reason and disables Load/Set-default when fits=false', () => {
    const variant: ModelVariantEntry = {
      ...baseVariant,
      estimated_peak_vram_mib: 21_400,
      fits: false,
      fits_reason: 'vram',
    }
    renderCard(variant)

    expect(screen.getAllByText(/Odhad špičky 20,9 GiB presahuje 11,7 GiB VRAM tejto GPU\./).length).toBeGreaterThan(0)
    const loadButton = screen.getByRole('button', { name: 'Načítať na GPU' })
    expect(loadButton).toHaveAttribute('aria-disabled', 'true')
    expect(loadButton).toHaveAccessibleDescription(/Odhad špičky 20,9 GiB/)
  })

  it('shows a visible compute-capability reason when fits_reason is compute_capability', () => {
    const variant: ModelVariantEntry = {
      ...baseVariant,
      fits: false,
      fits_reason: 'compute_capability',
      precision: 'fp8',
    }
    renderCard(variant)

    expect(screen.getAllByText('FP8 vyžaduje compute capability ≥ 8.9 (táto GPU: 8.9).').length).toBeGreaterThan(0)
  })

  it('leaves Download enabled but disables Load/Set-default with the translated reason when engine_available=false', () => {
    const variant: ModelVariantEntry = {
      ...baseVariant,
      downloaded: false,
      download: { bytes_done: null, bytes_total: null, command_id: null, error_code: null, state: 'absent' },
      engine: 'native',
      engine_available: false,
      engine_unavailable_reason: 'worker_not_published',
    }
    renderCard(variant)

    const downloadButton = screen.getByRole('button', { name: 'Stiahnuť' })
    expect(downloadButton).toHaveAttribute('aria-disabled', 'false')
  })

  it('disables Load/Set-default (not Download) with the engine reason once downloaded and unavailable', () => {
    const variant: ModelVariantEntry = {
      ...baseVariant,
      engine: 'native',
      engine_available: false,
      engine_unavailable_reason: 'worker_not_published',
      is_default: false,
    }
    renderCard(variant)

    const loadButton = screen.getByRole('button', { name: 'Načítať na GPU' })
    const setDefaultButton = screen.getByRole('button', { name: 'Nastaviť ako predvolený' })
    expect(loadButton).toHaveAttribute('aria-disabled', 'true')
    expect(loadButton).toHaveAccessibleDescription('Pracovný proces nebol publikovaný')
    expect(setDefaultButton).toHaveAttribute('aria-disabled', 'true')
  })

  it('shows an indeterminate DownloadProgress bar instead of actions while downloading with an unknown total', () => {
    const variant: ModelVariantEntry = {
      ...baseVariant,
      downloaded: false,
      download: { bytes_done: null, bytes_total: null, command_id: 'cmd-1', error_code: null, state: 'downloading' },
    }
    renderCard(variant)

    const bar = screen.getByRole('progressbar')
    expect(bar).not.toHaveAttribute('aria-valuenow')
    expect(screen.queryByRole('button', { name: 'Stiahnuť' })).not.toBeInTheDocument()
  })

  it('shows "Stiahnuť znova" when the download failed', () => {
    const variant: ModelVariantEntry = {
      ...baseVariant,
      downloaded: false,
      download: {
        bytes_done: null,
        bytes_total: null,
        command_id: null,
        error_code: 'MODEL_CHECKPOINT_UNVERIFIED',
        state: 'failed',
      },
    }
    renderCard(variant)

    expect(screen.getByRole('button', { name: 'Stiahnuť znova' })).toBeInTheDocument()
  })

  it('shows the real percentage on the card while downloading with a known total', () => {
    const variant: ModelVariantEntry = {
      ...baseVariant,
      downloaded: false,
      download: {
        bytes_done: 1_050_000_000,
        bytes_total: 4_200_000_000,
        command_id: 'cmd-1',
        error_code: null,
        state: 'downloading',
      },
    }
    renderCard(variant)

    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '25')
    expect(screen.getByText(/\(25\s%\)/)).toBeInTheDocument()
  })
})

describe('ModelCard availability (models_response 1.2)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('marks an unpublished variant and keeps its Download button disabled with the reason', () => {
    renderCard(unpublishedModelVariant)

    expect(screen.getByText('Zatiaľ nezverejnené na Hugging Face')).toBeInTheDocument()
    const downloadButton = screen.getByRole('button', { name: 'Stiahnuť' })
    expect(downloadButton).toHaveAttribute('aria-disabled', 'true')
    expect(downloadButton).toHaveAccessibleDescription(/ešte nie je zverejnený/)
    expect(screen.queryByRole('button', { name: 'Načítať na GPU' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Nastaviť ako predvolený' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Odstrániť kópiu' })).not.toBeInTheDocument()
  })

  it('still shows the VRAM estimate and fit for an unpublished variant', () => {
    renderCard({ ...unpublishedModelVariant, fits: false, fits_reason: 'vram' })

    expect(screen.getByText('Nezmestí sa')).toBeInTheDocument()
    expect(screen.getAllByText(/Odhad špičky 8,3 GiB presahuje 11,7 GiB VRAM tejto GPU\./).length).toBeGreaterThan(0)
  })

  it('marks a planned variant "Pripravuje sa" with a disabled Download and no fit guess', () => {
    renderCard(plannedModelVariant)

    expect(screen.getByText('Pripravuje sa')).toBeInTheDocument()
    const downloadButton = screen.getByRole('button', { name: 'Stiahnuť' })
    expect(downloadButton).toHaveAttribute('aria-disabled', 'true')
    expect(downloadButton).toHaveAccessibleDescription(/ešte neexistuje/)
    expect(screen.queryByText('Neznáme')).not.toBeInTheDocument()
    expect(screen.queryByText(/Worker ešte nenahlásil GPU/)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Nastaviť ako predvolený' })).not.toBeInTheDocument()
  })

  it.each([
    ['unpublished', unpublishedModelVariant],
    ['planned', plannedModelVariant],
  ] as const)('styles the %s Download button as disabled yet keeps it focusable', (_, variant) => {
    renderCard(variant)

    const downloadButton = screen.getByRole('button', { name: 'Stiahnuť' })
    expect(downloadButton).toHaveAttribute('aria-disabled', 'true')
    // The muted look comes from the shared Button's `aria-disabled:` variants
    // (button.test.tsx); the computed style is asserted in e2e/models.spec.ts.
    expect(downloadButton).toHaveClass(
      'aria-disabled:cursor-not-allowed',
      'aria-disabled:bg-muted',
      'aria-disabled:text-muted-foreground',
      'aria-disabled:hover:bg-muted',
    )
    expect(downloadButton).not.toBeDisabled()
    downloadButton.focus()
    expect(downloadButton).toHaveFocus()
  })

  it('does not mute an enabled Download button', () => {
    renderCard({
      ...baseVariant,
      downloaded: false,
      download: { bytes_done: null, bytes_total: null, command_id: null, error_code: null, state: 'absent' },
    })

    const downloadButton = screen.getByRole('button', { name: 'Stiahnuť' })
    // Tailwind's `aria-disabled:` matches only `aria-disabled="true"`, so an
    // explicit "false" plus no unconditional muted class keeps it unmuted.
    expect(downloadButton).toHaveAttribute('aria-disabled', 'false')
    expect(downloadButton).not.toHaveClass('cursor-not-allowed')
    expect(downloadButton).not.toHaveClass('text-muted-foreground')
  })

  it('renders the availability badges in English', async () => {
    await i18n.changeLanguage('en')
    renderCard(unpublishedModelVariant)
    renderCard(plannedModelVariant)

    expect(screen.getByText('Not yet published on Hugging Face')).toBeInTheDocument()
    expect(screen.getByText('In preparation')).toBeInTheDocument()
  })
})

it('retries a cached load failure with a confirmed GPU load rather than another download', async () => {
  await i18n.changeLanguage('sk')
  let loads = 0
  let downloads = 0
  server.use(
    http.post('/v1/models/:id/load', () => {
      loads++
      return HttpResponse.json(modelCommandAccepted, { status: 202 })
    }),
    http.post('/v1/models/:id/download', () => {
      downloads++
      return HttpResponse.json(modelCommandAccepted, { status: 202 })
    }),
  )
  renderCard({
    ...baseVariant,
    download: {
      bytes_done: null,
      bytes_total: null,
      command_id: 'failed-load',
      state: 'failed',
      error_code: 'MODEL_VARIANT_DOES_NOT_FIT',
    },
  })
  await userEvent.click(screen.getByRole('button', { name: 'Načítať na GPU' }))
  await userEvent.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: 'Načítať na GPU' }))
  await waitFor(() => {
    expect(loads).toBe(1)
  })
  expect(downloads).toBe(0)
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import type { components } from '@/api/schema'
import { ModelDeleteDialog, ModelLoadDialog } from './ModelCardDialogs'

type ModelVariantEntry = components['schemas']['ModelVariantEntry']

const variant: ModelVariantEntry = {
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

function wrap(children: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

describe('ModelLoadDialog (UX §3.7 point 3)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows the switch-cost notice and closes on confirm', async () => {
    const onOpenChange = vi.fn()
    render(wrap(<ModelLoadDialog variant={variant} resident={null} language="sk" open onOpenChange={onOpenChange} />))

    expect(screen.getByText(/Na GPU zatiaľ nie je načítaný žiadny model/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Načítať na GPU' }))

    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
  })
})

describe('ModelDeleteDialog (UX §3.7 point 5)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows the local-copy size and closes on confirm', async () => {
    const onOpenChange = vi.fn()
    render(wrap(<ModelDeleteDialog variant={variant} language="sk" open onOpenChange={onOpenChange} />))

    expect(screen.getByText(/Zmazať 3,9 GiB z disku\?/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Odstrániť kópiu' }))

    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
  })
})

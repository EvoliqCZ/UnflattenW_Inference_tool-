import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router'
import i18n from '@/i18n'
import type { components } from '@/api/schema'
import { plannedModelVariant, unpublishedModelVariant } from '@/test/msw/fixtures'
import { ModelPicker } from './ModelPicker'

type ModelsResponse = components['schemas']['ModelsResponse']
type ModelVariantEntry = components['schemas']['ModelVariantEntry']

const int8Variant: ModelVariantEntry = {
  availability: 'published',
  variant_id: 'qwen3-vl-int8',
  label: 'Unflatten W v15b · INT8',
  precision: 'int8',
  kv_cache: 'fp8',
  bundle_manifest_sha256: null,
  estimated_peak_vram_mib: 9100,
  fits: true,
  fits_reason: null,
  resident: true,
  source_kind: 'hf',
  downloaded: true,
  download_size_bytes: 1000,
  is_default: true,
  engine: 'native',
  engine_available: true,
  engine_unavailable_reason: null,
  download: null,
}

const bf16Variant: ModelVariantEntry = {
  availability: 'published',
  variant_id: 'qwen3-vl-bf16',
  label: 'Unflatten W v15b · BF16',
  precision: 'bf16',
  kv_cache: 'bf16',
  bundle_manifest_sha256: null,
  estimated_peak_vram_mib: 21400,
  fits: false,
  fits_reason: 'vram',
  resident: false,
  source_kind: 'hf',
  downloaded: false,
  download_size_bytes: 2000,
  is_default: false,
  engine: 'native',
  engine_available: true,
  engine_unavailable_reason: null,
  download: null,
}

const engineUnavailableVariant: ModelVariantEntry = {
  availability: 'published',
  variant_id: 'qwen3-vl-unavailable',
  label: 'Unflatten W v15b · nedostupný',
  precision: 'int8',
  kv_cache: 'fp8',
  bundle_manifest_sha256: null,
  estimated_peak_vram_mib: 9100,
  fits: true,
  fits_reason: null,
  resident: false,
  source_kind: 'hf',
  downloaded: true,
  download_size_bytes: 1000,
  is_default: false,
  engine: 'native',
  engine_available: false,
  engine_unavailable_reason: 'worker_not_published',
  download: null,
}

const twoVariants: ModelsResponse = {
  schema_version: '1.2',
  default_variant: 'qwen3-vl-int8',
  default_variant_source: 'config',
  device: null,
  resident_variant: null,
  variants: [int8Variant, bf16Variant],
}

function renderPicker(models: ModelsResponse, value: string | null, onChange = vi.fn()) {
  return render(
    <MemoryRouter>
      <ModelPicker models={models} value={value} onChange={onChange} language="sk" />
    </MemoryRouter>,
  )
}

describe('ModelPicker (models_response 1.1)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('lists every variant as a radio option', () => {
    renderPicker(twoVariants, 'qwen3-vl-int8')

    expect(screen.getByRole('radio', { name: /INT8/ })).toBeChecked()
    expect(screen.getByRole('radio', { name: /BF16/ })).toBeInTheDocument()
  })

  it('disables a variant whose fits is false', () => {
    renderPicker(twoVariants, 'qwen3-vl-int8')

    expect(screen.getByRole('radio', { name: /BF16/ })).toBeDisabled()
  })

  it('calls onChange when a selectable variant is picked', async () => {
    const onChange = vi.fn()
    const oneFitting: ModelsResponse = { ...twoVariants, variants: [bf16Variant, int8Variant] }
    renderPicker(oneFitting, 'qwen3-vl-bf16', onChange)

    await userEvent.click(screen.getByRole('radio', { name: /INT8/ }))

    expect(onChange).toHaveBeenCalledWith('qwen3-vl-int8')
  })

  it('collapses to a plain row when only one variant exists', () => {
    const single: ModelsResponse = { ...twoVariants, variants: [int8Variant] }
    renderPicker(single, 'qwen3-vl-int8')

    expect(screen.queryByRole('radio')).not.toBeInTheDocument()
    expect(screen.getByText(/Unflatten W v15b · INT8/)).toBeInTheDocument()
  })

  it('renders nothing when there are no variants', () => {
    const { container } = renderPicker({ ...twoVariants, variants: [] }, null)
    expect(container).toBeEmptyDOMElement()
  })

  it('owner decision D-A (b): shows an unavailable-engine variant, disabled, with its reason visible', () => {
    const withUnavailable: ModelsResponse = { ...twoVariants, variants: [int8Variant, engineUnavailableVariant] }
    renderPicker(withUnavailable, 'qwen3-vl-int8')

    const radio = screen.getByRole('radio', { name: /nedostupný/ })
    expect(radio).toBeInTheDocument()
    expect(radio).toBeDisabled()
    expect(screen.getByText(/Pracovný proces nebol publikovaný/)).toBeInTheDocument()
  })

  it('shows ModelSwitchNotice when the selected variant is not the resident one', () => {
    const withResident: ModelsResponse = { ...twoVariants, resident_variant: 'qwen3-vl-int8' }
    renderPicker(withResident, 'qwen3-vl-bf16')

    expect(screen.getByText(/Na GPU je teraz/)).toBeInTheDocument()
  })

  it('leaves unpublished and planned catalogue variants out of the picker (models_response 1.2)', () => {
    const withCatalogue: ModelsResponse = {
      ...twoVariants,
      variants: [int8Variant, unpublishedModelVariant, plannedModelVariant],
    }
    renderPicker(withCatalogue, 'qwen3-vl-int8')

    // Only one selectable variant remains, so the picker collapses to one row.
    expect(screen.queryByRole('radio')).not.toBeInTheDocument()
    expect(screen.getByText(/Unflatten W v15b · INT8/)).toBeInTheDocument()
    expect(screen.queryByText(/FP8 \(fp8 weights/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Unflatten W MoE/)).not.toBeInTheDocument()
  })

  it('does not show ModelSwitchNotice when the selected variant is already resident', () => {
    const withResident: ModelsResponse = { ...twoVariants, resident_variant: 'qwen3-vl-int8' }
    renderPicker(withResident, 'qwen3-vl-int8')

    expect(screen.queryByText(/Na GPU je teraz/)).not.toBeInTheDocument()
  })
})

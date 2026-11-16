import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import type { components } from '@/api/schema'
import { describeFitsReason, type Translate } from './fits'

type ModelVariantEntry = components['schemas']['ModelVariantEntry']
type ModelsResponseDevice = components['schemas']['ModelsResponseDevice']

const t: Translate = (key, params) => (i18n.t as (key: string, params?: Record<string, string>) => string)(key, params)

const device: ModelsResponseDevice = {
  compute_capability: [8, 9],
  free_vram_mib: 2000,
  observed_at: '2026-09-27T10:00:00Z',
  total_vram_mib: 12_000,
}

const baseVariant: ModelVariantEntry = {
  availability: 'published',
  bundle_manifest_sha256: null,
  download: null,
  download_size_bytes: null,
  downloaded: false,
  engine: 'native',
  engine_available: true,
  engine_unavailable_reason: null,
  estimated_peak_vram_mib: 21_400,
  fits: false,
  fits_reason: 'vram',
  is_default: false,
  kv_cache: 'bf16',
  label: 'Variant',
  precision: 'bf16',
  resident: false,
  source_kind: 'hf',
  variant_id: 'v1',
}

describe('describeFitsReason (UX §3.7, §12 A4)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('returns null when the variant fits', () => {
    expect(describeFitsReason({ ...baseVariant, fits: true, fits_reason: null }, device, 'sk', t)).toBeNull()
  })

  it('returns null when fits is unknown (null)', () => {
    expect(describeFitsReason({ ...baseVariant, fits: null, fits_reason: null }, device, 'sk', t)).toBeNull()
  })

  it('describes a vram overshoot with the estimate and device total', () => {
    const reason = describeFitsReason(baseVariant, device, 'sk', t)
    expect(reason).toBe('Odhad špičky 20,9 GiB presahuje 11,7 GiB VRAM tejto GPU.')
  })

  it('describes a compute-capability shortfall with the device CC', () => {
    const fp8Variant: ModelVariantEntry = { ...baseVariant, fits_reason: 'compute_capability', precision: 'fp8' }
    const reason = describeFitsReason(fp8Variant, { ...device, compute_capability: [7, 5] }, 'sk', t)
    expect(reason).toBe('FP8 vyžaduje compute capability ≥ 8.9 (táto GPU: 7.5).')
  })

  it('falls back to "?" when the device has not reported', () => {
    const reason = describeFitsReason(baseVariant, null, 'sk', t)
    expect(reason).toContain('presahuje ? VRAM')
  })
})

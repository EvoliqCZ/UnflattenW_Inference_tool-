import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import type { components } from '@/api/schema'
import { describeEngineUnavailableReason } from './engineReason'
import type { Translate } from './fits'

type ModelVariantEntry = components['schemas']['ModelVariantEntry']

const t: Translate = (key, params) => (i18n.t as (key: string, params?: Record<string, string>) => string)(key, params)

const baseVariant: ModelVariantEntry = {
  availability: 'published',
  bundle_manifest_sha256: null,
  download: null,
  download_size_bytes: null,
  downloaded: true,
  engine: 'native',
  engine_available: true,
  engine_unavailable_reason: null,
  estimated_peak_vram_mib: 11_240,
  fits: true,
  fits_reason: null,
  is_default: false,
  kv_cache: 'fp8',
  label: 'Variant',
  precision: 'int8',
  resident: false,
  source_kind: 'hf',
  variant_id: 'v1',
}

describe('describeEngineUnavailableReason (UX §17.5/§425)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('returns null when the engine is available', () => {
    expect(describeEngineUnavailableReason(baseVariant, t)).toBeNull()
  })

  it('translates worker_not_published', () => {
    const variant = {
      ...baseVariant,
      engine_available: false,
      engine_unavailable_reason: 'worker_not_published' as const,
    }
    expect(describeEngineUnavailableReason(variant, t)).toBe('Pracovný proces nebol publikovaný')
  })
})

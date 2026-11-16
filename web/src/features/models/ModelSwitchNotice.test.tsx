import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import type { components } from '@/api/schema'
import { ModelSwitchNotice } from './ModelSwitchNotice'

type ModelVariantEntry = components['schemas']['ModelVariantEntry']

const variant: ModelVariantEntry = {
  availability: 'published',
  bundle_manifest_sha256: null,
  download: null,
  download_size_bytes: 4_200_000_000,
  downloaded: false,
  engine: 'native',
  engine_available: true,
  engine_unavailable_reason: null,
  estimated_peak_vram_mib: 11_240,
  fits: true,
  fits_reason: null,
  is_default: false,
  kv_cache: 'fp8',
  label: 'Unflatten W v15b INT8',
  precision: 'int8',
  resident: false,
  source_kind: 'hf',
  variant_id: 'v1',
}

describe('ModelSwitchNotice (UX §3.7 "cena prepnutia")', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('says nothing is loaded yet when resident is null', () => {
    render(<ModelSwitchNotice selected={variant} resident={null} language="sk" />)
    expect(screen.getByText(/Na GPU zatiaľ nie je načítaný žiadny model/)).toBeInTheDocument()
  })

  it('names both variants when the selection would evict a downloaded resident model', () => {
    const resident = { ...variant, variant_id: 'v2', label: 'Iný variant', resident: true }
    render(<ModelSwitchNotice selected={{ ...variant, downloaded: true }} resident={resident} language="sk" />)
    expect(
      screen.getByText(/Na GPU je teraz Iný variant\. Úloha ho uvoľní a načíta Unflatten W v15b INT8/),
    ).toBeInTheDocument()
  })

  it('adds the download-size sentence when the selection is not downloaded yet', () => {
    const resident = { ...variant, variant_id: 'v2', label: 'Iný variant', resident: true }
    render(<ModelSwitchNotice selected={variant} resident={resident} language="sk" />)
    expect(screen.getByText(/Najprv sa stiahne 3,9 GiB/)).toBeInTheDocument()
  })
})

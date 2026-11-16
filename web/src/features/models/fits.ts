import type { components } from '@/api/schema'
import { formatMebibytes } from '@/lib/format'
import type { SupportedLanguage } from '@/i18n'

type ModelVariantEntry = components['schemas']['ModelVariantEntry']
type ModelsResponseDevice = components['schemas']['ModelsResponseDevice']

/**
 * A minimal, loosely-typed stand-in for `useTranslation()`'s `t` -- the
 * real `TFunction` overloads reject a plain `Record<string, string>`
 * interpolation object for a dynamically-chosen key, since `i18next`'s
 * generated types expect each key's exact param shape. Every call site
 * passes `t` from `useTranslation()` directly; this signature is only here
 * to keep this module free of the react-i18next type dependency.
 */
export type Translate = (key: string, params?: Record<string, string>) => string

/**
 * UX §3.7 "nezmestí sa": the general capacity sentence when `fits_reason =
 * 'vram'`, the compute-capability sentence when it is `'compute_capability'`
 * (§12 A4 -- distinguishing the two needs `fits_reason`, present since
 * `models_response` 1.1). `null` when the variant fits or the device has
 * not reported (`device = null`), so there is nothing to compute a reason
 * from.
 */
export function describeFitsReason(
  variant: ModelVariantEntry,
  device: ModelsResponseDevice | null,
  language: SupportedLanguage,
  t: Translate,
): string | null {
  if (variant.fits !== false || variant.fits_reason === null) {
    return null
  }
  if (variant.fits_reason === 'compute_capability') {
    const cc = device !== null ? `${String(device.compute_capability[0])}.${String(device.compute_capability[1])}` : '?'
    return t('models.fits.exceedsComputeCapability', { cc })
  }
  const total = device !== null ? formatMebibytes(device.total_vram_mib, language) : '?'
  const est =
    variant.estimated_peak_vram_mib !== null ? formatMebibytes(variant.estimated_peak_vram_mib, language) : '?'
  return t('models.fits.exceedsVram', { est, total })
}

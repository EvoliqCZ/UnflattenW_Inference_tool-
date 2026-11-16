import type { SupportedLanguage } from '@/i18n'

/** UX §9.3: `sk` renders as `sk-SK`, `en` as `en-GB` (24h time, day-month order). */
const INTL_LOCALE: Record<SupportedLanguage, string> = {
  sk: 'sk-SK',
  en: 'en-GB',
}

function localeFor(language: SupportedLanguage): string {
  return INTL_LOCALE[language]
}

const IEC_UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'] as const

/**
 * Formats a byte count in IEC (binary) units (UX §9.3: "IEC jednotky všade").
 * `1536` → `1,5 KiB` (sk) / `1.5 KiB` (en).
 */
export function formatBytes(bytes: number, language: SupportedLanguage): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    throw new RangeError(`formatBytes: bytes must be a finite, non-negative number, got ${String(bytes)}`)
  }
  let value = bytes
  let unitIndex = 0
  while (value >= 1024 && unitIndex < IEC_UNITS.length - 1) {
    value /= 1024
    unitIndex += 1
  }
  const unit = IEC_UNITS[unitIndex] as string
  const maximumFractionDigits = unitIndex === 0 ? 0 : 1
  const formatted = new Intl.NumberFormat(localeFor(language), { maximumFractionDigits }).format(value)
  return `${formatted} ${unit}`
}

/** Formats a size already expressed in mebibytes (VRAM figures arrive in MiB). */
export function formatMebibytes(mib: number, language: SupportedLanguage): string {
  return formatBytes(mib * 1024 * 1024, language)
}

/** `12345.6` → `12 345,6` (sk, NBSP thousands separator) / `12,345.6` (en). */
export function formatNumber(value: number, language: SupportedLanguage): string {
  return new Intl.NumberFormat(localeFor(language)).format(value)
}

/** `0.5` → `50 %` (sk, NBSP) / `50%` (en). */
export function formatPercent(fraction: number, language: SupportedLanguage): string {
  return new Intl.NumberFormat(localeFor(language), { style: 'percent' }).format(fraction)
}

import type { SupportedLanguage } from '@/i18n'

/** UX §9.3: `sk` renders as `sk-SK`, `en` as `en-GB` (24h time, day-month order). */
const INTL_LOCALE: Record<SupportedLanguage, string> = {
  sk: 'sk-SK',
  en: 'en-GB',
}

/** UX §9.3: below 1 minute, both directions collapse to a fixed "just now" phrase. */
const JUST_NOW: Record<SupportedLanguage, string> = {
  sk: 'práve teraz',
  en: 'just now',
}

const RELATIVE_UNITS: readonly [Intl.RelativeTimeFormatUnit, number][] = [
  ['day', 86400],
  ['hour', 3600],
  ['minute', 60],
]

/** `27. 9. 2026 14:05` (sk) / `27 Sept 2026, 14:05` (en). */
export function formatDateTime(date: Date, language: SupportedLanguage): string {
  return new Intl.DateTimeFormat(INTL_LOCALE[language], { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

/**
 * `pred 5 minútami` / `5 minutes ago`, `o 23 hodín` / `in 23 hours`.
 * Anything under 60 seconds either way is "práve teraz" / "just now" (UX §9.3).
 */
export function formatRelativeTime(date: Date, now: Date, language: SupportedLanguage): string {
  const diffSeconds = (date.getTime() - now.getTime()) / 1000
  if (Math.abs(diffSeconds) < 60) {
    return JUST_NOW[language]
  }
  const rtf = new Intl.RelativeTimeFormat(INTL_LOCALE[language], { numeric: 'auto' })
  for (const [unit, unitSeconds] of RELATIVE_UNITS) {
    if (Math.abs(diffSeconds) >= unitSeconds) {
      return rtf.format(Math.round(diffSeconds / unitSeconds), unit)
    }
  }
  return rtf.format(Math.round(diffSeconds), 'second')
}

/**
 * `1 min 12 s` / `1 min 12 s` for 60s and over, `14,2 s` (sk) / `14.2 s` (en) below.
 */
export function formatDuration(seconds: number, language: SupportedLanguage): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    throw new RangeError(`formatDuration: seconds must be a finite, non-negative number, got ${String(seconds)}`)
  }
  const locale = INTL_LOCALE[language]
  if (seconds < 60) {
    const value = new Intl.NumberFormat(locale, { maximumFractionDigits: 1 }).format(seconds)
    return `${value} s`
  }
  const wholeSeconds = Math.round(seconds)
  const minutes = Math.floor(wholeSeconds / 60)
  const remainderSeconds = wholeSeconds % 60
  const minuteValue = new Intl.NumberFormat(locale).format(minutes)
  const secondValue = new Intl.NumberFormat(locale).format(remainderSeconds)
  return `${minuteValue} min ${secondValue} s`
}

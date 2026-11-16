import type { SupportedLanguage } from '@/i18n'
import { formatBytes } from '@/lib/format'
import type { SelectionValidationError } from './validateSelection'

/**
 * `validateSelection` reports byte counts as raw numbers (UX §4.3); the
 * `validation.tooLarge` string interpolates IEC-formatted sizes (UX §9.3),
 * so the byte->text formatting happens here, once, for the i18n call site.
 */
export function selectionErrorParams(
  error: SelectionValidationError,
  language: SupportedLanguage,
): Record<string, string | number> {
  const params = error.params ?? {}
  if (error.code === 'tooLarge') {
    return {
      size: formatBytes(Number(params.size), language),
      max: formatBytes(Number(params.max), language),
    }
  }
  return params
}

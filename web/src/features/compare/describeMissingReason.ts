import type { TFunction } from 'i18next'
import { PROBLEM_CODES, type ProblemCode } from '@/i18n/codes'
import type { MissingReason } from './types'

function isProblemCode(code: string): code is ProblemCode {
  return (PROBLEM_CODES as readonly string[]).includes(code)
}

/** Translates `t` into the exact key/params for the reason (UX §5.6). Split out from `MissingPreview` so the text mapping is unit-testable without rendering. */
export function describeMissingReason(t: TFunction, reason: MissingReason): string {
  switch (reason.kind) {
    case 'pageFailed': {
      // Narrowed through a local binding, not the `reason.errorCode` member
      // expression directly -- `isProblemCode` only narrows a plain variable.
      const { errorCode } = reason
      const error = isProblemCode(errorCode) ? t(`codes.problem.${errorCode}.title`) : errorCode
      return t('compare.missing.pageFailed', { index: reason.index + 1, error })
    }
    case 'previewFailed':
      return t('compare.missing.previewFailed')
    case 'docxNotProduced':
      return t('compare.missing.docxNotProduced', {
        reasons: reason.reasonCodes.map((code) => t(`codes.reason.${code}`)).join(', '),
      })
    case 'noSource':
      return t('compare.missing.noSource', { index: reason.index + 1 })
    case 'outOfRange':
      return reason.side === 'docx'
        ? t('compare.missing.outOfRangeDocx', { index: reason.index + 1 })
        : t('compare.missing.outOfRangeSource', { index: reason.index + 1 })
    case 'loadFailed':
      return t('compare.missing.loadFailed')
  }
}

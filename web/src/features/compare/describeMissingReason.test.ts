import type { TFunction } from 'i18next'
import { describe, expect, it } from 'vitest'
import { describeMissingReason } from './describeMissingReason'

/** A fake `t` for exercising the key/param mapping without loading real i18next resources. */
function fakeT(impl: (key: string, params?: Record<string, unknown>) => string): TFunction {
  return impl as unknown as TFunction
}

const echoT = fakeT((key, params) => `${key}:${JSON.stringify(params)}`)

describe('describeMissingReason', () => {
  it('should resolve a known error code to its translated title for pageFailed', () => {
    const text = describeMissingReason(echoT, { kind: 'pageFailed', index: 2, errorCode: 'RENDERER_FAILED' })

    expect(text).toBe('compare.missing.pageFailed:{"index":3,"error":"codes.problem.RENDERER_FAILED.title:undefined"}')
  })

  it('should fall back to the raw code for pageFailed when it is not a known problem code', () => {
    const text = describeMissingReason(echoT, { kind: 'pageFailed', index: 0, errorCode: 'SOME_UNKNOWN_CODE' })

    expect(text).toContain('"error":"SOME_UNKNOWN_CODE"')
  })

  it('should join translated reason codes for docxNotProduced', () => {
    const t = fakeT((key, params) =>
      key.startsWith('codes.reason.') ? key.replace('codes.reason.', '') : `${key}:${JSON.stringify(params)}`,
    )

    const text = describeMissingReason(t, {
      kind: 'docxNotProduced',
      reasonCodes: ['DOCX_AST_NOT_STRICT', 'DOCX_CHAIN_ERROR'],
    })

    expect(text).toBe('compare.missing.docxNotProduced:{"reasons":"DOCX_AST_NOT_STRICT, DOCX_CHAIN_ERROR"}')
  })

  it('should pick the docx-side vs source-side text for outOfRange', () => {
    expect(describeMissingReason(echoT, { kind: 'outOfRange', index: 4, side: 'docx' })).toBe(
      'compare.missing.outOfRangeDocx:{"index":5}',
    )
    expect(describeMissingReason(echoT, { kind: 'outOfRange', index: 4, side: 'source' })).toBe(
      'compare.missing.outOfRangeSource:{"index":5}',
    )
  })

  it('should convert 0-based index to 1-based for noSource', () => {
    expect(describeMissingReason(echoT, { kind: 'noSource', index: 0 })).toBe('compare.missing.noSource:{"index":1}')
  })

  it('should need no params for previewFailed and loadFailed', () => {
    expect(describeMissingReason(echoT, { kind: 'previewFailed' })).toBe('compare.missing.previewFailed:undefined')
    expect(describeMissingReason(echoT, { kind: 'loadFailed' })).toBe('compare.missing.loadFailed:undefined')
  })
})

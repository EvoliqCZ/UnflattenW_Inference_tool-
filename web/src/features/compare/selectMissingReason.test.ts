import { describe, expect, it } from 'vitest'
import type { DiagnosticOutcome, PageReport } from './types'
import { selectMissingReason } from './selectMissingReason'

function outcome(overrides: Partial<DiagnosticOutcome> = {}): DiagnosticOutcome {
  return {
    canonical_ast_produced: true,
    claim_scope: 'diagnostic',
    docx_chain: null,
    docx_produced: true,
    prediction_ir_status: 'sealed',
    reason_codes: [],
    ...overrides,
  }
}

describe('selectMissingReason', () => {
  it('should report noSource when the source pane is missing but within range', () => {
    const reason = selectMissingReason({ side: 'source', pageIndex: 0, sourceCount: 2, pages: null, outcome: null })

    expect(reason).toEqual({ kind: 'noSource', index: 0 })
  })

  it('should report outOfRange for the source side once past sourceCount', () => {
    const reason = selectMissingReason({ side: 'source', pageIndex: 2, sourceCount: 2, pages: null, outcome: null })

    expect(reason).toEqual({ kind: 'outOfRange', index: 2, side: 'source' })
  })

  it('should prefer pageFailed when pages[] marks the index as failed (UX §5.6 priority 1)', () => {
    const pages: PageReport[] = [
      { decode_steps: 3, error_code: 'RENDERER_FAILED', index: 1, seconds: 1.2, state: 'failed' },
    ]

    const reason = selectMissingReason({
      side: 'docx',
      pageIndex: 1,
      sourceCount: 2,
      pages,
      outcome: outcome({ docx_produced: false, reason_codes: ['DOCX_CHAIN_ERROR'] }),
    })

    expect(reason).toEqual({ kind: 'pageFailed', index: 1, errorCode: 'RENDERER_FAILED' })
  })

  it('should report previewFailed from reason_codes ahead of docxNotProduced (priority 2 before 3)', () => {
    const reason = selectMissingReason({
      side: 'docx',
      pageIndex: 0,
      sourceCount: 1,
      pages: null,
      outcome: outcome({ docx_produced: false, reason_codes: ['PREVIEW_RENDER_FAILED'] }),
    })

    expect(reason).toEqual({ kind: 'previewFailed' })
  })

  it('should report previewFailed from docx_chain.failure_codes too', () => {
    const reason = selectMissingReason({
      side: 'docx',
      pageIndex: 0,
      sourceCount: 1,
      pages: null,
      outcome: outcome({
        docx_chain: {
          failure_codes: ['PREVIEW_RENDER_FAILED'],
          page_outcome: 'failed',
          policy_id: 'headless-defaults.v0',
          policy_status: 'RATIFIED',
        },
      }),
    })

    expect(reason).toEqual({ kind: 'previewFailed' })
  })

  it('should report docxNotProduced with the reason codes when docx_produced is false', () => {
    const reason = selectMissingReason({
      side: 'docx',
      pageIndex: 0,
      sourceCount: 1,
      pages: null,
      outcome: outcome({ docx_produced: false, reason_codes: ['DOCX_AST_NOT_STRICT'] }),
    })

    expect(reason).toEqual({ kind: 'docxNotProduced', reasonCodes: ['DOCX_AST_NOT_STRICT'] })
  })

  it('should fall back to outOfRange for the docx side when nothing else explains the gap', () => {
    const reason = selectMissingReason({ side: 'docx', pageIndex: 5, sourceCount: 6, pages: null, outcome: outcome() })

    expect(reason).toEqual({ kind: 'outOfRange', index: 5, side: 'docx' })
  })
})

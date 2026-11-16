import type { DiagnosticOutcome, MissingReason, PageReport } from './types'

export interface SelectMissingReasonInput {
  side: 'source' | 'docx'
  /** 0-based. */
  pageIndex: number
  sourceCount: number
  pages: PageReport[] | null
  outcome: DiagnosticOutcome | null
}

/**
 * UX §5.6 priority table, restricted to what `ComparisonViewer`'s own props
 * (design §7.3: `artifacts`, `outcome`, `pages`) can decide up front. Called
 * only for a pane that has no url at `pageIndex` -- `loadFailed` (a runtime
 * `<img>` failure) and `expired` (a job-level 410, not part of this
 * component's props) are decided elsewhere, not here.
 */
export function selectMissingReason({
  side,
  pageIndex,
  sourceCount,
  pages,
  outcome,
}: SelectMissingReasonInput): MissingReason {
  if (side === 'source') {
    return pageIndex >= sourceCount
      ? { kind: 'outOfRange', index: pageIndex, side: 'source' }
      : { kind: 'noSource', index: pageIndex }
  }

  const pageReport = pages?.find((page) => page.index === pageIndex)
  if (pageReport?.state === 'failed') {
    return { kind: 'pageFailed', index: pageIndex, errorCode: pageReport.error_code ?? 'INTERNAL_ERROR' }
  }

  const previewRenderFailed =
    (outcome?.reason_codes.includes('PREVIEW_RENDER_FAILED') ?? false) ||
    (outcome?.docx_chain?.failure_codes?.includes('PREVIEW_RENDER_FAILED') ?? false)
  if (previewRenderFailed) {
    return { kind: 'previewFailed' }
  }

  if (outcome?.docx_produced === false) {
    return { kind: 'docxNotProduced', reasonCodes: outcome.reason_codes }
  }

  return { kind: 'outOfRange', index: pageIndex, side: 'docx' }
}

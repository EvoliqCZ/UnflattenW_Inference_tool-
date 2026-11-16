import { describe, expect, it } from 'vitest'
import type { ArtifactEntry } from './types'
import { buildPageSequences } from './buildPageSequences'

function entry(overrides: Partial<ArtifactEntry> = {}): ArtifactEntry {
  return {
    page_index: 0,
    role: 'source_preview',
    url: '/v1/conversions/job-1/artifacts/source_preview/0',
    ...overrides,
  }
}

describe('buildPageSequences', () => {
  it('should split artifacts into independent source and docx sequences keyed by page_index', () => {
    const result = buildPageSequences([
      entry({ page_index: 0, role: 'source_preview', url: 's0' }),
      entry({ page_index: 1, role: 'source_preview', url: 's1' }),
      entry({ page_index: 0, role: 'docx_preview', url: 'd0' }),
    ])

    expect(result.source.get(0)).toBe('s0')
    expect(result.source.get(1)).toBe('s1')
    expect(result.docx.get(0)).toBe('d0')
    expect(result.docx.get(1)).toBeUndefined()
    expect(result.sourceCount).toBe(2)
    expect(result.docxCount).toBe(1)
  })

  it('should ignore roles other than source_preview/docx_preview and entries with a null page_index', () => {
    const result = buildPageSequences([
      entry({ page_index: null, role: 'diagnostic_docx', url: 'd' }),
      entry({ page_index: null, role: 'prediction_ir', url: 'ir' }),
    ])

    expect(result.source.size).toBe(0)
    expect(result.docx.size).toBe(0)
  })

  it('should return empty sequences for no artifacts', () => {
    const result = buildPageSequences([])

    expect(result.sourceCount).toBe(0)
    expect(result.docxCount).toBe(0)
  })
})

import { describe, expect, it } from 'vitest'
import type { components } from '@/api/schema'
import { estimateRemainingMinutes } from './pageProgress'

type PageReport = components['schemas']['PageReport']

function page(overrides: Partial<PageReport>): PageReport {
  return { index: 0, state: 'succeeded', error_code: null, decode_steps: 100, seconds: 10, ...overrides }
}

describe('estimateRemainingMinutes (UX §3.4)', () => {
  it('returns null with fewer than 2 finished pages', () => {
    const pages = [page({ index: 0, seconds: 30 })]
    expect(estimateRemainingMinutes(pages, 1, 5)).toBeNull()
  })

  it('returns null when pages is null', () => {
    expect(estimateRemainingMinutes(null, 3, 5)).toBeNull()
  })

  it('estimates from the median of finished pages once at least 2 have finished', () => {
    const pages = [page({ index: 0, seconds: 30 }), page({ index: 1, seconds: 60 })]
    // median 45s * 3 remaining pages = 135s = ~2 min (rounded)
    expect(estimateRemainingMinutes(pages, 2, 5)).toBe(2)
  })

  it('ignores failed pages when computing the median', () => {
    const pages = [
      page({ index: 0, seconds: 30 }),
      page({ index: 1, seconds: 60 }),
      page({ index: 2, state: 'failed', error_code: 'MODEL_DECODE_MAX_STEPS', seconds: 5, decode_steps: null }),
    ]
    expect(estimateRemainingMinutes(pages, 3, 5)).toBe(2)
  })

  it('never returns a negative estimate for a fully-done job', () => {
    const pages = [page({ index: 0, seconds: 30 }), page({ index: 1, seconds: 30 })]
    expect(estimateRemainingMinutes(pages, 2, 2)).toBe(0)
  })
})

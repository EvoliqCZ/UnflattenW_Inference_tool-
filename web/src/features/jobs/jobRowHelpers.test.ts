import { describe, expect, it } from 'vitest'
import { formatPagesColumn, isJobActive, jobDisplayName } from './jobRowHelpers'

describe('isJobActive', () => {
  it('is true only for queued and running', () => {
    expect(isJobActive('queued')).toBe(true)
    expect(isJobActive('running')).toBe(true)
    expect(isJobActive('succeeded')).toBe(false)
    expect(isJobActive('failed')).toBe(false)
    expect(isJobActive('cancelled')).toBe(false)
    expect(isJobActive('expired')).toBe(false)
  })
})

describe('formatPagesColumn (UX §6 "Strany")', () => {
  it('shows done/total while active', () => {
    expect(formatPagesColumn({ state: 'running', pages_done: 1, pages_total: 3 })).toBe('1 / 3')
  })

  it('defaults done to 0 while active if not yet reported', () => {
    expect(formatPagesColumn({ state: 'queued', pages_done: null, pages_total: 3 })).toBe('0 / 3')
  })

  it('shows just the total once terminal', () => {
    expect(formatPagesColumn({ state: 'succeeded', pages_done: 3, pages_total: 3 })).toBe('3')
  })

  it('shows an em dash when pages_total is unknown', () => {
    expect(formatPagesColumn({ state: 'queued', pages_done: null, pages_total: null })).toBe('—')
  })
})

describe('jobDisplayName', () => {
  it('prefers the label, falling back to the job id', () => {
    expect(jobDisplayName({ label: 'Faktúra', job_id: 'job-1' })).toBe('Faktúra')
    expect(jobDisplayName({ label: null, job_id: 'job-1' })).toBe('job-1')
  })
})

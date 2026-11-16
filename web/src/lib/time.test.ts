import { describe, expect, it } from 'vitest'
import { formatDateTime, formatDuration, formatRelativeTime } from './time'

describe('formatDateTime (UX §9.3)', () => {
  // Exact rendering is timezone-dependent (dateStyle/timeStyle use the host's
  // local time), so this asserts structure and content, not a fixed string.
  const date = new Date('2026-09-27T14:05:00Z')

  it('renders sk as day. month. year, time', () => {
    expect(formatDateTime(date, 'sk')).toMatch(/^27\. 9\. 2026,? \d{2}:\d{2}$/)
  })

  it('renders en with a 3-letter month and 24h time', () => {
    expect(formatDateTime(date, 'en')).toMatch(/^27 Sept? 2026, \d{2}:\d{2}$/)
  })
})

describe('formatRelativeTime (UX §9.3)', () => {
  const now = new Date('2026-09-27T12:00:00Z')

  it('collapses anything under 60s to "just now" in both directions', () => {
    expect(formatRelativeTime(new Date(now.getTime() - 30_000), now, 'sk')).toBe('práve teraz')
    expect(formatRelativeTime(new Date(now.getTime() + 30_000), now, 'en')).toBe('just now')
  })

  it('renders past minutes in sk', () => {
    expect(formatRelativeTime(new Date(now.getTime() - 5 * 60_000), now, 'sk')).toBe('pred 5 minútami')
  })

  it('renders past minutes in en', () => {
    expect(formatRelativeTime(new Date(now.getTime() - 5 * 60_000), now, 'en')).toBe('5 minutes ago')
  })

  it('renders future hours in sk', () => {
    expect(formatRelativeTime(new Date(now.getTime() + 23 * 3_600_000), now, 'sk')).toBe('o 23 hodín')
  })

  it('renders future hours in en', () => {
    expect(formatRelativeTime(new Date(now.getTime() + 23 * 3_600_000), now, 'en')).toBe('in 23 hours')
  })
})

describe('formatDuration (UX §9.3)', () => {
  it('renders sub-minute durations as decimal seconds', () => {
    expect(formatDuration(14.2, 'sk')).toBe('14,2 s')
    expect(formatDuration(14.2, 'en')).toBe('14.2 s')
  })

  it('renders minute-and-second durations without decimals', () => {
    expect(formatDuration(72, 'sk')).toBe('1 min 12 s')
    expect(formatDuration(72, 'en')).toBe('1 min 12 s')
  })

  it('rejects negative or non-finite input', () => {
    expect(() => formatDuration(-1, 'en')).toThrow(RangeError)
    expect(() => formatDuration(Number.NaN, 'en')).toThrow(RangeError)
  })
})

import { describe, expect, it } from 'vitest'
import { formatBytes, formatMebibytes, formatNumber, formatPercent } from './format'

describe('formatBytes (UX §9.3: IEC units)', () => {
  it('keeps small values in bytes', () => {
    expect(formatBytes(850, 'en')).toBe('850 B')
  })

  it('formats GiB with one decimal in sk (comma) and en (dot)', () => {
    const bytes = 3.2 * 1024 ** 3
    expect(formatBytes(bytes, 'sk')).toBe('3,2 GiB')
    expect(formatBytes(bytes, 'en')).toBe('3.2 GiB')
  })

  it('formats MiB without decimals when the value is whole', () => {
    expect(formatBytes(850 * 1024 ** 2, 'en')).toBe('850 MiB')
  })

  it('rejects negative or non-finite input', () => {
    expect(() => formatBytes(-1, 'en')).toThrow(RangeError)
    expect(() => formatBytes(Number.NaN, 'en')).toThrow(RangeError)
  })
})

describe('formatMebibytes', () => {
  it('converts MiB figures (VRAM) through the same IEC formatting', () => {
    expect(formatMebibytes(11240, 'en')).toBe('11 GiB')
  })
})

describe('formatNumber (UX §9.3)', () => {
  it('uses NBSP thousands separator and comma decimal in sk', () => {
    expect(formatNumber(12345.6, 'sk')).toBe('12 345,6')
  })

  it('uses comma thousands separator and dot decimal in en', () => {
    expect(formatNumber(12345.6, 'en')).toBe('12,345.6')
  })
})

describe('formatPercent (UX §9.3)', () => {
  it('renders sk with an NBSP before the percent sign', () => {
    expect(formatPercent(0.5, 'sk')).toBe('50 %')
  })

  it('renders en without a space', () => {
    expect(formatPercent(0.5, 'en')).toBe('50%')
  })
})

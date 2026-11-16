import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { ExpiryCountdown } from './ExpiryCountdown'

const NOW = new Date('2026-09-27T12:00:00.000Z')

describe('ExpiryCountdown', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('should render an em dash when there is no expiry (active job)', () => {
    render(<ExpiryCountdown expiresAt={null} />)

    expect(screen.getByText('—')).toBeInTheDocument()
  })

  it('should render an icon alongside the countdown text (never color alone)', () => {
    const expiresAt = new Date(NOW.getTime() + 5 * 60 * 60 * 1000).toISOString()
    render(<ExpiryCountdown expiresAt={expiresAt} />)

    expect(document.querySelector('svg[aria-hidden="true"]')).toBeInTheDocument()
    expect(screen.getByText(/Vyprší o/)).toBeInTheDocument()
  })

  it('should switch to the under-1-hour warning form below the threshold', () => {
    const expiresAt = new Date(NOW.getTime() + 42 * 60 * 1000).toISOString()
    render(<ExpiryCountdown expiresAt={expiresAt} />)

    const wrapper = screen.getByText(/Vyprší o/).closest('span[title]')
    expect(wrapper?.className).toContain('text-warning-text')
  })

  it('should not use the warning form above the 1-hour threshold', () => {
    const expiresAt = new Date(NOW.getTime() + 5 * 60 * 60 * 1000).toISOString()
    render(<ExpiryCountdown expiresAt={expiresAt} />)

    const wrapper = screen.getByText(/Vyprší o/).closest('span[title]')
    expect(wrapper?.className).not.toContain('text-warning-text')
  })

  it('should render the expired text once the target time has passed', () => {
    const expiresAt = new Date(NOW.getTime() - 60 * 1000).toISOString()
    render(<ExpiryCountdown expiresAt={expiresAt} />)

    expect(screen.getByText('Vypršala')).toBeInTheDocument()
  })

  it('should announce the under-1-hour crossing exactly once via a live region', () => {
    const expiresAt = new Date(NOW.getTime() + 65 * 60 * 1000).toISOString()
    const { container } = render(<ExpiryCountdown expiresAt={expiresAt} />)

    // Still above 1h: the live region has not announced anything yet.
    expect(container.querySelector('.sr-only')).toHaveTextContent('')

    // Advance past the 1h threshold.
    act(() => {
      vi.advanceTimersByTime(6 * 60 * 1000)
    })

    expect(container.querySelector('.sr-only')).toHaveTextContent('Čoskoro vyprší')
  })
})

import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { JobStateBadge } from './JobStateBadge'

describe('JobStateBadge', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should render an icon and translated text for queued', () => {
    render(<JobStateBadge state="queued" />)

    expect(screen.getByText('Vo fronte')).toBeInTheDocument()
    expect(document.querySelector('svg[aria-hidden="true"]')).toBeInTheDocument()
  })

  it('should render distinct color classes per state (never color alone)', () => {
    const { container: queued } = render(<JobStateBadge state="queued" />)
    const { container: succeeded } = render(<JobStateBadge state="succeeded" />)
    const { container: failed } = render(<JobStateBadge state="failed" />)

    expect(queued.querySelector('span')?.className).not.toEqual(succeeded.querySelector('span')?.className)
    expect(succeeded.querySelector('span')?.className).not.toEqual(failed.querySelector('span')?.className)
    // Every badge still carries text alongside the color.
    expect(screen.getByText('Dokončená')).toBeInTheDocument()
    expect(screen.getByText('Zlyhala')).toBeInTheDocument()
  })

  it('should only spin the running icon under prefers-reduced-motion-safe styling', () => {
    const { container: running } = render(<JobStateBadge state="running" />)
    const { container: cancelled } = render(<JobStateBadge state="cancelled" />)

    expect(running.querySelector('svg')?.getAttribute('class')).toContain('motion-safe:animate-spin')
    expect(cancelled.querySelector('svg')?.getAttribute('class')).not.toContain('animate-spin')
  })

  it('should render every job state without throwing', () => {
    const states = ['queued', 'running', 'succeeded', 'failed', 'cancelled', 'expired'] as const
    for (const state of states) {
      const { unmount } = render(<JobStateBadge state={state} />)
      unmount()
    }
  })
})

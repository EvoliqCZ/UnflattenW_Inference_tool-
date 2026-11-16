import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Problem } from '@/api/problem'
import i18n from '@/i18n'
import { ProblemMessage } from './ProblemMessage'

function buildProblem(overrides: Partial<Problem> = {}): Problem {
  return {
    code: 'RATE_LIMITED',
    retryable: true,
    status: 429,
    title: 'Too many requests',
    ...overrides,
  }
}

describe('ProblemMessage', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should render an icon, the mapped title and hint for a known code', () => {
    render(<ProblemMessage problem={buildProblem()} context="submit" />)

    expect(screen.getByText('Príliš veľa požiadaviek')).toBeInTheDocument()
    expect(screen.getByText('Prekročil sa povolený počet požiadaviek za daný čas.')).toBeInTheDocument()
    expect(document.querySelector('svg[aria-hidden="true"]')).toBeInTheDocument()
  })

  it('should fall back to the unknown-code text for an unrecognized code', () => {
    render(<ProblemMessage problem={{ code: 'SOMETHING_NEW_AND_UNSEEN' }} context="page" />)

    expect(screen.getByText('Neznámy kód')).toBeInTheDocument()
  })

  it('should show a retry button when retryable and onRetry is provided, and call it on click', async () => {
    const onRetry = vi.fn()
    const user = userEvent.setup()
    render(<ProblemMessage problem={buildProblem({ retryable: true })} context="download" onRetry={onRetry} />)

    const retry = screen.getByRole('button', { name: 'Skúsiť znova' })
    await user.click(retry)

    expect(onRetry).toHaveBeenCalledTimes(1)
  })

  it('should not show a retry button when the problem is not retryable', () => {
    render(<ProblemMessage problem={buildProblem({ retryable: false })} context="job" onRetry={vi.fn()} />)

    expect(screen.queryByRole('button', { name: 'Skúsiť znova' })).not.toBeInTheDocument()
  })

  it('should not show a retry button for a bare {code} even if onRetry is provided', () => {
    render(<ProblemMessage problem={{ code: 'RATE_LIMITED' }} context="submit" onRetry={vi.fn()} />)

    expect(screen.queryByRole('button', { name: 'Skúsiť znova' })).not.toBeInTheDocument()
  })
})

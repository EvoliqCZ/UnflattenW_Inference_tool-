import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { PageNavigator } from './PageNavigator'

describe('PageNavigator', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should announce the current page', () => {
    render(<PageNavigator index={2} total={5} onNavigate={vi.fn()} />)

    expect(screen.getByText('Strana 3 z 5')).toBeInTheDocument()
  })

  it('should disable prev at the first page and next at the last page', () => {
    const { rerender } = render(<PageNavigator index={0} total={3} onNavigate={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Späť' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Ďalej' })).not.toBeDisabled()

    rerender(<PageNavigator index={2} total={3} onNavigate={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Späť' })).not.toBeDisabled()
    expect(screen.getByRole('button', { name: 'Ďalej' })).toBeDisabled()
  })

  it('should call onNavigate with the next/previous 0-based index on click', async () => {
    const onNavigate = vi.fn()
    const user = userEvent.setup()
    render(<PageNavigator index={1} total={5} onNavigate={onNavigate} />)

    await user.click(screen.getByRole('button', { name: 'Ďalej' }))
    expect(onNavigate).toHaveBeenCalledWith(2)

    await user.click(screen.getByRole('button', { name: 'Späť' }))
    expect(onNavigate).toHaveBeenCalledWith(0)
  })

  it('should navigate to a typed 1-based page number on Enter, clamped to range', async () => {
    const onNavigate = vi.fn()
    const user = userEvent.setup()
    render(<PageNavigator index={0} total={5} onNavigate={onNavigate} />)

    const input = screen.getByRole('spinbutton', { name: 'Strana' })
    await user.clear(input)
    await user.type(input, '9{Enter}')

    expect(onNavigate).toHaveBeenCalledWith(4)
  })
})

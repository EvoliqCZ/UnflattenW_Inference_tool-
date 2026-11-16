import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { ModeToggle } from './ModeToggle'

describe('ModeToggle', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should render the three modes with the active one pressed', () => {
    render(<ModeToggle mode="overlay" onChange={vi.fn()} />)

    expect(screen.getByRole('radio', { name: 'Vedľa seba' })).toHaveAttribute('aria-checked', 'false')
    expect(screen.getByRole('radio', { name: 'Prekrytie' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('radio', { name: 'Posuvník' })).toHaveAttribute('aria-checked', 'false')
  })

  it('should call onChange with the selected mode', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<ModeToggle mode="side" onChange={onChange} />)

    await user.click(screen.getByRole('radio', { name: 'Posuvník' }))

    expect(onChange).toHaveBeenCalledWith('swipe')
  })
})

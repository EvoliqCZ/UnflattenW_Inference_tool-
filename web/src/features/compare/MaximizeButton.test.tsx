import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { MaximizeButton } from './MaximizeButton'

describe('MaximizeButton', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should show "Zväčšiť" when not maximized and call onToggle on click', async () => {
    const onToggle = vi.fn()
    const user = userEvent.setup()
    render(<MaximizeButton maximized={false} onToggle={onToggle} />)

    const button = screen.getByRole('button', { name: 'Zväčšiť' })
    await user.click(button)
    expect(onToggle).toHaveBeenCalledOnce()
  })

  it('should show "Obnoviť zobrazenie" when maximized', () => {
    render(<MaximizeButton maximized onToggle={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Obnoviť zobrazenie' })).toHaveAttribute('aria-pressed', 'true')
  })
})

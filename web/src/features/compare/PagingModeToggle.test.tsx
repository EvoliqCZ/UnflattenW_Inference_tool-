import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { PagingModeToggle } from './PagingModeToggle'

describe('PagingModeToggle', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should report the opposite state on click', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<PagingModeToggle linked onChange={onChange} />)

    await user.click(screen.getByRole('switch', { name: 'Prepojiť strany' }))
    expect(onChange).toHaveBeenCalledWith(false)
  })
})

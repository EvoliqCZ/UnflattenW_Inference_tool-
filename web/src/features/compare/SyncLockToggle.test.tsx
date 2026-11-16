import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { SyncLockToggle } from './SyncLockToggle'

describe('SyncLockToggle', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should toggle linked state on click', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<SyncLockToggle linked onChange={onChange} />)

    const toggle = screen.getByRole('button', { name: 'Prepojiť posuny' })
    expect(toggle).toHaveAttribute('aria-pressed', 'true')

    await user.click(toggle)
    expect(onChange).toHaveBeenCalledWith(false)
  })
})

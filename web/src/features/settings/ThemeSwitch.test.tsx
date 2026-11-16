import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { ThemeSwitch } from './ThemeSwitch'

describe('ThemeSwitch (UX §6 Nastavenia)', () => {
  beforeEach(async () => {
    window.localStorage.clear()
    document.documentElement.classList.remove('dark')
    await i18n.changeLanguage('sk')
  })

  afterEach(() => {
    window.localStorage.clear()
    document.documentElement.classList.remove('dark')
  })

  it('defaults to "Podľa systému" and switches to "Tmavá" on click, persisted', async () => {
    const user = userEvent.setup()
    render(<ThemeSwitch />)

    expect(screen.getByRole('radio', { name: 'Podľa systému' })).toBeChecked()

    await user.click(screen.getByRole('radio', { name: 'Tmavá' }))

    expect(screen.getByRole('radio', { name: 'Tmavá' })).toBeChecked()
    expect(window.localStorage.getItem('unflatten-theme')).toBe('dark')
    expect(document.documentElement.classList.contains('dark')).toBe(true)
  })

  it('does not crash when localStorage.setItem throws (private browsing / quota)', async () => {
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('quota exceeded')
    })
    const user = userEvent.setup()
    render(<ThemeSwitch />)

    await user.click(screen.getByRole('radio', { name: 'Tmavá' }))

    expect(screen.getByRole('radiogroup')).toBeInTheDocument()
    setItemSpy.mockRestore()
  })
})

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { LanguageSwitch } from './LanguageSwitch'

describe('LanguageSwitch (UX §6 Nastavenia)', () => {
  beforeEach(async () => {
    window.localStorage.clear()
    await i18n.changeLanguage('sk')
    document.documentElement.lang = 'sk'
  })

  afterEach(async () => {
    window.localStorage.clear()
    await i18n.changeLanguage('sk')
    document.documentElement.lang = 'sk'
  })

  it('renders both languages, current selection checked', () => {
    render(<LanguageSwitch />)

    expect(screen.getByRole('radio', { name: 'Slovenčina' })).toBeChecked()
    expect(screen.getByRole('radio', { name: 'English' })).not.toBeChecked()
  })

  it('switching to English persists it in localStorage and sets <html lang>', async () => {
    const user = userEvent.setup()
    render(<LanguageSwitch />)

    await user.click(screen.getByRole('radio', { name: 'English' }))

    expect(window.localStorage.getItem('unflatten.lang')).toBe('en')
    expect(document.documentElement.lang).toBe('en')
  })

  it('does not crash when localStorage.setItem throws (private browsing / quota)', async () => {
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('quota exceeded')
    })
    const user = userEvent.setup()
    render(<LanguageSwitch />)

    await user.click(screen.getByRole('radio', { name: 'English' }))

    expect(screen.getByRole('radiogroup')).toBeInTheDocument()
    setItemSpy.mockRestore()
  })
})

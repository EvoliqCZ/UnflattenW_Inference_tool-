import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { ShortcutHelpDialog } from './ShortcutHelpDialog'

describe('ShortcutHelpDialog', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should render nothing when closed', () => {
    render(<ShortcutHelpDialog open={false} onOpenChange={vi.fn()} />)
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('should list every shortcut with its key combo when open', () => {
    render(<ShortcutHelpDialog open onOpenChange={vi.fn()} />)

    const dialog = screen.getByRole('dialog')
    expect(dialog).toHaveTextContent('Klávesové skratky')
    expect(dialog).toHaveTextContent('Predchádzajúca / nasledujúca strana')
    expect(dialog).toHaveTextContent('Maximalizovať / obnoviť')
  })
})

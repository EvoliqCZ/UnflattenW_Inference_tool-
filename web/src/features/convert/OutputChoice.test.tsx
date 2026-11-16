import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { OutputChoice } from './OutputChoice'

describe('OutputChoice', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows both options with diagnostic_docx selected by default', () => {
    render(<OutputChoice value="diagnostic_docx" onChange={vi.fn()} />)

    expect(screen.getByRole('radio', { name: /Dokument Word/ })).toBeChecked()
    expect(screen.getByRole('radio', { name: /Medzivýstup/ })).not.toBeChecked()
  })

  it('calls onChange with prediction_ir when that option is picked', async () => {
    const onChange = vi.fn()
    render(<OutputChoice value="diagnostic_docx" onChange={onChange} />)

    await userEvent.click(screen.getByRole('radio', { name: /Medzivýstup/ }))

    expect(onChange).toHaveBeenCalledWith('prediction_ir')
  })
})

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { JobLabelInput, isLabelValid, LABEL_MAX_CODE_POINTS } from './JobLabelInput'

describe('isLabelValid (design §3.2, client-side mirror of the server rule)', () => {
  it('accepts an empty label (optional field)', () => {
    expect(isLabelValid('')).toBe(true)
  })

  it('accepts a normal label', () => {
    expect(isLabelValid('Zmluva o dielo')).toBe(true)
  })

  it('rejects a label longer than 120 code points', () => {
    expect(isLabelValid('a'.repeat(LABEL_MAX_CODE_POINTS + 1))).toBe(false)
  })

  it.each(['\\', '/', ':', '*', '?', '"', '<', '>', '|'])('rejects the forbidden character %s', (char) => {
    expect(isLabelValid(`bad${char}label`)).toBe(false)
  })

  it('rejects a control character', () => {
    expect(isLabelValid('bad\u0007label')).toBe(false)
  })
})

describe('JobLabelInput', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('renders the current value and calls onChange while typing', async () => {
    const onChange = vi.fn()
    render(<JobLabelInput value="" onChange={onChange} />)

    await userEvent.type(screen.getByLabelText('Označenie úlohy'), 'a')

    expect(onChange).toHaveBeenCalledWith('a')
  })

  it('shows the counter once the label reaches 100 code points', () => {
    render(<JobLabelInput value={'a'.repeat(100)} onChange={vi.fn()} />)

    expect(screen.getByText('100/120')).toBeInTheDocument()
  })

  it('does not show the counter below 100 code points', () => {
    render(<JobLabelInput value="short" onChange={vi.fn()} />)

    expect(screen.queryByText(/\/120/)).not.toBeInTheDocument()
  })

  it('flags an invalid label as aria-invalid', () => {
    render(<JobLabelInput value="bad/label" onChange={vi.fn()} />)

    expect(screen.getByLabelText('Označenie úlohy')).toHaveAttribute('aria-invalid', 'true')
  })
})

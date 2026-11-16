import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { VramMeter } from './VramMeter'

describe('VramMeter (UX §3.7 "nezmestí sa")', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows free-of-total text in the non-compact form', () => {
    render(<VramMeter totalMiB={12_000} freeMiB={2_000} language="sk" />)
    expect(screen.getByText(/voľné 2 GiB z 11,7 GiB/)).toBeInTheDocument()
  })

  it('renders a hatched overshoot marker when the estimate exceeds total capacity', () => {
    const { container } = render(<VramMeter totalMiB={12_000} estimateMiB={21_400} compact language="sk" />)
    expect(container.querySelector('[aria-hidden="true"]')).toBeInTheDocument()
  })

  it('omits the caption in compact form', () => {
    render(<VramMeter totalMiB={12_000} estimateMiB={11_240} compact language="sk" />)
    expect(screen.queryByText(/voľné/)).not.toBeInTheDocument()
  })
})

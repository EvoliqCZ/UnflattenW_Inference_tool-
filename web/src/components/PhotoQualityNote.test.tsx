import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { PhotoQualityNote } from './PhotoQualityNote'

describe('PhotoQualityNote', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should render an icon and the fixed H2 text (inline variant)', () => {
    render(<PhotoQualityNote variant="inline" />)

    expect(
      screen.getByText(/Pri fotografiách meranie S5 ukázalo pokles podobnosti textu približne o 2 %/),
    ).toBeInTheDocument()
    expect(document.querySelector('svg[aria-hidden="true"]')).toBeInTheDocument()
  })

  it('should render the same text in compact variant', () => {
    render(<PhotoQualityNote variant="compact" />)

    expect(screen.getByRole('note')).toHaveTextContent('meranie S5')
  })

  it('should never render a close button', () => {
    render(<PhotoQualityNote variant="inline" />)

    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })
})

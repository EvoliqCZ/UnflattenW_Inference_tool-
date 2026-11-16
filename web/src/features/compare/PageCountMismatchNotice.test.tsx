import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { PageCountMismatchNotice } from './PageCountMismatchNotice'

describe('PageCountMismatchNotice', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should render nothing when the counts match', () => {
    const { container } = render(<PageCountMismatchNotice sourceCount={4} docxCount={4} />)

    expect(container).toBeEmptyDOMElement()
  })

  it('should render both counts when they differ', () => {
    render(<PageCountMismatchNotice sourceCount={5} docxCount={4} />)

    expect(
      screen.getByText('Originál má 5 strán, Word sa vykreslil na 4 strán. Strany sa nemusia zhodovať jedna k jednej.'),
    ).toBeInTheDocument()
  })
})

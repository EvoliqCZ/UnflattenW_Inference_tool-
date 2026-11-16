import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { OverlayCanvas } from './OverlayCanvas'

describe('OverlayCanvas', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should render both panes stacked and no blob:/data: URLs', () => {
    const { container } = render(
      <OverlayCanvas
        sourcePane={{ src: '/v1/conversions/j/artifacts/source_preview/0', alt: 'source' }}
        docxPane={{ src: '/v1/conversions/j/artifacts/docx_preview/0', alt: 'docx' }}
      />,
    )

    expect(screen.getByRole('img', { name: 'source' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'docx' })).toBeInTheDocument()
    expect(container.innerHTML).not.toMatch(/blob:|data:/)
  })

  it('should update the overlay opacity custom property from the slider, without a literal style="opacity:" attribute', async () => {
    render(<OverlayCanvas sourcePane={{ src: 's', alt: 'source' }} docxPane={{ src: 'd', alt: 'docx' }} />)

    const top = screen.getByTestId('compare-overlay-top')
    expect(top.style.getPropertyValue('--compare-overlay-opacity')).toBe('0.5')
    expect(top.getAttribute('style')).not.toMatch(/(?:^|;|\s)opacity:/)

    const slider = screen.getByRole('slider', { name: 'Priehľadnosť' })
    const user = userEvent.setup()
    slider.focus()
    await user.keyboard('{ArrowRight}')

    expect(Number(top.style.getPropertyValue('--compare-overlay-opacity'))).toBeGreaterThan(0.5)
  })

  it('should toggle the diff tint switch and apply tint classes to both panes', async () => {
    const { container } = render(
      <OverlayCanvas sourcePane={{ src: 's', alt: 'source' }} docxPane={{ src: 'd', alt: 'docx' }} />,
    )

    const user = userEvent.setup()
    await user.click(screen.getByRole('switch', { name: 'Zvýraznenie rozdielov' }))

    expect(container.querySelector('.compare-tint-source')).not.toBeNull()
    expect(container.querySelector('.compare-tint-docx')).not.toBeNull()
  })
})

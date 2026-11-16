import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { SwipeCanvas } from './SwipeCanvas'

describe('SwipeCanvas', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should render both panes and expose the divider as a slider at 50% by default', () => {
    const { container } = render(
      <SwipeCanvas sourcePane={{ src: 's', alt: 'source' }} docxPane={{ src: 'd', alt: 'docx' }} />,
    )

    expect(screen.getByRole('img', { name: 'source' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'docx' })).toBeInTheDocument()
    const slider = screen.getByRole('slider')
    expect(slider).toHaveAttribute('aria-valuenow', '50')
    expect(container.innerHTML).not.toMatch(/blob:|data:/)
  })

  it('should move the divider (custom property, no literal clip-path/left style) on a track click', () => {
    render(<SwipeCanvas sourcePane={{ src: 's', alt: 'source' }} docxPane={{ src: 'd', alt: 'docx' }} />)
    const track = screen.getByTestId('compare-swipe-track')
    track.getBoundingClientRect = () => ({
      left: 0,
      width: 200,
      top: 0,
      height: 200,
      right: 200,
      bottom: 200,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    })

    fireEvent.pointerDown(track, { clientX: 150 })

    const slider = screen.getByRole('slider')
    expect(slider).toHaveAttribute('aria-valuenow', '75')
    const docxLayer = screen.getByTestId('compare-swipe-docx')
    expect(docxLayer.style.getPropertyValue('--compare-swipe-position')).toBe('75%')
    expect(docxLayer.getAttribute('style')).not.toMatch(/clip-path:/)
  })

  it('should move the handle by 1% on arrow keys and 10% with Shift, once it has focus', async () => {
    const user = userEvent.setup()
    render(<SwipeCanvas sourcePane={{ src: 's', alt: 'source' }} docxPane={{ src: 'd', alt: 'docx' }} />)

    const slider = screen.getByRole('slider')
    slider.focus()
    await user.keyboard('{ArrowRight}')
    expect(slider).toHaveAttribute('aria-valuenow', '51')

    await user.keyboard('{Shift>}{ArrowLeft}{/Shift}')
    expect(slider).toHaveAttribute('aria-valuenow', '41')
  })

  it('should jump to the ends on Home/End', async () => {
    const user = userEvent.setup()
    render(<SwipeCanvas sourcePane={{ src: 's', alt: 'source' }} docxPane={{ src: 'd', alt: 'docx' }} />)

    const slider = screen.getByRole('slider')
    slider.focus()
    await user.keyboard('{End}')
    expect(slider).toHaveAttribute('aria-valuenow', '100')

    await user.keyboard('{Home}')
    expect(slider).toHaveAttribute('aria-valuenow', '0')
  })
})

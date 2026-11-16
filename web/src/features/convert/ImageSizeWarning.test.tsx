import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { ImageSizeWarning, SMALL_IMAGE_LONG_SIDE_PX, estimatedDpi } from './ImageSizeWarning'

describe('estimatedDpi', () => {
  it('rounds long_side / 11.69 (UX §4.4)', () => {
    expect(estimatedDpi(1600)).toBe(137)
  })
})

describe('ImageSizeWarning', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('renders nothing when the long side is at or above the threshold', () => {
    const { container } = render(<ImageSizeWarning longSidePx={SMALL_IMAGE_LONG_SIDE_PX} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('renders the warning with the estimated DPI below the threshold', () => {
    render(<ImageSizeWarning longSidePx={1600} />)
    expect(screen.getByRole('note')).toHaveTextContent('137')
  })
})

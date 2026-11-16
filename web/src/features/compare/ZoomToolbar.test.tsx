import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { ZoomToolbar } from './ZoomToolbar'

describe('ZoomToolbar', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should show the current zoom level and call the matching handler for each button', async () => {
    const onZoomIn = vi.fn()
    const onZoomOut = vi.fn()
    const onFitPage = vi.fn()
    const onFitWidth = vi.fn()
    const onActual = vi.fn()
    const user = userEvent.setup()

    render(
      <ZoomToolbar
        zoomPercent={125}
        onZoomIn={onZoomIn}
        onZoomOut={onZoomOut}
        onFitPage={onFitPage}
        onFitWidth={onFitWidth}
        onActual={onActual}
      />,
    )

    expect(screen.getByText('Priblíženie 125 %')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Priblížiť' }))
    expect(onZoomIn).toHaveBeenCalledOnce()

    await user.click(screen.getByRole('button', { name: 'Oddialiť' }))
    expect(onZoomOut).toHaveBeenCalledOnce()

    await user.click(screen.getByRole('button', { name: 'Prispôsobiť strane' }))
    expect(onFitPage).toHaveBeenCalledOnce()

    await user.click(screen.getByRole('button', { name: 'Prispôsobiť šírke' }))
    expect(onFitWidth).toHaveBeenCalledOnce()

    await user.click(screen.getByRole('button', { name: 'Skutočná veľkosť' }))
    expect(onActual).toHaveBeenCalledOnce()
  })
})

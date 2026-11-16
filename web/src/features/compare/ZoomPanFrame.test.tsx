import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { ZoomPanFrame } from './ZoomPanFrame'

describe('ZoomPanFrame', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should apply zoom and pan as CSS custom properties, never a literal transform', () => {
    render(
      <ZoomPanFrame pane="source" value={{ zoom: 150, panX: 10, panY: -5 }} onPan={vi.fn()} onZoomBy={vi.fn()}>
        <span>page</span>
      </ZoomPanFrame>,
    )

    const frame = screen.getByTestId('compare-zoom-frame-source')
    expect(frame.style.getPropertyValue('--compare-zoom')).toBe('1.5')
    expect(frame.style.getPropertyValue('--compare-pan-x')).toBe('10%')
    expect(frame.style.getPropertyValue('--compare-pan-y')).toBe('-5%')
    expect(frame.getAttribute('style')).not.toMatch(/transform/)
  })

  it('should pan on pointer drag', () => {
    const onPan = vi.fn()
    render(
      <ZoomPanFrame pane="docx" value={{ zoom: 100, panX: 0, panY: 0 }} onPan={onPan} onZoomBy={vi.fn()}>
        <span>page</span>
      </ZoomPanFrame>,
    )

    const frame = screen.getByTestId('compare-zoom-frame-docx')
    fireEvent.pointerDown(frame, { clientX: 0, clientY: 0, button: 0 })
    fireEvent.pointerMove(window, { clientX: 10, clientY: 20 })

    expect(onPan).toHaveBeenCalledWith('docx', expect.any(Number), expect.any(Number))
  })

  it('should zoom on ctrl+wheel but not on a plain wheel', () => {
    const onZoomBy = vi.fn()
    render(
      <ZoomPanFrame pane="source" value={{ zoom: 100, panX: 0, panY: 0 }} onPan={vi.fn()} onZoomBy={onZoomBy}>
        <span>page</span>
      </ZoomPanFrame>,
    )

    const frame = screen.getByTestId('compare-zoom-frame-source')
    fireEvent.wheel(frame, { deltaY: -100 })
    expect(onZoomBy).not.toHaveBeenCalled()

    fireEvent.wheel(frame, { deltaY: -100, ctrlKey: true })
    expect(onZoomBy).toHaveBeenCalledWith('source', 1.1)
  })
})

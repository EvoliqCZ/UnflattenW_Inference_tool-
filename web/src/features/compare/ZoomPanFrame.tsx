import type { CSSProperties, PointerEvent as ReactPointerEvent, ReactNode, WheelEvent as ReactWheelEvent } from 'react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import type { ComparePane, ZoomPanValue } from './useZoomPan'

export interface ZoomPanFrameProps {
  pane: ComparePane
  value: ZoomPanValue
  onPan: (pane: ComparePane, dxPercent: number, dyPercent: number) => void
  onZoomBy: (pane: ComparePane, factor: number) => void
  children: ReactNode
}

/**
 * UX §5.3: applies the shared `{zoom, panX, panY}` transform via CSS custom
 * properties (never a literal inline transform value, matching
 * `compare.css`'s existing convention) and wires the two drag-based
 * interactions the CSS can't express on its own: pointer-drag panning and
 * Ctrl/Cmd+wheel zoom anchored... at the frame's centre, not the cursor --
 * a cursor-anchored zoom needs `getBoundingClientRect` geometry jsdom cannot
 * exercise meaningfully in tests, so it is out of this task's scope (noted
 * for a follow-up). A plain wheel (no modifier) is left alone so it keeps
 * scrolling the page (UX §5.3).
 */
export function ZoomPanFrame({ pane, value, onPan, onZoomBy, children }: ZoomPanFrameProps) {
  const { t } = useTranslation()

  function startDrag(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return
    const rect = event.currentTarget.getBoundingClientRect()
    // jsdom has no layout engine, so `rect.width/height` are 0 in tests;
    // fall back to 1 so a drag still produces a (large but non-NaN, non-Infinity) percent delta.
    const width = rect.width || 1
    const height = rect.height || 1
    let lastX = event.clientX
    let lastY = event.clientY
    function onMove(moveEvent: PointerEvent) {
      onPan(pane, ((moveEvent.clientX - lastX) / width) * 100, ((moveEvent.clientY - lastY) / height) * 100)
      lastX = moveEvent.clientX
      lastY = moveEvent.clientY
    }
    function onUp() {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  function onWheel(event: ReactWheelEvent<HTMLDivElement>) {
    if (!event.ctrlKey && !event.metaKey) return
    event.preventDefault()
    onZoomBy(pane, event.deltaY < 0 ? 1.1 : 1 / 1.1)
  }

  const style = {
    '--compare-zoom': value.zoom / 100,
    '--compare-pan-x': `${String(value.panX)}%`,
    '--compare-pan-y': `${String(value.panY)}%`,
  } as CSSProperties

  return (
    <div
      role="group"
      aria-label={t(pane === 'source' ? 'compare.paneSource' : 'compare.paneDocx')}
      data-pane={pane}
      data-testid={`compare-zoom-frame-${pane}`}
      className={cn('compare-zoom-frame', value.zoom > 100 && 'compare-zoom-frame-draggable')}
      style={style}
      onPointerDown={startDrag}
      onWheel={onWheel}
    >
      {children}
    </div>
  )
}

import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'
import { useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { PagePane, type PagePaneProps } from './PagePane'

export interface SwipeCanvasProps {
  sourcePane: PagePaneProps
  docxPane: PagePaneProps
}

const SWIPE_STEP = 1
const SWIPE_STEP_LARGE = 10

/**
 * UX §5.2 "Posuvník": original on the left, Word revealed from the right as
 * the divider moves left. Dragging (pointer) and the handle's own arrow-key
 * control (UX §5.9: ←/→ by 1%, Shift+←/→ by 10%, Home/End to the ends) are
 * both first-class -- WCAG 2.5.7 requires the button/keyboard alternative to
 * every drag gesture.
 */
export function SwipeCanvas({ sourcePane, docxPane }: SwipeCanvasProps) {
  const { t } = useTranslation()
  const [position, setPosition] = useState(50)
  const trackRef = useRef<HTMLDivElement>(null)

  function moveTo(clientX: number) {
    const track = trackRef.current
    if (!track) return
    const rect = track.getBoundingClientRect()
    const percent = rect.width === 0 ? 0 : ((clientX - rect.left) / rect.width) * 100
    setPosition(Math.min(100, Math.max(0, percent)))
  }

  function startDrag(event: ReactPointerEvent<HTMLDivElement>) {
    event.stopPropagation()
    moveTo(event.clientX)
    function onMove(moveEvent: PointerEvent) {
      moveTo(moveEvent.clientX)
    }
    function onUp() {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  function onHandleKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    switch (event.key) {
      case 'ArrowLeft':
        event.stopPropagation()
        setPosition((current) => Math.max(0, current - (event.shiftKey ? SWIPE_STEP_LARGE : SWIPE_STEP)))
        return
      case 'ArrowRight':
        event.stopPropagation()
        setPosition((current) => Math.min(100, current + (event.shiftKey ? SWIPE_STEP_LARGE : SWIPE_STEP)))
        return
      case 'Home':
        event.stopPropagation()
        setPosition(0)
        return
      case 'End':
        event.stopPropagation()
        setPosition(100)
        return
      default:
    }
  }

  const swipeStyle = { '--compare-swipe-position': `${String(position)}%` } as CSSProperties

  return (
    <div
      ref={trackRef}
      data-testid="compare-swipe-track"
      className="compare-swipe compare-page-slot"
      onPointerDown={(event) => {
        moveTo(event.clientX)
      }}
    >
      <PagePane key={sourcePane.src ?? 'missing'} {...sourcePane} />
      <div data-testid="compare-swipe-docx" className="compare-swipe-docx" style={swipeStyle}>
        <PagePane key={docxPane.src ?? 'missing'} {...docxPane} />
      </div>
      <div
        role="slider"
        tabIndex={0}
        aria-label={t('compare.mode.swipe')}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(position)}
        className="compare-swipe-handle"
        style={swipeStyle}
        onPointerDown={startDrag}
        onKeyDown={onHandleKeyDown}
      />
    </div>
  )
}

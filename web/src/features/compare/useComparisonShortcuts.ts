import type { KeyboardEvent as ReactKeyboardEvent } from 'react'
import type { CompareMode } from './types'

export interface ComparisonShortcutActions {
  prevPage: () => void
  nextPage: () => void
  firstPage: () => void
  lastPage: () => void
  zoomIn: () => void
  zoomOut: () => void
  fitPage: () => void
  actualSize: () => void
  /** Percent-of-frame steps, `dx`/`dy` in `[-1, 0, 1]`. */
  pan: (dx: number, dy: number) => void
  setMode: (mode: CompareMode) => void
  toggleDiffTint: () => void
  opacityStep: (deltaPercent: number) => void
  toggleLock: () => void
  toggleMaximize: () => void
  exitMaximize: () => void
  openHelp: () => void
}

function isTextInput(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  return target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable
}

/**
 * UX §5.9: builds the comparison viewer's `onKeyDown` handler. Bound to the
 * viewer's root region element (not `document`), so it only ever fires
 * while focus is inside the viewer -- exactly the "active only when the
 * viewer has focus" requirement, for free from event bubbling, with no
 * separate focus-tracking state. The one exception the table itself calls
 * out is a focused text field (`compare-page-input`), guarded here.
 */
export function useComparisonShortcuts(actions: ComparisonShortcutActions) {
  return function onKeyDown(event: ReactKeyboardEvent<HTMLElement>) {
    if (event.key === 'Escape') {
      actions.exitMaximize()
      return
    }
    if (isTextInput(event.target)) return

    switch (event.key) {
      case 'ArrowLeft':
        event.preventDefault()
        if (event.shiftKey) actions.pan(1, 0)
        else actions.prevPage()
        return
      case 'ArrowRight':
        event.preventDefault()
        if (event.shiftKey) actions.pan(-1, 0)
        else actions.nextPage()
        return
      case 'ArrowUp':
        event.preventDefault()
        actions.pan(0, 1)
        return
      case 'ArrowDown':
        event.preventDefault()
        actions.pan(0, -1)
        return
      case 'Home':
        actions.firstPage()
        return
      case 'End':
        actions.lastPage()
        return
      case '+':
      case '=':
        actions.zoomIn()
        return
      case '-':
        actions.zoomOut()
        return
      case '0':
        actions.fitPage()
        return
      case '1':
        actions.actualSize()
        return
      case 's':
      case 'S':
        actions.setMode('side')
        return
      case 'o':
      case 'O':
        actions.setMode('overlay')
        return
      case 'w':
      case 'W':
        actions.setMode('swipe')
        return
      case 'd':
      case 'D':
        actions.toggleDiffTint()
        return
      case '[':
        actions.opacityStep(-10)
        return
      case ']':
        actions.opacityStep(10)
        return
      case 'l':
      case 'L':
        actions.toggleLock()
        return
      case 'f':
      case 'F':
        actions.toggleMaximize()
        return
      case '?':
        actions.openHelp()
        return
      default:
    }
  }
}

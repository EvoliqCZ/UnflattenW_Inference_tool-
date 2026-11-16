import { useState } from 'react'

export type ComparePane = 'source' | 'docx'

export interface ZoomPanValue {
  /** Percent, 25-800. 100 = the shared "fit page" baseline (UX §5.3). */
  zoom: number
  /** Pan offset as a percent of the frame, both axes. */
  panX: number
  panY: number
}

const BASELINE: ZoomPanValue = { zoom: 100, panX: 0, panY: 0 }
const MIN_ZOOM = 25
const MAX_ZOOM = 800
const ZOOM_STEP_FACTOR = 1.25
const PAN_STEP_PERCENT = 10

function clampZoom(zoom: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom))
}

/**
 * UX §5.3: one shared `{zoom, panX, panY}` transform for both panes. The
 * "sync lock" (`linked`, UX §5.3 🔗, side-mode only) decides whether an
 * update to one pane also writes the other -- unlinked, each pane keeps its
 * own independent state so differently placed blocks can be compared.
 *
 * `fitPage` / `fitWidth` / `actual` (UX §5.3 zoom levels) all resolve to the
 * same `{zoom: 100, pan: 0}` baseline here: the page slot's fixed A4 aspect
 * ratio (`compare.css` `.compare-page-slot`) already fits both width and
 * height at 100%, so there is nothing left for "fit width" or "actual size"
 * to do differently without a layout change outside this task's scope.
 */
export function useZoomPan(linked: boolean) {
  const [source, setSource] = useState<ZoomPanValue>(BASELINE)
  const [docx, setDocx] = useState<ZoomPanValue>(BASELINE)

  // Functional updates throughout -- several callers (zoom-step keys, wheel
  // ticks) fire multiple updates before React re-renders, and reading the
  // `source`/`docx` closure variables would apply every one of them to the
  // same stale snapshot instead of accumulating.
  function write(pane: ComparePane, updater: (current: ZoomPanValue) => ZoomPanValue) {
    if (linked) {
      setSource((current) => updater(current))
      setDocx((current) => updater(current))
      return
    }
    if (pane === 'source') setSource(updater)
    else setDocx(updater)
  }

  function valueOf(pane: ComparePane): ZoomPanValue {
    return pane === 'source' ? source : docx
  }

  function zoomBy(pane: ComparePane, factor: number) {
    write(pane, (current) => ({ ...current, zoom: clampZoom(current.zoom * factor) }))
  }

  function reset(pane: ComparePane) {
    write(pane, () => BASELINE)
  }

  function pan(pane: ComparePane, dxPercent: number, dyPercent: number) {
    write(pane, (current) => ({ ...current, panX: current.panX + dxPercent, panY: current.panY + dyPercent }))
  }

  function panByKey(pane: ComparePane, dx: -1 | 0 | 1, dy: -1 | 0 | 1) {
    pan(pane, dx * PAN_STEP_PERCENT, dy * PAN_STEP_PERCENT)
  }

  return {
    source,
    docx,
    valueOf,
    zoomBy,
    zoomIn: (pane: ComparePane) => {
      zoomBy(pane, ZOOM_STEP_FACTOR)
    },
    zoomOut: (pane: ComparePane) => {
      zoomBy(pane, 1 / ZOOM_STEP_FACTOR)
    },
    setActual: reset,
    setFitPage: reset,
    setFitWidth: reset,
    reset,
    pan,
    panByKey,
  }
}

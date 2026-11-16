import { OverlayCanvas } from './OverlayCanvas'
import { PagePane, type PagePaneProps } from './PagePane'
import { SwipeCanvas } from './SwipeCanvas'
import type { CompareMode } from './types'
import type { ComparePane, ZoomPanValue } from './useZoomPan'
import { ZoomPanFrame } from './ZoomPanFrame'

export interface ComparisonCanvasProps {
  mode: CompareMode
  sourcePane: PagePaneProps
  docxPane: PagePaneProps
  sourceIndex: number
  docxIndex: number
  zoomSource: ZoomPanValue
  zoomDocx: ZoomPanValue
  onPan: (pane: ComparePane, dx: number, dy: number) => void
  onZoomBy: (pane: ComparePane, factor: number) => void
  opacity: number
  onOpacityChange: (opacity: number) => void
  diffTint: boolean
  onDiffTintChange: (diffTint: boolean) => void
}

/** The mode-specific content (UX §5.2), wrapped in the shared zoom/pan transform (UX §5.3). */
export function ComparisonCanvas({
  mode,
  sourcePane,
  docxPane,
  sourceIndex,
  docxIndex,
  zoomSource,
  zoomDocx,
  onPan,
  onZoomBy,
  opacity,
  onOpacityChange,
  diffTint,
  onDiffTintChange,
}: ComparisonCanvasProps) {
  if (mode === 'side') {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <ZoomPanFrame pane="source" value={zoomSource} onPan={onPan} onZoomBy={onZoomBy}>
          <div className="compare-page-slot">
            <PagePane key={sourcePane.src ?? `missing-${String(sourceIndex)}`} {...sourcePane} />
          </div>
        </ZoomPanFrame>
        <ZoomPanFrame pane="docx" value={zoomDocx} onPan={onPan} onZoomBy={onZoomBy}>
          <div className="compare-page-slot">
            <PagePane key={docxPane.src ?? `missing-${String(docxIndex)}`} {...docxPane} />
          </div>
        </ZoomPanFrame>
      </div>
    )
  }

  return (
    <ZoomPanFrame pane="source" value={zoomSource} onPan={onPan} onZoomBy={onZoomBy}>
      {mode === 'overlay' ? (
        <OverlayCanvas
          sourcePane={sourcePane}
          docxPane={docxPane}
          opacity={opacity}
          onOpacityChange={onOpacityChange}
          diffTint={diffTint}
          onDiffTintChange={onDiffTintChange}
        />
      ) : (
        <SwipeCanvas sourcePane={sourcePane} docxPane={docxPane} />
      )}
    </ZoomPanFrame>
  )
}

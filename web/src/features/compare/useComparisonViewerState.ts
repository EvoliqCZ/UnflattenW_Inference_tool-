import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { buildPageSequences } from './buildPageSequences'
import { selectMissingReason } from './selectMissingReason'
import type { ArtifactEntry, CompareMode, DiagnosticOutcome, PageReport } from './types'
import { useComparisonShortcuts } from './useComparisonShortcuts'
import { useZoomPan } from './useZoomPan'

export interface UseComparisonViewerStateArgs {
  artifacts: ArtifactEntry[]
  outcome: DiagnosticOutcome | null
  pages: PageReport[] | null
  initialPage: number
  initialMode: CompareMode
  onStateChange?: ((state: { page: number; mode: CompareMode }) => void) | undefined
}

function clamp(value: number, max: number): number {
  return Math.min(Math.max(value, 0), max)
}

/**
 * All of `ComparisonViewer`'s state and derived values, split out so the
 * component itself stays a render function (Karpathy simplicity: one file,
 * one job). Keyboard-driven zoom/pan always targets the `source` pane (see
 * `ComparisonViewer`'s own doc comment for why that is an accepted
 * simplification, not an oversight).
 */
export function useComparisonViewerState({
  artifacts,
  outcome,
  pages,
  initialPage,
  initialMode,
  onStateChange,
}: UseComparisonViewerStateArgs) {
  const { t } = useTranslation()
  const { source, docx, sourceCount, docxCount } = buildPageSequences(artifacts)
  const total = Math.max(sourceCount, docxCount, 1)
  const mismatched = sourceCount !== docxCount

  const [mode, setMode] = useState<CompareMode>(initialMode)
  const [linkedPaging, setLinkedPaging] = useState(true)
  const [rawIndex, setRawIndex] = useState(() => Math.max(initialPage, 0))
  const [rawSourceIndex, setRawSourceIndex] = useState(() => Math.max(initialPage, 0))
  const [rawDocxIndex, setRawDocxIndex] = useState(() => Math.max(initialPage, 0))
  const [zoomLinked, setZoomLinked] = useState(true)
  const [maximized, setMaximized] = useState(false)
  const [helpOpen, setHelpOpen] = useState(false)
  const [opacity, setOpacity] = useState(50)
  const [diffTint, setDiffTint] = useState(false)
  const zoomPan = useZoomPan(zoomLinked)

  const linkedIndex = clamp(rawIndex, total - 1)
  const sourceIndex = linkedPaging ? linkedIndex : clamp(rawSourceIndex, Math.max(sourceCount - 1, 0))
  const docxIndex = linkedPaging ? linkedIndex : clamp(rawDocxIndex, Math.max(docxCount - 1, 0))

  function navigate(nextIndex: number) {
    const clamped = clamp(nextIndex, total - 1)
    setRawIndex(clamped)
    onStateChange?.({ page: clamped, mode })
  }

  function changeMode(nextMode: CompareMode) {
    setMode(nextMode)
    onStateChange?.({ page: linkedIndex, mode: nextMode })
  }

  const onKeyDown = useComparisonShortcuts({
    prevPage: () => {
      navigate(linkedIndex - 1)
    },
    nextPage: () => {
      navigate(linkedIndex + 1)
    },
    firstPage: () => {
      navigate(0)
    },
    lastPage: () => {
      navigate(total - 1)
    },
    zoomIn: () => {
      zoomPan.zoomIn('source')
    },
    zoomOut: () => {
      zoomPan.zoomOut('source')
    },
    fitPage: () => {
      zoomPan.setFitPage('source')
    },
    actualSize: () => {
      zoomPan.setActual('source')
    },
    pan: (dx, dy) => {
      zoomPan.panByKey('source', dx as -1 | 0 | 1, dy as -1 | 0 | 1)
    },
    setMode: changeMode,
    toggleDiffTint: () => {
      setDiffTint((current) => !current)
    },
    opacityStep: (delta) => {
      setOpacity((current) => clamp(current + delta, 100))
    },
    toggleLock: () => {
      setZoomLinked((current) => !current)
    },
    toggleMaximize: () => {
      setMaximized((current) => !current)
    },
    exitMaximize: () => {
      setMaximized(false)
    },
    openHelp: () => {
      setHelpOpen(true)
    },
  })

  const sourceUrl = source.get(sourceIndex)
  const docxUrl = docx.get(docxIndex)
  const sourcePane = {
    src: sourceUrl,
    alt: t('compare.alt.source', { index: sourceIndex + 1, total: sourceCount }),
    missing: sourceUrl
      ? undefined
      : selectMissingReason({ side: 'source', pageIndex: sourceIndex, sourceCount, pages, outcome }),
  }
  const docxPane = {
    src: docxUrl,
    alt: t('compare.alt.docx', { index: docxIndex + 1, total: docxCount }),
    missing: docxUrl
      ? undefined
      : selectMissingReason({ side: 'docx', pageIndex: docxIndex, sourceCount, pages, outcome }),
  }

  return {
    source,
    docx,
    sourceCount,
    docxCount,
    total,
    mismatched,
    mode,
    changeMode,
    linkedPaging,
    setLinkedPaging,
    sourceIndex,
    docxIndex,
    linkedIndex,
    navigate,
    setRawSourceIndex,
    setRawDocxIndex,
    zoomPan,
    zoomLinked,
    setZoomLinked,
    maximized,
    setMaximized,
    helpOpen,
    setHelpOpen,
    opacity,
    setOpacity,
    diffTint,
    setDiffTint,
    onKeyDown,
    sourcePane,
    docxPane,
  }
}

import { useTranslation } from 'react-i18next'
import { ShortcutHelpDialog } from '@/components/ShortcutHelpDialog'
import { cn } from '@/lib/utils'
import { ComparisonCanvas } from './ComparisonCanvas'
import { ComparisonToolbar } from './ComparisonToolbar'
import './compare.css'
import { PageCountMismatchNotice } from './PageCountMismatchNotice'
import { PageNavigator } from './PageNavigator'
import { PagingModeToggle } from './PagingModeToggle'
import { SeparatePageNavigators } from './SeparatePageNavigators'
import { ThumbnailStrip } from './ThumbnailStrip'
import type { ArtifactEntry, CompareMode, DiagnosticOutcome, PageReport } from './types'
import { useComparisonViewerState } from './useComparisonViewerState'

export interface ComparisonViewerProps {
  artifacts: ArtifactEntry[]
  outcome: DiagnosticOutcome | null
  pages: PageReport[] | null
  /** 0-based. */
  initialPage?: number
  initialMode?: CompareMode
  onStateChange?: (state: { page: number; mode: CompareMode }) => void
}

/**
 * UX §5: the "Porovnanie" tab. Builds independent source/docx page
 * sequences from `artifacts` (§5.1), renders one of the three modes (§5.2)
 * plus the shared navigator/thumbnails (§5.4), zoom/pan (§5.3), single-key
 * shortcuts (§5.9) and maximize (§5.8, a CSS-only in-app overlay -- the
 * banner it must leave visible belongs to `AppShell`, outside this
 * component's ownership). Nearly all state and derived values live in
 * `useComparisonViewerState`; this file is just the render.
 */
export function ComparisonViewer({
  artifacts,
  outcome,
  pages,
  initialPage = 0,
  initialMode = 'side',
  onStateChange,
}: ComparisonViewerProps) {
  const { t } = useTranslation()
  const state = useComparisonViewerState({ artifacts, outcome, pages, initialPage, initialMode, onStateChange })

  return (
    <section
      tabIndex={0}
      role="region"
      aria-label={t('job.tab.compare')}
      className={cn('flex flex-col gap-4', state.maximized && 'compare-maximized')}
      onKeyDown={state.onKeyDown}
    >
      <ComparisonToolbar
        mode={state.mode}
        onModeChange={state.changeMode}
        zoomPercent={state.zoomPan.source.zoom}
        onZoomIn={() => {
          state.zoomPan.zoomIn('source')
        }}
        onZoomOut={() => {
          state.zoomPan.zoomOut('source')
        }}
        onFitPage={() => {
          state.zoomPan.setFitPage('source')
        }}
        onFitWidth={() => {
          state.zoomPan.setFitWidth('source')
        }}
        onActual={() => {
          state.zoomPan.setActual('source')
        }}
        zoomLinked={state.zoomLinked}
        onZoomLinkedChange={state.setZoomLinked}
        maximized={state.maximized}
        onToggleMaximize={() => {
          state.setMaximized((current) => !current)
        }}
        onOpenHelp={() => {
          state.setHelpOpen(true)
        }}
      />
      <PageCountMismatchNotice sourceCount={state.sourceCount} docxCount={state.docxCount} />
      {state.mismatched ? <PagingModeToggle linked={state.linkedPaging} onChange={state.setLinkedPaging} /> : null}
      {!state.linkedPaging ? (
        <SeparatePageNavigators
          sourceIndex={state.sourceIndex}
          docxIndex={state.docxIndex}
          sourceCount={state.sourceCount}
          docxCount={state.docxCount}
          onNavigateSource={state.setRawSourceIndex}
          onNavigateDocx={state.setRawDocxIndex}
        />
      ) : null}
      <ComparisonCanvas
        mode={state.mode}
        sourcePane={state.sourcePane}
        docxPane={state.docxPane}
        sourceIndex={state.sourceIndex}
        docxIndex={state.docxIndex}
        zoomSource={state.zoomPan.source}
        zoomDocx={state.zoomPan.docx}
        onPan={state.zoomPan.pan}
        onZoomBy={state.zoomPan.zoomBy}
        opacity={state.opacity}
        onOpacityChange={state.setOpacity}
        diffTint={state.diffTint}
        onDiffTintChange={state.setDiffTint}
      />
      {state.linkedPaging ? (
        <>
          <PageNavigator index={state.linkedIndex} total={state.total} onNavigate={state.navigate} />
          <ThumbnailStrip
            sourceThumbs={Array.from({ length: state.total }, (_, i) => state.source.get(i))}
            docxThumbs={Array.from({ length: state.total }, (_, i) => state.docx.get(i))}
            pages={pages}
            currentIndex={state.linkedIndex}
            onNavigate={state.navigate}
          />
        </>
      ) : (
        <ThumbnailStrip
          sourceThumbs={Array.from({ length: Math.max(state.sourceCount, 1) }, (_, i) => state.source.get(i))}
          docxThumbs={Array.from({ length: Math.max(state.docxCount, 1) }, (_, i) => state.docx.get(i))}
          pages={pages}
          linked={false}
          currentIndex={0}
          onNavigate={() => undefined}
          sourceIndex={state.sourceIndex}
          docxIndex={state.docxIndex}
          onNavigateSource={state.setRawSourceIndex}
          onNavigateDocx={state.setRawDocxIndex}
        />
      )}
      <ShortcutHelpDialog open={state.helpOpen} onOpenChange={state.setHelpOpen} />
    </section>
  )
}

import { HelpCircle } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'
import { MaximizeButton } from './MaximizeButton'
import { ModeToggle } from './ModeToggle'
import { SyncLockToggle } from './SyncLockToggle'
import type { CompareMode } from './types'
import { ZoomToolbar } from './ZoomToolbar'

export interface ComparisonToolbarProps {
  mode: CompareMode
  onModeChange: (mode: CompareMode) => void
  zoomPercent: number
  onZoomIn: () => void
  onZoomOut: () => void
  onFitPage: () => void
  onFitWidth: () => void
  onActual: () => void
  zoomLinked: boolean
  onZoomLinkedChange: (linked: boolean) => void
  maximized: boolean
  onToggleMaximize: () => void
  onOpenHelp: () => void
}

/**
 * The comparison viewer's control bar: mode + zoom + (side mode only) the
 * sync lock + maximize + shortcut help. `flex-wrap` so it degrades to
 * multiple rows on a phone instead of overflowing (UX has no dedicated
 * phone layout for this bar beyond "it must stay usable").
 */
export function ComparisonToolbar({
  mode,
  onModeChange,
  zoomPercent,
  onZoomIn,
  onZoomOut,
  onFitPage,
  onFitWidth,
  onActual,
  zoomLinked,
  onZoomLinkedChange,
  maximized,
  onToggleMaximize,
  onOpenHelp,
}: ComparisonToolbarProps) {
  const { t } = useTranslation()

  return (
    <div className="flex flex-wrap items-center gap-2">
      <ModeToggle mode={mode} onChange={onModeChange} />
      <ZoomToolbar
        zoomPercent={zoomPercent}
        onZoomIn={onZoomIn}
        onZoomOut={onZoomOut}
        onFitPage={onFitPage}
        onFitWidth={onFitWidth}
        onActual={onActual}
      />
      {mode === 'side' ? <SyncLockToggle linked={zoomLinked} onChange={onZoomLinkedChange} /> : null}
      <MaximizeButton maximized={maximized} onToggle={onToggleMaximize} />
      <Button type="button" variant="outline" size="icon" aria-label={t('compare.shortcuts.open')} onClick={onOpenHelp}>
        <HelpCircle aria-hidden="true" />
      </Button>
    </div>
  )
}

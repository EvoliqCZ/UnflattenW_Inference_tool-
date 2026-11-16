import { Minus, Plus, RotateCcw } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'

export interface ZoomToolbarProps {
  zoomPercent: number
  onZoomIn: () => void
  onZoomOut: () => void
  onFitPage: () => void
  onFitWidth: () => void
  onActual: () => void
}

/**
 * UX §5.3/§5.9/§5.10: the button alternative to every zoom gesture (WCAG
 * 2.5.7) -- keyboard shortcuts (`useComparisonShortcuts`) call the same
 * handlers this toolbar does, so there is exactly one place each action is
 * implemented. `aria-keyshortcuts` documents the key without depending on
 * it (UX §5.9's tooltip requirement).
 */
export function ZoomToolbar({ zoomPercent, onZoomIn, onZoomOut, onFitPage, onFitWidth, onActual }: ZoomToolbarProps) {
  const { t } = useTranslation()

  return (
    <div className="flex flex-wrap items-center gap-1" role="group" aria-label={t('compare.zoomGroup')}>
      <Button
        type="button"
        variant="outline"
        size="icon"
        aria-label={t('compare.zoom.out')}
        aria-keyshortcuts="-"
        onClick={onZoomOut}
      >
        <Minus aria-hidden="true" />
      </Button>
      <span aria-live="polite" className="w-14 text-center text-sm tabular-nums">
        {t('compare.zoomLevel', { percent: Math.round(zoomPercent) })}
      </span>
      <Button
        type="button"
        variant="outline"
        size="icon"
        aria-label={t('compare.zoom.in')}
        aria-keyshortcuts="+"
        onClick={onZoomIn}
      >
        <Plus aria-hidden="true" />
      </Button>
      <Button type="button" variant="outline" size="sm" aria-keyshortcuts="0" onClick={onFitPage}>
        {t('compare.zoom.fitPage')}
      </Button>
      <Button type="button" variant="outline" size="sm" onClick={onFitWidth}>
        {t('compare.zoom.fitWidth')}
      </Button>
      <Button type="button" variant="outline" size="sm" aria-keyshortcuts="1" onClick={onActual}>
        {t('compare.zoom.actual')}
      </Button>
      <Button type="button" variant="outline" size="icon" aria-label={t('compare.zoom.reset')} onClick={onFitPage}>
        <RotateCcw aria-hidden="true" />
      </Button>
    </div>
  )
}

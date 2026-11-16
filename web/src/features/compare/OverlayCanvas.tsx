import type { CSSProperties } from 'react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { PagePane, type PagePaneProps } from './PagePane'

export interface OverlayCanvasProps {
  sourcePane: PagePaneProps
  docxPane: PagePaneProps
  /**
   * Controlled opacity/diff-tint (T28: driven by `useComparisonShortcuts`'
   * `[`/`]`/`D` and the toolbar). Omitted -- as every existing standalone
   * caller/test does -- falls back to this component's own state, so it
   * still works with no parent wiring at all.
   */
  opacity?: number
  onOpacityChange?: (opacity: number) => void
  diffTint?: boolean
  onDiffTintChange?: (diffTint: boolean) => void
}

/** UX §5.2 "Prekrytie": Word over the original, opacity slider + optional colour diff tint. */
export function OverlayCanvas({
  sourcePane,
  docxPane,
  opacity: controlledOpacity,
  onOpacityChange,
  diffTint: controlledDiffTint,
  onDiffTintChange,
}: OverlayCanvasProps) {
  const { t } = useTranslation()
  const [internalOpacity, setInternalOpacity] = useState(50)
  const [internalDiffTint, setInternalDiffTint] = useState(false)
  const opacity = controlledOpacity ?? internalOpacity
  const diffTint = controlledDiffTint ?? internalDiffTint
  const setOpacity = onOpacityChange ?? setInternalOpacity
  const setDiffTint = onDiffTintChange ?? setInternalDiffTint

  return (
    <div>
      <div className="compare-overlay compare-page-slot">
        <PagePane key={sourcePane.src ?? 'missing'} {...sourcePane} tint={diffTint ? 'source' : undefined} />
        <div
          data-testid="compare-overlay-top"
          className="compare-overlay-top"
          style={{ '--compare-overlay-opacity': opacity / 100 } as CSSProperties}
        >
          <PagePane key={docxPane.src ?? 'missing'} {...docxPane} tint={diffTint ? 'docx' : undefined} />
        </div>
      </div>
      <div className="flex items-center gap-4">
        <label className="flex flex-1 items-center gap-2 text-sm">
          {t('compare.opacity')}
          <Slider
            aria-label={t('compare.opacity')}
            min={0}
            max={100}
            step={1}
            value={[opacity]}
            onValueChange={([value]) => {
              if (value !== undefined) setOpacity(value)
            }}
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <Switch checked={diffTint} onCheckedChange={setDiffTint} aria-label={t('compare.diffTint')} />
          {t('compare.diffTint')}
        </label>
      </div>
    </div>
  )
}

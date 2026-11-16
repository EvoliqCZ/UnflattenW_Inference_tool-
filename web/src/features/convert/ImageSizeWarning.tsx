import { AlertTriangle } from 'lucide-react'
import { useTranslation } from 'react-i18next'

/** UX §4.4: a warning constant, not a policy field yet (design §12 A5). */
export const SMALL_IMAGE_LONG_SIDE_PX = 2500
const A4_LONG_SIDE_INCHES = 11.69

/** `round(long_side / 11.69)` (UX §4.4): the DPI an A4 page would have at this pixel size. */
export function estimatedDpi(longSidePx: number): number {
  return Math.round(longSidePx / A4_LONG_SIDE_INCHES)
}

export interface ImageSizeWarningProps {
  longSidePx: number
}

/** UX §4.4: non-blocking warning under a queue row whose long side is below the threshold. */
export function ImageSizeWarning({ longSidePx }: ImageSizeWarningProps) {
  const { t } = useTranslation()
  if (longSidePx >= SMALL_IMAGE_LONG_SIDE_PX) {
    return null
  }
  return (
    <p role="note" className="flex items-start gap-1.5 text-xs text-warning-text">
      <AlertTriangle aria-hidden="true" className="size-3.5 shrink-0 translate-y-0.5" />
      <span>{t('convert.warning.smallImage', { dpi: estimatedDpi(longSidePx) })}</span>
    </p>
  )
}

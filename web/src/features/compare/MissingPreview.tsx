import { FileWarning } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'
import { describeMissingReason } from './describeMissingReason'
import type { MissingReason } from './types'

export interface MissingPreviewProps {
  reason: MissingReason
  /** Only rendered for `loadFailed` -- UX §5.6 "Skúsiť znova". */
  onRetry?: (() => void) | undefined
}

/** UX §5.6: an A4-shaped hatched placeholder carrying the reason a pane has no image. */
export function MissingPreview({ reason, onRetry }: MissingPreviewProps) {
  const { t } = useTranslation()
  const text = describeMissingReason(t, reason)

  return (
    <div className="compare-missing-preview" role="img" aria-label={text}>
      <FileWarning aria-hidden="true" className="size-6" />
      <p>{text}</p>
      {reason.kind === 'loadFailed' && onRetry ? (
        <Button type="button" size="sm" variant="outline" onClick={onRetry}>
          {t('common.retry')}
        </Button>
      ) : null}
    </div>
  )
}

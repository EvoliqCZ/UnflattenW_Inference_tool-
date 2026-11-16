import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { formatBytes, formatPercent } from '@/lib/format'
import type { SupportedLanguage } from '@/i18n'
import { PROBLEM_CODES, type DownloadState, type ProblemCode } from '@/i18n/codes'

export interface DownloadProgressProps {
  state: DownloadState
  bytesDone?: number | null
  bytesTotal?: number | null
  errorCode?: string | null
  language: SupportedLanguage
  onRetry?: () => void
  retryLabel?: string
}

function isProblemCode(code: string): code is ProblemCode {
  return (PROBLEM_CODES as readonly string[]).includes(code)
}

/**
 * UX §3.7 points 2/6: a determinate bar while `bytesTotal` is known,
 * indeterminate (Radix `data-state="indeterminate"`, no `aria-valuenow`)
 * while it is still `null`, and a translated error plus "Stiahnuť znova" on
 * `failed`.
 */
export function DownloadProgress({
  state,
  bytesDone,
  bytesTotal,
  errorCode,
  language,
  onRetry,
  retryLabel,
}: DownloadProgressProps) {
  const { t } = useTranslation()

  if (state === 'failed') {
    const title = errorCode && isProblemCode(errorCode) ? t(`codes.problem.${errorCode}.title`) : t('codes.unknown')
    return (
      <div className="flex flex-col gap-2">
        <p className="text-sm text-destructive">{title}</p>
        {onRetry ? (
          <Button type="button" size="sm" variant="outline" onClick={onRetry}>
            {retryLabel ?? t('models.action.retryDownload')}
          </Button>
        ) : null}
      </div>
    )
  }

  if (state !== 'downloading' && state !== 'verifying' && state !== 'loading') {
    return null
  }

  const value =
    bytesTotal !== undefined && bytesTotal !== null && bytesDone != null ? (bytesDone / bytesTotal) * 100 : null
  const valueLabel =
    value !== null && bytesDone != null && bytesTotal != null
      ? `${formatBytes(bytesDone, language)} / ${formatBytes(bytesTotal, language)} (${formatPercent(value / 100, language)})`
      : t(`models.state.${state}`)

  return (
    <div className="flex flex-col gap-1">
      <Progress value={value} valueLabel={valueLabel} />
      <p className="text-xs text-muted-foreground">{valueLabel}</p>
      {state === 'downloading' ? <p className="text-xs text-muted-foreground">{t('models.download.waiting')}</p> : null}
    </div>
  )
}

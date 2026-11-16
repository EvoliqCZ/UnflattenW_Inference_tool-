import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { formatMebibytes } from '@/lib/format'
import type { SupportedLanguage } from '@/i18n'

export interface VramMeterProps {
  totalMiB: number
  freeMiB?: number
  estimateMiB?: number
  compact?: boolean
  language: SupportedLanguage
}

/**
 * UX §3.7 "nezmestí sa": a capacity bar with an estimated-peak overlay. When
 * `estimateMiB` exceeds `totalMiB` the overshoot renders as a hatched bar
 * past the capacity line instead of clipping silently.
 */
export function VramMeter({ totalMiB, freeMiB, estimateMiB, compact, language }: VramMeterProps) {
  const { t } = useTranslation()
  const usedMiB = freeMiB !== undefined ? Math.max(totalMiB - freeMiB, 0) : undefined
  const estimatePct = estimateMiB !== undefined && totalMiB > 0 ? Math.min((estimateMiB / totalMiB) * 100, 100) : null
  const overshoot = estimateMiB !== undefined && estimateMiB > totalMiB

  return (
    <div
      className={cn('flex flex-col gap-1', compact ? 'w-32' : 'w-full')}
      role="img"
      aria-label={t('models.vram.label')}
    >
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-surface-sunken">
        {usedMiB !== undefined ? (
          <div
            className="absolute inset-y-0 left-0 rounded-full bg-muted-foreground/40"
            style={{ width: `${String(Math.min((usedMiB / totalMiB) * 100, 100))}%` }}
          />
        ) : null}
        {estimatePct !== null ? (
          <div
            className={cn('absolute inset-y-0 left-0 rounded-full', overshoot ? 'bg-destructive/70' : 'bg-accent')}
            style={{ width: `${String(estimatePct)}%` }}
          />
        ) : null}
        {overshoot ? (
          <div
            className="absolute inset-y-0 right-0 w-2 bg-[repeating-linear-gradient(45deg,theme(colors.destructive.DEFAULT)_0,theme(colors.destructive.DEFAULT)_2px,transparent_2px,transparent_4px)]"
            aria-hidden="true"
          />
        ) : null}
      </div>
      {!compact ? (
        <p className="font-mono text-xs text-muted-foreground">
          {freeMiB !== undefined
            ? t('models.vram.freeOfTotal', {
                free: formatMebibytes(freeMiB, language),
                total: formatMebibytes(totalMiB, language),
              })
            : formatMebibytes(totalMiB, language)}
        </p>
      ) : null}
    </div>
  )
}

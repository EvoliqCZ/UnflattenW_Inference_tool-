import { CheckCircle2, Circle, Loader2, XCircle } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'
import { formatDuration } from '@/lib/time'
import type { SupportedLanguage } from '@/i18n'
import { estimateRemainingMinutes } from './pageProgress'

type PageReport = components['schemas']['PageReport']

export interface PageProgressListProps {
  pagesTotal: number
  pagesDone: number
  currentPage: number | null
  pages: PageReport[] | null
  language: SupportedLanguage
}

const SCROLLABLE_ABOVE = 8

/** UX §3.4 "Viacstranový priebeh": per-page rows plus a remaining-time estimate. */
export function PageProgressList({ pagesTotal, pagesDone, currentPage, pages, language }: PageProgressListProps) {
  const { t } = useTranslation()
  const failedCount = pages?.filter((page) => page.state === 'failed').length ?? 0
  const estimateMinutes = estimateRemainingMinutes(pages, pagesDone, pagesTotal)

  const rows = Array.from({ length: pagesTotal }, (_, index) => {
    const report = pages?.find((page) => page.index === index) ?? null
    return { index, report }
  })

  const list = (
    <ol className="flex flex-col gap-1">
      {rows.map(({ index, report }) => (
        <PageProgressRow
          key={index}
          index={index}
          report={report}
          isCurrent={index === currentPage}
          language={language}
        />
      ))}
    </ol>
  )

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between text-sm">
        <span>
          {t('job.progress.pageSummary', {
            current: (currentPage ?? pagesDone) + 1,
            total: pagesTotal,
            done: pagesDone,
            failed: failedCount,
          })}
        </span>
        {estimateMinutes !== null ? (
          <span className="text-muted-foreground">{t('job.progress.etaMinutes', { minutes: estimateMinutes })}</span>
        ) : null}
      </div>
      <Progress
        value={pagesTotal > 0 ? (pagesDone / pagesTotal) * 100 : 0}
        valueLabel={t('job.progress.pageSummary', {
          current: (currentPage ?? pagesDone) + 1,
          total: pagesTotal,
          done: pagesDone,
          failed: failedCount,
        })}
      />
      {pagesTotal > SCROLLABLE_ABOVE ? (
        <div
          role="region"
          aria-label={t('job.progress.pages')}
          tabIndex={0}
          className="max-h-80 overflow-y-auto overscroll-contain pr-3"
        >
          {list}
        </div>
      ) : (
        list
      )}
    </div>
  )
}

function PageProgressRow({
  index,
  report,
  isCurrent,
  language,
}: {
  index: number
  report: PageReport | null
  isCurrent: boolean
  language: SupportedLanguage
}) {
  const { t } = useTranslation()

  if (report?.state === 'succeeded') {
    return (
      <li className="flex items-center gap-2 text-sm">
        <CheckCircle2 aria-hidden="true" className="size-4 text-success-text" />
        <span>{t('job.artifacts.page', { n: index + 1 })}</span>
        <span className="font-mono text-xs text-muted-foreground">{formatDuration(report.seconds, language)}</span>
        {report.decode_steps !== null ? (
          <span className="font-mono text-xs text-muted-foreground">{report.decode_steps}</span>
        ) : null}
      </li>
    )
  }

  if (report?.state === 'failed') {
    return (
      <li className="flex items-center gap-2 text-sm text-destructive">
        <XCircle aria-hidden="true" className="size-4" />
        <span>{t('job.artifacts.page', { n: index + 1 })}</span>
        <span className="font-mono text-xs">{report.error_code}</span>
      </li>
    )
  }

  if (isCurrent) {
    return (
      <li className="flex items-center gap-2 text-sm text-accent">
        <Loader2 aria-hidden="true" className="size-4 motion-safe:animate-spin" />
        <span>{t('job.artifacts.page', { n: index + 1 })}</span>
        <span className="text-xs">{t('job.progress.page.running')}</span>
      </li>
    )
  }

  return (
    <li className={cn('flex items-center gap-2 text-sm text-muted-foreground')}>
      <Circle aria-hidden="true" className="size-4" />
      <span>{t('job.artifacts.page', { n: index + 1 })}</span>
      <span className="text-xs">{t('job.progress.page.waiting')}</span>
    </li>
  )
}

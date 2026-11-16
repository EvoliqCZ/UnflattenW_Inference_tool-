import { ArrowLeft } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router'
import type { components } from '@/api/schema'
import { Button } from '@/components/ui/button'
import { JobStateBadge } from '@/components/JobStateBadge'
import { CancelJobDialog } from '@/features/jobs/CancelJobDialog'
import { isJobActive, jobDisplayName } from '@/features/jobs/jobRowHelpers'
import { PurgeJobDialog } from '@/features/jobs/PurgeJobDialog'
import type { SupportedLanguage } from '@/i18n'
import { formatDuration, formatRelativeTime } from '@/lib/time'

type ConversionStatus = components['schemas']['ConversionStatus']

export interface JobHeaderProps {
  job: ConversionStatus
  language: SupportedLanguage
}

/** UX §6 "Detail úlohy" header: label, mono meta line, state badge and per-state actions. */
export function JobHeader({ job, language }: JobHeaderProps) {
  const { t } = useTranslation()
  const [now, setNow] = useState(() => new Date())
  const [cancelOpen, setCancelOpen] = useState(false)
  const [purgeOpen, setPurgeOpen] = useState(false)
  const active = isJobActive(job.state)

  useEffect(() => {
    if (!active) return
    const id = window.setInterval(() => {
      setNow(new Date())
    }, 1000)
    return () => {
      window.clearInterval(id)
    }
  }, [active])

  const createdAt = new Date(job.created_at)
  const timeLabel = job.terminal_at
    ? `${t('job.duration')} ${formatDuration((new Date(job.terminal_at).getTime() - createdAt.getTime()) / 1000, language)}`
    : `${t('job.elapsed')} ${formatDuration(Math.max((now.getTime() - createdAt.getTime()) / 1000, 0), language)}`

  return (
    <header className="flex flex-col gap-2">
      <Link to="/jobs" className="inline-flex w-fit items-center gap-1 text-sm text-muted-foreground hover:underline">
        <ArrowLeft aria-hidden="true" className="size-4" />
        {t('job.header.back')}
      </Link>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-xl font-semibold">{jobDisplayName(job)}</h1>
        <div className="flex items-center gap-2">
          <JobStateBadge state={job.state} />
          {active ? (
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                setCancelOpen(true)
              }}
            >
              {t('jobs.action.cancel')}
            </Button>
          ) : (
            <>
              {job.state === 'failed' ? (
                <Button asChild variant="outline">
                  <Link to="/">{t('job.header.reupload')}</Link>
                </Button>
              ) : null}
              <Button
                type="button"
                variant="outline"
                className="text-destructive"
                onClick={() => {
                  setPurgeOpen(true)
                }}
              >
                {t('jobs.action.purge')}
              </Button>
            </>
          )}
        </div>
      </div>
      <p className="font-mono text-xs text-muted-foreground">
        {job.job_id} · {job.model_variant ?? '—'} · {formatRelativeTime(createdAt, now, language)} · {timeLabel}
      </p>

      {cancelOpen ? <CancelJobDialog job={job} open={cancelOpen} onOpenChange={setCancelOpen} /> : null}
      {purgeOpen ? <PurgeJobDialog job={job} open={purgeOpen} onOpenChange={setPurgeOpen} /> : null}
    </header>
  )
}

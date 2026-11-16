import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router'
import { useJobQuery, useModelsQuery } from '@/api/queries'
import { DownloadProgress } from '@/features/models/DownloadProgress'
import { fetchJobsList } from '@/features/jobs/api'
import { jobDisplayName } from '@/features/jobs/jobRowHelpers'
import type { SupportedLanguage } from '@/i18n'

export interface QueueReasonProps {
  jobId: string
}

const DOWNLOADING_STATES = new Set(['downloading', 'verifying'])

/**
 * UX §3.4 "Dôvod čakania pri queued": a client-side guess, not a server
 * claim, checked in priority order: (1) the job's chosen variant is still
 * downloading/verifying in `/v1/models` 1.1 (`download.state`, now on
 * master since T40) -- shown with a compact `DownloadProgress`; (2) another
 * job is `running`; (3) falls through to "waiting for a worker".
 */
export function QueueReason({ jobId }: QueueReasonProps) {
  const { t, i18n } = useTranslation()
  const language = i18n.language as SupportedLanguage
  const jobQuery = useJobQuery(jobId)
  const modelsQuery = useModelsQuery()
  const otherJobQuery = useQuery({
    queryKey: ['jobs', 'other-running', jobId],
    queryFn: () => fetchJobsList({ states: ['running'], limit: 5 }),
    select: (data) => data.items.find((job) => job.job_id !== jobId) ?? null,
  })

  const modelVariant = jobQuery.data?.model_variant ?? null
  const variant = modelsQuery.data?.variants.find((entry) => entry.variant_id === modelVariant) ?? null
  const download = variant?.download ?? null

  if (download !== null && DOWNLOADING_STATES.has(download.state)) {
    return (
      <div role="status" className="flex flex-col gap-1 text-sm text-muted-foreground">
        <p>{t('job.queue.waitingDownload', { label: variant?.label })}</p>
        <DownloadProgress
          state={download.state}
          bytesDone={download.bytes_done}
          bytesTotal={download.bytes_total}
          errorCode={download.error_code}
          language={language}
        />
      </div>
    )
  }

  const otherJob = otherJobQuery.data ?? null

  if (otherJob) {
    return (
      <p role="status" className="text-sm text-muted-foreground">
        {t('job.queue.waitingJob')} —{' '}
        <Link to={`/jobs/${otherJob.job_id}`} className="underline underline-offset-4">
          {jobDisplayName(otherJob)}
        </Link>
      </p>
    )
  }

  return (
    <p role="status" className="text-sm text-muted-foreground">
      {t('job.queue.waitingWorker')}
    </p>
  )
}

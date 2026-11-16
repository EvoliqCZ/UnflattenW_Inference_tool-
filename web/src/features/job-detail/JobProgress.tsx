import type { components } from '@/api/schema'
import type { SupportedLanguage } from '@/i18n'
import { PageProgressList } from './PageProgressList'
import { QueueReason } from './QueueReason'
import { StageStepper } from './StageStepper'

type ConversionStatus = components['schemas']['ConversionStatus']

export interface JobProgressProps {
  job: ConversionStatus
  language: SupportedLanguage
}

/**
 * UX §3.4/§7.2: active-job view -- `QueueReason` while queued, the 5-step
 * `StageStepper` always, `PageProgressList` only for a multi-page job
 * (`pages_total !== null`).
 */
export function JobProgress({ job, language }: JobProgressProps) {
  return (
    <div className="flex flex-col gap-4">
      {job.state === 'queued' ? <QueueReason jobId={job.job_id} /> : null}
      <StageStepper stage={job.stage} state={job.state} requestedOutput={job.requested_output} />
      {job.pages_total !== null ? (
        <PageProgressList
          pagesTotal={job.pages_total}
          pagesDone={job.pages_done ?? 0}
          currentPage={job.current_page}
          pages={job.pages}
          language={language}
        />
      ) : null}
    </div>
  )
}

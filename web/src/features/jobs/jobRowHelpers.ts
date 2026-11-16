import type { JobState } from '@/i18n/codes'
import type { JobSummary } from './api'

const ACTIVE_STATES: readonly JobState[] = ['queued', 'running']

/** UX §3.5/§6: only an active job can be cancelled; only a terminal one can be purged. */
export function isJobActive(state: JobState): boolean {
  return ACTIVE_STATES.includes(state)
}

/** UX §6 "Strany" column: `done/total` while active, else just `total`, `—` when unknown. */
export function formatPagesColumn(job: Pick<JobSummary, 'state' | 'pages_done' | 'pages_total'>): string {
  if (job.pages_total === null) {
    return '—'
  }
  if (isJobActive(job.state)) {
    return `${String(job.pages_done ?? 0)} / ${String(job.pages_total)}`
  }
  return String(job.pages_total)
}

/** The label shown for a job row: its label, or the raw job id when unlabelled. */
export function jobDisplayName(job: Pick<JobSummary, 'label' | 'job_id'>): string {
  return job.label ?? job.job_id
}

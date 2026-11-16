import { Link } from 'react-router'
import { ExpiryCountdown } from '@/components/ExpiryCountdown'
import { JobStateBadge } from '@/components/JobStateBadge'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import type { SupportedLanguage } from '@/i18n'
import { formatRelativeTime } from '@/lib/time'
import { formatPagesColumn, jobDisplayName } from './jobRowHelpers'
import { JobRowMenu } from './JobRowMenu'
import type { JobSummary } from './api'

export interface JobCardListProps {
  jobs: readonly JobSummary[]
  language: SupportedLanguage
  onCancel: (job: JobSummary) => void
  onPurge: (job: JobSummary) => void
}

/** UX §6 "< 768px": card list -- title + state on line 1, model/pages/time (mono) on line 2. */
export function JobCardList({ jobs, language, onCancel, onPurge }: JobCardListProps) {
  const now = new Date()

  return (
    <ul className="flex flex-col gap-2">
      {jobs.map((job) => (
        <li key={job.job_id}>
          <Card>
            <CardHeader className="flex-row items-center justify-between gap-2 space-y-0">
              <Link to={`/jobs/${job.job_id}`} className="font-medium underline-offset-4 hover:underline">
                {jobDisplayName(job)}
              </Link>
              <div className="flex items-center gap-2">
                <JobStateBadge state={job.state} />
                <JobRowMenu
                  job={job}
                  onCancel={() => {
                    onCancel(job)
                  }}
                  onPurge={() => {
                    onPurge(job)
                  }}
                />
              </div>
            </CardHeader>
            <CardContent className="flex items-center justify-between font-mono text-xs text-muted-foreground">
              <span>
                {job.model_variant ?? '—'} · {formatPagesColumn(job)} ·{' '}
                {formatRelativeTime(new Date(job.created_at), now, language)}
              </span>
              <ExpiryCountdown expiresAt={job.expires_at} compact />
            </CardContent>
          </Card>
        </li>
      ))}
    </ul>
  )
}

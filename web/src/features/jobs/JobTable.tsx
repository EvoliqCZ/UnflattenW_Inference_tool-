import { useTranslation } from 'react-i18next'
import { Link } from 'react-router'
import { ExpiryCountdown } from '@/components/ExpiryCountdown'
import { JobStateBadge } from '@/components/JobStateBadge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import type { SupportedLanguage } from '@/i18n'
import { formatDateTime, formatRelativeTime } from '@/lib/time'
import { formatPagesColumn, jobDisplayName } from './jobRowHelpers'
import { JobRowMenu } from './JobRowMenu'
import type { JobSummary } from './api'

export interface JobTableProps {
  jobs: readonly JobSummary[]
  language: SupportedLanguage
  onCancel: (job: JobSummary) => void
  onPurge: (job: JobSummary) => void
}

/** UX §6/§11.4: desktop table, `>= 1024px` per the shell breakpoint (design §8.7). */
export function JobTable({ jobs, language, onCancel, onPurge }: JobTableProps) {
  const { t } = useTranslation()
  const now = new Date()

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>{t('jobs.column.label')}</TableHead>
          <TableHead>{t('jobs.column.pages')}</TableHead>
          <TableHead>{t('jobs.column.model')}</TableHead>
          <TableHead>{t('jobs.column.state')}</TableHead>
          <TableHead>{t('jobs.column.created')}</TableHead>
          <TableHead>{t('jobs.column.expires')}</TableHead>
          <TableHead>
            <span className="sr-only">{t('jobs.column.actions')}</span>
          </TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {jobs.map((job) => (
          <TableRow key={job.job_id}>
            <TableCell>
              <Link to={`/jobs/${job.job_id}`} className="font-medium underline-offset-4 hover:underline">
                {jobDisplayName(job)}
              </Link>
            </TableCell>
            <TableCell>{formatPagesColumn(job)}</TableCell>
            <TableCell>{job.model_variant ?? '—'}</TableCell>
            <TableCell>
              <JobStateBadge state={job.state} />
            </TableCell>
            <TableCell title={formatDateTime(new Date(job.created_at), language)}>
              {formatRelativeTime(new Date(job.created_at), now, language)}
            </TableCell>
            <TableCell>
              <ExpiryCountdown expiresAt={job.expires_at} compact />
            </TableCell>
            <TableCell>
              <JobRowMenu
                job={job}
                onCancel={() => {
                  onCancel(job)
                }}
                onPurge={() => {
                  onPurge(job)
                }}
              />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

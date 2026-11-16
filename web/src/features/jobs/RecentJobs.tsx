import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router'
import { JobStateBadge } from '@/components/JobStateBadge'
import { fetchJobsList } from './api'
import { jobDisplayName } from './jobRowHelpers'

const RECENT_JOBS_LIMIT = 3

/** UX §4.1 "Posledné úlohy": the last three jobs, no filter, no pagination. */
export function RecentJobs() {
  const { t } = useTranslation()
  const query = useQuery({
    queryKey: ['jobs', { state: 'recent' }],
    queryFn: () => fetchJobsList({ limit: RECENT_JOBS_LIMIT }),
  })
  const jobs = query.data?.items ?? []

  if (jobs.length === 0) {
    return null
  }

  return (
    <section aria-label={t('convert.recentJobs.title')} className="flex flex-col gap-2">
      <h2 className="text-sm font-medium text-foreground">{t('convert.recentJobs.title')}</h2>
      <ul className="flex flex-col gap-1">
        {jobs.map((job) => (
          <li key={job.job_id} className="flex items-center justify-between gap-2 text-sm">
            <Link to={`/jobs/${job.job_id}`} className="truncate underline-offset-4 hover:underline">
              {jobDisplayName(job)}
            </Link>
            <JobStateBadge state={job.state} />
          </li>
        ))}
      </ul>
      <Link to="/jobs" className="text-sm text-accent underline-offset-4 hover:underline">
        {t('convert.recentJobs.viewAll')}
      </Link>
    </section>
  )
}

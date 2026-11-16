import { MoreHorizontal } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { isJobActive } from './jobRowHelpers'
import type { JobSummary } from './api'

export interface JobRowMenuProps {
  job: Pick<JobSummary, 'job_id' | 'state'>
  onCancel: () => void
  onPurge: () => void
}

/** UX §6 "Úlohy" row menu: Otvoriť always, Zrušiť while active, Zmazať once terminal. */
export function JobRowMenu({ job, onCancel, onPurge }: JobRowMenuProps) {
  const { t } = useTranslation()
  const active = isJobActive(job.state)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button type="button" variant="ghost" size="icon" aria-label={t('jobs.column.actions')}>
          <MoreHorizontal aria-hidden="true" className="size-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem asChild>
          <Link to={`/jobs/${job.job_id}`}>{t('jobs.action.open')}</Link>
        </DropdownMenuItem>
        {active ? (
          <DropdownMenuItem onSelect={onCancel}>{t('jobs.action.cancel')}</DropdownMenuItem>
        ) : (
          <DropdownMenuItem onSelect={onPurge} className="text-destructive">
            {t('jobs.action.purge')}
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

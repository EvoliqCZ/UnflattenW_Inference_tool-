import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import type { SupportedLanguage } from '@/i18n'
import { CancelJobDialog } from './CancelJobDialog'
import { JobCardList } from './JobCardList'
import { JobFilters } from './JobFilters'
import { JobTable } from './JobTable'
import { PurgeJobDialog } from './PurgeJobDialog'
import { useJobsListQuery, type JobFilterId, type JobSummary } from './api'
import { jobDisplayName } from './jobRowHelpers'

const DESKTOP_QUERY = '(min-width: 768px)'

/**
 * UX §6: table `>= 768px`, card list below. A single `matchMedia`-driven
 * choice (not both rendered with `hidden`/CSS) -- otherwise every job would
 * exist twice in the accessibility tree (`JobStateBadge` text, row actions),
 * which also fires "multiple elements" everywhere in tests.
 */
function useIsDesktop(): boolean {
  const [isDesktop, setIsDesktop] = useState(() => window.matchMedia(DESKTOP_QUERY).matches)

  useEffect(() => {
    const media = window.matchMedia(DESKTOP_QUERY)
    const apply = () => {
      setIsDesktop(media.matches)
    }
    apply()
    media.addEventListener('change', apply)
    return () => {
      media.removeEventListener('change', apply)
    }
  }, [])

  return isDesktop
}

/** UX §6 "Úlohy": filters, table/card list, load-more, and the two destructive dialogs. */
export function JobsPage() {
  const { t, i18n } = useTranslation()
  const language = i18n.language as SupportedLanguage
  const [filter, setFilter] = useState<JobFilterId>('all')
  const [search, setSearch] = useState('')
  const [cancelTarget, setCancelTarget] = useState<JobSummary | null>(null)
  const [purgeTarget, setPurgeTarget] = useState<JobSummary | null>(null)

  const isDesktop = useIsDesktop()
  const query = useJobsListQuery(filter)
  const jobs = useMemo(() => query.data?.pages.flatMap((page) => page.items) ?? [], [query.data])

  const normalizedSearch = search.trim().toLowerCase()
  const visibleJobs =
    normalizedSearch === '' ? jobs : jobs.filter((job) => jobDisplayName(job).toLowerCase().includes(normalizedSearch))

  return (
    <main className="mx-auto flex max-w-5xl flex-col gap-4 p-4">
      <h1 className="text-xl font-semibold">{t('jobs.title')}</h1>

      <JobFilters filter={filter} onFilterChange={setFilter} search={search} onSearchChange={setSearch} />

      {query.isPending ? (
        <div className="flex flex-col gap-2">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      ) : visibleJobs.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          {jobs.length === 0 && filter === 'all' ? t('jobs.empty') : t('jobs.emptyFilter')}
        </p>
      ) : isDesktop ? (
        <JobTable jobs={visibleJobs} language={language} onCancel={setCancelTarget} onPurge={setPurgeTarget} />
      ) : (
        <JobCardList jobs={visibleJobs} language={language} onCancel={setCancelTarget} onPurge={setPurgeTarget} />
      )}

      {query.hasNextPage ? (
        <Button
          type="button"
          variant="outline"
          className="self-center"
          onClick={() => void query.fetchNextPage()}
          aria-disabled={query.isFetchingNextPage}
        >
          {t('jobs.loadMore')}
        </Button>
      ) : null}

      {cancelTarget ? (
        <CancelJobDialog
          job={cancelTarget}
          open={true}
          onOpenChange={(open) => {
            if (!open) setCancelTarget(null)
          }}
        />
      ) : null}
      {purgeTarget ? (
        <PurgeJobDialog
          job={purgeTarget}
          open={true}
          onOpenChange={(open) => {
            if (!open) setPurgeTarget(null)
          }}
        />
      ) : null}
    </main>
  )
}

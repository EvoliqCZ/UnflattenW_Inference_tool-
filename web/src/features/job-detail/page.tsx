import { useTranslation } from 'react-i18next'
import { Link, useParams, useSearchParams } from 'react-router'
import { useJobQuery } from '@/api/queries'
import { ProblemError } from '@/api/problem'
import { ProblemMessage } from '@/components/ProblemMessage'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ComparisonViewer } from '@/features/compare/ComparisonViewer'
import type { SupportedLanguage } from '@/i18n'
import { DiagnosticsPanel } from './DiagnosticsPanel'
import { JobHeader } from './JobHeader'
import { JobProgress } from './JobProgress'
import { ResultPanel } from './ResultPanel'
import { useProgressAnnouncement } from './useProgressAnnouncement'
import { useTerminalToast } from './useTerminalToast'

const TERMINAL_STATES = new Set(['succeeded', 'failed', 'cancelled', 'expired'])
const TABS = ['result', 'compare', 'diagnostics'] as const
type DetailTab = (typeof TABS)[number]

function isDetailTab(value: string | null): value is DetailTab {
  return (TABS as readonly string[]).includes(value ?? '')
}

/**
 * UX §6/§7.2 "/jobs/:jobId": progress while active, `Výsledok`/`Porovnanie`/
 * `Diagnostika` tabs once terminal (state kept in `?tab=`, design §7.1/§7.5
 * convention for restorable screen state). T24 -> T28/T36 scope note: once
 * a job's artefacts have expired, `ComparisonViewer` is not rendered at all
 * (it has no `expires_at`/`expired` prop to tell it) -- the simpler of the
 * two options that note left open.
 */
export function JobDetailPage() {
  const { t, i18n } = useTranslation()
  const language = i18n.language as SupportedLanguage
  const { jobId } = useParams<{ jobId: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const query = useJobQuery(jobId ?? '')
  const job = query.data

  useTerminalToast(job)
  const announcement = useProgressAnnouncement(job)

  const requestedTab = searchParams.get('tab')
  const activeTab: DetailTab = isDetailTab(requestedTab) ? requestedTab : 'result'

  if (query.isPending) {
    return (
      <main className="mx-auto flex max-w-4xl flex-col gap-4 p-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-24 w-full" />
      </main>
    )
  }

  if (query.isError) {
    if (query.error instanceof ProblemError && query.error.code === 'RESOURCE_NOT_FOUND') {
      return (
        <main className="mx-auto flex max-w-4xl flex-col items-start gap-2 p-4">
          <p>{t('job.notFound')}</p>
          <Link to="/jobs" className="text-accent underline-offset-4 hover:underline">
            {t('nav.jobs')}
          </Link>
        </main>
      )
    }
    return (
      <main className="mx-auto max-w-4xl p-4">
        <ProblemMessage
          problem={query.error instanceof ProblemError ? query.error.problem : { code: 'INTERNAL_ERROR' }}
          context="job"
          onRetry={() => void query.refetch()}
        />
      </main>
    )
  }

  if (job === undefined) return <></>

  const isTerminal = TERMINAL_STATES.has(job.state)
  const isExpired = job.state === 'expired'

  return (
    <main className="mx-auto flex max-w-4xl flex-col gap-4 p-4">
      <div aria-live="polite" className="sr-only">
        {announcement}
      </div>

      <JobHeader job={job} language={language} />

      {isTerminal ? (
        <Tabs
          value={activeTab}
          onValueChange={(value) => {
            setSearchParams((params) => {
              params.set('tab', value)
              return params
            })
          }}
        >
          <TabsList>
            {TABS.map((tab) => (
              <TabsTrigger key={tab} value={tab}>
                {t(`job.tab.${tab}`)}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      ) : (
        <JobProgress job={job} language={language} />
      )}

      {isTerminal && activeTab === 'result' ? <ResultPanel job={job} /> : null}
      {isTerminal && activeTab === 'compare' ? (
        isExpired ? (
          <p className="text-sm text-muted-foreground">{t('job.compareExpired')}</p>
        ) : (
          <ComparisonViewer
            artifacts={job.artifacts.filter(
              (artifact): artifact is Exclude<typeof artifact, string> => typeof artifact !== 'string',
            )}
            outcome={job.diagnostic_outcome}
            pages={job.pages}
          />
        )
      ) : null}
      {isTerminal && activeTab === 'diagnostics' ? (
        <DiagnosticsPanel outcome={job.diagnostic_outcome} inputKind={job.input_kind} />
      ) : null}
    </main>
  )
}

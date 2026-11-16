import { useQuery } from '@tanstack/react-query'
import type { JobState } from '@/i18n/codes'
import { apiClient, call } from './client'
import { retryDelay, shouldRetry, TransportError } from './problem'
import type { components } from './schema'

type ConversionStatus = components['schemas']['ConversionStatus']
type ModelsResponse = components['schemas']['ModelsResponse']

/**
 * Query key factories, one per resource in design §7.3.
 *
 * `session` and `settings` are keyed here for the future `auth/session.tsx`
 * and settings-screen tasks to invalidate on login/logout/401 and on save,
 * but this task does not build hooks for them:
 * - `GET /v1/session` landed with T21 (master, not yet merged into this
 *   branch), so it is not in this branch's `schema.d.ts` yet. `POST
 *   /v1/session`, `POST /v1/setup` and `PUT /v1/setup/password` parse their
 *   JSON bodies by hand server-side (`api.py`), so they carry no OpenAPI
 *   `requestBody` even once merged -- whichever task wires them must type
 *   `{ password: string }` (session/setup) and `{ current_password,
 *   new_password }` (setup/password) from `schemas.py`/design §2.4, not from
 *   `schema.d.ts`.
 * - `GET /v1/settings` does not exist yet on any branch.
 */
export const queryKeys = {
  session: () => ['session'] as const,
  readiness: () => ['readiness'] as const,
  jobs: (filter: { state?: string } = {}) => ['jobs', filter] as const,
  job: (jobId: string) => ['job', jobId] as const,
  models: () => ['models'] as const,
  settings: () => ['settings'] as const,
}

const READY_INTERVAL_MS = 30000
const NOT_READY_INTERVAL_MS = 5000

/** design §7.3: 5s while not ready, 30s once the service reports ready. */
export function readinessRefetchInterval(serviceReady: boolean): number {
  return serviceReady ? READY_INTERVAL_MS : NOT_READY_INTERVAL_MS
}

const ACTIVE_JOBS_LIST_INTERVAL_MS = 2000

/** design §7.3: poll the jobs list only while it holds an active job. */
export function jobsListRefetchInterval(hasActiveJob: boolean): number | false {
  return hasActiveJob ? ACTIVE_JOBS_LIST_INTERVAL_MS : false
}

const ACTIVE_JOB_INTERVAL_MS = 1500
const ACTIVE_JOB_STATES: readonly JobState[] = ['queued', 'running']

/** design §7.3: poll a single job only while `state ∈ {queued, running}`. */
export function jobRefetchInterval(state: JobState): number | false {
  return ACTIVE_JOB_STATES.includes(state) ? ACTIVE_JOB_INTERVAL_MS : false
}

const ACTIVE_MODEL_DOWNLOAD_INTERVAL_MS = 2000
const IDLE_MODELS_INTERVAL_MS = 15000
const ACTIVE_DOWNLOAD_STATES = new Set(['downloading', 'verifying', 'loading'])

/**
 * design §7.3: 2s while any variant's `download.state` is downloading,
 * verifying or loading, else 15s. A `load` command in flight also surfaces
 * as `download.state = 'loading'` (`ModelVariantDownload`, design §3.8), so
 * this one check covers UX §3.7 point 1's "or a `load` command is pending"
 * clause too.
 */
export function modelsRefetchInterval(downloadStates: readonly string[]): number {
  return downloadStates.some((state) => ACTIVE_DOWNLOAD_STATES.has(state))
    ? ACTIVE_MODEL_DOWNLOAD_INTERVAL_MS
    : IDLE_MODELS_INTERVAL_MS
}

/** Extracts each variant's `download.state` from a models response, for `modelsRefetchInterval`. */
export function modelDownloadStates(models: ModelsResponse | undefined): string[] {
  const states: string[] = []
  for (const variant of models?.variants ?? []) {
    if (variant.download !== null) {
      states.push(variant.download.state)
    }
  }
  return states
}

/** design §7.3/§7.4: retry transport failures and >=500 problems with backoff. */
const retryOptions = {
  retry: (_failureCount: number, error: unknown) => shouldRetry(error),
  retryDelay,
}

export function useJobQuery(jobId: string) {
  return useQuery({
    queryKey: queryKeys.job(jobId),
    queryFn: () =>
      call<ConversionStatus>(apiClient.GET('/v1/conversions/{job_id}', { params: { path: { job_id: jobId } } })),
    refetchInterval: (query) => jobRefetchInterval(query.state.data?.state ?? 'queued'),
    ...retryOptions,
  })
}

export function useModelsQuery() {
  return useQuery({
    queryKey: queryKeys.models(),
    queryFn: () => call<ModelsResponse>(apiClient.GET('/v1/models')),
    refetchInterval: (query) => modelsRefetchInterval(modelDownloadStates(query.state.data)),
    ...retryOptions,
  })
}

/**
 * `GET /v1/readiness` is registered but not yet given a typed OpenAPI
 * response model (T13 predates T15's `responses=` pass for this route), so
 * `schema.d.ts` types its body as `unknown`. Narrowed locally to the one
 * field this task needs; the full contract belongs to the readiness screen
 * task once it exists in `schema.d.ts`.
 */
/** UX §4.3: the browser-side validation limits (`readiness.limits`), when the server has published them. */
export interface ReadinessLimits {
  max_pages: number
  max_upload_bytes: number
  accepted_media_types: readonly string[]
}

interface ReadinessReport {
  service_ready: boolean
  limits?: ReadinessLimits
}

function isReadinessReport(value: unknown): value is ReadinessReport {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as { service_ready?: unknown }).service_ready === 'boolean'
  )
}

export function useReadinessQuery() {
  return useQuery({
    queryKey: queryKeys.readiness(),
    queryFn: async () => {
      const data = await call<unknown>(apiClient.GET('/v1/readiness'))
      if (!isReadinessReport(data)) {
        throw new TransportError()
      }
      return data
    },
    refetchInterval: (query) => readinessRefetchInterval(query.state.data?.service_ready ?? false),
    ...retryOptions,
  })
}

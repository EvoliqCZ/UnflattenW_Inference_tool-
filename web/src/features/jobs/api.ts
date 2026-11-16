import { type InfiniteData, useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient, call } from '@/api/client'
import { isProblemResponse, ProblemError, retryDelay, shouldRetry, TransportError } from '@/api/problem'
import { jobsListRefetchInterval, queryKeys } from '@/api/queries'
import type { JobState } from '@/i18n/codes'
import type { components } from '@/api/schema'

export type JobList = components['schemas']['JobList']
export type JobSummary = components['schemas']['JobSummary']
export type ConversionStatus = components['schemas']['ConversionStatus']

/**
 * UX §6 "Úlohy" filter tabs. `all` sends no `state=` query param at all
 * (design §3.4: "Bez... " -- an absent filter means every state); every
 * other tab maps to one or more repeated `state=` values.
 */
export const JOB_FILTERS = ['all', 'active', 'succeeded', 'failed', 'ended'] as const
export type JobFilterId = (typeof JOB_FILTERS)[number]

const FILTER_STATES: Record<JobFilterId, readonly JobState[] | undefined> = {
  all: undefined,
  active: ['queued', 'running'],
  succeeded: ['succeeded'],
  failed: ['failed'],
  ended: ['cancelled', 'expired'],
}

export function statesForFilter(filter: JobFilterId): readonly JobState[] | undefined {
  return FILTER_STATES[filter]
}

const JOBS_LIST_LIMIT = 25

/**
 * Pure query-string builder for `GET /v1/conversions` (design §3.4): `state`
 * is repeatable, `limit` and `cursor` are singular. `api.py` reads these off
 * the raw `Request` rather than declared FastAPI `Query()` params, so they
 * never made it into `openapi.json` / `schema.d.ts` -- this function is the
 * one place that encodes the wire contract, and callers below embed its
 * output directly into the (otherwise query-less) typed path.
 */
export interface JobsListSearchParams {
  states: readonly JobState[] | undefined
  cursor: string | null | undefined
  limit: number | undefined
}

export function buildJobsListSearch(params: Partial<JobsListSearchParams>): string {
  const search = new URLSearchParams()
  for (const state of params.states ?? []) {
    search.append('state', state)
  }
  if (params.limit !== undefined) {
    search.set('limit', String(params.limit))
  }
  if (params.cursor) {
    search.set('cursor', params.cursor)
  }
  return search.toString()
}

export async function fetchJobsList(params: Partial<JobsListSearchParams> = {}): Promise<JobList> {
  const search = buildJobsListSearch({
    states: params.states,
    cursor: params.cursor,
    limit: params.limit ?? JOBS_LIST_LIMIT,
  })
  const path = (search ? `/v1/conversions?${search}` : '/v1/conversions') as '/v1/conversions'
  return call<JobList>(apiClient.GET(path))
}

export async function cancelJob(jobId: string): Promise<ConversionStatus> {
  return call<ConversionStatus>(apiClient.DELETE('/v1/conversions/{job_id}', { params: { path: { job_id: jobId } } }))
}

/**
 * `DELETE /v1/conversions/{job_id}?mode=purge` (design §3.4/§6): 204 on
 * success. Not routed through `call()` -- that helper treats an `undefined`
 * body as a transport failure (correct for the JSON endpoints it was built
 * for), which would misclassify every successful purge.
 */
export async function purgeJob(jobId: string): Promise<void> {
  const { error, response } = await apiClient.DELETE(
    '/v1/conversions/{job_id}?mode=purge' as '/v1/conversions/{job_id}',
    { params: { path: { job_id: jobId } } },
  )
  if (error !== undefined) {
    if (isProblemResponse(response)) {
      throw new ProblemError(error)
    }
    throw new TransportError(response.status)
  }
  if (!response.ok) {
    throw new TransportError(response.status)
  }
}

const retryOptions = {
  retry: (_failureCount: number, error: unknown) => shouldRetry(error),
  retryDelay,
}

function hasActiveJob(data: InfiniteData<JobList> | undefined): boolean {
  const items = data?.pages.flatMap((page) => page.items) ?? []
  return items.some((job) => job.state === 'queued' || job.state === 'running')
}

/** design §7.3: `useInfiniteQuery` keyed by filter, cursor-paginated, polling only while a page holds an active job. */
export function useJobsListQuery(filter: JobFilterId) {
  const states = statesForFilter(filter)
  return useInfiniteQuery({
    queryKey: queryKeys.jobs({ state: filter }),
    queryFn: ({ pageParam }: { pageParam: string | null }) =>
      fetchJobsList({ states, cursor: pageParam, limit: JOBS_LIST_LIMIT }),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage: JobList) => lastPage.next_cursor,
    refetchInterval: (query) => jobsListRefetchInterval(hasActiveJob(query.state.data)),
    ...retryOptions,
  })
}

export function useCancelJobMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) => cancelJob(jobId),
    onSuccess: (_status, jobId) => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.job(jobId) })
      void queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

export function usePurgeJobMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) => purgeJob(jobId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

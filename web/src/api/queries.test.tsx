import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import type { ReactNode } from 'react'
import { describe, expect, it } from 'vitest'
import { server } from '@/test/msw/server'
import { modelsResponse, runningJob, succeededJob } from '@/test/msw/fixtures'
import {
  jobRefetchInterval,
  jobsListRefetchInterval,
  modelDownloadStates,
  modelsRefetchInterval,
  queryKeys,
  readinessRefetchInterval,
  useJobQuery,
  useModelsQuery,
  useReadinessQuery,
} from './queries'
import { ProblemError, TransportError } from './problem'

describe('query keys', () => {
  it('are stable, distinct tuples per resource', () => {
    expect(queryKeys.session()).toEqual(['session'])
    expect(queryKeys.readiness()).toEqual(['readiness'])
    expect(queryKeys.jobs({ state: 'active' })).toEqual(['jobs', { state: 'active' }])
    expect(queryKeys.job('job-1')).toEqual(['job', 'job-1'])
    expect(queryKeys.models()).toEqual(['models'])
    expect(queryKeys.settings()).toEqual(['settings'])
  })
})

describe('readinessRefetchInterval (design §7.3)', () => {
  it('polls every 5s while not ready', () => {
    expect(readinessRefetchInterval(false)).toBe(5000)
  })

  it('polls every 30s once ready', () => {
    expect(readinessRefetchInterval(true)).toBe(30000)
  })
})

describe('jobsListRefetchInterval (design §7.3)', () => {
  it('polls every 2s while an active job is present', () => {
    expect(jobsListRefetchInterval(true)).toBe(2000)
  })

  it('stops polling once nothing is active', () => {
    expect(jobsListRefetchInterval(false)).toBe(false)
  })
})

describe('jobRefetchInterval (design §7.3)', () => {
  it.each(['queued', 'running'] as const)('polls every 1.5s while %s', (state) => {
    expect(jobRefetchInterval(state)).toBe(1500)
  })

  it.each(['succeeded', 'failed', 'cancelled', 'expired'] as const)('stops polling once %s', (state) => {
    expect(jobRefetchInterval(state)).toBe(false)
  })
})

describe('modelsRefetchInterval (design §7.3)', () => {
  it.each(['downloading', 'verifying', 'loading'] as const)('polls every 2s while a variant is %s', (state) => {
    expect(modelsRefetchInterval([state])).toBe(2000)
  })

  it('polls every 15s when no variant is busy', () => {
    expect(modelsRefetchInterval([])).toBe(15000)
    expect(modelsRefetchInterval(['ready', 'absent'])).toBe(15000)
  })
})

function withQueryClient() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  function wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>
  }
  return wrapper
}

describe('useJobQuery', () => {
  it('fetches a job by id', async () => {
    const { result } = renderHook(() => useJobQuery(runningJob.job_id), { wrapper: withQueryClient() })
    await waitFor(() => {
      expect(result.current.data).toEqual(runningJob)
    })
  })

  it('throws ProblemError for a well-formed problem response', async () => {
    server.use(
      http.get('/v1/conversions/:jobId', () =>
        HttpResponse.json(
          {
            blockers: null,
            code: 'RESOURCE_NOT_FOUND',
            max_pages: null,
            retry_after_seconds: null,
            retryable: false,
            status: 404,
            title: 'Resource not found',
            type: null,
          },
          { status: 404, headers: { 'content-type': 'application/problem+json' } },
        ),
      ),
    )
    const { result } = renderHook(() => useJobQuery(succeededJob.job_id), { wrapper: withQueryClient() })
    await waitFor(() => {
      expect(result.current.isError).toBe(true)
    })
    expect(result.current.error).toBeInstanceOf(ProblemError)
  })
})

describe('modelDownloadStates', () => {
  it('collects each variant download.state', () => {
    const [first] = modelsResponse.variants
    if (first?.download === undefined || first.download === null) {
      throw new Error('modelsResponse fixture must have at least one downloadable variant')
    }
    expect(
      modelDownloadStates({
        ...modelsResponse,
        variants: [
          { ...first, download: { ...first.download, state: 'downloading' } },
          { ...first, variant_id: 'other', download: null },
        ],
      }),
    ).toEqual(['downloading'])
  })

  it('returns an empty list for undefined data', () => {
    expect(modelDownloadStates(undefined)).toEqual([])
  })
})

describe('useModelsQuery', () => {
  it('fetches the models list', async () => {
    const { result } = renderHook(() => useModelsQuery(), { wrapper: withQueryClient() })
    await waitFor(() => {
      expect(result.current.data).toEqual(modelsResponse)
    })
  })
})

describe('useReadinessQuery', () => {
  it('narrows the unknown readiness body to service_ready', async () => {
    server.use(http.get('/v1/readiness', () => HttpResponse.json({ service_ready: true })))
    const { result } = renderHook(() => useReadinessQuery(), { wrapper: withQueryClient() })
    await waitFor(() => {
      expect(result.current.data).toEqual({ service_ready: true })
    })
  })

  it('rejects a malformed readiness body as a transport error', async () => {
    // `TransportError` is retried indefinitely with backoff (design §7.3), so
    // the query never settles into `isError` -- it keeps retrying, as
    // intended for "service unreachable" (UX §3.8). Assert on the failed
    // attempt itself instead of a terminal state that will never arrive.
    server.use(http.get('/v1/readiness', () => HttpResponse.json({ unexpected: true })))
    const { result } = renderHook(() => useReadinessQuery(), { wrapper: withQueryClient() })
    await waitFor(() => {
      expect(result.current.failureCount).toBeGreaterThan(0)
    })
    expect(result.current.failureReason).toBeInstanceOf(TransportError)
    expect(result.current.isError).toBe(false)
  })
})

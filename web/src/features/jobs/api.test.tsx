import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import type { ReactNode } from 'react'
import { describe, expect, it } from 'vitest'
import { server } from '@/test/msw/server'
import { emptyJobList, makeJobSummary } from '@/test/msw/fixtures'
import { buildJobsListSearch, cancelJob, purgeJob, statesForFilter, useJobsListQuery } from './api'

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

describe('buildJobsListSearch (design §3.4)', () => {
  it('repeats state= once per state', () => {
    expect(buildJobsListSearch({ states: ['queued', 'running'] })).toBe('state=queued&state=running')
  })

  it('omits state entirely for the "all" filter', () => {
    expect(buildJobsListSearch({})).toBe('')
  })

  it('includes limit and cursor when given', () => {
    expect(buildJobsListSearch({ limit: 25, cursor: 'abc' })).toBe('limit=25&cursor=abc')
  })

  it('omits an empty cursor', () => {
    expect(buildJobsListSearch({ cursor: null })).toBe('')
  })
})

describe('statesForFilter', () => {
  it('maps each filter tab to its server states', () => {
    expect(statesForFilter('all')).toBeUndefined()
    expect(statesForFilter('active')).toEqual(['queued', 'running'])
    expect(statesForFilter('succeeded')).toEqual(['succeeded'])
    expect(statesForFilter('failed')).toEqual(['failed'])
    expect(statesForFilter('ended')).toEqual(['cancelled', 'expired'])
  })
})

describe('useJobsListQuery (design §7.3 infinite query)', () => {
  it('sends the filter states as repeated state= query params', async () => {
    let capturedUrl: string | undefined
    server.use(
      http.get('/v1/conversions', ({ request }) => {
        capturedUrl = request.url
        return HttpResponse.json(emptyJobList)
      }),
    )

    const { result } = renderHook(() => useJobsListQuery('active'), { wrapper })
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })

    expect(capturedUrl).toContain('state=queued')
    expect(capturedUrl).toContain('state=running')
  })

  it('paginates with the cursor from next_cursor', async () => {
    const firstPage = {
      items: [makeJobSummary({ job_id: 'job-1' })],
      next_cursor: 'cursor-2',
      schema_version: '1.0' as const,
    }
    const secondPage = {
      items: [makeJobSummary({ job_id: 'job-2' })],
      next_cursor: null,
      schema_version: '1.0' as const,
    }
    let requestCount = 0
    server.use(
      http.get('/v1/conversions', ({ request }) => {
        requestCount += 1
        const cursor = new URL(request.url).searchParams.get('cursor')
        return HttpResponse.json(cursor === 'cursor-2' ? secondPage : firstPage)
      }),
    )

    const { result } = renderHook(() => useJobsListQuery('all'), { wrapper })
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })
    expect(result.current.data?.pages).toHaveLength(1)
    expect(result.current.hasNextPage).toBe(true)

    await result.current.fetchNextPage()
    await waitFor(() => {
      expect(result.current.data?.pages).toHaveLength(2)
    })
    expect(requestCount).toBe(2)
    expect(result.current.hasNextPage).toBe(false)
  })
})

describe('cancelJob / purgeJob', () => {
  it('cancels a job via DELETE without a mode param', async () => {
    let capturedUrl: string | undefined
    server.use(
      http.delete('/v1/conversions/:jobId', ({ request }) => {
        capturedUrl = request.url
        return HttpResponse.json(makeJobSummary())
      }),
    )

    await cancelJob('job-1')
    expect(capturedUrl).not.toContain('mode=')
  })

  it('purges a job via DELETE with mode=purge and resolves on 204', async () => {
    let capturedUrl: string | undefined
    server.use(
      http.delete('/v1/conversions/:jobId', ({ request }) => {
        capturedUrl = request.url
        return new HttpResponse(null, { status: 204 })
      }),
    )

    await expect(purgeJob('job-1')).resolves.toBeUndefined()
    expect(capturedUrl).toContain('mode=purge')
  })
})

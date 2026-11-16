import { useMutation, useQuery } from '@tanstack/react-query'
import { QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it } from 'vitest'
import { sessionExpiredController } from '@/auth/SessionExpiredDialog'
import { queryKeys } from './queries'
import { ProblemError, TransportError } from './problem'
import { createQueryClient, isSessionExpiryError } from './queryClient'

function buildProblem(status: number, code = 'AUTHENTICATION_FAILED') {
  return new ProblemError({
    blockers: null,
    code: code as never,
    max_pages: null,
    retry_after_seconds: null,
    retryable: false,
    status,
    title: 'x',
    type: null,
  })
}

describe('isSessionExpiryError', () => {
  it('is true for a 401 ProblemError while auth_kind is session', () => {
    const queryClient = createQueryClient()
    queryClient.setQueryData(queryKeys.session(), { auth_kind: 'session' })

    expect(isSessionExpiryError(buildProblem(401), queryClient)).toBe(true)
  })

  it('is false for 403 WEB_REQUEST_REJECTED even with a session auth_kind (never logs out)', () => {
    const queryClient = createQueryClient()
    queryClient.setQueryData(queryKeys.session(), { auth_kind: 'session' })

    expect(isSessionExpiryError(buildProblem(403, 'WEB_REQUEST_REJECTED'), queryClient)).toBe(false)
  })

  it('is false when the last known auth_kind is bearer (setup wizard) or null (no session yet)', () => {
    const queryClient = createQueryClient()
    queryClient.setQueryData(queryKeys.session(), { auth_kind: 'bearer' })
    expect(isSessionExpiryError(buildProblem(401), queryClient)).toBe(false)

    queryClient.setQueryData(queryKeys.session(), { auth_kind: null })
    expect(isSessionExpiryError(buildProblem(401), queryClient)).toBe(false)
  })

  it('is false for a non-ProblemError (TransportError)', () => {
    const queryClient = createQueryClient()
    queryClient.setQueryData(queryKeys.session(), { auth_kind: 'session' })

    expect(isSessionExpiryError(new TransportError(401), queryClient)).toBe(false)
  })
})

describe('createQueryClient (design §7.4 QueryCache.onError)', () => {
  afterEach(() => {
    sessionExpiredController.close()
  })

  it('opens the session-expired dialog when a query fails with a session 401', async () => {
    const queryClient = createQueryClient()
    queryClient.setQueryData(queryKeys.session(), { auth_kind: 'session' })
    function wrapper({ children }: { children: ReactNode }) {
      return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    }

    renderHook(
      () =>
        useQuery({
          queryKey: ['job', 'some-job'],
          queryFn: () => Promise.reject(buildProblem(401)),
          retry: false,
        }),
      { wrapper },
    )

    await waitFor(() => {
      expect(sessionExpiredController.getSnapshot()).toBe(true)
    })
  })

  it('never opens the dialog for a mutation failure (UX §3.2: a submit is not replayed automatically)', async () => {
    const queryClient = createQueryClient()
    queryClient.setQueryData(queryKeys.session(), { auth_kind: 'session' })
    function wrapper({ children }: { children: ReactNode }) {
      return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    }

    const { result } = renderHook(
      () =>
        useMutation({
          mutationFn: () => Promise.reject(buildProblem(401)),
        }),
      { wrapper },
    )
    result.current.mutate()

    await waitFor(() => {
      expect(result.current.isError).toBe(true)
    })
    expect(sessionExpiredController.getSnapshot()).toBe(false)
  })
})

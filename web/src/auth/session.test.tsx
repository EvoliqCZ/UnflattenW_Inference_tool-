import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import type { ReactNode } from 'react'
import { describe, expect, it } from 'vitest'
import { server } from '@/test/msw/server'
import { ProblemError } from '@/api/problem'
import { authenticatedSession, authenticationFailedProblem, unauthenticatedSession } from '@/test/msw/fixtures'
import {
  deleteSession,
  fetchSessionInfo,
  postLogin,
  useLoginMutation,
  useLogoutMutation,
  useSessionQuery,
} from './session'

function wrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}

describe('fetchSessionInfo', () => {
  it('calls GET /v1/session and returns the SessionInfo body', async () => {
    server.use(http.get('/v1/session', () => HttpResponse.json(authenticatedSession)))

    await expect(fetchSessionInfo()).resolves.toEqual(authenticatedSession)
  })
})

describe('useSessionQuery', () => {
  it("is keyed ['session'] and exposes the fetched SessionInfo", async () => {
    server.use(http.get('/v1/session', () => HttpResponse.json(unauthenticatedSession)))

    const { result } = renderHook(() => useSessionQuery(), { wrapper })

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })
    expect(result.current.data).toEqual(unauthenticatedSession)
  })

  it('does not retry on failure (design: an invalid session is a real 401)', async () => {
    server.use(http.get('/v1/session', () => HttpResponse.json({ code: 'AUTHENTICATION_FAILED' }, { status: 401 })))

    const { result } = renderHook(() => useSessionQuery(), { wrapper })

    await waitFor(() => {
      expect(result.current.isError).toBe(true)
    })
  })
})

describe('postLogin', () => {
  it('POSTs the password to /v1/session and returns the SessionInfo body', async () => {
    server.use(http.post('/v1/session', () => HttpResponse.json(authenticatedSession, { status: 201 })))

    await expect(postLogin('correct horse battery staple')).resolves.toEqual(authenticatedSession)
  })

  it('rejects with ProblemError on a 401 (UX §3.2 "Nesprávne heslo")', async () => {
    server.use(
      http.post('/v1/session', () =>
        HttpResponse.json(authenticationFailedProblem, {
          status: 401,
          headers: { 'content-type': 'application/problem+json' },
        }),
      ),
    )

    await expect(postLogin('wrong')).rejects.toBeInstanceOf(ProblemError)
  })
})

describe('useLoginMutation', () => {
  it("seeds the ['session'] query with the returned SessionInfo on success", async () => {
    server.use(http.post('/v1/session', () => HttpResponse.json(authenticatedSession, { status: 201 })))
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    function localWrapper({ children }: { children: ReactNode }) {
      return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    }

    const { result } = renderHook(() => useLoginMutation(), { wrapper: localWrapper })
    result.current.mutate('correct horse battery staple')

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })
    expect(queryClient.getQueryData(['session'])).toEqual(authenticatedSession)
  })
})

describe('deleteSession', () => {
  it('DELETEs /v1/session', async () => {
    server.use(http.delete('/v1/session', () => new HttpResponse(null, { status: 204 })))

    await expect(deleteSession()).resolves.toBeUndefined()
  })
})

describe('useLogoutMutation', () => {
  it("invalidates the ['session'] query on success", async () => {
    server.use(
      http.delete('/v1/session', () => new HttpResponse(null, { status: 204 })),
      http.get('/v1/session', () => HttpResponse.json(unauthenticatedSession)),
    )
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    queryClient.setQueryData(['session'], authenticatedSession)
    function localWrapper({ children }: { children: ReactNode }) {
      return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    }

    const { result } = renderHook(() => useLogoutMutation(), { wrapper: localWrapper })
    result.current.mutate()

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })
    expect(queryClient.getQueryState(['session'])?.isInvalidated).toBe(true)
  })
})

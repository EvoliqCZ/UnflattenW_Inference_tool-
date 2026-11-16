import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import type { ReactNode } from 'react'
import { describe, expect, it } from 'vitest'
import { server } from '@/test/msw/server'
import { ProblemError, TransportError } from '@/api/problem'
import { settingsResponse } from '@/test/msw/fixtures'
import { fetchSettings, putChangePassword, useChangePasswordMutation, useSettingsQuery } from './api'

function wrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}

describe('fetchSettings', () => {
  it('calls GET /v1/settings and returns the SettingsResponse body', async () => {
    server.use(http.get('/v1/settings', () => HttpResponse.json(settingsResponse)))

    await expect(fetchSettings()).resolves.toEqual(settingsResponse)
  })
})

describe('useSettingsQuery', () => {
  it("is keyed ['settings'] and exposes the fetched SettingsResponse", async () => {
    server.use(http.get('/v1/settings', () => HttpResponse.json(settingsResponse)))

    const { result } = renderHook(() => useSettingsQuery(), { wrapper })

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })
    expect(result.current.data).toEqual(settingsResponse)
  })
})

describe('putChangePassword', () => {
  it('PUTs current_password/new_password to /v1/setup/password and resolves on 204', async () => {
    server.use(
      http.put('/v1/setup/password', async ({ request }) => {
        expect(await request.json()).toEqual({ current_password: 'old-pw', new_password: 'new-and-long-enough' })
        return new HttpResponse(null, { status: 204 })
      }),
    )

    await expect(putChangePassword('old-pw', 'new-and-long-enough')).resolves.toBeUndefined()
  })

  it('rejects with ProblemError on a 401 (wrong current password)', async () => {
    server.use(
      http.put('/v1/setup/password', () =>
        HttpResponse.json(
          {
            blockers: null,
            code: 'AUTHENTICATION_FAILED',
            max_pages: null,
            retry_after_seconds: null,
            retryable: false,
            status: 401,
            title: 'Authentication failed',
            type: null,
          },
          { status: 401, headers: { 'content-type': 'application/problem+json' } },
        ),
      ),
    )

    await expect(putChangePassword('wrong', 'new-and-long-enough')).rejects.toBeInstanceOf(ProblemError)
  })

  it('rejects with ProblemError on a 429 carrying retry_after_seconds', async () => {
    server.use(
      http.put('/v1/setup/password', () =>
        HttpResponse.json(
          {
            blockers: null,
            code: 'RATE_LIMITED',
            max_pages: null,
            retry_after_seconds: 42,
            retryable: true,
            status: 429,
            title: 'Rate limited',
            type: null,
          },
          { status: 429, headers: { 'content-type': 'application/problem+json' } },
        ),
      ),
    )

    const error = await putChangePassword('old-pw', 'new-and-long-enough').catch((cause: unknown) => cause)
    expect(error).toBeInstanceOf(ProblemError)
    expect((error as ProblemError).problem.retry_after_seconds).toBe(42)
  })

  it('rejects with TransportError on a network failure', async () => {
    server.use(http.put('/v1/setup/password', () => HttpResponse.error()))

    await expect(putChangePassword('old-pw', 'new-and-long-enough')).rejects.toBeInstanceOf(TransportError)
  })
})

describe('useChangePasswordMutation', () => {
  it('resolves on success', async () => {
    server.use(http.put('/v1/setup/password', () => new HttpResponse(null, { status: 204 })))

    const { result } = renderHook(() => useChangePasswordMutation(), { wrapper })
    result.current.mutate({ currentPassword: 'old-pw', newPassword: 'new-and-long-enough' })

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })
  })
})

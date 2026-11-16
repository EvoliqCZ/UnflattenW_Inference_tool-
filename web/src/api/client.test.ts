import { describe, expect, it } from 'vitest'
import { server } from '@/test/msw/server'
import { http, HttpResponse } from 'msw'
import { call, apiClient, callRaw } from './client'
import { ProblemError, TransportError } from './problem'
import { modelsResponse, authenticatedSession, authenticationFailedProblem } from '@/test/msw/fixtures'

describe('call (design §7.4)', () => {
  it('returns the data on success', async () => {
    server.use(http.get('/v1/models', () => HttpResponse.json(modelsResponse)))
    await expect(call(apiClient.GET('/v1/models'))).resolves.toEqual(modelsResponse)
  })

  it('throws TransportError when the request itself fails (network error)', async () => {
    server.use(http.get('/v1/models', () => HttpResponse.error()))
    await expect(call(apiClient.GET('/v1/models'))).rejects.toBeInstanceOf(TransportError)
  })

  it('throws ProblemError for a well-formed application/problem+json error body', async () => {
    server.use(
      http.get('/v1/models', () =>
        HttpResponse.json(
          {
            blockers: null,
            code: 'INTERNAL_ERROR',
            max_pages: null,
            retry_after_seconds: null,
            retryable: true,
            status: 500,
            title: 'Internal error',
            type: null,
          },
          { status: 500, headers: { 'content-type': 'application/problem+json' } },
        ),
      ),
    )
    await expect(call(apiClient.GET('/v1/models'))).rejects.toBeInstanceOf(ProblemError)
  })

  it('throws TransportError for a non-2xx response without a problem body', async () => {
    server.use(http.get('/v1/models', () => new HttpResponse('nope', { status: 502 })))
    const error = await call(apiClient.GET('/v1/models')).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(TransportError)
    expect((error as TransportError).status).toBe(502)
  })

  it('throws TransportError when a 2xx response carries no body', async () => {
    server.use(http.get('/v1/models', () => new HttpResponse(null, { status: 200 })))
    await expect(call(apiClient.GET('/v1/models'))).rejects.toBeInstanceOf(TransportError)
  })
})

describe('callRaw (design §2.3: POST /v1/setup, POST/DELETE /v1/session have no OpenAPI requestBody)', () => {
  it('POSTs the body as JSON and returns the response data on success', async () => {
    server.use(
      http.post('/v1/session', async ({ request }) => {
        expect(request.headers.get('content-type')).toBe('application/json')
        expect(request.headers.get('x-unflatten-client')).toBe('web')
        expect(await request.json()).toEqual({ password: 'correct horse battery staple' })
        return HttpResponse.json(authenticatedSession, { status: 201 })
      }),
    )

    await expect(callRaw('/v1/session', 'POST', { password: 'correct horse battery staple' })).resolves.toEqual(
      authenticatedSession,
    )
  })

  it('sends the bootstrap token as a bearer header (POST /v1/setup)', async () => {
    server.use(
      http.post('/v1/setup', ({ request }) => {
        expect(request.headers.get('authorization')).toBe('Bearer bootstrap-token')
        return HttpResponse.json(authenticatedSession, { status: 201 })
      }),
    )

    await expect(
      callRaw('/v1/setup', 'POST', { password: 'correct horse battery staple' }, 'bootstrap-token'),
    ).resolves.toEqual(authenticatedSession)
  })

  it('returns undefined for a 204 with no body (DELETE /v1/session)', async () => {
    server.use(http.delete('/v1/session', () => new HttpResponse(null, { status: 204 })))

    await expect(callRaw('/v1/session', 'DELETE')).resolves.toBeUndefined()
  })

  it('throws ProblemError for a well-formed application/problem+json error body', async () => {
    server.use(
      http.post('/v1/session', () =>
        HttpResponse.json(authenticationFailedProblem, {
          status: 401,
          headers: { 'content-type': 'application/problem+json' },
        }),
      ),
    )

    await expect(callRaw('/v1/session', 'POST', { password: 'wrong' })).rejects.toBeInstanceOf(ProblemError)
  })

  it('throws TransportError when the request itself fails (network error)', async () => {
    server.use(http.post('/v1/session', () => HttpResponse.error()))

    await expect(callRaw('/v1/session', 'POST', { password: 'x' })).rejects.toBeInstanceOf(TransportError)
  })

  it('PUTs the body as JSON and returns undefined for a 204 (PUT /v1/setup/password)', async () => {
    server.use(
      http.put('/v1/setup/password', async ({ request }) => {
        expect(request.headers.get('content-type')).toBe('application/json')
        expect(await request.json()).toEqual({ current_password: 'old', new_password: 'new-and-long-enough' })
        return new HttpResponse(null, { status: 204 })
      }),
    )

    await expect(
      callRaw('/v1/setup/password', 'PUT', { current_password: 'old', new_password: 'new-and-long-enough' }),
    ).resolves.toBeUndefined()
  })
})

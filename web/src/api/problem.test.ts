import { describe, expect, it } from 'vitest'
import en from '@/i18n/en.json'
import { isProblemResponse, ProblemError, retryDelay, shouldRetry, TransportError } from './problem'
import { authenticationFailedProblem, internalErrorProblem } from '@/test/msw/fixtures'

describe('ProblemError', () => {
  it('copies code, status, retryable and the full problem from the server body', () => {
    const error = new ProblemError(authenticationFailedProblem)
    expect(error.code).toBe('AUTHENTICATION_FAILED')
    expect(error.status).toBe(401)
    expect(error.retryable).toBe(false)
    expect(error.problem).toBe(authenticationFailedProblem)
    expect(error.message).toBe(authenticationFailedProblem.title)
    expect(error.name).toBe('ProblemError')
  })

  it('has a translated title for its code in en.json (seam: codes vs. client)', () => {
    const error = new ProblemError(internalErrorProblem)
    const titles = en.codes.problem as Record<string, { title: string }>
    expect(titles[error.code]?.title).toBeTypeOf('string')
  })
})

describe('TransportError', () => {
  it('carries an optional status and cause', () => {
    const cause = new Error('network down')
    const error = new TransportError(503, cause)
    expect(error.status).toBe(503)
    expect(error.cause).toBe(cause)
    expect(error.name).toBe('TransportError')
  })

  it('allows a missing status for outright network failures', () => {
    const error = new TransportError()
    expect(error.status).toBeUndefined()
  })
})

describe('isProblemResponse', () => {
  it('detects application/problem+json', () => {
    const response = new Response(null, { headers: { 'content-type': 'application/problem+json' } })
    expect(isProblemResponse(response)).toBe(true)
  })

  it('rejects plain json and missing content-type', () => {
    expect(isProblemResponse(new Response(null, { headers: { 'content-type': 'application/json' } }))).toBe(false)
    expect(isProblemResponse(new Response(null))).toBe(false)
  })
})

describe('shouldRetry (design §7.3/§7.4)', () => {
  it('never retries a 401 ProblemError', () => {
    expect(shouldRetry(new ProblemError(authenticationFailedProblem))).toBe(false)
  })

  it('never retries a ProblemError below 500', () => {
    expect(shouldRetry(new ProblemError({ ...internalErrorProblem, status: 404, retryable: false }))).toBe(false)
  })

  it('retries a >=500 ProblemError that is not 401', () => {
    expect(shouldRetry(new ProblemError(internalErrorProblem))).toBe(true)
  })

  it('retries a TransportError', () => {
    expect(shouldRetry(new TransportError(503))).toBe(true)
  })

  it('does not retry unrelated errors', () => {
    expect(shouldRetry(new Error('boom'))).toBe(false)
  })
})

describe('retryDelay (design §7.3: exponential 2-30s)', () => {
  it('starts at 2s and doubles', () => {
    expect(retryDelay(0)).toBe(2000)
    expect(retryDelay(1)).toBe(4000)
    expect(retryDelay(2)).toBe(8000)
  })

  it('caps at 30s', () => {
    expect(retryDelay(10)).toBe(30000)
  })
})

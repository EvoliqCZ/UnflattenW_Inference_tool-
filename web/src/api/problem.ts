import type { components } from './schema'

export type Problem = components['schemas']['Problem']

/**
 * A well-formed RFC 7807 `application/problem+json` error from the API
 * (design §7.4). `retryable` is computed server-side, not re-derived here.
 */
export class ProblemError extends Error {
  readonly code: Problem['code']
  readonly status: number
  readonly retryable: boolean
  readonly problem: Problem

  constructor(problem: Problem) {
    super(problem.title)
    this.name = 'ProblemError'
    this.code = problem.code
    this.status = problem.status
    this.retryable = problem.retryable
    this.problem = problem
  }
}

/**
 * A network failure, a non-2xx response without a `problem+json` body, or any
 * other transport-level failure the client cannot attribute to the server's
 * contract (design §7.4, UX §3.8 "Služba neodpovedá").
 */
export class TransportError extends Error {
  readonly status: number | undefined

  constructor(status?: number, cause?: unknown) {
    super(status === undefined ? 'transport error' : `transport error (${String(status)})`)
    this.name = 'TransportError'
    this.status = status
    this.cause = cause
  }
}

const PROBLEM_CONTENT_TYPE = 'application/problem+json'

export function isProblemResponse(response: Response): boolean {
  return response.headers.get('content-type')?.includes(PROBLEM_CONTENT_TYPE) ?? false
}

/**
 * `retry` predicate for TanStack Query's `retry` option (design §7.3):
 * never retry a 401 or any `ProblemError` below 500; `TransportError` and
 * server (>=500) `ProblemError`s are retryable.
 */
export function shouldRetry(error: unknown): boolean {
  if (error instanceof ProblemError) {
    return error.status >= 500 && error.status !== 401
  }
  return error instanceof TransportError
}

const MIN_RETRY_DELAY_MS = 2000
const MAX_RETRY_DELAY_MS = 30000

/** Exponential backoff 2s-30s (design §7.3) for the `retryDelay` option. */
export function retryDelay(attemptIndex: number): number {
  const delay = MIN_RETRY_DELAY_MS * 2 ** attemptIndex
  return Math.min(delay, MAX_RETRY_DELAY_MS)
}

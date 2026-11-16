import createClient from 'openapi-fetch'
import type { paths } from './schema'
import { isProblemResponse, ProblemError, TransportError, type Problem } from './problem'

/**
 * design §7.4: same-origin cookies, no bearer token in the browser, and the
 * server-side host allow-list / rate limiter key off this header.
 */
export const apiClient = createClient<paths>({
  baseUrl: '',
  headers: { 'X-Unflatten-Client': 'web' },
  credentials: 'same-origin',
})

interface OpenApiFetchResult<T> {
  data?: T | undefined
  error?: unknown
  response: Response
}

/**
 * Unwraps an `openapi-fetch` call: returns the data on success, or throws
 * `ProblemError` (server sent a well-formed `application/problem+json` body)
 * or `TransportError` (anything else: network failure, or a non-2xx response
 * without a problem body) -- design §7.4.
 */
export async function call<T>(request: Promise<OpenApiFetchResult<T>>): Promise<T> {
  let result: OpenApiFetchResult<T>
  try {
    result = await request
  } catch (cause) {
    throw new TransportError(undefined, cause)
  }
  const { data, error, response } = result
  if (error !== undefined) {
    if (isProblemResponse(response)) {
      throw new ProblemError(error as Problem)
    }
    throw new TransportError(response.status)
  }
  if (data === undefined) {
    throw new TransportError(response.status)
  }
  return data
}

/**
 * `POST /v1/setup`, `POST /v1/session`, `DELETE /v1/session`, `PUT
 * /v1/setup/password` and `PUT /v1/models/default` parse their JSON body by
 * hand server-side (`api.py`) and carry no OpenAPI `requestBody` (see the
 * note in `queries.ts`), so `openapi-fetch` cannot type a call to them --
 * this mirrors `call()`'s error handling for those routes by hand.
 * `bearerToken` is only used by `POST /v1/setup` (design §2.3: bootstrap
 * token from the terminal). `body` values may be `null` (`PUT
 * /v1/models/default {variant_id: null}` clears the user override).
 */
export async function callRaw<T>(
  path: string,
  method: 'POST' | 'PUT' | 'DELETE',
  body?: Record<string, string | null>,
  bearerToken?: string,
): Promise<T> {
  const headers: Record<string, string> = { 'X-Unflatten-Client': 'web' }
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
  }
  if (bearerToken !== undefined) {
    headers.Authorization = `Bearer ${bearerToken}`
  }
  let response: Response
  try {
    response = await fetch(
      path,
      body === undefined
        ? { method, credentials: 'same-origin', headers }
        : { method, credentials: 'same-origin', headers, body: JSON.stringify(body) },
    )
  } catch (cause) {
    throw new TransportError(undefined, cause)
  }
  if (response.status === 204) {
    return undefined as T
  }
  let data: unknown
  try {
    data = await response.json()
  } catch (cause) {
    throw new TransportError(response.status, cause)
  }
  if (!response.ok) {
    if (isProblemResponse(response)) {
      throw new ProblemError(data as Problem)
    }
    throw new TransportError(response.status)
  }
  return data as T
}

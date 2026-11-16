import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient, call, callRaw } from '@/api/client'
import { queryKeys } from '@/api/queries'
import type { components } from '@/api/schema'

export type SessionInfo = components['schemas']['SessionInfo']

/**
 * `GET /v1/session` (design §2.3): never itself a hard failure -- an absent
 * or invalid credential is `authenticated: false` in a `200`, not a thrown
 * error, so the router guard (`auth/guard.ts`) always has a `SessionInfo` to
 * decide from. Only an invalid *session cookie* is a genuine `401`; that
 * still resolves the query with an error, which callers treat as
 * unauthenticated (design §2.3 note: "s neplatným credentialom 401").
 */
export function fetchSessionInfo(): Promise<SessionInfo> {
  return call<SessionInfo>(apiClient.GET('/v1/session'))
}

/**
 * design §7.3: keyed `['session']`, no polling -- invalidated explicitly by
 * login/logout/setup mutations and on a 401 from any other query.
 */
export function useSessionQuery() {
  return useQuery({
    queryKey: queryKeys.session(),
    queryFn: fetchSessionInfo,
    retry: false,
  })
}

/**
 * `POST /v1/session` (design §2.3): body `{password}` parsed by hand
 * server-side, so it has no OpenAPI `requestBody` (see `queries.ts`'s note
 * and `callRaw`'s). Throws `ProblemError` (401 wrong password, 409
 * `SETUP_REQUIRED`, 429 `RATE_LIMITED`) or `TransportError`.
 */
export function postLogin(password: string): Promise<SessionInfo> {
  return callRaw<SessionInfo>('/v1/session', 'POST', { password })
}

/** UX §3.2 login form: on success, seed the session query so the guard sees it immediately. */
export function useLoginMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: postLogin,
    onSuccess: (session) => {
      queryClient.setQueryData(queryKeys.session(), session)
    },
  })
}

/** `DELETE /v1/session` (design §2.3): revokes the cookie session, `204` on success. */
export function deleteSession(): Promise<void> {
  return callRaw('/v1/session', 'DELETE')
}

/** Logout (UX §2.3 `UserMenu` / §3.9 settings): invalidates the session query so every guard re-evaluates. */
export function useLogoutMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: deleteSession,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.session() })
    },
  })
}

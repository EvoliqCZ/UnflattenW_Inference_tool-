import { QueryCache, QueryClient } from '@tanstack/react-query'
import { sessionExpiredController } from '@/auth/SessionExpiredDialog'
import type { SessionInfo } from '@/auth/session'
import { ProblemError } from './problem'
import { queryKeys } from './queries'

/**
 * design §7.4: "`ProblemError.status === 401 && session.auth_kind ===
 * 'session'` -> `SessionExpiredDialog`". A bearer-authenticated caller (no
 * browser session yet, e.g. `/setup`) or a `WEB_REQUEST_REJECTED` (403)
 * never reaches this -- both fail the `status === 401` check on their own.
 */
export function isSessionExpiryError(error: unknown, queryClient: Pick<QueryClient, 'getQueryData'>): boolean {
  if (!(error instanceof ProblemError) || error.status !== 401) {
    return false
  }
  const session = queryClient.getQueryData<SessionInfo>(queryKeys.session())
  return session?.auth_kind === 'session'
}

/**
 * design §7.4: wired on `QueryCache`, not `MutationCache` -- this is what
 * makes the session-expired retry GET-only (UX §3.2 point 5): `QueryCache`
 * only ever sees query (read) failures, so a submit mutation's own 401
 * never opens this dialog from here (screens handle that error locally).
 */
export function createQueryClient(): QueryClient {
  const queryClient: QueryClient = new QueryClient({
    queryCache: new QueryCache({
      onError: (error, query) => {
        if (!isSessionExpiryError(error, queryClient)) {
          return
        }
        sessionExpiredController.open(() => {
          void queryClient.refetchQueries({ queryKey: query.queryKey })
        })
      },
    }),
  })
  return queryClient
}

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { callRaw } from '@/api/client'
import { queryKeys } from '@/api/queries'
import type { SessionInfo } from '@/auth/session'

/**
 * `POST /v1/setup` (design §2.3), body `{password}` -- like `POST
 * /v1/session`, this has no OpenAPI `requestBody` (see `queries.ts`'s note),
 * hence `callRaw`. `token` is the bootstrap token from the setup link's
 * `#token=` fragment, sent as the bearer header; `null` on a loopback-only
 * deployment, where the server offers tokenless setup (owner 2026-09-29,
 * `SessionInfo.setup_tokenless`).
 */
export function postSetup(token: string | null, password: string): Promise<SessionInfo> {
  return callRaw<SessionInfo>('/v1/setup', 'POST', { password }, token ?? undefined)
}

export interface SetupInput {
  token: string | null
  password: string
}

/** UX §3.1 step "password": on success, seed the session query so the guard advances past `/setup`. */
export function useSetupMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ token, password }: SetupInput) => postSetup(token, password),
    onSuccess: (session) => {
      queryClient.setQueryData(queryKeys.session(), session)
    },
  })
}

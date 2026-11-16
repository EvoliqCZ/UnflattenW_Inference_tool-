import { useMutation, useQuery } from '@tanstack/react-query'
import { apiClient, call, callRaw } from '@/api/client'
import { queryKeys } from '@/api/queries'
import { shouldRetry, retryDelay } from '@/api/problem'
import type { components } from '@/api/schema'

export type SettingsResponse = components['schemas']['SettingsResponse']

/** `GET /v1/settings` (design §3.10): version, session limits, default variant and sandbox mode for "O službe". */
export function fetchSettings(): Promise<SettingsResponse> {
  return call<SettingsResponse>(apiClient.GET('/v1/settings'))
}

export function useSettingsQuery() {
  return useQuery({
    queryKey: queryKeys.settings(),
    queryFn: fetchSettings,
    retry: (_failureCount: number, error: unknown) => shouldRetry(error),
    retryDelay,
  })
}

/**
 * `PUT /v1/setup/password` (design §3.10/§2.4): body `{current_password,
 * new_password}` parsed by hand server-side, so it has no OpenAPI
 * `requestBody` (see the note in `queries.ts` and `callRaw`'s) -- mirrors
 * `postLogin`'s use of `callRaw` in `auth/session.tsx`. `204` on success;
 * throws `ProblemError` (401 wrong current password, 429 `RATE_LIMITED`) or
 * `TransportError`.
 */
export function putChangePassword(currentPassword: string, newPassword: string): Promise<void> {
  return callRaw('/v1/setup/password', 'PUT', {
    current_password: currentPassword,
    new_password: newPassword,
  })
}

export interface ChangePasswordInput {
  currentPassword: string
  newPassword: string
}

export function useChangePasswordMutation() {
  return useMutation({
    mutationFn: ({ currentPassword, newPassword }: ChangePasswordInput) =>
      putChangePassword(currentPassword, newPassword),
  })
}

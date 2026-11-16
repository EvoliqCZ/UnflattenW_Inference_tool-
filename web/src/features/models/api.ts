import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient, call, callRaw } from '@/api/client'
import { queryKeys } from '@/api/queries'
import type { components } from '@/api/schema'

export type ModelsResponse = components['schemas']['ModelsResponse']
export type ModelVariantEntry = components['schemas']['ModelVariantEntry']
export type ModelCommandAccepted = components['schemas']['ModelCommandAccepted']

/** design §3.9: `POST /v1/models/{variant_id}/download` -- 202, polled via `useModelsQuery`. */
export function postModelDownload(variantId: string): Promise<ModelCommandAccepted> {
  return call<ModelCommandAccepted>(
    apiClient.POST('/v1/models/{variant_id}/download', { params: { path: { variant_id: variantId } } }),
  )
}

/** design §3.9/§17.5: `POST /v1/models/{variant_id}/load`. */
export function postModelLoad(variantId: string): Promise<ModelCommandAccepted> {
  return call<ModelCommandAccepted>(
    apiClient.POST('/v1/models/{variant_id}/load', { params: { path: { variant_id: variantId } } }),
  )
}

/** design §3.9: `DELETE /v1/models/{variant_id}/download` -- removes the local copy. */
export function deleteModelDownload(variantId: string): Promise<ModelCommandAccepted> {
  return call<ModelCommandAccepted>(
    apiClient.DELETE('/v1/models/{variant_id}/download', { params: { path: { variant_id: variantId } } }),
  )
}

/**
 * design §3.9: `PUT /v1/models/default {variant_id}`. Parsed by hand
 * server-side (no OpenAPI `requestBody`, see `client.ts`'s `callRaw` note)
 * -- `null` clears the user override and falls back to config/auto.
 */
export function putModelDefault(variantId: string | null): Promise<ModelsResponse> {
  return callRaw<ModelsResponse>('/v1/models/default', 'PUT', { variant_id: variantId })
}

/** Every command mutation below invalidates the models list so polling (design §7.3) picks up the new state. */
function useInvalidateModelsOnSettle() {
  const queryClient = useQueryClient()
  return () => {
    void queryClient.invalidateQueries({ queryKey: queryKeys.models() })
  }
}

export function useDownloadModelMutation() {
  const invalidate = useInvalidateModelsOnSettle()
  return useMutation({
    mutationFn: (variantId: string) => postModelDownload(variantId),
    onSettled: invalidate,
  })
}

export function useLoadModelMutation() {
  const invalidate = useInvalidateModelsOnSettle()
  return useMutation({
    mutationFn: (variantId: string) => postModelLoad(variantId),
    onSettled: invalidate,
  })
}

export function useDeleteModelDownloadMutation() {
  const invalidate = useInvalidateModelsOnSettle()
  return useMutation({
    mutationFn: (variantId: string) => deleteModelDownload(variantId),
    onSettled: invalidate,
  })
}

export function useSetDefaultModelMutation() {
  const invalidate = useInvalidateModelsOnSettle()
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (variantId: string | null) => putModelDefault(variantId),
    onSettled: () => {
      invalidate()
      void queryClient.invalidateQueries({ queryKey: queryKeys.settings() })
    },
  })
}

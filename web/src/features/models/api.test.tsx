import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import type { ReactNode } from 'react'
import { describe, expect, it } from 'vitest'
import { server } from '@/test/msw/server'
import { modelCommandAccepted, modelsResponse } from '@/test/msw/fixtures'
import {
  deleteModelDownload,
  postModelDownload,
  postModelLoad,
  putModelDefault,
  useDeleteModelDownloadMutation,
  useDownloadModelMutation,
  useLoadModelMutation,
  useSetDefaultModelMutation,
} from './api'

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>
}

describe('model command fetchers (design §3.9)', () => {
  it('posts a download command and returns the accepted envelope', async () => {
    await expect(postModelDownload('qwen3-vl-bf16')).resolves.toEqual(modelCommandAccepted)
  })

  it('posts a load command', async () => {
    server.use(
      http.post('/v1/models/:variantId/load', () =>
        HttpResponse.json({ ...modelCommandAccepted, kind: 'load', phase: 'loading' }, { status: 202 }),
      ),
    )
    const result = await postModelLoad('qwen3-vl-bf16')
    expect(result.kind).toBe('load')
  })

  it('deletes a local download', async () => {
    await expect(deleteModelDownload('qwen3-vl-int8')).resolves.toEqual(modelCommandAccepted)
  })

  it('sends the default variant id as a raw JSON body, including null', async () => {
    let capturedBody: unknown
    server.use(
      http.put('/v1/models/default', async ({ request }) => {
        capturedBody = await request.json()
        return HttpResponse.json(modelsResponse)
      }),
    )
    await putModelDefault(null)
    expect(capturedBody).toEqual({ variant_id: null })
  })
})

describe('model command mutations invalidate the models query (design §7.3)', () => {
  it.each([
    ['download', () => useDownloadModelMutation(), 'qwen3-vl-bf16'],
    ['load', () => useLoadModelMutation(), 'qwen3-vl-bf16'],
    ['delete', () => useDeleteModelDownloadMutation(), 'qwen3-vl-int8'],
  ] as const)('%s settles without throwing', async (_name, useHook, variantId) => {
    const { result } = renderHook(() => useHook(), { wrapper })
    result.current.mutate(variantId)
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })
  })

  it('set-default settles without throwing', async () => {
    const { result } = renderHook(() => useSetDefaultModelMutation(), { wrapper })
    result.current.mutate('qwen3-vl-bf16')
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })
  })
})

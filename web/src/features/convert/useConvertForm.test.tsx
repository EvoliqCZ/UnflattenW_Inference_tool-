import { act, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { beforeEach, expect, it, vi } from 'vitest'
import { submitConversion } from '@/api/upload'
import { TransportError } from '@/api/problem'
import { useConvertForm } from './useConvertForm'

vi.mock('@/api/upload', () => ({ submitConversion: vi.fn() }))
const upload = vi.mocked(submitConversion)
const wrapper = ({ children }: { children: ReactNode }) => <MemoryRouter>{children}</MemoryRouter>
const pdf = () => new File(['%PDF-1.4'], 'source.pdf', { type: 'application/pdf' })

beforeEach(() => {
  upload.mockReset()
})

it('guards two synchronous submissions and disables submission until the response arrives', async () => {
  let reject!: (reason: unknown) => void
  upload.mockImplementation(
    () =>
      new Promise((_resolve, failure) => {
        reject = failure
      }),
  )
  const { result } = renderHook(() => useConvertForm(undefined, true), { wrapper })
  await act(async () => result.current.addFiles([pdf()]))
  let first!: Promise<void>
  let second!: Promise<void>
  act(() => {
    first = result.current.submit()
    second = result.current.submit()
  })
  expect(upload).toHaveBeenCalledTimes(1)
  expect(result.current.canSubmit).toBe(false)
  await act(async () => {
    reject(new TransportError(undefined))
    await Promise.all([first, second])
  })
  expect(result.current.canSubmit).toBe(true)
  expect(result.current.submitError).toEqual({ code: 'INTERNAL_ERROR' })
})

it.each(['files', 'order', 'output', 'model', 'label'] as const)(
  'reuses an uncertain upload key for unchanged retries and replaces it when %s changes',
  async (change) => {
    upload.mockRejectedValue(new TransportError(undefined))
    const { result } = renderHook(() => useConvertForm(undefined, true), { wrapper })
    await act(async () =>
      result.current.addFiles(
        change === 'order'
          ? [new File(['a'], 'a.png', { type: 'image/png' }), new File(['b'], 'b.png', { type: 'image/png' })]
          : [pdf()],
      ),
    )
    await act(async () => result.current.submit())
    const key = upload.mock.calls[0]?.[0].idempotencyKey
    expect(key).toMatch(/^[0-9a-f-]{36}$/)
    await act(async () => result.current.submit())
    expect(upload.mock.calls[1]?.[0].idempotencyKey).toBe(key)
    await act(async () => {
      if (change === 'files') await result.current.addFiles([pdf()])
      if (change === 'order') result.current.reorder([...result.current.files].reverse())
      if (change === 'output') result.current.setOutput('prediction_ir')
      if (change === 'model') result.current.setModelVariant('dense-1.0-bf16')
      if (change === 'label') result.current.setLabel('Changed label')
    })
    await act(async () => result.current.submit())
    expect(upload.mock.calls[2]?.[0].idempotencyKey).toMatch(/^[0-9a-f-]{36}$/)
    expect(upload.mock.calls[2]?.[0].idempotencyKey).not.toBe(key)
  },
)

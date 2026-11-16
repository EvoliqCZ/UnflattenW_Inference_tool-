import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { modelsResponse, plannedModelVariant, unpublishedModelVariant } from '@/test/msw/fixtures'
import { ModelStep } from './ModelStep'

function renderStep(onNext = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <ModelStep language="sk" onNext={onNext} />
    </QueryClientProvider>,
  )
  return onNext
}

function modelsWith(state: 'downloading' | 'verifying' | 'failed' | 'ready') {
  const [first, ...rest] = modelsResponse.variants
  if (first === undefined) throw new Error('fixture has no variants')
  return {
    ...modelsResponse,
    variants: [
      {
        ...first,
        resident: state === 'ready',
        downloaded: state !== 'failed',
        download: {
          bytes_done: state === 'downloading' ? 1_000_000 : null,
          bytes_total: state === 'downloading' ? 4_200_000_000 : null,
          command_id: null,
          error_code: state === 'failed' ? 'INTERNAL_ERROR' : null,
          state,
        },
      },
      ...rest,
    ],
  }
}

describe('ModelStep (UX §3.1 step 4 / §11.2)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows a determinate progress bar while downloading', async () => {
    server.use(http.get('/v1/models', () => HttpResponse.json(modelsWith('downloading'))))
    renderStep()

    await waitFor(() => expect(screen.getByRole('progressbar')).toBeInTheDocument())
    expect(screen.getByRole('button', { name: 'Pokračovať' })).toHaveAttribute('aria-disabled', 'true')
  })

  it('shows an indeterminate state while verifying', async () => {
    server.use(http.get('/v1/models', () => HttpResponse.json(modelsWith('verifying'))))
    renderStep()

    await waitFor(() => expect(screen.getByText('Overuje sa')).toBeInTheDocument())
  })

  it('shows the error and a retry action when the download failed', async () => {
    server.use(http.get('/v1/models', () => HttpResponse.json(modelsWith('failed'))))
    renderStep()

    expect(await screen.findByRole('button', { name: 'Stiahnuť znova' })).toBeInTheDocument()
  })

  it('enables "Pokračovať" once the variant is ready', async () => {
    const onNext = vi.fn()
    server.use(http.get('/v1/models', () => HttpResponse.json(modelsWith('ready'))))
    renderStep(onNext)

    const continueButton = await screen.findByRole('button', { name: 'Pokračovať' })
    await waitFor(() => expect(continueButton).toHaveAttribute('aria-disabled', 'false'))

    await userEvent.click(continueButton)
    expect(onNext).toHaveBeenCalled()
  })

  it('offers no unpublished or planned catalogue variant to choose from', async () => {
    const [first] = modelsWith('ready').variants
    if (first === undefined) throw new Error('fixture has no variants')
    server.use(
      http.get('/v1/models', () =>
        HttpResponse.json({ ...modelsResponse, variants: [first, unpublishedModelVariant, plannedModelVariant] }),
      ),
    )
    renderStep()

    await waitFor(() => expect(screen.getByText('Qwen3-VL INT8')).toBeInTheDocument())
    expect(screen.queryByText(/Vybrať iný model/)).not.toBeInTheDocument()
  })

  it('"Dokončiť neskôr" always advances, even mid-download', async () => {
    const onNext = vi.fn()
    server.use(http.get('/v1/models', () => HttpResponse.json(modelsWith('downloading'))))
    renderStep(onNext)

    await userEvent.click(await screen.findByRole('button', { name: 'Dokončiť neskôr' }))
    expect(onNext).toHaveBeenCalled()
  })
})

it('retries a cached GPU failure with load and keeps Continue disabled', async () => {
  let loads = 0
  let downloads = 0
  const data = modelsWith('failed')
  const first = data.variants[0]
  if (first === undefined) throw new Error('missing fixture variant')
  data.variants[0] = {
    ...first,
    downloaded: true,
    resident: false,
    download: {
      bytes_done: null,
      bytes_total: null,
      command_id: 'failed-load',
      error_code: 'MODEL_VARIANT_DOES_NOT_FIT',
      state: 'failed',
    },
  }
  server.use(
    http.get('/v1/models', () => HttpResponse.json(data)),
    http.post('/v1/models/:id/load', () => {
      loads++
      return HttpResponse.json({}, { status: 202 })
    }),
    http.post('/v1/models/:id/download', () => {
      downloads++
      return HttpResponse.json({}, { status: 202 })
    }),
  )
  renderStep()
  await userEvent.click(await screen.findByRole('button', { name: 'Načítať na GPU' }))
  await waitFor(() => {
    expect(loads).toBe(1)
  })
  expect(downloads).toBe(0)
  expect(screen.getByRole('button', { name: 'Pokračovať' })).toHaveAttribute('aria-disabled', 'true')
})

it('does not treat a complete cache as a GPU-resident model', async () => {
  const data = modelsWith('ready')
  const first = data.variants[0]
  if (first === undefined) throw new Error('missing fixture variant')
  data.variants[0] = { ...first, resident: false }
  server.use(http.get('/v1/models', () => HttpResponse.json(data)))
  renderStep()
  await waitFor(() => expect(screen.getByText('Qwen3-VL INT8')).toBeInTheDocument())
  expect(screen.getByRole('button', { name: 'Pokračovať' })).toHaveAttribute('aria-disabled', 'true')
})

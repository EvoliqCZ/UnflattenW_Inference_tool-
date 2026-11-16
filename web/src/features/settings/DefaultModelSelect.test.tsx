import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { modelsResponse, plannedModelVariant, settingsResponse, unpublishedModelVariant } from '@/test/msw/fixtures'
import { DefaultModelSelect } from './DefaultModelSelect'

function renderSelect() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <DefaultModelSelect />
    </QueryClientProvider>,
  )
}

describe('DefaultModelSelect (UX §6 / design §3.10)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('lists only variants with fits !== false, with a note about the excluded one', async () => {
    server.use(http.get('/v1/models', () => HttpResponse.json(modelsResponse)))
    renderSelect()

    await userEvent.click(await screen.findByRole('combobox'))

    expect(screen.getByRole('option', { name: 'Qwen3-VL INT8' })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: 'Qwen3-VL BF16' })).not.toBeInTheDocument()
    expect(screen.getByText('Varianty, ktoré sa nezmestia, nie je možné nastaviť ako predvolené.')).toBeInTheDocument()
  })

  it('never offers an unpublished or planned catalogue variant as the default', async () => {
    server.use(
      http.get('/v1/models', () =>
        HttpResponse.json({
          ...modelsResponse,
          variants: [...modelsResponse.variants, unpublishedModelVariant, plannedModelVariant],
        }),
      ),
    )
    renderSelect()

    await userEvent.click(await screen.findByRole('combobox'))

    expect(screen.getByRole('option', { name: 'Qwen3-VL INT8' })).toBeInTheDocument()
    expect(screen.queryByRole('option', { name: unpublishedModelVariant.label })).not.toBeInTheDocument()
    expect(screen.queryByRole('option', { name: plannedModelVariant.label })).not.toBeInTheDocument()
  })

  it('sends PUT /v1/models/default with the chosen variant id', async () => {
    let sentBody: unknown = null
    server.use(
      http.get('/v1/models', () => HttpResponse.json(modelsResponse)),
      http.get('/v1/settings', () => HttpResponse.json(settingsResponse)),
      http.put('/v1/models/default', async ({ request }) => {
        sentBody = await request.json()
        return HttpResponse.json(modelsResponse)
      }),
    )
    renderSelect()

    await userEvent.click(await screen.findByRole('combobox'))
    await userEvent.click(screen.getByRole('option', { name: 'Qwen3-VL INT8' }))

    await waitFor(() => {
      expect(sentBody).toEqual({ variant_id: 'qwen3-vl-int8' })
    })
  })

  it('sends null when "Automaticky" is chosen', async () => {
    let sentBody: unknown = null
    server.use(
      http.get('/v1/models', () => HttpResponse.json(modelsResponse)),
      http.get('/v1/settings', () =>
        HttpResponse.json({
          ...settingsResponse,
          default_variant: { ...settingsResponse.default_variant, override: 'qwen3-vl-int8' },
        }),
      ),
      http.put('/v1/models/default', async ({ request }) => {
        sentBody = await request.json()
        return HttpResponse.json(modelsResponse)
      }),
    )
    renderSelect()

    await userEvent.click(await screen.findByRole('combobox'))
    await userEvent.click(screen.getByRole('option', { name: 'Automaticky' }))

    await waitFor(() => {
      expect(sentBody).toEqual({ variant_id: null })
    })
  })
})

it('refreshes the selected default after saving and allows clearing it', async () => {
  let override: string | null = null
  const saved: (string | null)[] = []
  server.use(
    http.get('/v1/models', () => HttpResponse.json(modelsResponse)),
    http.get('/v1/settings', () =>
      HttpResponse.json({
        ...settingsResponse,
        default_variant: { ...settingsResponse.default_variant, override },
      }),
    ),
    http.put('/v1/models/default', async ({ request }) => {
      override = ((await request.json()) as { variant_id: string | null }).variant_id
      saved.push(override)
      return HttpResponse.json(modelsResponse)
    }),
  )
  renderSelect()
  await userEvent.click(await screen.findByRole('combobox'))
  await userEvent.click(screen.getByRole('option', { name: 'Qwen3-VL INT8' }))
  await waitFor(() => expect(screen.getByRole('combobox')).toHaveTextContent('Qwen3-VL INT8'))
  await userEvent.click(screen.getByRole('combobox'))
  await userEvent.click(screen.getByRole('option', { name: 'Automaticky' }))
  await waitFor(() => expect(screen.getByRole('combobox')).toHaveTextContent('Automaticky'))
  expect(saved).toEqual(['qwen3-vl-int8', null])
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { modelsResponse } from '@/test/msw/fixtures'
import { ModelsPage } from './page'

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <ModelsPage />
    </QueryClientProvider>,
  )
}

describe('ModelsPage (UX §7.2/§7.3 "/models")', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('renders the device panel and one card per variant', async () => {
    server.use(http.get('/v1/models', () => HttpResponse.json(modelsResponse)))
    renderPage()

    await waitFor(() => {
      expect(screen.getByText('Worker ešte nenahlásil GPU.')).toBeInTheDocument()
    })
    expect(screen.getByText('Qwen3-VL INT8')).toBeInTheDocument()
    expect(screen.getByText('Qwen3-VL BF16')).toBeInTheDocument()
    expect(screen.getByText(/Modely sa sťahujú z Hugging Face/)).toBeInTheDocument()
  })

  it('shows the empty-list message when no variant is configured', async () => {
    server.use(
      http.get('/v1/models', () =>
        HttpResponse.json({ ...modelsResponse, default_variant: null, resident_variant: null, variants: [] }),
      ),
    )
    renderPage()

    await waitFor(() => {
      expect(screen.getByText('Nie je nakonfigurovaný žiadny model.')).toBeInTheDocument()
    })
  })
})

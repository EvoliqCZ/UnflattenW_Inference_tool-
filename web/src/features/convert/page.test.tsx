import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { modelsResponse } from '@/test/msw/fixtures'
import { SAMPLE_PDF_URL } from '@/features/setup/ReadyStep'
import { ConvertPage } from './page'

function renderPage(initialEntries: string[] = ['/']) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={initialEntries}>
        <ConvertPage />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

const readyReadiness = {
  service_ready: true,
  limits: {
    max_pages: 20,
    max_upload_bytes: 50_000_000,
    accepted_media_types: ['application/pdf', 'image/png', 'image/jpeg'],
  },
}

beforeEach(async () => {
  await i18n.changeLanguage('sk')
  server.use(
    http.get('/v1/readiness', () => HttpResponse.json(readyReadiness)),
    http.get('/v1/models', () => HttpResponse.json(modelsResponse)),
  )
  // Large enough that ImageSizeWarning (also `role="note"`) never renders here --
  // this suite only exercises PhotoQualityNote's visibility.
  vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue({ width: 2800, height: 3600, close: vi.fn() }))
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('ConvertPage (UX §4)', () => {
  it('disables submission with no files selected', async () => {
    renderPage()

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Odoslať/ })).toHaveAttribute('aria-disabled', 'true'),
    )
    expect(screen.getByText('Najprv vyber súbor.')).toBeInTheDocument()
  })

  it('shows PhotoQualityNote once an image is selected', async () => {
    renderPage()
    await waitFor(() => expect(screen.getByLabelText('Vybrať súbory')).not.toBeDisabled())

    await userEvent.upload(screen.getByLabelText('Vybrať súbory'), new File(['x'], 'scan.png', { type: 'image/png' }))

    await waitFor(() => expect(screen.getByRole('note')).toHaveTextContent('meranie S5'))
  })

  it('does not show PhotoQualityNote for a PDF-only selection', async () => {
    renderPage()
    await waitFor(() => expect(screen.getByLabelText('Vybrať súbory')).not.toBeDisabled())

    await userEvent.upload(
      screen.getByLabelText('Vybrať súbory'),
      new File(['%PDF-1.4'], 'doc.pdf', { type: 'application/pdf' }),
    )

    await waitFor(() => expect(screen.getByText('doc.pdf')).toBeInTheDocument())
    expect(screen.queryByRole('note')).not.toBeInTheDocument()
  })

  it('rejects a selection above the readiness page limit and shows the reason', async () => {
    renderPage()
    await waitFor(() => expect(screen.getByLabelText('Vybrať súbory')).not.toBeDisabled())
    const files = Array.from({ length: 25 }, (_, i) => new File(['x'], `p${String(i)}.png`, { type: 'image/png' }))

    await userEvent.upload(screen.getByLabelText('Vybrať súbory'), files)

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('25'))
    expect(screen.queryByText('p0.png')).not.toBeInTheDocument()
  })

  it('preloads the sample PDF into the queue when linked via ?sample=', async () => {
    server.use(
      http.get(SAMPLE_PDF_URL, () => HttpResponse.text('%PDF-1.4', { headers: { 'content-type': 'application/pdf' } })),
    )
    renderPage([`/?sample=${encodeURIComponent(SAMPLE_PDF_URL)}`])

    await waitFor(() => expect(screen.getByText('sample.pdf')).toBeInTheDocument())
  })
})

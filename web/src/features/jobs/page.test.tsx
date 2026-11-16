import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { emptyJobList, makeJobSummary } from '@/test/msw/fixtures'
import { JobsPage } from './page'

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <JobsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('JobsPage (UX §6)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows the empty state with no jobs at all', async () => {
    server.use(http.get('/v1/conversions', () => HttpResponse.json(emptyJobList)))
    renderPage()

    expect(await screen.findByText('Zatiaľ žiadne úlohy.')).toBeInTheDocument()
  })

  it('shows the empty-filter state when a filter matches nothing', async () => {
    let requestedStates: string[] = []
    server.use(
      http.get('/v1/conversions', ({ request }) => {
        requestedStates = new URL(request.url).searchParams.getAll('state')
        return HttpResponse.json(emptyJobList)
      }),
    )
    renderPage()
    await screen.findByText('Zatiaľ žiadne úlohy.')

    await userEvent.click(screen.getByRole('tab', { name: 'Zlyhané' }))

    await waitFor(() => {
      expect(requestedStates).toEqual(['failed'])
    })
    expect(await screen.findByText('Žiadna úloha nezodpovedá filtru.')).toBeInTheDocument()
  })

  it('loads a second page with "Načítať ďalšie" using the cursor', async () => {
    const page1 = {
      items: [makeJobSummary({ job_id: 'job-1', label: 'Prvá' })],
      next_cursor: 'cursor-2',
      schema_version: '1.0' as const,
    }
    const page2 = {
      items: [makeJobSummary({ job_id: 'job-2', label: 'Druhá' })],
      next_cursor: null,
      schema_version: '1.0' as const,
    }
    server.use(
      http.get('/v1/conversions', ({ request }) => {
        const cursor = new URL(request.url).searchParams.get('cursor')
        return HttpResponse.json(cursor === 'cursor-2' ? page2 : page1)
      }),
    )
    renderPage()

    await screen.findByText('Prvá')
    await userEvent.click(screen.getByRole('button', { name: 'Načítať ďalšie' }))

    expect(await screen.findByText('Druhá')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Načítať ďalšie' })).not.toBeInTheDocument()
  })

  it('filters loaded rows by the search box without a server request', async () => {
    const page = {
      items: [
        makeJobSummary({ job_id: 'job-1', label: 'Faktúra-2026' }),
        makeJobSummary({ job_id: 'job-2', label: 'Zmluva.pdf' }),
      ],
      next_cursor: null,
      schema_version: '1.0' as const,
    }
    server.use(http.get('/v1/conversions', () => HttpResponse.json(page)))
    renderPage()

    await screen.findByText('Faktúra-2026')
    await userEvent.type(screen.getByRole('searchbox', { name: 'Hľadať podľa označenia' }), 'zmluva')

    expect(screen.queryByText('Faktúra-2026')).not.toBeInTheDocument()
    expect(screen.getByText('Zmluva.pdf')).toBeInTheDocument()
  })

  it('purges a job from the table and returns to the (now empty) list', async () => {
    const page = {
      items: [makeJobSummary({ job_id: 'job-1', label: 'Stará úloha', state: 'succeeded' })],
      next_cursor: null,
      schema_version: '1.0' as const,
    }
    server.use(
      http.get('/v1/conversions', () => HttpResponse.json(page)),
      http.delete('/v1/conversions/:jobId', () => new HttpResponse(null, { status: 204 })),
    )
    renderPage()

    await screen.findByText('Stará úloha')
    await userEvent.click(screen.getByRole('button', { name: 'Akcie' }))
    await userEvent.click(screen.getByRole('menuitem', { name: 'Odstrániť natrvalo' }))
    await userEvent.click(screen.getByRole('button', { name: 'Odstrániť' }))

    await waitFor(() => expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument())
  })
})

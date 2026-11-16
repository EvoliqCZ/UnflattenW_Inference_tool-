import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { emptyJobList, makeJobSummary } from '@/test/msw/fixtures'
import { RecentJobs } from './RecentJobs'

function renderRecentJobs() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <RecentJobs />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('RecentJobs (UX §4.1)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('renders nothing while there are no jobs', async () => {
    server.use(http.get('/v1/conversions', () => HttpResponse.json(emptyJobList)))
    const { container } = renderRecentJobs()

    await waitFor(() => expect(container).toBeEmptyDOMElement())
  })

  it('lists recent jobs with a link to the full jobs list', async () => {
    server.use(
      http.get('/v1/conversions', () =>
        HttpResponse.json({
          items: [makeJobSummary({ job_id: 'job-1', label: 'Faktúra-2026' })],
          next_cursor: null,
          schema_version: '1.0',
        }),
      ),
    )
    renderRecentJobs()

    expect(await screen.findByRole('link', { name: 'Faktúra-2026' })).toHaveAttribute('href', '/jobs/job-1')
    expect(screen.getByRole('link', { name: 'Všetky úlohy' })).toHaveAttribute('href', '/jobs')
  })
})

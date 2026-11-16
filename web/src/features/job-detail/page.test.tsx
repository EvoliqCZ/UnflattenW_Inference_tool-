import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { MemoryRouter, Route, Routes } from 'react-router'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { runningJob, succeededJob } from '@/test/msw/fixtures'
import { JobDetailPage } from './page'

function renderDetail(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/jobs/:jobId" element={<JobDetailPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('JobDetailPage (UX §6/§7.2)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows the progress view with no tabs while the job is active', async () => {
    server.use(http.get('/v1/conversions/:jobId', () => HttpResponse.json(runningJob)))
    renderDetail(`/jobs/${runningJob.job_id}`)

    expect(await screen.findByText('Model')).toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: 'Výsledok' })).not.toBeInTheDocument()
  })

  it('shows the Result/Compare/Diagnostics tabs once the job is terminal, defaulting to Result', async () => {
    server.use(http.get('/v1/conversions/:jobId', () => HttpResponse.json(succeededJob)))
    renderDetail(`/jobs/${succeededJob.job_id}`)

    expect(await screen.findByRole('tab', { name: 'Výsledok' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Stiahnuť všetko (ZIP)' })).toBeInTheDocument()
  })

  it('switches to the Diagnostics tab and keeps it selected via ?tab=', async () => {
    server.use(http.get('/v1/conversions/:jobId', () => HttpResponse.json(succeededJob)))
    renderDetail(`/jobs/${succeededJob.job_id}`)

    await screen.findByRole('tab', { name: 'Výsledok' })
    await userEvent.click(screen.getByRole('tab', { name: 'Diagnostika' }))

    await waitFor(() => {
      expect(screen.getByRole('tab', { name: 'Diagnostika' })).toHaveAttribute('data-state', 'active')
    })
  })

  it('shows an expired notice instead of the comparison viewer for an expired job', async () => {
    const expiredJob = { ...succeededJob, state: 'expired' as const, expires_at: null }
    server.use(http.get('/v1/conversions/:jobId', () => HttpResponse.json(expiredJob)))
    renderDetail(`/jobs/${expiredJob.job_id}?tab=compare`)

    expect(await screen.findByText(/artefakty úlohy už expirovali/)).toBeInTheDocument()
  })

  it('shows a not-found message for a 404', async () => {
    server.use(
      http.get('/v1/conversions/:jobId', () =>
        HttpResponse.json(
          { code: 'RESOURCE_NOT_FOUND', title: 'Not found', status: 404, retryable: false },
          { status: 404, headers: { 'Content-Type': 'application/problem+json' } },
        ),
      ),
    )
    renderDetail('/jobs/job-missing')

    expect(await screen.findByText('Úloha neexistuje alebo bola zmazaná.')).toBeInTheDocument()
  })
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { makeJobSummary, modelsResponse, runningJob } from '@/test/msw/fixtures'
import { QueueReason } from './QueueReason'

function renderReason(jobId = 'job-queued') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <QueueReason jobId={jobId} />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('QueueReason (UX §3.4)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows "waiting for a worker" when no other job is running', () => {
    server.use(
      http.get('/v1/conversions', () => HttpResponse.json({ items: [], next_cursor: null, schema_version: '1.0' })),
    )
    renderReason()
    expect(screen.getByText('Čaká sa na pracovný proces')).toBeInTheDocument()
  })

  it('shows "waiting for the job queue" and links to the other running job', async () => {
    const otherJob = makeJobSummary({ job_id: 'job-other', label: 'Iná úloha', state: 'running' })
    server.use(
      http.get('/v1/conversions', () =>
        HttpResponse.json({ items: [otherJob], next_cursor: null, schema_version: '1.0' }),
      ),
    )
    renderReason('job-queued')
    const link = await screen.findByRole('link', { name: 'Iná úloha' })
    expect(link).toHaveAttribute('href', '/jobs/job-other')
    expect(screen.getByRole('status')).toHaveTextContent('Čaká sa vo fronte úloh')
  })

  it('shows "waiting for the model download" when the job\'s variant is still downloading', async () => {
    const [first, ...rest] = modelsResponse.variants
    if (first === undefined) throw new Error('fixture has no variants')
    server.use(
      http.get('/v1/conversions/{job_id}', () => HttpResponse.json(runningJob)),
      http.get('/v1/models', () =>
        HttpResponse.json({
          ...modelsResponse,
          variants: [
            {
              ...first,
              download: {
                bytes_done: 1000,
                bytes_total: 4000,
                command_id: null,
                error_code: null,
                state: 'downloading',
              },
            },
            ...rest,
          ],
        }),
      ),
      http.get('/v1/conversions', () => HttpResponse.json({ items: [], next_cursor: null, schema_version: '1.0' })),
    )
    renderReason(runningJob.job_id)

    await waitFor(() =>
      expect(screen.getByRole('status')).toHaveTextContent(`Čaká sa na stiahnutie modelu ${first.label}`),
    )
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
  })
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { runningJob, succeededJob } from '@/test/msw/fixtures'
import { JobHeader } from './JobHeader'

function renderHeader(job: Parameters<typeof JobHeader>[0]['job']) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <JobHeader job={job} language="sk" />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('JobHeader (UX §6)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows the job id, model, state badge and a Cancel action for an active job', () => {
    renderHeader(runningJob)
    expect(screen.getByRole('heading', { name: runningJob.job_id })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Zrušiť' })).toBeInTheDocument()
  })

  it('opens the cancel dialog on click', async () => {
    renderHeader(runningJob)
    await userEvent.click(screen.getByRole('button', { name: 'Zrušiť' }))
    expect(screen.getByRole('alertdialog')).toBeInTheDocument()
  })

  it('shows a Purge action instead of Cancel for a terminal job', () => {
    renderHeader(succeededJob)
    expect(screen.queryByRole('button', { name: 'Zrušiť' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Odstrániť natrvalo' })).toBeInTheDocument()
  })

  it('shows "Duration" instead of "Elapsed" once the job is terminal', () => {
    renderHeader(succeededJob)
    expect(screen.getByText(/Trvanie/)).toBeInTheDocument()
  })
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { MemoryRouter, Route, Routes } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { makeJobSummary } from '@/test/msw/fixtures'
import { PurgeJobDialog } from './PurgeJobDialog'

function LocationProbe({ label }: { label: string }) {
  return <p>{label}</p>
}

function renderDialog(onOpenChange = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={['/jobs/job-1']}>
        <Routes>
          <Route
            path="/jobs/:jobId"
            element={
              <PurgeJobDialog
                job={makeJobSummary({ job_id: 'job-1', label: 'Zmluva.pdf' })}
                open
                onOpenChange={onOpenChange}
              />
            }
          />
          <Route path="/jobs" element={<LocationProbe label="jobs-list" />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
  return onOpenChange
}

describe('PurgeJobDialog (UX §3.6)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('navigates back to the jobs list after a successful purge', async () => {
    server.use(http.delete('/v1/conversions/:jobId', () => new HttpResponse(null, { status: 204 })))
    const onOpenChange = renderDialog()

    await userEvent.click(screen.getByRole('button', { name: 'Odstrániť' }))

    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
    expect(await screen.findByText('jobs-list')).toBeInTheDocument()
  })
})

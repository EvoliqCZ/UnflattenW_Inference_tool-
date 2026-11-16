import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { Toaster } from '@/components/ui/toaster'
import { server } from '@/test/msw/server'
import { jobTerminalProblem, makeJobSummary } from '@/test/msw/fixtures'
import { CancelJobDialog } from './CancelJobDialog'

function renderDialog(onOpenChange = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <Toaster />
      <CancelJobDialog
        job={makeJobSummary({ job_id: 'job-1', label: 'Faktúra-2026' })}
        open
        onOpenChange={onOpenChange}
      />
    </QueryClientProvider>,
  )
  return onOpenChange
}

describe('CancelJobDialog (UX §3.5)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('closes after a successful cancel', async () => {
    server.use(http.delete('/v1/conversions/:jobId', () => HttpResponse.json(makeJobSummary({ state: 'cancelled' }))))
    const onOpenChange = renderDialog()

    await userEvent.click(screen.getByRole('button', { name: 'Zrušiť' }))

    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
  })

  it('closes and shows a toast on a 409 JOB_TERMINAL response, instead of treating it as an error', async () => {
    server.use(
      http.delete('/v1/conversions/:jobId', () =>
        HttpResponse.json(jobTerminalProblem, { status: 409, headers: { 'Content-Type': 'application/problem+json' } }),
      ),
    )
    const onOpenChange = renderDialog()

    await userEvent.click(screen.getByRole('button', { name: 'Zrušiť' }))

    await waitFor(() => {
      expect(onOpenChange).toHaveBeenCalledWith(false)
    })
    expect(await screen.findByText('Úloha už skončila.')).toBeInTheDocument()
  })
})

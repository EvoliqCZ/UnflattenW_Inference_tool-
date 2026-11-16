import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { queryKeys } from '@/api/queries'
import { server } from '@/test/msw/server'
import { authenticatedSession, authenticationFailedProblem } from '@/test/msw/fixtures'
import { SessionExpiredDialog, sessionExpiredController } from './SessionExpiredDialog'

describe('SessionExpiredDialog', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  afterEach(() => {
    sessionExpiredController.close()
  })

  function renderDialog() {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={queryClient}>
        <SessionExpiredDialog />
      </QueryClientProvider>,
    )
    return queryClient
  }

  it('is hidden until the controller opens it', () => {
    renderDialog()

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('shows the password form on the spot when the controller opens', async () => {
    renderDialog()

    sessionExpiredController.open()

    expect(await screen.findByRole('dialog', { name: 'Relácia vypršala' })).toBeInTheDocument()
    expect(screen.getByLabelText('Heslo')).toBeInTheDocument()
  })

  it('logs in, closes, and replays the stored retry on a successful submit (GET-only retry)', async () => {
    server.use(http.post('/v1/session', () => HttpResponse.json(authenticatedSession, { status: 201 })))
    const user = userEvent.setup()
    const queryClient = renderDialog()
    const retry = vi.fn()
    sessionExpiredController.open(retry)

    await user.type(await screen.findByLabelText('Heslo'), 'correct horse battery staple')
    await user.click(screen.getByRole('button', { name: 'Prihlásiť sa' }))

    await waitFor(() => {
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    })
    expect(retry).toHaveBeenCalledTimes(1)
    expect(queryClient.getQueryData(queryKeys.session())).toEqual(authenticatedSession)
  })

  it('shows "Nesprávne heslo" on a 401 and keeps the dialog (and the retry) pending', async () => {
    server.use(
      http.post('/v1/session', () =>
        HttpResponse.json(authenticationFailedProblem, {
          status: 401,
          headers: { 'content-type': 'application/problem+json' },
        }),
      ),
    )
    const user = userEvent.setup()
    renderDialog()
    const retry = vi.fn()
    sessionExpiredController.open(retry)

    await user.type(await screen.findByLabelText('Heslo'), 'wrong password')
    await user.click(screen.getByRole('button', { name: 'Prihlásiť sa' }))

    expect(await screen.findByText('Prihlásenie zlyhalo')).toBeInTheDocument()
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(retry).not.toHaveBeenCalled()
  })
})

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { Toaster } from '@/components/ui/toaster'
import { ChangePasswordDialog } from './ChangePasswordDialog'

function renderDialog() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <ChangePasswordDialog />
      <Toaster />
    </QueryClientProvider>,
  )
}

function lastSubmitButton(): HTMLElement {
  const buttons = screen.getAllByRole('button', { name: 'Zmeniť heslo' })
  const last = buttons.at(-1)
  if (last === undefined) {
    throw new Error('expected a "Zmeniť heslo" button')
  }
  return last
}

async function fillAndSubmit(
  user: ReturnType<typeof userEvent.setup>,
  { current, next, confirm }: { current: string; next: string; confirm: string },
) {
  await user.click(screen.getByRole('button', { name: 'Zmeniť heslo' }))
  await user.type(screen.getByLabelText('Súčasné heslo'), current)
  await user.type(screen.getByLabelText('Nové heslo'), next)
  await user.type(screen.getByLabelText('Potvrdenie nového hesla'), confirm)
  await user.click(lastSubmitButton())
}

describe('ChangePasswordDialog (UX §6 Nastavenia -> Účet)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('submits current_password/new_password and closes with a toast on 204', async () => {
    let receivedBody: unknown
    server.use(
      http.put('/v1/setup/password', async ({ request }) => {
        receivedBody = await request.json()
        return new HttpResponse(null, { status: 204 })
      }),
    )
    const user = userEvent.setup()
    renderDialog()

    await fillAndSubmit(user, {
      current: 'old-password',
      next: 'new-and-long-enough',
      confirm: 'new-and-long-enough',
    })

    expect(await screen.findByText('Heslo bolo zmenené.')).toBeInTheDocument()
    expect(receivedBody).toEqual({ current_password: 'old-password', new_password: 'new-and-long-enough' })
    expect(screen.queryByLabelText('Súčasné heslo')).not.toBeInTheDocument()
  })

  it('shows an inline error and keeps the dialog open on a 401 (wrong current password)', async () => {
    server.use(
      http.put('/v1/setup/password', () =>
        HttpResponse.json(
          {
            blockers: null,
            code: 'AUTHENTICATION_FAILED',
            max_pages: null,
            retry_after_seconds: null,
            retryable: false,
            status: 401,
            title: 'Authentication failed',
            type: null,
          },
          { status: 401, headers: { 'content-type': 'application/problem+json' } },
        ),
      ),
    )
    const user = userEvent.setup()
    renderDialog()

    await fillAndSubmit(user, { current: 'wrong', next: 'new-and-long-enough', confirm: 'new-and-long-enough' })

    expect(await screen.findByText('Súčasné heslo je nesprávne.')).toBeInTheDocument()
    expect(screen.getByLabelText('Súčasné heslo')).toBeInTheDocument()
  })

  it('disables submit with the retry_after_seconds countdown on a 429', async () => {
    server.use(
      http.put('/v1/setup/password', () =>
        HttpResponse.json(
          {
            blockers: null,
            code: 'RATE_LIMITED',
            max_pages: null,
            retry_after_seconds: 42,
            retryable: true,
            status: 429,
            title: 'Rate limited',
            type: null,
          },
          { status: 429, headers: { 'content-type': 'application/problem+json' } },
        ),
      ),
    )
    const user = userEvent.setup()
    renderDialog()

    await fillAndSubmit(user, {
      current: 'old-password',
      next: 'new-and-long-enough',
      confirm: 'new-and-long-enough',
    })

    expect(await screen.findByText('Príliš veľa pokusov. Skús to znova o 42 s.')).toHaveAttribute('aria-disabled')
  })

  it('shows a client-side error when the new password is too short', async () => {
    const user = userEvent.setup()
    renderDialog()

    await fillAndSubmit(user, { current: 'old-password', next: 'short', confirm: 'short' })

    expect(await screen.findByText('Heslo musí mať aspoň 12 znakov.')).toBeInTheDocument()
  })

  it('shows a client-side error when new and confirm do not match', async () => {
    const user = userEvent.setup()
    renderDialog()

    await fillAndSubmit(user, { current: 'old-password', next: 'new-and-long-enough', confirm: 'different-value' })

    expect(await screen.findByText('Heslá sa nezhodujú.')).toBeInTheDocument()
  })
})

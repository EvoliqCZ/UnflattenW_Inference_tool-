import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { MemoryRouter, Route, Routes } from 'react-router'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { authenticatedSession, authenticationFailedProblem } from '@/test/msw/fixtures'
import { LoginPage } from './page'

function renderLoginPage(initialEntries: string[] = ['/login']) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={initialEntries}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<h1>Nová konverzia</h1>} />
          <Route path="/jobs" element={<h1>Úlohy</h1>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('LoginPage (UX §3.2)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('renders a password field and a submit button', () => {
    renderLoginPage()

    expect(screen.getByLabelText('Heslo')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Prihlásiť sa' })).toBeInTheDocument()
  })

  it('shows "Nesprávne heslo." on a 401 and keeps the typed password (does not clear the field)', async () => {
    server.use(
      http.post('/v1/session', () =>
        HttpResponse.json(authenticationFailedProblem, {
          status: 401,
          headers: { 'content-type': 'application/problem+json' },
        }),
      ),
    )
    const user = userEvent.setup()
    renderLoginPage()

    const passwordField = screen.getByLabelText('Heslo')
    await user.type(passwordField, 'typo-password')
    await user.click(screen.getByRole('button', { name: 'Prihlásiť sa' }))

    expect(await screen.findByText('Nesprávne heslo.')).toBeInTheDocument()
    expect(passwordField).toHaveValue('typo-password')
  })

  it('shows a countdown and marks the submit button aria-disabled on 429 (design retry_after_seconds)', async () => {
    server.use(
      http.post('/v1/session', () =>
        HttpResponse.json(
          {
            blockers: null,
            code: 'RATE_LIMITED',
            max_pages: null,
            retry_after_seconds: 42,
            retryable: true,
            status: 429,
            title: 'Too many requests',
            type: null,
          },
          { status: 429, headers: { 'content-type': 'application/problem+json' } },
        ),
      ),
    )
    const user = userEvent.setup()
    renderLoginPage()

    await user.type(screen.getByLabelText('Heslo'), 'some-password')
    await user.click(screen.getByRole('button', { name: 'Prihlásiť sa' }))

    const submitButton = await screen.findByRole('button', { name: /42 s/ })
    expect(submitButton).toHaveAttribute('aria-disabled', 'true')
  })

  it('navigates to "/" on a successful login with no next param', async () => {
    server.use(http.post('/v1/session', () => HttpResponse.json(authenticatedSession, { status: 201 })))
    const user = userEvent.setup()
    renderLoginPage()

    await user.type(screen.getByLabelText('Heslo'), 'correct horse battery staple')
    await user.click(screen.getByRole('button', { name: 'Prihlásiť sa' }))

    expect(await screen.findByRole('heading', { name: 'Nová konverzia' })).toBeInTheDocument()
  })

  it('navigates to a sanitized "next" route on success', async () => {
    server.use(http.post('/v1/session', () => HttpResponse.json(authenticatedSession, { status: 201 })))
    const user = userEvent.setup()
    renderLoginPage(['/login?next=%2Fjobs'])

    await user.type(screen.getByLabelText('Heslo'), 'correct horse battery staple')
    await user.click(screen.getByRole('button', { name: 'Prihlásiť sa' }))

    expect(await screen.findByRole('heading', { name: 'Úlohy' })).toBeInTheDocument()
  })

  it('toggles the password field visibility', async () => {
    const user = userEvent.setup()
    renderLoginPage()

    const passwordField = screen.getByLabelText('Heslo')
    expect(passwordField).toHaveAttribute('type', 'password')

    await user.click(screen.getByRole('button', { name: 'Zobraziť heslo' }))
    expect(passwordField).toHaveAttribute('type', 'text')
  })

  it('reveals the reset-password command under "Zabudnuté heslo?"', async () => {
    const user = userEvent.setup()
    renderLoginPage()

    await user.click(screen.getByText('Zabudnuté heslo?'))

    expect(await screen.findByText('./Spustit.sh reset-password')).toBeInTheDocument()
  })
})

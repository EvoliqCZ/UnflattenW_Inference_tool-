import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import type { components } from '@/api/schema'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import {
  authenticatedSession,
  authenticationFailedProblem,
  setupRequiredSession,
  setupTokenlessSession,
} from '@/test/msw/fixtures'
import { SetupPage } from './page'

type SessionInfo = components['schemas']['SessionInfo']

const VALID_PASSWORD = 'correct horse battery staple'

function renderSetupPage(session: SessionInfo = setupTokenlessSession) {
  server.use(http.get('/v1/session', () => HttpResponse.json(session)))
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/setup']}>
        <Routes>
          <Route path="/setup" element={<SetupPage />} />
          <Route path="/login" element={<h1>Prihlásenie</h1>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

async function fillPasswordStep(
  user: ReturnType<typeof userEvent.setup>,
  password = VALID_PASSWORD,
  confirm = password,
) {
  await user.type(await screen.findByLabelText('Heslo'), password)
  await user.type(screen.getByLabelText('Potvrdenie hesla'), confirm)
  await user.click(screen.getByRole('button', { name: 'Vytvoriť heslo' }))
}

describe('SetupPage (UX §3.1, owner 2026-09-29: no token field)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
    window.history.replaceState(null, '', '/setup')
  })

  afterEach(() => {
    window.history.replaceState(null, '', '/')
  })

  it('tokenless: opens on the password step with no token field anywhere', async () => {
    renderSetupPage()

    expect(await screen.findByLabelText('Heslo')).toBeInTheDocument()
    expect(screen.queryByText(/token/i)).not.toBeInTheDocument()
    expect(screen.queryByText('Prístupový token')).not.toBeInTheDocument()
  })

  it('tokenless: POST /v1/setup carries no Authorization header and advances to step "system"', async () => {
    server.use(
      http.post('/v1/setup', async ({ request }) => {
        expect(request.headers.get('authorization')).toBeNull()
        expect(await request.json()).toEqual({ password: VALID_PASSWORD })
        return HttpResponse.json(authenticatedSession, { status: 201 })
      }),
      http.get('/v1/readiness', () => HttpResponse.json({ service_ready: true })),
    )
    const user = userEvent.setup()
    renderSetupPage()

    await fillPasswordStep(user)

    expect(await screen.findByText('Systém je pripravený.')).toBeInTheDocument()
  })

  it('setup link: reads #token=, strips it from the address bar at once, and sends it as a bearer header', async () => {
    window.history.replaceState(null, '', '/setup#token=link-token')
    server.use(
      http.post('/v1/setup', ({ request }) => {
        expect(request.headers.get('authorization')).toBe('Bearer link-token')
        return HttpResponse.json(authenticatedSession, { status: 201 })
      }),
      http.get('/v1/readiness', () => HttpResponse.json({ service_ready: true })),
    )
    const user = userEvent.setup()
    renderSetupPage(setupRequiredSession)

    expect(await screen.findByLabelText('Heslo')).toBeInTheDocument()
    expect(window.location.hash).toBe('')
    expect(window.location.pathname).toBe('/setup')

    await fillPasswordStep(user)

    expect(await screen.findByText('Systém je pripravený.')).toBeInTheDocument()
  })

  it('token required and no link token: shows the "open the setup link" message instead of a form', async () => {
    renderSetupPage(setupRequiredSession)

    expect(
      await screen.findByText('Otvor odkaz na prvotné nastavenie, ktorý vypísal run-local.sh.'),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText('Heslo')).not.toBeInTheDocument()
  })

  it('token required and no link token: the message is translated to English', async () => {
    await i18n.changeLanguage('en')
    renderSetupPage(setupRequiredSession)

    expect(await screen.findByText('Open the setup link printed by run-local.sh.')).toBeInTheDocument()
  })

  it('setup link: a 401 (wrong or stale token) shows the invalid-link message', async () => {
    window.history.replaceState(null, '', '/setup#token=stale-token')
    server.use(
      http.post('/v1/setup', () =>
        HttpResponse.json(authenticationFailedProblem, {
          status: 401,
          headers: { 'content-type': 'application/problem+json' },
        }),
      ),
    )
    const user = userEvent.setup()
    renderSetupPage(setupRequiredSession)

    await fillPasswordStep(user)

    expect(
      await screen.findByText('Odkaz na nastavenie neplatí. Otvor znova odkaz, ktorý vypísal run-local.sh.'),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText('Heslo')).not.toBeInTheDocument()
  })

  it('tokenless: a 401 (server no longer offers tokenless setup) shows the "open the setup link" message', async () => {
    server.use(
      http.post('/v1/setup', () =>
        HttpResponse.json(authenticationFailedProblem, {
          status: 401,
          headers: { 'content-type': 'application/problem+json' },
        }),
      ),
    )
    const user = userEvent.setup()
    renderSetupPage()

    await fillPasswordStep(user)

    expect(
      await screen.findByText('Otvor odkaz na prvotné nastavenie, ktorý vypísal run-local.sh.'),
    ).toBeInTheDocument()
  })

  it('rejects a password shorter than 12 characters without an API call', async () => {
    const user = userEvent.setup()
    renderSetupPage()

    await fillPasswordStep(user, 'short', 'short')

    expect(await screen.findByText('Heslo musí mať aspoň 12 znakov.')).toBeInTheDocument()
  })

  it('rejects a mismatched confirmation without an API call', async () => {
    const user = userEvent.setup()
    renderSetupPage()

    await fillPasswordStep(user, VALID_PASSWORD, 'a different password entirely')

    expect(await screen.findByText('Heslá sa nezhodujú.')).toBeInTheDocument()
  })

  it('a network failure on submit shows the generic network-error text', async () => {
    server.use(http.post('/v1/setup', () => HttpResponse.error()))
    const user = userEvent.setup()
    renderSetupPage()

    await fillPasswordStep(user)

    expect(await screen.findByText('Službu sa nepodarilo dosiahnuť. Skús to znova.')).toBeInTheDocument()
  })

  it('a 429 RATE_LIMITED shows the problem catalog message', async () => {
    server.use(
      http.post('/v1/setup', () =>
        HttpResponse.json(
          {
            blockers: null,
            code: 'RATE_LIMITED',
            max_pages: null,
            retry_after_seconds: 30,
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
    renderSetupPage()

    await fillPasswordStep(user)

    expect(await screen.findByText('Príliš veľa požiadaviek')).toBeInTheDocument()
  })

  it('a 409 SETUP_ALREADY_COMPLETE redirects to /login', async () => {
    server.use(
      http.post('/v1/setup', () =>
        HttpResponse.json(
          {
            blockers: null,
            code: 'SETUP_ALREADY_COMPLETE',
            max_pages: null,
            retry_after_seconds: null,
            retryable: false,
            status: 409,
            title: 'Setup already complete',
            type: null,
          },
          { status: 409, headers: { 'content-type': 'application/problem+json' } },
        ),
      ),
    )
    const user = userEvent.setup()
    renderSetupPage()

    await fillPasswordStep(user)

    expect(await screen.findByRole('heading', { name: 'Prihlásenie' })).toBeInTheDocument()
  })
})

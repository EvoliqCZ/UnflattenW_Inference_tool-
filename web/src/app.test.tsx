import { render, screen } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import { beforeEach, describe, expect, it } from 'vitest'
import { createQueryClient } from '@/api/queryClient'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { authenticatedSession, setupRequiredSession, unauthenticatedSession } from '@/test/msw/fixtures'
import { App } from './app'
import { createAppRouter } from './router'

/** Each test gets its own `QueryClient`/router pair (see `app.tsx`'s `AppProps` note) -- no shared session cache. */
function renderApp() {
  const queryClient = createQueryClient()
  const router = createAppRouter(queryClient)
  return render(<App queryClient={queryClient} router={router} />)
}

/**
 * T26 -> T29 (plan §I): regression coverage for the real `createBrowserRouter`
 * running under MSW/jsdom. Before the `test/setup.ts` fix, mounting the real
 * router (as opposed to `router.test.tsx`'s route-list-only assertions)
 * threw synchronously from the guard loader's first fetch, because
 * react-router's own `AbortController` (jsdom's) and the patched `Request`
 * (undici's) disagreed about whose `AbortSignal` this is.
 */
describe('App (real router + QueryClientProvider, design §7.1/§7.5)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('renders the default route once the guard loader resolves an authenticated session', async () => {
    server.use(http.get('/v1/session', () => HttpResponse.json(authenticatedSession)))

    renderApp()

    expect(await screen.findByRole('heading', { name: 'Prevod dokumentu' })).toBeInTheDocument()
  })

  it('redirects to /login when the guard loader resolves no session', async () => {
    server.use(http.get('/v1/session', () => HttpResponse.json(unauthenticatedSession)))

    renderApp()

    expect(await screen.findByRole('heading', { name: 'Prihlásenie' })).toBeInTheDocument()
  })

  it('redirects to /setup when setup_required is true, even before /login', async () => {
    server.use(http.get('/v1/session', () => HttpResponse.json(setupRequiredSession)))

    renderApp()

    expect(await screen.findByRole('heading', { name: 'Prvotné nastavenie' })).toBeInTheDocument()
  })
})

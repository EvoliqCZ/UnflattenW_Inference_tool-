import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { settingsResponse } from '@/test/msw/fixtures'
import { AboutService } from './AboutService'

function renderAboutService() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <AboutService />
    </QueryClientProvider>,
  )
}

describe('AboutService (UX §6 Nastavenia -> O službe)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows git SHA, API version and session limits from GET /v1/settings', async () => {
    server.use(http.get('/v1/settings', () => HttpResponse.json(settingsResponse)))
    renderAboutService()

    expect(await screen.findByText('57db768')).toBeInTheDocument()
    expect(screen.getByText('1.1')).toBeInTheDocument()
    expect(screen.getByText('Časový limit nečinnosti relácie: 1800 s')).toBeInTheDocument()
    expect(screen.getByText('Maximálna dĺžka relácie: 28800 s')).toBeInTheDocument()
    expect(screen.getByText('Unflatten W 1.0')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'EVOLIQ s.r.o.' })).toHaveAttribute('href', 'https://www.evoliq.cz/sk')
  })

  it('does not render the rootful sentence when sandbox.rootful is false', async () => {
    server.use(http.get('/v1/settings', () => HttpResponse.json(settingsResponse)))
    renderAboutService()

    await screen.findByText('57db768')
    expect(screen.queryByText('Sandbox beží cez docker socket hostiteľa (práva roota).')).not.toBeInTheDocument()
  })

  it('renders the rootful sentence when sandbox.rootful is true', async () => {
    server.use(http.get('/v1/settings', () => HttpResponse.json({ ...settingsResponse, sandbox: { rootful: true } })))
    renderAboutService()

    expect(await screen.findByText('Sandbox beží cez docker socket hostiteľa (práva roota).')).toBeInTheDocument()
  })

  it('does not render the rootful sentence when sandbox.rootful is null (not yet published)', async () => {
    server.use(http.get('/v1/settings', () => HttpResponse.json({ ...settingsResponse, sandbox: { rootful: null } })))
    renderAboutService()

    await screen.findByText('57db768')
    expect(screen.queryByText('Sandbox beží cez docker socket hostiteľa (práva roota).')).not.toBeInTheDocument()
  })
})

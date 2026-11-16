import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { MemoryRouter, Route, Routes } from 'react-router'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { server } from '@/test/msw/server'
import { settingsResponse } from '@/test/msw/fixtures'
import { SettingsPage } from './page'

function renderSettingsPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/settings']}>
        <Routes>
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/login" element={<h1>Prihlásenie</h1>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('SettingsPage (UX §6 Nastavenia)', () => {
  beforeEach(async () => {
    window.localStorage.clear()
    document.documentElement.classList.remove('dark')
    await i18n.changeLanguage('sk')
  })

  it('renders language, theme, account and about sections', async () => {
    server.use(http.get('/v1/settings', () => HttpResponse.json(settingsResponse)))
    renderSettingsPage()

    expect(screen.getByRole('heading', { name: 'Nastavenia', level: 1 })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Slovenčina' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Podľa systému' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Účet' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'O aplikácii' })).toBeInTheDocument()
    expect(await screen.findByText('57db768')).toBeInTheDocument()
  })

  it('logs out (DELETE /v1/session) and navigates to /login when "Odhlásiť" is clicked', async () => {
    let deleteCalled = false
    server.use(
      http.get('/v1/settings', () => HttpResponse.json(settingsResponse)),
      http.delete('/v1/session', () => {
        deleteCalled = true
        return new HttpResponse(null, { status: 204 })
      }),
    )
    const user = userEvent.setup()
    renderSettingsPage()

    await user.click(screen.getByRole('button', { name: 'Odhlásiť sa' }))

    expect(await screen.findByRole('heading', { name: 'Prihlásenie' })).toBeInTheDocument()
    expect(deleteCalled).toBe(true)
  })
})

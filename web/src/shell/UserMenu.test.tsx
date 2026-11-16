import { setLanguage } from '@/i18n'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { MemoryRouter, Route, Routes } from 'react-router'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { server } from '@/test/msw/server'
import { UserMenu } from './UserMenu'

function renderUserMenu() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/']}>
        <Routes>
          <Route path="/" element={<UserMenu />} />
          <Route path="/settings" element={<h1>Nastavenia účtu</h1>} />
          <Route path="/login" element={<h1>Prihlásenie</h1>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  setLanguage('sk')
})

describe('UserMenu', () => {
  beforeEach(() => {
    window.localStorage.clear()
    document.documentElement.classList.remove('dark')
  })

  afterEach(() => {
    document.documentElement.classList.remove('dark')
  })

  it('should list theme and account entries', async () => {
    const user = userEvent.setup()
    renderUserMenu()

    await user.click(screen.getByRole('button', { name: 'Nastavenia a téma' }))

    expect(screen.getByRole('menuitem', { name: 'Nastavenia' })).toBeInTheDocument()
    expect(screen.getByRole('menuitemradio', { name: /Podľa systému/ })).toBeInTheDocument()
    expect(screen.getByRole('menuitemradio', { name: /Svetlá/ })).toBeInTheDocument()
    expect(screen.getByRole('menuitemradio', { name: /Tmavá/ })).toBeInTheDocument()
    expect(screen.getByRole('menuitem', { name: 'Odhlásiť sa' })).not.toHaveAttribute('aria-disabled')
  })

  it('should apply the dark class and persist the choice when Tmavá is selected', async () => {
    const user = userEvent.setup()
    renderUserMenu()

    await user.click(screen.getByRole('button', { name: 'Nastavenia a téma' }))
    await user.click(screen.getByRole('menuitemradio', { name: /Tmavá/ }))

    expect(document.documentElement.classList.contains('dark')).toBe(true)
    expect(window.localStorage.getItem('unflatten-theme')).toBe('dark')
  })

  it('logs out (DELETE /v1/session) and navigates to /login when "Odhlásiť" is selected', async () => {
    let deleteCalled = false
    server.use(
      http.delete('/v1/session', () => {
        deleteCalled = true
        return new HttpResponse(null, { status: 204 })
      }),
    )
    const user = userEvent.setup()
    renderUserMenu()

    await user.click(screen.getByRole('button', { name: 'Nastavenia a téma' }))
    await user.click(screen.getByRole('menuitem', { name: 'Odhlásiť sa' }))

    expect(await screen.findByRole('heading', { name: 'Prihlásenie' })).toBeInTheDocument()
    expect(deleteCalled).toBe(true)
  })

  it('still navigates to /login even if the logout request itself fails', async () => {
    server.use(http.delete('/v1/session', () => HttpResponse.error()))
    const user = userEvent.setup()
    renderUserMenu()

    await user.click(screen.getByRole('button', { name: 'Nastavenia a téma' }))
    await user.click(screen.getByRole('menuitem', { name: 'Odhlásiť sa' }))

    expect(await screen.findByRole('heading', { name: 'Prihlásenie' })).toBeInTheDocument()
  })
})

it('opens settings from the user menu', async () => {
  renderUserMenu()
  await userEvent.click(screen.getByRole('button', { name: 'Nastavenia a téma' }))
  await userEvent.click(screen.getByRole('menuitem', { name: 'Nastavenia' }))
  expect(await screen.findByRole('heading', { name: 'Nastavenia účtu' })).toBeInTheDocument()
})

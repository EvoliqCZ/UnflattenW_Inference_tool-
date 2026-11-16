import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { HttpResponse, http } from 'msw'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { LoginPage } from '@/features/login/page'
import { SetupPage } from '@/features/setup/page'
import { ChangePasswordDialog } from '@/features/settings/ChangePasswordDialog'
import { Toast, ToastClose, ToastProvider, ToastTitle, ToastViewport } from '@/components/ui/toast'
import { AppShell } from '@/shell/AppShell'
import { AuthShell } from '@/shell/AuthShell'
import { server } from '@/test/msw/server'
import { setupTokenlessSession } from '@/test/msw/fixtures'
import i18n, { LANGUAGE_STORAGE_KEY, setLanguage } from './index'

function providers(children: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  localStorage.clear()
  setLanguage('en')
})
afterEach(() => {
  vi.restoreAllMocks()
})

describe('English application-owned shell', () => {
  it('translates navigation, theme menu and skip link in the product shell', async () => {
    providers(
      <AppShell>
        <p>Faktúra zákazníka — user content</p>
      </AppShell>,
    )
    expect(screen.queryByRole('note')).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Skip to content' })).toHaveAttribute('href', '#main-content')
    const nav = screen.getByRole('navigation', { name: 'Main navigation' })
    expect(within(nav).getByRole('link', { name: 'Convert' })).toHaveAttribute('href', '/')
    expect(within(nav).getByRole('link', { name: 'Jobs' })).toHaveAttribute('href', '/jobs')
    expect(within(nav).getByRole('link', { name: 'Models' })).toHaveAttribute('href', '/models')
    expect(screen.getByText('Faktúra zákazníka — user content')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Settings and theme' }))
    expect(screen.getByRole('menuitem', { name: 'Settings' })).toBeInTheDocument()
    for (const name of ['System', 'Light', 'Dark'])
      expect(screen.getByRole('menuitemradio', { name })).toBeInTheDocument()
    expect(screen.getByRole('menuitem', { name: 'Log out' })).toBeInTheDocument()
  })

  it('opens and dismisses English mobile navigation', async () => {
    providers(
      <AppShell>
        <p>content</p>
      </AppShell>,
    )
    await userEvent.click(screen.getByRole('button', { name: 'Open navigation' }))
    const sheet = screen.getByRole('dialog')
    expect(within(sheet).getByRole('link', { name: 'Models' })).toBeInTheDocument()
    await userEvent.click(within(sheet).getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('localizes portalled dialog and toast dismissal controls', async () => {
    providers(<ChangePasswordDialog />)
    await userEvent.click(screen.getByRole('button', { name: 'Change password' }))
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('names the toast dismissal in English', () => {
    render(
      <ToastProvider>
        <Toast open>
          <ToastTitle>Saved</ToastTitle>
          <ToastClose />
        </Toast>
        <ToastViewport />
      </ToastProvider>,
    )
    expect(screen.getByRole('button', { name: 'Close' })).toBeInTheDocument()
  })
})

describe('language choice before authentication', () => {
  it('switches login immediately and retains entered password', async () => {
    providers(
      <AuthShell>
        <LoginPage />
      </AuthShell>,
    )
    await userEvent.type(screen.getByLabelText('Password'), 'entered-but-not-sent')
    await userEvent.click(screen.getByRole('button', { name: 'Slovenčina' }))
    expect(screen.getByLabelText('Heslo')).toHaveValue('entered-but-not-sent')
    expect(screen.getByRole('button', { name: 'Prihlásiť sa' })).toBeInTheDocument()
    expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('sk')
    expect(document.documentElement.lang).toBe('sk')
    await userEvent.click(screen.getByRole('button', { name: 'English' }))
    expect(screen.getByRole('button', { name: 'Log in' })).toBeInTheDocument()
    expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('en')
  })

  it('offers English tokenless setup before password creation', async () => {
    server.use(http.get('/v1/session', () => HttpResponse.json(setupTokenlessSession)))
    providers(
      <AuthShell>
        <SetupPage />
      </AuthShell>,
    )
    expect(await screen.findByLabelText('Password')).toBeInTheDocument()
    expect(screen.getByLabelText('Confirm password')).toBeInTheDocument()
    expect(screen.queryByLabelText(/token/i)).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Slovenčina' }))
    expect(screen.getByLabelText('Potvrdenie hesla')).toBeInTheDocument()
    expect(screen.queryByRole('note')).not.toBeInTheDocument()
  })

  it('applies a chosen language even when storage writes are denied', async () => {
    await i18n.changeLanguage('sk')
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('Denied', 'SecurityError')
    })
    providers(
      <AuthShell>
        <LoginPage />
      </AuthShell>,
    )
    await userEvent.click(screen.getByRole('button', { name: 'English' }))
    expect(screen.getByLabelText('Password')).toBeInTheDocument()
    expect(document.documentElement.lang).toBe('en')
  })
})

describe('authenticated app with unavailable browser storage', () => {
  it.each(['property', 'read', 'write'] as const)(
    'keeps navigation and theme switching usable after a denied %s',
    async (failure) => {
      const denied = () => {
        throw new DOMException('Denied', 'SecurityError')
      }
      if (failure === 'property') vi.spyOn(window, 'localStorage', 'get').mockImplementation(denied)
      else vi.spyOn(Storage.prototype, failure === 'read' ? 'getItem' : 'setItem').mockImplementation(denied)
      providers(
        <AppShell>
          <p>Usable after login</p>
        </AppShell>,
      )
      expect(screen.getByRole('navigation', { name: 'Main navigation' })).toBeInTheDocument()
      expect(screen.getByText('Usable after login')).toBeInTheDocument()
      await userEvent.click(screen.getByRole('button', { name: 'Settings and theme' }))
      await userEvent.click(screen.getByRole('menuitemradio', { name: 'Dark' }))
      expect(document.documentElement).toHaveClass('dark')
    },
  )
})

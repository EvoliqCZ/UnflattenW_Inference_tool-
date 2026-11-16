import { setLanguage } from '@/i18n'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it } from 'vitest'
import { AppShell } from './AppShell'

beforeEach(() => {
  setLanguage('sk')
})

describe('AppShell', () => {
  it('should render the product nav, children and skip link without a diagnostic banner', () => {
    // T29: `AppShell` renders `TopNav` -> `UserMenu`, which now uses
    // `useNavigate`/`useLogoutMutation` for "Odhlásiť".
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <AppShell>
            <p>page content</p>
          </AppShell>
        </MemoryRouter>
      </QueryClientProvider>,
    )

    expect(screen.queryByRole('note')).not.toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Hlavná navigácia' })).toBeInTheDocument()
    expect(screen.getByText('page content')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Preskočiť na obsah' })).toHaveAttribute('href', '#main-content')
  })
})

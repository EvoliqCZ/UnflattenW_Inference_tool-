import { setLanguage } from '@/i18n'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it } from 'vitest'
import { TopNav } from './TopNav'

// T29: `UserMenu`'s "Odhlásiť" now calls `useNavigate`/`useLogoutMutation`,
// so any test rendering it (`TopNav` includes it) needs both providers.
function renderTopNav() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <TopNav />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  setLanguage('sk')
})

describe('TopNav', () => {
  it('should render the primary navigation links', () => {
    renderTopNav()

    const nav = screen.getByRole('navigation', { name: 'Hlavná navigácia' })
    expect(nav).toHaveTextContent('Nová konverzia')
    expect(nav).toHaveTextContent('Úlohy')
    expect(nav).toHaveTextContent('Modely')
  })

  it('should render the Unflatten W mark as the image named by the brand', () => {
    renderTopNav()

    const header = screen.getByRole('banner')
    const mark = within(header).getByRole('img', { name: 'Unflatten W' })
    expect(mark).toHaveAttribute('src', expect.stringContaining('unflatten-w-mark'))
    expect(header).toHaveTextContent('Unflatten W')
  })

  it('should open the mobile navigation sheet with the same links', async () => {
    const user = userEvent.setup()
    renderTopNav()

    await user.click(screen.getByRole('button', { name: 'Otvoriť navigáciu' }))

    expect(await screen.findByRole('dialog')).toHaveTextContent('Nová konverzia')
  })
})

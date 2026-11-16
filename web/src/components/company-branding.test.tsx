import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router'
import { beforeEach, expect, it } from 'vitest'
import { setLanguage } from '@/i18n'
import { AuthShell } from '@/shell/AuthShell'
import { AppShell } from '@/shell/AppShell'
import { AboutService } from '@/features/settings/AboutService'

beforeEach(() => {
  setLanguage('en')
})

function renderApp(children: React.ReactNode) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>,
  )
}

it('shows the original EVOLIQ logo and safe website link even before login', async () => {
  render(
    <AuthShell>
      <p>Login</p>
    </AuthShell>,
  )
  const footer = within(screen.getByRole('contentinfo'))
  const link = footer.getByRole('link', { name: 'EVOLIQ s.r.o.' })
  expect(within(link).getAllByRole('img', { name: 'EVOLIQ' })).toHaveLength(2)
  expect(link).toHaveAttribute('href', 'https://www.evoliq.cz/en')
  expect(link).toHaveAttribute('target', '_blank')
  expect(link).toHaveAttribute('rel', 'noopener noreferrer')
  await userEvent.click(screen.getByRole('button', { name: 'Slovenčina' }))
  expect(link).toHaveAttribute('href', 'https://www.evoliq.cz/sk')
})

it('brands the authenticated header as well as the footer', () => {
  renderApp(
    <AppShell>
      <p>Conversion</p>
    </AppShell>,
  )
  const header = within(screen.getByRole('banner'))
  expect(
    within(header.getByRole('link', { name: 'EVOLIQ s.r.o.' })).getAllByRole('img', { name: 'EVOLIQ' }),
  ).toHaveLength(2)
  expect(within(screen.getByRole('contentinfo')).getByRole('link', { name: 'EVOLIQ s.r.o.' })).toBeInTheDocument()
})

it('includes the company identity and logo in About', async () => {
  renderApp(<AboutService />)
  await screen.findByText('57db768')
  const link = screen.getByRole('link', { name: 'EVOLIQ s.r.o.' })
  expect(within(link).getAllByRole('img', { name: 'EVOLIQ' })).toHaveLength(2)
  expect(link).toHaveAttribute('href', 'https://www.evoliq.cz/en')
})

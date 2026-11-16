import { setLanguage } from '@/i18n'
import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import { AuthShell } from './AuthShell'

beforeEach(() => {
  setLanguage('sk')
})

describe('AuthShell', () => {
  it('should render children without a diagnostic banner or authenticated nav', () => {
    render(
      <AuthShell>
        <p>login form</p>
      </AuthShell>,
    )

    expect(screen.queryByRole('note')).not.toBeInTheDocument()
    expect(screen.getByText('login form')).toBeInTheDocument()
    expect(screen.queryByRole('navigation')).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'EVOLIQ s.r.o.' })).toHaveAttribute('href', 'https://www.evoliq.cz/sk')
  })

  it('should show the Unflatten W mark above the login and setup content', () => {
    render(
      <AuthShell>
        <p>login form</p>
      </AuthShell>,
    )

    const mark = screen.getByRole('img', { name: 'Unflatten W' })
    expect(mark).toHaveAttribute('src', expect.stringContaining('unflatten-w-mark'))
    expect(screen.getByRole('main')).toContainElement(mark)
  })
})

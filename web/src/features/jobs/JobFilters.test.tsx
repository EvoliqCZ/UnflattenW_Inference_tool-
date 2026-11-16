import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { JobFilters } from './JobFilters'

describe('JobFilters (UX §6)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('reports the selected tab when a different one is clicked', async () => {
    const onFilterChange = vi.fn()
    render(<JobFilters filter="all" onFilterChange={onFilterChange} search="" onSearchChange={vi.fn()} />)

    await userEvent.click(screen.getByRole('tab', { name: 'Aktívne' }))

    expect(onFilterChange).toHaveBeenCalledWith('active')
  })

  it('reports search text as it is typed', async () => {
    const onSearchChange = vi.fn()
    render(<JobFilters filter="all" onFilterChange={vi.fn()} search="" onSearchChange={onSearchChange} />)

    await userEvent.type(screen.getByRole('searchbox', { name: 'Hľadať podľa označenia' }), 'x')

    expect(onSearchChange).toHaveBeenCalledWith('x')
  })
})

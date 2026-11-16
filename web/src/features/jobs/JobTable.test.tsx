import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { makeJobSummary } from '@/test/msw/fixtures'
import { JobTable } from './JobTable'

function nth(elements: readonly HTMLElement[], index: number): HTMLElement {
  const element = elements[index]
  if (element === undefined) {
    throw new Error(`expected an element at index ${String(index)}`)
  }
  return element
}

function renderTable(jobs: Parameters<typeof JobTable>[0]['jobs'], onCancel = vi.fn(), onPurge = vi.fn()) {
  return render(
    <MemoryRouter>
      <JobTable jobs={jobs} language="sk" onCancel={onCancel} onPurge={onPurge} />
    </MemoryRouter>,
  )
}

describe('JobTable (UX §6/§11.4)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('links the job label to its detail page', () => {
    renderTable([makeJobSummary({ job_id: 'job-42', label: 'Zmluva.pdf' })])

    expect(screen.getByRole('link', { name: 'Zmluva.pdf' })).toHaveAttribute('href', '/jobs/job-42')
  })

  it('shows done/total pages for an active job and just total for a terminal one', () => {
    renderTable([
      makeJobSummary({ job_id: 'a', state: 'running', pages_done: 1, pages_total: 3 }),
      makeJobSummary({ job_id: 'b', state: 'succeeded', pages_done: 3, pages_total: 3 }),
    ])

    expect(screen.getByText('1 / 3')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('offers cancel for an active job and purge for a terminal one', async () => {
    const onCancel = vi.fn()
    const onPurge = vi.fn()
    renderTable(
      [
        makeJobSummary({ job_id: 'active-job', state: 'running' }),
        makeJobSummary({ job_id: 'done-job', state: 'succeeded' }),
      ],
      onCancel,
      onPurge,
    )

    const menus = screen.getAllByRole('button', { name: 'Akcie' })
    await userEvent.click(nth(menus, 0))
    await userEvent.click(screen.getByRole('menuitem', { name: 'Zrušiť' }))
    expect(onCancel).toHaveBeenCalledWith(expect.objectContaining({ job_id: 'active-job' }))

    await userEvent.click(nth(menus, 1))
    await userEvent.click(screen.getByRole('menuitem', { name: 'Odstrániť natrvalo' }))
    expect(onPurge).toHaveBeenCalledWith(expect.objectContaining({ job_id: 'done-job' }))
  })
})

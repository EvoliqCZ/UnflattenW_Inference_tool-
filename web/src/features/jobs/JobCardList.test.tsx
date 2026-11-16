import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { makeJobSummary } from '@/test/msw/fixtures'
import { JobCardList } from './JobCardList'

describe('JobCardList (UX §6, < 768px)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('renders a card per job with label, state and a mono meta line', () => {
    render(
      <MemoryRouter>
        <JobCardList
          jobs={[makeJobSummary({ job_id: 'job-9', label: 'scan-staré.pdf', model_variant: 'INT8·KV FP8' })]}
          language="sk"
          onCancel={vi.fn()}
          onPurge={vi.fn()}
        />
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: 'scan-staré.pdf' })).toHaveAttribute('href', '/jobs/job-9')
    expect(screen.getByText('Dokončená')).toBeInTheDocument()
    expect(screen.getByText(/INT8·KV FP8/)).toBeInTheDocument()
  })
})

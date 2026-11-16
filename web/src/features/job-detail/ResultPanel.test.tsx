import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { succeededJob } from '@/test/msw/fixtures'
import type { components } from '@/api/schema'
import { ResultPanel } from './ResultPanel'

type DiagnosticOutcome = components['schemas']['DiagnosticOutcome']

function outcome(overrides: Partial<DiagnosticOutcome> = {}): DiagnosticOutcome {
  return {
    canonical_ast_produced: true,
    claim_scope: 'diagnostic',
    docx_chain: null,
    docx_produced: true,
    prediction_ir_status: 'sealed',
    reason_codes: [],
    ...overrides,
  }
}

describe('ResultPanel (UX §3.6)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows the H5 font substitution notice when the outcome carries it', () => {
    render(
      <ResultPanel
        job={{ ...succeededJob, diagnostic_outcome: outcome({ reason_codes: ['DOCX_FONT_SUBSTITUTED'] }) }}
      />,
    )
    expect(screen.getByText(/FONT_FALLBACK/)).toBeInTheDocument()
  })

  it('does not show the font notice when nothing was substituted', () => {
    render(<ResultPanel job={{ ...succeededJob, diagnostic_outcome: outcome() }} />)
    expect(screen.queryByText(/FONT_FALLBACK/)).not.toBeInTheDocument()
  })

  it('shows "Word nevznikol" with translated reasons when docx_produced is false', () => {
    render(
      <ResultPanel
        job={{
          ...succeededJob,
          diagnostic_outcome: outcome({ docx_produced: false, reason_codes: ['DOCX_AST_NOT_STRICT'] }),
        }}
      />,
    )
    expect(screen.getByText('Word dokument sa nepodarilo vytvoriť.')).toBeInTheDocument()
    expect(screen.getByText('AST nespĺňa prísny profil')).toBeInTheDocument()
  })

  it('renders the ZIP link built from the job id', () => {
    render(<ResultPanel job={{ ...succeededJob, diagnostic_outcome: outcome() }} />)
    expect(screen.getByRole('link', { name: 'Stiahnuť všetko (ZIP)' })).toHaveAttribute(
      'href',
      `/v1/conversions/${succeededJob.job_id}/artifacts.zip`,
    )
  })

  it('shows the expiry countdown from expires_at', () => {
    // Pin the clock before the fixture's expires_at (2026-09-28T10:03Z);
    // on the wall clock this test started failing once that date passed.
    vi.useFakeTimers({ toFake: ['Date'] })
    vi.setSystemTime(new Date('2026-09-27T12:00:00Z'))
    try {
      render(<ResultPanel job={{ ...succeededJob, diagnostic_outcome: outcome() }} />)
      expect(screen.getByText(/Vyprší o/)).toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })
})

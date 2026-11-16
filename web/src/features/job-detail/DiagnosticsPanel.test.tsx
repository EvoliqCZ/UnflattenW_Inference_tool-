import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import type { components } from '@/api/schema'
import { DiagnosticsPanel } from './DiagnosticsPanel'

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

describe('DiagnosticsPanel (UX §6, H2)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows the H2 photo quality note for an image input', () => {
    render(<DiagnosticsPanel outcome={outcome()} inputKind="image" />)
    expect(screen.getByRole('note')).toBeInTheDocument()
  })

  it('does not show the photo quality note for a pdf input', () => {
    render(<DiagnosticsPanel outcome={outcome()} inputKind="pdf" />)
    expect(screen.queryByRole('note')).not.toBeInTheDocument()
  })

  it('shows an unavailable message when there is no outcome yet', () => {
    render(<DiagnosticsPanel outcome={null} inputKind="pdf" />)
    expect(screen.getByText('Diagnostika zatiaľ nie je k dispozícii.')).toBeInTheDocument()
  })

  it('renders canonical_ast_produced and docx_produced as Áno/Nie', () => {
    render(
      <DiagnosticsPanel outcome={outcome({ canonical_ast_produced: true, docx_produced: false })} inputKind="pdf" />,
    )
    expect(screen.getAllByText('Áno')).toHaveLength(1)
    expect(screen.getAllByText('Nie')).toHaveLength(1)
  })

  it('shows the headless-defaults explanation only when that reason code is present', () => {
    render(<DiagnosticsPanel outcome={outcome({ reason_codes: ['HEADLESS_DEFAULTS_V0_APPLIED'] })} inputKind="pdf" />)
    expect(screen.getByText('Použité predvolené hodnoty bez obsluhy')).toBeInTheDocument()
  })

  it('shows "bez úprav" for a policy with empty counts', () => {
    const { container } = render(
      <DiagnosticsPanel
        outcome={outcome({
          docx_chain: {
            page_outcome: 'ok',
            policies: [{ policy_id: 'headless-defaults.v0', policy_status: 'RATIFIED', counts: {} }],
            policy_id: 'headless-defaults.v0',
            policy_status: 'RATIFIED',
          },
        })}
        inputKind="pdf"
      />,
    )
    expect(container.querySelector('li')?.textContent).toContain('Táto politika nemení obsah')
  })

  it('expands the raw JSON collapsible and copies it', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.assign(navigator, { clipboard: { writeText } })
    render(<DiagnosticsPanel outcome={outcome()} inputKind="pdf" />)
    await userEvent.click(screen.getByText('Surové dáta (JSON)'))
    await userEvent.click(screen.getByRole('button', { name: 'Kopírovať' }))
    expect(writeText).toHaveBeenCalled()
  })
})

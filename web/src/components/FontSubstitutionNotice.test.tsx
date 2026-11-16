import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import type { components } from '@/api/schema'
import { FontSubstitutionNotice } from './FontSubstitutionNotice'

type DiagnosticOutcome = components['schemas']['DiagnosticOutcome']

function buildOutcome(overrides: Partial<DiagnosticOutcome> = {}): DiagnosticOutcome {
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

describe('FontSubstitutionNotice', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should render nothing when the H5 condition is not met', () => {
    const { container } = render(<FontSubstitutionNotice outcome={buildOutcome()} />)

    expect(container).toBeEmptyDOMElement()
  })

  it('should render the notice (with icon) when reason_codes carries DOCX_FONT_SUBSTITUTED', () => {
    render(<FontSubstitutionNotice outcome={buildOutcome({ reason_codes: ['DOCX_FONT_SUBSTITUTED'] })} />)

    expect(screen.getByText(/FONT_FALLBACK/)).toBeInTheDocument()
    expect(document.querySelector('svg[aria-hidden="true"]')).toBeInTheDocument()
  })

  it('should render the notice when renderer_approximations carries FONT_FALLBACK', () => {
    render(
      <FontSubstitutionNotice
        outcome={buildOutcome({
          docx_chain: {
            page_outcome: 'ok',
            policy_id: 'headless-defaults.v0',
            policy_status: 'RATIFIED',
            renderer_approximations: ['FONT_FALLBACK'],
          },
        })}
      />,
    )

    expect(screen.getByText(/FONT_FALLBACK/)).toBeInTheDocument()
  })

  it('should never render a close button', () => {
    render(<FontSubstitutionNotice outcome={buildOutcome({ reason_codes: ['DOCX_FONT_SUBSTITUTED'] })} />)

    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })
})

import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { JOB_STAGES } from '@/i18n/codes'
import { StageStepper } from './StageStepper'

describe('StageStepper (UX §3.4)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it.each(JOB_STAGES)('renders every JOB_STAGES code "%s" without crashing, with exactly one active step', (stage) => {
    const { unmount } = render(<StageStepper stage={stage} state="running" requestedOutput="diagnostic_docx" />)
    expect(screen.getAllByText('Prebieha')).toHaveLength(1)
    expect(screen.getByText(stage)).toBeInTheDocument()
    unmount()
  })

  it('labels the assembly step "Zapečatenie IR" when the requested output is prediction_ir', () => {
    render(<StageStepper stage="ast_promotion" state="running" requestedOutput="prediction_ir" />)
    expect(screen.getByText('Zapečatenie IR')).toBeInTheDocument()
  })

  it('labels the assembly step "Zostavenie" for diagnostic_docx output', () => {
    render(<StageStepper stage="ast_promotion" state="running" requestedOutput="diagnostic_docx" />)
    expect(screen.getByText('Zostavenie')).toBeInTheDocument()
  })

  it('marks every step done when the job succeeded', () => {
    render(<StageStepper stage="admission_complete" state="succeeded" requestedOutput="diagnostic_docx" />)
    expect(screen.getAllByText('Hotovo')).toHaveLength(5)
  })

  it('marks the current step failed when the job failed', () => {
    render(<StageStepper stage="renderer_gate" state="failed" requestedOutput="diagnostic_docx" />)
    expect(screen.getByText('Zlyhalo')).toBeInTheDocument()
  })
})

import { describe, expect, it } from 'vitest'
import { JOB_STAGES } from '@/i18n/codes'
import { buildSteps, STAGE_STEP, STEP_IDS } from './stages'

describe('STAGE_STEP completeness (StageStepper)', () => {
  it.each(JOB_STAGES)('maps stage "%s" to one of the 5 visible steps', (stage) => {
    expect(STEP_IDS).toContain(STAGE_STEP[stage])
  })

  it('covers every JOB_STAGES code, no more and no less', () => {
    expect(Object.keys(STAGE_STEP).sort()).toEqual([...JOB_STAGES].sort())
  })
})

describe('buildSteps', () => {
  it('marks every step done when state is succeeded, regardless of stage', () => {
    const steps = buildSteps('pdf_inspection', 'succeeded')
    expect(steps.every((step) => step.status === 'done')).toBe(true)
  })

  it('marks steps before the current stage done, the current one "current", and later ones upcoming', () => {
    const steps = buildSteps('model_decode', 'running')
    expect(steps.map((s) => s.status)).toEqual(['done', 'done', 'done', 'current', 'upcoming'])
  })

  it('marks the current step "failed" when the job failed', () => {
    const steps = buildSteps('renderer_gate', 'failed')
    expect(steps.find((s) => s.id === 'assembly')?.status).toBe('failed')
  })

  it('marks the current step "cancelled" when the job was cancelled', () => {
    const steps = buildSteps('loading_model', 'cancelled')
    expect(steps.find((s) => s.id === 'model')?.status).toBe('cancelled')
  })
})

import type { JobStage } from '@/i18n/codes'

/** UX §3.4 stage table: the server's 12 `stage` codes group into 5 visible steps. */
export const STEP_IDS = ['receipt', 'model', 'input', 'recognition', 'assembly'] as const
export type StepId = (typeof STEP_IDS)[number]

/**
 * `stage` -> visible step. A `Record<JobStage, StepId>` makes TypeScript
 * reject a build that drops a `JobStage` from this map; `stages.test.ts`
 * additionally checks it at runtime so a future widening of `JobStage`
 * cannot silently produce an "Iné" (unknown) step.
 */
export const STAGE_STEP: Record<JobStage, StepId> = {
  admission_complete: 'receipt',
  loading_model: 'model',
  pdf_inspection: 'input',
  pdf_rasterization: 'input',
  png_validation: 'input',
  model_input_projection: 'recognition',
  feature_extraction: 'recognition',
  model_decode: 'recognition',
  prediction_ir_sealing: 'assembly',
  ast_promotion: 'assembly',
  renderer_gate: 'assembly',
  artifact_finalization: 'assembly',
}

export type StepStatus = 'done' | 'current' | 'upcoming' | 'failed' | 'cancelled'

export interface StepDescriptor {
  id: StepId
  status: StepStatus
}

/**
 * UX §3.4: steps before the current stage's step are done, the current
 * step's status depends on `state` (running/queued -> current, failed ->
 * failed, cancelled -> cancelled), everything after is upcoming.
 * `state: 'succeeded'` marks every step done regardless of `stage`.
 */
export function buildSteps(
  stage: JobStage,
  state: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'expired',
): StepDescriptor[] {
  if (state === 'succeeded') {
    return STEP_IDS.map((id) => ({ id, status: 'done' }))
  }
  const currentIndex = STEP_IDS.indexOf(STAGE_STEP[stage])
  return STEP_IDS.map((id, index) => {
    if (index < currentIndex) return { id, status: 'done' }
    if (index > currentIndex) return { id, status: 'upcoming' }
    if (state === 'failed') return { id, status: 'failed' }
    if (state === 'cancelled' || state === 'expired') return { id, status: 'cancelled' }
    return { id, status: 'current' }
  })
}

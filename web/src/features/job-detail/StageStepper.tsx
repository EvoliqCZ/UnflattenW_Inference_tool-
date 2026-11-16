import { Ban, CheckCircle2, Circle, Loader2, XCircle } from 'lucide-react'
import type { ComponentType } from 'react'
import { useTranslation } from 'react-i18next'
import type { JobStage, JobState } from '@/i18n/codes'
import { cn } from '@/lib/utils'
import { buildSteps, type StepStatus } from './stages'

export interface StageStepperProps {
  stage: JobStage
  state: Exclude<JobState, 'succeeded'> | 'succeeded'
  /** UX §3.4: the assembly step is labelled "Zapečatenie IR" for `prediction_ir` output. */
  requestedOutput: 'diagnostic_docx' | 'prediction_ir'
}

const STATUS_ICON: Record<StepStatus, ComponentType<{ className?: string }>> = {
  done: CheckCircle2,
  current: Loader2,
  upcoming: Circle,
  failed: XCircle,
  cancelled: Ban,
}

/**
 * UX §3.4: server sends 12 `stage` codes, grouped here into 5 visible
 * steps (`stages.ts`). The exact `stage` code is always shown in mono under
 * the active step -- "a measuring instrument, nothing hidden" -- rather
 * than translated prose, since the field itself carries no i18n contract.
 */
export function StageStepper({ stage, state, requestedOutput }: StageStepperProps) {
  const { t } = useTranslation()
  const steps = buildSteps(stage, state)
  const activeIndex = steps.findIndex((step) => step.status !== 'done' && step.status !== 'upcoming')

  return (
    <ol className="flex flex-wrap items-center gap-x-2 gap-y-3" aria-label={t('job.progress.summary')}>
      {steps.map((step, index) => {
        const Icon = STATUS_ICON[step.status]
        const label =
          step.id === 'assembly' && requestedOutput === 'prediction_ir'
            ? t('job.progress.step.assemblyIr')
            : t(`job.progress.step.${step.id}`)
        return (
          <li key={step.id} className="flex items-center gap-2">
            {index > 0 ? <span aria-hidden="true" className="h-px w-4 bg-border-strong" /> : null}
            <div
              className="flex flex-col items-center gap-1 text-xs"
              aria-current={index === activeIndex ? 'step' : undefined}
            >
              <span
                className={cn(
                  'flex items-center gap-1 font-medium',
                  step.status === 'done' && 'text-success-text',
                  step.status === 'current' && 'text-accent',
                  step.status === 'upcoming' && 'text-muted-foreground',
                  step.status === 'failed' && 'text-destructive',
                  step.status === 'cancelled' && 'text-muted-foreground line-through',
                )}
              >
                <Icon
                  aria-hidden="true"
                  className={cn('size-4', step.status === 'current' && 'motion-safe:animate-spin')}
                />
                {label}
                <span className="sr-only">{t(`job.progress.step.state.${step.status}`)}</span>
              </span>
              {index === activeIndex ? <code className="text-[10px] text-muted-foreground">{stage}</code> : null}
            </div>
          </li>
        )
      })}
    </ol>
  )
}

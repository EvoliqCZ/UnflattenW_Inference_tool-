import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'

/**
 * UX §3.1: the stepper is informational, not clickable forward navigation.
 * Four steps (T45 added "model" between "system" and "ready" -- see
 * `ModelStep`); the former "token" step is gone (owner 2026-09-29, DEC-052
 * addendum: tokenless setup on loopback, otherwise the token comes from the
 * setup link's URL fragment).
 */
export const SETUP_STEPS = ['password', 'system', 'model', 'ready'] as const
export type SetupStep = (typeof SETUP_STEPS)[number]

export function SetupStepper({ current }: { current: SetupStep }) {
  const { t } = useTranslation()
  const currentIndex = SETUP_STEPS.indexOf(current)

  return (
    <ol aria-label={t('setup.title')} className="flex flex-wrap gap-x-2 gap-y-1 text-sm">
      {SETUP_STEPS.map((step, index) => (
        <li
          key={step}
          aria-current={step === current ? 'step' : undefined}
          className={cn('text-muted-foreground', index <= currentIndex && 'font-medium text-foreground')}
        >
          {t(`setup.step.${step}.title`)}
          {index < SETUP_STEPS.length - 1 ? <span aria-hidden="true"> &middot; </span> : null}
        </li>
      ))}
    </ol>
  )
}

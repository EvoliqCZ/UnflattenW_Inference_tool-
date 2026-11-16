import { Ban, CheckCircle2, Clock, Hourglass, Loader2, XCircle } from 'lucide-react'
import type { ComponentType } from 'react'
import { useTranslation } from 'react-i18next'
import { Badge, type BadgeProps } from '@/components/ui/badge'
import type { JobState } from '@/i18n/codes'
import { cn } from '@/lib/utils'

/**
 * UX §3.4 state table: each `JobState` gets a badge variant, an icon and a
 * translated label. Color is never the only carrier of meaning (WCAG
 * 1.4.1) -- the icon and text are always rendered alongside it.
 */
const STATE_CONFIG: Record<
  JobState,
  { variant: NonNullable<BadgeProps['variant']>; icon: ComponentType<{ className?: string }>; spin?: boolean }
> = {
  queued: { variant: 'secondary', icon: Clock },
  running: { variant: 'default', icon: Loader2, spin: true },
  succeeded: { variant: 'success', icon: CheckCircle2 },
  failed: { variant: 'destructive', icon: XCircle },
  cancelled: { variant: 'outline', icon: Ban },
  expired: { variant: 'secondary', icon: Hourglass },
}

export interface JobStateBadgeProps {
  state: JobState
}

export function JobStateBadge({ state }: JobStateBadgeProps) {
  const { t } = useTranslation()
  const { variant, icon: Icon, spin } = STATE_CONFIG[state]

  return (
    <Badge variant={variant}>
      {/* `motion-safe:` keeps the spin off for `prefers-reduced-motion` (UX §3.4 `running`). */}
      <Icon aria-hidden="true" className={cn('size-3.5', spin && 'motion-safe:animate-spin')} />
      <span>{t(`codes.state.${state}`)}</span>
    </Badge>
  )
}

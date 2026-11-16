import { AlertTriangle, Clock } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { formatDateTime, formatRelativeTime } from '@/lib/time'
import { cn } from '@/lib/utils'
import type { SupportedLanguage } from '@/i18n'

const ONE_HOUR_MS = 60 * 60 * 1000
const NORMAL_TICK_MS = 60 * 1000
const UNDER_HOUR_TICK_MS = 10 * 1000

export interface ExpiryCountdownProps {
  /** ISO timestamp, or `null` for an active job that has no expiry yet (UX §3.4/§3.6.4). */
  expiresAt: string | null
  compact?: boolean
}

/**
 * UX §3.6.4: relative countdown to `expires_at`, switching to a warning
 * form under one hour (icon + color), and to `expired` once past. The
 * remaining-time text is `aria-live="off"` (re-announcing it every tick
 * would be noise); a separate, silent-until-crossing region announces the
 * under-1-hour transition exactly once.
 */
export function ExpiryCountdown({ expiresAt, compact = false }: ExpiryCountdownProps) {
  const { t, i18n } = useTranslation()
  const language = i18n.language as SupportedLanguage
  const [now, setNow] = useState(() => new Date())
  const announcedUnderHour = useRef(false)

  const target = useMemo(() => (expiresAt === null ? null : new Date(expiresAt)), [expiresAt])
  const remainingMs = target === null ? null : target.getTime() - now.getTime()
  const isUnderHour = remainingMs !== null && remainingMs > 0 && remainingMs < ONE_HOUR_MS
  const isExpired = remainingMs !== null && remainingMs <= 0

  useEffect(() => {
    if (target === null) {
      return
    }
    const tickMs = isUnderHour ? UNDER_HOUR_TICK_MS : NORMAL_TICK_MS
    const id = window.setInterval(() => {
      setNow(new Date())
    }, tickMs)
    return () => {
      window.clearInterval(id)
    }
  }, [target, isUnderHour])

  const [announcement, setAnnouncement] = useState('')
  useEffect(() => {
    if (isUnderHour && !announcedUnderHour.current) {
      announcedUnderHour.current = true
      setAnnouncement(t('job.expiry.soon'))
    }
  }, [isUnderHour, t])

  if (target === null) {
    return <span className={cn('text-muted-foreground', compact && 'text-xs')}>—</span>
  }

  const label = isExpired
    ? t('job.expiry.expired')
    : t('job.expiry.in', { time: formatRelativeTime(target, now, language) })
  const Icon = isUnderHour ? AlertTriangle : Clock

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1',
        compact && 'text-xs',
        isUnderHour && 'text-warning-text',
        isExpired && 'text-muted-foreground line-through',
      )}
      title={formatDateTime(target, language)}
    >
      <Icon aria-hidden="true" className="size-3.5 shrink-0" />
      <span aria-live="off">{label}</span>
      <span className="sr-only" aria-live="polite">
        {announcement}
      </span>
    </span>
  )
}

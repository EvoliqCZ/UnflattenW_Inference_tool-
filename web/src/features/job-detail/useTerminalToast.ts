import { useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'
import { toast } from '@/hooks/use-toast'
import { jobDisplayName } from '@/features/jobs/jobRowHelpers'

type ConversionStatus = components['schemas']['ConversionStatus']

const TERMINAL_STATES: readonly ConversionStatus['state'][] = ['succeeded', 'failed', 'cancelled', 'expired']

/**
 * UX §3.4: "a terminal state additionally fires a short Toast, even when the
 * visitor is on another screen" -- fires once, the moment a job (already
 * seen non-terminal) crosses into a terminal `state`, not on the first
 * render of an already-finished job.
 */
export function useTerminalToast(job: ConversionStatus | undefined) {
  const { t } = useTranslation()
  const sawActive = useRef(false)

  useEffect(() => {
    if (job === undefined) return
    const isTerminal = TERMINAL_STATES.includes(job.state)
    if (!isTerminal) {
      sawActive.current = true
      return
    }
    if (sawActive.current) {
      toast({ title: t('job.terminalToast', { label: jobDisplayName(job), state: t(`codes.state.${job.state}`) }) })
    }
    sawActive.current = false
  }, [job, t])
}

import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'

type ConversionStatus = components['schemas']['ConversionStatus']

/**
 * UX §3.4: the `aria-live="polite"` region announces only a step change, a
 * page finishing, or the terminal state -- never every poll tick. Tracked by
 * comparing `stage`/`pages_done`/`state` against the previous render.
 */
export function useProgressAnnouncement(job: ConversionStatus | undefined): string {
  const { t } = useTranslation()
  const [announcement, setAnnouncement] = useState('')
  const previous = useRef<{ stage: string; pagesDone: number | null; state: string } | null>(null)

  useEffect(() => {
    if (job === undefined) return
    const current = { stage: job.stage, pagesDone: job.pages_done, state: job.state }
    const last = previous.current
    previous.current = current

    if (last === null) return

    if (current.state !== last.state && current.state !== 'queued' && current.state !== 'running') {
      setAnnouncement(t(`codes.state.${current.state}`))
      return
    }
    if (current.stage !== last.stage) {
      setAnnouncement(t(`codes.stage.${current.stage}`))
      return
    }
    if (current.pagesDone !== null && current.pagesDone !== last.pagesDone) {
      setAnnouncement(`${t('job.artifacts.page', { n: current.pagesDone })} ${t('job.progress.page.done')}`)
    }
  }, [job, t])

  return announcement
}

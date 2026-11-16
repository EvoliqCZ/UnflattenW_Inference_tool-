import { Info } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'

export interface PhotoQualityNoteProps {
  variant: 'inline' | 'compact'
}

/**
 * UX H2: shown whenever the upload contains at least one image, and again
 * in the job-detail Diagnostics section for an image input. Fixed i18n
 * text, never dismissible -- no close affordance is ever rendered.
 */
export function PhotoQualityNote({ variant }: PhotoQualityNoteProps) {
  const { t } = useTranslation()

  return (
    <p
      role="note"
      className={cn('flex items-start gap-2 text-muted-foreground', variant === 'compact' ? 'text-xs' : 'text-sm')}
    >
      <Info aria-hidden="true" className="size-4 shrink-0 translate-y-0.5" />
      <span>{t('domainComponents.photoQualityNote.text')}</span>
    </p>
  )
}

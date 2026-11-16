import { AlertTriangle } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'
import { Alert, AlertDescription } from '@/components/ui/alert'

type DiagnosticOutcome = components['schemas']['DiagnosticOutcome']

export interface FontSubstitutionNoticeProps {
  outcome: DiagnosticOutcome
}

/**
 * UX H5 "font substitution is never hidden": renders only when the outcome
 * actually carries a font substitution -- `DOCX_FONT_SUBSTITUTED` in
 * `reason_codes`, or `FONT_FALLBACK` in the renderer's approximations --
 * otherwise `null`. Never closable: no `onClose`/dismiss prop exists at all.
 */
export function FontSubstitutionNotice({ outcome }: FontSubstitutionNoticeProps) {
  const { t } = useTranslation()

  const fontSubstituted =
    outcome.reason_codes.includes('DOCX_FONT_SUBSTITUTED') ||
    (outcome.docx_chain?.renderer_approximations?.includes('FONT_FALLBACK') ?? false)

  if (!fontSubstituted) {
    return null
  }

  return (
    <Alert variant="warning">
      <AlertTriangle aria-hidden="true" className="size-4" />
      <AlertDescription>{t('domainComponents.fontSubstitutionNotice.text')}</AlertDescription>
    </Alert>
  )
}

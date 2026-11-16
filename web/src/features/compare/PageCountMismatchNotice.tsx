import { Info } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Alert, AlertDescription } from '@/components/ui/alert'

export interface PageCountMismatchNoticeProps {
  sourceCount: number
  docxCount: number
}

/** UX §5.5: renders only when the original and Word page counts differ. */
export function PageCountMismatchNotice({ sourceCount, docxCount }: PageCountMismatchNoticeProps) {
  const { t } = useTranslation()

  if (sourceCount === docxCount) {
    return null
  }

  return (
    <Alert>
      <Info aria-hidden="true" className="size-4" />
      <AlertDescription>{t('compare.mismatchNotice', { sourceCount, docxCount })}</AlertDescription>
    </Alert>
  )
}

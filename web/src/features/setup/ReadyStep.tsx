import { useTranslation } from 'react-i18next'
import { Link } from 'react-router'
import { Button } from '@/components/ui/button'

/**
 * UX §3.1 step 5 "ready": the sample document is referenced by URL/constant
 * only -- bundling the actual file is a devops task (owner decision D-B,
 * plan §I). `ConvertPage` (T31) does not yet read `?sample=`; wiring the
 * pre-fill is that task's job once the dropzone exists.
 */
export const SAMPLE_PDF_URL = '/assets/sample.pdf'

export function ReadyStep() {
  const { t } = useTranslation()

  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-base font-semibold">{t('setup.step.ready.title')}</h2>
      <p className="text-sm text-muted-foreground">{t('setup.step.ready.body')}</p>
      <Button asChild>
        <Link to={{ pathname: '/', search: `?sample=${encodeURIComponent(SAMPLE_PDF_URL)}` }}>
          {t('setup.step.ready.tryDemo')}
        </Link>
      </Button>
      <Button asChild variant="outline">
        <Link to="/">{t('setup.step.ready.uploadOwn')}</Link>
      </Button>
    </div>
  )
}

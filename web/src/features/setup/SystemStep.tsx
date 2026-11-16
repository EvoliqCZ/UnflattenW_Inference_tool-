import { useTranslation } from 'react-i18next'
import { useModelsQuery, useReadinessQuery } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { DevicePanel } from '@/features/models/DevicePanel'
import type { SupportedLanguage } from '@/i18n'

export interface SystemStepProps {
  language: SupportedLanguage
  onNext: () => void
}

/**
 * UX §3.1 step 3 "system". The per-component OK/fail breakdown from the
 * readiness contract's full component list is still out of scope (`useReadinessQuery`
 * only exposes `service_ready` today, `queries.ts`); the GPU part (T45) reuses
 * `DevicePanel` over `/v1/models` 1.1's `device` field, which is `null` while
 * the worker has not published it yet. "Pokracovat aj tak" stays enabled
 * either way (UX: uploads get blocked later, with a reason, not here).
 */
export function SystemStep({ language, onNext }: SystemStepProps) {
  const { t } = useTranslation()
  const readinessQuery = useReadinessQuery()
  const modelsQuery = useModelsQuery()

  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-base font-semibold">{t('setup.step.system.title')}</h2>
      <p className="text-sm text-muted-foreground">{t('setup.step.system.body')}</p>
      <p aria-live="polite">
        {readinessQuery.isLoading
          ? t('setup.step.system.checking')
          : readinessQuery.data?.service_ready === true
            ? t('setup.step.system.ready')
            : t('setup.step.system.notReady')}
      </p>
      <DevicePanel device={modelsQuery.data?.device ?? null} now={new Date()} language={language} />
      <Button type="button" onClick={onNext}>
        {readinessQuery.data?.service_ready === true ? t('setup.continue') : t('setup.continueAnyway')}
      </Button>
    </div>
  )
}

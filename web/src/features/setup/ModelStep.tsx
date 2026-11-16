import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useModelsQuery } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { useDownloadModelMutation, useLoadModelMutation } from '@/features/models/api'
import { isPublished } from '@/features/models/availability'
import { DownloadProgress } from '@/features/models/DownloadProgress'
import type { SupportedLanguage } from '@/i18n'
import { ModelStepVariantList } from './ModelStepVariantList'

export interface ModelStepProps {
  language: SupportedLanguage
  onNext: () => void
}

/**
 * UX §3.1 step 4 / §11.2 "model": the worker already auto-downloads a
 * variant on first start (owner §12/2), so this step is a progress display,
 * not a picker the user must complete -- "Dokončiť neskôr" always advances,
 * mirroring the ASCII wireframe's disabled-until-ready "Pokračovať" plus a
 * second, always-enabled button.
 */
export function ModelStep({ language, onNext }: ModelStepProps) {
  const { t } = useTranslation()
  const modelsQuery = useModelsQuery()
  const downloadMutation = useDownloadModelMutation()
  const loadMutation = useLoadModelMutation()
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // models_response 1.2: only published variants can be downloaded here.
  const variants = (modelsQuery.data?.variants ?? []).filter(isPublished)
  const defaultId = modelsQuery.data?.default_variant ?? null
  const activeId = selectedId ?? defaultId ?? variants[0]?.variant_id ?? null
  const active = variants.find((variant) => variant.variant_id === activeId) ?? null
  const state = active?.download?.state ?? (active !== null ? 'ready' : null)
  const ready = state === 'ready' && active?.resident === true

  function handleSelect(variantId: string): void {
    setSelectedId(variantId)
    const variant = variants.find((entry) => entry.variant_id === variantId)
    if (variant?.download?.state === 'absent') {
      downloadMutation.mutate(variantId)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-base font-semibold">{t('setup.step.model.title')}</h2>
      <p className="text-sm text-muted-foreground">{t('setup.step.model.body')}</p>
      {active !== null ? (
        <div className="rounded-md border border-border p-3">
          <p className="flex items-center gap-2 text-sm font-medium text-foreground">
            {active.label}
            {active.variant_id === defaultId ? (
              <span className="text-xs font-normal text-muted-foreground">{t('convert.model.default')}</span>
            ) : null}
          </p>
          {active.download !== null ? (
            <DownloadProgress
              state={active.download.state}
              bytesDone={active.download.bytes_done}
              bytesTotal={active.download.bytes_total}
              errorCode={active.download.error_code}
              language={language}
              retryLabel={t(active.downloaded ? 'models.action.load' : 'models.action.retryDownload')}
              onRetry={() => {
                if (active.downloaded) loadMutation.mutate(active.variant_id)
                else downloadMutation.mutate(active.variant_id)
              }}
            />
          ) : (
            <p className="text-xs text-muted-foreground">{t('models.state.ready')}</p>
          )}
          {state === 'ready' && !ready && active.downloaded ? (
            <Button
              type="button"
              aria-disabled={loadMutation.isPending}
              onClick={() => {
                if (!loadMutation.isPending) loadMutation.mutate(active.variant_id)
              }}
            >
              {t('models.action.load')}
            </Button>
          ) : null}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">{t('models.empty')}</p>
      )}
      {variants.length > 1 ? (
        <ModelStepVariantList variants={variants} activeId={activeId} onSelect={handleSelect} />
      ) : null}
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          aria-disabled={!ready}
          onClick={() => {
            if (ready) onNext()
          }}
        >
          {t('setup.continue')}
        </Button>
        <Button type="button" variant="outline" onClick={onNext}>
          {t('setup.step.model.finishLater')}
        </Button>
      </div>
    </div>
  )
}

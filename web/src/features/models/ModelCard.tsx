import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { formatMebibytes } from '@/lib/format'
import type { SupportedLanguage } from '@/i18n'
import { useDownloadModelMutation, useSetDefaultModelMutation, type ModelVariantEntry } from './api'
import { ModelCardActions, type CardAction } from './ModelCardActions'
import { ModelCardBadges } from './ModelCardBadges'
import { ModelDeleteDialog, ModelLoadDialog } from './ModelCardDialogs'
import { DownloadProgress } from './DownloadProgress'
import { describeEngineUnavailableReason } from './engineReason'
import { describeFitsReason, type Translate } from './fits'
import { VramMeter } from './VramMeter'

type ModelsResponseDevice = components['schemas']['ModelsResponseDevice']

export interface ModelCardProps {
  variant: ModelVariantEntry
  resident: ModelVariantEntry | null
  device: ModelsResponseDevice | null
  language: SupportedLanguage
}

const TRANSFER_STATES = new Set(['downloading', 'verifying', 'loading', 'failed'])

/** UX §3.7/§7.3 "Modely": one variant's card -- badges, VRAM, transfer progress or actions. */
export function ModelCard({ variant, resident, device, language }: ModelCardProps) {
  const { t } = useTranslation()
  const [loadOpen, setLoadOpen] = useState(false)
  const [deleteOpen, setDeleteOpen] = useState(false)
  const downloadMutation = useDownloadModelMutation()
  const setDefaultMutation = useSetDefaultModelMutation()

  const planned = variant.availability === 'planned'
  const translate: Translate = (key, params) =>
    (t as (key: string, params?: Record<string, string>) => string)(key, params)
  const fitsReasonText = describeFitsReason(variant, device, language, translate)
  const engineReasonText = describeEngineUnavailableReason(variant, translate)
  const gatedReasonText = fitsReasonText ?? engineReasonText
  const deleteDisabledReason = variant.resident
    ? t('models.confirm.delete.deniedResident')
    : variant.source_kind === 'local'
      ? t('models.confirm.delete.deniedLocal')
      : null

  const actions: CardAction[] = []
  if (variant.availability !== 'published') {
    // models_response 1.2: listed, but nothing exists to download, load or
    // select yet -- one disabled Download with the reason right below it.
    actions.push({
      id: `${variant.variant_id}-download`,
      label: t('models.action.download'),
      onClick: () => undefined,
      disabledReason: t(`models.availability.${variant.availability}.reason`),
    })
  } else if (!variant.downloaded && variant.source_kind === 'hf') {
    actions.push({
      id: `${variant.variant_id}-download`,
      label: t('models.action.download'),
      onClick: () => {
        downloadMutation.mutate(variant.variant_id)
      },
      disabledReason: fitsReasonText,
      pending: downloadMutation.isPending,
    })
  } else {
    if (!variant.resident) {
      actions.push({
        id: `${variant.variant_id}-load`,
        label: t('models.action.load'),
        onClick: () => {
          setLoadOpen(true)
        },
        disabledReason: gatedReasonText,
      })
    }
    actions.push({
      id: `${variant.variant_id}-delete`,
      label: t('models.action.delete'),
      onClick: () => {
        setDeleteOpen(true)
      },
      disabledReason: deleteDisabledReason,
      variant: 'destructive',
    })
  }
  if (!variant.is_default && variant.availability === 'published') {
    actions.push({
      id: `${variant.variant_id}-set-default`,
      label: t('models.action.setDefault'),
      onClick: () => {
        setDefaultMutation.mutate(variant.variant_id)
      },
      disabledReason: gatedReasonText,
      pending: setDefaultMutation.isPending,
    })
  }

  const showTransfer = variant.download !== null && TRANSFER_STATES.has(variant.download.state)

  return (
    <Card>
      <CardHeader>
        <ModelCardBadges variant={variant} />
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {variant.estimated_peak_vram_mib !== null ? (
          <VramMeter
            totalMiB={device?.total_vram_mib ?? variant.estimated_peak_vram_mib}
            estimateMiB={variant.estimated_peak_vram_mib}
            compact
            language={language}
          />
        ) : null}
        <p className="text-xs text-muted-foreground">
          {variant.estimated_peak_vram_mib !== null
            ? formatMebibytes(variant.estimated_peak_vram_mib, language)
            : planned
              ? t('models.availability.planned.vram')
              : t('common.unknown')}
        </p>
        {fitsReasonText !== null ? <p className="text-xs text-muted-foreground">{fitsReasonText}</p> : null}
        {variant.fits === null && !planned ? (
          <p className="text-xs text-muted-foreground">{t('models.fits.unknownNote')}</p>
        ) : null}
        {showTransfer && variant.download !== null ? (
          <DownloadProgress
            state={variant.download.state}
            bytesDone={variant.download.bytes_done}
            bytesTotal={variant.download.bytes_total}
            errorCode={variant.download.error_code}
            language={language}
            retryLabel={t(variant.downloaded ? 'models.action.load' : 'models.action.retryDownload')}
            onRetry={() => {
              if (variant.downloaded) setLoadOpen(true)
              else downloadMutation.mutate(variant.variant_id)
            }}
          />
        ) : (
          <ModelCardActions actions={actions} />
        )}
      </CardContent>
      <ModelLoadDialog
        variant={variant}
        resident={resident}
        language={language}
        open={loadOpen}
        onOpenChange={setLoadOpen}
      />
      <ModelDeleteDialog variant={variant} language={language} open={deleteOpen} onOpenChange={setDeleteOpen} />
    </Card>
  )
}

import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'
import { formatBytes } from '@/lib/format'
import type { SupportedLanguage } from '@/i18n'

type ModelVariantEntry = components['schemas']['ModelVariantEntry']

export interface ModelSwitchNoticeProps {
  selected: ModelVariantEntry
  resident: ModelVariantEntry | null
  language: SupportedLanguage
}

/** UX §3.7 "Upozornenie na cenu prepnutia": shown whenever `selected` is not the resident variant. */
export function ModelSwitchNotice({ selected, resident, language }: ModelSwitchNoticeProps) {
  const { t } = useTranslation()

  if (resident === null) {
    return <p className="text-sm text-muted-foreground">{t('models.switchNotice.noneResident')}</p>
  }

  return (
    <p className="text-sm text-muted-foreground">
      {t('models.switchNotice.downloaded', { resident: resident.label, selected: selected.label })}
      {!selected.downloaded && selected.download_size_bytes !== null
        ? ` ${t('models.switchNotice.notDownloaded', { size: formatBytes(selected.download_size_bytes, language) })}`
        : ''}
    </p>
  )
}

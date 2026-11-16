import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'
import { formatMebibytes } from '@/lib/format'
import { formatRelativeTime } from '@/lib/time'
import type { SupportedLanguage } from '@/i18n'
import { VramMeter } from './VramMeter'

type ModelsResponseDevice = components['schemas']['ModelsResponseDevice']

export interface DevicePanelProps {
  device: ModelsResponseDevice | null
  now: Date
  language: SupportedLanguage
}

/** UX §7.3 "Modely" header: total/free VRAM, compute capability, and a meter; `device = null` while the worker has not reported yet. */
export function DevicePanel({ device, now, language }: DevicePanelProps) {
  const { t } = useTranslation()

  if (device === null) {
    return (
      <section aria-label={t('models.device.label')} className="rounded-md border border-border bg-card p-4">
        <p className="text-sm text-muted-foreground">{t('models.device.unknown')}</p>
      </section>
    )
  }

  const [major, minor] = device.compute_capability

  return (
    <section
      aria-label={t('models.device.label')}
      className="flex flex-col gap-2 rounded-md border border-border bg-card p-4"
    >
      <p className="text-sm text-foreground">
        {t('models.device.summary', {
          total: formatMebibytes(device.total_vram_mib, language),
          free: formatMebibytes(device.free_vram_mib, language),
          observed: formatRelativeTime(new Date(device.observed_at), now, language),
          cc: `${String(major)}.${String(minor)}`,
        })}
      </p>
      <VramMeter totalMiB={device.total_vram_mib} freeMiB={device.free_vram_mib} language={language} />
    </section>
  )
}

import { Link } from 'react-router'
import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { isPublished } from '@/features/models/availability'
import { describeEngineUnavailableReason } from '@/features/models/engineReason'
import { describeFitsReason, type Translate } from '@/features/models/fits'
import { ModelSwitchNotice } from '@/features/models/ModelSwitchNotice'
import { formatMebibytes } from '@/lib/format'
import type { SupportedLanguage } from '@/i18n'

type ModelsResponse = components['schemas']['ModelsResponse']
type ModelVariantEntry = components['schemas']['ModelVariantEntry']

export interface ModelPickerProps {
  models: ModelsResponse
  value: string | null
  onChange: (variantId: string) => void
  language: SupportedLanguage
}

function isSelectable(variant: ModelVariantEntry): boolean {
  return variant.fits !== false && variant.engine_available
}

/**
 * UX §4.6/§3.7, `models_response` 1.1: a variant that does not fit or whose
 * engine is unavailable stays listed and disabled with its reason visible
 * next to it (owner decision D-A (b) -- never filtered out, the user must
 * see why it is missing). `ModelSwitchNotice` appears whenever the current
 * selection is not the resident variant. A single variant collapses to one
 * plain row instead of a radio group.
 */
export function ModelPicker({ models, value, onChange, language }: ModelPickerProps) {
  const { t } = useTranslation()
  const translate: Translate = (key, params) =>
    (t as (key: string, params?: Record<string, string>) => string)(key, params)

  // models_response 1.2: unpublished/planned variants cannot run a job at
  // all (not a fit/engine gate the user could change), so they stay on the
  // Models screen and never appear here.
  const variants = models.variants.filter(isPublished)

  if (variants.length === 0) {
    return null
  }

  if (variants.length === 1) {
    const [only] = variants
    return <p className="text-sm text-foreground">{t('convert.model.single', { label: only?.label })}</p>
  }

  const selected = variants.find((variant) => variant.variant_id === value) ?? null
  const resident = variants.find((variant) => variant.variant_id === models.resident_variant) ?? null

  return (
    <div className="flex flex-col gap-2">
      <RadioGroup value={value ?? ''} onValueChange={onChange} aria-label={t('convert.model.label')}>
        {variants.map((variant) => {
          const id = `convert-model-${variant.variant_id}`
          const selectable = isSelectable(variant)
          const reasonText =
            describeFitsReason(variant, models.device, language, translate) ??
            describeEngineUnavailableReason(variant, translate)
          return (
            <div key={variant.variant_id} className="flex items-start gap-2">
              <RadioGroupItem value={variant.variant_id} id={id} disabled={!selectable} className="mt-1" />
              <Label htmlFor={id} className="flex flex-col gap-0.5">
                <span className="flex items-center gap-2">
                  {variant.label}
                  {variant.resident ? <span className="text-xs text-accent">{t('models.switch.loaded')}</span> : null}
                  {variant.is_default ? (
                    <span className="text-xs text-muted-foreground">{t('convert.model.default')}</span>
                  ) : null}
                </span>
                <span className="text-xs text-muted-foreground">
                  {variant.estimated_peak_vram_mib !== null
                    ? formatMebibytes(variant.estimated_peak_vram_mib, language)
                    : t('common.unknown')}
                  {reasonText !== null ? ` · ${reasonText}` : ''}
                </span>
              </Label>
            </div>
          )
        })}
      </RadioGroup>
      {selected !== null && selected.variant_id !== models.resident_variant ? (
        <ModelSwitchNotice selected={selected} resident={resident} language={language} />
      ) : null}
      <Link to="/models" className="text-sm text-accent underline-offset-4 hover:underline">
        {t('convert.model.manage')}
      </Link>
    </div>
  )
}

import { useTranslation } from 'react-i18next'
import { useModelsQuery } from '@/api/queries'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useSetDefaultModelMutation } from '@/features/models/api'
import { isPublished } from '@/features/models/availability'
import { useSettingsQuery } from './api'

const AUTO_VALUE = 'auto'

/**
 * UX §6 "Nastavenia" / design §3.10: `Select` over variants with `fits !==
 * false`, `PUT /v1/models/default`. A variant that does not fit is left out
 * of the options entirely (not disabled) -- the accompanying note explains
 * why the list is shorter than the Models screen's. `AUTO_VALUE` clears the
 * user override (`variant_id: null`), falling back to config/auto.
 */
export function DefaultModelSelect() {
  const { t } = useTranslation()
  const modelsQuery = useModelsQuery()
  const settingsQuery = useSettingsQuery()
  const setDefaultMutation = useSetDefaultModelMutation()

  // models_response 1.2: an unpublished/planned variant can never be the
  // default (the API refuses it) -- it is not an "excluded because it does
  // not fit" case, so it does not count toward `hasExcluded` either.
  const published = (modelsQuery.data?.variants ?? []).filter(isPublished)
  const variants = published.filter((variant) => variant.fits !== false)
  const hasExcluded = published.length > variants.length
  const override = settingsQuery.data?.default_variant.override ?? null
  const value = override ?? AUTO_VALUE

  function handleChange(next: string): void {
    setDefaultMutation.mutate(next === AUTO_VALUE ? null : next)
  }

  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor="settings-default-model">{t('settings.defaultModel.label')}</Label>
      <Select value={value} onValueChange={handleChange}>
        <SelectTrigger id="settings-default-model" className="max-w-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={AUTO_VALUE}>{t('settings.defaultModel.auto')}</SelectItem>
          {variants.map((variant) => (
            <SelectItem key={variant.variant_id} value={variant.variant_id}>
              {variant.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {hasExcluded ? <p className="text-xs text-muted-foreground">{t('settings.defaultModel.excludedNote')}</p> : null}
    </div>
  )
}

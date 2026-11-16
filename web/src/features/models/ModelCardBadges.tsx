import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'
import { Badge } from '@/components/ui/badge'
import { CardTitle } from '@/components/ui/card'

type ModelVariantEntry = components['schemas']['ModelVariantEntry']

export interface ModelCardBadgesProps {
  variant: ModelVariantEntry
}

/**
 * UX §7.3 `ModelCard` header: label plus default/resident/engine/fits
 * badges, and (`models_response` 1.2) an availability badge for a variant
 * that is not published yet. A planned model has no fit or engine to state,
 * so it carries neither badge.
 */
export function ModelCardBadges({ variant }: ModelCardBadgesProps) {
  const { t } = useTranslation()
  const planned = variant.availability === 'planned'

  return (
    <CardTitle className="flex flex-wrap items-center gap-2 text-base">
      {variant.label}
      {variant.availability === 'unpublished' ? (
        <Badge variant="secondary">{t('models.availability.unpublished.badge')}</Badge>
      ) : null}
      {planned ? <Badge variant="secondary">{t('models.availability.planned.badge')}</Badge> : null}
      {variant.is_default ? <Badge variant="secondary">{t('models.badge.default')}</Badge> : null}
      {variant.resident ? <Badge variant="success">{t('models.badge.resident')}</Badge> : null}
      {planned ? null : <Badge variant="outline">{t('codes.engine.native')}</Badge>}
      {variant.fits === false ? <Badge variant="destructive">{t('models.fits.no')}</Badge> : null}
      {variant.fits === null && !planned ? <Badge variant="outline">{t('models.fits.unknown')}</Badge> : null}
    </CardTitle>
  )
}

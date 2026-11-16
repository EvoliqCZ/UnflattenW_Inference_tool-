import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'

type ModelVariantEntry = components['schemas']['ModelVariantEntry']

export interface ModelStepVariantListProps {
  variants: readonly ModelVariantEntry[]
  activeId: string | null
  onSelect: (variantId: string) => void
}

/** UX §11.2 "Vybrať iný model (N)": a collapsed radio group over the other variants. */
export function ModelStepVariantList({ variants, activeId, onSelect }: ModelStepVariantListProps) {
  const { t } = useTranslation()

  return (
    <Collapsible>
      <CollapsibleTrigger className="text-sm text-accent underline-offset-4 hover:underline">
        {t('setup.step.model.chooseOther', { count: variants.length })}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <RadioGroup value={activeId ?? ''} onValueChange={onSelect} className="mt-2 flex flex-col gap-2">
          {variants.map((variant) => {
            const id = `setup-model-${variant.variant_id}`
            return (
              <div key={variant.variant_id} className="flex items-center gap-2">
                <RadioGroupItem value={variant.variant_id} id={id} disabled={variant.fits === false} />
                <Label htmlFor={id}>{variant.label}</Label>
              </div>
            )
          })}
        </RadioGroup>
      </CollapsibleContent>
    </Collapsible>
  )
}

import { useTranslation } from 'react-i18next'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import type { components } from '@/api/schema'

export type ConversionOutput = components['schemas']['SubmissionResponse']['requested_output']

export interface OutputChoiceProps {
  value: ConversionOutput
  onChange: (value: ConversionOutput) => void
}

/** UX §4.5: two visible options, `diagnostic_docx` pre-selected, no other switches (H4). */
export function OutputChoice({ value, onChange }: OutputChoiceProps) {
  const { t } = useTranslation()

  return (
    <RadioGroup
      value={value}
      onValueChange={(next) => {
        onChange(next as ConversionOutput)
      }}
      aria-label={t('convert.output.legend')}
    >
      {(['diagnostic_docx', 'prediction_ir'] as const).map((option) => {
        const key = option === 'diagnostic_docx' ? 'docx' : 'ir'
        const id = `convert-output-${option}`
        return (
          <div key={option} className="flex items-start gap-2">
            <RadioGroupItem value={option} id={id} className="mt-1" />
            <Label htmlFor={id} className="flex flex-col gap-0.5 font-normal">
              <span className="font-medium">{t(`convert.output.${key}.label`)}</span>
              <span className="text-sm text-muted-foreground">{t(`convert.output.${key}.hint`)}</span>
            </Label>
          </div>
        )
      })}
    </RadioGroup>
  )
}

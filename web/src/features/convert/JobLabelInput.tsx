import { useTranslation } from 'react-i18next'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

export const LABEL_MAX_CODE_POINTS = 120
const LABEL_COUNTER_THRESHOLD = 100
const FORBIDDEN_LABEL_CHARS = /[\\/:*?"<>|]/

const FORBIDDEN_LABEL_CATEGORY = /[\p{Cc}\p{Cf}\p{Zl}\p{Zp}]/u

/**
 * Client-side mirror of the server's label rule (design §3.2: NFC, forbidden
 * characters/categories, 1-120 code points). The server re-validates and
 * remains authoritative; an empty label is valid (the field is optional).
 */
export function isLabelValid(value: string): boolean {
  const normalized = value.normalize('NFC').trim()
  if (normalized === '') {
    return true
  }
  const codePoints = Array.from(normalized).length
  return (
    codePoints <= LABEL_MAX_CODE_POINTS &&
    !FORBIDDEN_LABEL_CHARS.test(normalized) &&
    !FORBIDDEN_LABEL_CATEGORY.test(normalized)
  )
}

export interface JobLabelInputProps {
  value: string
  onChange: (value: string) => void
}

/** UX §4.7: job label field, pre-filled by the caller from the selected file name(s). */
export function JobLabelInput({ value, onChange }: JobLabelInputProps) {
  const { t } = useTranslation()
  const codePoints = Array.from(value).length
  const valid = isLabelValid(value)

  return (
    <div className="flex flex-col gap-1">
      <Label htmlFor="convert-job-label">{t('convert.label.job')}</Label>
      <Input
        id="convert-job-label"
        value={value}
        aria-invalid={!valid}
        onChange={(event) => {
          onChange(event.target.value)
        }}
      />
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{t('convert.label.hint')}</span>
        {codePoints >= LABEL_COUNTER_THRESHOLD ? (
          <span>{t('convert.label.counter', { count: codePoints, max: LABEL_MAX_CODE_POINTS })}</span>
        ) : null}
      </div>
      {!valid ? <p className="text-xs text-destructive">{t('convert.label.invalid')}</p> : null}
    </div>
  )
}

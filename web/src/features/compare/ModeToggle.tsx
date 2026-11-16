import { useTranslation } from 'react-i18next'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { CompareMode } from './types'

export interface ModeToggleProps {
  mode: CompareMode
  onChange: (mode: CompareMode) => void
}

const MODES: CompareMode[] = ['side', 'overlay', 'swipe']

/** UX §5.2: side-by-side / overlay / swipe. Keyboard letter shortcuts (S/O/W) are T28 scope. */
export function ModeToggle({ mode, onChange }: ModeToggleProps) {
  const { t } = useTranslation()

  return (
    <ToggleGroup
      type="single"
      value={mode}
      onValueChange={(value) => {
        if (value) onChange(value as CompareMode)
      }}
      aria-label={t('compare.modeLabel')}
    >
      {MODES.map((candidate) => (
        <ToggleGroupItem key={candidate} value={candidate} aria-label={t(`compare.mode.${candidate}`)}>
          {t(`compare.mode.${candidate}`)}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  )
}

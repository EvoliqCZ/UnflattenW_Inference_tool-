import { useTranslation } from 'react-i18next'
import { Switch } from '@/components/ui/switch'

export interface PagingModeToggleProps {
  linked: boolean
  onChange: (linked: boolean) => void
}

/**
 * UX §5.5 "Listovať spolu/samostatne" -- only rendered by the caller when
 * `sourceCount !== docxCount`; checked = "spolu" (one shared index, the
 * default), unchecked = "samostatne" (each pane pages independently).
 */
export function PagingModeToggle({ linked, onChange }: PagingModeToggleProps) {
  const { t } = useTranslation()

  return (
    <label className="flex items-center gap-2 text-sm">
      <Switch checked={linked} onCheckedChange={onChange} aria-label={t('compare.linkPages')} />
      {t('compare.linkPages')}
    </label>
  )
}

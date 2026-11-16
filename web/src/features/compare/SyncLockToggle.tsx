import { Link, Unlink } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Toggle } from '@/components/ui/toggle'

export interface SyncLockToggleProps {
  linked: boolean
  onChange: (linked: boolean) => void
}

/** UX §5.3 🔗 "Zámok synchronizácie" -- side mode only (caller decides when to render this). */
export function SyncLockToggle({ linked, onChange }: SyncLockToggleProps) {
  const { t } = useTranslation()

  return (
    <Toggle pressed={linked} onPressedChange={onChange} aria-label={t('compare.link')} aria-keyshortcuts="l" size="sm">
      {linked ? <Link aria-hidden="true" /> : <Unlink aria-hidden="true" />}
    </Toggle>
  )
}

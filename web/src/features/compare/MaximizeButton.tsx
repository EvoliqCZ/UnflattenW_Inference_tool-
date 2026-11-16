import { Minimize2, Maximize2 } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'

export interface MaximizeButtonProps {
  maximized: boolean
  onToggle: () => void
}

/** UX §5.8 "Maximalizovať" (F) -- see `ComparisonViewer`'s `.compare-maximized` for the scope decision. */
export function MaximizeButton({ maximized, onToggle }: MaximizeButtonProps) {
  const { t } = useTranslation()

  return (
    <Button type="button" variant="outline" size="sm" aria-keyshortcuts="f" aria-pressed={maximized} onClick={onToggle}>
      {maximized ? <Minimize2 aria-hidden="true" /> : <Maximize2 aria-hidden="true" />}
      {t(maximized ? 'compare.restore' : 'compare.maximize')}
    </Button>
  )
}

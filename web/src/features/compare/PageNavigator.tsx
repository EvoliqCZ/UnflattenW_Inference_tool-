import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

export interface PageNavigatorProps {
  /** 0-based. */
  index: number
  /** Number of pages in the linked range (UX §5.4/§5.5: `max(sourceCount, docxCount)`). */
  total: number
  onNavigate: (index: number) => void
  /** UX §5.5 "samostatne": overrides id/label so two navigators (source, docx) can coexist. */
  id?: string
  label?: string
}

/**
 * UX §5.4: prev/next and a "Strana [ n ] z N" field. Home/End and single-key
 * shortcuts are wired at the viewer level (`useComparisonShortcuts`, T28);
 * this ships click/Enter navigation and polls `polite` for screen readers
 * on page change.
 */
export function PageNavigator({ index, total, onNavigate, id = 'compare-page-input', label }: PageNavigatorProps) {
  const { t } = useTranslation()
  const navLabel = label ?? t('compare.pageInputLabel')
  const [draft, setDraft] = useState(String(index + 1))
  // React's "adjusting state when a prop changes" pattern (not an effect):
  // an external navigation (prev/next/thumbnail click) must overwrite
  // whatever the user was mid-typing.
  const [syncedIndex, setSyncedIndex] = useState(index)
  if (index !== syncedIndex) {
    setSyncedIndex(index)
    setDraft(String(index + 1))
  }

  function commitDraft() {
    const parsed = Number.parseInt(draft, 10)
    if (Number.isNaN(parsed)) {
      setDraft(String(index + 1))
      return
    }
    const clamped = Math.min(Math.max(parsed - 1, 0), total - 1)
    onNavigate(clamped)
  }

  return (
    <nav aria-label={navLabel} className="flex items-center gap-2">
      <Button
        type="button"
        variant="outline"
        size="icon"
        disabled={index <= 0}
        onClick={() => {
          onNavigate(index - 1)
        }}
      >
        <ChevronLeft aria-hidden="true" />
        <span className="sr-only">{t('common.back')}</span>
      </Button>
      <Label htmlFor={id} className="sr-only">
        {navLabel}
      </Label>
      <Input
        id={id}
        type="number"
        min={1}
        max={total}
        value={draft}
        className="w-16"
        onChange={(event) => {
          setDraft(event.target.value)
        }}
        onBlur={commitDraft}
        onKeyDown={(event) => {
          if (event.key === 'Enter') commitDraft()
        }}
      />
      <span aria-live="polite">{t('compare.pageOf', { current: index + 1, total })}</span>
      <Button
        type="button"
        variant="outline"
        size="icon"
        disabled={index >= total - 1}
        onClick={() => {
          onNavigate(index + 1)
        }}
      >
        <ChevronRight aria-hidden="true" />
        <span className="sr-only">{t('common.next')}</span>
      </Button>
    </nav>
  )
}

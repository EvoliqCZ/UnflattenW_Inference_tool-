import { useTranslation } from 'react-i18next'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { SHORTCUT_ROWS } from '@/features/compare/shortcuts'

export interface ShortcutHelpDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

/**
 * UX §5.9 "?": lists every comparison-viewer shortcut alongside its button
 * alternative (UX §5.10 WCAG 2.5.7 -- every dragging/one-key action also has
 * a toolbar button; this dialog is the discoverability surface for the key
 * itself, not a second way to trigger the action).
 */
export function ShortcutHelpDialog({ open, onOpenChange }: ShortcutHelpDialogProps) {
  const { t } = useTranslation()

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t('compare.shortcuts.title')}</DialogTitle>
        </DialogHeader>
        <table className="w-full text-sm">
          <tbody>
            {SHORTCUT_ROWS.map((row) => (
              <tr key={row.labelKey} className="border-b border-border last:border-0">
                <td className="py-1.5 pr-4 font-mono text-xs text-muted-foreground">{row.keys}</td>
                <td className="py-1.5">{t(row.labelKey)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </DialogContent>
    </Dialog>
  )
}

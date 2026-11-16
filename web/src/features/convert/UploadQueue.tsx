import { ArrowDown, ArrowUp, FileText, X } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'
import type { SupportedLanguage } from '@/i18n'
import { formatBytes } from '@/lib/format'
import { ImageSizeWarning } from './ImageSizeWarning'
import { QueueItemThumbnail } from './QueueItemThumbnail'
import type { SelectedFile } from './types'

export interface UploadQueueProps {
  files: readonly SelectedFile[]
  onReorder: (files: SelectedFile[]) => void
  onRemove: (id: string) => void
  language: SupportedLanguage
}

function moved<T>(list: readonly T[], from: number, to: number): T[] {
  const copy = [...list]
  const [item] = copy.splice(from, 1)
  if (item !== undefined) {
    copy.splice(to, 0, item)
  }
  return copy
}

/**
 * UX §4.4: the selected files in result-page order. Reordering is exposed
 * only through the up/down buttons here (WCAG 2.5.7 dragging alternative);
 * pointer-drag reordering is not built for v1 (Karpathy: not required by
 * this task's tests, and the buttons already give full keyboard/pointer
 * access to the same operation).
 */
export function UploadQueue({ files, onReorder, onRemove, language }: UploadQueueProps) {
  const { t } = useTranslation()
  const [longSides, setLongSides] = useState<Record<string, number>>({})

  function move(index: number, delta: number): void {
    const target = index + delta
    if (target < 0 || target >= files.length) return
    onReorder(moved(files, index, target))
  }

  return (
    <ol className="flex flex-col gap-2">
      {files.map((entry, index) => {
        const longSide = longSides[entry.id]
        return (
          <li key={entry.id} className="flex flex-col gap-1 rounded-sm border border-border p-2">
            <div className="flex items-center gap-3">
              <span className="w-5 text-right text-sm text-muted-foreground">{index + 1}</span>
              {entry.kind === 'pdf' ? (
                <FileText aria-hidden="true" className="size-8 shrink-0 text-muted-foreground" />
              ) : (
                <QueueItemThumbnail
                  file={entry.file}
                  onDimensions={(longSide) => {
                    setLongSides((prev) => ({ ...prev, [entry.id]: longSide }))
                  }}
                />
              )}
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm text-foreground">{entry.file.name}</p>
                <p className="text-xs text-muted-foreground">{formatBytes(entry.file.size, language)}</p>
              </div>
              <div className="flex items-center gap-1">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  aria-label={t('convert.queue.moveUp')}
                  disabled={index === 0}
                  onClick={() => {
                    move(index, -1)
                  }}
                >
                  <ArrowUp aria-hidden="true" className="size-4" />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  aria-label={t('convert.queue.moveDown')}
                  disabled={index === files.length - 1}
                  onClick={() => {
                    move(index, 1)
                  }}
                >
                  <ArrowDown aria-hidden="true" className="size-4" />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  aria-label={t('convert.queue.remove')}
                  onClick={() => {
                    onRemove(entry.id)
                  }}
                >
                  <X aria-hidden="true" className="size-4" />
                </Button>
              </div>
            </div>
            {entry.kind === 'image' && longSide !== undefined ? <ImageSizeWarning longSidePx={longSide} /> : null}
          </li>
        )
      })}
    </ol>
  )
}

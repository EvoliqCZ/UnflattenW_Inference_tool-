import { useRef, useState, type DragEvent } from 'react'
import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export interface DropzoneProps {
  onFilesSelected: (files: File[]) => void
  disabled?: boolean
  disabledReason?: string | undefined
  hasFiles?: boolean
}

/**
 * UX §4.2: a full-width drop target with a real `<button>` to open the file
 * picker (never a clickable `<div>`) and a visually-hidden but still
 * accessible `<input type="file">`. Drag-over highlighting and an
 * `invalid-drag` state (unsupported type in `DataTransfer.items`) give
 * feedback before the drop; the actual accept/reject decision is
 * `validateSelection` on the dropped/picked files, run by the caller.
 */
export function Dropzone({ onFilesSelected, disabled = false, disabledReason, hasFiles = false }: DropzoneProps) {
  const { t } = useTranslation()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragState, setDragState] = useState<'idle' | 'drag-over' | 'invalid-drag'>('idle')

  function hasUnsupportedDragType(event: DragEvent<HTMLDivElement>): boolean {
    return Array.from(event.dataTransfer.items).some(
      (item) =>
        item.kind === 'file' && item.type !== '' && item.type !== 'application/pdf' && !item.type.startsWith('image/'),
    )
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault()
    if (disabled) return
    setDragState(hasUnsupportedDragType(event) ? 'invalid-drag' : 'drag-over')
  }

  function handleDragLeave(): void {
    setDragState('idle')
  }

  function handleDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault()
    setDragState('idle')
    if (disabled) return
    onFilesSelected(Array.from(event.dataTransfer.files))
  }

  function handleInputChange(event: React.ChangeEvent<HTMLInputElement>): void {
    if (event.target.files) {
      onFilesSelected(Array.from(event.target.files))
      event.target.value = ''
    }
  }

  return (
    <div
      data-state={disabled ? 'disabled' : dragState}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        'flex flex-col items-center justify-center gap-2 rounded-sm border border-dashed border-border-strong text-center',
        hasFiles ? 'h-24' : 'h-[200px]',
        dragState === 'drag-over' && 'border-accent bg-accent-subtle',
        dragState === 'invalid-drag' && 'border-destructive',
        disabled && 'opacity-50',
      )}
    >
      <p className="text-sm text-foreground">{t('convert.dropzone.label')}</p>
      {dragState === 'invalid-drag' ? (
        <p className="text-sm text-destructive">{t('convert.dropzone.invalidType')}</p>
      ) : null}
      <Button type="button" variant="outline" size="sm" disabled={disabled} onClick={() => inputRef.current?.click()}>
        {t('convert.dropzone.choose')}
      </Button>
      {disabled && disabledReason ? <p className="text-sm text-muted-foreground">{disabledReason}</p> : null}
      <input
        ref={inputRef}
        type="file"
        multiple
        accept="application/pdf,image/png,image/jpeg"
        disabled={disabled}
        onChange={handleInputChange}
        className="sr-only"
        aria-label={t('convert.dropzone.choose')}
      />
    </div>
  )
}

import type { ReactNode } from 'react'
import { Button } from '@/components/ui/button'

export interface CardAction {
  id: string
  label: ReactNode
  onClick: () => void
  disabledReason: string | null
  pending?: boolean
  variant?: 'default' | 'outline' | 'destructive'
}

/**
 * UX §3.7: a disabled action stays focusable and keeps `aria-disabled`
 * (never `disabled`), with its reason rendered right below so a screen
 * reader announces it via `aria-describedby` -- never hidden in a tooltip
 * alone. The shared Button styles `aria-disabled="true"` itself (muted
 * tokens, not-allowed cursor, no hover change), whatever the variant.
 */
function ActionButton({ id, label, onClick, disabledReason, pending, variant }: CardAction) {
  const disabled = disabledReason !== null || (pending ?? false)
  const reasonId = `${id}-reason`
  return (
    <div className="flex flex-col gap-1">
      <Button
        type="button"
        size="sm"
        variant={variant ?? 'outline'}
        aria-disabled={disabled}
        aria-describedby={disabledReason !== null ? reasonId : undefined}
        onClick={() => {
          if (!disabled) onClick()
        }}
      >
        {label}
      </Button>
      {disabledReason !== null ? (
        <p id={reasonId} className="text-xs text-muted-foreground">
          {disabledReason}
        </p>
      ) : null}
    </div>
  )
}

export interface ModelCardActionsProps {
  actions: CardAction[]
}

/** UX §7.3 `ModelCard`: the row of `onDownload`/`onLoad`/`onSetDefault`/`onDelete` buttons. */
export function ModelCardActions({ actions }: ModelCardActionsProps) {
  return (
    <div className="flex flex-wrap gap-3">
      {actions.map((action) => (
        <ActionButton key={action.id} {...action} />
      ))}
    </div>
  )
}

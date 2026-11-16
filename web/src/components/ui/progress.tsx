import * as ProgressPrimitive from '@radix-ui/react-progress'
import type { ComponentProps } from 'react'
import { cn } from '@/lib/utils'

export interface ProgressProps extends ComponentProps<typeof ProgressPrimitive.Root> {
  /** UX §8.5: `aria-valuetext` for screen readers, e.g. "strana 3 z 12" -- a percentage alone is not enough context. */
  valueLabel?: string
}

export function Progress({ className, value, valueLabel, ...props }: ProgressProps) {
  return (
    <ProgressPrimitive.Root
      value={value}
      aria-valuetext={valueLabel}
      className={cn('h-2 w-full overflow-hidden rounded-full bg-surface-sunken', className)}
      {...props}
    >
      <ProgressPrimitive.Indicator
        className="h-full bg-accent transition-[width] duration-200 ease-out"
        style={{ width: `${String(value ?? 0)}%` }}
      />
    </ProgressPrimitive.Root>
  )
}

import * as TogglePrimitive from '@radix-ui/react-toggle'
import { type VariantProps, cva } from 'class-variance-authority'
import type { ComponentProps } from 'react'
import { cn } from '@/lib/utils'

export const toggleVariants = cva(
  'inline-flex items-center justify-center gap-1 rounded-sm border border-transparent text-sm font-medium ' +
    'data-[state=on]:border-border-strong data-[state=on]:bg-accent-subtle disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      size: {
        default: 'h-9 px-3',
        sm: 'h-8 px-2',
      },
    },
    defaultVariants: {
      size: 'default',
    },
  },
)

export interface ToggleProps extends ComponentProps<typeof TogglePrimitive.Root>, VariantProps<typeof toggleVariants> {}

export function Toggle({ className, size, ...props }: ToggleProps) {
  return <TogglePrimitive.Root className={cn(toggleVariants({ size }), className)} {...props} />
}

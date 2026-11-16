import { type VariantProps, cva } from 'class-variance-authority'
import type { HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

export const badgeVariants = cva('inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium', {
  variants: {
    variant: {
      default: 'border-transparent bg-accent text-accent-ink',
      secondary: 'border-border bg-secondary text-secondary-foreground',
      success: 'border-transparent bg-success text-accent-ink',
      warning: 'border-transparent bg-warning-bg text-warning-text',
      destructive: 'border-transparent bg-destructive text-destructive-foreground',
      outline: 'border-border-strong text-foreground',
    },
  },
  defaultVariants: {
    variant: 'default',
  },
})

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement>, VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />
}

import { type VariantProps, cva } from 'class-variance-authority'
import type { HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

export const alertVariants = cva('relative w-full rounded-md border p-4 text-sm', {
  variants: {
    variant: {
      default: 'border-border bg-surface text-foreground',
      warning: 'border-warning-text/30 bg-warning-bg text-warning-text',
      destructive: 'border-destructive/30 bg-transparent text-destructive',
    },
  },
  defaultVariants: {
    variant: 'default',
  },
})

export interface AlertProps extends HTMLAttributes<HTMLDivElement>, VariantProps<typeof alertVariants> {}

export function Alert({ className, variant, ...props }: AlertProps) {
  return <div role="alert" className={cn(alertVariants({ variant }), className)} {...props} />
}

export function AlertTitle({ className, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return <h5 className={cn('mb-1 font-medium leading-none', className)} {...props} />
}

export function AlertDescription({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('text-sm [&_p]:leading-relaxed', className)} {...props} />
}

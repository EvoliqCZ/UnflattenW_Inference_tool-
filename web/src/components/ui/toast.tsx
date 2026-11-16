import { useTranslation } from 'react-i18next'
import * as ToastPrimitive from '@radix-ui/react-toast'
import { type VariantProps, cva } from 'class-variance-authority'
import { X } from 'lucide-react'
import type { ComponentProps } from 'react'
import { cn } from '@/lib/utils'

export const ToastProvider = ToastPrimitive.Provider

export function ToastViewport({ className, ...props }: ComponentProps<typeof ToastPrimitive.Viewport>) {
  return (
    <ToastPrimitive.Viewport
      className={cn(
        'fixed bottom-0 right-0 z-50 flex w-full max-w-sm flex-col gap-2 p-4 sm:bottom-4 sm:right-4',
        className,
      )}
      {...props}
    />
  )
}

const toastVariants = cva(
  'pointer-events-auto relative flex w-full items-start gap-3 rounded-md border p-4 shadow-none',
  {
    variants: {
      variant: {
        default: 'border-border bg-surface text-foreground',
        destructive: 'border-destructive/30 bg-surface text-destructive',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  },
)

export interface ToastProps extends ComponentProps<typeof ToastPrimitive.Root>, VariantProps<typeof toastVariants> {}

export function Toast({ className, variant, ...props }: ToastProps) {
  return <ToastPrimitive.Root className={cn(toastVariants({ variant }), className)} {...props} />
}

export function ToastTitle({ className, ...props }: ComponentProps<typeof ToastPrimitive.Title>) {
  return <ToastPrimitive.Title className={cn('text-sm font-semibold', className)} {...props} />
}

export function ToastDescription({ className, ...props }: ComponentProps<typeof ToastPrimitive.Description>) {
  return <ToastPrimitive.Description className={cn('text-sm text-muted-foreground', className)} {...props} />
}

export function ToastClose({ className, ...props }: ComponentProps<typeof ToastPrimitive.Close>) {
  const { t } = useTranslation()
  return (
    <ToastPrimitive.Close
      className={cn('absolute right-2 top-2 rounded-sm text-muted-foreground hover:text-foreground', className)}
      aria-label={t('common.close')}
      {...props}
    >
      <X className="size-4" />
    </ToastPrimitive.Close>
  )
}

export function ToastAction({ className, ...props }: ComponentProps<typeof ToastPrimitive.Action>) {
  return (
    <ToastPrimitive.Action
      className={cn('shrink-0 rounded-sm border border-border-strong px-2 py-1 text-sm font-medium', className)}
      {...props}
    />
  )
}

import { Slot } from '@radix-ui/react-slot'
import { type VariantProps, cva } from 'class-variance-authority'
import type { ButtonHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

// UX §3.7: a disabled action keeps `aria-disabled` (never native `disabled`)
// so it stays focusable and can announce its reason. Tailwind's `disabled:`
// variant only matches the native attribute, so `aria-disabled="true"` gets
// its own muted look: muted tokens, not-allowed cursor, and a neutralizer for
// every variant's hover utility (no hover change). No pointer-events-none --
// that would also hide the not-allowed cursor.
const ARIA_DISABLED =
  'aria-disabled:cursor-not-allowed aria-disabled:border aria-disabled:border-border ' +
  'aria-disabled:bg-muted aria-disabled:text-muted-foreground ' +
  'aria-disabled:hover:bg-muted aria-disabled:hover:opacity-100'

export const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-sm text-sm font-medium ' +
    'transition-colors disabled:pointer-events-none disabled:opacity-50 ' +
    ARIA_DISABLED +
    ' [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
  {
    variants: {
      variant: {
        default: 'bg-accent text-accent-ink hover:opacity-90',
        secondary: 'bg-secondary text-secondary-foreground border border-border hover:bg-surface-sunken',
        outline: 'border border-border-strong bg-transparent text-foreground hover:bg-secondary',
        ghost: 'text-foreground hover:bg-secondary',
        destructive: 'bg-destructive text-destructive-foreground hover:opacity-90',
        // a disabled link stays text: drop the base's muted box.
        link:
          'text-accent underline-offset-4 hover:underline ' +
          'aria-disabled:border-0 aria-disabled:bg-transparent ' +
          'aria-disabled:hover:bg-transparent aria-disabled:hover:no-underline',
      },
      size: {
        default: 'h-9 px-4',
        sm: 'h-8 px-3',
        lg: 'h-11 px-6',
        icon: 'size-9',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  },
)

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

export function Button({ className, variant, size, asChild = false, ...props }: ButtonProps) {
  const Comp = asChild ? Slot : 'button'
  return <Comp className={cn(buttonVariants({ variant, size }), className)} {...props} />
}

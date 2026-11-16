import type { HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

/** UX §8.4: reduced-motion turns the pulse off via the global `prefers-reduced-motion` rule in tokens.css. */
export function Skeleton({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('animate-pulse rounded-sm bg-surface-sunken', className)} {...props} />
}

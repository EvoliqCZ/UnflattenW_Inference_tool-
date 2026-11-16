import * as ToggleGroupPrimitive from '@radix-ui/react-toggle-group'
import type { VariantProps } from 'class-variance-authority'
import type { ComponentProps } from 'react'
import { toggleVariants } from '@/components/ui/toggle'
import { cn } from '@/lib/utils'

export function ToggleGroup({ className, children, ...props }: ComponentProps<typeof ToggleGroupPrimitive.Root>) {
  return (
    <ToggleGroupPrimitive.Root className={cn('inline-flex items-center gap-1', className)} {...props}>
      {children}
    </ToggleGroupPrimitive.Root>
  )
}

export function ToggleGroupItem({
  className,
  size,
  ...props
}: ComponentProps<typeof ToggleGroupPrimitive.Item> & VariantProps<typeof toggleVariants>) {
  return <ToggleGroupPrimitive.Item className={cn(toggleVariants({ size }), className)} {...props} />
}

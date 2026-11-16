import * as SliderPrimitive from '@radix-ui/react-slider'
import type { ComponentProps } from 'react'
import { cn } from '@/lib/utils'

export function Slider({ className, 'aria-label': ariaLabel, ...props }: ComponentProps<typeof SliderPrimitive.Root>) {
  return (
    <SliderPrimitive.Root className={cn('relative flex w-full touch-none items-center', className)} {...props}>
      <SliderPrimitive.Track className="relative h-1 w-full grow overflow-hidden rounded-full bg-surface-sunken">
        <SliderPrimitive.Range className="absolute h-full bg-accent" />
      </SliderPrimitive.Track>
      {/* `role="slider"` lives on the thumb, not the root (Radix source) -- an `aria-label` given
          here must be forwarded there, or a single-thumb slider has no accessible name at all. */}
      <SliderPrimitive.Thumb
        aria-label={ariaLabel}
        className="block size-4 rounded-full border border-border-strong bg-surface disabled:pointer-events-none disabled:opacity-50"
      />
    </SliderPrimitive.Root>
  )
}

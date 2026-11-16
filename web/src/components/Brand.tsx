import markUrl from '@/assets/brand/unflatten-w-mark.webp'
import { cn } from '@/lib/utils'

/**
 * Unflatten W icon-only mark with the product name as HTML text beside it.
 * The official lockup's navy wordmark is unreadable on the dark theme, so the
 * mark (`scripts/build_brand_assets.py`) never carries the name itself. The
 * image is the accessible name; the visible text is `aria-hidden` so a screen
 * reader hears "Unflatten W" once. No tagline: branding here must not read as
 * a product-quality claim (DEC-044/DEC-052).
 */
export function Brand({
  className,
  markClassName,
  textClassName,
}: {
  className?: string
  markClassName?: string
  textClassName?: string
}) {
  return (
    <span className={cn('flex items-center gap-2', className)}>
      <img src={markUrl} alt="Unflatten W" width={130} height={128} className={cn('w-auto', markClassName)} />
      <span aria-hidden="true" className={textClassName}>
        Unflatten W 1.0
      </span>
    </span>
  )
}

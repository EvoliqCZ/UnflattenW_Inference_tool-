import { useState } from 'react'
import { cn } from '@/lib/utils'
import { MissingPreview } from './MissingPreview'
import type { MissingReason } from './types'

export interface PagePaneProps {
  /** Server preview URL (design §3.7) -- never `blob:`/`data:` (CSP `img-src 'self'`). */
  src?: string | undefined
  alt: string
  missing?: MissingReason | undefined
  /** UX §5.2 overlay diff tint: colours this pane's image via CSS only. */
  tint?: 'source' | 'docx' | undefined
}

/**
 * One side of the comparison. `missing` (decided by the caller from
 * `artifacts`/`outcome`/`pages`) takes priority; otherwise renders the
 * `<img>` and falls back to a `loadFailed` `MissingPreview` on its own if
 * the browser fails to load `src` (UX §5.6 last row) -- retryable by
 * remounting the `<img>` with a fresh key. Callers key `<PagePane>` by
 * `src` (page navigation) so a stale `loadFailed` never survives a page
 * change -- a fresh `src` means a fresh component instance, not an effect.
 */
export function PagePane({ src, alt, missing, tint }: PagePaneProps) {
  const [loadFailed, setLoadFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)

  const effectiveMissing: MissingReason | undefined = missing ?? (loadFailed ? { kind: 'loadFailed' } : undefined)

  if (effectiveMissing) {
    return (
      <MissingPreview
        reason={effectiveMissing}
        onRetry={
          effectiveMissing.kind === 'loadFailed'
            ? () => {
                setLoadFailed(false)
                setAttempt((current) => current + 1)
              }
            : undefined
        }
      />
    )
  }

  return (
    <img
      key={attempt}
      src={src}
      alt={alt}
      loading="lazy"
      className={cn(
        'compare-page-image',
        tint === 'source' && 'compare-tint-source',
        tint === 'docx' && 'compare-tint-docx',
      )}
      onError={() => {
        setLoadFailed(true)
      }}
    />
  )
}

import { CircleDashed, XCircle } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import type { PageReport } from './types'

export interface ThumbnailStripProps {
  /** One entry per page 0..total-1; `undefined` = no preview at that index. */
  sourceThumbs: (string | undefined)[]
  docxThumbs: (string | undefined)[]
  pages: PageReport[] | null
  /** UX §5.5 "samostatne": defaults to the paired strip; `false` splits into two single-image rows. */
  linked?: boolean
  currentIndex: number
  onNavigate: (index: number) => void
  sourceIndex?: number
  docxIndex?: number
  onNavigateSource?: (index: number) => void
  onNavigateDocx?: (index: number) => void
}

function isFailed(pages: PageReport[] | null, index: number): boolean {
  return pages?.some((page) => page.index === index && page.state === 'failed') ?? false
}

/** UX §5.4/§5.5: a strip of "O | W" thumbnail pairs (linked), or two independent rows (separate). */
export function ThumbnailStrip({
  sourceThumbs,
  docxThumbs,
  pages,
  linked = true,
  currentIndex,
  onNavigate,
  sourceIndex = 0,
  docxIndex = 0,
  onNavigateSource,
  onNavigateDocx,
}: ThumbnailStripProps) {
  const { t } = useTranslation()
  const total = Math.max(sourceThumbs.length, docxThumbs.length)

  function label(index: number, count: number) {
    return t('compare.pageOf', { current: index + 1, total: count })
  }

  function row(thumbs: (string | undefined)[], activeIndex: number, onSelect: (index: number) => void) {
    return (
      <ol className="flex gap-2 overflow-x-auto">
        {thumbs.map((thumb, index) => (
          <li key={index}>
            <button
              type="button"
              aria-current={index === activeIndex ? 'page' : undefined}
              aria-label={label(index, thumbs.length)}
              className={cn(
                'flex gap-1 rounded-sm border p-0.5',
                index === activeIndex ? 'border-accent' : 'border-transparent',
              )}
              onClick={() => {
                onSelect(index)
              }}
            >
              <img src={thumb} alt="" loading="lazy" className="compare-thumbnail" />
              {isFailed(pages, index) ? <XCircle aria-hidden="true" className="size-3.5 text-destructive" /> : null}
              {!isFailed(pages, index) && thumb === undefined ? (
                <CircleDashed aria-hidden="true" className="size-3.5 text-muted-foreground" />
              ) : null}
            </button>
          </li>
        ))}
      </ol>
    )
  }

  if (!linked) {
    return (
      <div className="flex flex-col gap-2">
        {row(sourceThumbs, sourceIndex, onNavigateSource ?? (() => undefined))}
        {row(docxThumbs, docxIndex, onNavigateDocx ?? (() => undefined))}
      </div>
    )
  }

  return (
    <ol className="flex gap-2 overflow-x-auto">
      {Array.from({ length: total }, (_, index) => {
        const failed = isFailed(pages, index)
        const missing = !failed && (sourceThumbs[index] === undefined || docxThumbs[index] === undefined)
        const isCurrent = index === currentIndex

        return (
          <li key={index}>
            <button
              type="button"
              aria-current={isCurrent ? 'page' : undefined}
              aria-label={label(index, total)}
              className={cn('flex gap-1 rounded-sm border p-0.5', isCurrent ? 'border-accent' : 'border-transparent')}
              onClick={() => {
                onNavigate(index)
              }}
            >
              <img src={sourceThumbs[index]} alt="" loading="lazy" className="compare-thumbnail" />
              <img src={docxThumbs[index]} alt="" loading="lazy" className="compare-thumbnail" />
              {failed ? <XCircle aria-hidden="true" className="size-3.5 text-destructive" /> : null}
              {missing ? <CircleDashed aria-hidden="true" className="size-3.5 text-muted-foreground" /> : null}
            </button>
          </li>
        )
      })}
    </ol>
  )
}

import type { components } from '@/api/schema'

type PageReport = components['schemas']['PageReport']

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  const middle = sorted[mid] ?? 0
  if (sorted.length % 2 !== 0) {
    return middle
  }
  const beforeMiddle = sorted[mid - 1] ?? 0
  return (beforeMiddle + middle) / 2
}

/**
 * UX §3.4: "median `seconds` of finished pages x remaining pages, only from
 * the 2nd finished page on -- below that, show nothing (not 'computing...')".
 * Returns whole minutes, or `null` when fewer than 2 pages have finished.
 */
export function estimateRemainingMinutes(
  pages: readonly PageReport[] | null,
  pagesDone: number,
  pagesTotal: number,
): number | null {
  if (pagesDone < 2 || pages === null) {
    return null
  }
  const finishedSeconds = pages.filter((page) => page.state === 'succeeded').map((page) => page.seconds)
  if (finishedSeconds.length < 2) {
    return null
  }
  const remainingPages = Math.max(pagesTotal - pagesDone, 0)
  return Math.round((median(finishedSeconds) * remainingPages) / 60)
}

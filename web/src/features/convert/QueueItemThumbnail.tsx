import { useEffect, useRef, useState } from 'react'

const THUMBNAIL_WIDTH = 40
const THUMBNAIL_HEIGHT = 56

export interface QueueItemThumbnailProps {
  file: File
  onDimensions?: (longSidePx: number) => void
}

/**
 * UX §4.4/H8: draws the decoded `ImageBitmap` straight into a `<canvas>` --
 * never through a `blob:`/`data:` URL in an `<img>`, which CSP forbids
 * (design §6.3). `createImageBitmap` failing (undecodable file that slipped
 * past `validateSelection`, e.g. a race with a file removed on disk) falls
 * back to a plain placeholder instead of throwing.
 */
export function QueueItemThumbnail({ file, onDimensions }: QueueItemThumbnailProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [failed, setFailed] = useState(false)
  // `onDimensions` is an inline callback recreated on every parent render
  // (`UploadQueue`'s `.map`); keeping it out of the effect's dependency array
  // via a ref -- instead of depending on it directly -- stops the decode
  // from re-running (and looping, since reporting dimensions itself causes
  // a parent re-render) every time the parent renders.
  const onDimensionsRef = useRef(onDimensions)
  useEffect(() => {
    onDimensionsRef.current = onDimensions
  }, [onDimensions])

  useEffect(() => {
    let cancelled = false
    createImageBitmap(file)
      .then((bitmap) => {
        if (cancelled) {
          bitmap.close()
          return
        }
        onDimensionsRef.current?.(Math.max(bitmap.width, bitmap.height))
        const context = canvasRef.current?.getContext('2d')
        context?.drawImage(bitmap, 0, 0, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
        bitmap.close()
      })
      .catch(() => {
        if (!cancelled) {
          setFailed(true)
        }
      })
    return () => {
      cancelled = true
    }
  }, [file])

  if (failed) {
    return (
      <div
        aria-hidden="true"
        className="flex h-14 w-10 shrink-0 items-center justify-center rounded-sm border border-border bg-secondary text-xs text-muted-foreground"
      >
        ?
      </div>
    )
  }
  return (
    <canvas
      ref={canvasRef}
      width={THUMBNAIL_WIDTH}
      height={THUMBNAIL_HEIGHT}
      aria-hidden="true"
      className="shrink-0 rounded-sm border border-border"
    />
  )
}

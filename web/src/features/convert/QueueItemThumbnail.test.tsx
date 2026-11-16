import { render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { QueueItemThumbnail } from './QueueItemThumbnail'

function fakeBitmap(width: number, height: number) {
  return { width, height, close: vi.fn() }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('QueueItemThumbnail (UX §4.4/H8: canvas only, never a blob:/data: <img>)', () => {
  it('draws the decoded bitmap into a <canvas>, never an <img>', async () => {
    const bitmap = fakeBitmap(1200, 1600)
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue(bitmap))
    const file = new File(['x'], 'page.png', { type: 'image/png' })

    const { container } = render(<QueueItemThumbnail file={file} />)

    await waitFor(() => {
      expect(bitmap.close).toHaveBeenCalled()
    })
    expect(container.querySelector('canvas')).toBeInTheDocument()
    expect(container.querySelector('img')).not.toBeInTheDocument()
  })

  it('reports the long side in pixels via onDimensions', async () => {
    const bitmap = fakeBitmap(1200, 1600)
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue(bitmap))
    const onDimensions = vi.fn()
    const file = new File(['x'], 'page.png', { type: 'image/png' })

    render(<QueueItemThumbnail file={file} onDimensions={onDimensions} />)

    await waitFor(() => {
      expect(onDimensions).toHaveBeenCalledWith(1600)
    })
  })

  it('falls back to a placeholder when decoding fails', async () => {
    vi.stubGlobal('createImageBitmap', vi.fn().mockRejectedValue(new Error('cannot decode')))
    const file = new File(['x'], 'page.png', { type: 'image/png' })

    const { container, findByText } = render(<QueueItemThumbnail file={file} />)

    await findByText('?')
    expect(container.querySelector('canvas')).not.toBeInTheDocument()
  })
})

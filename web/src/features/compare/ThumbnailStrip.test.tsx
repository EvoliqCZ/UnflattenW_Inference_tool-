import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import type { PageReport } from './types'
import { ThumbnailStrip } from './ThumbnailStrip'

describe('ThumbnailStrip', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should render one item per page and mark the current one with aria-current', () => {
    render(
      <ThumbnailStrip
        sourceThumbs={['s0', 's1']}
        docxThumbs={['d0', 'd1']}
        pages={null}
        currentIndex={1}
        onNavigate={vi.fn()}
      />,
    )

    const items = screen.getAllByRole('button')
    expect(items).toHaveLength(2)
    expect(items[1]).toHaveAttribute('aria-current', 'page')
    expect(items[0]).not.toHaveAttribute('aria-current')
  })

  it('should call onNavigate with the clicked index', async () => {
    const onNavigate = vi.fn()
    const user = userEvent.setup()
    render(
      <ThumbnailStrip
        sourceThumbs={['s0', 's1']}
        docxThumbs={['d0', 'd1']}
        pages={null}
        currentIndex={0}
        onNavigate={onNavigate}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Strana 2 z 2' }))

    expect(onNavigate).toHaveBeenCalledWith(1)
  })

  it('should mark a failed page with the failed icon, and a page missing either preview (without failing) with the missing icon', () => {
    const pages: PageReport[] = [
      { decode_steps: null, error_code: 'RENDERER_FAILED', index: 0, seconds: 0.4, state: 'failed' },
    ]

    const { container } = render(
      <ThumbnailStrip
        sourceThumbs={['s0', 's1', 's2']}
        docxThumbs={[undefined, undefined, 'd2']}
        pages={pages}
        currentIndex={0}
        onNavigate={vi.fn()}
      />,
    )

    expect(container.querySelector('.text-destructive')).not.toBeNull()
    expect(container.querySelector('.text-muted-foreground')).not.toBeNull()
  })

  it('should split into two independent rows when linked is false', async () => {
    const onNavigateSource = vi.fn()
    const onNavigateDocx = vi.fn()
    const user = userEvent.setup()
    render(
      <ThumbnailStrip
        sourceThumbs={['s0', 's1']}
        docxThumbs={['d0']}
        pages={null}
        linked={false}
        currentIndex={0}
        onNavigate={vi.fn()}
        sourceIndex={1}
        docxIndex={0}
        onNavigateSource={onNavigateSource}
        onNavigateDocx={onNavigateDocx}
      />,
    )

    const items = screen.getAllByRole('button')
    expect(items).toHaveLength(3)
    expect(screen.getAllByRole('button', { name: 'Strana 2 z 2' })[0]).toHaveAttribute('aria-current', 'page')

    await user.click(screen.getByRole('button', { name: 'Strana 1 z 1' }))
    expect(onNavigateDocx).toHaveBeenCalledWith(0)
    expect(onNavigateSource).not.toHaveBeenCalled()
  })
})

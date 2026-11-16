import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { PagePane } from './PagePane'

describe('PagePane', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should render an <img> with a server URL when src is given and no missing reason applies', () => {
    render(<PagePane src="/v1/conversions/job-1/artifacts/source_preview/0" alt="Originál, strana 1 z 2" />)

    const img = screen.getByRole('img', { name: 'Originál, strana 1 z 2' })
    expect(img).toHaveAttribute('src', '/v1/conversions/job-1/artifacts/source_preview/0')
    expect(img.getAttribute('src')).not.toMatch(/^blob:|^data:/)
  })

  it('should render MissingPreview instead of an <img> when a missing reason is supplied', () => {
    render(<PagePane alt="ignored" missing={{ kind: 'noSource', index: 0 }} />)

    expect(screen.queryByRole('img', { name: 'ignored' })).not.toBeInTheDocument()
    expect(screen.getByText('Náhľad originálu pre stranu 1 chýba.')).toBeInTheDocument()
  })

  it('should switch to a loadFailed MissingPreview when the <img> fails to load, and retry re-mounts it', () => {
    render(<PagePane src="/v1/conversions/job-1/artifacts/docx_preview/0" alt="Word, strana 1" />)

    fireEvent.error(screen.getByRole('img', { name: 'Word, strana 1' }))

    expect(screen.getByText('Náhľad sa nepodarilo načítať.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Skúsiť znova' }))

    expect(screen.getByRole('img', { name: 'Word, strana 1' })).toBeInTheDocument()
  })

  it('should not carry a stale load failure into a fresh instance for a different page (callers key <PagePane> by src)', () => {
    // A remount (new `key`, i.e. a fresh component instance), not a `rerender` of the same one --
    // that is exactly what "callers key by src" (see PagePane.tsx docstring) means in practice.
    const { unmount } = render(<PagePane src="/v1/conversions/job-1/artifacts/docx_preview/0" alt="strana 1" />)
    fireEvent.error(screen.getByRole('img', { name: 'strana 1' }))
    expect(screen.getByText('Náhľad sa nepodarilo načítať.')).toBeInTheDocument()
    unmount()

    render(<PagePane src="/v1/conversions/job-1/artifacts/docx_preview/1" alt="strana 2" />)

    expect(screen.getByRole('img', { name: 'strana 2' })).toBeInTheDocument()
  })
})

import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import type { components } from '@/api/schema'
import { PageProgressList } from './PageProgressList'

type PageReport = components['schemas']['PageReport']

function page(overrides: Partial<PageReport>): PageReport {
  return { index: 0, state: 'succeeded', error_code: null, decode_steps: 100, seconds: 14.2, ...overrides }
}

describe('PageProgressList (UX §3.4)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows no time estimate with only 1 finished page', () => {
    const pages = [page({ index: 0, seconds: 30 })]
    render(<PageProgressList pagesTotal={3} pagesDone={1} currentPage={1} pages={pages} language="sk" />)
    expect(screen.queryByText(/odhad/)).not.toBeInTheDocument()
  })

  it('shows a time estimate once 2 pages have finished', () => {
    const pages = [page({ index: 0, seconds: 30 }), page({ index: 1, seconds: 60 })]
    render(<PageProgressList pagesTotal={5} pagesDone={2} currentPage={2} pages={pages} language="sk" />)
    expect(screen.getByText(/odhad/)).toBeInTheDocument()
  })

  it('renders a succeeded row with its seconds and decode steps', () => {
    const pages = [page({ index: 0, seconds: 14.2, decode_steps: 1842 })]
    render(<PageProgressList pagesTotal={1} pagesDone={1} currentPage={null} pages={pages} language="sk" />)
    expect(screen.getByText('14,2 s')).toBeInTheDocument()
    expect(screen.getByText('1842')).toBeInTheDocument()
  })

  it('renders a failed row with its translated error code', () => {
    const pages = [page({ index: 0, state: 'failed', error_code: 'MODEL_DECODE_MAX_STEPS', decode_steps: null })]
    render(<PageProgressList pagesTotal={1} pagesDone={0} currentPage={null} pages={pages} language="sk" />)
    expect(screen.getByText('MODEL_DECODE_MAX_STEPS')).toBeInTheDocument()
  })

  it('renders the summary line with current/total/done/failed', () => {
    const pages = [page({ index: 0 }), page({ index: 1, state: 'failed', error_code: 'X', decode_steps: null })]
    render(<PageProgressList pagesTotal={3} pagesDone={1} currentPage={2} pages={pages} language="sk" />)
    expect(screen.getByText('Strana 3 z 3 · hotové 1 · zlyhané 1')).toBeInTheDocument()
  })
})

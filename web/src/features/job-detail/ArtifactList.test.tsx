import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import type { components } from '@/api/schema'
import { ArtifactList } from './ArtifactList'

type ArtifactEntry = components['schemas']['ArtifactEntry']

function artifact(overrides: Partial<ArtifactEntry>): ArtifactEntry {
  return {
    role: 'diagnostic_docx',
    page_index: null,
    url: '/v1/conversions/job-1/artifacts/diagnostic_docx',
    ...overrides,
  }
}

describe('ArtifactList (UX §3.6, keyed by role)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('renders the primary Word download when a diagnostic_docx artifact is present', () => {
    render(<ArtifactList artifacts={[artifact({})]} zipUrl="/v1/conversions/job-1/artifacts.zip" />)
    expect(screen.getByRole('link', { name: 'Stiahnuť Word' })).toHaveAttribute(
      'href',
      '/v1/conversions/job-1/artifacts/diagnostic_docx',
    )
  })

  it('never lists source_preview or docx_preview rows', () => {
    render(
      <ArtifactList
        artifacts={[
          artifact({}),
          artifact({ role: 'source_preview', page_index: 0, url: '/v1/conversions/job-1/artifacts/source_preview/0' }),
          artifact({ role: 'docx_preview', page_index: 0, url: '/v1/conversions/job-1/artifacts/docx_preview/0' }),
        ]}
        zipUrl="/v1/conversions/job-1/artifacts.zip"
      />,
    )
    expect(screen.queryByText(/Náhľad/)).not.toBeInTheDocument()
  })

  it('groups prediction_ir artifacts into a collapsible section keyed by role, counted and page-ordered', async () => {
    render(
      <ArtifactList
        artifacts={[
          artifact({ role: 'prediction_ir', page_index: 1, url: '/v1/conversions/job-1/artifacts/prediction_ir/1' }),
          artifact({ role: 'prediction_ir', page_index: 0, url: '/v1/conversions/job-1/artifacts/prediction_ir/0' }),
        ]}
        zipUrl="/v1/conversions/job-1/artifacts.zip"
      />,
    )
    const trigger = screen.getByText('Prediction IR (JSON, 2 strán)')
    expect(trigger).toBeInTheDocument()
    await userEvent.click(trigger)
    const links = screen.getAllByRole('link', { name: /Strana/ })
    expect(links.map((link) => link.getAttribute('href'))).toEqual([
      '/v1/conversions/job-1/artifacts/prediction_ir/0',
      '/v1/conversions/job-1/artifacts/prediction_ir/1',
    ])
  })

  it('always renders the ZIP download with the given url', () => {
    render(<ArtifactList artifacts={[]} zipUrl="/v1/conversions/job-1/artifacts.zip" />)
    expect(screen.getByRole('link', { name: 'Stiahnuť všetko (ZIP)' })).toHaveAttribute(
      'href',
      '/v1/conversions/job-1/artifacts.zip',
    )
  })
})

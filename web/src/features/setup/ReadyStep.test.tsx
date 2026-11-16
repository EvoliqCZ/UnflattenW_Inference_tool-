import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { ReadyStep, SAMPLE_PDF_URL } from './ReadyStep'

describe('ReadyStep (UX §3.1 step 5)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('links the demo button to "/" with the sample document referenced by URL', () => {
    render(
      <MemoryRouter>
        <ReadyStep />
      </MemoryRouter>,
    )

    const demoLink = screen.getByRole('link', { name: 'Skúsiť ukážkový dokument' })
    expect(demoLink).toHaveAttribute('href', `/?sample=${encodeURIComponent(SAMPLE_PDF_URL)}`)
    expect(screen.getByRole('link', { name: 'Nahrať vlastný súbor' })).toHaveAttribute('href', '/')
  })
})

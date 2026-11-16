import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { ComparisonViewer } from './ComparisonViewer'
import type { ArtifactEntry, DiagnosticOutcome } from './types'

function artifact(overrides: Partial<ArtifactEntry>): ArtifactEntry {
  return { page_index: 0, role: 'source_preview', url: 'x', ...overrides }
}

function outcome(overrides: Partial<DiagnosticOutcome> = {}): DiagnosticOutcome {
  return {
    canonical_ast_produced: true,
    claim_scope: 'diagnostic',
    docx_chain: null,
    docx_produced: true,
    prediction_ir_status: 'sealed',
    reason_codes: [],
    ...overrides,
  }
}

const twoMatchedPages: ArtifactEntry[] = [
  artifact({ page_index: 0, role: 'source_preview', url: '/v1/conversions/j/artifacts/source_preview/0' }),
  artifact({ page_index: 1, role: 'source_preview', url: '/v1/conversions/j/artifacts/source_preview/1' }),
  artifact({ page_index: 0, role: 'docx_preview', url: '/v1/conversions/j/artifacts/docx_preview/0' }),
  artifact({ page_index: 1, role: 'docx_preview', url: '/v1/conversions/j/artifacts/docx_preview/1' }),
]

describe('ComparisonViewer', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should never render a blob: or data: URL anywhere, regardless of mode', () => {
    for (const mode of ['side', 'overlay', 'swipe'] as const) {
      const { container, unmount } = render(
        <ComparisonViewer artifacts={twoMatchedPages} outcome={outcome()} pages={null} initialMode={mode} />,
      )
      expect(container.innerHTML).not.toMatch(/blob:|data:/)
      unmount()
    }
  })

  it('should switch modes via the toggle and report the change through onStateChange', async () => {
    const onStateChange = vi.fn()
    const user = userEvent.setup()
    render(
      <ComparisonViewer artifacts={twoMatchedPages} outcome={outcome()} pages={null} onStateChange={onStateChange} />,
    )

    expect(screen.getAllByRole('img')).toHaveLength(2)

    await user.click(screen.getByRole('radio', { name: 'Posuvník' }))

    expect(onStateChange).toHaveBeenCalledWith({ page: 0, mode: 'swipe' })
    expect(screen.getByRole('slider')).toBeInTheDocument()
  })

  it('should render a mismatch notice and a MissingPreview for the page the shorter side lacks', () => {
    const artifacts: ArtifactEntry[] = [
      artifact({ page_index: 0, role: 'source_preview', url: 's0' }),
      artifact({ page_index: 1, role: 'source_preview', url: 's1' }),
      artifact({ page_index: 0, role: 'docx_preview', url: 'd0' }),
    ]
    render(<ComparisonViewer artifacts={artifacts} outcome={outcome()} pages={null} initialPage={1} />)

    expect(
      screen.getByText('Originál má 2 strán, Word sa vykreslil na 1 strán. Strany sa nemusia zhodovať jedna k jednej.'),
    ).toBeInTheDocument()
    expect(screen.getByText('Word nemá stranu 2.')).toBeInTheDocument()
  })

  it('should show docxNotProduced across every page when docx_produced is false', () => {
    const artifacts: ArtifactEntry[] = [artifact({ page_index: 0, role: 'source_preview', url: 's0' })]
    render(
      <ComparisonViewer
        artifacts={artifacts}
        outcome={outcome({ docx_produced: false, reason_codes: ['DOCX_AST_NOT_STRICT'] })}
        pages={null}
      />,
    )

    expect(screen.getByText('Word nevznikol: AST nespĺňa prísny profil.')).toBeInTheDocument()
  })

  it('should navigate between pages with the navigator and keep the current thumbnail in sync', async () => {
    const user = userEvent.setup()
    render(<ComparisonViewer artifacts={twoMatchedPages} outcome={outcome()} pages={null} />)

    await user.click(screen.getByRole('button', { name: 'Ďalej' }))

    expect(screen.getByText('Strana 2 z 2')).toBeInTheDocument()
    const currentThumbnail = screen.getAllByRole('button', { name: 'Strana 2 z 2' })[0]
    expect(currentThumbnail).toHaveAttribute('aria-current', 'page')
  })

  it('should zoom both panes together via the toolbar (sync lock on by default)', async () => {
    const user = userEvent.setup()
    render(<ComparisonViewer artifacts={twoMatchedPages} outcome={outcome()} pages={null} />)

    await user.click(screen.getByRole('button', { name: 'Priblížiť' }))

    expect(screen.getByTestId('compare-zoom-frame-source').style.getPropertyValue('--compare-zoom')).toBe('1.25')
    expect(screen.getByTestId('compare-zoom-frame-docx').style.getPropertyValue('--compare-zoom')).toBe('1.25')
  })

  it('should keep panes independent once the sync lock is turned off', async () => {
    const user = userEvent.setup()
    render(<ComparisonViewer artifacts={twoMatchedPages} outcome={outcome()} pages={null} />)

    await user.click(screen.getByRole('button', { name: 'Prepojiť posuny' }))
    await user.click(screen.getByRole('button', { name: 'Priblížiť' }))

    expect(screen.getByTestId('compare-zoom-frame-source').style.getPropertyValue('--compare-zoom')).toBe('1.25')
    expect(screen.getByTestId('compare-zoom-frame-docx').style.getPropertyValue('--compare-zoom')).toBe('1')
  })

  it('should only react to shortcut keys while focus is inside the viewer', async () => {
    const user = userEvent.setup()
    render(<ComparisonViewer artifacts={twoMatchedPages} outcome={outcome()} pages={null} />)

    document.body.focus()
    await user.keyboard('w')
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()

    screen.getByRole('region', { name: 'Porovnanie' }).focus()
    await user.keyboard('w')
    expect(screen.getByRole('slider')).toBeInTheDocument()
  })

  it('should exit maximize on Escape', async () => {
    const user = userEvent.setup()
    render(<ComparisonViewer artifacts={twoMatchedPages} outcome={outcome()} pages={null} />)

    const region = screen.getByRole('region', { name: 'Porovnanie' })
    region.focus()
    await user.keyboard('f')
    expect(screen.getByRole('button', { name: 'Obnoviť zobrazenie' })).toBeInTheDocument()

    await user.keyboard('{Escape}')
    expect(screen.getByRole('button', { name: 'Zväčšiť' })).toBeInTheDocument()
  })

  it('should offer a "Listovať spolu/samostatne" toggle only when the page counts differ, with a button navigator per side once separated', async () => {
    const user = userEvent.setup()
    const mismatched: ArtifactEntry[] = [
      artifact({ page_index: 0, role: 'source_preview', url: 's0' }),
      artifact({ page_index: 1, role: 'source_preview', url: 's1' }),
      artifact({ page_index: 0, role: 'docx_preview', url: 'd0' }),
    ]
    render(<ComparisonViewer artifacts={mismatched} outcome={outcome()} pages={null} />)

    const linkToggle = screen.getByRole('switch', { name: 'Prepojiť strany' })
    expect(linkToggle).toBeInTheDocument()

    await user.click(linkToggle)

    expect(screen.getByRole('navigation', { name: 'Strana (originál)' })).toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Strana (Word)' })).toBeInTheDocument()
  })

  it('should never render the linked/separate paging toggle when page counts match', () => {
    render(<ComparisonViewer artifacts={twoMatchedPages} outcome={outcome()} pages={null} />)
    expect(screen.queryByRole('switch', { name: 'Prepojiť strany' })).not.toBeInTheDocument()
  })

  it('should keep every zoom/lock/maximize/help action reachable through a button, not just a key', () => {
    render(<ComparisonViewer artifacts={twoMatchedPages} outcome={outcome()} pages={null} />)

    for (const name of ['Priblížiť', 'Oddialiť', 'Prispôsobiť strane', 'Prispôsobiť šírke', 'Skutočná veľkosť']) {
      expect(screen.getByRole('button', { name })).toBeInTheDocument()
    }
    expect(screen.getByRole('button', { name: 'Prepojiť posuny' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Zväčšiť' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Klávesové skratky' })).toBeInTheDocument()
  })

  it('should wrap the toolbar instead of overflowing on a narrow (phone) layout', () => {
    const { container } = render(<ComparisonViewer artifacts={twoMatchedPages} outcome={outcome()} pages={null} />)
    const toolbar = container.querySelector('.flex-wrap')
    expect(toolbar).not.toBeNull()
  })
})

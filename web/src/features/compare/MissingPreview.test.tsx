import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { MissingPreview } from './MissingPreview'

describe('MissingPreview', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('should render the reason text and no blob:/data: URLs anywhere in the DOM', () => {
    const { container } = render(<MissingPreview reason={{ kind: 'outOfRange', index: 3, side: 'docx' }} />)

    expect(screen.getByText('Word nemá stranu 4.')).toBeInTheDocument()
    expect(container.innerHTML).not.toMatch(/blob:|data:/)
  })

  it('should show a retry button only for loadFailed, and call onRetry when clicked', async () => {
    const onRetry = vi.fn()
    const user = userEvent.setup()
    render(<MissingPreview reason={{ kind: 'loadFailed' }} onRetry={onRetry} />)

    await user.click(screen.getByRole('button', { name: 'Skúsiť znova' }))

    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('should not show a retry button for non-loadFailed reasons even if onRetry is passed', () => {
    render(<MissingPreview reason={{ kind: 'previewFailed' }} onRetry={vi.fn()} />)

    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })
})

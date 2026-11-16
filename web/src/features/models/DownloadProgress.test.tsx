import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { DownloadProgress } from './DownloadProgress'

describe('DownloadProgress (UX §3.7 points 2/6)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('renders a determinate bar with bytes done/total when bytes_total is known', () => {
    render(<DownloadProgress state="downloading" bytesDone={1_000_000} bytesTotal={4_000_000} language="sk" />)

    const bar = screen.getByRole('progressbar')
    expect(bar).toHaveAttribute('aria-valuenow', '25')
    expect(screen.getByText(/Sťahovanie beží vo workeri/)).toBeInTheDocument()
  })

  it('renders an indeterminate bar (no aria-valuenow) when bytes_total is null', () => {
    render(<DownloadProgress state="downloading" bytesDone={null} bytesTotal={null} language="sk" />)

    const bar = screen.getByRole('progressbar')
    expect(bar).not.toHaveAttribute('aria-valuenow')
    expect(bar).toHaveAttribute('data-state', 'indeterminate')
  })

  it('shows the translated error and a "Stiahnuť znova" retry button when failed', async () => {
    const onRetry = vi.fn()
    render(<DownloadProgress state="failed" errorCode="MODEL_CHECKPOINT_UNVERIFIED" language="sk" onRetry={onRetry} />)

    expect(screen.getByText('Kontrolný bod modelu nebol overený')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Stiahnuť znova' }))
    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('falls back to the generic unknown-code text for an unrecognized error_code', () => {
    render(<DownloadProgress state="failed" errorCode="SOMETHING_NEW" language="sk" />)

    expect(screen.getByText(i18n.t('codes.unknown'))).toBeInTheDocument()
  })

  it('renders nothing for a settled state (ready/absent)', () => {
    const { container } = render(<DownloadProgress state="ready" language="sk" />)
    expect(container).toBeEmptyDOMElement()
  })
})

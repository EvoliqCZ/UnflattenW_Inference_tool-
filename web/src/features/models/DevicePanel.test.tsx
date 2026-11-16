import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import i18n from '@/i18n'
import { DevicePanel } from './DevicePanel'

describe('DevicePanel (UX §7.3 "Modely" header)', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('shows the "worker has not reported" text when device is null', () => {
    render(<DevicePanel device={null} now={new Date('2026-09-27T10:00:00Z')} language="sk" />)

    expect(screen.getByText('Worker ešte nenahlásil GPU.')).toBeInTheDocument()
  })

  it('summarizes total/free VRAM and compute capability when device is known', () => {
    render(
      <DevicePanel
        device={{
          compute_capability: [8, 9],
          free_vram_mib: 2_000,
          observed_at: '2026-09-27T09:59:30Z',
          total_vram_mib: 12_000,
        }}
        now={new Date('2026-09-27T10:00:00Z')}
        language="sk"
      />,
    )

    expect(screen.getByText(/compute capability 8\.9/)).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Zariadenie' })).toBeInTheDocument()
  })
})

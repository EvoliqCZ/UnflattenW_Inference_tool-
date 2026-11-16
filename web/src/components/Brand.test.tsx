import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { Brand } from './Brand'

describe('distribution brand', () => {
  it('shows the public 1.0 version using the existing mark', () => {
    render(<Brand />)
    expect(screen.getByText('Unflatten W 1.0')).toBeVisible()
    expect(screen.getByRole('img', { name: 'Unflatten W' })).toHaveAttribute(
      'src',
      expect.stringContaining('unflatten-w-mark'),
    )
  })
})

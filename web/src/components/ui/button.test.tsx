import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { Button } from './button'

// UX §3.7: a disabled action keeps `aria-disabled` (never native `disabled`)
// so it stays focusable and can announce its reason. Tailwind's `disabled:`
// variant only matches the native attribute, so the shared Button carries an
// `aria-disabled:` set of its own -- muted tokens, not-allowed cursor, and a
// neutralizer for every hover utility so nothing changes on hover.
const MUTED = [
  'aria-disabled:cursor-not-allowed',
  'aria-disabled:text-muted-foreground',
  'aria-disabled:hover:opacity-100',
]
const MUTED_BOX = [
  'aria-disabled:border',
  'aria-disabled:border-border',
  'aria-disabled:bg-muted',
  'aria-disabled:hover:bg-muted',
]

const VARIANTS = ['default', 'secondary', 'outline', 'ghost', 'destructive', 'link'] as const

function renderDisabled(variant: (typeof VARIANTS)[number]) {
  render(
    <Button variant={variant} aria-disabled="true">
      Act
    </Button>,
  )
  return screen.getByRole('button', { name: 'Act' })
}

describe('Button aria-disabled styling', () => {
  it.each(VARIANTS.filter((v) => v !== 'link'))('mutes the %s variant with the muted box tokens', (variant) => {
    const button = renderDisabled(variant)
    expect(button).toHaveAttribute('aria-disabled', 'true')
    expect(button).toHaveClass(...MUTED, ...MUTED_BOX)
  })

  it('mutes the link variant as text only (no box, no hover underline)', () => {
    const button = renderDisabled('link')
    expect(button).toHaveClass(
      ...MUTED,
      'aria-disabled:border-0',
      'aria-disabled:bg-transparent',
      'aria-disabled:hover:bg-transparent',
      'aria-disabled:hover:no-underline',
    )
    // tailwind-merge must drop the base box tokens the link variant overrides.
    expect(button).not.toHaveClass('aria-disabled:border')
    expect(button).not.toHaveClass('aria-disabled:bg-muted')
  })

  it.each(VARIANTS)('neutralizes every hover utility of the %s variant', (variant) => {
    const button = renderDisabled(variant)
    const classes = button.className.split(/\s+/)
    // utility family = first dash segment: `bg-secondary` -> `bg`, `opacity-90` -> `opacity`.
    const family = (utility: string) => (utility === 'no-underline' ? 'underline' : (utility.split('-')[0] ?? utility))
    const hoverFamilies = classes.filter((c) => c.startsWith('hover:')).map((c) => family(c.slice('hover:'.length)))
    expect(hoverFamilies.length).toBeGreaterThan(0)
    const neutralized = classes
      .filter((c) => c.startsWith('aria-disabled:hover:'))
      .map((c) => family(c.slice('aria-disabled:hover:'.length)))
    for (const prop of hoverFamilies) {
      expect(neutralized, `hover utility '${prop}' has no aria-disabled:hover: counterpart`).toContain(prop)
    }
  })

  it('keeps an aria-disabled button focusable (no native disabled, no pointer-events-none)', () => {
    const button = renderDisabled('default')
    expect(button).not.toBeDisabled()
    expect(button.className).not.toMatch(/aria-disabled:pointer-events-none/)
    button.focus()
    expect(button).toHaveFocus()
  })

  it('keeps the native disabled styles unchanged', () => {
    render(<Button disabled>Off</Button>)
    expect(screen.getByRole('button', { name: 'Off' })).toHaveClass(
      'disabled:pointer-events-none',
      'disabled:opacity-50',
    )
  })
})

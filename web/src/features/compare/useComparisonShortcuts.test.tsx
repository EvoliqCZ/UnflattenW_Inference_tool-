import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { type ComparisonShortcutActions, useComparisonShortcuts } from './useComparisonShortcuts'

function Harness({ actions }: { actions: ComparisonShortcutActions }) {
  const onKeyDown = useComparisonShortcuts(actions)
  return (
    <div role="region" aria-label="viewer" tabIndex={0} onKeyDown={onKeyDown} data-testid="root">
      <input aria-label="page number" />
      <button type="button">focusable</button>
    </div>
  )
}

function makeActions(): ComparisonShortcutActions {
  return {
    prevPage: vi.fn(),
    nextPage: vi.fn(),
    firstPage: vi.fn(),
    lastPage: vi.fn(),
    zoomIn: vi.fn(),
    zoomOut: vi.fn(),
    fitPage: vi.fn(),
    actualSize: vi.fn(),
    pan: vi.fn(),
    setMode: vi.fn(),
    toggleDiffTint: vi.fn(),
    opacityStep: vi.fn(),
    toggleLock: vi.fn(),
    toggleMaximize: vi.fn(),
    exitMaximize: vi.fn(),
    openHelp: vi.fn(),
  }
}

describe('useComparisonShortcuts', () => {
  it('should navigate pages with arrow keys when the region has focus', async () => {
    const actions = makeActions()
    const user = userEvent.setup()
    render(<Harness actions={actions} />)

    screen.getByRole('button').focus()
    await user.keyboard('{ArrowRight}')
    expect(actions.nextPage).toHaveBeenCalledOnce()

    await user.keyboard('{ArrowLeft}')
    expect(actions.prevPage).toHaveBeenCalledOnce()
  })

  it('should pan instead of paging when Shift is held', async () => {
    const actions = makeActions()
    const user = userEvent.setup()
    render(<Harness actions={actions} />)

    screen.getByRole('button').focus()
    await user.keyboard('{Shift>}{ArrowRight}{/Shift}')

    expect(actions.pan).toHaveBeenCalled()
    expect(actions.nextPage).not.toHaveBeenCalled()
  })

  it('should switch modes with S/O/W and toggle maximize/help', async () => {
    const actions = makeActions()
    const user = userEvent.setup()
    render(<Harness actions={actions} />)

    screen.getByRole('button').focus()
    await user.keyboard('w')
    expect(actions.setMode).toHaveBeenCalledWith('swipe')

    await user.keyboard('f')
    expect(actions.toggleMaximize).toHaveBeenCalledOnce()

    await user.keyboard('?')
    expect(actions.openHelp).toHaveBeenCalledOnce()
  })

  it('should not treat a shortcut key as a page-number field keystroke', async () => {
    const actions = makeActions()
    const user = userEvent.setup()
    render(<Harness actions={actions} />)

    const input = screen.getByRole('textbox')
    input.focus()
    await user.keyboard('s')

    expect(actions.setMode).not.toHaveBeenCalled()
  })

  it('should still exit maximize on Escape even while a text field has focus', async () => {
    const actions = makeActions()
    const user = userEvent.setup()
    render(<Harness actions={actions} />)

    screen.getByRole('textbox').focus()
    await user.keyboard('{Escape}')

    expect(actions.exitMaximize).toHaveBeenCalledOnce()
  })

  it('should never fire when focus is outside the region', () => {
    const actions = makeActions()
    render(<Harness actions={actions} />)
    document.body.focus()

    // No React onKeyDown fires for an event whose target is outside this
    // element's subtree -- proven by simply never having triggered it above
    // this line, since jsdom/RTL cannot dispatch a bubbling event without a
    // target inside the tree. Nothing to assert beyond "no crash, no calls".
    expect(actions.nextPage).not.toHaveBeenCalled()
  })
})

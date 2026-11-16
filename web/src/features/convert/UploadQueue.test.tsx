import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { UploadQueue } from './UploadQueue'
import type { SelectedFile } from './types'

function entry(name: string, kind: SelectedFile['kind'] = 'image'): SelectedFile {
  return { id: name, file: new File(['x'], name, { type: kind === 'pdf' ? 'application/pdf' : 'image/png' }), kind }
}

function nth(elements: readonly HTMLElement[], index: number): HTMLElement {
  const element = elements[index]
  if (element === undefined) {
    throw new Error(`expected an element at index ${String(index)}`)
  }
  return element
}

beforeEach(async () => {
  await i18n.changeLanguage('sk')
  vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue({ width: 100, height: 100, close: vi.fn() }))
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('UploadQueue (image ordering, UX §4.4)', () => {
  it('renders files in the given order with 1-based position numbers', () => {
    const files = [entry('a.png'), entry('b.png'), entry('c.png')]
    render(<UploadQueue files={files} onReorder={vi.fn()} onRemove={vi.fn()} language="sk" />)

    const items = screen.getAllByRole('listitem')
    expect(items.map((item) => item.textContent)).toEqual([
      expect.stringContaining('a.png'),
      expect.stringContaining('b.png'),
      expect.stringContaining('c.png'),
    ])
    expect(items[0]).toHaveTextContent('1')
    expect(items[1]).toHaveTextContent('2')
  })

  it('moves a file down and re-numbers the queue when "move down" is used', async () => {
    const files = [entry('a.png'), entry('b.png')]
    const onReorder = vi.fn()
    render(<UploadQueue files={files} onReorder={onReorder} onRemove={vi.fn()} language="sk" />)

    await userEvent.click(nth(screen.getAllByRole('button', { name: 'Posunúť nižšie' }), 0))

    expect(onReorder).toHaveBeenCalledWith([files[1], files[0]])
  })

  it('moves a file up when "move up" is used', async () => {
    const files = [entry('a.png'), entry('b.png')]
    const onReorder = vi.fn()
    render(<UploadQueue files={files} onReorder={onReorder} onRemove={vi.fn()} language="sk" />)

    await userEvent.click(nth(screen.getAllByRole('button', { name: 'Posunúť vyššie' }), 1))

    expect(onReorder).toHaveBeenCalledWith([files[1], files[0]])
  })

  it('disables "move up" on the first row and "move down" on the last row', () => {
    const files = [entry('a.png'), entry('b.png')]
    render(<UploadQueue files={files} onReorder={vi.fn()} onRemove={vi.fn()} language="sk" />)

    expect(screen.getAllByRole('button', { name: 'Posunúť vyššie' })[0]).toBeDisabled()
    expect(screen.getAllByRole('button', { name: 'Posunúť nižšie' })[1]).toBeDisabled()
  })

  it('calls onRemove with the row id when its remove button is clicked', async () => {
    const files = [entry('a.png'), entry('b.png')]
    const onRemove = vi.fn()
    render(<UploadQueue files={files} onReorder={vi.fn()} onRemove={onRemove} language="sk" />)

    await userEvent.click(nth(screen.getAllByRole('button', { name: 'Odstrániť' }), 0))

    expect(onRemove).toHaveBeenCalledWith('a.png')
  })

  it('renders a document icon (not a thumbnail) for a PDF row', () => {
    const files = [entry('a.pdf', 'pdf')]
    const { container } = render(<UploadQueue files={files} onReorder={vi.fn()} onRemove={vi.fn()} language="sk" />)

    expect(container.querySelector('canvas')).not.toBeInTheDocument()
  })
})

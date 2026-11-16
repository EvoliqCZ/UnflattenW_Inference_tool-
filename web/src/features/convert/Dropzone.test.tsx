import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import i18n from '@/i18n'
import { Dropzone } from './Dropzone'

describe('Dropzone', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('sk')
  })

  it('renders a real button that opens the file picker, and a still-accessible hidden input', () => {
    render(<Dropzone onFilesSelected={vi.fn()} />)

    expect(screen.getByRole('button', { name: 'Vybrať súbory' })).toBeInTheDocument()
    expect(screen.getByLabelText('Vybrať súbory')).toHaveAttribute('type', 'file')
  })

  it('calls onFilesSelected with the chosen files', async () => {
    const onFilesSelected = vi.fn()
    render(<Dropzone onFilesSelected={onFilesSelected} />)
    const file = new File(['x'], 'a.pdf', { type: 'application/pdf' })

    await userEvent.upload(screen.getByLabelText('Vybrať súbory'), file)

    expect(onFilesSelected).toHaveBeenCalledWith([file])
  })

  it('calls onFilesSelected with dropped files', () => {
    const onFilesSelected = vi.fn()
    render(<Dropzone onFilesSelected={onFilesSelected} />)
    const file = new File(['x'], 'a.png', { type: 'image/png' })
    const dropTarget = screen.getByText('Sem presuň PDF alebo obrázky').closest('div') as HTMLElement

    fireDrop(dropTarget, [file])

    expect(onFilesSelected).toHaveBeenCalledWith([file])
  })

  it('disables the button and shows the reason when the service is not ready', () => {
    render(<Dropzone onFilesSelected={vi.fn()} disabled disabledReason="Služba nie je pripravená." />)

    expect(screen.getByRole('button', { name: 'Vybrať súbory' })).toBeDisabled()
    expect(screen.getByText('Služba nie je pripravená.')).toBeInTheDocument()
  })
})

function fireDrop(element: HTMLElement, files: File[]): void {
  const dataTransfer = {
    files,
    items: files.map((file) => ({ kind: 'file', type: file.type })),
  }
  const event = new Event('drop', { bubbles: true, cancelable: true }) as unknown as DragEvent & {
    dataTransfer: typeof dataTransfer
  }
  Object.defineProperty(event, 'dataTransfer', { value: dataTransfer })
  element.dispatchEvent(event)
}

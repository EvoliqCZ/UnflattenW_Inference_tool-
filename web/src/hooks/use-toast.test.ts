import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { dismissToast, removeToast, toast, useToasts } from './use-toast'

describe('use-toast', () => {
  it('should add a toast that a subscribed hook observes', () => {
    const { result } = renderHook(() => useToasts())
    const before = result.current.length

    let id = ''
    act(() => {
      id = toast({ title: 'Hotovo' })
    })

    expect(result.current).toHaveLength(before + 1)
    expect(result.current.at(-1)).toMatchObject({ id, title: 'Hotovo', open: true })
  })

  it('should mark a toast closed without removing it', () => {
    const { result } = renderHook(() => useToasts())
    let id = ''
    act(() => {
      id = toast({ title: 'Zlyhalo', variant: 'destructive' })
    })

    act(() => {
      dismissToast(id)
    })

    expect(result.current.find((t) => t.id === id)).toMatchObject({ open: false })
  })

  it('should remove a toast entirely', () => {
    const { result } = renderHook(() => useToasts())
    let id = ''
    act(() => {
      id = toast({ title: 'docasny' })
    })

    act(() => {
      removeToast(id)
    })

    expect(result.current.find((t) => t.id === id)).toBeUndefined()
  })
})

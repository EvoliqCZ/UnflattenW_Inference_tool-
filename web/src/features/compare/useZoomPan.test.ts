import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { useZoomPan } from './useZoomPan'

describe('useZoomPan', () => {
  it('should zoom both panes together when linked', () => {
    const { result } = renderHook(() => useZoomPan(true))

    act(() => {
      result.current.zoomIn('source')
    })

    expect(result.current.source.zoom).toBe(125)
    expect(result.current.docx.zoom).toBe(125)
  })

  it('should keep panes independent when unlinked', () => {
    const { result } = renderHook(() => useZoomPan(false))

    act(() => {
      result.current.zoomIn('source')
    })

    expect(result.current.source.zoom).toBe(125)
    expect(result.current.docx.zoom).toBe(100)
  })

  it('should clamp zoom to the 25-800 range', () => {
    const { result } = renderHook(() => useZoomPan(true))

    act(() => {
      for (let i = 0; i < 40; i += 1) result.current.zoomIn('source')
    })
    expect(result.current.source.zoom).toBe(800)

    act(() => {
      for (let i = 0; i < 40; i += 1) result.current.zoomOut('source')
    })
    expect(result.current.source.zoom).toBe(25)
  })

  it('should pan by keyboard steps of 10% and reset back to the baseline', () => {
    const { result } = renderHook(() => useZoomPan(true))

    act(() => {
      result.current.panByKey('source', 1, -1)
    })
    expect(result.current.source.panX).toBe(10)
    expect(result.current.source.panY).toBe(-10)

    act(() => {
      result.current.reset('source')
    })
    expect(result.current.source).toEqual({ zoom: 100, panX: 0, panY: 0 })
  })
})

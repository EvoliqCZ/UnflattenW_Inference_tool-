import { describe, expect, it } from 'vitest'
import { validateSelection, validateSelectionSync, type SelectionLimits } from './validateSelection'

const limits: SelectionLimits = {
  maxPages: 2,
  maxUploadBytes: 1000,
  acceptedMediaTypes: ['application/pdf', 'image/png', 'image/jpeg'],
}

function file(name: string, type: string, size = 10): File {
  return new File([new Uint8Array(size)], name, { type })
}

describe('validateSelectionSync (UX §4.3, limits from readiness)', () => {
  it('accepts a single PDF within limits', () => {
    expect(validateSelectionSync([file('a.pdf', 'application/pdf')], limits)).toBeNull()
  })

  it('accepts images within the page and byte limits', () => {
    expect(validateSelectionSync([file('a.png', 'image/png'), file('b.png', 'image/png')], limits)).toBeNull()
  })

  it('rejects an unsupported media type before any other check', () => {
    expect(validateSelectionSync([file('a.gif', 'image/gif')], limits)).toEqual({
      code: 'type',
      params: { name: 'a.gif' },
    })
  })

  it('rejects mixing a PDF with images', () => {
    expect(validateSelectionSync([file('a.pdf', 'application/pdf'), file('b.png', 'image/png')], limits)).toEqual({
      code: 'mixed',
    })
  })

  it('rejects more than one PDF', () => {
    expect(validateSelectionSync([file('a.pdf', 'application/pdf'), file('b.pdf', 'application/pdf')], limits)).toEqual(
      { code: 'onePdf' },
    )
  })

  it('rejects more images than limits.maxPages', () => {
    const files = [file('a.png', 'image/png'), file('b.png', 'image/png'), file('c.png', 'image/png')]
    expect(validateSelectionSync(files, limits)).toEqual({ code: 'tooManyImages', params: { n: 3, max: 2 } })
  })

  it('rejects a selection whose total size exceeds limits.maxUploadBytes', () => {
    const files = [file('a.png', 'image/png', 600), file('b.png', 'image/png', 600)]
    expect(validateSelectionSync(files, limits)).toEqual({
      code: 'tooLarge',
      params: { size: 1200, max: 1000 },
    })
  })

  it('is all-or-nothing: one bad file among good ones still rejects the whole selection', () => {
    const files = [file('good.png', 'image/png'), file('bad.gif', 'image/gif')]
    const result = validateSelectionSync(files, limits)
    expect(result?.code).toBe('type')
  })
})

describe('validateSelection (async decode check)', () => {
  it('accepts images that decode successfully', async () => {
    const result = await validateSelection([file('a.png', 'image/png')], limits, () => Promise.resolve(true))
    expect(result).toBeNull()
  })

  it('rejects an image that fails to decode', async () => {
    const result = await validateSelection([file('a.png', 'image/png')], limits, () => Promise.resolve(false))
    expect(result).toEqual({ code: 'undecodable', params: { name: 'a.png' } })
  })

  it('never calls the decode check for a PDF', async () => {
    let called = false
    const result = await validateSelection([file('a.pdf', 'application/pdf')], limits, () => {
      called = true
      return Promise.resolve(false)
    })
    expect(result).toBeNull()
    expect(called).toBe(false)
  })

  it('short-circuits on the sync error and never runs the decode check', async () => {
    let called = false
    const result = await validateSelection([file('a.gif', 'image/gif')], limits, () => {
      called = true
      return Promise.resolve(true)
    })
    expect(result).toEqual({ code: 'type', params: { name: 'a.gif' } })
    expect(called).toBe(false)
  })
})

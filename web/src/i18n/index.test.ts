import { describe, expect, it, vi } from 'vitest'
import { detectLanguage, LANGUAGE_STORAGE_KEY, readStoredLanguage, setLanguage } from './index'

describe('detectLanguage', () => {
  it('picks sk for any sk* navigator language', () => {
    expect(detectLanguage('sk-SK')).toBe('sk')
    expect(detectLanguage('sk')).toBe('sk')
  })

  it('falls back to en otherwise', () => {
    expect(detectLanguage('en-GB')).toBe('en')
    expect(detectLanguage('de-DE')).toBe('en')
  })
})

describe('readStoredLanguage', () => {
  it('returns the stored language when valid', () => {
    const storage = { getItem: () => 'sk' }
    expect(readStoredLanguage(storage)).toBe('sk')
  })

  it('returns null when nothing or something invalid is stored', () => {
    expect(readStoredLanguage({ getItem: () => null })).toBeNull()
    expect(readStoredLanguage({ getItem: () => 'fr' })).toBeNull()
  })
})

describe('setLanguage', () => {
  it('persists the language and updates <html lang>', () => {
    setLanguage('en')
    expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('en')
    expect(document.documentElement.lang).toBe('en')
    setLanguage('sk')
    expect(window.localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('sk')
    expect(document.documentElement.lang).toBe('sk')
  })
})

it('treats inaccessible storage as no saved preference', () => {
  expect(
    readStoredLanguage({
      getItem: () => {
        throw new DOMException('Denied')
      },
    }),
  ).toBeNull()
})

it('still applies the active language when persistence is denied', () => {
  const spy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
    throw new DOMException('Denied')
  })
  try {
    expect(() => {
      setLanguage('en')
    }).not.toThrow()
    expect(document.documentElement.lang).toBe('en')
  } finally {
    spy.mockRestore()
  }
})

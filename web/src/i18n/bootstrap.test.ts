import { afterEach, beforeEach, expect, it, vi } from 'vitest'

beforeEach(() => {
  localStorage.clear()
  vi.resetModules()
})
afterEach(() => {
  vi.restoreAllMocks()
})

it('uses a saved language before the browser language on a new visit', async () => {
  localStorage.setItem('unflatten.lang', 'en')
  vi.spyOn(window.navigator, 'language', 'get').mockReturnValue('sk-SK')
  const { default: fresh } = await import('./index')
  expect(fresh.resolvedLanguage).toBe('en')
  expect(document.documentElement.lang).toBe('en')
})

it('selects Slovak for a Slovak browser without a saved preference', async () => {
  vi.spyOn(window.navigator, 'language', 'get').mockReturnValue('sk-SK')
  const { default: fresh } = await import('./index')
  expect(fresh.resolvedLanguage).toBe('sk')
})

it('can start in English when the browser denies the storage property itself', async () => {
  vi.spyOn(window, 'localStorage', 'get').mockImplementation(() => {
    throw new DOMException('Denied', 'SecurityError')
  })
  vi.spyOn(window.navigator, 'language', 'get').mockReturnValue('de-DE')
  const { default: fresh } = await import('./index')
  expect(fresh.resolvedLanguage).toBe('en')
  expect(document.documentElement.lang).toBe('en')
})

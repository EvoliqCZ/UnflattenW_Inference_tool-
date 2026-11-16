import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// `public/theme-init.js` is copied verbatim to `dist/` root by Vite (not
// bundled), so it cannot be `import`ed as a module. This test loads and
// executes the exact file that ships, against a real jsdom `document`, to
// prove the pre-render bootstrap script behaves like `src/lib/theme.ts`.
const scriptPath = resolve(import.meta.dirname, '../../public/theme-init.js')
const scriptSource = readFileSync(scriptPath, 'utf-8')

function runThemeInit(): void {
  // eslint-disable-next-line @typescript-eslint/no-implied-eval, @typescript-eslint/no-unsafe-call -- executing our own static asset under test.
  new Function(scriptSource)()
}

function mockMatchMedia(prefersDark: boolean): void {
  vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches: prefersDark }) as unknown)
}

describe('theme-init.js', () => {
  beforeEach(() => {
    window.localStorage.clear()
    document.documentElement.classList.remove('dark')
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('should add the dark class when stored preference is dark', () => {
    window.localStorage.setItem('unflatten-theme', 'dark')
    mockMatchMedia(false)

    runThemeInit()

    expect(document.documentElement.classList.contains('dark')).toBe(true)
  })

  it('should not add the dark class when stored preference is light', () => {
    window.localStorage.setItem('unflatten-theme', 'light')
    mockMatchMedia(true)

    runThemeInit()

    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })

  it('should follow the system preference when nothing is stored', () => {
    mockMatchMedia(true)

    runThemeInit()

    expect(document.documentElement.classList.contains('dark')).toBe(true)
  })

  it('should treat an invalid stored value as system', () => {
    window.localStorage.setItem('unflatten-theme', 'sepia')
    mockMatchMedia(false)

    runThemeInit()

    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })

  it('should default to light when matchMedia is unavailable', () => {
    vi.stubGlobal('matchMedia', undefined)

    runThemeInit()

    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })

  it('should default to system when localStorage access throws', () => {
    const getItem = vi.spyOn(window.localStorage.__proto__ as Storage, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    mockMatchMedia(true)

    runThemeInit()

    expect(document.documentElement.classList.contains('dark')).toBe(true)
    getItem.mockRestore()
  })
})

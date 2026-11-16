import { describe, expect, it, vi } from 'vitest'
import {
  applyEffectiveTheme,
  isThemePreference,
  readStoredThemePreference,
  resolveEffectiveTheme,
  writeStoredThemePreference,
} from './theme'

describe('isThemePreference', () => {
  it('should accept the three known preferences', () => {
    expect(isThemePreference('system')).toBe(true)
    expect(isThemePreference('light')).toBe(true)
    expect(isThemePreference('dark')).toBe(true)
  })

  it('should reject anything else', () => {
    expect(isThemePreference('sepia')).toBe(false)
    expect(isThemePreference(null)).toBe(false)
    expect(isThemePreference(undefined)).toBe(false)
  })
})

describe('readStoredThemePreference', () => {
  it('should return the stored value when valid', () => {
    const storage = { getItem: vi.fn().mockReturnValue('dark') }
    expect(readStoredThemePreference(storage)).toBe('dark')
  })

  it('should fall back to system when missing', () => {
    const storage = { getItem: vi.fn().mockReturnValue(null) }
    expect(readStoredThemePreference(storage)).toBe('system')
  })

  it('should fall back to system when the stored value is invalid', () => {
    const storage = { getItem: vi.fn().mockReturnValue('sepia') }
    expect(readStoredThemePreference(storage)).toBe('system')
  })
})

describe('writeStoredThemePreference', () => {
  it('should write under the theme storage key', () => {
    const storage = { setItem: vi.fn() }
    writeStoredThemePreference(storage, 'light')
    expect(storage.setItem).toHaveBeenCalledWith('unflatten-theme', 'light')
  })
})

describe('resolveEffectiveTheme', () => {
  it('should resolve system to dark when the OS prefers dark', () => {
    expect(resolveEffectiveTheme('system', true)).toBe('dark')
  })

  it('should resolve system to light when the OS prefers light', () => {
    expect(resolveEffectiveTheme('system', false)).toBe('light')
  })

  it('should ignore the OS preference for an explicit choice', () => {
    expect(resolveEffectiveTheme('light', true)).toBe('light')
    expect(resolveEffectiveTheme('dark', false)).toBe('dark')
  })
})

describe('applyEffectiveTheme', () => {
  it('should add the dark class for dark', () => {
    const classList = { toggle: vi.fn() }
    applyEffectiveTheme({ classList: classList as unknown as DOMTokenList }, 'dark')
    expect(classList.toggle).toHaveBeenCalledWith('dark', true)
  })

  it('should remove the dark class for light', () => {
    const classList = { toggle: vi.fn() }
    applyEffectiveTheme({ classList: classList as unknown as DOMTokenList }, 'light')
    expect(classList.toggle).toHaveBeenCalledWith('dark', false)
  })
})

import { useCallback, useEffect, useState } from 'react'
import {
  applyEffectiveTheme,
  readStoredThemePreference,
  resolveEffectiveTheme,
  writeStoredThemePreference,
  type ThemePreference,
} from '@/lib/theme'

/**
 * Single consumer today (`UserMenu`'s theme toggle) -- a plain hook, not a
 * context provider, per Karpathy simplicity-first: nothing else needs shared
 * theme state yet. `public/theme-init.js` already applied the initial class
 * before React mounted; this hook keeps it in sync with user changes and
 * with the OS preference while `preference === 'system'`.
 */
export function useThemePreference(): [ThemePreference, (next: ThemePreference) => void] {
  const [preference, setPreference] = useState<ThemePreference>(() => {
    try {
      return readStoredThemePreference(window.localStorage)
    } catch {
      return 'system'
    }
  })

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const apply = () => {
      applyEffectiveTheme(document.documentElement, resolveEffectiveTheme(preference, media.matches))
    }
    apply()
    media.addEventListener('change', apply)
    return () => {
      media.removeEventListener('change', apply)
    }
  }, [preference])

  const setThemePreference = useCallback((next: ThemePreference) => {
    setPreference(next)
    try {
      writeStoredThemePreference(window.localStorage, next)
    } catch {
      // Storage can be denied or full; keep the active theme usable in this tab.
    }
  }, [])

  return [preference, setThemePreference]
}

// UX §8.1 / design §6.2: theme is `system | light | dark`, persisted in
// localStorage, and applied as a `dark` class on `<html>`. This module holds
// the pure decision logic shared by `public/theme-init.js` (the synchronous
// pre-render bootstrap, duplicated there because static `public/` files are
// not bundled) and the in-app `ThemeProvider` toggle. Keeping the logic pure
// and framework-free makes both call sites testable without a DOM.

export const THEME_STORAGE_KEY = 'unflatten-theme'

export type ThemePreference = 'system' | 'light' | 'dark'
export type EffectiveTheme = 'light' | 'dark'

const PREFERENCES: readonly ThemePreference[] = ['system', 'light', 'dark']

export function isThemePreference(value: unknown): value is ThemePreference {
  return typeof value === 'string' && (PREFERENCES as readonly string[]).includes(value)
}

/** Reads the stored preference, falling back to `system` for anything missing or invalid. */
export function readStoredThemePreference(storage: Pick<Storage, 'getItem'>): ThemePreference {
  const raw = storage.getItem(THEME_STORAGE_KEY)
  return isThemePreference(raw) ? raw : 'system'
}

export function writeStoredThemePreference(storage: Pick<Storage, 'setItem'>, preference: ThemePreference): void {
  storage.setItem(THEME_STORAGE_KEY, preference)
}

/** `system` resolves against the OS/browser media query; `light`/`dark` are explicit. */
export function resolveEffectiveTheme(preference: ThemePreference, systemPrefersDark: boolean): EffectiveTheme {
  if (preference === 'system') {
    return systemPrefersDark ? 'dark' : 'light'
  }
  return preference
}

/** Applies (or removes) the `dark` class -- the only thing tokens.css keys off. */
export function applyEffectiveTheme(root: Pick<HTMLElement, 'classList'>, effective: EffectiveTheme): void {
  root.classList.toggle('dark', effective === 'dark')
}

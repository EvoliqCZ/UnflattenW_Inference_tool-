import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import en from './en.json'
import sk from './sk.json'

export const LANGUAGE_STORAGE_KEY = 'unflatten.lang'
export const SUPPORTED_LANGUAGES = ['sk', 'en'] as const
export type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number]

// design §7.5: sk.json is the type source, en.json is checked against it
// (same keys) by i18n-completeness.test.ts.
declare module 'i18next' {
  interface CustomTypeOptions {
    defaultNS: 'translation'
    resources: { translation: typeof sk }
  }
}

/** `navigator.language.startsWith("sk") ? "sk" : "en"` (design §7.5). */
export function detectLanguage(navigatorLanguage: string): SupportedLanguage {
  return navigatorLanguage.startsWith('sk') ? 'sk' : 'en'
}

export function readStoredLanguage(storage: Pick<Storage, 'getItem'>): SupportedLanguage | null {
  try {
    const stored = storage.getItem(LANGUAGE_STORAGE_KEY)
    return stored === 'sk' || stored === 'en' ? stored : null
  } catch {
    return null
  }
}

function initialStoredLanguage(): SupportedLanguage | null {
  try {
    return readStoredLanguage(window.localStorage)
  } catch {
    // Browsers can deny access to the storage property itself.
    return null
  }
}

const initialLanguage: SupportedLanguage = initialStoredLanguage() ?? detectLanguage(window.navigator.language)

void i18n.use(initReactI18next).init({
  resources: {
    sk: { translation: sk },
    en: { translation: en },
  },
  lng: initialLanguage,
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  returnNull: false,
})

/** Keeps `<html lang>` and `localStorage` in sync with the active language. */
export function setLanguage(language: SupportedLanguage): void {
  document.documentElement.lang = language
  void i18n.changeLanguage(language)
  try {
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language)
  } catch {
    // A denied write must not prevent changing this tab's language.
  }
}

document.documentElement.lang = initialLanguage

export default i18n

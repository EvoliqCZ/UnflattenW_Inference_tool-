import { useTranslation } from 'react-i18next'
import { setLanguage, SUPPORTED_LANGUAGES } from '@/i18n'

/** Available before login as well as on every authenticated page. */
export function LanguageSelector() {
  const { t, i18n } = useTranslation()
  return (
    <div role="group" aria-label={t('settings.language')} className="inline-flex gap-1">
      {SUPPORTED_LANGUAGES.map((language) => (
        <button
          key={language}
          type="button"
          lang={language}
          aria-label={language === 'sk' ? 'Slovenčina' : 'English'}
          aria-pressed={i18n.resolvedLanguage === language}
          className="rounded-sm px-2 py-1 hover:text-accent focus-visible:outline-2 focus-visible:outline-accent aria-pressed:bg-secondary aria-pressed:text-foreground"
          onClick={() => {
            setLanguage(language)
          }}
        >
          {language.toUpperCase()}
        </button>
      ))}
    </div>
  )
}

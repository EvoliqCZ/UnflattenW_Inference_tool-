import { useTranslation } from 'react-i18next'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { setLanguage, type SupportedLanguage } from '@/i18n'

/** Language names are shown in their own language (UX §2.3 footer "SK | EN"), not translated. */
const LANGUAGE_LABELS: Record<SupportedLanguage, string> = {
  sk: 'Slovenčina',
  en: 'English',
}

/** UX §6 Nastavenia: language switches immediately, no save step. */
export function LanguageSwitch() {
  const { t, i18n } = useTranslation()
  const current: SupportedLanguage = i18n.language === 'sk' ? 'sk' : 'en'

  function handleChange(value: string): void {
    setLanguage(value as SupportedLanguage)
  }

  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-sm font-medium">{t('settings.language')}</span>
      <RadioGroup value={current} onValueChange={handleChange} className="flex gap-4">
        {(Object.keys(LANGUAGE_LABELS) as SupportedLanguage[]).map((language) => (
          <div key={language} className="flex items-center gap-2">
            <RadioGroupItem value={language} id={`settings-language-${language}`} />
            <Label htmlFor={`settings-language-${language}`}>{LANGUAGE_LABELS[language]}</Label>
          </div>
        ))}
      </RadioGroup>
    </div>
  )
}

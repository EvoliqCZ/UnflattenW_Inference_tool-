import { useTranslation } from 'react-i18next'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { useThemePreference } from '@/hooks/use-theme'
import type { ThemePreference } from '@/lib/theme'

const THEME_OPTIONS: ThemePreference[] = ['system', 'light', 'dark']

/** UX §6 Nastavenia: Podľa systému / Svetlá / Tmavá, applied immediately. */
export function ThemeSwitch() {
  const { t } = useTranslation()
  const [theme, setTheme] = useThemePreference()

  function handleChange(value: string): void {
    try {
      setTheme(value as ThemePreference)
    } catch {
      // localStorage.setItem can throw (private browsing, quota, disabled
      // storage); swallow so one denied write doesn't crash the settings
      // screen -- the theme change is simply not persisted this time.
    }
  }

  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-sm font-medium">{t('settings.theme.label')}</span>
      <RadioGroup value={theme} onValueChange={handleChange} className="flex gap-4">
        {THEME_OPTIONS.map((option) => (
          <div key={option} className="flex items-center gap-2">
            <RadioGroupItem value={option} id={`settings-theme-${option}`} />
            <Label htmlFor={`settings-theme-${option}`}>{t(`settings.theme.${option}`)}</Label>
          </div>
        ))}
      </RadioGroup>
    </div>
  )
}

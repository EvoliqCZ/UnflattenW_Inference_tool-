import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router'
import { useLogoutMutation } from '@/auth/session'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { AboutService } from './AboutService'
import { ChangePasswordDialog } from './ChangePasswordDialog'
import { DefaultModelSelect } from './DefaultModelSelect'
import { LanguageSwitch } from './LanguageSwitch'
import { ThemeSwitch } from './ThemeSwitch'

// UX §6 "/settings" -- Nastavenia: language/theme (client-side), default
// model (`DefaultModelSelect`), account (change password, logout), about
// (git SHA, API version, session limits, rootful sandbox sentence).
export function SettingsPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const logoutMutation = useLogoutMutation()

  async function handleLogout(): Promise<void> {
    try {
      await logoutMutation.mutateAsync()
    } catch {
      // Ignored: the user lands on /login either way (mirrors `UserMenu`'s
      // logout -- an already-expired session should still take them there).
    } finally {
      void navigate('/login')
    }
  }

  return (
    <main className="mx-auto flex max-w-2xl flex-col gap-6 p-4">
      <h1 className="text-xl font-semibold">{t('nav.settings')}</h1>

      <section aria-label={t('settings.language')}>
        <LanguageSwitch />
      </section>

      <section aria-label={t('settings.theme.label')}>
        <ThemeSwitch />
      </section>

      <section aria-label={t('settings.defaultModel.label')}>
        <DefaultModelSelect />
      </section>

      <Card>
        <CardHeader>
          <CardTitle>{t('settings.account.title')}</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          <ChangePasswordDialog />
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              void handleLogout()
            }}
          >
            {t('nav.logout')}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t('settings.about.title')}</CardTitle>
        </CardHeader>
        <CardContent>
          <AboutService />
        </CardContent>
      </Card>
    </main>
  )
}

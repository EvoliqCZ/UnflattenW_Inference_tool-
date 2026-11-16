import { useTranslation } from 'react-i18next'
import { Monitor, Moon, Settings, Sun } from 'lucide-react'
import { useNavigate } from 'react-router'
import { useLogoutMutation } from '@/auth/session'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useThemePreference } from '@/hooks/use-theme'
import type { ThemePreference } from '@/lib/theme'

/**
 * UX §2.3: `⚙ ▾` dropdown -- Nastavenia, Jazyk, Téma, Odhlásiť. Language is
 * wired to real i18n state in a later task (T39 settings); "Odhlásiť" (T29)
 * calls `DELETE /v1/session` and lands on `/login` regardless of the
 * result -- an already-expired session should still take the user there.
 */
export function UserMenu() {
  const { t } = useTranslation()
  const [theme, setTheme] = useThemePreference()
  const navigate = useNavigate()
  const logoutMutation = useLogoutMutation()

  async function handleLogout(): Promise<void> {
    try {
      await logoutMutation.mutateAsync()
    } catch {
      // Ignored: the user lands on /login either way -- a failed DELETE
      // (already-expired session, network hiccup) shouldn't trap them here.
    } finally {
      void navigate('/login')
    }
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t('nav.settingsAndTheme')}
        className="flex size-9 items-center justify-center rounded-sm text-foreground hover:bg-secondary"
      >
        <Settings className="size-4" aria-hidden="true" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem
          onSelect={() => {
            void navigate('/settings')
          }}
        >
          {t('nav.settings')}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuLabel>{t('settings.theme.label')}</DropdownMenuLabel>
        <DropdownMenuRadioGroup
          value={theme}
          onValueChange={(value) => {
            setTheme(value as ThemePreference)
          }}
        >
          <DropdownMenuRadioItem value="system">
            <Monitor className="mr-2 size-4" aria-hidden="true" />
            {t('settings.theme.system')}
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="light">
            <Sun className="mr-2 size-4" aria-hidden="true" />
            {t('settings.theme.light')}
          </DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="dark">
            <Moon className="mr-2 size-4" aria-hidden="true" />
            {t('settings.theme.dark')}
          </DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onSelect={() => {
            void handleLogout()
          }}
        >
          {t('nav.logout')}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

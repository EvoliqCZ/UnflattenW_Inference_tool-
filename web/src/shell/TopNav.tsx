import { useTranslation } from 'react-i18next'
import { Menu } from 'lucide-react'
import { Brand } from '@/components/Brand'
import { CompanyBrand } from '@/components/CompanyBrand'
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { UserMenu } from './UserMenu'

// UX §2.1 routes.json paths. Plain <a> for now -- `react-router` `NavLink`
// takes over once the router lands in T26; these hrefs already match
// `routes.json` so that swap is a like-for-like replacement.
const NAV_LINKS = [
  { href: '/', label: 'nav.convert' },
  { href: '/jobs', label: 'nav.jobs' },
  { href: '/models', label: 'nav.models' },
] as const

function NavLinks({ className }: { className?: string }) {
  const { t } = useTranslation()
  return (
    <ul className={className}>
      {NAV_LINKS.map((link) => (
        <li key={link.href}>
          <a href={link.href} className="text-sm font-medium text-foreground hover:text-accent">
            {t(link.label)}
          </a>
        </li>
      ))}
    </ul>
  )
}

/** UX §2.3: 56px shell header (48px < 768), nav collapses into a `Sheet` below 768px. */
export function TopNav() {
  const { t } = useTranslation()
  return (
    <header className="flex h-14 items-center gap-4 border-b border-border px-4 max-sm:h-12">
      <Sheet>
        <SheetTrigger
          aria-label={t('nav.openNavigation')}
          className="flex size-9 items-center justify-center rounded-sm sm:hidden"
        >
          <Menu className="size-5" aria-hidden="true" />
        </SheetTrigger>
        <SheetContent side="left">
          <SheetTitle>Unflatten W</SheetTitle>
          <NavLinks className="mt-4 flex flex-col gap-4" />
        </SheetContent>
      </Sheet>

      <Brand markClassName="h-7" textClassName="font-mono text-sm font-semibold" />

      <nav aria-label={t('nav.mainNavigation')} className="hidden sm:block">
        <NavLinks className="flex items-center gap-6" />
      </nav>

      <div className="ml-auto flex items-center gap-2">
        <CompanyBrand compact className="hidden md:inline-flex" />
        <UserMenu />
      </div>
    </header>
  )
}

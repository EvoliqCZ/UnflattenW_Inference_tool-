import type { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { BrandFooter } from '@/components/BrandFooter'
import { Toaster } from '@/components/ui/toaster'
import { TopNav } from './TopNav'

/**
 * UX §2.3 shell layout for authenticated routes: nav, content
 * (max-width 1200px), footer. `Toaster` and the skip link are mounted once
 * here per the UX §7.2 component tree.
 */
export function AppShell({ children }: { children: ReactNode }) {
  const { t } = useTranslation()
  return (
    <div className="flex min-h-dvh flex-col bg-bg text-foreground">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-2 focus:top-10 focus:z-50 focus:rounded-sm focus:bg-accent focus:px-3 focus:py-2 focus:text-accent-ink"
      >
        {t('nav.skipToContent')}
      </a>
      <TopNav />
      <main id="main-content" className="mx-auto w-full max-w-[1200px] flex-1 px-4 py-8 sm:px-6">
        {children}
      </main>
      <BrandFooter />
      <Toaster />
    </div>
  )
}

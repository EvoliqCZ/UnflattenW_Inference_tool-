import type { ReactNode } from 'react'
import { Brand } from '@/components/Brand'
import { BrandFooter } from '@/components/BrandFooter'
import { Toaster } from '@/components/ui/toaster'

/** UX §2.3 / §7.2: layout for `/login` and `/setup` -- no TopNav (no session yet). */
export function AuthShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-dvh flex-col bg-bg text-foreground">
      <main className="mx-auto flex w-full max-w-md flex-1 flex-col justify-center px-4 py-8">
        <Brand
          className="mb-6 justify-center gap-3"
          markClassName="h-12"
          textClassName="font-mono text-lg font-semibold"
        />
        {children}
      </main>
      <BrandFooter />
      <Toaster />
    </div>
  )
}

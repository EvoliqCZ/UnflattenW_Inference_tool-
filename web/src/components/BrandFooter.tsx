import { CompanyBrand } from './CompanyBrand'
import { LanguageSelector } from './LanguageSelector'

export function BrandFooter() {
  return (
    <footer className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2 border-t border-border px-4 py-3 text-center font-mono text-xs text-ink-muted">
      <span>Unflatten W 1.0</span>
      <CompanyBrand />
      <LanguageSelector />
    </footer>
  )
}

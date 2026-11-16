import { useTranslation } from 'react-i18next'
import darkLogo from '@/assets/brand/evoliq-logo-dark.png'
import whiteLogo from '@/assets/brand/evoliq-logo-white.png'
import { cn } from '@/lib/utils'

/** Original corporate artwork, with a readable variant for each theme. */
export function CompanyBrand({ compact = false, className }: { compact?: boolean; className?: string }) {
  const { t, i18n } = useTranslation()
  const imageClass = compact ? 'h-7 w-auto' : 'h-9 w-auto'
  return (
    <a
      href={i18n.resolvedLanguage === 'sk' ? 'https://www.evoliq.cz/sk' : 'https://www.evoliq.cz/en'}
      target="_blank"
      rel="noopener noreferrer"
      aria-label="EVOLIQ s.r.o."
      title={t('brand.companyWebsite')}
      className={cn(
        'inline-flex shrink-0 items-center rounded-sm focus-visible:outline-2 focus-visible:outline-accent',
        className,
      )}
    >
      <img src={darkLogo} width={1011} height={348} alt="EVOLIQ" className={cn(imageClass, 'dark:hidden')} />
      <img src={whiteLogo} width={933} height={313} alt="EVOLIQ" className={cn(imageClass, 'hidden dark:block')} />
    </a>
  )
}

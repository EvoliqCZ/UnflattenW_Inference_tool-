import { useTranslation } from 'react-i18next'
import { useModelsQuery } from '@/api/queries'
import { ProblemMessage } from '@/components/ProblemMessage'
import { Skeleton } from '@/components/ui/skeleton'
import type { SupportedLanguage } from '@/i18n'
import { DevicePanel } from './DevicePanel'
import { ModelCard } from './ModelCard'

/** UX §3.7/§7.2 "/models": device panel + a grid of `ModelCard`, one per variant. */
export function ModelsPage() {
  const { t, i18n } = useTranslation()
  const language = i18n.language as SupportedLanguage
  const query = useModelsQuery()
  const models = query.data

  return (
    <main className="mx-auto flex max-w-5xl flex-col gap-4 p-4">
      <h1 className="text-xl font-semibold">{t('models.title')}</h1>

      {query.isPending ? (
        <div className="flex flex-col gap-4">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-40 w-full" />
        </div>
      ) : null}

      {query.isError ? <ProblemMessage problem={{ code: 'INTERNAL_ERROR' }} context="page" /> : null}

      {models ? (
        <>
          <DevicePanel device={models.device} now={new Date()} language={language} />
          {models.variants.length === 0 ? (
            <p className="text-sm text-muted-foreground">{t('models.empty')}</p>
          ) : (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
              {models.variants.map((variant) => (
                <ModelCard
                  key={variant.variant_id}
                  variant={variant}
                  resident={models.variants.find((candidate) => candidate.resident) ?? null}
                  device={models.device}
                  language={language}
                />
              ))}
            </div>
          )}
          <p className="text-xs text-muted-foreground">{t('models.footer')}</p>
        </>
      ) : null}
    </main>
  )
}

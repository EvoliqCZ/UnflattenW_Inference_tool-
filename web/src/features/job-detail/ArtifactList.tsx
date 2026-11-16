import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'

type ArtifactEntry = components['schemas']['ArtifactEntry']

export interface ArtifactListProps {
  artifacts: ArtifactEntry[]
  zipUrl: string
}

/**
 * UX §3.6 "Výsledky, sťahovanie": keyed by `role` (H9). `source_preview`/
 * `docx_preview` are never listed here (they live in the comparison tab),
 * but stay in `artifacts` for the ZIP. A plain `<a href download>` per row
 * -- no `fetch`-to-blob -- so the same-origin cookie rides along on its own.
 */
export function ArtifactList({ artifacts, zipUrl }: ArtifactListProps) {
  const { t } = useTranslation()
  const diagnosticDocx = artifacts.find((artifact) => artifact.role === 'diagnostic_docx')
  const predictionIrPages = artifacts
    .filter((artifact) => artifact.role === 'prediction_ir')
    .sort((a, b) => (a.page_index ?? 0) - (b.page_index ?? 0))

  return (
    <div className="flex flex-col gap-3">
      {diagnosticDocx ? (
        <Button asChild size="lg">
          <a href={diagnosticDocx.url} download>
            {t('job.artifacts.downloadWord')}
          </a>
        </Button>
      ) : null}

      {predictionIrPages.length > 0 ? (
        <Collapsible>
          <CollapsibleTrigger className="text-sm font-medium text-accent hover:underline">
            {t('job.artifacts.predictionIr', { n: predictionIrPages.length })}
          </CollapsibleTrigger>
          <CollapsibleContent>
            <ul className="mt-2 flex flex-col gap-1">
              {predictionIrPages.map((artifact) => (
                <li key={artifact.url}>
                  <a href={artifact.url} download className="text-sm underline-offset-4 hover:underline">
                    {t('job.artifacts.page', { n: (artifact.page_index ?? 0) + 1 })}
                  </a>
                </li>
              ))}
            </ul>
          </CollapsibleContent>
        </Collapsible>
      ) : null}

      <div className="flex flex-col gap-1">
        <Button asChild variant="outline" className="self-start">
          <a href={zipUrl} download>
            {t('job.result.zip')}
          </a>
        </Button>
        <p className="text-xs text-muted-foreground">{t('job.result.zipHint')}</p>
      </div>
    </div>
  )
}

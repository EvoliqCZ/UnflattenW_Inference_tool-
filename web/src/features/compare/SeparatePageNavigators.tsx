import { useTranslation } from 'react-i18next'
import { PageNavigator } from './PageNavigator'

export interface SeparatePageNavigatorsProps {
  sourceIndex: number
  docxIndex: number
  sourceCount: number
  docxCount: number
  onNavigateSource: (index: number) => void
  onNavigateDocx: (index: number) => void
}

/** UX §5.5 "samostatne": one "◀ n/N ▶" navigator per pane instead of the single shared one. */
export function SeparatePageNavigators({
  sourceIndex,
  docxIndex,
  sourceCount,
  docxCount,
  onNavigateSource,
  onNavigateDocx,
}: SeparatePageNavigatorsProps) {
  const { t } = useTranslation()

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <PageNavigator
        index={sourceIndex}
        total={Math.max(sourceCount, 1)}
        onNavigate={onNavigateSource}
        id="compare-page-input-source"
        label={t('compare.pageInputLabelSource')}
      />
      <PageNavigator
        index={docxIndex}
        total={Math.max(docxCount, 1)}
        onNavigate={onNavigateDocx}
        id="compare-page-input-docx"
        label={t('compare.pageInputLabelDocx')}
      />
    </div>
  )
}

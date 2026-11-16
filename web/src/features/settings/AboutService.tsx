import { useTranslation } from 'react-i18next'
import { ProblemError } from '@/api/problem'
import { ProblemMessage } from '@/components/ProblemMessage'
import { CompanyBrand } from '@/components/CompanyBrand'
import { useSettingsQuery } from './api'

/** UX §6 Nastavenia -> "O službe": git SHA, API version, session limits, and the rootful-sandbox sentence when it applies. */
export function AboutService() {
  const { t } = useTranslation()
  const { data, isLoading, isError, error } = useSettingsQuery()

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">{t('common.loading')}</p>
  }

  if (isError || data === undefined) {
    return error instanceof ProblemError ? (
      <ProblemMessage problem={error.problem} context="page" />
    ) : (
      <p role="alert" className="text-sm text-destructive">
        {t('login.networkError')}
      </p>
    )
  }

  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-3 text-sm font-medium">
        <span>Unflatten W 1.0</span>
        <CompanyBrand />
      </div>
      <dl className="flex flex-col gap-1 text-sm">
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">{t('settings.about.version')}</dt>
          <dd className="font-mono">{data.service_version.git_sha}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">{t('settings.about.apiVersion')}</dt>
          <dd className="font-mono">{data.service_version.api_version}</dd>
        </div>
        <p className="text-muted-foreground">
          {t('settings.about.sessionIdle', { seconds: data.session.idle_seconds })}
        </p>
        <p className="text-muted-foreground">{t('settings.about.sessionMax', { seconds: data.session.max_seconds })}</p>
        {data.sandbox.rootful === true ? <p>{t('settings.about.sandboxRootful')}</p> : null}
      </dl>
    </>
  )
}

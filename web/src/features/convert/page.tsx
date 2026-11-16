import { useTranslation } from 'react-i18next'
import { useModelsQuery, useReadinessQuery } from '@/api/queries'
import type { ReadinessLimits } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { PhotoQualityNote } from '@/components/PhotoQualityNote'
import { ProblemMessage } from '@/components/ProblemMessage'
import { RecentJobs } from '@/features/jobs/RecentJobs'
import type { SupportedLanguage } from '@/i18n'
import { formatBytes } from '@/lib/format'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Dropzone } from './Dropzone'
import { JobLabelInput } from './JobLabelInput'
import { ModelPicker } from './ModelPicker'
import { OutputChoice } from './OutputChoice'
import { selectionErrorParams } from './selectionErrorMessage'
import { UploadQueue } from './UploadQueue'
import { useConvertForm } from './useConvertForm'

/** UX §4: single-column form, top to bottom follows the decision order (§4.1). */
export function ConvertPage() {
  const { t, i18n } = useTranslation()
  const language = i18n.language as SupportedLanguage
  const readiness = useReadinessQuery()
  const models = useModelsQuery()
  const limits: ReadinessLimits | undefined = readiness.data?.limits
  const serviceReady = readiness.data?.service_ready ?? false
  const form = useConvertForm(limits, serviceReady)

  const blockedReason = !serviceReady ? t('convert.blocked.notReady') : undefined

  return (
    <main className="mx-auto flex max-w-2xl flex-col gap-6 p-4">
      <h1 className="text-xl font-semibold">{t('convert.title')}</h1>

      <section aria-label={t('convert.dropzone.label')} className="flex flex-col gap-2">
        <Dropzone
          onFilesSelected={(files) => void form.addFiles(files)}
          disabled={!serviceReady}
          disabledReason={blockedReason}
          hasFiles={form.files.length > 0}
        />
        {limits ? (
          <p className="text-xs text-muted-foreground">
            {t('convert.limits', {
              maxPages: limits.max_pages,
              maxBytes: formatBytes(limits.max_upload_bytes, language),
            })}
          </p>
        ) : null}
        {form.selectionError ? (
          <Alert variant="destructive">
            <AlertDescription>
              {t(`validation.${form.selectionError.code}`, selectionErrorParams(form.selectionError, language))}
            </AlertDescription>
          </Alert>
        ) : null}
        {form.files.length > 0 ? (
          <UploadQueue files={form.files} onReorder={form.reorder} onRemove={form.removeFile} language={language} />
        ) : null}
        {form.hasImages ? <PhotoQualityNote variant="inline" /> : null}
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-medium text-foreground">{t('convert.output.legend')}</h2>
        <OutputChoice value={form.output} onChange={form.setOutput} />
      </section>

      {models.data && models.data.variants.length > 0 ? (
        <section className="flex flex-col gap-2">
          <h2 className="text-sm font-medium text-foreground">{t('convert.model.label')}</h2>
          <ModelPicker
            models={models.data}
            value={form.modelVariant}
            onChange={form.setModelVariant}
            language={language}
          />
        </section>
      ) : null}

      <JobLabelInput value={form.label} onChange={form.setLabel} />

      {form.submitError ? <ProblemMessage problem={form.submitError} context="submit" /> : null}

      <Button
        type="button"
        aria-disabled={!form.canSubmit}
        onClick={() => {
          if (form.canSubmit) void form.submit()
        }}
      >
        {form.progress !== null ? t('convert.submitting') : t('convert.submit')}
      </Button>
      {!form.canSubmit ? (
        <p className="text-sm text-muted-foreground">
          {form.files.length === 0 ? t('convert.blocked.noFiles') : blockedReason}
        </p>
      ) : null}

      <RecentJobs />
    </main>
  )
}

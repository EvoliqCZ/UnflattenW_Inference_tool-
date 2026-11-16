import { AlertTriangle } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { ExpiryCountdown } from '@/components/ExpiryCountdown'
import { FontSubstitutionNotice } from '@/components/FontSubstitutionNotice'
import { ProblemMessage } from '@/components/ProblemMessage'
import { ArtifactList } from './ArtifactList'

type ConversionStatus = components['schemas']['ConversionStatus']

export interface ResultPanelProps {
  job: ConversionStatus
}

/** UX §3.6/§7.2 "Výsledok" tab: font notice, "Word nevznikol" alert, artifacts, expiry, error. */
export function ResultPanel({ job }: ResultPanelProps) {
  const { t } = useTranslation()
  const outcome = job.diagnostic_outcome

  return (
    <div className="flex flex-col gap-4">
      {outcome ? <FontSubstitutionNotice outcome={outcome} /> : null}

      {outcome && !outcome.docx_produced ? (
        <Alert variant="warning">
          <AlertTriangle aria-hidden="true" className="size-4" />
          <AlertTitle>{t('job.result.docxMissing')}</AlertTitle>
          <AlertDescription>
            <ul className="list-disc pl-4">
              {outcome.reason_codes.map((code) => (
                <li key={code}>{t(`codes.reason.${code}`)}</li>
              ))}
            </ul>
            <p>{t('job.result.irOnly')}</p>
          </AlertDescription>
        </Alert>
      ) : null}

      <ArtifactList
        artifacts={job.artifacts.filter(
          (artifact): artifact is Exclude<typeof artifact, string> => typeof artifact !== 'string',
        )}
        zipUrl={`/v1/conversions/${job.job_id}/artifacts.zip`}
      />

      <ExpiryCountdown expiresAt={job.expires_at} />

      {job.error ? <ProblemMessage problem={job.error} context="job" /> : null}
    </div>
  )
}

import { useTranslation } from 'react-i18next'
import type { components } from '@/api/schema'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { PhotoQualityNote } from '@/components/PhotoQualityNote'
import { toast } from '@/hooks/use-toast'
import { RENDERER_APPROXIMATIONS, type InputKind, type RendererApproximation } from '@/i18n/codes'

type DiagnosticOutcome = components['schemas']['DiagnosticOutcome']

function isRendererApproximation(value: string): value is RendererApproximation {
  return (RENDERER_APPROXIMATIONS as readonly string[]).includes(value)
}

export interface DiagnosticsPanelProps {
  outcome: DiagnosticOutcome | null
  inputKind: InputKind | null
}

/**
 * UX §6 "Detail úlohy" Diagnostika tab, read-only. Free-form `docx_chain`
 * fields (`page_outcome`, `profile`, `ast_status`, `renderer_status`,
 * `error_class`) carry no i18n contract of their own -- they are the
 * pipeline's raw diagnostic strings, shown as-is in mono next to their raw
 * field name, the same "nothing hidden" convention as `StageStepper`'s
 * stage code. Only the closed enums (`renderer_approximations`,
 * `reason_codes`) go through `codes.*`.
 */
export function DiagnosticsPanel({ outcome, inputKind }: DiagnosticsPanelProps) {
  const { t } = useTranslation()

  function copyRawJson() {
    void navigator.clipboard.writeText(JSON.stringify(outcome, null, 2)).then(() => {
      toast({ title: t('common.copied') })
    })
  }

  return (
    <div className="flex flex-col gap-4">
      {inputKind === 'image' ? <PhotoQualityNote variant="inline" /> : null}

      {outcome === null ? (
        <p className="text-sm text-muted-foreground">{t('diagnostics.unavailable')}</p>
      ) : (
        <>
          <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-2 text-sm">
            <RawField label="prediction_ir_status" value={outcome.prediction_ir_status} />
            <BoolField label="canonical_ast_produced" value={outcome.canonical_ast_produced} />
            <BoolField label="docx_produced" value={outcome.docx_produced} />
            <RawField label="page_outcome" value={outcome.docx_chain?.page_outcome ?? null} />
            <RawField label="profile" value={outcome.docx_chain?.profile ?? null} />
            <RawField label="ast_status" value={outcome.docx_chain?.ast_status ?? null} />
            <RawField label="renderer_status" value={outcome.docx_chain?.renderer_status ?? null} />
            <RawField label="error_class" value={outcome.docx_chain?.error_class ?? null} />
            {outcome.docx_chain?.renderer_approximations && outcome.docx_chain.renderer_approximations.length > 0 ? (
              <>
                <dt>
                  <code>renderer_approximations</code>
                </dt>
                <dd>
                  {outcome.docx_chain.renderer_approximations
                    .map((code) => (isRendererApproximation(code) ? t(`codes.approximation.${code}`) : code))
                    .join(', ')}
                </dd>
              </>
            ) : null}
            {outcome.docx_chain?.failure_codes && outcome.docx_chain.failure_codes.length > 0 ? (
              <>
                <dt>
                  <code>failure_codes</code>
                </dt>
                <dd className="font-mono text-xs">{outcome.docx_chain.failure_codes.join(', ')}</dd>
              </>
            ) : null}
          </dl>

          {outcome.docx_chain?.policies && outcome.docx_chain.policies.length > 0 ? (
            <ul className="flex flex-col gap-1 text-sm">
              {outcome.docx_chain.policies.map((policy) => {
                const counts = Object.entries(policy.counts)
                return (
                  <li key={policy.policy_id}>
                    <code>{policy.policy_id}</code> · {policy.policy_status} ·{' '}
                    {counts.length === 0
                      ? t('diagnostics.policyNoEdits')
                      : counts.map(([key, value]) => `${key}: ${String(value)}`).join(', ')}
                  </li>
                )
              })}
            </ul>
          ) : null}

          {outcome.reason_codes.length > 0 ? (
            <ul className="flex flex-col gap-1 text-sm">
              {outcome.reason_codes.map((code) => (
                <li key={code}>
                  {t(`codes.reason.${code}`)} <code className="text-xs text-muted-foreground">{code}</code>
                </li>
              ))}
            </ul>
          ) : null}

          {outcome.reason_codes.includes('HEADLESS_DEFAULTS_V0_APPLIED') ? (
            <p className="text-sm text-muted-foreground">{t('diagnostics.headlessDefaults')}</p>
          ) : null}

          <Collapsible>
            <CollapsibleTrigger className="text-sm font-medium text-accent hover:underline">
              {t('diagnostics.rawJson')}
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-2 flex flex-col gap-2">
              <pre className="overflow-x-auto rounded-sm bg-secondary p-2 font-mono text-xs">
                {JSON.stringify(outcome, null, 2)}
              </pre>
              <Button type="button" variant="outline" size="sm" className="self-start" onClick={copyRawJson}>
                {t('common.copy')}
              </Button>
            </CollapsibleContent>
          </Collapsible>
        </>
      )}
    </div>
  )
}

function RawField({ label, value }: { label: string; value: string | null }) {
  if (value === null) return null
  return (
    <>
      <dt>
        <code>{label}</code>
      </dt>
      <dd className="font-mono text-xs">{value}</dd>
    </>
  )
}

function BoolField({ label, value }: { label: string; value: boolean }) {
  const { t } = useTranslation()
  return (
    <>
      <dt>
        <code>{label}</code>
      </dt>
      <dd>{value ? t('diagnostics.yes') : t('diagnostics.no')}</dd>
    </>
  )
}

import { AlertCircle } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { Problem } from '@/api/problem'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { PROBLEM_CODES, type ProblemCode } from '@/i18n/codes'

function isProblemCode(code: string): code is ProblemCode {
  return (PROBLEM_CODES as readonly string[]).includes(code)
}

/**
 * UX §7 component table: `problem: Problem | {code: string}` -- a full
 * `Problem` from the API, or just a code when the caller only has that
 * (e.g. a client-side validation result shaped like one). `retryable` only
 * exists on the full shape, so a bare `{code}` never renders the retry
 * button even if `onRetry` is supplied.
 */
export interface ProblemMessageProps {
  problem: Problem | { code: string }
  context: 'submit' | 'job' | 'page' | 'download'
  onRetry?: () => void
}

function isRetryable(problem: ProblemMessageProps['problem']): boolean {
  return 'retryable' in problem && problem.retryable
}

export function ProblemMessage({ problem, context, onRetry }: ProblemMessageProps) {
  const { t } = useTranslation()
  const code = problem.code
  // PROBLEM_CODES (codes.ts) is the closed, catalog-verified register
  // (i18n-completeness.test.ts) -- an unrecognized code falls back to the
  // generic "unknown code" text instead of a raw i18next key string.
  const title = isProblemCode(code) ? t(`codes.problem.${code}.title`) : t('codes.unknown')
  const hint = isProblemCode(code) ? t(`codes.problem.${code}.hint`) : null

  return (
    <Alert variant="destructive" data-context={context}>
      <AlertCircle aria-hidden="true" className="size-4" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>
        {hint !== null ? <p>{hint}</p> : null}
        {isRetryable(problem) && onRetry ? (
          <Button type="button" size="sm" variant="outline" className="mt-2" onClick={onRetry}>
            {t('common.retry')}
          </Button>
        ) : null}
      </AlertDescription>
    </Alert>
  )
}

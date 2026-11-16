import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router'
import { ProblemError, TransportError } from '@/api/problem'
import { useSessionQuery } from '@/auth/session'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { toast } from '@/hooks/use-toast'
import type { SupportedLanguage } from '@/i18n'
import { ModelStep } from './ModelStep'
import { PasswordStep } from './PasswordStep'
import { ReadyStep } from './ReadyStep'
import { type SetupStep, SetupStepper } from './SetupStepper'
import { useSetupMutation } from './api'
import { SystemStep } from './SystemStep'

/**
 * The bootstrap token from the setup link `run-local.sh` prints for a
 * non-loopback deployment (`/setup#token=<t>`). A fragment is never sent to
 * a server, so the token reaches no access log; the page strips it from the
 * address bar right after reading it (`useEffect` below).
 */
function readSetupLinkToken(): string | null {
  const token = new URLSearchParams(window.location.hash.slice(1)).get('token')
  return token === null || token === '' ? null : token
}

// UX §3.1 "/setup" -- setup wizard. Owner 2026-09-29 (DEC-052 addendum): no
// token field. A loopback-only deployment offers tokenless setup
// (`SessionInfo.setup_tokenless`); any other one needs the setup link, and
// without it the page only says where to find that link.
export function SetupPage() {
  const { t, i18n } = useTranslation()
  const language = i18n.language as SupportedLanguage
  const navigate = useNavigate()
  const session = useSessionQuery()
  const setupMutation = useSetupMutation()
  const [linkToken] = useState(readSetupLinkToken)
  const [step, setStep] = useState<SetupStep>('password')
  const [setupRefused, setSetupRefused] = useState(false)
  const [passwordStepError, setPasswordStepError] = useState<ProblemError | TransportError | null>(null)

  useEffect(() => {
    if (new URLSearchParams(window.location.hash.slice(1)).has('token')) {
      // Keep the router's own history state; only the URL loses its fragment.
      window.history.replaceState(window.history.state, '', window.location.pathname + window.location.search)
    }
  }, [])

  const canSubmit = linkToken !== null || session.data?.setup_tokenless === true

  async function handlePasswordSubmit(password: string): Promise<void> {
    setPasswordStepError(null)
    try {
      await setupMutation.mutateAsync({ token: linkToken, password })
      setStep('system')
    } catch (cause) {
      if (cause instanceof ProblemError) {
        if (cause.status === 401) {
          setSetupRefused(true)
        } else if (cause.code === 'SETUP_ALREADY_COMPLETE') {
          toast({ title: t('setup.step.password.alreadyComplete') })
          void navigate('/login')
        } else {
          setPasswordStepError(cause)
        }
      } else if (cause instanceof TransportError) {
        setPasswordStepError(cause)
      } else {
        throw cause
      }
    }
  }

  function renderPasswordStep() {
    if (session.isPending && linkToken === null) {
      return null
    }
    if (setupRefused || !canSubmit) {
      return (
        <p role="alert" className="text-sm">
          {setupRefused && linkToken !== null ? t('setup.link.invalid') : t('setup.link.missing')}
        </p>
      )
    }
    return (
      <PasswordStep
        onSubmit={(password) => {
          void handlePasswordSubmit(password)
        }}
        isSubmitting={setupMutation.isPending}
        error={passwordStepError}
      />
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t('setup.title')}</CardTitle>
        <SetupStepper current={step} />
      </CardHeader>
      <CardContent>
        {step === 'password' ? renderPasswordStep() : null}
        {step === 'system' ? (
          <SystemStep
            language={language}
            onNext={() => {
              setStep('model')
            }}
          />
        ) : null}
        {step === 'model' ? (
          <ModelStep
            language={language}
            onNext={() => {
              setStep('ready')
            }}
          />
        ) : null}
        {step === 'ready' ? <ReadyStep /> : null}
      </CardContent>
    </Card>
  )
}

import { Eye, EyeOff } from 'lucide-react'
import { type SubmitEvent, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate, useSearchParams } from 'react-router'
import { ProblemError, TransportError } from '@/api/problem'
import { sanitizeNextPath } from '@/auth/guard'
import { useLoginMutation } from '@/auth/session'
import { ProblemMessage } from '@/components/ProblemMessage'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ForgotPasswordCollapsible } from './ForgotPasswordCollapsible'

const RATE_LIMIT_TICK_MS = 1000
const PASSWORD_INPUT_ID = 'login-password'

/** Ticks a rate-limit countdown down to 0, one second at a time (UX §3.2 point 3). */
function useCountdownSeconds(): [number, (seconds: number) => void] {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    if (seconds <= 0) {
      return
    }
    const id = window.setInterval(() => {
      setSeconds((current) => Math.max(0, current - 1))
    }, RATE_LIMIT_TICK_MS)
    return () => {
      window.clearInterval(id)
    }
  }, [seconds])
  return [seconds, setSeconds]
}

// UX §3.2 "/login" -- Login screen: single password field, 401 keeps the
// field and its value (only inline text changes), 429 disables the submit
// button for the server-given `retry_after_seconds`, and "Zabudnuté heslo?"
// shows the CLI reset command instead of any email flow.
export function LoginPage() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const loginMutation = useLoginMutation()
  const [password, setPassword] = useState('')
  const [isPasswordVisible, setIsPasswordVisible] = useState(false)
  const [wrongPassword, setWrongPassword] = useState(false)
  const [otherError, setOtherError] = useState<ProblemError | TransportError | null>(null)
  const [retrySeconds, setRetrySeconds] = useCountdownSeconds()

  async function handleSubmit(event: SubmitEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    if (retrySeconds > 0) {
      return
    }
    setWrongPassword(false)
    setOtherError(null)
    try {
      await loginMutation.mutateAsync(password)
      // `sanitizeNextPath` only ever returns a route from `routes.json` --
      // no open redirect (`auth/guard.ts`).
      const next = sanitizeNextPath(searchParams.get('next'))
      void navigate(next, { replace: true })
    } catch (cause) {
      if (cause instanceof ProblemError) {
        if (cause.code === 'RATE_LIMITED') {
          setRetrySeconds(cause.problem.retry_after_seconds ?? 60)
          setOtherError(cause)
        } else if (cause.status === 401) {
          // UX §3.2 point 2: field keeps its value and focus, only the
          // inline message appears -- the user needs to see the typo.
          setWrongPassword(true)
          document.getElementById(PASSWORD_INPUT_ID)?.focus()
        } else {
          setOtherError(cause)
        }
      } else if (cause instanceof TransportError) {
        setOtherError(cause)
      } else {
        throw cause
      }
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t('login.title')}</CardTitle>
      </CardHeader>
      <CardContent>
        <form
          className="flex flex-col gap-3"
          onSubmit={(event) => {
            void handleSubmit(event)
          }}
        >
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={PASSWORD_INPUT_ID}>{t('login.password')}</Label>
            <div className="flex gap-2">
              <Input
                id={PASSWORD_INPUT_ID}
                type={isPasswordVisible ? 'text' : 'password'}
                autoComplete="current-password"
                required
                value={password}
                aria-describedby={wrongPassword ? 'login-password-error' : undefined}
                aria-invalid={wrongPassword}
                onChange={(event) => {
                  setPassword(event.target.value)
                }}
              />
              <Button
                type="button"
                variant="outline"
                size="icon"
                aria-label={isPasswordVisible ? t('login.hidePassword') : t('login.showPassword')}
                onClick={() => {
                  setIsPasswordVisible((visible) => !visible)
                }}
              >
                {isPasswordVisible ? <EyeOff aria-hidden="true" /> : <Eye aria-hidden="true" />}
              </Button>
            </div>
            {wrongPassword ? (
              <p id="login-password-error" role="alert" className="text-sm text-destructive">
                {t('login.wrong')}
              </p>
            ) : null}
          </div>

          {otherError !== null ? (
            otherError instanceof ProblemError ? (
              <ProblemMessage problem={otherError.problem} context="submit" />
            ) : (
              <p role="alert" className="text-sm text-destructive">
                {t('login.networkError')}
              </p>
            )
          ) : null}

          <Button type="submit" aria-disabled={retrySeconds > 0} disabled={loginMutation.isPending}>
            {retrySeconds > 0 ? t('login.rateLimited', { seconds: retrySeconds }) : t('login.submit')}
          </Button>
        </form>

        <ForgotPasswordCollapsible />
      </CardContent>
    </Card>
  )
}

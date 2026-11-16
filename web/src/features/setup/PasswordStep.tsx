import { type SubmitEvent, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ProblemError, TransportError } from '@/api/problem'
import { ProblemMessage } from '@/components/ProblemMessage'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

const MIN_PASSWORD_LENGTH = 12

export interface PasswordStepProps {
  onSubmit: (password: string) => void
  isSubmitting: boolean
  /** A non-401, non-`SETUP_ALREADY_COMPLETE` failure of the `POST /v1/setup` call (UX §3.1 step 2: "sieť -> Alert"). */
  error: ProblemError | TransportError | null
}

/** UX §3.1 step 2 "password": min 12 code points, no composition rules (NIST 800-63B), length indicator only. */
export function PasswordStep({ onSubmit, isSubmitting, error }: PasswordStepProps) {
  const { t } = useTranslation()
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [tooShort, setTooShort] = useState(false)
  const [mismatch, setMismatch] = useState(false)

  function handleSubmit(event: SubmitEvent<HTMLFormElement>): void {
    event.preventDefault()
    const isTooShort = password.length < MIN_PASSWORD_LENGTH
    const isMismatched = password !== confirmPassword
    setTooShort(isTooShort)
    setMismatch(isMismatched)
    if (isTooShort || isMismatched) {
      return
    }
    onSubmit(password)
  }

  return (
    <form className="flex flex-col gap-3" onSubmit={handleSubmit}>
      <h2 className="text-base font-semibold">{t('setup.step.password.title')}</h2>
      <p className="text-sm text-muted-foreground">{t('setup.step.password.body')}</p>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="setup-password">{t('setup.step.password.passwordLabel')}</Label>
        <Input
          id="setup-password"
          type="password"
          autoComplete="new-password"
          required
          value={password}
          onChange={(event) => {
            setPassword(event.target.value)
          }}
        />
        {tooShort ? (
          <p role="alert" className="text-sm text-destructive">
            {t('setup.step.password.tooShort')}
          </p>
        ) : null}
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="setup-password-confirm">{t('setup.step.password.confirmLabel')}</Label>
        <Input
          id="setup-password-confirm"
          type="password"
          autoComplete="new-password"
          required
          value={confirmPassword}
          onChange={(event) => {
            setConfirmPassword(event.target.value)
          }}
        />
        {mismatch ? (
          <p role="alert" className="text-sm text-destructive">
            {t('setup.step.password.mismatch')}
          </p>
        ) : null}
      </div>
      {error !== null ? (
        error instanceof ProblemError ? (
          <ProblemMessage problem={error.problem} context="submit" />
        ) : (
          <p role="alert" className="text-sm text-destructive">
            {t('login.networkError')}
          </p>
        )
      ) : null}
      <Button type="submit" disabled={isSubmitting}>
        {t('setup.step.password.submit')}
      </Button>
    </form>
  )
}

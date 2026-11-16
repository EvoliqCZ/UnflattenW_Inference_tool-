import { type SubmitEvent, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ProblemError, TransportError } from '@/api/problem'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { toast } from '@/hooks/use-toast'
import { useChangePasswordMutation } from './api'
import { ChangePasswordForm } from './ChangePasswordForm'

const MIN_PASSWORD_LENGTH = 12
const RATE_LIMIT_TICK_MS = 1000

/** Ticks a rate-limit countdown down to 0, one second at a time (mirrors `LoginPage`'s, UX §3.2 point 3). */
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

/** UX §6 Nastavenia -> Účet -> "Zmeniť heslo" dialog: current/new/confirm, `PUT /v1/setup/password`. */
export function ChangePasswordDialog() {
  const { t } = useTranslation()
  const mutation = useChangePasswordMutation()
  const [open, setOpen] = useState(false)
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [tooShort, setTooShort] = useState(false)
  const [mismatch, setMismatch] = useState(false)
  const [wrongCurrentPassword, setWrongCurrentPassword] = useState(false)
  const [otherError, setOtherError] = useState<ProblemError | TransportError | null>(null)
  const [retrySeconds, setRetrySeconds] = useCountdownSeconds()

  function resetForm(): void {
    setCurrentPassword('')
    setNewPassword('')
    setConfirmPassword('')
    setTooShort(false)
    setMismatch(false)
    setWrongCurrentPassword(false)
    setOtherError(null)
  }

  async function handleSubmit(event: SubmitEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    if (retrySeconds > 0) {
      return
    }
    const isTooShort = newPassword.length < MIN_PASSWORD_LENGTH
    const isMismatched = newPassword !== confirmPassword
    setTooShort(isTooShort)
    setMismatch(isMismatched)
    setWrongCurrentPassword(false)
    setOtherError(null)
    if (isTooShort || isMismatched) {
      return
    }
    try {
      await mutation.mutateAsync({ currentPassword, newPassword })
      toast({ title: t('settings.changePasswordDialog.success') })
      setOpen(false)
      resetForm()
    } catch (cause) {
      if (cause instanceof ProblemError) {
        if (cause.code === 'RATE_LIMITED') {
          setRetrySeconds(cause.problem.retry_after_seconds ?? 60)
          setOtherError(cause)
        } else if (cause.status === 401) {
          setWrongCurrentPassword(true)
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
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        setOpen(nextOpen)
        if (!nextOpen) {
          resetForm()
        }
      }}
    >
      <DialogTrigger asChild>
        <Button type="button" variant="outline">
          {t('settings.account.changePassword')}
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t('settings.changePasswordDialog.title')}</DialogTitle>
        </DialogHeader>
        <ChangePasswordForm
          currentPassword={currentPassword}
          newPassword={newPassword}
          confirmPassword={confirmPassword}
          onCurrentPasswordChange={setCurrentPassword}
          onNewPasswordChange={setNewPassword}
          onConfirmPasswordChange={setConfirmPassword}
          tooShort={tooShort}
          mismatch={mismatch}
          wrongCurrentPassword={wrongCurrentPassword}
          error={otherError}
          retrySeconds={retrySeconds}
          isSubmitting={mutation.isPending}
          onSubmit={(event) => {
            void handleSubmit(event)
          }}
        />
      </DialogContent>
    </Dialog>
  )
}

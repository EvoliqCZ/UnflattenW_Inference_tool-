import type { SubmitEvent } from 'react'
import { useTranslation } from 'react-i18next'
import { ProblemError, TransportError } from '@/api/problem'
import { ProblemMessage } from '@/components/ProblemMessage'
import { Button } from '@/components/ui/button'
import { DialogFooter } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'

export interface ChangePasswordFormProps {
  currentPassword: string
  newPassword: string
  confirmPassword: string
  onCurrentPasswordChange: (value: string) => void
  onNewPasswordChange: (value: string) => void
  onConfirmPasswordChange: (value: string) => void
  tooShort: boolean
  mismatch: boolean
  wrongCurrentPassword: boolean
  error: ProblemError | TransportError | null
  retrySeconds: number
  isSubmitting: boolean
  onSubmit: (event: SubmitEvent<HTMLFormElement>) => void
}

/** The form body of `ChangePasswordDialog`, split out to keep both files under the 150-line limit. */
export function ChangePasswordForm({
  currentPassword,
  newPassword,
  confirmPassword,
  onCurrentPasswordChange,
  onNewPasswordChange,
  onConfirmPasswordChange,
  tooShort,
  mismatch,
  wrongCurrentPassword,
  error,
  retrySeconds,
  isSubmitting,
  onSubmit,
}: ChangePasswordFormProps) {
  const { t } = useTranslation()

  return (
    <form className="flex flex-col gap-3" onSubmit={onSubmit}>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="change-password-current">{t('settings.changePasswordDialog.currentPasswordLabel')}</Label>
        <Input
          id="change-password-current"
          type="password"
          autoComplete="current-password"
          required
          value={currentPassword}
          aria-invalid={wrongCurrentPassword}
          onChange={(event) => {
            onCurrentPasswordChange(event.target.value)
          }}
        />
        {wrongCurrentPassword ? (
          <p role="alert" className="text-sm text-destructive">
            {t('settings.changePasswordDialog.wrongCurrentPassword')}
          </p>
        ) : null}
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="change-password-new">{t('settings.changePasswordDialog.newPasswordLabel')}</Label>
        <Input
          id="change-password-new"
          type="password"
          autoComplete="new-password"
          required
          value={newPassword}
          onChange={(event) => {
            onNewPasswordChange(event.target.value)
          }}
        />
        {tooShort ? (
          <p role="alert" className="text-sm text-destructive">
            {t('settings.changePasswordDialog.tooShort')}
          </p>
        ) : null}
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="change-password-confirm">{t('settings.changePasswordDialog.confirmLabel')}</Label>
        <Input
          id="change-password-confirm"
          type="password"
          autoComplete="new-password"
          required
          value={confirmPassword}
          onChange={(event) => {
            onConfirmPasswordChange(event.target.value)
          }}
        />
        {mismatch ? (
          <p role="alert" className="text-sm text-destructive">
            {t('settings.changePasswordDialog.mismatch')}
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
      <DialogFooter>
        <Button type="submit" aria-disabled={retrySeconds > 0} disabled={isSubmitting}>
          {retrySeconds > 0
            ? t('login.rateLimited', { seconds: retrySeconds })
            : t('settings.changePasswordDialog.submit')}
        </Button>
      </DialogFooter>
    </form>
  )
}

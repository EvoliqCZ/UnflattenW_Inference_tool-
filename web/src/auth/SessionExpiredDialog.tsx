import { useQueryClient } from '@tanstack/react-query'
import { type SubmitEvent, useState, useSyncExternalStore } from 'react'
import { useTranslation } from 'react-i18next'
import { queryKeys } from '@/api/queries'
import { ProblemError } from '@/api/problem'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ProblemMessage } from '@/components/ProblemMessage'
import { postLogin } from './session'

type Listener = () => void

let isOpenState = false
let pendingRetry: (() => void) | null = null
const listeners = new Set<Listener>()

function emit(): void {
  for (const listener of listeners) {
    listener()
  }
}

/**
 * design §7.4 / UX §3.2 point 5: a single global controller, opened by
 * `api/queryClient.ts`'s `QueryCache.onError` from *outside* the React tree
 * that made the failing call -- any screen's query can 401 with an expired
 * session. `retry` is only ever a re-fetch of the query that failed (a GET,
 * design §7.3): `QueryCache.onError` fires for queries, never for mutations
 * (those go through the separate, unwired `MutationCache`), so a submit
 * never gets replayed here (UX §3.2 point 5: "odoslanie úlohy sa
 * nezopakuje automaticky").
 */
export const sessionExpiredController = {
  open(retry?: () => void): void {
    isOpenState = true
    if (retry) {
      pendingRetry = retry
    }
    emit()
  },
  close(): void {
    isOpenState = false
    pendingRetry = null
    emit()
  },
  consumeRetry(): (() => void) | null {
    const retry = pendingRetry
    pendingRetry = null
    return retry
  },
  subscribe(listener: Listener): () => void {
    listeners.add(listener)
    return () => {
      listeners.delete(listener)
    }
  },
  getSnapshot(): boolean {
    return isOpenState
  },
}

/**
 * UX §3.2 point 5: the password field appears "on the spot" (a global
 * dialog, not a redirect to `/login`) so a rendered-but-unsubmitted form
 * elsewhere on the page survives. On success, the query that triggered the
 * dialog (if any) is retried exactly once.
 */
export function SessionExpiredDialog() {
  const { t } = useTranslation()
  const isOpen = useSyncExternalStore(
    (listener) => sessionExpiredController.subscribe(listener),
    () => sessionExpiredController.getSnapshot(),
  )
  const queryClient = useQueryClient()
  const [password, setPassword] = useState('')
  const [error, setError] = useState<ProblemError | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  async function handleSubmit(event: SubmitEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    setIsSubmitting(true)
    setError(null)
    try {
      const session = await postLogin(password)
      queryClient.setQueryData(queryKeys.session(), session)
      setPassword('')
      const retry = sessionExpiredController.consumeRetry()
      sessionExpiredController.close()
      retry?.()
    } catch (cause) {
      if (cause instanceof ProblemError) {
        setError(cause)
      } else {
        throw cause
      }
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) {
          sessionExpiredController.close()
        }
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t('session.expired.title')}</DialogTitle>
          <DialogDescription>{t('session.expired.body')}</DialogDescription>
        </DialogHeader>
        <form
          className="flex flex-col gap-3"
          onSubmit={(event) => {
            void handleSubmit(event)
          }}
        >
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="session-expired-password">{t('login.password')}</Label>
            <Input
              id="session-expired-password"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(event) => {
                setPassword(event.target.value)
              }}
            />
          </div>
          {error !== null ? <ProblemMessage problem={error.problem} context="submit" /> : null}
          <DialogFooter>
            <Button type="submit" disabled={isSubmitting || password.length === 0}>
              {t('login.submit')}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

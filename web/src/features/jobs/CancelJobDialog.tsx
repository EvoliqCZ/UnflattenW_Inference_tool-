import { useTranslation } from 'react-i18next'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { ProblemError } from '@/api/problem'
import { toast } from '@/hooks/use-toast'
import { useCancelJobMutation, type JobSummary } from './api'
import { jobDisplayName } from './jobRowHelpers'

export interface CancelJobDialogProps {
  job: Pick<JobSummary, 'job_id' | 'label'>
  open: boolean
  onOpenChange: (open: boolean) => void
}

/**
 * UX §3.5: confirm, then cancel. A `JOB_TERMINAL` 409 (the job finished
 * before the cancel reached the server) is not an error here -- the dialog
 * closes and a toast explains it (design §3.5 point 3); any other failure
 * stays on the dialog via the mutation's own error state.
 */
export function CancelJobDialog({ job, open, onOpenChange }: CancelJobDialogProps) {
  const { t } = useTranslation()
  const mutation = useCancelJobMutation()

  async function handleConfirm() {
    try {
      await mutation.mutateAsync(job.job_id)
      onOpenChange(false)
    } catch (error) {
      if (error instanceof ProblemError && error.code === 'JOB_TERMINAL') {
        onOpenChange(false)
        toast({ title: t('job.cancel.alreadyTerminal') })
        return
      }
      // Any other failure (network, 5xx): leave the dialog open so the
      // pending state below (and a retry) are visible; `mutation.error`
      // is not re-thrown further.
    }
  }

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t('job.cancel.title')}</AlertDialogTitle>
          <AlertDialogDescription>{t('job.cancel.body', { label: jobDisplayName(job) })}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t('job.cancel.keep')}</AlertDialogCancel>
          <AlertDialogAction
            aria-disabled={mutation.isPending}
            onClick={(event) => {
              event.preventDefault()
              if (!mutation.isPending) void handleConfirm()
            }}
          >
            {t('job.cancel.confirm')}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

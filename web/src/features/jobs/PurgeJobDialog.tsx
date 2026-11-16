import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router'
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
import { toast } from '@/hooks/use-toast'
import { usePurgeJobMutation, type JobSummary } from './api'
import { jobDisplayName } from './jobRowHelpers'

export interface PurgeJobDialogProps {
  job: Pick<JobSummary, 'job_id' | 'label'>
  open: boolean
  onOpenChange: (open: boolean) => void
}

/** UX §3.6 "Zmazanie z histórie": permanent delete, then back to the jobs list (already there from the list, a no-op navigation). */
export function PurgeJobDialog({ job, open, onOpenChange }: PurgeJobDialogProps) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const mutation = usePurgeJobMutation()

  async function handleConfirm() {
    try {
      await mutation.mutateAsync(job.job_id)
    } catch {
      // Left on the dialog: `mutation.error`/`isError` reflect the failure
      // (e.g. a race where the job became active again is not expected in
      // v1, but a transport failure is); no further action here.
      return
    }
    onOpenChange(false)
    toast({ title: t('job.purge.toast') })
    void navigate('/jobs')
  }

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t('job.purge.title')}</AlertDialogTitle>
          <AlertDialogDescription>{t('job.purge.body', { label: jobDisplayName(job) })}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t('job.purge.keep')}</AlertDialogCancel>
          <AlertDialogAction
            aria-disabled={mutation.isPending}
            onClick={(event) => {
              event.preventDefault()
              if (!mutation.isPending) void handleConfirm()
            }}
          >
            {t('job.purge.confirm')}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

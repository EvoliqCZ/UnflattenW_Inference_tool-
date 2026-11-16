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
import { formatBytes } from '@/lib/format'
import type { SupportedLanguage } from '@/i18n'
import { useDeleteModelDownloadMutation, useLoadModelMutation, type ModelVariantEntry } from './api'
import { ModelSwitchNotice } from './ModelSwitchNotice'

interface DialogProps {
  variant: ModelVariantEntry
  open: boolean
  onOpenChange: (open: boolean) => void
}

/** UX §3.7 point 3: confirm before evicting the resident model and loading `variant`. */
export function ModelLoadDialog({
  variant,
  resident,
  language,
  open,
  onOpenChange,
}: DialogProps & {
  resident: ModelVariantEntry | null
  language: SupportedLanguage
}) {
  const { t } = useTranslation()
  const mutation = useLoadModelMutation()

  async function handleConfirm() {
    try {
      await mutation.mutateAsync(variant.variant_id)
      onOpenChange(false)
    } catch {
      // Left open: `mutation.error`/`isError` reflect the failure for a retry.
    }
  }

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t('models.confirm.load.title', { label: variant.label })}</AlertDialogTitle>
          <AlertDialogDescription asChild>
            <ModelSwitchNotice selected={variant} resident={resident} language={language} />
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
          <AlertDialogAction
            aria-disabled={mutation.isPending}
            onClick={(event) => {
              event.preventDefault()
              if (!mutation.isPending) void handleConfirm()
            }}
          >
            {t('models.action.load')}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

/** UX §3.7 point 5: confirm before removing the local copy. */
export function ModelDeleteDialog({
  variant,
  language,
  open,
  onOpenChange,
}: DialogProps & { language: SupportedLanguage }) {
  const { t } = useTranslation()
  const mutation = useDeleteModelDownloadMutation()

  async function handleConfirm() {
    try {
      await mutation.mutateAsync(variant.variant_id)
      onOpenChange(false)
    } catch {
      // Left open, mutation.error reflects the failure.
    }
  }

  const size =
    variant.download_size_bytes !== null ? formatBytes(variant.download_size_bytes, language) : t('common.unknown')

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t('models.confirm.delete.title', { size })}</AlertDialogTitle>
          <AlertDialogDescription>{t('models.confirm.delete.body')}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
          <AlertDialogAction
            aria-disabled={mutation.isPending}
            onClick={(event) => {
              event.preventDefault()
              if (!mutation.isPending) void handleConfirm()
            }}
          >
            {t('models.action.delete')}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

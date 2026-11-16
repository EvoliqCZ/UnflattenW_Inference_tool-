import { useTranslation } from 'react-i18next'
import { toast } from '@/hooks/use-toast'
import { Button } from '@/components/ui/button'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'

/** UX §3.2 point 4: no e-mail flow, just the CLI reset command (with a copy button). */
export function ForgotPasswordCollapsible() {
  const { t } = useTranslation()

  function copyResetCommand(): void {
    void navigator.clipboard.writeText(t('login.resetCommand')).then(() => {
      toast({ title: t('common.copied') })
    })
  }

  return (
    <Collapsible className="mt-6">
      <CollapsibleTrigger className="text-sm font-medium text-accent hover:underline">
        {t('login.forgot')}
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 flex flex-col gap-2 text-sm text-muted-foreground">
        <p>{t('login.forgotBody')}</p>
        <div className="flex items-center gap-2">
          <code className="rounded-sm bg-secondary px-2 py-1 font-mono text-xs">{t('login.resetCommand')}</code>
          <Button type="button" variant="outline" size="sm" onClick={copyResetCommand}>
            {t('common.copy')}
          </Button>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

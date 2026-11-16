import { removeToast, useToasts } from '@/hooks/use-toast'
import { Toast, ToastClose, ToastDescription, ToastProvider, ToastTitle, ToastViewport } from './toast'

/** Mounted once in `AppShell`/`AuthShell` (UX §7.2 tree). */
export function Toaster() {
  const toasts = useToasts()

  return (
    <ToastProvider>
      {toasts.map(({ id, title, description, variant, open }) => (
        <Toast
          key={id}
          variant={variant}
          open={open}
          onOpenChange={(nextOpen) => {
            if (!nextOpen) {
              removeToast(id)
            }
          }}
        >
          <div className="flex flex-1 flex-col gap-1">
            <ToastTitle>{title}</ToastTitle>
            {description ? <ToastDescription>{description}</ToastDescription> : null}
          </div>
          <ToastClose />
        </Toast>
      ))}
      <ToastViewport />
    </ToastProvider>
  )
}

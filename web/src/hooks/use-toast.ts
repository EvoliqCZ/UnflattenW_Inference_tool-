import { useEffect, useState } from 'react'

// Minimal toast store (shadcn/ui pattern, trimmed): a module-level list plus
// a subscriber set, so any component can call `toast()` and any number of
// `<Toaster>` instances stay in sync without prop drilling or a context
// provider. No auto-dismiss timers -- `Toast` closes on user action
// (Radix `onOpenChange`) or an explicit `dismiss()` call.

export interface ToastOptions {
  title: string
  description?: string
  variant?: 'default' | 'destructive'
}

export interface ToastRecord extends ToastOptions {
  id: string
  open: boolean
}

let toasts: ToastRecord[] = []
const listeners = new Set<(toasts: ToastRecord[]) => void>()

function emit() {
  for (const listener of listeners) {
    listener(toasts)
  }
}

export function toast(options: ToastOptions): string {
  const id = crypto.randomUUID()
  toasts = [...toasts, { ...options, id, open: true }]
  emit()
  return id
}

export function dismissToast(id: string): void {
  toasts = toasts.map((t) => (t.id === id ? { ...t, open: false } : t))
  emit()
}

export function removeToast(id: string): void {
  toasts = toasts.filter((t) => t.id !== id)
  emit()
}

export function useToasts(): ToastRecord[] {
  const [state, setState] = useState(toasts)

  useEffect(() => {
    listeners.add(setState)
    return () => {
      listeners.delete(setState)
    }
  }, [])

  return state
}

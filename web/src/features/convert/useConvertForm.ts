import { useEffect, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router'
import { submitConversion, type ConversionInputKind } from '@/api/upload'
import type { ReadinessLimits } from '@/api/queries'
import { ProblemError, TransportError } from '@/api/problem'
import type { components } from '@/api/schema'
import { SAMPLE_PDF_URL } from '@/features/setup/ReadyStep'
import { validateSelection, type SelectionValidationError } from './validateSelection'
import { selectedFileKind, type SelectedFile } from './types'

type Problem = components['schemas']['Problem']
type ConversionOutput = components['schemas']['SubmissionResponse']['requested_output']

function defaultLabel(files: readonly SelectedFile[]): string {
  const [first, ...rest] = files
  if (first === undefined) {
    return ''
  }
  const name = first.file.name.replace(/\.[^.]+$/, '')
  return rest.length > 0 ? `${name} +${String(rest.length)}` : name
}

function toSelectedFiles(files: readonly File[]): SelectedFile[] {
  return files.map((file) => ({ id: crypto.randomUUID(), file, kind: selectedFileKind(file) }))
}

/**
 * UX §3.3/§4: the "New conversion" form's state and submission, split out of
 * `page.tsx` so the page component stays presentational. Selection is
 * all-or-nothing (`validateSelection`, UX §4.3): a rejected drop/pick never
 * merges with the existing queue.
 */
export function useConvertForm(limits: ReadinessLimits | undefined, serviceReady: boolean) {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const samplePreloadedRef = useRef(false)
  const inFlight = useRef(false)
  const retry = useRef<{ signature: string; key: string } | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [files, setFiles] = useState<SelectedFile[]>([])
  const [output, setOutput] = useState<ConversionOutput>('diagnostic_docx')
  const [modelVariant, setModelVariant] = useState<string | null>(null)
  const [label, setLabel] = useState('')
  const [labelTouched, setLabelTouched] = useState(false)
  const [selectionError, setSelectionError] = useState<SelectionValidationError | null>(null)
  const [submitError, setSubmitError] = useState<Problem | { code: string } | null>(null)
  const [progress, setProgress] = useState<number | null>(null)

  async function addFiles(picked: File[]): Promise<void> {
    setSelectionError(null)
    if (limits === undefined) {
      // UX §4.3: readiness has not loaded yet -- accept and validate limits later.
      const next = toSelectedFiles(picked)
      setFiles(next)
      if (!labelTouched) setLabel(defaultLabel(next))
      return
    }
    const error = await validateSelection(picked, {
      maxPages: limits.max_pages,
      maxUploadBytes: limits.max_upload_bytes,
      acceptedMediaTypes: limits.accepted_media_types,
    })
    if (error !== null) {
      setSelectionError(error)
      return
    }
    const next = toSelectedFiles(picked)
    setFiles(next)
    if (!labelTouched) setLabel(defaultLabel(next))
  }

  // UX §3.1 step 5 "Skúsiť ukážkový dokument": `ReadyStep` links to
  // `/?sample=<SAMPLE_PDF_URL>`; a same-origin fetch turns the bundled asset
  // into a `File` and runs it through the normal `addFiles` path (CSP: the
  // file itself is never assigned to a `blob:`/`data:` URL, only fetched).
  useEffect(() => {
    const sample = searchParams.get('sample')
    if (sample !== SAMPLE_PDF_URL || samplePreloadedRef.current) return
    samplePreloadedRef.current = true
    void (async () => {
      try {
        const response = await fetch(sample)
        if (!response.ok) return
        const blob = await response.blob()
        await addFiles([new File([blob], 'sample.pdf', { type: 'application/pdf' })])
      } catch {
        // Best-effort preload: a failed fetch just leaves the form empty for
        // a manual pick -- no error surface is designed for this shortcut.
      }
    })()
    // `addFiles` is not memoized and is guarded by `samplePreloadedRef` above
    // (runs at most once), so tracking its identity here would only make the
    // effect re-fire on every render for no behavioural difference.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams])

  function removeFile(id: string): void {
    const next = files.filter((entry) => entry.id !== id)
    setFiles(next)
    if (!labelTouched) setLabel(defaultLabel(next))
  }

  function setLabelValue(value: string): void {
    setLabelTouched(true)
    setLabel(value)
  }

  const hasImages = files.some((entry) => entry.kind === 'image')
  const kind: ConversionInputKind = files.some((entry) => entry.kind === 'pdf') ? 'pdf' : 'image'
  const canSubmit = files.length > 0 && serviceReady && !submitting

  async function submit(): Promise<void> {
    if (!canSubmit || inFlight.current) return
    inFlight.current = true
    setSubmitting(true)
    setSubmitError(null)
    setProgress(0)
    try {
      const submittedLabel = label.trim() === '' ? null : label
      const signature = JSON.stringify([files.map((entry) => entry.id), output, modelVariant, submittedLabel])
      if (retry.current?.signature !== signature) {
        retry.current = { signature, key: crypto.randomUUID() }
      }
      const response = await submitConversion({
        kind,
        files: files.map((entry) => entry.file),
        output,
        modelVariant,
        label: submittedLabel,
        idempotencyKey: retry.current.key,
        onProgress: setProgress,
      })
      void navigate(`/jobs/${response.job_id}`)
    } catch (error) {
      if (error instanceof ProblemError) {
        setSubmitError(error.problem)
      } else if (error instanceof TransportError) {
        setSubmitError({ code: 'INTERNAL_ERROR' })
      } else {
        throw error
      }
    } finally {
      inFlight.current = false
      setSubmitting(false)
      setProgress(null)
    }
  }

  return {
    files,
    addFiles,
    removeFile,
    reorder: setFiles,
    output,
    setOutput,
    modelVariant,
    setModelVariant,
    label,
    setLabel: setLabelValue,
    hasImages,
    selectionError,
    submitError,
    progress,
    canSubmit,
    submit,
  }
}

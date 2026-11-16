import { ProblemError, TransportError, type Problem } from './problem'
import type { components } from './schema'

export type SubmissionResponse = components['schemas']['SubmissionResponse']

/** UX §3.3/§4: a raw PDF body, or `multipart/form-data` with ordered `page` parts. */
export type ConversionInputKind = 'pdf' | 'image'

export interface SubmitConversionParams {
  kind: ConversionInputKind
  /** In result order (§4.4 `UploadQueue` reordering) -- exactly one file when `kind === 'pdf'`. */
  files: readonly File[]
  output: SubmissionResponse['requested_output']
  modelVariant?: string | null
  label?: string | null
  /** Reuse after an uncertain response to avoid creating a duplicate job. */
  idempotencyKey?: string
  onProgress?: (fraction: number) => void
  signal?: AbortSignal
}

const SUBMIT_URL = '/v1/conversions'
const PDF_CONTENT_TYPE = 'application/pdf'
const PROBLEM_CONTENT_TYPE = 'application/problem+json'

function buildBody(
  kind: ConversionInputKind,
  files: readonly File[],
): { body: XMLHttpRequestBodyInit; contentType: string | null } {
  if (kind === 'pdf') {
    const [file] = files
    if (file === undefined) {
      throw new Error('submitConversion: at least one file is required')
    }
    return { body: file, contentType: PDF_CONTENT_TYPE }
  }
  const form = new FormData()
  for (const file of files) {
    form.append('page', file, file.name)
  }
  // FormData sets its own `multipart/form-data; boundary=...` Content-Type --
  // setting one here would drop the boundary and break the server's parser.
  return { body: form, contentType: null }
}

function parseJsonBody(xhr: XMLHttpRequest): unknown {
  if (xhr.responseText === '') {
    return undefined
  }
  try {
    return JSON.parse(xhr.responseText) as unknown
  } catch {
    return undefined
  }
}

/**
 * Uploads a new conversion via `XMLHttpRequest` (design §7.4/A8): `fetch`
 * has no upload progress event, so the submit form's progress bar needs XHR
 * specifically, carrying the same headers as `apiClient` (`api/client.ts`)
 * plus the submission-only ones (`X-Unflatten-Output`, `Idempotency-Key`,
 * `X-Unflatten-Label`). This is a same-origin request, so -- like `fetch`'s
 * default `credentials: 'same-origin'` -- the browser sends the session
 * cookie without setting `xhr.withCredentials` (that flag only affects
 * cross-origin requests).
 */
export function submitConversion(params: SubmitConversionParams): Promise<SubmissionResponse> {
  const { kind, files, output, modelVariant, label, onProgress, signal } = params
  const { body, contentType } = buildBody(kind, files)

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', SUBMIT_URL)
    xhr.setRequestHeader('X-Unflatten-Client', 'web')
    xhr.setRequestHeader('X-Unflatten-Profile', 'word-core-v0')
    xhr.setRequestHeader('X-Unflatten-Output', output)
    xhr.setRequestHeader('Idempotency-Key', params.idempotencyKey ?? crypto.randomUUID())
    if (contentType !== null) {
      xhr.setRequestHeader('Content-Type', contentType)
    }
    if (modelVariant !== null && modelVariant !== undefined) {
      xhr.setRequestHeader('X-Unflatten-Model', modelVariant)
    }
    if (label !== null && label !== undefined && label !== '') {
      // design §3.2: header value is percent-encoded UTF-8, decoded server-side.
      xhr.setRequestHeader('X-Unflatten-Label', encodeURIComponent(label))
    }

    xhr.upload.onprogress = (event) => {
      if (onProgress && event.lengthComputable) {
        onProgress(event.loaded / event.total)
      }
    }

    xhr.onload = () => {
      const data = parseJsonBody(xhr)
      if (xhr.status === 202 && data !== undefined) {
        resolve(data as SubmissionResponse)
        return
      }
      const responseContentType = xhr.getResponseHeader('content-type') ?? ''
      if (responseContentType.includes(PROBLEM_CONTENT_TYPE) && data !== undefined) {
        reject(new ProblemError(data as Problem))
        return
      }
      reject(new TransportError(xhr.status))
    }
    xhr.onerror = () => {
      reject(new TransportError(undefined))
    }
    xhr.onabort = () => {
      reject(new TransportError(undefined))
    }
    if (signal) {
      if (signal.aborted) {
        xhr.abort()
        return
      }
      signal.addEventListener(
        'abort',
        () => {
          xhr.abort()
        },
        { once: true },
      )
    }

    xhr.send(body)
  })
}

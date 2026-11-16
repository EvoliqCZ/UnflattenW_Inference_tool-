import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ProblemError, TransportError } from './problem'
import { submitConversion } from './upload'

/**
 * A fake `XMLHttpRequest` (plan T31: "falošný XHR"): real `XMLHttpRequest` in
 * jsdom does not model upload progress or response events precisely enough,
 * and `upload.ts` deliberately depends on the raw XHR API (`fetch` has no
 * upload progress event), so the seam under test is this fake, not jsdom's.
 */
class FakeXMLHttpRequest {
  static instances: FakeXMLHttpRequest[] = []

  method = ''
  url = ''
  status = 0
  responseText = ''
  headers = new Map<string, string>()
  body: unknown
  aborted = false
  upload: { onprogress: ((event: ProgressEvent) => void) | null } = { onprogress: null }
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  onabort: (() => void) | null = null
  private responseHeaders = new Map<string, string>()

  constructor() {
    FakeXMLHttpRequest.instances.push(this)
  }

  open(method: string, url: string): void {
    this.method = method
    this.url = url
  }

  setRequestHeader(name: string, value: string): void {
    this.headers.set(name.toLowerCase(), value)
  }

  send(body: unknown): void {
    this.body = body
  }

  abort(): void {
    this.aborted = true
    this.onabort?.()
  }

  getResponseHeader(name: string): string | null {
    return this.responseHeaders.get(name.toLowerCase()) ?? null
  }

  respondJson(status: number, data: unknown, contentType = 'application/json'): void {
    this.status = status
    this.responseText = JSON.stringify(data)
    this.responseHeaders.set('content-type', contentType)
    this.onload?.()
  }
}

function lastRequest(): FakeXMLHttpRequest {
  const instance = FakeXMLHttpRequest.instances.at(-1)
  if (instance === undefined) {
    throw new Error('no XMLHttpRequest was constructed')
  }
  return instance
}

beforeEach(() => {
  FakeXMLHttpRequest.instances = []
  vi.stubGlobal('XMLHttpRequest', FakeXMLHttpRequest)
})

const pdfFile = new File([new Uint8Array([1, 2, 3])], 'source.pdf', { type: 'application/pdf' })

describe('submitConversion', () => {
  it('sends the required headers for a raw PDF submission', () => {
    void submitConversion({ kind: 'pdf', files: [pdfFile], output: 'diagnostic_docx' })

    const xhr = lastRequest()
    expect(xhr.method).toBe('POST')
    expect(xhr.url).toBe('/v1/conversions')
    expect(xhr.headers.get('x-unflatten-client')).toBe('web')
    expect(xhr.headers.get('x-unflatten-profile')).toBe('word-core-v0')
    expect(xhr.headers.get('x-unflatten-output')).toBe('diagnostic_docx')
    expect(xhr.headers.get('content-type')).toBe('application/pdf')
    expect(xhr.headers.get('idempotency-key')).toMatch(/^[0-9a-f-]{36}$/)
    expect(xhr.body).toBe(pdfFile)
  })

  it('percent-encodes a label with non-ASCII characters into X-Unflatten-Label', () => {
    void submitConversion({ kind: 'pdf', files: [pdfFile], output: 'diagnostic_docx', label: 'Zmluva č. 1' })

    expect(lastRequest().headers.get('x-unflatten-label')).toBe(encodeURIComponent('Zmluva č. 1'))
  })

  it('omits X-Unflatten-Label when no label is given', () => {
    void submitConversion({ kind: 'pdf', files: [pdfFile], output: 'diagnostic_docx' })

    expect(lastRequest().headers.has('x-unflatten-label')).toBe(false)
  })

  it('sends X-Unflatten-Model only when a variant is chosen', () => {
    void submitConversion({ kind: 'pdf', files: [pdfFile], output: 'diagnostic_docx', modelVariant: 'qwen3-vl-int8' })

    expect(lastRequest().headers.get('x-unflatten-model')).toBe('qwen3-vl-int8')
  })

  it('builds a FormData body with ordered page parts for images, without a manual Content-Type', () => {
    const first = new File(['a'], 'page-1.png', { type: 'image/png' })
    const second = new File(['b'], 'page-2.png', { type: 'image/png' })

    void submitConversion({ kind: 'image', files: [first, second], output: 'diagnostic_docx' })

    const xhr = lastRequest()
    expect(xhr.headers.has('content-type')).toBe(false)
    const form = xhr.body as FormData
    expect(form.getAll('page')).toEqual([first, second])
  })

  it('reports upload progress fractions via onProgress', () => {
    const onProgress = vi.fn()
    void submitConversion({ kind: 'pdf', files: [pdfFile], output: 'diagnostic_docx', onProgress })

    lastRequest().upload.onprogress?.({ lengthComputable: true, loaded: 50, total: 200 } as ProgressEvent)

    expect(onProgress).toHaveBeenCalledWith(0.25)
  })

  it('does not set withCredentials for the same-origin request', () => {
    void submitConversion({ kind: 'pdf', files: [pdfFile], output: 'diagnostic_docx' })

    expect((lastRequest() as unknown as { withCredentials?: boolean }).withCredentials).toBeUndefined()
  })

  it('resolves with the parsed SubmissionResponse on 202', async () => {
    const response = {
      schema_version: '1.0',
      job_id: 'job_1',
      state: 'queued',
      requested_output: 'diagnostic_docx',
      claim_scope: 'diagnostic',
      product_quality_claim: false,
      status_url: '/v1/conversions/job_1',
    }
    const promise = submitConversion({ kind: 'pdf', files: [pdfFile], output: 'diagnostic_docx' })

    lastRequest().respondJson(202, response)

    await expect(promise).resolves.toEqual(response)
  })

  it('rejects with ProblemError for a problem+json error response', async () => {
    const problem = {
      type: null,
      code: 'REQUEST_INVALID',
      title: 'Label is invalid',
      status: 400,
      retryable: false,
      blockers: null,
      max_pages: null,
      retry_after_seconds: null,
    }
    const promise = submitConversion({ kind: 'pdf', files: [pdfFile], output: 'diagnostic_docx' })

    lastRequest().respondJson(400, problem, 'application/problem+json')

    await expect(promise).rejects.toBeInstanceOf(ProblemError)
  })

  it('rejects with TransportError on a network error', async () => {
    const promise = submitConversion({ kind: 'pdf', files: [pdfFile], output: 'diagnostic_docx' })

    lastRequest().onerror?.()

    await expect(promise).rejects.toBeInstanceOf(TransportError)
  })

  it('aborts the underlying request when the signal aborts', async () => {
    const controller = new AbortController()
    const promise = submitConversion({
      kind: 'pdf',
      files: [pdfFile],
      output: 'diagnostic_docx',
      signal: controller.signal,
    })

    controller.abort()

    expect(lastRequest().aborted).toBe(true)
    await expect(promise).rejects.toBeInstanceOf(TransportError)
  })
})

it('sends a caller-supplied key so retries refer to the same job', () => {
  void submitConversion({
    kind: 'pdf',
    files: [pdfFile],
    output: 'diagnostic_docx',
    idempotencyKey: 'fixed-upload-key',
  })
  expect(lastRequest().headers.get('idempotency-key')).toBe('fixed-upload-key')
})

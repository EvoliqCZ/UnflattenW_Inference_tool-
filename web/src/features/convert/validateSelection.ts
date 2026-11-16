/** UX §4.3: limits come from `GET /v1/readiness` (`limits.*`), never hard-coded. */
export interface SelectionLimits {
  maxPages: number
  maxUploadBytes: number
  acceptedMediaTypes: readonly string[]
}

export type SelectionValidationCode = 'type' | 'mixed' | 'onePdf' | 'tooManyImages' | 'tooLarge' | 'undecodable'

export interface SelectionValidationError {
  code: SelectionValidationCode
  params?: Record<string, string | number>
}

const PDF_MEDIA_TYPE = 'application/pdf'

function isPdf(file: File): boolean {
  return file.type === PDF_MEDIA_TYPE
}

/**
 * UX §4.3: the checks run in this exact order, and the whole selection is
 * accepted or rejected together -- a bad file never causes a partial accept.
 * PDF page count is deliberately not checked here (no pdf.js in the
 * browser, design §12 A11); the server is the sole authority and returns
 * `TOO_MANY_PAGES`/`UNSUPPORTED_PAGE_COUNT` on submit.
 */
export function validateSelectionSync(
  files: readonly File[],
  limits: SelectionLimits,
): SelectionValidationError | null {
  for (const file of files) {
    if (!limits.acceptedMediaTypes.includes(file.type)) {
      return { code: 'type', params: { name: file.name } }
    }
  }
  const pdfCount = files.filter(isPdf).length
  const hasImages = files.some((file) => !isPdf(file))
  if (pdfCount > 0 && hasImages) {
    return { code: 'mixed' }
  }
  if (pdfCount > 1) {
    return { code: 'onePdf' }
  }
  if (pdfCount === 0 && files.length > limits.maxPages) {
    return { code: 'tooManyImages', params: { n: files.length, max: limits.maxPages } }
  }
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0)
  if (totalBytes > limits.maxUploadBytes) {
    return { code: 'tooLarge', params: { size: totalBytes, max: limits.maxUploadBytes } }
  }
  return null
}

export type DecodeCheck = (file: File) => Promise<boolean>

/** UX §4.3: an image must decode in the browser (`createImageBitmap`). */
export const defaultDecodeCheck: DecodeCheck = async (file) => {
  try {
    const bitmap = await createImageBitmap(file)
    bitmap.close()
    return true
  } catch {
    return false
  }
}

/**
 * Full browser-side validation (UX §4.3): cheap sync checks first, then each
 * image's decodability. All-or-nothing throughout -- the first failure
 * rejects the entire selection, none of it is accepted.
 */
export async function validateSelection(
  files: readonly File[],
  limits: SelectionLimits,
  decodeCheck: DecodeCheck = defaultDecodeCheck,
): Promise<SelectionValidationError | null> {
  const syncError = validateSelectionSync(files, limits)
  if (syncError !== null) {
    return syncError
  }
  for (const file of files) {
    if (!isPdf(file) && !(await decodeCheck(file))) {
      return { code: 'undecodable', params: { name: file.name } }
    }
  }
  return null
}

/**
 * T41: a minimal, synthetic, single-page PDF built in memory -- no fixture
 * binary checked into the repo, no external/scanned document (CLAUDE.md:
 * "Use only synthetic test inputs"). The admission path only ever checks the
 * `%PDF-` magic bytes (`api.py` `_admit_pdf_stream`/`_stage_pdf_bytes`) and
 * `fake_worker.py` never reads the uploaded bytes at all (it publishes canned
 * fixtures), so this does not need to be a realistic rendering -- just a
 * structurally valid, minimal one-page PDF.
 */
export function minimalPdfBytes(): Buffer {
  const text = [
    '%PDF-1.4',
    '1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj',
    '2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj',
    '3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >> endobj',
    'trailer << /Root 1 0 R >>',
    '%%EOF',
    '',
  ].join('\n')
  return Buffer.from(text, 'ascii')
}

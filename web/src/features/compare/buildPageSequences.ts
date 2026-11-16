import type { ArtifactEntry } from './types'

export interface PageSequences {
  /** page_index -> preview url */
  source: Map<number, string>
  docx: Map<number, string>
  sourceCount: number
  docxCount: number
}

/**
 * UX §5.1: `artifacts[]` holds both roles interleaved; build two independent,
 * `page_index`-keyed sequences. **No assumption** that source page *i*
 * corresponds to docx page *i* -- the caller indexes each map separately.
 */
export function buildPageSequences(artifacts: ArtifactEntry[]): PageSequences {
  const source = new Map<number, string>()
  const docx = new Map<number, string>()

  for (const artifact of artifacts) {
    if (artifact.page_index === null) continue
    if (artifact.role === 'source_preview') {
      source.set(artifact.page_index, artifact.url)
    } else if (artifact.role === 'docx_preview') {
      docx.set(artifact.page_index, artifact.url)
    }
  }

  return { source, docx, sourceCount: source.size, docxCount: docx.size }
}

import type { components } from '@/api/schema'

export type ArtifactEntry = components['schemas']['ArtifactEntry']
export type DiagnosticOutcome = components['schemas']['DiagnosticOutcome']
export type PageReport = components['schemas']['PageReport']

/** UX §5.2: the three comparison modes. */
export type CompareMode = 'side' | 'overlay' | 'swipe'

/**
 * UX §5.6: the reason a pane has no image to show, in the priority order
 * the table specifies. `loadFailed` is not decided here -- it is a runtime
 * condition `PagePane` discovers from its own `<img onError>` -- everything
 * else is decided from data already in hand (`selectMissingReason.ts`).
 */
export type MissingReason =
  | { kind: 'pageFailed'; index: number; errorCode: string }
  | { kind: 'previewFailed' }
  | { kind: 'docxNotProduced'; reasonCodes: DiagnosticOutcome['reason_codes'] }
  | { kind: 'noSource'; index: number }
  | { kind: 'outOfRange'; index: number; side: 'source' | 'docx' }
  | { kind: 'loadFailed' }

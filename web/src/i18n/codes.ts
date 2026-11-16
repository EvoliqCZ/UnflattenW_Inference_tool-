// Typed literal registers mirroring `web/codes.json` (design §5.3), which is
// exported by the Python service and is the source of truth. TypeScript cannot
// recover literal string-union types from a plain JSON import (arrays widen to
// `string[]`), so the code lists are restated here as `as const` tuples. Drift
// is not silent: `i18n-completeness.test.ts` asserts every tuple below equals
// (as a set) the matching group in `codes.json`.
//
// `CODE_GROUPS` is the single map the completeness test iterates — adding a
// group to `codes.json` without adding it here fails that test immediately.

export const PROBLEM_CODES = [
  'ARTIFACT_EXPIRED',
  'AST_PROMOTION_FAILED',
  'AST_PROMOTION_UNAVAILABLE',
  'AUTHENTICATION_FAILED',
  'CAPABILITY_DENIED',
  'CAPABILITY_NOT_READY',
  'HOST_NOT_ALLOWED',
  'IDEMPOTENCY_KEY_REUSED',
  'IMAGE_INPUT_INVALID',
  'INTERNAL_ERROR',
  'JOB_ACTIVE',
  'JOB_CANCELLED',
  'JOB_TERMINAL',
  'MODEL_CHECKPOINT_UNVERIFIED',
  'MODEL_COMMAND_CONFLICT',
  'MODEL_COMPONENT_LOCK_MISMATCH',
  'MODEL_DECODE_MAX_STEPS',
  'MODEL_DECODE_VOCABULARY_EXHAUSTED',
  'MODEL_DOWNLOAD_FAILED',
  'MODEL_ENGINE_UNAVAILABLE',
  'MODEL_INPUT_FORBIDDEN_CHANNEL',
  'MODEL_VARIANT_DOES_NOT_FIT',
  'MODEL_VARIANT_NOT_DOWNLOADABLE',
  'MODEL_VARIANT_NOT_PUBLISHED',
  'MODEL_VARIANT_PLANNED',
  'PDF_ACTIVE_CONTENT',
  'PDF_ANNOTATION_UNSUPPORTED',
  'PDF_EMBEDDED_FILE',
  'PDF_ENCRYPTED',
  'PDF_HIDDEN_TEXT',
  'PDF_MALFORMED',
  'PDF_PAGE_DIMENSIONS_EXCEEDED',
  'PDF_SIGNATURE_INVALID',
  'PDF_TEXT_VISIBILITY_AMBIGUOUS',
  'PNG_INVALID',
  'POPPLER_RESOURCE_LIMIT',
  'POPPLER_TIMEOUT',
  'PREDICTION_IR_INVALID',
  'QUEUE_CAPACITY_EXCEEDED',
  'RASTER_OUTPUT_INVALID',
  'RASTER_PIXEL_LIMIT_EXCEEDED',
  'RATE_LIMITED',
  'RENDERER_FAILED',
  'REQUEST_INVALID',
  'RESOURCE_NOT_FOUND',
  'RUNTIME_POLICY_MISMATCH',
  'SETUP_ALREADY_COMPLETE',
  'SETUP_REQUIRED',
  'TOO_MANY_PAGES',
  'UNKNOWN_MODEL_VARIANT',
  'UNSUPPORTED_MEDIA_TYPE',
  'UNSUPPORTED_PAGE_COUNT',
  'UPLOAD_TOO_LARGE',
  'WEB_REQUEST_REJECTED',
] as const

export const DIAGNOSTIC_REASON_CODES = [
  'DIAGNOSTIC_DOCX_NOT_REQUESTED',
  'DOCX_AST_NOT_STRICT',
  'DOCX_CHAIN_ERROR',
  'DOCX_FONT_SUBSTITUTED',
  'DOCX_OUTPUT_TOO_LARGE',
  'DOCX_PAGE_NOT_TERMINATED',
  'DOCX_PAGE_STREAM_INVALID',
  'DOCX_RENDER_NOT_READY',
  'HEADLESS_DEFAULTS_V0_APPLIED',
  'PREVIEW_RENDER_FAILED',
] as const

export const JOB_STATES = ['cancelled', 'expired', 'failed', 'queued', 'running', 'succeeded'] as const

export const JOB_STAGES = [
  'admission_complete',
  'artifact_finalization',
  'ast_promotion',
  'feature_extraction',
  'loading_model',
  'model_decode',
  'model_input_projection',
  'pdf_inspection',
  'pdf_rasterization',
  'png_validation',
  'prediction_ir_sealing',
  'renderer_gate',
] as const

export const DOCX_BLOCKERS = [
  'AST_PROMOTER_UNAVAILABLE',
  'PRODUCT_CHECKPOINT_UNAVAILABLE',
  'RESOURCE_BRANCH_UNAVAILABLE',
  'RFC004_RENDERER_GATE_UNAVAILABLE',
  'TYPED_PROPERTY_HEADS_UNAVAILABLE',
] as const

export const COMPONENT_CODES = [
  'artifact_store',
  'job_store',
  'pdf_sandbox',
  'principal_store',
  'runtime_policy',
  'worker',
] as const

export const RENDERER_APPROXIMATIONS = ['FONT_FALLBACK'] as const

export const ARTIFACT_ROLES = ['diagnostic_docx', 'docx_preview', 'prediction_ir', 'source_preview'] as const

export const CAPABILITIES = [
  'cancel_own_job',
  'download_docx',
  'download_prediction_ir',
  'read_status',
  'submit_diagnostic',
] as const

export const MODEL_COMMAND_KINDS = ['delete', 'download', 'load'] as const

export const MODEL_COMMAND_STATES = ['failed', 'queued', 'running', 'succeeded'] as const

export const MODEL_COMMAND_PHASES = ['deleting', 'downloading', 'loading', 'verifying'] as const

export const DOWNLOAD_STATES = ['absent', 'downloading', 'failed', 'loading', 'ready', 'verifying'] as const

export const FITS_REASONS = ['compute_capability', 'vram'] as const

export const INPUT_KINDS = ['image', 'pdf'] as const

export const ENGINE_IDS = ['native'] as const

export const ENGINE_UNAVAILABLE_REASONS = ['worker_not_published'] as const

export type ProblemCode = (typeof PROBLEM_CODES)[number]
export type DiagnosticReasonCode = (typeof DIAGNOSTIC_REASON_CODES)[number]
export type JobState = (typeof JOB_STATES)[number]
export type JobStage = (typeof JOB_STAGES)[number]
export type DocxBlocker = (typeof DOCX_BLOCKERS)[number]
export type ComponentCode = (typeof COMPONENT_CODES)[number]
export type RendererApproximation = (typeof RENDERER_APPROXIMATIONS)[number]
export type ArtifactRole = (typeof ARTIFACT_ROLES)[number]
export type Capability = (typeof CAPABILITIES)[number]
export type ModelCommandKind = (typeof MODEL_COMMAND_KINDS)[number]
export type ModelCommandState = (typeof MODEL_COMMAND_STATES)[number]
export type ModelCommandPhase = (typeof MODEL_COMMAND_PHASES)[number]
export type DownloadState = (typeof DOWNLOAD_STATES)[number]
export type FitsReason = (typeof FITS_REASONS)[number]
export type InputKind = (typeof INPUT_KINDS)[number]
export type EngineId = (typeof ENGINE_IDS)[number]
export type EngineUnavailableReason = (typeof ENGINE_UNAVAILABLE_REASONS)[number]

/**
 * Maps each `codes.json` group name to its register and the i18n key prefix
 * that carries its translations (design §5.3 table), plus whether each code
 * maps to a single leaf key or an object with sub-keys (`title`/`hint` etc.).
 */
export const CODE_GROUPS = {
  problem_codes: { register: PROBLEM_CODES, prefix: 'codes.problem', shape: ['title', 'hint'] },
  diagnostic_reason_codes: { register: DIAGNOSTIC_REASON_CODES, prefix: 'codes.reason', shape: null },
  job_states: { register: JOB_STATES, prefix: 'codes.state', shape: null },
  job_stages: { register: JOB_STAGES, prefix: 'codes.stage', shape: null },
  docx_blockers: { register: DOCX_BLOCKERS, prefix: 'codes.blocker', shape: null },
  component_codes: { register: COMPONENT_CODES, prefix: 'codes.component', shape: ['label', 'hint'] },
  renderer_approximations: { register: RENDERER_APPROXIMATIONS, prefix: 'codes.approximation', shape: null },
  artifact_roles: { register: ARTIFACT_ROLES, prefix: 'artifact.role', shape: null },
  capabilities: { register: CAPABILITIES, prefix: 'codes.capability', shape: null },
  model_command_kinds: { register: MODEL_COMMAND_KINDS, prefix: 'codes.commandKind', shape: null },
  model_command_states: { register: MODEL_COMMAND_STATES, prefix: 'codes.commandState', shape: null },
  model_command_phases: { register: MODEL_COMMAND_PHASES, prefix: 'codes.commandPhase', shape: null },
  download_states: { register: DOWNLOAD_STATES, prefix: 'models.state', shape: null },
  fits_reasons: { register: FITS_REASONS, prefix: 'codes.fitsReason', shape: null },
  input_kinds: { register: INPUT_KINDS, prefix: 'codes.inputKind', shape: null },
  engine_ids: { register: ENGINE_IDS, prefix: 'codes.engine', shape: null },
  engine_unavailable_reasons: {
    register: ENGINE_UNAVAILABLE_REASONS,
    prefix: 'codes.engineUnavailable',
    shape: null,
  },
} as const

export type CodeGroupName = keyof typeof CODE_GROUPS

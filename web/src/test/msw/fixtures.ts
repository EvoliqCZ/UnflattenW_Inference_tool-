import type { components } from '@/api/schema'

type ConversionStatus = components['schemas']['ConversionStatus']
type JobList = components['schemas']['JobList']
type JobSummary = components['schemas']['JobSummary']
type ModelCommandAccepted = components['schemas']['ModelCommandAccepted']
type ModelsResponse = components['schemas']['ModelsResponse']
type Problem = components['schemas']['Problem']
type SessionInfo = components['schemas']['SessionInfo']
type SettingsResponse = components['schemas']['SettingsResponse']

/** `GET /v1/session` default fixture for router-guard tests: a valid session. */
export const authenticatedSession: SessionInfo = {
  auth_kind: 'session',
  authenticated: true,
  capabilities: ['submit_conversion'],
  expires_at: '2026-10-04T00:00:00Z',
  principal_id: 'local',
  schema_version: '1.1',
  setup_required: false,
  setup_tokenless: false,
}

export const unauthenticatedSession: SessionInfo = {
  auth_kind: null,
  authenticated: false,
  capabilities: null,
  expires_at: null,
  principal_id: null,
  schema_version: '1.1',
  setup_required: false,
  setup_tokenless: false,
}

/** Setup still required, token-protected (non-loopback deployment): the wizard needs the setup link. */
export const setupRequiredSession: SessionInfo = {
  ...unauthenticatedSession,
  setup_required: true,
}

/** Owner 2026-09-29 (DEC-052 addendum): loopback-only deployment, password-only setup. */
export const setupTokenlessSession: SessionInfo = {
  ...setupRequiredSession,
  setup_tokenless: true,
}

/** A running job with one page in flight (design §7.3 polling test fixture). */
export const runningJob: ConversionStatus = {
  artifacts: [],
  attempt: 1,
  claim_scope: 'diagnostic',
  created_at: '2026-09-27T10:00:00.000Z',
  current_page: 0,
  diagnostic_outcome: null,
  engine: 'native',
  error: null,
  expires_at: null,
  input_kind: 'pdf',
  job_id: 'job-running',
  label: null,
  model_variant: 'qwen3-vl-int8',
  pages: null,
  pages_done: 0,
  pages_total: 1,
  product_quality_claim: false,
  requested_output: 'diagnostic_docx',
  schema_version: '1.1',
  stage: 'model_decode',
  state: 'running',
  terminal_at: null,
  updated_at: '2026-09-27T10:00:05.000Z',
}

/** A finished job -- no further polling should occur. */
export const succeededJob: ConversionStatus = {
  ...runningJob,
  job_id: 'job-succeeded',
  current_page: null,
  pages_done: 1,
  stage: 'artifact_finalization',
  state: 'succeeded',
  terminal_at: '2026-09-27T10:03:00.000Z',
  updated_at: '2026-09-27T10:03:00.000Z',
  expires_at: '2026-09-28T10:03:00.000Z',
}

export const modelsResponse: ModelsResponse = {
  default_variant: 'qwen3-vl-int8',
  default_variant_source: 'config',
  device: null,
  resident_variant: null,
  schema_version: '1.2',
  variants: [
    {
      availability: 'published',
      bundle_manifest_sha256: null,
      download: {
        bytes_done: null,
        bytes_total: null,
        command_id: null,
        error_code: null,
        state: 'ready',
      },
      download_size_bytes: 4_200_000_000,
      downloaded: true,
      engine: 'native',
      engine_available: true,
      engine_unavailable_reason: null,
      estimated_peak_vram_mib: 11_240,
      fits: true,
      fits_reason: null,
      is_default: true,
      kv_cache: 'fp8',
      label: 'Qwen3-VL INT8',
      precision: 'int8',
      resident: false,
      source_kind: 'hf',
      variant_id: 'qwen3-vl-int8',
    },
    {
      availability: 'published',
      bundle_manifest_sha256: null,
      download: {
        bytes_done: null,
        bytes_total: null,
        command_id: null,
        error_code: null,
        state: 'absent',
      },
      download_size_bytes: 21_400_000_000,
      downloaded: false,
      engine: 'native',
      engine_available: true,
      engine_unavailable_reason: null,
      estimated_peak_vram_mib: 21_400,
      fits: false,
      fits_reason: 'vram',
      is_default: false,
      kv_cache: 'bf16',
      label: 'Qwen3-VL BF16',
      precision: 'bf16',
      resident: false,
      source_kind: 'hf',
      variant_id: 'qwen3-vl-bf16',
    },
  ],
}

type ModelVariantEntry = ModelsResponse['variants'][number]

/** `models_response` 1.2: a known precision whose Hugging Face repo is not public yet (DEC-050). */
export const unpublishedModelVariant: ModelVariantEntry = {
  availability: 'unpublished',
  bundle_manifest_sha256: null,
  download: null,
  download_size_bytes: null,
  downloaded: false,
  engine: 'native',
  engine_available: true,
  engine_unavailable_reason: null,
  estimated_peak_vram_mib: 8_488,
  fits: true,
  fits_reason: null,
  is_default: false,
  kv_cache: 'fp8',
  label: 'FP8 (fp8 weights, fp8 KV cache)',
  precision: 'fp8',
  resident: false,
  source_kind: 'hf',
  variant_id: 'fp8-hf',
}

/** `models_response` 1.2: a model that does not exist yet (Unflatten W MoE). */
export const plannedModelVariant: ModelVariantEntry = {
  ...unpublishedModelVariant,
  availability: 'planned',
  estimated_peak_vram_mib: null,
  fits: null,
  kv_cache: null,
  label: 'Unflatten W MoE (future model)',
  precision: null,
  source_kind: null,
  variant_id: 'unflatten-w-moe',
}

/** `POST/DELETE /v1/models/{variant_id}/download`, `POST /v1/models/{variant_id}/load` default fixture (design §3.9). */
export const modelCommandAccepted: ModelCommandAccepted = {
  bytes_done: null,
  bytes_total: null,
  command_id: 'cmd-1',
  error_code: null,
  finished_at: null,
  kind: 'download',
  phase: 'downloading',
  requested_at: '2026-09-27T10:00:00.000Z',
  schema_version: '1.0',
  started_at: null,
  state: 'queued',
  variant_id: 'qwen3-vl-bf16',
}

/** `GET /v1/conversions` row fixture (design §3.4 `JobSummary`). */
export function makeJobSummary(overrides: Partial<JobSummary> = {}): JobSummary {
  return {
    created_at: '2026-09-27T10:00:00.000Z',
    docx_produced: true,
    error_code: null,
    expires_at: '2026-09-28T10:00:00.000Z',
    input_kind: 'pdf',
    job_id: 'job-1',
    label: 'Faktúra-2026',
    model_variant: 'qwen3-vl-int8',
    pages_done: 3,
    pages_total: 3,
    requested_output: 'diagnostic_docx',
    stage: 'artifact_finalization',
    state: 'succeeded',
    terminal_at: '2026-09-27T10:02:00.000Z',
    updated_at: '2026-09-27T10:02:00.000Z',
    ...overrides,
  }
}

export const emptyJobList: JobList = {
  items: [],
  next_cursor: null,
  schema_version: '1.0',
}

export const authenticationFailedProblem: Problem = {
  blockers: null,
  code: 'AUTHENTICATION_FAILED',
  max_pages: null,
  retry_after_seconds: null,
  retryable: false,
  status: 401,
  title: 'Authentication failed',
  type: null,
}

/** `GET /v1/settings` default fixture (design §3.10): non-rootful sandbox, no user default override. */
export const settingsResponse: SettingsResponse = {
  default_variant: {
    configured: 'qwen3-vl-int8',
    effective: 'qwen3-vl-int8',
    override: null,
    source: 'config',
  },
  sandbox: {
    rootful: false,
  },
  schema_version: '1.0',
  service_version: {
    api_version: '1.1',
    git_sha: '57db768',
  },
  session: {
    idle_seconds: 1800,
    max_seconds: 28800,
  },
}

export const internalErrorProblem: Problem = {
  blockers: null,
  code: 'INTERNAL_ERROR',
  max_pages: null,
  retry_after_seconds: null,
  retryable: true,
  status: 500,
  title: 'Internal error',
  type: null,
}

/** `DELETE /v1/conversions/{job_id}` when the job already reached a terminal state (UX §3.5 point 3). */
export const jobTerminalProblem: Problem = {
  blockers: null,
  code: 'JOB_TERMINAL',
  max_pages: null,
  retry_after_seconds: null,
  retryable: false,
  status: 409,
  title: 'Job is already terminal',
  type: null,
}

import type { components } from '@/api/schema'

type ModelVariantEntry = components['schemas']['ModelVariantEntry']

/**
 * `models_response` 1.2: only a `published` variant can be downloaded,
 * loaded or selected. `unpublished` (Hugging Face repo not public yet,
 * DEC-050) and `planned` (a model that does not exist yet) appear on the
 * Models screen only -- every picker leaves them out, since the API refuses
 * to select them (`MODEL_VARIANT_NOT_PUBLISHED` / `MODEL_VARIANT_PLANNED`).
 */
export function isPublished(variant: ModelVariantEntry): boolean {
  return variant.availability === 'published'
}

import type { components } from '@/api/schema'
import type { Translate } from './fits'

type ModelVariantEntry = components['schemas']['ModelVariantEntry']

/**
 * UX §17.5/§425: the translated `engine_unavailable_reason`, or `null` when
 * the engine is available.
 */
export function describeEngineUnavailableReason(variant: ModelVariantEntry, t: Translate): string | null {
  if (variant.engine_available || variant.engine_unavailable_reason === null) {
    return null
  }
  return t('codes.engineUnavailable.worker_not_published')
}
